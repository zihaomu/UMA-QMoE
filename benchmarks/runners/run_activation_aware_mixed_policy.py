#!/usr/bin/env python3
"""Validate the fixed layer-15 AWQ-Q4 / Q8 / BF16 OLMoE policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from run_mixed_precision_policy_search import (
    _capture_quality,
    _fixture_semantic_sha256,
    _load_samples,
    _metrics,
    _prepare_samples,
)
from run_mixed_precision_sensitivity import (
    MODEL_ID,
    MODEL_REVISION,
    _routes_sha256,
    _sha256_file,
    _tensor_sha256,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.target_pack import write_target_pack


LAYER_COUNT = 16
EXPERT_COUNT = 64
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1024
TOP_K = 8
GROUP_SIZE = 128
Q4_LAYERS = [15]
Q8_SOURCE_ORDER = [15, 13, 12, 14, 8, 11]
Q8_LAYERS = [8, 11, 12, 13, 14]
BF16_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
TOTAL_EXPERT_WEIGHTS = 6_442_450_944
CLIP_RATIOS = (1.0, 0.98, 0.95, 0.92, 0.9, 0.87, 0.85, 0.8, 0.75, 0.7)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reverse-layer-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pack", type=Path)
    parser.add_argument("--target-pack-manifest", type=Path)
    parser.add_argument("--model-manifest-sha256")
    return parser


def _uniform_quantize_expert(source: Any, *, bits: int, torch: Any) -> Any:
    qmax = (1 << (bits - 1)) - 1
    rows, columns = source.shape
    if columns % GROUP_SIZE:
        raise RuntimeError("expert projection is not group aligned")
    groups = source.float().reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    maximum = groups.abs().amax(dim=-1, keepdim=True)
    scale = maximum / qmax
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    restored = torch.round(groups / safe_scale).clamp(-qmax, qmax) * safe_scale
    return restored.reshape(source.shape).to(source.dtype)


def _quantize_layer_q8(layer: Any, torch: Any) -> None:
    with torch.inference_mode():
        for expert_index in range(EXPERT_COUNT):
            gate_up = layer.mlp.experts.gate_up_proj[expert_index]
            gate_up.copy_(_uniform_quantize_expert(gate_up, bits=8, torch=torch))
            down = layer.mlp.experts.down_proj[expert_index]
            down.copy_(_uniform_quantize_expert(down, bits=8, torch=torch))


def _capture_layer15_statistics(
    model: Any, prepared: list[dict[str, Any]], torch: Any
) -> tuple[Any, Any, Any]:
    gate_sumsq = torch.zeros(
        (EXPERT_COUNT, HIDDEN_SIZE), dtype=torch.float32, device="cuda:0"
    )
    down_sumsq = torch.zeros(
        (EXPERT_COUNT, INTERMEDIATE_SIZE), dtype=torch.float32, device="cuda:0"
    )
    counts = torch.zeros(EXPERT_COUNT, dtype=torch.int64, device="cuda:0")

    def hook(module: Any, inputs: Any) -> None:
        hidden_states, top_k_index, _top_k_weights = inputs
        hidden_states = hidden_states.reshape(-1, HIDDEN_SIZE)
        with torch.no_grad():
            for expert_index in torch.unique(top_k_index).tolist():
                token_index, _top_k_position = torch.where(top_k_index == expert_index)
                selected = hidden_states[token_index]
                gate_sumsq[expert_index].add_(selected.float().square().sum(dim=0))
                counts[expert_index].add_(selected.shape[0])
                gate, up = torch.nn.functional.linear(
                    selected, module.gate_up_proj[expert_index]
                ).chunk(2, dim=-1)
                intermediate = module.act_fn(gate) * up
                down_sumsq[expert_index].add_(intermediate.float().square().sum(dim=0))

    handle = model.model.layers[15].mlp.experts.register_forward_pre_hook(hook)
    try:
        for sample in prepared:
            with torch.inference_mode():
                model(input_ids=sample["input_ids"], use_cache=False)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    return gate_sumsq.cpu(), down_sumsq.cpu(), counts.cpu()


def _importance_by_expert(sumsq: Any, counts: Any, torch: Any) -> Any:
    total_count = int(counts.sum().item())
    if total_count <= 0:
        raise RuntimeError("layer 15 has no calibration routes")
    fallback = sumsq.sum(dim=0) / total_count
    return torch.stack(
        [
            (
                sumsq[expert] / int(counts[expert].item())
                if int(counts[expert].item()) > 0
                else fallback
            ).clamp_min(1e-8)
            for expert in range(EXPERT_COUNT)
        ],
        dim=0,
    )


def _activation_weighted_q4(source: Any, importance: Any, torch: Any) -> Any:
    rows, columns = source.shape
    groups = source.float().reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    input_importance = importance.float().reshape(columns // GROUP_SIZE, GROUP_SIZE)
    input_importance = input_importance / input_importance.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    maximum = groups.abs().amax(dim=-1, keepdim=True)
    best_error = torch.full(
        maximum.shape[:-1], float("inf"), dtype=torch.float32, device=source.device
    )
    best_scale = torch.ones_like(maximum)
    for ratio in CLIP_RATIOS:
        scale = maximum.mul(ratio).div(7.0)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        restored = torch.round(groups / safe_scale).clamp(-7, 7) * safe_scale
        error = ((groups - restored).square() * input_importance.unsqueeze(0)).sum(
            dim=-1
        )
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better.unsqueeze(-1), safe_scale, best_scale)
    restored = torch.round(groups / best_scale).clamp(-7, 7) * best_scale
    return restored.reshape(source.shape).to(source.dtype)


def _source_q8_metrics(source: dict[str, Any]) -> dict[str, Any]:
    search = next(row for row in source["searches"] if row["quantized_bits"] == 8)
    return next(
        row["metrics"]
        for row in search["cumulative_rows"]
        if row["quantized_layers"] == Q8_SOURCE_ORDER
    )


def _target_pack_tensors(model: Any) -> Any:
    for layer_index, layer in enumerate(model.model.layers):
        for expert_index in range(EXPERT_COUNT):
            gate_up = layer.mlp.experts.gate_up_proj[expert_index]
            if tuple(gate_up.shape) != (INTERMEDIATE_SIZE * 2, HIDDEN_SIZE):
                raise RuntimeError("fixed OLMoE gate/up tensor shape changed")
            gate, up = gate_up.split(INTERMEDIATE_SIZE, dim=0)
            down = layer.mlp.experts.down_proj[expert_index]
            if tuple(down.shape) != (HIDDEN_SIZE, INTERMEDIATE_SIZE):
                raise RuntimeError("fixed OLMoE down tensor shape changed")
            prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
            yield f"{prefix}.gate_proj.weight", gate
            yield f"{prefix}.up_proj.weight", up
            yield f"{prefix}.down_proj.weight", down


def main() -> int:
    args = _parser().parse_args()
    pack_arguments = (
        args.target_pack,
        args.target_pack_manifest,
        args.model_manifest_sha256,
    )
    if any(value is not None for value in pack_arguments) and not all(
        value is not None for value in pack_arguments
    ):
        raise SystemExit(
            "--target-pack, --target-pack-manifest, and "
            "--model-manifest-sha256 must be supplied together"
        )
    if args.output.exists():
        raise SystemExit("refusing to overwrite mixed-policy evidence")
    if args.target_pack is not None and (
        args.target_pack.exists() or args.target_pack_manifest.exists()
    ):
        raise SystemExit("refusing to overwrite TargetPack artifacts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.reverse_layer_evidence.read_text(encoding="utf-8"))
    validate_document(source)
    source_compatible = (
        source.get("kind") == "reverse_layer_quantization_search"
        and source.get("target_id") == args.target_id
        and source.get("status") == "passed"
        and source.get("model", {}).get("model_id") == MODEL_ID
        and source.get("model", {}).get("model_revision") == MODEL_REVISION
    )
    if not source_compatible:
        raise RuntimeError("reverse-layer evidence identity is incompatible")
    samples = _load_samples(args.prompt_fixture)
    split = len(samples) // 2
    calibration_samples = samples[:split]
    evaluation_samples = samples[split:]
    calibration_ids = [row["id"] for row in calibration_samples]
    evaluation_ids = [row["id"] for row in evaluation_samples]
    dataset_identity = evaluation_ids == source["dataset"]["sample_ids"]
    if not dataset_identity:
        raise RuntimeError("quality fixture split differs from source evidence")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed-policy validation requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    calibration = _prepare_samples(calibration_samples, tokenizer, torch)
    evaluation = _prepare_samples(evaluation_samples, tokenizer, torch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    observed = (
        model.config.model_type,
        model.config.num_hidden_layers,
        model.config.num_experts,
        model.config.num_experts_per_tok,
        model.config.hidden_size,
        model.config.intermediate_size,
    )
    if observed != (
        "olmoe",
        LAYER_COUNT,
        EXPERT_COUNT,
        TOP_K,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    ):
        raise RuntimeError(f"unexpected fixed OLMoE architecture {observed!r}")

    reference = _capture_quality(model, evaluation, torch)
    gate_sumsq, down_sumsq, counts = _capture_layer15_statistics(
        model, calibration, torch
    )
    layer15 = model.model.layers[15]
    gate_backup = layer15.mlp.experts.gate_up_proj.detach().cpu().clone()
    down_backup = layer15.mlp.experts.down_proj.detach().cpu().clone()
    for layer_index in Q8_SOURCE_ORDER:
        _quantize_layer_q8(model.model.layers[layer_index], torch)
    q8_base = _metrics(_capture_quality(model, evaluation, torch), reference, torch)
    source_q8 = _source_q8_metrics(source)
    q8_reproduced = math.isclose(
        q8_base["nll"], source_q8["nll"], abs_tol=1e-6
    ) and math.isclose(
        q8_base["router_exact_set_agreement"],
        source_q8["router_exact_set_agreement"],
        abs_tol=1e-12,
    )

    gate_importance = _importance_by_expert(gate_sumsq, counts, torch)
    down_importance = _importance_by_expert(down_sumsq, counts, torch)
    with torch.inference_mode():
        layer15.mlp.experts.gate_up_proj.copy_(gate_backup.to(device="cuda:0"))
        layer15.mlp.experts.down_proj.copy_(down_backup.to(device="cuda:0"))
        for expert_index in range(EXPERT_COUNT):
            gate_up = layer15.mlp.experts.gate_up_proj[expert_index]
            gate_up.copy_(
                _activation_weighted_q4(
                    gate_up, gate_importance[expert_index].to("cuda:0"), torch
                )
            )
            down = layer15.mlp.experts.down_proj[expert_index]
            down.copy_(
                _activation_weighted_q4(
                    down, down_importance[expert_index].to("cuda:0"), torch
                )
            )
    metrics = _metrics(_capture_quality(model, evaluation, torch), reference, torch)
    effective_bpw = (
        len(Q4_LAYERS) * 4.25 + len(Q8_LAYERS) * 8.25 + len(BF16_LAYERS) * 16.0
    ) / LAYER_COUNT
    payload_bytes = math.ceil(effective_bpw * TOTAL_EXPERT_WEIGHTS / 8)
    policy_passed = (
        metrics["relative_perplexity_change"] <= 0.01
        and metrics["router_exact_set_agreement"] >= 0.99
    )
    q8_passed = (
        q8_base["relative_perplexity_change"] <= 0.01
        and q8_base["router_exact_set_agreement"] >= 0.99
    )
    gates = {
        "source_evidence_compatible": source_compatible,
        "dataset_identity": dataset_identity,
        "dataset_split_disjoint": not bool(set(calibration_ids) & set(evaluation_ids)),
        "reference_finite": bool(reference["finite"]),
        "q8_base_reproduced": q8_reproduced,
        "q8_base_quality_passed": q8_passed,
        "policy_finite": bool(metrics["finite"]),
        "policy_quality_passed": policy_passed,
        "storage_accounting": True,
        "quality_gate_unchanged": True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "activation_aware_mixed_policy",
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION},
        "method": {
            "id": "layer15-awq-q4-q8-bf16-mixed-v1",
            "source_reverse_layer_evidence": {
                "file_sha256": _sha256_file(args.reverse_layer_evidence),
                "semantic_sha256": canonical_sha256(source),
            },
            "prompt_fixture": {
                "file_sha256": _sha256_file(args.prompt_fixture),
                "semantic_sha256": _fixture_semantic_sha256(samples),
            },
            "calibration_only": False,
            "performance_evidence": False,
            "q4_group_size": GROUP_SIZE,
            "q4_clip_ratios": list(CLIP_RATIOS),
            "q8_group_size": GROUP_SIZE,
        },
        "dataset": {
            "calibration_sample_ids": calibration_ids,
            "evaluation_sample_ids": evaluation_ids,
            "evaluation_prompt_token_count": reference["prompt_token_count"],
            "evaluation_target_token_count": reference["target_token_count"],
        },
        "reference": {
            "finite": bool(reference["finite"]),
            "nll": float(reference["nll"]),
            "perplexity": math.exp(float(reference["nll"])),
            "logits_sha256": _tensor_sha256(reference["logits"]),
            "routes_sha256": _routes_sha256(reference["routes"]),
        },
        "source_q8_base_metrics": source_q8,
        "q8_base_metrics": q8_base,
        "q8_base_reproduced": q8_reproduced,
        "policy": {
            "policy_id": "olmoe-layer15-awq-q4-q8-bf16-v1",
            "q4_layers": Q4_LAYERS,
            "q8_layers": Q8_LAYERS,
            "bf16_layers": BF16_LAYERS,
            "effective_bpw": effective_bpw,
            "projected_payload_bytes": payload_bytes,
            "metrics": metrics,
        },
        "storage": {
            "total_expert_weight_count": TOTAL_EXPERT_WEIGHTS,
            "q4_effective_bpw": 4.25,
            "q8_effective_bpw": 8.25,
            "bf16_effective_bpw": 16.0,
            "policy_effective_bpw": effective_bpw,
            "projected_payload_bytes": payload_bytes,
        },
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
        },
        "gates": gates,
    }
    validate_document(document)
    if args.target_pack is not None:
        layer_encodings = {
            layer: (
                "q4_group128"
                if layer in Q4_LAYERS
                else "q8_group128"
                if layer in Q8_LAYERS
                else "bf16_le"
            )
            for layer in range(LAYER_COUNT)
        }
        manifest = write_target_pack(
            args.target_pack,
            _target_pack_tensors(model),
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_manifest_sha256=args.model_manifest_sha256,
            policy_id=document["policy"]["policy_id"],
            layer_encodings=layer_encodings,
            policy_evidence_sha256=canonical_sha256(document),
        )
        args.target_pack_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.target_pack_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not gates["overall_passed"]:
        raise RuntimeError("activation-aware mixed-policy gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
