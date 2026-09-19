#!/usr/bin/env python3
"""Measure single-layer BF16 restoration sensitivity for canonical OLMoE Q4."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from uma_qmoe.contracts import validate_document
from uma_qmoe.expert_pack import ExpertPackReader
from uma_qmoe.numeric import clamp_cosine_similarity
from uma_qmoe.q4 import dequantize_q4


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
LAYER_COUNT = 16
EXPERT_COUNT = 64
INTERMEDIATE_SIZE = 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt(path: Path, prompt_id: str) -> str:
    matches: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("id") == prompt_id:
                matches.append(item.get("prompt"))
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise RuntimeError(f"prompt fixture must contain exactly one {prompt_id!r}")
    return matches[0]


def _tensor_sha256(value: Any) -> str:
    payload = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _routes_sha256(routes: dict[int, Any]) -> str:
    digest = hashlib.sha256()
    for layer_index in range(LAYER_COUNT):
        digest.update(layer_index.to_bytes(2, "little"))
        digest.update(routes[layer_index].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _capture(model: Any, device_inputs: dict[str, Any], torch: Any) -> dict[str, Any]:
    routes: dict[int, Any] = {}
    handles = []

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE router hook returned an unexpected value")
            routes[layer_index] = result[2].detach().cpu()

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(gate_hook(layer_index)))
    try:
        with torch.inference_mode():
            result = model(**device_inputs, use_cache=False)
            logits = result.logits[:, -1, :].detach().float().cpu()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if set(routes) != set(range(LAYER_COUNT)):
        raise RuntimeError("sensitivity capture did not observe all OLMoE routers")
    finite = bool(torch.isfinite(logits).all().item())
    return {"logits": logits, "routes": routes, "finite": finite}


def _metrics(
    candidate: dict[str, Any], reference: dict[str, Any], torch: Any
) -> dict[str, Any]:
    candidate_logits = candidate["logits"]
    reference_logits = reference["logits"]
    absolute = (candidate_logits - reference_logits).abs().reshape(-1)
    p99_index = max(0, math.ceil(0.99 * absolute.numel()) - 1)
    p99 = float(torch.sort(absolute).values[p99_index].item())
    cosine = clamp_cosine_similarity(
        float(
            torch.nn.functional.cosine_similarity(
                candidate_logits.reshape(1, -1), reference_logits.reshape(1, -1)
            ).item()
        )
    )
    exact_count = 0
    decision_count = 0
    overlap_total = 0
    per_layer = []
    for layer_index in range(LAYER_COUNT):
        candidate_routes = candidate["routes"][layer_index].reshape(-1, 8)
        reference_routes = reference["routes"][layer_index].reshape(-1, 8)
        candidate_sorted = candidate_routes.sort(dim=-1).values
        reference_sorted = reference_routes.sort(dim=-1).values
        exact = (candidate_sorted == reference_sorted).all(dim=-1)
        intersection = (
            (candidate_routes[:, :, None] == reference_routes[:, None, :])
            .any(dim=-1)
            .sum(dim=-1)
        )
        layer_decisions = int(exact.numel())
        layer_exact = int(exact.sum().item())
        per_layer.append(layer_exact / layer_decisions)
        exact_count += layer_exact
        decision_count += layer_decisions
        overlap_total += int(intersection.sum().item())
    top1 = int(candidate_logits.argmax(dim=-1).item())
    reference_top1 = int(reference_logits.argmax(dim=-1).item())
    return {
        "finite": bool(candidate["finite"]),
        "top1_token_id": top1,
        "top1_matches_reference": top1 == reference_top1,
        "logit_max_absolute_error": float(absolute.max().item()),
        "logit_p99_absolute_error": p99,
        "logit_cosine_similarity": cosine,
        "router_exact_set_agreement": exact_count / decision_count,
        "router_mean_set_overlap": overlap_total / (decision_count * 8),
        "per_layer_router_exact_set_agreement": per_layer,
    }


def _copy_q4_layer(
    model: Any, reader: ExpertPackReader, layer_index: int, torch: Any
) -> None:
    experts = model.model.layers[layer_index].mlp.experts
    with torch.no_grad():
        for expert_index in range(EXPERT_COUNT):
            prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
            gate = torch.from_numpy(
                dequantize_q4(reader.tensor_q4(f"{prefix}.gate_proj.weight"))
            ).to(device="cuda:0", dtype=torch.bfloat16)
            up = torch.from_numpy(
                dequantize_q4(reader.tensor_q4(f"{prefix}.up_proj.weight"))
            ).to(device="cuda:0", dtype=torch.bfloat16)
            down = torch.from_numpy(
                dequantize_q4(reader.tensor_q4(f"{prefix}.down_proj.weight"))
            ).to(device="cuda:0", dtype=torch.bfloat16)
            experts.gate_up_proj[expert_index, :INTERMEDIATE_SIZE].copy_(gate)
            experts.gate_up_proj[expert_index, INTERMEDIATE_SIZE:].copy_(up)
            experts.down_proj[expert_index].copy_(down)
            del gate, up, down


def _copy_bf16_layer(
    model: Any,
    sources: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    torch: Any,
) -> None:
    experts = model.model.layers[layer_index].mlp.experts

    def source(name: str) -> Any:
        value = sources[weight_map[name]].get_tensor(name)
        if value.dtype != torch.bfloat16:
            raise RuntimeError(f"sensitivity BF16 source {name!r} has wrong dtype")
        return value.to(device="cuda:0", non_blocking=False)

    with torch.no_grad():
        for expert_index in range(EXPERT_COUNT):
            prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
            gate = source(f"{prefix}.gate_proj.weight")
            up = source(f"{prefix}.up_proj.weight")
            down = source(f"{prefix}.down_proj.weight")
            experts.gate_up_proj[expert_index, :INTERMEDIATE_SIZE].copy_(gate)
            experts.gate_up_proj[expert_index, INTERMEDIATE_SIZE:].copy_(up)
            experts.down_proj[expert_index].copy_(down)
            del gate, up, down


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite mixed-precision sensitivity evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "mixed-precision sensitivity requires a BF16 CUDA/HIP device"
        )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device_inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
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
        getattr(model.config, "model_type", None),
        getattr(model.config, "num_hidden_layers", None),
        getattr(model.config, "num_experts", None),
        getattr(model.config, "num_experts_per_tok", None),
    )
    if observed != ("olmoe", LAYER_COUNT, EXPERT_COUNT, 8):
        raise RuntimeError(f"unexpected fixed OLMoE architecture {observed!r}")

    reference = _capture(model, device_inputs, torch)
    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    try:
        with ExitStack() as stack:
            sources = {
                shard: stack.enter_context(
                    safe_open(args.model / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(weight_map.values()))
            }
            for layer_index in range(LAYER_COUNT):
                print(f"quantizing layer {layer_index}/15", flush=True)
                _copy_q4_layer(model, reader, layer_index, torch)
            all_q4_capture = _capture(model, device_inputs, torch)
            all_q4 = _metrics(all_q4_capture, reference, torch)

            rows = []
            for layer_index in range(LAYER_COUNT):
                print(f"restoring BF16 layer {layer_index}/15", flush=True)
                _copy_bf16_layer(model, sources, weight_map, layer_index, torch)
                candidate = _capture(model, device_inputs, torch)
                metrics = _metrics(candidate, reference, torch)
                rows.append(
                    {
                        "restored_layer": layer_index,
                        "metrics": metrics,
                        "delta_vs_all_q4": {
                            "router_exact_set_agreement": (
                                metrics["router_exact_set_agreement"]
                                - all_q4["router_exact_set_agreement"]
                            ),
                            "router_mean_set_overlap": (
                                metrics["router_mean_set_overlap"]
                                - all_q4["router_mean_set_overlap"]
                            ),
                            "logit_cosine_similarity": (
                                metrics["logit_cosine_similarity"]
                                - all_q4["logit_cosine_similarity"]
                            ),
                        },
                        "finite": bool(metrics["finite"]),
                    }
                )
                _copy_q4_layer(model, reader, layer_index, torch)

        tensor_headers = reader.header["tensors"]
        total_weight_count = sum(item["element_count"] for item in tensor_headers)
        layer_zero = [
            item
            for item in tensor_headers
            if item["name"].startswith("model.layers.0.mlp.experts.")
        ]
        replaced_q4_bytes = sum(
            item["storage_end_offset"] - item["packed_offset"] for item in layer_zero
        )
        bf16_layer_bytes = sum(item["element_count"] * 2 for item in layer_zero)
        all_q4_bytes = args.expert_pack.stat().st_size
        mixed_bytes = all_q4_bytes - replaced_q4_bytes + bf16_layer_bytes
        ranking = [
            row["restored_layer"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["delta_vs_all_q4"]["router_exact_set_agreement"],
                    -row["delta_vs_all_q4"]["logit_cosine_similarity"],
                    row["restored_layer"],
                ),
            )
        ]
        gates = {
            "reference_finite": bool(reference["finite"]),
            "all_q4_finite": bool(all_q4["finite"]),
            "all_layers_covered": sorted(row["restored_layer"] for row in rows)
            == list(range(LAYER_COUNT)),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": True,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "mixed_precision_sensitivity",
            "captured_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "target_id": args.target_id,
            "status": "passed" if gates["overall_passed"] else "failed",
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expert_pack_sha256": _sha256_file(args.expert_pack),
            },
            "method": {
                "id": "single-layer-bf16-restore-v1",
                "diagnostic_only": True,
                "materializes_expert_parameters": True,
                "performance_evidence": False,
                "candidate_definition": "one BF16 expert layer plus fifteen canonical Q4-dequantized expert layers",
                "restored_layers": list(range(LAYER_COUNT)),
            },
            "reference": {
                "finite": bool(reference["finite"]),
                "logits_sha256": _tensor_sha256(reference["logits"]),
                "routes_sha256": _routes_sha256(reference["routes"]),
                "top1_token_id": int(reference["logits"].argmax(dim=-1).item()),
            },
            "all_q4_baseline": all_q4,
            "storage": {
                "all_q4_bytes": all_q4_bytes,
                "single_bf16_layer_bytes": bf16_layer_bytes,
                "replaced_q4_layer_bytes": replaced_q4_bytes,
                "single_layer_mixed_bytes": mixed_bytes,
                "all_q4_effective_bpw": all_q4_bytes * 8 / total_weight_count,
                "single_layer_mixed_effective_bpw": mixed_bytes
                * 8
                / total_weight_count,
            },
            "rows": rows,
            "ranking": ranking,
            "gates": gates,
        }
        validate_document(document)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not gates["overall_passed"]:
            raise RuntimeError("mixed-precision sensitivity integrity gate failed")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
