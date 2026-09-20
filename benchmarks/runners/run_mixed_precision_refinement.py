#!/usr/bin/env python3
"""Refine OLMoE Q4 quality with cumulative layers and layer-2 experts."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from run_mixed_precision_sensitivity import (
    EXPERT_COUNT,
    INTERMEDIATE_SIZE,
    LAYER_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    _capture,
    _copy_bf16_layer,
    _copy_q4_layer,
    _metrics,
    _prompt,
    _routes_sha256,
    _sha256_file,
    _tensor_sha256,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.expert_pack import ExpertPackReader
from uma_qmoe.q4 import dequantize_q4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--single-layer-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "router_exact_set_agreement": metrics["router_exact_set_agreement"]
        - baseline["router_exact_set_agreement"],
        "router_mean_set_overlap": metrics["router_mean_set_overlap"]
        - baseline["router_mean_set_overlap"],
        "logit_cosine_similarity": metrics["logit_cosine_similarity"]
        - baseline["logit_cosine_similarity"],
    }


def _source_tensor(
    sources: dict[str, Any], weight_map: dict[str, str], name: str, torch: Any
) -> Any:
    value = sources[weight_map[name]].get_tensor(name)
    if value.dtype != torch.bfloat16:
        raise RuntimeError(f"refinement BF16 source {name!r} has wrong dtype")
    return value.to(device="cuda:0", non_blocking=False)


def _copy_bf16_expert(
    model: Any,
    sources: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    expert_index: int,
    torch: Any,
) -> None:
    experts = model.model.layers[layer_index].mlp.experts
    prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
    gate = _source_tensor(sources, weight_map, f"{prefix}.gate_proj.weight", torch)
    up = _source_tensor(sources, weight_map, f"{prefix}.up_proj.weight", torch)
    down = _source_tensor(sources, weight_map, f"{prefix}.down_proj.weight", torch)
    with torch.no_grad():
        experts.gate_up_proj[expert_index, :INTERMEDIATE_SIZE].copy_(gate)
        experts.gate_up_proj[expert_index, INTERMEDIATE_SIZE:].copy_(up)
        experts.down_proj[expert_index].copy_(down)
    del gate, up, down


def _copy_q4_expert(
    model: Any,
    reader: ExpertPackReader,
    layer_index: int,
    expert_index: int,
    torch: Any,
) -> None:
    experts = model.model.layers[layer_index].mlp.experts
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
    with torch.no_grad():
        experts.gate_up_proj[expert_index, :INTERMEDIATE_SIZE].copy_(gate)
        experts.gate_up_proj[expert_index, INTERMEDIATE_SIZE:].copy_(up)
        experts.down_proj[expert_index].copy_(down)
    del gate, up, down


def _route_count(capture: dict[str, Any], expert_index: int) -> int:
    routes = capture["routes"][2]
    return int((routes == expert_index).sum().item())


def _storage_by_prefix(
    headers: list[dict[str, Any]], prefix: str
) -> tuple[int, int, int]:
    selected = [item for item in headers if item["name"].startswith(prefix)]
    if not selected:
        raise RuntimeError(f"ExpertPack has no tensors below {prefix!r}")
    q4_bytes = sum(
        item["storage_end_offset"] - item["packed_offset"] for item in selected
    )
    weight_count = sum(item["element_count"] for item in selected)
    return q4_bytes, weight_count * 2, weight_count


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite mixed-precision refinement evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_evidence = json.loads(
        args.single_layer_evidence.read_text(encoding="utf-8")
    )
    validate_document(source_evidence)
    if (
        source_evidence.get("kind") != "mixed_precision_sensitivity"
        or source_evidence.get("target_id") != args.target_id
        or source_evidence.get("status") != "passed"
    ):
        raise RuntimeError("single-layer sensitivity evidence identity is incompatible")
    layer_order = source_evidence["ranking"]
    if layer_order[0] != 2 or sorted(layer_order) != list(range(LAYER_COUNT)):
        raise RuntimeError("single-layer ranking must cover all layers starting at 2")
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
        raise RuntimeError("mixed-precision refinement requires a BF16 CUDA/HIP device")
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
    pack_sha256 = _sha256_file(args.expert_pack)
    if source_evidence["model"]["expert_pack_sha256"] != pack_sha256:
        raise RuntimeError("single-layer sensitivity used a different ExpertPack")
    headers = reader.header["tensors"]
    total_weight_count = sum(item["element_count"] for item in headers)
    layer_storage = {
        layer: _storage_by_prefix(headers, f"model.layers.{layer}.mlp.experts.")
        for layer in range(LAYER_COUNT)
    }
    layer2_q4, layer2_bf16, _layer2_weights = layer_storage[2]
    all_q4_bytes = args.expert_pack.stat().st_size
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

            cumulative_rows = []
            restored: list[int] = []
            mixed_bytes = all_q4_bytes
            for layer_index in layer_order:
                print(f"cumulative BF16 layer {layer_index}", flush=True)
                _copy_bf16_layer(model, sources, weight_map, layer_index, torch)
                restored.append(layer_index)
                q4_bytes, bf16_bytes, _weights = layer_storage[layer_index]
                mixed_bytes += bf16_bytes - q4_bytes
                candidate = _capture(model, device_inputs, torch)
                metrics = _metrics(candidate, reference, torch)
                cumulative_rows.append(
                    {
                        "added_layer": layer_index,
                        "restored_layers": list(restored),
                        "metrics": metrics,
                        "delta_vs_all_q4": _delta(metrics, all_q4),
                        "mixed_bytes": mixed_bytes,
                        "effective_bpw": mixed_bytes * 8 / total_weight_count,
                        "finite": bool(metrics["finite"]),
                    }
                )

            for layer_index in range(LAYER_COUNT):
                print(f"resetting Q4 layer {layer_index}/15", flush=True)
                _copy_q4_layer(model, reader, layer_index, torch)

            expert_rows = []
            for expert_index in range(EXPERT_COUNT):
                print(f"restoring BF16 layer2 expert {expert_index}/63", flush=True)
                _copy_bf16_expert(
                    model, sources, weight_map, 2, expert_index, torch
                )
                candidate = _capture(model, device_inputs, torch)
                metrics = _metrics(candidate, reference, torch)
                expert_q4, expert_bf16, _weights = _storage_by_prefix(
                    headers, f"model.layers.2.mlp.experts.{expert_index}."
                )
                expert_rows.append(
                    {
                        "restored_expert": expert_index,
                        "reference_route_count": _route_count(
                            reference, expert_index
                        ),
                        "q4_route_count": _route_count(
                            all_q4_capture, expert_index
                        ),
                        "metrics": metrics,
                        "delta_vs_all_q4": _delta(metrics, all_q4),
                        "extra_bytes": expert_bf16 - expert_q4,
                        "finite": bool(metrics["finite"]),
                    }
                )
                _copy_q4_expert(model, reader, 2, expert_index, torch)

        expert_ranking = [
            row["restored_expert"]
            for row in sorted(
                expert_rows,
                key=lambda row: (
                    -row["delta_vs_all_q4"]["router_exact_set_agreement"],
                    -row["delta_vs_all_q4"]["logit_cosine_similarity"],
                    -row["reference_route_count"],
                    row["restored_expert"],
                ),
            )
        ]
        passing = [
            row
            for row in cumulative_rows
            if row["metrics"]["router_exact_set_agreement"] >= 0.99
        ]
        gates = {
            "reference_finite": bool(reference["finite"]),
            "all_q4_finite": bool(all_q4["finite"]),
            "layer2_first": layer_order[0] == 2,
            "all_layers_covered": len(cumulative_rows) == LAYER_COUNT,
            "all_layer2_experts_covered": len(expert_rows) == EXPERT_COUNT,
            "matrix_finite": all(
                row["finite"] for row in cumulative_rows + expert_rows
            ),
            "quality_gate_unchanged": True,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "mixed_precision_refinement",
            "captured_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "target_id": args.target_id,
            "status": "passed" if gates["overall_passed"] else "failed",
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expert_pack_sha256": pack_sha256,
            },
            "method": {
                "id": "cumulative-layer-and-layer2-expert-bf16-restore-v1",
                "diagnostic_only": True,
                "materializes_expert_parameters": True,
                "performance_evidence": False,
                "source_single_layer_evidence": {
                    "file_sha256": _sha256_file(args.single_layer_evidence),
                    "semantic_sha256": canonical_sha256(source_evidence),
                },
                "layer_order": layer_order,
                "expert_layer": 2,
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
                "total_expert_weight_count": total_weight_count,
                "layer2_q4_bytes": layer2_q4,
                "layer2_bf16_bytes": layer2_bf16,
            },
            "cumulative_layer_rows": cumulative_rows,
            "layer2_expert_rows": expert_rows,
            "layer2_expert_ranking": expert_ranking,
            "quality_gate": {
                "minimum_router_exact_set_agreement": 0.99,
                "first_passing_restored_layers": (
                    passing[0]["restored_layers"] if passing else None
                ),
            },
            "gates": gates,
        }
        validate_document(document)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not gates["overall_passed"]:
            raise RuntimeError("mixed-precision refinement integrity gate failed")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
