#!/usr/bin/env python3
"""Search uniform and sparse-residual expert quantization quality."""

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
    EXPERT_COUNT,
    LAYER_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    _copy_q4_layer,
    _routes_sha256,
    _sha256_file,
    _tensor_sha256,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.expert_pack import ExpertPackReader


CANDIDATES = [
    ("q4-g128", 4, 128, 0),
    ("q4-g128-r1", 4, 128, 1),
    ("q4-g64", 4, 64, 0),
    ("q4-g128-r2", 4, 128, 2),
    ("q4-g32", 4, 32, 0),
    ("q4-g128-r4", 4, 128, 4),
    ("q5-g128", 5, 128, 0),
    ("q4-g128-r8", 4, 128, 8),
    ("q4-g16", 4, 16, 0),
    ("q6-g128", 6, 128, 0),
    ("q4-g128-r16", 4, 128, 16),
    ("q8-g128", 8, 128, 0),
    ("q9-g128", 9, 128, 0),
    ("q10-g128", 10, 128, 0),
    ("q12-g128", 12, 128, 0),
    ("bf16-upper-bound", 16, None, 0),
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--route-coverage-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _expert_parameters(model: Any) -> list[tuple[str, Any]]:
    parameters = []
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        parameters.extend(
            (
                (
                    f"model.layers.{layer_index}.mlp.experts.gate_up_proj",
                    experts.gate_up_proj,
                ),
                (
                    f"model.layers.{layer_index}.mlp.experts.down_proj",
                    experts.down_proj,
                ),
            )
        )
    expected = LAYER_COUNT * 2
    if len(parameters) != expected or any(
        parameter.ndim != 3 or parameter.shape[0] != EXPERT_COUNT
        for _name, parameter in parameters
    ):
        raise RuntimeError("unexpected stacked OLMoE expert parameter layout")
    return parameters


def _quantize_dequantize(
    source: Any,
    *,
    bits: int,
    group_size: int,
    residual_values_per_group: int,
    torch: Any,
) -> Any:
    flat = source.float().reshape(-1)
    if flat.numel() % group_size:
        raise RuntimeError("expert tensor is not divisible by the quantization group")
    groups = flat.reshape(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    maximum = groups.abs().amax(dim=1, keepdim=True)
    scales = maximum / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    restored = torch.round(groups / scales).clamp(-qmax, qmax) * scales
    if residual_values_per_group:
        residual = (groups - restored).abs()
        indices = torch.topk(
            residual,
            k=residual_values_per_group,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        restored.scatter_(1, indices, groups.gather(1, indices))
    return restored.reshape(source.shape).to(source.dtype)


def _restore_sparse_residual(
    source: Any,
    base: Any,
    *,
    group_size: int,
    residual_values_per_group: int,
    torch: Any,
) -> Any:
    source_groups = source.float().reshape(-1, group_size)
    restored = base.float().reshape(-1, group_size).clone()
    residual = (source_groups - restored).abs()
    indices = torch.topk(
        residual,
        k=residual_values_per_group,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    restored.scatter_(1, indices, source_groups.gather(1, indices))
    return restored.reshape(source.shape).to(source.dtype)


def _storage(
    candidate_id: str,
    bits: int,
    group_size: int | None,
    residual_count: int,
    *,
    total_weights: int,
    all_q4_bytes: int,
) -> dict[str, float | int]:
    primary_bpw = float(bits)
    scale_bpw = 0.0 if group_size is None else 32.0 / group_size
    residual_bpw = 0.0 if group_size is None else residual_count * 24.0 / group_size
    pack_bpw = all_q4_bytes * 8 / total_weights
    if candidate_id == "q4-g128":
        effective_bpw = pack_bpw
        payload_bytes = all_q4_bytes
    elif candidate_id.startswith("q4-g128-r"):
        effective_bpw = pack_bpw + residual_bpw
        payload_bytes = math.ceil(effective_bpw * total_weights / 8)
    else:
        effective_bpw = primary_bpw + scale_bpw + residual_bpw
        payload_bytes = math.ceil(effective_bpw * total_weights / 8)
    return {
        "primary_bpw": primary_bpw,
        "scale_bpw": scale_bpw,
        "residual_bpw": residual_bpw,
        "effective_bpw": effective_bpw,
        "projected_payload_bytes": payload_bytes,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite compensation search evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.route_coverage_evidence.read_text(encoding="utf-8"))
    validate_document(source)
    source_compatible = (
        source.get("kind") == "route_coverage_policy_search"
        and source.get("target_id") == args.target_id
        and source.get("status") == "passed"
        and source.get("model", {}).get("model_id") == MODEL_ID
        and source.get("model", {}).get("model_revision") == MODEL_REVISION
    )
    if not source_compatible:
        raise RuntimeError("route coverage evidence identity is incompatible")
    samples = _load_samples(args.prompt_fixture)
    evaluation_samples = samples[len(samples) // 2 :]
    evaluation_ids = [sample["id"] for sample in evaluation_samples]
    dataset_identity = evaluation_ids == source["dataset"]["evaluation_sample_ids"]
    if not dataset_identity:
        raise RuntimeError("quality fixture split differs from route coverage evidence")
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
        raise RuntimeError("compensation search requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prepared = _prepare_samples(evaluation_samples, tokenizer, torch)
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

    reference = _capture_quality(model, prepared, torch)
    parameters = _expert_parameters(model)
    total_weights = sum(parameter.numel() for _name, parameter in parameters)
    if total_weights != source["storage"]["total_expert_weight_count"]:
        raise RuntimeError("expert parameter count differs from source evidence")
    print(f"backing up {len(parameters)} BF16 expert tensors", flush=True)
    backups = [
        (parameter, parameter.detach().clone()) for _name, parameter in parameters
    ]
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    pack_sha256 = _sha256_file(args.expert_pack)
    if source["model"]["expert_pack_sha256"] != pack_sha256:
        raise RuntimeError("route coverage evidence used a different ExpertPack")
    for layer_index in range(LAYER_COUNT):
        _copy_q4_layer(model, reader, layer_index, torch)
    q4_backups = [(parameter, parameter.detach().clone()) for parameter, _ in backups]
    all_q4_bytes = source["storage"]["all_q4_bytes"]
    rows = []
    try:
        for candidate_id, bits, group_size, residual_count in CANDIDATES:
            print(f"evaluating {candidate_id}", flush=True)
            with torch.inference_mode():
                for (parameter, backup), (_q4_parameter, q4_backup) in zip(
                    backups, q4_backups, strict=True
                ):
                    if group_size is None:
                        parameter.copy_(backup)
                    elif candidate_id == "q4-g128":
                        parameter.copy_(q4_backup)
                    elif candidate_id.startswith("q4-g128-r"):
                        parameter.copy_(
                            _restore_sparse_residual(
                                backup,
                                q4_backup,
                                group_size=group_size,
                                residual_values_per_group=residual_count,
                                torch=torch,
                            )
                        )
                    else:
                        parameter.copy_(
                            _quantize_dequantize(
                                backup,
                                bits=bits,
                                group_size=group_size,
                                residual_values_per_group=residual_count,
                                torch=torch,
                            )
                        )
            capture = _capture_quality(model, prepared, torch)
            metrics = _metrics(capture, reference, torch)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "bits": bits,
                    "group_size": group_size,
                    "residual_values_per_group": residual_count,
                    "metrics": metrics,
                    **_storage(
                        candidate_id,
                        bits,
                        group_size,
                        residual_count,
                        total_weights=total_weights,
                        all_q4_bytes=all_q4_bytes,
                    ),
                    "finite": bool(metrics["finite"]),
                }
            )
    finally:
        del backups, q4_backups
        reader.close()

    source_baseline = source["all_q4_baseline"]
    q4_baseline_reproduced = math.isclose(
        rows[0]["metrics"]["nll"], source_baseline["nll"], abs_tol=1e-6
    ) and math.isclose(
        rows[0]["metrics"]["router_exact_set_agreement"],
        source_baseline["router_exact_set_agreement"],
        abs_tol=1e-12,
    )
    passing = [
        row
        for row in rows
        if row["metrics"]["relative_perplexity_change"] <= 0.01
        and row["metrics"]["router_exact_set_agreement"] >= 0.99
    ]
    upper = rows[-1]["metrics"]
    gates = {
        "source_evidence_compatible": source_compatible,
        "dataset_identity": dataset_identity,
        "reference_finite": bool(reference["finite"]),
        "candidate_progression": len(rows) == len(CANDIDATES),
        "matrix_finite": all(row["finite"] for row in rows),
        "q4_baseline_reproduced": q4_baseline_reproduced,
        "bf16_upper_bound": math.isclose(
            upper["relative_perplexity_change"], 0.0, abs_tol=1e-12
        )
        and math.isclose(upper["router_exact_set_agreement"], 1.0)
        and math.isclose(upper["logit_max_absolute_error"], 0.0, abs_tol=1e-12),
        "quality_gate_unchanged": True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "quantization_compensation_search",
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "expert_pack_sha256": source["model"]["expert_pack_sha256"],
        },
        "method": {
            "id": "expert-weight-qformat-compensation-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "source_route_coverage_evidence": {
                "file_sha256": _sha256_file(args.route_coverage_evidence),
                "semantic_sha256": canonical_sha256(source),
            },
            "prompt_fixture": {
                "file_sha256": _sha256_file(args.prompt_fixture),
                "semantic_sha256": _fixture_semantic_sha256(samples),
            },
            "candidate_ids": [candidate[0] for candidate in CANDIDATES],
        },
        "dataset": {
            "sample_ids": evaluation_ids,
            "prompt_token_count": reference["prompt_token_count"],
            "target_token_count": reference["target_token_count"],
        },
        "reference": {
            "finite": bool(reference["finite"]),
            "nll": float(reference["nll"]),
            "perplexity": math.exp(float(reference["nll"])),
            "logits_sha256": _tensor_sha256(reference["logits"]),
            "routes_sha256": _routes_sha256(reference["routes"]),
        },
        "source_all_q4_baseline": source_baseline,
        "storage": {
            "total_expert_weight_count": total_weights,
            "all_q4_pack_bytes": all_q4_bytes,
            "all_q4_pack_effective_bpw": all_q4_bytes * 8 / total_weights,
        },
        "candidate_rows": rows,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_candidate_id": (
                passing[0]["candidate_id"] if passing else None
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
        raise RuntimeError("compensation search integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
