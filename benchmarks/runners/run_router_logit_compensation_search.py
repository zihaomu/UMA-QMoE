#!/usr/bin/env python3
"""Fit tiny post-Q4 router-logit corrections on a disjoint calibration split."""

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


EXPERT_COUNT = 64
TOP_K = 8
REGULARIZATION = 1e-4
CANDIDATE_SPECS = [
    ("q4-baseline", "identity", None, None),
    ("bias", "bias", None, None),
    ("diagonal-affine", "diagonal_affine", None, None),
    ("ridge-delta-r8-l1e-4", "ridge_delta", 8, REGULARIZATION),
    ("ridge-delta-r16-l1e-4", "ridge_delta", 16, REGULARIZATION),
    ("ridge-delta-r32-l1e-4", "ridge_delta", 32, REGULARIZATION),
    ("ridge-delta-full-l1e-4", "ridge_delta_full", None, REGULARIZATION),
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-id", choices=("halo3", "local-halo", "spark1"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--route-coverage-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _capture_router_logits(
    model: Any, prepared: list[dict[str, Any]], torch: Any
) -> dict[int, Any]:
    """Capture raw gate logits in deterministic sample/token order."""

    captured: dict[int, list[Any]] = {layer: [] for layer in range(LAYER_COUNT)}
    current: dict[int, Any] = {}

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE router hook returned an unexpected value")
            current[layer_index] = result[0].detach().float().cpu()

        return hook

    handles = [
        layer.mlp.gate.register_forward_hook(gate_hook(layer_index))
        for layer_index, layer in enumerate(model.model.layers)
    ]
    try:
        for sample in prepared:
            current.clear()
            with torch.inference_mode():
                model(input_ids=sample["input_ids"], use_cache=False)
            torch.cuda.synchronize()
            if set(current) != set(range(LAYER_COUNT)):
                raise RuntimeError("router calibration did not observe every layer")
            for layer_index in range(LAYER_COUNT):
                captured[layer_index].append(current[layer_index])
    finally:
        for handle in handles:
            handle.remove()
    result = {layer: torch.cat(values, dim=0) for layer, values in captured.items()}
    token_counts = {tensor.shape[0] for tensor in result.values()}
    if len(token_counts) != 1 or any(
        tensor.ndim != 2 or tensor.shape[1] != EXPERT_COUNT
        for tensor in result.values()
    ):
        raise RuntimeError("router calibration produced inconsistent shapes")
    return result


def _fit_layer_transform(
    source: Any,
    reference: Any,
    *,
    transform: str,
    rank: int | None,
    regularization: float | None,
    torch: Any,
) -> dict[str, Any]:
    """Fit delta(reference - Q4) in float64 on CPU."""

    x = source.to(dtype=torch.float64, device="cpu")
    delta = reference.to(dtype=torch.float64, device="cpu") - x
    if transform == "identity":
        return {"transform": transform}
    if transform == "bias":
        return {"transform": transform, "bias": delta.mean(dim=0).float()}
    x_mean = x.mean(dim=0)
    delta_mean = delta.mean(dim=0)
    x_centered = x - x_mean
    delta_centered = delta - delta_mean
    if transform == "diagonal_affine":
        denominator = x_centered.square().sum(dim=0).clamp_min(1e-12)
        slope = (x_centered * delta_centered).sum(dim=0) / denominator
        bias = delta_mean - x_mean * slope
        return {
            "transform": transform,
            "slope": slope.float(),
            "bias": bias.float(),
        }
    if transform not in {"ridge_delta", "ridge_delta_full"}:
        raise ValueError(f"unsupported router transform {transform!r}")
    if regularization is None:
        raise ValueError("ridge transform requires regularization")
    gram = x_centered.T @ x_centered
    scale = max(float(torch.trace(gram).item()) / EXPERT_COUNT, 1e-12)
    system = gram + torch.eye(EXPERT_COUNT, dtype=torch.float64) * (
        regularization * scale
    )
    weight = torch.linalg.solve(system, x_centered.T @ delta_centered)
    bias = delta_mean - x_mean @ weight
    if transform == "ridge_delta_full":
        return {
            "transform": transform,
            "weight": weight.float(),
            "bias": bias.float(),
        }
    if rank is None:
        raise ValueError("low-rank ridge transform requires a rank")
    left, singular, right = torch.linalg.svd(weight, full_matrices=False)
    return {
        "transform": transform,
        "left": (left[:, :rank] * singular[:rank]).float(),
        "right": right[:rank, :].float(),
        "bias": bias.float(),
    }


def _apply_layer_transform(logits: Any, transform: dict[str, Any]) -> Any:
    kind = transform["transform"]
    if kind == "identity":
        return logits
    if kind == "bias":
        return logits + transform["bias"]
    if kind == "diagonal_affine":
        return logits + logits * transform["slope"] + transform["bias"]
    if kind == "ridge_delta":
        return (
            logits
            + (logits @ transform["left"]) @ transform["right"]
            + transform["bias"]
        )
    if kind == "ridge_delta_full":
        return logits + logits @ transform["weight"] + transform["bias"]
    raise ValueError(f"unsupported router transform {kind!r}")


def _fit_candidate(
    source: dict[int, Any],
    reference: dict[int, Any],
    *,
    transform: str,
    rank: int | None,
    regularization: float | None,
    torch: Any,
) -> dict[int, dict[str, Any]]:
    return {
        layer_index: _fit_layer_transform(
            source[layer_index],
            reference[layer_index],
            transform=transform,
            rank=rank,
            regularization=regularization,
            torch=torch,
        )
        for layer_index in range(LAYER_COUNT)
    }


def _router_exact(
    candidate: dict[int, Any], reference: dict[int, Any], torch: Any
) -> float:
    exact = 0
    total = 0
    for layer_index in range(LAYER_COUNT):
        candidate_routes = (
            torch.topk(candidate[layer_index], TOP_K, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        reference_routes = (
            torch.topk(reference[layer_index], TOP_K, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        matches = (candidate_routes == reference_routes).all(dim=-1)
        exact += int(matches.sum().item())
        total += matches.numel()
    return exact / total


def _transformed_logits(
    source: dict[int, Any], transforms: dict[int, dict[str, Any]]
) -> dict[int, Any]:
    return {
        layer_index: _apply_layer_transform(
            source[layer_index], transforms[layer_index]
        )
        for layer_index in range(LAYER_COUNT)
    }


def _install_candidate(
    model: Any, transforms: dict[int, dict[str, Any]], torch: Any
) -> list[Any]:
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        device_transform = {
            name: (
                value.to(device="cuda:0", dtype=torch.float32)
                if hasattr(value, "to")
                else value
            )
            for name, value in transforms[layer_index].items()
        }

        def hook(
            module: Any,
            _inputs: Any,
            result: Any,
            *,
            fitted: dict[str, Any] = device_transform,
        ) -> Any:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE router hook returned an unexpected value")
            original_logits = result[0]
            corrected = _apply_layer_transform(original_logits.float(), fitted)
            probabilities = torch.nn.functional.softmax(
                corrected, dtype=torch.float32, dim=-1
            )
            scores, indices = torch.topk(probabilities, module.top_k, dim=-1)
            if module.norm_topk_prob:
                scores = scores / scores.sum(dim=-1, keepdim=True)
            return (
                corrected.to(original_logits.dtype),
                scores.to(original_logits.dtype),
                indices,
            )

        handles.append(layer.mlp.gate.register_forward_hook(hook))
    return handles


def _parameter_count(transform: str, rank: int | None) -> int:
    if transform == "identity":
        per_layer = 0
    elif transform == "bias":
        per_layer = EXPERT_COUNT
    elif transform == "diagonal_affine":
        per_layer = EXPERT_COUNT * 2
    elif transform == "ridge_delta":
        if rank is None:
            raise ValueError("low-rank transform requires a rank")
        per_layer = EXPERT_COUNT * rank + rank * EXPERT_COUNT + EXPERT_COUNT
    elif transform == "ridge_delta_full":
        per_layer = EXPERT_COUNT * EXPERT_COUNT + EXPERT_COUNT
    else:
        raise ValueError(f"unsupported router transform {transform!r}")
    return LAYER_COUNT * per_layer


def _router_logits_sha256(values: dict[int, Any], torch: Any) -> str:
    return _tensor_sha256(
        torch.cat([values[layer] for layer in range(LAYER_COUNT)], dim=0)
    )


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite router compensation evidence")
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
    split = len(samples) // 2
    calibration_samples = samples[:split]
    evaluation_samples = samples[split:]
    calibration_ids = [sample["id"] for sample in calibration_samples]
    evaluation_ids = [sample["id"] for sample in evaluation_samples]
    dataset_identity = (
        calibration_ids == source["dataset"]["calibration_sample_ids"]
        and evaluation_ids == source["dataset"]["evaluation_sample_ids"]
    )
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
        raise RuntimeError("router compensation search requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    calibration_prepared = _prepare_samples(calibration_samples, tokenizer, torch)
    evaluation_prepared = _prepare_samples(evaluation_samples, tokenizer, torch)
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
    if observed != ("olmoe", LAYER_COUNT, EXPERT_COUNT, TOP_K):
        raise RuntimeError(f"unexpected fixed OLMoE architecture {observed!r}")

    print("capturing BF16 calibration and held-out reference", flush=True)
    reference_calibration = _capture_router_logits(model, calibration_prepared, torch)
    reference = _capture_quality(model, evaluation_prepared, torch)
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    pack_sha256 = _sha256_file(args.expert_pack)
    if source["model"]["expert_pack_sha256"] != pack_sha256:
        raise RuntimeError("route coverage evidence used a different ExpertPack")
    total_weights = source["storage"]["total_expert_weight_count"]
    all_q4_bytes = source["storage"]["all_q4_bytes"]
    try:
        for layer_index in range(LAYER_COUNT):
            print(f"quantizing layer {layer_index}/15", flush=True)
            _copy_q4_layer(model, reader, layer_index, torch)
        print("capturing all-Q4 calibration and held-out baseline", flush=True)
        q4_calibration = _capture_router_logits(model, calibration_prepared, torch)
        all_q4_capture = _capture_quality(model, evaluation_prepared, torch)
        all_q4 = _metrics(all_q4_capture, reference, torch)
        source_baseline = source["all_q4_baseline"]
        q4_reproduced = math.isclose(
            all_q4["nll"], source_baseline["nll"], abs_tol=1e-6
        ) and math.isclose(
            all_q4["router_exact_set_agreement"],
            source_baseline["router_exact_set_agreement"],
            abs_tol=1e-12,
        )
        baseline_calibration_exact = _router_exact(
            q4_calibration, reference_calibration, torch
        )
        rows = []
        for candidate_id, transform, rank, regularization in CANDIDATE_SPECS:
            print(f"evaluating {candidate_id}", flush=True)
            fitted = _fit_candidate(
                q4_calibration,
                reference_calibration,
                transform=transform,
                rank=rank,
                regularization=regularization,
                torch=torch,
            )
            calibration_exact = _router_exact(
                _transformed_logits(q4_calibration, fitted),
                reference_calibration,
                torch,
            )
            if transform == "identity":
                metrics = all_q4
            else:
                handles = _install_candidate(model, fitted, torch)
                try:
                    capture = _capture_quality(model, evaluation_prepared, torch)
                    metrics = _metrics(capture, reference, torch)
                finally:
                    for handle in handles:
                        handle.remove()
            parameter_count = _parameter_count(transform, rank)
            parameter_bytes = parameter_count * 2
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "transform": transform,
                    "rank": rank,
                    "regularization_relative": regularization,
                    "parameter_count": parameter_count,
                    "parameter_bytes_bf16": parameter_bytes,
                    "projected_effective_bpw": (
                        (all_q4_bytes + parameter_bytes) * 8 / total_weights
                    ),
                    "calibration_router_exact_set_agreement": calibration_exact,
                    "metrics": metrics,
                    "finite": bool(metrics["finite"]),
                }
            )
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"] <= 0.01
            and row["metrics"]["router_exact_set_agreement"] >= 0.99
        ]
        gates = {
            "source_evidence_compatible": source_compatible,
            "dataset_identity": dataset_identity,
            "dataset_split_disjoint": not bool(
                set(calibration_ids) & set(evaluation_ids)
            ),
            "reference_finite": bool(reference["finite"]),
            "all_q4_finite": bool(all_q4["finite"]),
            "q4_baseline_reproduced": q4_reproduced,
            "candidate_progression": len(rows) == len(CANDIDATE_SPECS),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": True,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "router_logit_compensation_search",
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
                "id": "post-q4-router-logit-calibration-v1",
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
                "candidate_ids": [spec[0] for spec in CANDIDATE_SPECS],
            },
            "dataset": {
                "calibration_sample_ids": calibration_ids,
                "evaluation_sample_ids": evaluation_ids,
                "calibration_router_token_count": int(
                    reference_calibration[0].shape[0]
                ),
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
            "source_all_q4_baseline": source_baseline,
            "all_q4_baseline": all_q4,
            "calibration": {
                "reference_router_logits_sha256": _router_logits_sha256(
                    reference_calibration, torch
                ),
                "all_q4_router_logits_sha256": _router_logits_sha256(
                    q4_calibration, torch
                ),
                "all_q4_router_exact_set_agreement": baseline_calibration_exact,
            },
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
            raise RuntimeError("router compensation integrity gate failed")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
