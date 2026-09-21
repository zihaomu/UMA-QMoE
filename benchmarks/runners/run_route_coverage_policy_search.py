#!/usr/bin/env python3
"""Search per-layer BF16 expert policies from calibration route coverage."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from run_mixed_precision_refinement import (
    _copy_bf16_expert,
    _storage_by_prefix,
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


COVERAGE_THRESHOLDS = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-id", choices=("halo3", "local-halo", "spark1"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--policy-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _route_counts(capture: dict[str, Any], torch: Any) -> list[list[int]]:
    return [
        torch.bincount(
            capture["routes"][layer_index].reshape(-1), minlength=EXPERT_COUNT
        ).tolist()
        for layer_index in range(LAYER_COUNT)
    ]


def _selection(counts: list[int], threshold: float) -> tuple[list[int], float]:
    total = sum(counts)
    if total <= 0:
        raise RuntimeError("calibration route layer has no assignments")
    ranking = sorted(range(EXPERT_COUNT), key=lambda expert: (-counts[expert], expert))
    selected = []
    covered = 0
    for expert in ranking:
        if counts[expert] == 0:
            break
        selected.append(expert)
        covered += counts[expert]
        if covered / total >= threshold:
            break
    return selected, covered / total


def _candidate_selections(
    route_counts: list[list[int]],
) -> list[tuple[str, str, float | None, list[tuple[list[int], float]]]]:
    candidates = []
    for threshold in COVERAGE_THRESHOLDS:
        candidates.append(
            (
                f"coverage-{round(threshold * 100)}-bf16",
                "route_coverage",
                threshold,
                [_selection(counts, threshold) for counts in route_counts],
            )
        )
    candidates.append(
        (
            "all-experts-bf16",
            "all_experts",
            None,
            [(list(range(EXPERT_COUNT)), 1.0) for _ in range(LAYER_COUNT)],
        )
    )
    return candidates


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite route coverage policy evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.policy_evidence.read_text(encoding="utf-8"))
    validate_document(source)
    if (
        source.get("kind") != "mixed_precision_policy_search"
        or source.get("target_id") != args.target_id
        or source.get("status") != "passed"
    ):
        raise RuntimeError("source policy evidence identity is incompatible")
    samples = _load_samples(args.prompt_fixture)
    split = len(samples) // 2
    calibration_samples = samples[:split]
    evaluation_samples = samples[split:]
    if len(calibration_samples) < 2 or len(evaluation_samples) < 2:
        raise RuntimeError("quality fixture cannot form two non-trivial splits")
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
        raise RuntimeError("route coverage policy search requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    calibration_inputs = _prepare_samples(calibration_samples, tokenizer, torch)
    evaluation_inputs = _prepare_samples(evaluation_samples, tokenizer, torch)
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

    calibration_capture = _capture_quality(model, calibration_inputs, torch)
    counts = _route_counts(calibration_capture, torch)
    reference = _capture_quality(model, evaluation_inputs, torch)
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
    if source["model"]["expert_pack_sha256"] != pack_sha256:
        raise RuntimeError("source policy evidence used a different ExpertPack")
    headers = reader.header["tensors"]
    total_weight_count = sum(item["element_count"] for item in headers)
    q4_bytes, bf16_bytes, _weights = _storage_by_prefix(
        headers, "model.layers.0.mlp.experts.0."
    )
    single_expert_extra = bf16_bytes - q4_bytes
    all_q4_bytes = args.expert_pack.stat().st_size
    candidates = _candidate_selections(counts)
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
            all_q4_capture = _capture_quality(model, evaluation_inputs, torch)
            all_q4 = _metrics(all_q4_capture, reference, torch)
            restored = [set() for _ in range(LAYER_COUNT)]
            rows = []
            for policy_id, mode, threshold, selections in candidates:
                print(f"evaluating {policy_id}", flush=True)
                for layer_index, (expert_ids, _coverage) in enumerate(selections):
                    for expert_index in expert_ids:
                        if expert_index not in restored[layer_index]:
                            _copy_bf16_expert(
                                model,
                                sources,
                                weight_map,
                                layer_index,
                                expert_index,
                                torch,
                            )
                            restored[layer_index].add(expert_index)
                capture = _capture_quality(model, evaluation_inputs, torch)
                metrics = _metrics(capture, reference, torch)
                restored_count = sum(len(experts) for experts in restored)
                extra_bytes = restored_count * single_expert_extra
                mixed_bytes = all_q4_bytes + extra_bytes
                rows.append(
                    {
                        "policy_id": policy_id,
                        "mode": mode,
                        "coverage_threshold": threshold,
                        "layers": [
                            {
                                "layer_index": layer_index,
                                "expert_ids": expert_ids,
                                "assignment_coverage": coverage,
                            }
                            for layer_index, (expert_ids, coverage) in enumerate(
                                selections
                            )
                        ],
                        "restored_expert_count": restored_count,
                        "metrics": metrics,
                        "extra_bytes": extra_bytes,
                        "mixed_bytes": mixed_bytes,
                        "effective_bpw": mixed_bytes * 8 / total_weight_count,
                        "finite": bool(metrics["finite"]),
                    }
                )
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"] <= 0.01
            and row["metrics"]["router_exact_set_agreement"] >= 0.99
        ]
        upper = rows[-1]["metrics"]
        split_disjoint = not bool(
            {sample["id"] for sample in calibration_samples}
            & {sample["id"] for sample in evaluation_samples}
        )
        gates = {
            "reference_finite": bool(reference["finite"]),
            "all_q4_finite": bool(all_q4["finite"]),
            "dataset_split_disjoint": split_disjoint,
            "calibration_complete": len(counts) == LAYER_COUNT,
            "candidate_progression": len(rows) == len(COVERAGE_THRESHOLDS) + 1,
            "matrix_finite": all(row["finite"] for row in rows),
            "all_experts_upper_bound": math.isclose(
                upper["relative_perplexity_change"], 0.0, abs_tol=1e-12
            )
            and math.isclose(upper["router_exact_set_agreement"], 1.0)
            and math.isclose(upper["logit_max_absolute_error"], 0.0, abs_tol=1e-12),
            "quality_gate_unchanged": True,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "route_coverage_policy_search",
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
                "id": "per-layer-reference-route-coverage-bf16-v1",
                "diagnostic_only": True,
                "materializes_expert_parameters": True,
                "performance_evidence": False,
                "coverage_thresholds": COVERAGE_THRESHOLDS,
                "source_policy_evidence": {
                    "file_sha256": _sha256_file(args.policy_evidence),
                    "semantic_sha256": canonical_sha256(source),
                },
                "prompt_fixture": {
                    "file_sha256": _sha256_file(args.prompt_fixture),
                    "semantic_sha256": _fixture_semantic_sha256(samples),
                },
            },
            "dataset": {
                "calibration_sample_ids": [
                    sample["id"] for sample in calibration_samples
                ],
                "evaluation_sample_ids": [
                    sample["id"] for sample in evaluation_samples
                ],
                "evaluation_prompt_token_count": reference["prompt_token_count"],
                "evaluation_target_token_count": reference["target_token_count"],
            },
            "calibration": {
                "route_counts_by_layer": [
                    {
                        "layer_index": layer_index,
                        "expert_counts": layer_counts,
                        "total_assignments": sum(layer_counts),
                    }
                    for layer_index, layer_counts in enumerate(counts)
                ]
            },
            "reference": {
                "finite": bool(reference["finite"]),
                "nll": float(reference["nll"]),
                "perplexity": math.exp(float(reference["nll"])),
                "logits_sha256": _tensor_sha256(reference["logits"]),
                "routes_sha256": _routes_sha256(reference["routes"]),
            },
            "all_q4_baseline": all_q4,
            "storage": {
                "all_q4_bytes": all_q4_bytes,
                "total_expert_weight_count": total_weight_count,
                "single_expert_extra_bytes": single_expert_extra,
            },
            "candidate_rows": rows,
            "quality_gate": {
                "maximum_relative_perplexity_increase": 0.01,
                "minimum_router_exact_set_agreement": 0.99,
                "first_passing_policy_id": (
                    passing[0]["policy_id"] if passing else None
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
            raise RuntimeError("route coverage policy integrity gate failed")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
