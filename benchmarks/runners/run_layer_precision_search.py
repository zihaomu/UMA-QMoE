#!/usr/bin/env python3
"""Search cumulative BF16 layer protection over uniform expert bases."""

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
    _routes_sha256,
    _sha256_file,
    _tensor_sha256,
)
from run_quantization_compensation_search import (
    _expert_parameters,
    _quantize_dequantize,
)
from uma_qmoe.contracts import canonical_sha256, validate_document


BASE_BITS = [8, 9, 12]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compensation-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _copy_layer(
    destination: list[tuple[Any, Any]],
    source: list[tuple[Any, Any]],
    layer_index: int,
) -> None:
    for index in (layer_index * 2, layer_index * 2 + 1):
        destination[index][0].copy_(source[index][1])


def _passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["relative_perplexity_change"] <= 0.01
        and metrics["router_exact_set_agreement"] >= 0.99
    )


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite layer precision evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.compensation_evidence.read_text(encoding="utf-8"))
    validate_document(source)
    source_compatible = (
        source.get("kind") == "quantization_compensation_search"
        and source.get("target_id") == args.target_id
        and source.get("status") == "passed"
        and source.get("model", {}).get("model_id") == MODEL_ID
        and source.get("model", {}).get("model_revision") == MODEL_REVISION
    )
    if not source_compatible:
        raise RuntimeError("compensation evidence identity is incompatible")
    samples = _load_samples(args.prompt_fixture)
    evaluation_samples = samples[len(samples) // 2 :]
    evaluation_ids = [sample["id"] for sample in evaluation_samples]
    dataset_identity = evaluation_ids == source["dataset"]["sample_ids"]
    if not dataset_identity:
        raise RuntimeError("quality fixture differs from compensation evidence")
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
        raise RuntimeError("layer precision search requires a BF16 device")
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
    if total_weights % LAYER_COUNT:
        raise RuntimeError("expert weights are not evenly partitioned by layer")
    backups = [
        (parameter, parameter.detach().clone()) for _name, parameter in parameters
    ]
    source_by_bits = {
        row["bits"]: row
        for row in source["candidate_rows"]
        if row["candidate_id"] in {"q8-g128", "q9-g128", "q12-g128"}
    }
    source_rows = [
        {
            "base_bits": bits,
            "candidate_id": f"q{bits}-g128",
            "metrics": source_by_bits[bits]["metrics"],
        }
        for bits in BASE_BITS
    ]
    searches = []
    base_reproduced = True
    try:
        for bits in BASE_BITS:
            print(f"building Q{bits} base", flush=True)
            with torch.inference_mode():
                for parameter, backup in backups:
                    parameter.copy_(
                        _quantize_dequantize(
                            backup,
                            bits=bits,
                            group_size=128,
                            residual_values_per_group=0,
                            torch=torch,
                        )
                    )
            base_backups = [
                (parameter, parameter.detach().clone()) for parameter, _ in backups
            ]
            base_capture = _capture_quality(model, prepared, torch)
            base_metrics = _metrics(base_capture, reference, torch)
            source_metrics = source_by_bits[bits]["metrics"]
            base_reproduced = (
                base_reproduced
                and math.isclose(
                    base_metrics["nll"], source_metrics["nll"], abs_tol=1e-6
                )
                and math.isclose(
                    base_metrics["router_exact_set_agreement"],
                    source_metrics["router_exact_set_agreement"],
                    abs_tol=1e-12,
                )
            )
            singles = []
            for layer_index in range(LAYER_COUNT):
                print(f"Q{bits} single BF16 layer {layer_index}/15", flush=True)
                with torch.inference_mode():
                    _copy_layer(backups, backups, layer_index)
                capture = _capture_quality(model, prepared, torch)
                metrics = _metrics(capture, reference, torch)
                singles.append(
                    {
                        "restored_layer": layer_index,
                        "metrics": metrics,
                        "finite": bool(metrics["finite"]),
                    }
                )
                with torch.inference_mode():
                    _copy_layer(backups, base_backups, layer_index)
            ranking = [
                row["restored_layer"]
                for row in sorted(
                    singles,
                    key=lambda row: (
                        -row["metrics"]["router_exact_set_agreement"],
                        row["metrics"]["relative_perplexity_change"],
                        row["restored_layer"],
                    ),
                )
            ]
            cumulative = []
            restored = []
            base_bpw = bits + 0.25
            for layer_index in ranking:
                print(f"Q{bits} cumulative add layer {layer_index}", flush=True)
                with torch.inference_mode():
                    _copy_layer(backups, backups, layer_index)
                restored.append(layer_index)
                capture = _capture_quality(model, prepared, torch)
                metrics = _metrics(capture, reference, torch)
                effective_bpw = base_bpw + len(restored) * (16.0 - base_bpw) / 16
                cumulative.append(
                    {
                        "added_layer": layer_index,
                        "restored_layers": list(restored),
                        "metrics": metrics,
                        "effective_bpw": effective_bpw,
                        "projected_payload_bytes": math.ceil(
                            effective_bpw * total_weights / 8
                        ),
                        "finite": bool(metrics["finite"]),
                    }
                )
            qualifying = [row for row in cumulative if _passes(row["metrics"])]
            searches.append(
                {
                    "base_bits": bits,
                    "base_effective_bpw": base_bpw,
                    "base_metrics": base_metrics,
                    "single_layer_rows": singles,
                    "ranking": ranking,
                    "cumulative_rows": cumulative,
                    "first_passing_restored_layers": (
                        qualifying[0]["restored_layers"] if qualifying else None
                    ),
                }
            )
            del base_backups
    finally:
        del backups

    passing_policies = []
    for search in searches:
        if _passes(search["base_metrics"]):
            passing_policies.append(
                {
                    "base_bits": search["base_bits"],
                    "restored_layers": [],
                    "effective_bpw": search["base_effective_bpw"],
                }
            )
        passing_policies.extend(
            {
                "base_bits": search["base_bits"],
                "restored_layers": row["restored_layers"],
                "effective_bpw": row["effective_bpw"],
            }
            for row in search["cumulative_rows"]
            if _passes(row["metrics"])
        )
    lowest = (
        min(
            passing_policies,
            key=lambda row: (
                row["effective_bpw"],
                row["base_bits"],
                len(row["restored_layers"]),
            ),
        )
        if passing_policies
        else None
    )
    matrix_finite = all(
        search["base_metrics"]["finite"]
        and all(
            row["finite"]
            for row in search["single_layer_rows"] + search["cumulative_rows"]
        )
        for search in searches
    )
    upper_bounds = all(
        math.isclose(
            search["cumulative_rows"][-1]["metrics"]["relative_perplexity_change"],
            0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            search["cumulative_rows"][-1]["metrics"]["router_exact_set_agreement"],
            1.0,
        )
        and math.isclose(
            search["cumulative_rows"][-1]["metrics"]["logit_max_absolute_error"],
            0.0,
            abs_tol=1e-12,
        )
        for search in searches
    )
    gates = {
        "source_evidence_compatible": source_compatible,
        "dataset_identity": dataset_identity,
        "reference_finite": bool(reference["finite"]),
        "base_metrics_reproduced": base_reproduced,
        "all_layers_covered": all(
            [row["restored_layer"] for row in search["single_layer_rows"]]
            == list(range(16))
            and sorted(search["ranking"]) == list(range(16))
            for search in searches
        ),
        "matrix_finite": matrix_finite,
        "bf16_upper_bounds": upper_bounds,
        "quality_gate_unchanged": True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "layer_precision_search",
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": source["model"],
        "method": {
            "id": "uniform-base-cumulative-bf16-layer-restore-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "base_bits": BASE_BITS,
            "source_compensation_evidence": {
                "file_sha256": _sha256_file(args.compensation_evidence),
                "semantic_sha256": canonical_sha256(source),
            },
            "prompt_fixture": {
                "file_sha256": _sha256_file(args.prompt_fixture),
                "semantic_sha256": _fixture_semantic_sha256(samples),
            },
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
        "source_uniform_metrics": source_rows,
        "storage": {
            "total_expert_weight_count": total_weights,
            "expert_weight_count_per_layer": total_weights // LAYER_COUNT,
        },
        "searches": searches,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "lowest_bpw_passing_policy": lowest,
        },
        "gates": gates,
    }
    validate_document(document)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not gates["overall_passed"]:
        raise RuntimeError("layer precision integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
