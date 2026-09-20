#!/usr/bin/env python3
"""Search safe cumulative quantized-layer subsets from a BF16 base."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from run_layer_precision_search import _copy_layer
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
from run_quantization_compensation_search import (
    _expert_parameters,
    _quantize_dequantize,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.expert_pack import ExpertPackReader


QUANTIZED_BITS = [4, 8, 12]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--compensation-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["relative_perplexity_change"] <= 0.01
        and metrics["router_exact_set_agreement"] >= 0.99
    )


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite reverse layer evidence")
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
        raise RuntimeError("reverse layer search requires a BF16 device")
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
    bf16_backups = [
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
        raise RuntimeError("compensation evidence used a different ExpertPack")
    for layer_index in range(LAYER_COUNT):
        _copy_q4_layer(model, reader, layer_index, torch)
    q4_backups = [
        (parameter, parameter.detach().clone()) for parameter, _ in bf16_backups
    ]
    with torch.inference_mode():
        for parameter, backup in bf16_backups:
            parameter.copy_(backup)
    source_by_bits = {
        row["bits"]: row
        for row in source["candidate_rows"]
        if row["candidate_id"] in {"q4-g128", "q8-g128", "q12-g128"}
    }
    source_rows = [
        {
            "quantized_bits": bits,
            "candidate_id": f"q{bits}-g128",
            "metrics": source_by_bits[bits]["metrics"],
        }
        for bits in QUANTIZED_BITS
    ]
    q4_bpw = source["storage"]["all_q4_pack_effective_bpw"]
    searches = []
    endpoints_reproduced = True
    try:
        for bits in QUANTIZED_BITS:
            print(f"building Q{bits} reverse candidates", flush=True)
            if bits == 4:
                quantized_backups = q4_backups
                quantized_bpw = q4_bpw
            else:
                quantized_backups = [
                    (
                        parameter,
                        _quantize_dequantize(
                            backup,
                            bits=bits,
                            group_size=128,
                            residual_values_per_group=0,
                            torch=torch,
                        ),
                    )
                    for parameter, backup in bf16_backups
                ]
                quantized_bpw = bits + 0.25
            singles = []
            for layer_index in range(LAYER_COUNT):
                print(f"Q{bits} single quantized layer {layer_index}/15", flush=True)
                with torch.inference_mode():
                    _copy_layer(bf16_backups, quantized_backups, layer_index)
                capture = _capture_quality(model, prepared, torch)
                metrics = _metrics(capture, reference, torch)
                singles.append(
                    {
                        "quantized_layer": layer_index,
                        "metrics": metrics,
                        "finite": bool(metrics["finite"]),
                    }
                )
                with torch.inference_mode():
                    _copy_layer(bf16_backups, bf16_backups, layer_index)
            ranking = [
                row["quantized_layer"]
                for row in sorted(
                    singles,
                    key=lambda row: (
                        -row["metrics"]["router_exact_set_agreement"],
                        row["metrics"]["relative_perplexity_change"],
                        row["quantized_layer"],
                    ),
                )
            ]
            cumulative = []
            quantized_layers = []
            for layer_index in ranking:
                print(f"Q{bits} cumulative quantize layer {layer_index}", flush=True)
                with torch.inference_mode():
                    _copy_layer(bf16_backups, quantized_backups, layer_index)
                quantized_layers.append(layer_index)
                capture = _capture_quality(model, prepared, torch)
                metrics = _metrics(capture, reference, torch)
                effective_bpw = (
                    16.0 - len(quantized_layers) * (16.0 - quantized_bpw) / 16
                )
                cumulative.append(
                    {
                        "added_quantized_layer": layer_index,
                        "quantized_layers": list(quantized_layers),
                        "metrics": metrics,
                        "effective_bpw": effective_bpw,
                        "projected_payload_bytes": math.ceil(
                            effective_bpw * total_weights / 8
                        ),
                        "finite": bool(metrics["finite"]),
                    }
                )
            source_metrics = source_by_bits[bits]["metrics"]
            endpoint = cumulative[-1]["metrics"]
            endpoints_reproduced = (
                endpoints_reproduced
                and math.isclose(endpoint["nll"], source_metrics["nll"], abs_tol=1e-6)
                and math.isclose(
                    endpoint["router_exact_set_agreement"],
                    source_metrics["router_exact_set_agreement"],
                    abs_tol=1e-12,
                )
            )
            qualifying = [row for row in cumulative if _passes(row["metrics"])]
            lowest = (
                min(qualifying, key=lambda row: row["effective_bpw"])
                if qualifying
                else None
            )
            searches.append(
                {
                    "quantized_bits": bits,
                    "quantized_effective_bpw": quantized_bpw,
                    "single_layer_rows": singles,
                    "ranking": ranking,
                    "cumulative_rows": cumulative,
                    "lowest_bpw_passing_quantized_layers": (
                        lowest["quantized_layers"] if lowest else None
                    ),
                }
            )
            with torch.inference_mode():
                for parameter, backup in bf16_backups:
                    parameter.copy_(backup)
            if bits != 4:
                del quantized_backups
    finally:
        del bf16_backups, q4_backups
        reader.close()

    passing_policies = [
        {
            "quantized_bits": search["quantized_bits"],
            "quantized_layers": row["quantized_layers"],
            "effective_bpw": row["effective_bpw"],
        }
        for search in searches
        for row in search["cumulative_rows"]
        if _passes(row["metrics"])
    ]
    lowest_policy = (
        min(
            passing_policies,
            key=lambda row: (
                row["effective_bpw"],
                row["quantized_bits"],
                -len(row["quantized_layers"]),
            ),
        )
        if passing_policies
        else None
    )
    gates = {
        "source_evidence_compatible": source_compatible,
        "dataset_identity": dataset_identity,
        "reference_finite": bool(reference["finite"]),
        "uniform_endpoints_reproduced": endpoints_reproduced,
        "all_layers_covered": all(
            [row["quantized_layer"] for row in search["single_layer_rows"]]
            == list(range(16))
            and sorted(search["ranking"]) == list(range(16))
            for search in searches
        ),
        "matrix_finite": all(
            all(
                row["finite"]
                for row in search["single_layer_rows"] + search["cumulative_rows"]
            )
            for search in searches
        ),
        "quality_gate_unchanged": True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "reverse_layer_quantization_search",
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": source["model"],
        "method": {
            "id": "bf16-base-cumulative-quantized-layer-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "quantized_bits": QUANTIZED_BITS,
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
            "q4_pack_effective_bpw": q4_bpw,
        },
        "searches": searches,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "lowest_bpw_passing_policy": lowest_policy,
        },
        "gates": gates,
    }
    validate_document(document)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not gates["overall_passed"]:
        raise RuntimeError("reverse layer integrity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
