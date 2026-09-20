from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import sys

import pytest

from uma_qmoe.contracts import ContractError, validate_document


RUNNER_DIRECTORY = Path(__file__).parents[2] / "benchmarks" / "runners"
sys.path.insert(0, str(RUNNER_DIRECTORY))
runner = importlib.import_module("run_router_logit_compensation_search")


def _metrics(exact: float) -> dict:
    return {
        "finite": True,
        "nll": 1.0,
        "perplexity": math.e,
        "relative_nll_change": 0.0,
        "relative_perplexity_change": 0.0,
        "top1_agreement": 1.0,
        "logit_max_absolute_error": 0.1,
        "logit_p99_absolute_error": 0.05,
        "logit_cosine_similarity": 0.999,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 0.999,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _document() -> dict:
    total_weights = 10_000_000
    pack_bytes = 5_312_500
    baseline = _metrics(0.7)
    rows = []
    for index, (candidate_id, transform, rank, regularization) in enumerate(
        runner.CANDIDATE_SPECS
    ):
        parameter_count = runner._parameter_count(transform, rank)
        parameter_bytes = parameter_count * 2
        metrics = copy.deepcopy(baseline)
        metrics["router_exact_set_agreement"] = 0.7 + index * 0.049
        metrics["per_layer_router_exact_set_agreement"] = [
            metrics["router_exact_set_agreement"]
        ] * 16
        rows.append(
            {
                "candidate_id": candidate_id,
                "transform": transform,
                "rank": rank,
                "regularization_relative": regularization,
                "parameter_count": parameter_count,
                "parameter_bytes_bf16": parameter_bytes,
                "projected_effective_bpw": (
                    (pack_bytes + parameter_bytes) * 8 / total_weights
                ),
                "calibration_router_exact_set_agreement": min(1.0, 0.75 + index * 0.04),
                "metrics": metrics,
                "finite": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "router_logit_compensation_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "post-q4-router-logit-calibration-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "source_route_coverage_evidence": {
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
            "prompt_fixture": {
                "file_sha256": "d" * 64,
                "semantic_sha256": "e" * 64,
            },
            "candidate_ids": [spec[0] for spec in runner.CANDIDATE_SPECS],
        },
        "dataset": {
            "calibration_sample_ids": ["cal-1", "cal-2"],
            "evaluation_sample_ids": ["eval-1", "eval-2"],
            "calibration_router_token_count": 12,
            "evaluation_prompt_token_count": 10,
            "evaluation_target_token_count": 4,
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.e,
            "logits_sha256": "f" * 64,
            "routes_sha256": "0" * 64,
        },
        "source_all_q4_baseline": copy.deepcopy(baseline),
        "all_q4_baseline": copy.deepcopy(baseline),
        "calibration": {
            "reference_router_logits_sha256": "1" * 64,
            "all_q4_router_logits_sha256": "2" * 64,
            "all_q4_router_exact_set_agreement": 0.75,
        },
        "storage": {
            "total_expert_weight_count": total_weights,
            "all_q4_pack_bytes": pack_bytes,
            "all_q4_pack_effective_bpw": pack_bytes * 8 / total_weights,
        },
        "candidate_rows": rows,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_candidate_id": "ridge-delta-full-l1e-4",
        },
        "gates": {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "dataset_split_disjoint": True,
            "reference_finite": True,
            "all_q4_finite": True,
            "q4_baseline_reproduced": True,
            "candidate_progression": True,
            "matrix_finite": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_router_compensation_contract_accepts_complete_search() -> None:
    validate_document(_document())


def test_router_compensation_parameter_accounting() -> None:
    assert runner._parameter_count("identity", None) == 0
    assert runner._parameter_count("bias", None) == 1024
    assert runner._parameter_count("ridge_delta", 8) == 17_408
    assert runner._parameter_count("ridge_delta_full", None) == 66_560


def test_router_compensation_rejects_split_leakage() -> None:
    document = _document()
    document["dataset"]["evaluation_sample_ids"][0] = "cal-1"
    with pytest.raises(ContractError, match="disjoint"):
        validate_document(document)


def test_router_compensation_rejects_tampered_storage() -> None:
    document = _document()
    document["candidate_rows"][3]["parameter_bytes_bf16"] += 2
    with pytest.raises(ContractError, match="storage"):
        validate_document(document)


def test_router_compensation_rejects_false_quality_winner() -> None:
    document = _document()
    document["quality_gate"]["first_passing_candidate_id"] = None
    with pytest.raises(ContractError, match="quality gate"):
        validate_document(document)
