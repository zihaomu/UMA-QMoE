from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(nll: float, exact: float, *, zero_error: bool = False) -> dict:
    reference_nll = 1.0
    perplexity = math.exp(nll)
    return {
        "finite": True,
        "nll": nll,
        "perplexity": perplexity,
        "relative_nll_change": (nll / reference_nll) - 1.0,
        "relative_perplexity_change": (perplexity / math.exp(reference_nll)) - 1.0,
        "top1_agreement": 1.0,
        "logit_max_absolute_error": 0.0 if zero_error else 0.1,
        "logit_p99_absolute_error": 0.0 if zero_error else 0.05,
        "logit_cosine_similarity": 1.0 if zero_error else 0.999,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 1.0 if zero_error else 0.999,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _selection(threshold: float) -> tuple[list[int], float]:
    if threshold <= 0.5:
        return [0], 0.5
    if threshold <= 0.75:
        return [0, 1], 0.8
    return [0, 1, 2], 1.0


def _document() -> dict:
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    rows = []
    for index, threshold in enumerate(thresholds):
        experts, coverage = _selection(threshold)
        restored = len(experts) * 16
        extra = restored * 100
        mixed = 1000 + extra
        rows.append(
            {
                "policy_id": f"coverage-{round(threshold * 100)}-bf16",
                "mode": "route_coverage",
                "coverage_threshold": threshold,
                "layers": [
                    {
                        "layer_index": layer,
                        "expert_ids": list(experts),
                        "assignment_coverage": coverage,
                    }
                    for layer in range(16)
                ],
                "restored_expert_count": restored,
                "metrics": _metrics(1.02 - index * 0.002, 0.8 + index * 0.02),
                "extra_bytes": extra,
                "mixed_bytes": mixed,
                "effective_bpw": mixed * 8 / 2000,
                "finite": True,
            }
        )
    extra = 1024 * 100
    mixed = 1000 + extra
    rows.append(
        {
            "policy_id": "all-experts-bf16",
            "mode": "all_experts",
            "coverage_threshold": None,
            "layers": [
                {
                    "layer_index": layer,
                    "expert_ids": list(range(64)),
                    "assignment_coverage": 1.0,
                }
                for layer in range(16)
            ],
            "restored_expert_count": 1024,
            "metrics": _metrics(1.0, 1.0, zero_error=True),
            "extra_bytes": extra,
            "mixed_bytes": mixed,
            "effective_bpw": mixed * 8 / 2000,
            "finite": True,
        }
    )
    return {
        "schema_version": 1,
        "kind": "route_coverage_policy_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "per-layer-reference-route-coverage-bf16-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "coverage_thresholds": thresholds,
            "source_policy_evidence": {
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
            "prompt_fixture": {
                "file_sha256": "d" * 64,
                "semantic_sha256": "e" * 64,
            },
        },
        "dataset": {
            "calibration_sample_ids": ["cal-1", "cal-2"],
            "evaluation_sample_ids": ["eval-1", "eval-2"],
            "evaluation_prompt_token_count": 10,
            "evaluation_target_token_count": 4,
        },
        "calibration": {
            "route_counts_by_layer": [
                {
                    "layer_index": layer,
                    "expert_counts": [5, 3, 2] + [0] * 61,
                    "total_assignments": 10,
                }
                for layer in range(16)
            ]
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.exp(1.0),
            "logits_sha256": "f" * 64,
            "routes_sha256": "0" * 64,
        },
        "all_q4_baseline": _metrics(1.03, 0.7),
        "storage": {
            "all_q4_bytes": 1000,
            "total_expert_weight_count": 2000,
            "single_expert_extra_bytes": 100,
        },
        "candidate_rows": rows,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_policy_id": "all-experts-bf16",
        },
        "gates": {
            "reference_finite": True,
            "all_q4_finite": True,
            "dataset_split_disjoint": True,
            "calibration_complete": True,
            "candidate_progression": True,
            "matrix_finite": True,
            "all_experts_upper_bound": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_route_coverage_contract_accepts_complete_search() -> None:
    validate_document(_document())


def test_route_coverage_contract_rejects_tampered_selection() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][0]["layers"][0]["expert_ids"] = [1]
    with pytest.raises(ContractError, match="expert selection"):
        validate_document(document)


def test_route_coverage_contract_rejects_tampered_storage() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][0]["mixed_bytes"] += 1
    with pytest.raises(ContractError, match="storage"):
        validate_document(document)


def test_route_coverage_contract_rejects_overlapping_split() -> None:
    document = copy.deepcopy(_document())
    document["dataset"]["evaluation_sample_ids"][0] = "cal-1"
    with pytest.raises(ContractError, match="splits"):
        validate_document(document)


def test_route_coverage_contract_rejects_tampered_upper_bound() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][-1]["metrics"]["logit_max_absolute_error"] = 0.1
    with pytest.raises(ContractError, match="gate"):
        validate_document(document)
