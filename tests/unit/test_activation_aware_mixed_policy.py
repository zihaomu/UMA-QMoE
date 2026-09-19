from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(nll: float = 1.0, exact: float = 0.995) -> dict:
    perplexity = math.exp(nll)
    return {
        "finite": True,
        "nll": nll,
        "perplexity": perplexity,
        "relative_nll_change": nll - 1.0,
        "relative_perplexity_change": perplexity / math.e - 1.0,
        "top1_agreement": 1.0,
        "logit_max_absolute_error": 0.1,
        "logit_p99_absolute_error": 0.05,
        "logit_cosine_similarity": 0.999,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 0.999,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _document() -> dict:
    total_weights = 6_442_450_944
    effective_bpw = 12.84375
    payload_bytes = math.ceil(effective_bpw * total_weights / 8)
    q8_metrics = _metrics(1.0, 0.995)
    policy_metrics = _metrics(0.999, 0.995)
    return {
        "schema_version": 1,
        "kind": "activation_aware_mixed_policy",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
        },
        "method": {
            "id": "layer15-awq-q4-q8-bf16-mixed-v1",
            "source_reverse_layer_evidence": {
                "file_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "prompt_fixture": {
                "file_sha256": "c" * 64,
                "semantic_sha256": "d" * 64,
            },
            "calibration_only": False,
            "performance_evidence": False,
            "q4_group_size": 128,
            "q4_clip_ratios": [
                1.0,
                0.98,
                0.95,
                0.92,
                0.9,
                0.87,
                0.85,
                0.8,
                0.75,
                0.7,
            ],
            "q8_group_size": 128,
        },
        "dataset": {
            "calibration_sample_ids": ["cal-1", "cal-2"],
            "evaluation_sample_ids": ["eval-1", "eval-2"],
            "evaluation_prompt_token_count": 10,
            "evaluation_target_token_count": 4,
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.e,
            "logits_sha256": "e" * 64,
            "routes_sha256": "f" * 64,
        },
        "source_q8_base_metrics": copy.deepcopy(q8_metrics),
        "q8_base_metrics": copy.deepcopy(q8_metrics),
        "q8_base_reproduced": True,
        "policy": {
            "policy_id": "olmoe-layer15-awq-q4-q8-bf16-v1",
            "q4_layers": [15],
            "q8_layers": [8, 11, 12, 13, 14],
            "bf16_layers": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            "effective_bpw": effective_bpw,
            "projected_payload_bytes": payload_bytes,
            "metrics": policy_metrics,
        },
        "storage": {
            "total_expert_weight_count": total_weights,
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
        "gates": {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "dataset_split_disjoint": True,
            "reference_finite": True,
            "q8_base_reproduced": True,
            "q8_base_quality_passed": True,
            "policy_finite": True,
            "policy_quality_passed": True,
            "storage_accounting": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_activation_aware_policy_accepts_complete_evidence() -> None:
    validate_document(_document())


def test_activation_aware_policy_rejects_layer_tampering() -> None:
    document = _document()
    document["policy"]["q4_layers"] = [14]
    with pytest.raises(ContractError):
        validate_document(document)


def test_activation_aware_policy_rejects_split_leakage() -> None:
    document = _document()
    document["dataset"]["evaluation_sample_ids"][0] = "cal-1"
    with pytest.raises(ContractError, match="disjoint"):
        validate_document(document)


def test_activation_aware_policy_rejects_storage_tampering() -> None:
    document = _document()
    document["policy"]["effective_bpw"] += 0.1
    with pytest.raises(ContractError, match="storage"):
        validate_document(document)


def test_activation_aware_policy_rejects_false_q8_reproduction() -> None:
    document = _document()
    document["q8_base_metrics"]["nll"] += 0.01
    document["q8_base_metrics"]["perplexity"] = math.exp(
        document["q8_base_metrics"]["nll"]
    )
    document["q8_base_metrics"]["relative_nll_change"] = 0.01
    document["q8_base_metrics"]["relative_perplexity_change"] = (
        document["q8_base_metrics"]["perplexity"] / math.e - 1.0
    )
    with pytest.raises(ContractError, match="reproduction"):
        validate_document(document)
