from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(exact: float, cosine: float = 0.9) -> dict:
    return {
        "finite": True,
        "top1_token_id": 1,
        "top1_matches_reference": True,
        "logit_max_absolute_error": 1.0,
        "logit_p99_absolute_error": 0.5,
        "logit_cosine_similarity": cosine,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 0.95,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _delta(metrics: dict, baseline: dict) -> dict:
    return {
        "router_exact_set_agreement": metrics["router_exact_set_agreement"]
        - baseline["router_exact_set_agreement"],
        "router_mean_set_overlap": metrics["router_mean_set_overlap"]
        - baseline["router_mean_set_overlap"],
        "logit_cosine_similarity": metrics["logit_cosine_similarity"]
        - baseline["logit_cosine_similarity"],
    }


def _document() -> dict:
    baseline = _metrics(0.5)
    order = [2, 0, 1, *range(3, 16)]
    cumulative = []
    restored: list[int] = []
    for index, layer in enumerate(order):
        restored.append(layer)
        exact = 0.5 + (0.5 * index / 15)
        metrics = _metrics(exact, 0.9 + 0.1 * index / 15)
        cumulative.append(
            {
                "added_layer": layer,
                "restored_layers": list(restored),
                "metrics": metrics,
                "delta_vs_all_q4": _delta(metrics, baseline),
                "mixed_bytes": 1000 + 100 * (index + 1),
                "effective_bpw": 4.0 + index / 10,
                "finite": True,
            }
        )
    experts = []
    for expert in range(64):
        metrics = _metrics(0.5)
        experts.append(
            {
                "restored_expert": expert,
                "reference_route_count": 64 - expert,
                "q4_route_count": 64 - expert,
                "metrics": metrics,
                "delta_vs_all_q4": _delta(metrics, baseline),
                "extra_bytes": 100,
                "finite": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "mixed_precision_refinement",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "cumulative-layer-and-layer2-expert-bf16-restore-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "source_single_layer_evidence": {
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
            "layer_order": order,
            "expert_layer": 2,
        },
        "reference": {
            "finite": True,
            "logits_sha256": "d" * 64,
            "routes_sha256": "e" * 64,
            "top1_token_id": 1,
        },
        "all_q4_baseline": baseline,
        "storage": {
            "all_q4_bytes": 1000,
            "total_expert_weight_count": 2000,
            "layer2_q4_bytes": 100,
            "layer2_bf16_bytes": 400,
        },
        "cumulative_layer_rows": cumulative,
        "layer2_expert_rows": experts,
        "layer2_expert_ranking": list(range(64)),
        "quality_gate": {
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_restored_layers": order,
        },
        "gates": {
            "reference_finite": True,
            "all_q4_finite": True,
            "layer2_first": True,
            "all_layers_covered": True,
            "all_layer2_experts_covered": True,
            "matrix_finite": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_mixed_precision_refinement_contract_accepts_complete_matrix() -> None:
    validate_document(_document())


def test_mixed_precision_refinement_rejects_non_layer2_start() -> None:
    document = _document()
    document["method"]["layer_order"][:2] = [0, 2]
    with pytest.raises(ContractError, match="from layer 2"):
        validate_document(document)


def test_mixed_precision_refinement_rejects_tampered_expert_ranking() -> None:
    document = copy.deepcopy(_document())
    document["layer2_expert_ranking"][:2] = [1, 0]
    with pytest.raises(ContractError, match="expert ranking"):
        validate_document(document)
