from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(nll: float, exact: float) -> dict:
    reference_nll = 1.0
    perplexity = math.exp(nll)
    return {
        "finite": True,
        "nll": nll,
        "perplexity": perplexity,
        "relative_nll_change": (nll / reference_nll) - 1.0,
        "relative_perplexity_change": (perplexity / math.exp(reference_nll)) - 1.0,
        "top1_agreement": 1.0,
        "logit_max_absolute_error": 0.1,
        "logit_p99_absolute_error": 0.05,
        "logit_cosine_similarity": 0.999,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 0.999,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _document() -> dict:
    order = [30, 26, 19]
    rows = []
    restored: list[int] = []
    for index, expert in enumerate(order):
        restored.append(expert)
        extra = 100 * len(restored)
        mixed = 1000 + extra
        rows.append(
            {
                "policy_id": "layer2-" + "-".join(f"e{item}" for item in restored),
                "added_expert": expert,
                "restored_experts": list(restored),
                "metrics": _metrics(1.005, [0.8, 0.9, 0.99][index]),
                "extra_bytes": extra,
                "mixed_bytes": mixed,
                "effective_bpw": mixed * 8 / 2000,
                "finite": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "mixed_precision_policy_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "cumulative-layer2-expert-prefix-quality-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "expert_layer": 2,
            "expert_order": order,
            "source_refinement_evidence": {
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
            "prompt_fixture": {
                "file_sha256": "d" * 64,
                "semantic_sha256": "e" * 64,
            },
        },
        "dataset": {
            "sample_ids": ["one", "two"],
            "sample_count": 2,
            "prompt_token_count": 10,
            "target_token_count": 4,
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.exp(1.0),
            "logits_sha256": "f" * 64,
            "routes_sha256": "0" * 64,
        },
        "all_q4_baseline": _metrics(1.005, 0.7),
        "storage": {
            "all_q4_bytes": 1000,
            "total_expert_weight_count": 2000,
            "single_expert_extra_bytes": 100,
        },
        "candidate_rows": rows,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_policy_id": "layer2-e30-e26-e19",
        },
        "gates": {
            "reference_finite": True,
            "all_q4_finite": True,
            "dataset_complete": True,
            "prefix_progression": True,
            "matrix_finite": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_policy_search_contract_accepts_complete_prefix() -> None:
    validate_document(_document())


def test_policy_search_rejects_tampered_prefix() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][1]["restored_experts"] = [30, 19]
    with pytest.raises(ContractError, match="prefix or storage"):
        validate_document(document)


def test_policy_search_rejects_tampered_perplexity() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][0]["metrics"]["perplexity"] += 1.0
    with pytest.raises(ContractError, match="perplexity"):
        validate_document(document)


def test_policy_search_rejects_tampered_gate_result() -> None:
    document = copy.deepcopy(_document())
    document["quality_gate"]["first_passing_policy_id"] = None
    with pytest.raises(ContractError, match="gate result"):
        validate_document(document)
