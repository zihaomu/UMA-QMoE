from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(nll: float, exact: float, *, zero_error: bool = False) -> dict:
    perplexity = math.exp(nll)
    return {
        "finite": True,
        "nll": nll,
        "perplexity": perplexity,
        "relative_nll_change": nll - 1.0,
        "relative_perplexity_change": perplexity / math.e - 1.0,
        "top1_agreement": 1.0,
        "logit_max_absolute_error": 0.0 if zero_error else 0.1,
        "logit_p99_absolute_error": 0.0 if zero_error else 0.05,
        "logit_cosine_similarity": 1.0 if zero_error else 0.999,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 1.0 if zero_error else 0.999,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _search(bits: int, total_weights: int) -> dict:
    base_metrics = _metrics(1.02, 0.8)
    singles = [
        {
            "restored_layer": layer,
            "metrics": _metrics(1.02, 0.8 + layer / 1000),
            "finite": True,
        }
        for layer in range(16)
    ]
    ranking = list(reversed(range(16)))
    cumulative = []
    restored = []
    base_bpw = bits + 0.25
    for index, layer in enumerate(ranking):
        restored.append(layer)
        upper = index == 15
        exact = 1.0 if upper else 0.9 + (index + 1) * 0.00625
        effective_bpw = base_bpw + len(restored) * (16.0 - base_bpw) / 16
        cumulative.append(
            {
                "added_layer": layer,
                "restored_layers": list(restored),
                "metrics": _metrics(
                    1.0 if upper else 1.005,
                    exact,
                    zero_error=upper,
                ),
                "effective_bpw": effective_bpw,
                "projected_payload_bytes": math.ceil(effective_bpw * total_weights / 8),
                "finite": True,
            }
        )
    return {
        "base_bits": bits,
        "base_effective_bpw": base_bpw,
        "base_metrics": base_metrics,
        "single_layer_rows": singles,
        "ranking": ranking,
        "cumulative_rows": cumulative,
        "first_passing_restored_layers": ranking[:15],
    }


def _document() -> dict:
    total_weights = 16000
    searches = [_search(bits, total_weights) for bits in (8, 9, 12)]
    q8_row = searches[0]["cumulative_rows"][14]
    return {
        "schema_version": 1,
        "kind": "layer_precision_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "uniform-base-cumulative-bf16-layer-restore-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "base_bits": [8, 9, 12],
            "source_compensation_evidence": {
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
            "prompt_fixture": {
                "file_sha256": "d" * 64,
                "semantic_sha256": "e" * 64,
            },
        },
        "dataset": {
            "sample_ids": ["eval-1", "eval-2"],
            "prompt_token_count": 10,
            "target_token_count": 4,
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.e,
            "logits_sha256": "f" * 64,
            "routes_sha256": "0" * 64,
        },
        "source_uniform_metrics": [
            {
                "base_bits": bits,
                "candidate_id": f"q{bits}-g128",
                "metrics": _metrics(1.02, 0.8),
            }
            for bits in (8, 9, 12)
        ],
        "storage": {
            "total_expert_weight_count": total_weights,
            "expert_weight_count_per_layer": 1000,
        },
        "searches": searches,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "lowest_bpw_passing_policy": {
                "base_bits": 8,
                "restored_layers": q8_row["restored_layers"],
                "effective_bpw": q8_row["effective_bpw"],
            },
        },
        "gates": {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": True,
            "base_metrics_reproduced": True,
            "all_layers_covered": True,
            "matrix_finite": True,
            "bf16_upper_bounds": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_layer_precision_contract_accepts_complete_search() -> None:
    validate_document(_document())


def test_layer_precision_contract_rejects_tampered_ranking() -> None:
    document = copy.deepcopy(_document())
    document["searches"][0]["ranking"][:2] = [14, 15]
    with pytest.raises(ContractError, match="ranking"):
        validate_document(document)


def test_layer_precision_contract_rejects_tampered_storage() -> None:
    document = copy.deepcopy(_document())
    document["searches"][0]["cumulative_rows"][0]["effective_bpw"] += 0.1
    with pytest.raises(ContractError, match="progression"):
        validate_document(document)


def test_layer_precision_contract_rejects_tampered_lowest_policy() -> None:
    document = copy.deepcopy(_document())
    document["quality_gate"]["lowest_bpw_passing_policy"]["base_bits"] = 9
    with pytest.raises(ContractError, match="lowest-bpw"):
        validate_document(document)


def test_layer_precision_contract_rejects_unreproduced_base() -> None:
    document = copy.deepcopy(_document())
    document["source_uniform_metrics"][0]["metrics"] = _metrics(1.03, 0.8)
    with pytest.raises(ContractError, match="gate"):
        validate_document(document)
