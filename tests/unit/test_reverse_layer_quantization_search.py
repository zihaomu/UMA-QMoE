from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics(nll: float, exact: float) -> dict:
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


def _search(bits: int, quantized_bpw: float, total_weights: int) -> dict:
    singles = [
        {
            "quantized_layer": layer,
            "metrics": _metrics(1.001, 1.0 - layer / 1000),
            "finite": True,
        }
        for layer in range(16)
    ]
    ranking = list(range(16))
    cumulative = []
    quantized = []
    for index, layer in enumerate(ranking):
        quantized.append(layer)
        final = index == 15
        exact = 0.968 if final else 1.0 - (index + 1) * 0.002
        nll = 1.02 if final else 1.005
        effective_bpw = 16.0 - len(quantized) * (16.0 - quantized_bpw) / 16
        cumulative.append(
            {
                "added_quantized_layer": layer,
                "quantized_layers": list(quantized),
                "metrics": _metrics(nll, exact),
                "effective_bpw": effective_bpw,
                "projected_payload_bytes": math.ceil(effective_bpw * total_weights / 8),
                "finite": True,
            }
        )
    return {
        "quantized_bits": bits,
        "quantized_effective_bpw": quantized_bpw,
        "single_layer_rows": singles,
        "ranking": ranking,
        "cumulative_rows": cumulative,
        "lowest_bpw_passing_quantized_layers": ranking[:5],
    }


def _document() -> dict:
    total_weights = 16000
    q4_bpw = 4.25
    bpw = {4: q4_bpw, 8: 8.25, 12: 12.25}
    searches = [_search(bits, bpw[bits], total_weights) for bits in (4, 8, 12)]
    q4_row = searches[0]["cumulative_rows"][4]
    return {
        "schema_version": 1,
        "kind": "reverse_layer_quantization_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "bf16-base-cumulative-quantized-layer-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "quantized_bits": [4, 8, 12],
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
                "quantized_bits": bits,
                "candidate_id": f"q{bits}-g128",
                "metrics": _metrics(1.02, 0.968),
            }
            for bits in (4, 8, 12)
        ],
        "storage": {
            "total_expert_weight_count": total_weights,
            "expert_weight_count_per_layer": 1000,
            "q4_pack_effective_bpw": q4_bpw,
        },
        "searches": searches,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "lowest_bpw_passing_policy": {
                "quantized_bits": 4,
                "quantized_layers": q4_row["quantized_layers"],
                "effective_bpw": q4_row["effective_bpw"],
            },
        },
        "gates": {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": True,
            "uniform_endpoints_reproduced": True,
            "all_layers_covered": True,
            "matrix_finite": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_reverse_layer_contract_accepts_complete_search() -> None:
    validate_document(_document())


def test_reverse_layer_contract_rejects_tampered_ranking() -> None:
    document = copy.deepcopy(_document())
    document["searches"][0]["ranking"][:2] = [1, 0]
    with pytest.raises(ContractError, match="ranking"):
        validate_document(document)


def test_reverse_layer_contract_rejects_tampered_storage() -> None:
    document = copy.deepcopy(_document())
    document["searches"][0]["cumulative_rows"][0]["effective_bpw"] += 0.1
    with pytest.raises(ContractError, match="progression"):
        validate_document(document)


def test_reverse_layer_contract_rejects_tampered_subset() -> None:
    document = copy.deepcopy(_document())
    document["searches"][0]["lowest_bpw_passing_quantized_layers"] = [0]
    with pytest.raises(ContractError, match="passing subset"):
        validate_document(document)


def test_reverse_layer_contract_rejects_unreproduced_endpoint() -> None:
    document = copy.deepcopy(_document())
    document["source_uniform_metrics"][0]["metrics"] = _metrics(1.03, 0.968)
    with pytest.raises(ContractError, match="gate"):
        validate_document(document)
