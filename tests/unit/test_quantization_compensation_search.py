from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from uma_qmoe.contracts import ContractError, validate_document


RUNNER_DIRECTORY = Path(__file__).parents[2] / "benchmarks" / "runners"
sys.path.insert(0, str(RUNNER_DIRECTORY))
_expert_parameters = importlib.import_module(
    "run_quantization_compensation_search"
)._expert_parameters


SPECS = [
    ("q4-g128", 4, 128, 0),
    ("q4-g128-r1", 4, 128, 1),
    ("q4-g64", 4, 64, 0),
    ("q4-g128-r2", 4, 128, 2),
    ("q4-g32", 4, 32, 0),
    ("q4-g128-r4", 4, 128, 4),
    ("q5-g128", 5, 128, 0),
    ("q4-g128-r8", 4, 128, 8),
    ("q4-g16", 4, 16, 0),
    ("q6-g128", 6, 128, 0),
    ("q4-g128-r16", 4, 128, 16),
    ("q8-g128", 8, 128, 0),
    ("q9-g128", 9, 128, 0),
    ("q10-g128", 10, 128, 0),
    ("q12-g128", 12, 128, 0),
    ("bf16-upper-bound", 16, None, 0),
]


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


def _document() -> dict:
    total_weights = 2000
    all_q4_bytes = 1063
    pack_bpw = all_q4_bytes * 8 / total_weights
    rows = []
    for index, (candidate_id, bits, group_size, residual_count) in enumerate(SPECS):
        primary_bpw = float(bits)
        scale_bpw = 0.0 if group_size is None else 32.0 / group_size
        residual_bpw = 0.0 if group_size is None else residual_count * 24.0 / group_size
        if candidate_id == "q4-g128":
            effective_bpw = pack_bpw
            payload_bytes = all_q4_bytes
        elif candidate_id.startswith("q4-g128-r"):
            effective_bpw = pack_bpw + residual_bpw
            payload_bytes = math.ceil(effective_bpw * total_weights / 8)
        else:
            effective_bpw = primary_bpw + scale_bpw + residual_bpw
            payload_bytes = math.ceil(effective_bpw * total_weights / 8)
        upper = candidate_id == "bf16-upper-bound"
        rows.append(
            {
                "candidate_id": candidate_id,
                "bits": bits,
                "group_size": group_size,
                "residual_values_per_group": residual_count,
                "metrics": _metrics(
                    1.0 if upper else 1.02 - min(index, 9) * 0.001,
                    1.0 if upper else 0.7 + index * 0.02,
                    zero_error=upper,
                ),
                "primary_bpw": primary_bpw,
                "scale_bpw": scale_bpw,
                "residual_bpw": residual_bpw,
                "effective_bpw": effective_bpw,
                "projected_payload_bytes": payload_bytes,
                "finite": True,
            }
        )
    source_baseline = copy.deepcopy(rows[0]["metrics"])
    return {
        "schema_version": 1,
        "kind": "quantization_compensation_search",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "expert-weight-qformat-compensation-v1",
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
            "candidate_ids": [spec[0] for spec in SPECS],
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
        "source_all_q4_baseline": source_baseline,
        "storage": {
            "total_expert_weight_count": total_weights,
            "all_q4_pack_bytes": all_q4_bytes,
            "all_q4_pack_effective_bpw": pack_bpw,
        },
        "candidate_rows": rows,
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
            "first_passing_candidate_id": "bf16-upper-bound",
        },
        "gates": {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": True,
            "candidate_progression": True,
            "matrix_finite": True,
            "q4_baseline_reproduced": True,
            "bf16_upper_bound": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_compensation_contract_accepts_complete_search() -> None:
    validate_document(_document())


def test_compensation_runner_accepts_stacked_transformers_experts() -> None:
    parameter = SimpleNamespace(ndim=3, shape=(64, 2, 3))
    layers = [
        SimpleNamespace(
            mlp=SimpleNamespace(
                experts=SimpleNamespace(
                    gate_up_proj=parameter,
                    down_proj=parameter,
                )
            )
        )
        for _ in range(16)
    ]
    observed = _expert_parameters(SimpleNamespace(model=SimpleNamespace(layers=layers)))
    assert len(observed) == 32
    assert observed[0][0] == "model.layers.0.mlp.experts.gate_up_proj"
    assert observed[-1][0] == "model.layers.15.mlp.experts.down_proj"


def test_compensation_contract_rejects_tampered_candidate() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][1]["residual_values_per_group"] = 2
    with pytest.raises(ContractError, match="identity"):
        validate_document(document)


def test_compensation_contract_rejects_tampered_storage() -> None:
    document = copy.deepcopy(_document())
    document["candidate_rows"][4]["effective_bpw"] += 0.1
    with pytest.raises(ContractError, match="storage"):
        validate_document(document)


def test_compensation_contract_rejects_unreproduced_baseline() -> None:
    document = copy.deepcopy(_document())
    document["source_all_q4_baseline"]["nll"] += 0.01
    document["source_all_q4_baseline"]["perplexity"] = math.exp(
        document["source_all_q4_baseline"]["nll"]
    )
    document["source_all_q4_baseline"]["relative_nll_change"] = (
        document["source_all_q4_baseline"]["nll"] - 1.0
    )
    document["source_all_q4_baseline"]["relative_perplexity_change"] = (
        document["source_all_q4_baseline"]["perplexity"] / math.e - 1.0
    )
    with pytest.raises(ContractError, match="gate"):
        validate_document(document)


def test_compensation_contract_rejects_tampered_quality_result() -> None:
    document = copy.deepcopy(_document())
    document["quality_gate"]["first_passing_candidate_id"] = None
    with pytest.raises(ContractError, match="quality gate"):
        validate_document(document)
