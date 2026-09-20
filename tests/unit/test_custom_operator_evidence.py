from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _evidence() -> dict:
    gates = {
        "operator_registered": True,
        "architecture_dispatch": True,
        "cpu_reference_forward": True,
        "target_reference_forward": True,
        "cpu_target_agreement": True,
        "performance_mode_fail_closed": True,
        "overall_passed": True,
    }
    return {
        "schema_version": 1,
        "kind": "custom_operator_evidence",
        "captured_at": "2026-09-18T08:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "operator": {
            "qualified_name": "uma_qmoe::moe_forward",
            "registration": "torch.library",
            "registered": True,
            "platform": "hip_gfx1151",
            "device_name": "AMD Radeon Graphics",
        },
        "reference": {
            "layer_index": 0,
            "input_shape": [1, 2048],
            "routes_shape": [1, 8],
            "unique_experts": 1,
            "cpu_finite": True,
            "target_finite": True,
            "cpu_output_sha256": "b" * 64,
            "target_output_sha256": "c" * 64,
            "max_abs_error": 0.25,
            "cosine_similarity": 0.999,
            "acceptance": {
                "max_abs_error": 1.0,
                "minimum_cosine_similarity": 0.99,
            },
        },
        "performance_mode": {
            "requested": True,
            "backend_registered": False,
            "rejected": True,
            "error": "performance mode has no registered hip_gfx1151 backend; silent reference fallback is forbidden",
        },
        "gates": gates,
    }


def test_custom_operator_evidence_recomputes_gates() -> None:
    validate_document(_evidence())

    tampered = copy.deepcopy(_evidence())
    tampered["operator"]["platform"] = "cuda_sm121"
    with pytest.raises(ContractError, match="gate"):
        validate_document(tampered)


def test_custom_operator_evidence_rejects_false_performance_claim() -> None:
    tampered = copy.deepcopy(_evidence())
    tampered["performance_mode"]["rejected"] = False
    with pytest.raises(ContractError, match="gate"):
        validate_document(tampered)
