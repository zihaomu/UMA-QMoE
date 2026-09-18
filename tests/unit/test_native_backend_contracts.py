from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.native_backend import PackedQ4NativeBackend, native_kernel_source_sha256


def _packed_evidence() -> dict:
    projections = [
        {
            "name": name,
            "shape": shape,
            "finite": True,
            "max_absolute_error": 0.01,
            "cosine_similarity": 0.999,
        }
        for name, shape in (
            ("gate_proj", [1024, 2048]),
            ("up_proj", [1024, 2048]),
            ("down_proj", [2048, 1024]),
        )
    ]
    samples = [1.0, 1.1, 1.2]
    gates = {
        "target_architecture": True,
        "target_compilation": True,
        "direct_packed_input": True,
        "no_dequantized_weight_cache": True,
        "projection_correctness": True,
        "moe_correctness": True,
        "performance_mode_executed": True,
        "overall_passed": True,
    }
    return {
        "schema_version": 1,
        "kind": "packed_q4_kernel_evidence",
        "captured_at": "2026-09-18T09:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "kernel": {
            "platform": "hip_gfx1151",
            "device_name": "AMD Radeon Graphics",
            "source_sha256": "b" * 64,
            "abi": "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-v1",
            "compiled_for_target": True,
            "reads_packed_weights_directly": True,
            "full_dequantized_weight_cache": False,
        },
        "workload": {
            "layer_index": 0,
            "tokens": 1,
            "top_k": 8,
            "unique_experts": 8,
            "warmup_iterations": 1,
            "measured_iterations": 3,
        },
        "correctness": {
            "projection_results": projections,
            "moe_forward": {
                "finite": True,
                "max_absolute_error": 0.1,
                "cosine_similarity": 0.999,
                "output_sha256": "c" * 64,
            },
            "acceptance": {
                "max_absolute_error": 1.0,
                "minimum_cosine_similarity": 0.99,
            },
        },
        "performance": {
            "scope": "single_moe_layer_microbenchmark",
            "samples_milliseconds": samples,
            "median_milliseconds": 1.1,
            "p95_milliseconds": 1.2,
            "compressed_cache_tensor_count": 24,
            "compressed_cache_bytes": 1024,
            "dequantized_weight_cache_bytes": 0,
            "formal_tps_claim": False,
        },
        "gates": gates,
    }


def _metrics(exact: float, cosine: float) -> dict:
    return {
        "finite": True,
        "top1_token_id": 7785,
        "top1_matches_reference": True,
        "logit_max_absolute_error": 1.0,
        "logit_p99_absolute_error": 0.5,
        "logit_cosine_similarity": cosine,
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": 0.95,
        "per_layer_router_exact_set_agreement": [exact] * 16,
    }


def _sensitivity() -> dict:
    baseline = _metrics(0.65, 0.998)
    rows = []
    for layer in range(16):
        metrics = _metrics(0.65 + layer / 1000, 0.998 + layer / 100000)
        rows.append(
            {
                "restored_layer": layer,
                "metrics": metrics,
                "delta_vs_all_q4": {
                    "router_exact_set_agreement": metrics["router_exact_set_agreement"]
                    - baseline["router_exact_set_agreement"],
                    "router_mean_set_overlap": 0.0,
                    "logit_cosine_similarity": metrics["logit_cosine_similarity"]
                    - baseline["logit_cosine_similarity"],
                },
                "finite": True,
            }
        )
    return {
        "schema_version": 1,
        "kind": "mixed_precision_sensitivity",
        "captured_at": "2026-09-18T09:00:00Z",
        "target_id": "spark1",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "method": {
            "id": "single-layer-bf16-restore-v1",
            "diagnostic_only": True,
            "materializes_expert_parameters": True,
            "performance_evidence": False,
            "candidate_definition": "one BF16 expert layer plus fifteen canonical Q4-dequantized expert layers",
            "restored_layers": list(range(16)),
        },
        "reference": {
            "finite": True,
            "logits_sha256": "b" * 64,
            "routes_sha256": "c" * 64,
            "top1_token_id": 7785,
        },
        "all_q4_baseline": baseline,
        "storage": {
            "all_q4_bytes": 1000,
            "single_bf16_layer_bytes": 100,
            "replaced_q4_layer_bytes": 25,
            "single_layer_mixed_bytes": 1075,
            "all_q4_effective_bpw": 4.25,
            "single_layer_mixed_effective_bpw": 4.98,
        },
        "rows": rows,
        "ranking": list(reversed(range(16))),
        "gates": {
            "reference_finite": True,
            "all_q4_finite": True,
            "all_layers_covered": True,
            "matrix_finite": True,
            "quality_gate_unchanged": True,
            "overall_passed": True,
        },
    }


def test_native_source_identity_is_stable_sha256() -> None:
    digest = native_kernel_source_sha256()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_native_backend_releases_closed_pack_cache() -> None:
    backend = PackedQ4NativeBackend(
        "cuda_sm121", SimpleNamespace(q4_linear=lambda *_args: None)
    )
    backend._cache[(1, "first", "cuda:0")] = object()
    backend._cache[(2, "second", "cuda:0")] = object()
    backend.release_pack(1)
    assert set(backend._cache) == {(2, "second", "cuda:0")}


def test_packed_q4_evidence_recomputes_gates_and_timing() -> None:
    validate_document(_packed_evidence())
    tampered = copy.deepcopy(_packed_evidence())
    tampered["performance"]["median_milliseconds"] = 99.0
    with pytest.raises(ContractError, match="timing summary"):
        validate_document(tampered)


def test_mixed_precision_matrix_recomputes_ranking_and_deltas() -> None:
    validate_document(_sensitivity())
    tampered = copy.deepcopy(_sensitivity())
    tampered["rows"][0]["delta_vs_all_q4"]["router_exact_set_agreement"] = 0.5
    with pytest.raises(ContractError, match="delta"):
        validate_document(tampered)
