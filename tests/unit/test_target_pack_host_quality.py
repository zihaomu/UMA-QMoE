from __future__ import annotations

import copy
import math

import pytest

from uma_qmoe.contracts import ContractError, validate_document


def _metrics() -> dict:
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
        "router_exact_set_agreement": 0.995,
        "router_mean_set_overlap": 0.999,
        "per_layer_router_exact_set_agreement": [0.995] * 16,
    }


def _document() -> dict:
    gates = {
        "manifest_identity": True,
        "policy_evidence_identity": True,
        "dataset_identity": True,
        "reference_finite": True,
        "candidate_finite": True,
        "quality_passed": True,
        "dense_only_checkpoint_load": True,
        "no_expert_parameters": True,
        "single_pack_mapping": True,
        "full_model_executed": True,
        "overall_passed": True,
    }
    return {
        "schema_version": 1,
        "kind": "target_pack_host_quality",
        "captured_at": "2026-09-19T11:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "model_manifest_sha256": "a" * 64,
        },
        "target_pack": {
            "artifact_sha256": "b" * 64,
            "artifact_size_bytes": 10_344_579_072,
            "manifest_file_sha256": "c" * 64,
            "manifest_semantic_sha256": "d" * 64,
            "policy_id": "olmoe-layer15-awq-q4-q8-bf16-v1",
            "policy_evidence_semantic_sha256": "e" * 64,
            "layer_encodings": {
                "q4_layers": [15],
                "q8_layers": [8, 11, 12, 13, 14],
                "bf16_layers": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            },
        },
        "dataset": {
            "fixture_file_sha256": "f" * 64,
            "fixture_semantic_sha256": "1" * 64,
            "evaluation_sample_ids": ["eval-1", "eval-2"],
            "evaluation_prompt_token_count": 10,
            "evaluation_target_token_count": 4,
        },
        "reference": {
            "finite": True,
            "nll": 1.0,
            "perplexity": math.e,
            "logits_sha256": "2" * 64,
            "routes_sha256": "3" * 64,
        },
        "candidate": _metrics(),
        "loader": {
            "performance_mode": False,
            "dense_tensor_count": 147,
            "dense_tensor_bytes": 953_421_824,
            "skipped_expert_tensor_count": 3072,
            "loaded_expert_tensor_count": 0,
            "expert_parameter_count": 0,
            "quantized_moe_block_count": 16,
            "expert_pack_mapping_count": 1,
            "expert_pack_vma_count": 1,
        },
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
        },
        "gates": gates,
    }


def test_target_pack_host_quality_accepts_complete_evidence() -> None:
    validate_document(_document())


def test_target_pack_host_quality_accepts_v2_policy_layers() -> None:
    document = _document()
    document["target_pack"].update(
        {
            "policy_id": "olmoe-layer15-awq-q4-q8-bf16-v2",
            "layer_encodings": {
                "q4_layers": [15],
                "q8_layers": [11, 12, 13, 14],
                "bf16_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
        }
    )
    validate_document(document)


def test_target_pack_host_quality_rejects_metric_tampering() -> None:
    document = _document()
    document["candidate"]["relative_perplexity_change"] = 0.5
    with pytest.raises(ContractError, match="relative quality"):
        validate_document(document)


def test_target_pack_host_quality_rejects_loader_gate_tampering() -> None:
    document = _document()
    document["gates"]["single_pack_mapping"] = False
    with pytest.raises(ContractError, match="gate"):
        validate_document(document)


def test_target_pack_host_quality_rejects_status_tampering() -> None:
    document = copy.deepcopy(_document())
    document["status"] = "failed"
    with pytest.raises(ContractError, match="status"):
        validate_document(document)
