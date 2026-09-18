from __future__ import annotations

import copy
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.oracle_smoke import _select_fixture_prompt


def _evidence() -> dict:
    return {
        "schema_version": 1,
        "kind": "oracle_smoke",
        "generated_at": "2026-09-17T14:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9" * 40,
            "derivation_id": "olmoe-bf16-rne-v1",
            "derivation_path": "models/manifests/olmoe_bf16.yaml",
            "derivation_file_sha256": "a" * 64,
            "derivation_semantic_sha256": "b" * 64,
            "artifact_root": "models/derived/olmoe/bf16-rne-v1",
            "weight_dtype": "BF16",
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.14.0",
            "transformers": "5.10.2",
            "cuda": None,
            "hip": "7.15.0",
            "accelerator_name": "AMD Radeon 8060S Graphics",
            "compute_capability": None,
            "bf16_supported": True,
            "model_type": "olmoe",
            "num_experts": 64,
            "num_experts_per_token": 8,
        },
        "input": {
            "fixture_path": "benchmarks/fixtures/olmoe_smoke_v1.jsonl",
            "fixture_file_sha256": "c" * 64,
            "prompt_id": "general-001",
            "prompt_sha256": "d" * 64,
            "token_ids": [1, 2, 3],
        },
        "timing": {"load_seconds": 1.0, "forward_seconds": 0.1},
        "memory": {
            "free_before_load_bytes": 100,
            "total_bytes": 200,
            "peak_allocated_bytes": 50,
            "peak_reserved_bytes": 60,
        },
        "result": {
            "finite": True,
            "logits_shape": [1, 50304],
            "final_token_logits_sha256": "e" * 64,
            "top_token_ids": [10, 11, 12, 13, 14],
            "top_logits": [5.0, 4.0, 3.0, 2.0, 1.0],
        },
    }


def test_oracle_smoke_evidence_schema_accepts_bound_pass() -> None:
    validate_document(_evidence())


def test_oracle_smoke_evidence_rejects_nonfinite_or_unsafe_paths() -> None:
    failed = copy.deepcopy(_evidence())
    failed["result"]["finite"] = False
    with pytest.raises(ContractError):
        validate_document(failed)

    unsafe = copy.deepcopy(_evidence())
    unsafe["input"]["fixture_path"] = "../fixture.jsonl"
    with pytest.raises(ContractError, match="safe project-relative"):
        validate_document(unsafe)


def test_select_fixture_prompt_requires_unique_id(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        '{"id":"one","prompt":"hello"}\n'
        '{"id":"two","prompt":"world"}\n',
        encoding="utf-8",
    )
    assert _select_fixture_prompt(fixture, "two") == "world"

    fixture.write_text(
        '{"id":"one","prompt":"hello"}\n'
        '{"id":"one","prompt":"again"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="duplicate"):
        _select_fixture_prompt(fixture, "one")
