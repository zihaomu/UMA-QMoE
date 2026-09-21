from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, canonical_sha256, validate_document
from uma_qmoe.quantization_calibration import (
    build_expert_balanced_sample_manifest,
    build_expert_calibration_coverage,
    build_quantization_dataset_manifest,
    route_capture_sha256,
    split_disjointness_report,
)


def _encode(text: str) -> list[int]:
    return [ord(character) for character in text]


def _fixture(path: Path, *, suffix: str = "") -> None:
    rows = [
        {
            "id": f"sample-a{suffix}",
            "category": "fact",
            "prompt": f"alpha{suffix}",
            "completion": " answer",
        },
        {
            "id": f"sample-b{suffix}",
            "category": "reasoning",
            "prompt": f"beta{suffix}",
            "completion": " result",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _manifest(path: Path, *, partition: str = "calibration") -> dict:
    return build_quantization_dataset_manifest(
        path,
        partition=partition,
        source_name="unit fixture",
        source_uri="repo://unit-fixture",
        source_revision="v1",
        license_id="Apache-2.0",
        tokenizer_id="Qwen/Qwen1.5-MoE-A2.7B",
        tokenizer_revision="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
        tokenizer_artifacts_sha256="a" * 64,
        encode=_encode,
    )


def _observations(manifest: dict) -> list[dict]:
    observations = []
    ordinal = 0
    for sample in manifest["samples"]:
        for token_index, token_id in enumerate(sample["prompt_token_ids"]):
            layers = []
            for layer_index in range(24):
                first = (ordinal + layer_index) % 60
                layers.append(
                    {
                        "layer_index": layer_index,
                        "expert_indices": [
                            first,
                            (first + 1) % 60,
                            (first + 2) % 60,
                            (first + 3) % 60,
                        ],
                        "routing_weights": [0.4, 0.3, 0.2, 0.1],
                        "router_entropy": 2.0,
                        "route_margin": 0.1,
                    }
                )
            observations.append(
                {
                    "sample_id": sample["sample_id"],
                    "token_index": token_index,
                    "token_id": token_id,
                    "layers": layers,
                }
            )
            ordinal += 1
    return observations


def test_builds_draft_dataset_coverage_and_deterministic_ebss(tmp_path: Path) -> None:
    fixture = tmp_path / "calibration.jsonl"
    _fixture(fixture)
    manifest = _manifest(fixture)
    observations = _observations(manifest)

    assert manifest["status"] == "draft"
    assert manifest["token_floor"] == {
        "minimum_prompt_tokens": 32768,
        "minimum_target_tokens": 0,
        "passed": False,
    }
    coverage = build_expert_calibration_coverage(
        manifest, observations, target_id="local-halo"
    )
    assert coverage["status"] == "insufficient_coverage"
    assert coverage["summary"]["prompt_tokens"] == manifest["totals"][
        "prompt_tokens"
    ]
    assert coverage["summary"]["layer_expert_units"] == 1440
    assert coverage["gates"]["activation_statistics_complete"] is False
    assert coverage["inputs"]["route_capture_sha256"] == route_capture_sha256(
        observations
    )

    first = build_expert_balanced_sample_manifest(manifest, observations, coverage)
    second = build_expert_balanced_sample_manifest(manifest, observations, coverage)
    assert first["status"] == "insufficient_coverage"
    assert first["selection"]["selected_tokens"] == len(observations)
    assert first["selection"]["selected_order_sha256"] == second["selection"][
        "selected_order_sha256"
    ]
    assert canonical_sha256(first) == canonical_sha256(second)


def test_dataset_manifest_rejects_freeze_below_floor_and_tampering(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "calibration.jsonl"
    _fixture(fixture)
    with pytest.raises(ContractError, match="cannot freeze"):
        build_quantization_dataset_manifest(
            fixture,
            partition="calibration",
            source_name="unit fixture",
            source_uri="repo://unit-fixture",
            source_revision="v1",
            license_id="Apache-2.0",
            tokenizer_id="Qwen/Qwen1.5-MoE-A2.7B",
            tokenizer_revision="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
            tokenizer_artifacts_sha256="a" * 64,
            encode=_encode,
            frozen=True,
        )

    manifest = _manifest(fixture)
    tampered = copy.deepcopy(manifest)
    tampered["samples"][0]["prompt_token_count"] += 1
    with pytest.raises(ContractError, match="prompt token count"):
        validate_document(tampered)
    with pytest.raises(ContractError, match="not frozen"):
        validate_document(manifest, require_frozen=True)


def test_coverage_rejects_missing_tokens_and_arithmetic_tampering(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "calibration.jsonl"
    _fixture(fixture)
    manifest = _manifest(fixture)
    observations = _observations(manifest)
    with pytest.raises(ContractError, match="missing"):
        build_expert_calibration_coverage(
            manifest, observations[:-1], target_id="local-halo"
        )

    coverage = build_expert_calibration_coverage(
        manifest, observations, target_id="local-halo"
    )
    tampered = copy.deepcopy(coverage)
    tampered["layers"][0]["experts"][0]["effective_samples"] += 1
    with pytest.raises(ContractError, match="effective sample"):
        validate_document(tampered)


def test_split_disjointness_audits_text_hash_overlap(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _fixture(left)
    _fixture(right, suffix="-different")
    calibration = _manifest(left)
    search = _manifest(right, partition="search")
    report = split_disjointness_report([calibration, search])
    assert report["passed"] is True
    assert report["pairwise"][0]["overlap_count"] == 0

    overlapping = _manifest(left, partition="held_out")
    report = split_disjointness_report([calibration, overlapping])
    assert report["passed"] is False
    assert report["pairwise"][0]["overlap_count"] == 2
