from __future__ import annotations

import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.reference_host_baseline import (
    build_reference_host_baseline,
    build_reference_host_run_manifest,
)


def _write_run(directory: Path) -> None:
    directory.mkdir()
    result = {
        "duration_seconds": 2.0,
        "requests": [
            {
                "input_tokens": 4,
                "output_tokens": 3,
                "ttft_ms": 100.0 + index,
                "inter_token_latencies_ms": [20.0, 30.0],
                "generated_token_ids": [1, 2, 3],
                "error": None,
            }
            for index in range(3)
        ],
    }
    metadata = {
        "target_id": "halo3",
        "status": "succeeded",
        "returncode": 0,
        "started_at": "2026-09-18T01:00:00Z",
        "completed_at": "2026-09-18T01:10:00Z",
        "runner_argv": ["python3", "benchmarks/runners/pytorch_reference_host.py"],
        "artifact_paths": ["result.json", "runner-metadata.json"],
        "offline_local_model": True,
        "generation_loop": "manual_past_key_values_greedy",
        "runtime": {
            "source_commit": "a" * 40,
            "backend": "hip",
            "container_image": "example/host@sha256:" + "b" * 64,
            "torch_version": "2.14.0",
            "transformers_version": "5.10.2",
            "backend_version": "7.15.0",
        },
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "c" * 40,
        },
        "workload": {
            "input_tokens": 4,
            "output_tokens": 3,
            "warmup_requests": 1,
            "measured_requests": 3,
            "max_concurrency": 1,
        },
        "cgroup_memory_peak_bytes": 123456,
        "torch_peak_allocated_bytes": 100000,
        "torch_peak_reserved_bytes": 110000,
        "samples": [
            {
                "memory_available_bytes": 1000,
                "telemetry": {"temperature_c": 50.0, "socket_power_w": 75.0},
            },
            {
                "memory_available_bytes": 900,
                "telemetry": {"temperature_c": 55.0, "socket_power_w": 80.0},
            },
        ],
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "runner-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _build(directory: Path) -> dict:
    return build_reference_host_baseline(
        directory,
        target_id="halo3",
        source_commit="a" * 40,
        backend="hip",
        container_image="example/host@sha256:" + "b" * 64,
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="c" * 40,
        derivation_semantic_sha256="d" * 64,
        input_tokens=4,
        output_tokens=3,
        warmup_requests=1,
        measured_requests=3,
    )


def test_reference_host_baseline_binds_manual_cache_runtime(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)

    document = _build(run)

    assert document["status"] == "passed"
    assert document["implementation"]["name"] == "pytorch_hf_reference_host"
    assert document["gates"]["manual_cache_decode"] is True
    assert document["results"]["total_output_tokens"] == 9
    validate_document(document)


def test_reference_host_baseline_rejects_relabelled_workload(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    metadata_path = run / "runner-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["workload"]["measured_requests"] = 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContractError, match="workload does not match"):
        _build(run)


def test_reference_host_run_manifest_binds_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    baseline = run / "reference-host-baseline.json"
    baseline.write_text(json.dumps(_build(run)), encoding="utf-8")

    manifest = build_reference_host_run_manifest(
        baseline,
        run,
        run_id="halo3-reference-host-001",
        git_commit="e" * 40,
        git_dirty=False,
        dirty_patch_sha256=None,
        machine_baseline_sha256="f" * 64,
        benchmark_contract_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
    )

    assert manifest["status"] == "succeeded"
    assert len(manifest["raw_artifacts"]) == 3
    validate_document(manifest)
