from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.external_baseline import (
    build_external_baseline,
    build_external_baseline_run_manifest,
)


def _write_run(directory: Path) -> None:
    directory.mkdir()
    result = {
        "duration_seconds": 2.0,
        "requests": [
            {
                "input_tokens": 4,
                "output_tokens": 3,
                "ttft_ms": 100.0 + index * 100.0,
                "inter_token_latencies_ms": [20.0 + index * 10.0, 30.0 + index * 10.0],
                "error": None,
            }
            for index in range(3)
        ],
    }
    metadata = {
        "target_id": "spark1",
        "status": "succeeded",
        "returncode": 0,
        "started_at": "2026-09-18T01:00:00Z",
        "completed_at": "2026-09-18T01:10:00Z",
        "runner_argv": ["python3", "benchmarks/external/example/runner.py"],
        "artifact_paths": ["result.json", "runner-metadata.json", "runner.log"],
        "isolation": {
            "execution_boundary": "separate_container",
            "imports_uma_qmoe_runtime": False,
            "loads_uma_qmoe_expert_pack": False,
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
        "runtime": {
            "implementation_name": "example_runtime",
            "implementation_version": "1.0.0",
            "source_commit": "a" * 40,
            "backend": "cuda",
            "container_image": "example/runtime@sha256:" + "b" * 64,
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
    (directory / "runner.log").write_text("done\n", encoding="utf-8")


def _build(directory: Path) -> dict:
    return build_external_baseline(
        directory,
        target_id="spark1",
        implementation_name="example_runtime",
        implementation_version="1.0.0",
        source_commit="a" * 40,
        backend="cuda",
        container_image="example/runtime@sha256:" + "b" * 64,
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="c" * 40,
        derivation_semantic_sha256="d" * 64,
        input_tokens=4,
        output_tokens=3,
        warmup_requests=1,
        measured_requests=3,
        max_concurrency=1,
    )


def test_external_baseline_is_runtime_neutral_and_hash_bound(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)

    document = _build(run)

    assert document["implementation"]["name"] == "example_runtime"
    assert document["status"] == "passed"
    assert document["results"]["tpot_ms"]["mean"] == pytest.approx(35.0)
    assert document["memory"]["torch_peak_allocated_bytes"] == 100000
    assert {item["path"] for item in document["raw_artifacts"]} == {
        "result.json",
        "runner-metadata.json",
        "runner.log",
    }
    validate_document(document)


def test_external_baseline_rejects_core_runtime_coupling(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    metadata_path = run / "runner-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["isolation"]["imports_uma_qmoe_runtime"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContractError):
        _build(run)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("model", "model_revision", "f" * 40),
        ("workload", "warmup_requests", 2),
        ("workload", "max_concurrency", 2),
        ("runtime", "implementation_version", "2.0.0"),
        ("runtime", "container_image", "other/image@sha256:" + "9" * 64),
    ],
)
def test_external_baseline_rejects_relabelled_runner_metadata(
    tmp_path: Path, section: str, field: str, replacement: object
) -> None:
    run = tmp_path / "run"
    _write_run(run)
    metadata_path = run / "runner-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[section][field] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ContractError, match="does not match normalization"):
        _build(run)


def test_external_baseline_rejects_symlink_artifact_escape(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    outside = tmp_path / "outside.log"
    outside.write_text("secret\n", encoding="utf-8")
    (run / "runner.log").unlink()
    (run / "runner.log").symlink_to(outside)

    with pytest.raises(ContractError, match="must not use symbolic links"):
        _build(run)


def test_external_baseline_rejects_missing_run_directory(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="cannot read baseline input"):
        _build(tmp_path / "missing")


def test_external_baseline_validator_detects_tampering(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    document = _build(run)

    tampered = copy.deepcopy(document)
    tampered["results"]["request_throughput"] = 999.0
    with pytest.raises(ContractError, match="throughput"):
        validate_document(tampered)


def test_external_baseline_run_manifest_binds_generic_artifacts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    baseline = run / "external-baseline.json"
    baseline.write_text(json.dumps(_build(run)), encoding="utf-8")

    manifest = build_external_baseline_run_manifest(
        baseline,
        run,
        run_id="spark1-example-001",
        git_commit="e" * 40,
        git_dirty=False,
        dirty_patch_sha256=None,
        machine_baseline_sha256="f" * 64,
        benchmark_contract_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
    )

    assert manifest["status"] == "succeeded"
    assert len(manifest["raw_artifacts"]) == 4
    validate_document(manifest)
