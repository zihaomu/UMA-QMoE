from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.public_baseline import (
    build_public_baseline,
    build_public_baseline_run_manifest,
)


def _write_run(directory: Path) -> None:
    directory.mkdir()
    raw = {
        "duration": 2.0,
        "completed": 3,
        "failed": 0,
        "total_input_tokens": 12,
        "total_output_tokens": 9,
        "request_throughput": 1.5,
        "output_throughput": 4.5,
        "total_token_throughput": 10.5,
        "input_lens": [4, 4, 4],
        "output_lens": [3, 3, 3],
        "ttfts": [0.1, 0.2, 0.3],
        "itls": [[0.02, 0.03], [0.03, 0.04], [0.04, 0.05]],
        "errors": ["", "", ""],
    }
    metadata = {
        "target_id": "spark1",
        "status": "succeeded",
        "started_at": "2026-09-18T01:00:00Z",
        "completed_at": "2026-09-18T01:10:00Z",
        "runner_argv": ["python3", "benchmarks/runners/vllm_public_baseline.py"],
        "benchmark_returncode": 0,
        "cgroup_memory_peak_bytes": 123456,
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
    (directory / "vllm-result.json").write_text(json.dumps(raw), encoding="utf-8")
    (directory / "runner-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (directory / "server.log").write_text("ready\n", encoding="utf-8")
    (directory / "benchmark.log").write_text("done\n", encoding="utf-8")


def _build(directory: Path) -> dict:
    return build_public_baseline(
        directory,
        target_id="spark1",
        implementation_version="0.29.0",
        source_commit="a" * 40,
        backend="cuda",
        container_image="example/vllm@sha256:" + "b" * 64,
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="c" * 40,
        derivation_semantic_sha256="d" * 64,
        input_tokens=4,
        output_tokens=3,
        warmup_requests=1,
        measured_requests=3,
        max_concurrency=1,
    )


def test_build_public_baseline_recomputes_latency_and_resource_evidence(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_run(run)

    document = _build(run)

    assert document["status"] == "passed"
    assert document["gates"]["overall_passed"] is True
    assert document["results"]["tpot_ms"]["mean"] == pytest.approx(35.0)
    assert document["results"]["e2el_ms"]["max"] == pytest.approx(390.0)
    assert document["memory"]["peak_bytes"] == 123456
    assert document["telemetry"]["maximum_temperature_c"] == 55.0
    assert document["telemetry"]["minimum_memory_available_bytes"] == 900
    assert {item["path"] for item in document["raw_artifacts"]} == {
        "vllm-result.json",
        "runner-metadata.json",
        "server.log",
        "benchmark.log",
    }
    validate_document(document)


def test_public_baseline_rejects_missing_detailed_latency(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    raw_path = run / "vllm-result.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["itls"][0] = [0.01]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="latency count"):
        _build(run)


def test_public_baseline_validator_recomputes_throughput_and_status(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_run(run)
    document = _build(run)

    tampered = copy.deepcopy(document)
    tampered["results"]["output_token_throughput"] = 99.0
    with pytest.raises(ContractError, match="throughput"):
        validate_document(tampered)

    tampered = copy.deepcopy(document)
    tampered["gates"]["overall_passed"] = False
    with pytest.raises(ContractError, match="overall gate"):
        validate_document(tampered)


def test_build_public_baseline_run_manifest_binds_raw_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    baseline_path = run / "baseline.json"
    baseline_path.write_text(json.dumps(_build(run)), encoding="utf-8")

    manifest = build_public_baseline_run_manifest(
        baseline_path,
        run,
        run_id="spark1-bf16-vllm-001",
        git_commit="e" * 40,
        git_dirty=False,
        dirty_patch_sha256=None,
        machine_baseline_sha256="f" * 64,
        benchmark_contract_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
    )

    assert manifest["status"] == "succeeded"
    assert manifest["git"]["dirty"] is False
    assert manifest["command"]["argv"][1].endswith("vllm_public_baseline.py")
    assert len(manifest["raw_artifacts"]) == 5
    validate_document(manifest)
