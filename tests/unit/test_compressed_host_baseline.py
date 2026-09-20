from __future__ import annotations

import json
from pathlib import Path

import pytest

from uma_qmoe.compressed_host_baseline import (
    build_compressed_host_baseline,
    build_compressed_host_run_manifest,
)
from uma_qmoe.contracts import ContractError, validate_document


def _write_run(directory: Path, *, target_policy_id: str | None = None) -> None:
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
    cache = {
        "tensor_count": 24,
        "device_storage_bytes": 1000,
        "dequantized_weight_bytes": 0,
    }
    loader = {
        "performance_mode": True,
        "dense_tensor_count": 147,
        "dense_tensor_bytes": 953421824,
        "skipped_expert_tensor_count": 3072,
        "loaded_expert_tensor_count": 0,
        "expert_parameter_count": 0,
        "quantized_moe_block_count": 16,
        "model_parameter_bytes": 953421824,
        "expert_pack_size_bytes": 3423698944,
        "expert_pack_mapping_count": 1,
        "expert_pack_vma_count": 1,
    }
    metadata = {
        "target_id": "halo3",
        "status": "succeeded",
        "returncode": 0,
        "started_at": "2026-09-19T01:00:00Z",
        "completed_at": "2026-09-19T01:10:00Z",
        "runner_argv": [
            "python3",
            "benchmarks/runners/run_compressed_olmoe_baseline.py",
        ],
        "artifact_paths": ["result.json", "runner-metadata.json"],
        "offline_local_model": True,
        "generation_loop": "manual_past_key_values_greedy",
        "runtime": {
            "source_commit": "a" * 40,
            "backend": "hip",
            "platform": "hip_gfx1151",
            "container_image": "example/host@sha256:" + "b" * 64,
            "torch_version": "2.14.0",
            "transformers_version": "5.10.2",
            "backend_version": "7.15.0",
            "native_source_sha256": "c" * 64,
            "fallback_count": 0,
        },
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "d" * 64,
            "target_policy_id": target_policy_id,
        },
        "workload": {
            "input_tokens": 4,
            "output_tokens": 3,
            "warmup_requests": 1,
            "measured_requests": 3,
            "max_concurrency": 1,
        },
        "loader": loader,
        "backend_cache": {
            "after_warmup": cache,
            "after_measurement": dict(cache),
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


def _build(directory: Path, *, target_policy_id: str | None = None) -> dict:
    return build_compressed_host_baseline(
        directory,
        target_id="halo3",
        source_commit="a" * 40,
        backend="hip",
        container_image="example/host@sha256:" + "b" * 64,
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="9b0c1aa87e34a20052389dce1f0cf01da783f654",
        expert_pack_sha256="d" * 64,
        target_policy_id=target_policy_id,
        input_tokens=4,
        output_tokens=3,
        warmup_requests=1,
        measured_requests=3,
    )


def test_compressed_host_baseline_binds_native_performance_mode(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_run(run)
    document = _build(run)
    assert document["status"] == "passed"
    assert document["implementation"]["platform"] == "hip_gfx1151"
    assert document["gates"]["cache_stable_after_warmup"] is True
    assert document["gates"]["no_dequantized_weight_cache"] is True
    validate_document(document)


def test_compressed_host_v2_binds_mixed_target_policy(tmp_path: Path) -> None:
    run = tmp_path / "run"
    policy_id = "olmoe-layer15-awq-q4-q8-bf16-v2"
    _write_run(run, target_policy_id=policy_id)
    document = _build(run, target_policy_id=policy_id)
    assert document["schema_version"] == 2
    assert document["model"]["expert_quantization"] == (
        "mixed_target_pack_q4_q8_bf16"
    )
    assert document["model"]["target_policy_id"] == policy_id
    validate_document(document)

    document["model"]["target_policy_id"] = (
        "olmoe-layer15-awq-q4-q8-bf16-v1"
    )
    with pytest.raises(ContractError):
        validate_document(document)


def test_compressed_host_baseline_rejects_cache_growth(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    metadata_path = run / "runner-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["backend_cache"]["after_measurement"]["tensor_count"] = 25
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    document = _build(run)
    assert document["status"] == "failed"
    assert document["gates"]["cache_stable_after_warmup"] is False
    validate_document(document)


def test_compressed_host_baseline_rejects_relabelled_pack(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    with pytest.raises(ContractError, match="model identity"):
        build_compressed_host_baseline(
            run,
            target_id="halo3",
            source_commit="a" * 40,
            backend="hip",
            container_image="example/host@sha256:" + "b" * 64,
            model_id="allenai/OLMoE-1B-7B-0125",
            model_revision="9b0c1aa87e34a20052389dce1f0cf01da783f654",
            expert_pack_sha256="e" * 64,
            input_tokens=4,
            output_tokens=3,
            warmup_requests=1,
            measured_requests=3,
        )


def test_compressed_host_run_manifest_binds_pack_and_trace(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    baseline = run / "compressed-host-baseline.json"
    baseline.write_text(json.dumps(_build(run)), encoding="utf-8")
    manifest = build_compressed_host_run_manifest(
        baseline,
        run,
        route_trace_sha256="e" * 64,
        run_id="halo3-compressed-host-001",
        git_commit="f" * 40,
        git_dirty=False,
        dirty_patch_sha256=None,
        machine_baseline_sha256="1" * 64,
        benchmark_contract_sha256="2" * 64,
        model_manifest_sha256="3" * 64,
    )
    assert manifest["inputs"]["expert_pack_sha256"] == "d" * 64
    assert manifest["inputs"]["route_trace_sha256"] == "e" * 64
    validate_document(manifest)
