from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.memory_bandwidth import _summarize_samples


def _evidence() -> dict:
    size = 512 * 1024 * 1024
    base = {
        "schema_version": 1,
        "kind": "memory_bandwidth_benchmark",
        "generated_at": "2026-09-17T14:30:00Z",
        "target_id": "spark1",
        "status": "passed",
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": ["provisional"],
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.13.0",
            "cuda": "13.0",
            "hip": None,
            "accelerator_name": "NVIDIA GB10",
            "compute_capability": [12, 1],
        },
        "configuration": {
            "requested_buffer_bytes": size,
            "actual_buffer_bytes": size,
            "dtype": "float32",
            "element_count": size // 4,
            "warmup_iterations": 3,
            "measured_iterations": 3,
            "inner_iterations": 1,
        },
        "memory": {
            "free_before_allocation_bytes": 100 * size,
            "total_bytes": 200 * size,
            "free_after_allocation_bytes": 98 * size,
            "peak_allocated_bytes": 2 * size,
            "peak_reserved_bytes": 2 * size,
        },
        "operations": [],
    }
    for operation_id, read_bytes, write_bytes in (
        ("read_reduce", size, 0),
        ("write_fill", 0, size),
        ("copy", size, size),
    ):
        base["operations"].append(
            {
                "id": operation_id,
                "implementation": "test",
                "algorithmic_read_bytes": read_bytes,
                "algorithmic_write_bytes": write_bytes,
                "algorithmic_total_bytes": read_bytes + write_bytes,
                "samples_seconds": [0.01, 0.011, 0.009],
                "median_seconds": 0.01,
                "mean_seconds": 0.01,
                "coefficient_of_variation": 0.08,
                "effective_gbps": (read_bytes + write_bytes) / 0.01 / 1e9,
            }
        )
    return base


def test_memory_bandwidth_evidence_validates() -> None:
    validate_document(_evidence())


def test_memory_bandwidth_rejects_wrong_traffic_accounting() -> None:
    evidence = copy.deepcopy(_evidence())
    evidence["operations"][2]["algorithmic_total_bytes"] -= 1
    with pytest.raises(ContractError, match="algorithmic_total_bytes"):
        validate_document(evidence)


def test_memory_bandwidth_telemetry_requires_complete_ordered_boundaries() -> None:
    evidence = _evidence()
    samples = []
    for operation_id in ("read_reduce", "write_fill", "copy"):
        for sample_index in range(3):
            for boundary in ("before", "after"):
                samples.append(
                    {
                        "operation_id": operation_id,
                        "sample_index": sample_index,
                        "boundary": boundary,
                        "monotonic_ns": len(samples) + 1,
                        "temperature_c": 40.0,
                        "socket_power_w": 10.0,
                        "graphics_clock_mhz": 1000.0,
                        "memory_clock_mhz": None,
                        "utilization_percent": 50.0,
                    }
                )
    evidence["telemetry"] = {
        "status": "available",
        "provider": "nvidia_smi",
        "collection_mode": "sample_boundaries_outside_timed_region",
        "samples": samples,
    }
    validate_document(evidence)

    evidence["telemetry"]["samples"][0], evidence["telemetry"]["samples"][1] = (
        evidence["telemetry"]["samples"][1],
        evidence["telemetry"]["samples"][0],
    )
    with pytest.raises(ContractError, match="ordered before/after"):
        validate_document(evidence)


def test_sample_summary_uses_median_and_rejects_invalid_values() -> None:
    summary = _summarize_samples([1.0, 2.0, 100.0], 1_000_000_000)
    assert summary["median_seconds"] == 2.0
    assert summary["effective_gbps"] == 0.5

    with pytest.raises(ContractError, match="finite and positive"):
        _summarize_samples([1.0, 0.0, 2.0], 1)
