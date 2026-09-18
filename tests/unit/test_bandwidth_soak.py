from __future__ import annotations

import copy
from pathlib import Path

import pytest

from uma_qmoe.bandwidth_soak import (
    _cgroup_memory_state,
    _summarize_bandwidth,
    _summarize_telemetry,
)
from uma_qmoe.contracts import ContractError, validate_document


def _evidence(*, full_duration: bool = False) -> dict:
    size = 512 * 1024 * 1024
    inner = 64
    requested_duration = 1800 if full_duration else 10
    elapsed = [450.0, 900.0, 1350.0, 1800.5] if full_duration else [2.5, 5.0, 7.5, 10.5]
    timings = {"read_reduce": 0.16, "write_fill": 0.15, "copy": 0.32}
    samples = [
        {
            "sample_index": index,
            "monotonic_ns": 1_000_000_000 + index * 10_000_000,
            "elapsed_from_start_seconds": elapsed[index],
            "timings_seconds": dict(timings),
        }
        for index in range(4)
    ]
    telemetry_samples = [
        {
            "operation_id": "soak_cycle",
            "sample_index": index,
            "boundary": "after",
            "monotonic_ns": 1_005_000_000 + index * 20_000_000,
            "temperature_c": 40.0 + index,
            "socket_power_w": 25.0,
            "graphics_clock_mhz": 2000.0,
            "memory_clock_mhz": None,
            "utilization_percent": 99.0,
        }
        for index in (0, 2)
    ]
    traffic = {
        "read_reduce": (size * inner, 0),
        "write_fill": (0, size * inner),
        "copy": (size * inner, size * inner),
    }
    operations = []
    for operation_id in ("read_reduce", "write_fill", "copy"):
        read_bytes, write_bytes = traffic[operation_id]
        operations.append(
            {
                "id": operation_id,
                "algorithmic_read_bytes": read_bytes,
                "algorithmic_write_bytes": write_bytes,
                "algorithmic_total_bytes": read_bytes + write_bytes,
                **_summarize_bandwidth(
                    [sample["timings_seconds"][operation_id] for sample in samples],
                    read_bytes + write_bytes,
                ),
            }
        )
    gates = {
        "minimum_duration_passed": full_duration,
        "no_swap_activity_passed": True,
        "coefficient_of_variation_passed": True,
        "drift_passed": True,
        "overall_passed": full_duration,
    }
    return {
        "schema_version": 1,
        "kind": "bandwidth_soak",
        "generated_at": "2026-09-18T02:15:00Z",
        "completed_at": "2026-09-18T02:45:00Z",
        "target_id": "halo3",
        "status": "passed" if full_duration else "diagnostic",
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": ["test"],
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.14.0",
            "cuda": None,
            "hip": "7.15",
            "accelerator_name": "AMD Radeon 8060S Graphics",
            "compute_capability": None,
        },
        "configuration": {
            "requested_buffer_bytes": size,
            "actual_buffer_bytes": size,
            "dtype": "float32",
            "element_count": size // 4,
            "requested_duration_seconds": requested_duration,
            "minimum_full_soak_seconds": 1800,
            "actual_duration_seconds": elapsed[-1],
            "warmup_cycles": 3,
            "inner_iterations": inner,
            "telemetry_interval_seconds": 5.0,
            "maximum_coefficient_of_variation": 0.03,
            "maximum_absolute_drift_fraction": 0.05,
        },
        "memory": {
            "free_before_allocation_bytes": 100 * size,
            "total_bytes": 200 * size,
            "free_after_allocation_bytes": 98 * size,
            "free_after_soak_bytes": 98 * size,
            "peak_allocated_bytes": 2 * size,
            "peak_reserved_bytes": 2 * size,
            "swap_in_pages_before": 0,
            "swap_in_pages_after": 0,
            "swap_in_pages_delta": 0,
            "swap_out_pages_before": 0,
            "swap_out_pages_after": 0,
            "swap_out_pages_delta": 0,
        },
        "samples": samples,
        "operations": operations,
        "telemetry": {
            "status": "available",
            "provider": "amd_smi",
            "collection_mode": "periodic_after_cycle_outside_timed_regions",
            "samples": telemetry_samples,
            "summary": _summarize_telemetry(telemetry_samples),
        },
        "gates": gates,
    }


def test_bandwidth_soak_diagnostic_and_full_evidence_validate() -> None:
    validate_document(_evidence())
    validate_document(_evidence(full_duration=True))


def test_bandwidth_soak_rejects_tampered_summary_and_gates() -> None:
    summary = copy.deepcopy(_evidence())
    summary["operations"][0]["p50_gbps"] += 1.0
    with pytest.raises(ContractError, match="invalid p50_gbps"):
        validate_document(summary)

    gates = copy.deepcopy(_evidence())
    gates["gates"]["overall_passed"] = True
    with pytest.raises(ContractError, match="gates"):
        validate_document(gates)


def test_bandwidth_soak_summary_reports_tail_and_drift() -> None:
    summary = _summarize_bandwidth([1.0, 1.0, 1.0, 2.0], 1_000_000_000)
    assert summary["sample_count"] == 4
    assert summary["p50_gbps"] == 1.0
    assert summary["p95_gbps"] == 1.0
    assert summary["minimum_gbps"] == 0.5


def test_bandwidth_soak_validates_workload_cgroup_swap_and_oom_gate() -> None:
    evidence = _evidence(full_duration=True)
    events = {"high": 0, "max": 0, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}
    evidence["memory"]["workload_cgroup"] = {
        "gate_scope": "workload_cgroup_v2",
        "swap_current_bytes_before": 0,
        "swap_current_bytes_after": 0,
        "swap_current_bytes_delta": 0,
        "swap_max_bytes": 0,
        "memory_events_before": dict(events),
        "memory_events_after": dict(events),
        "memory_events_delta": dict(events),
    }
    evidence["gates"]["workload_cgroup_swap_disabled_passed"] = True
    evidence["gates"]["no_oom_events_passed"] = True
    validate_document(evidence)

    tampered = copy.deepcopy(evidence)
    tampered["memory"]["workload_cgroup"]["memory_events_delta"]["oom"] = 1
    with pytest.raises(ContractError, match="event deltas"):
        validate_document(tampered)


def test_cgroup_memory_state_parses_v2_files(tmp_path: Path) -> None:
    (tmp_path / "memory.swap.current").write_text("0\n", encoding="utf-8")
    (tmp_path / "memory.swap.max").write_text("0\n", encoding="utf-8")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 1\nmax 2\noom 0\noom_kill 0\noom_group_kill 0\n",
        encoding="utf-8",
    )
    assert _cgroup_memory_state(tmp_path) == {
        "swap_current_bytes": 0,
        "swap_max_bytes": 0,
        "events": {"high": 1, "max": 2, "oom": 0, "oom_kill": 0, "oom_group_kill": 0},
    }
