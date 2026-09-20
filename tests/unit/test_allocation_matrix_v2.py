from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from uma_qmoe.allocation_matrix_v2 import (
    _allocation_compile_command,
    _normalize_native_output,
    _read_cgroup_state,
    _read_vmstat,
    _system_activity,
    validate_allocation_matrix_v2_document,
)
from uma_qmoe.contracts import ContractError


_SIZE = 8 * 1024 * 1024


def _touch(touch_id: str) -> dict:
    following_actor = "gpu" if touch_id == "cpu_first" else "cpu"
    return {
        "id": touch_id,
        "status": "measured",
        "seconds": 0.01,
        "minor_faults_delta": 10,
        "major_faults_delta": 0,
        "following_access": {
            "actor": following_actor,
            "status": "measured",
            "seconds": 0.02,
        },
    }


def _measured_case(case_id: str = "managed_unified") -> dict:
    return {
        "id": case_id,
        "status": "measured",
        "allocation_api": "native_test_allocator",
        "access_path": "native_test_access",
        "buffer_bytes": _SIZE,
        "allocation_seconds": 0.001,
        "touches": [_touch("cpu_first"), _touch("gpu_first")],
        "operations": [
            {"id": operation_id, "samples_seconds": [0.01, 0.011, 0.009]}
            for operation_id in ("read", "write", "copy")
        ],
        "contention": {
            "status": "measured",
            "wall_samples_seconds": [0.02, 0.021, 0.019],
            "gpu_samples_seconds": [0.01, 0.011, 0.009],
        },
    }


def _unavailable_case(case_id: str) -> dict:
    return {
        "id": case_id,
        "status": "unavailable",
        "allocation_api": "native_test_allocator",
        "access_path": "native_test_access",
        "reason": "capability_attribute_false",
    }


def _raw() -> dict:
    return {
        "backend": "cuda",
        "device_name": "NVIDIA GB10",
        "total_global_memory_bytes": 128_495_218_688,
        "multiprocessor_count": 48,
        "threads_per_block": 256,
        "blocks": 1536,
        "requested_buffer_bytes": _SIZE,
        "capabilities": {
            "managed_memory": 1,
            "concurrent_managed_access": 1,
            "pageable_memory_access": 1,
            "pageable_memory_access_uses_host_page_tables": 1,
        },
        "cases": [
            _measured_case(),
            _unavailable_case("system_pageable_direct"),
            _unavailable_case("platform_vmm"),
        ],
    }


def _point() -> dict:
    passing = _raw()
    passing["cases"] = [
        _measured_case("managed_unified"),
        _measured_case("system_pageable_direct"),
        _measured_case("platform_vmm"),
    ]
    normalized = _normalize_native_output(
        passing,
        backend="cuda",
        requested_buffer_bytes=_SIZE,
        measured_iterations=3,
        inner_iterations=2,
    )
    normalized.pop("runtime")
    normalized["measured_case_count"] = 3
    normalized["unavailable_case_count"] = 0
    return normalized


def _events(value: int = 0) -> dict[str, int]:
    return {
        "high": value,
        "max": value,
        "oom": value,
        "oom_kill": value,
        "oom_group_kill": value,
    }


def _document() -> dict:
    activity, memory_gates = _system_activity(
        {"swap_in_pages": 1, "swap_out_pages": 2},
        {"swap_in_pages": 1, "swap_out_pages": 2},
        {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()},
        {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()},
    )
    gates = {
        "required_paths_passed": True,
        "required_touch_chains_passed": True,
        "required_contention_passed": True,
        "steady_state_stability_passed": True,
        **memory_gates,
        "overall_passed": True,
    }
    return {
        "schema_version": 2,
        "kind": "allocation_matrix_v2",
        "generated_at": "2026-09-18T08:00:00Z",
        "target_id": "spark1-shanghai-zihaomu",
        "status": "passed",
        "source": {
            "path": "benchmarks/native/allocation_matrix_v2.cu",
            "sha256": "a" * 64,
        },
        "build": {
            "backend": "cuda",
            "compiler": "nvcc",
            "compiler_version": "Cuda compilation tools, release 13.0",
            "architecture": "sm_121",
            "optimization": "O3",
        },
        "runtime": {
            "backend": "cuda",
            "device_name": "NVIDIA GB10",
            "total_global_memory_bytes": 128_495_218_688,
            "multiprocessor_count": 48,
            "threads_per_block": 256,
            "blocks": 1536,
            "capabilities": {
                "managed_memory": 1,
                "concurrent_managed_access": 1,
                "pageable_memory_access": 1,
                "pageable_memory_access_uses_host_page_tables": 1,
            },
        },
        "configuration": {
            "buffer_sizes_bytes": [_SIZE],
            "warmup_iterations": 1,
            "measured_iterations": 3,
            "inner_iterations": 2,
            "maximum_coefficient_of_variation": 0.20,
        },
        "measurement_scope": {
            "elapsed_time": "gpu_events_and_host_monotonic",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "page_fault_scope": "calling_process_host_faults",
            "limitations": ["algorithmic bytes are not hardware counters"],
        },
        "pressure_points": [_point()],
        "system_activity": activity,
        "gates": gates,
    }


def test_normalize_native_output_keeps_raw_samples_and_accounts_bytes() -> None:
    point = _normalize_native_output(
        _raw(),
        backend="cuda",
        requested_buffer_bytes=_SIZE,
        measured_iterations=3,
        inner_iterations=2,
    )
    case = point["cases"][0]
    assert case["operations"][0]["samples_seconds"] == [0.01, 0.011, 0.009]
    assert case["operations"][0]["algorithmic_read_bytes"] == _SIZE * 2
    assert case["operations"][1]["algorithmic_write_bytes"] == _SIZE * 2
    assert case["operations"][2]["algorithmic_total_bytes"] == _SIZE * 4
    assert case["touches"][0]["host_minor_faults_delta"] == 10
    assert case["contention"]["wall"]["samples_seconds"] == [0.02, 0.021, 0.019]


def test_normalize_native_output_is_fail_closed() -> None:
    boolean_capability = _raw()
    boolean_capability["capabilities"]["managed_memory"] = True
    with pytest.raises(ContractError, match="integer booleans"):
        _normalize_native_output(
            boolean_capability,
            backend="cuda",
            requested_buffer_bytes=_SIZE,
            measured_iterations=3,
            inner_iterations=1,
        )

    missing_sample = _raw()
    missing_sample["cases"][0]["operations"][0]["samples_seconds"].pop()
    with pytest.raises(ContractError, match="every raw measured sample"):
        _normalize_native_output(
            missing_sample,
            backend="cuda",
            requested_buffer_bytes=_SIZE,
            measured_iterations=3,
            inner_iterations=1,
        )

    duplicate_case = _raw()
    duplicate_case["cases"][2]["id"] = "managed_unified"
    with pytest.raises(ContractError, match="each allocation path"):
        _normalize_native_output(
            duplicate_case,
            backend="cuda",
            requested_buffer_bytes=_SIZE,
            measured_iterations=3,
            inner_iterations=1,
        )


def test_v2_document_validates_and_recomputes_gates() -> None:
    evidence = _document()
    validate_allocation_matrix_v2_document(evidence)

    tampered_gate = copy.deepcopy(evidence)
    tampered_gate["gates"]["no_oom_events_passed"] = False
    with pytest.raises(ContractError, match="gates do not match"):
        validate_allocation_matrix_v2_document(tampered_gate)

    tampered_bytes = copy.deepcopy(evidence)
    tampered_bytes["pressure_points"][0]["cases"][0]["operations"][2][
        "algorithmic_total_bytes"
    ] -= 1
    with pytest.raises(ContractError, match="invalid algorithmic traffic"):
        validate_allocation_matrix_v2_document(tampered_bytes)

    tampered_summary = copy.deepcopy(evidence)
    tampered_summary["pressure_points"][0]["cases"][0]["operations"][0][
        "coefficient_of_variation"
    ] = 0.0
    with pytest.raises(ContractError, match="inconsistent with raw samples"):
        validate_allocation_matrix_v2_document(tampered_summary)

    missing_required_path = copy.deepcopy(evidence)
    missing_required_path["pressure_points"][0]["cases"][1] = {
        "id": "system_pageable_direct",
        "status": "unavailable",
        "allocation_api": "native_test_allocator",
        "access_path": "native_test_access",
        "reason": "pageable_allocation_failed",
    }
    missing_required_path["pressure_points"][0]["measured_case_count"] = 2
    missing_required_path["pressure_points"][0]["unavailable_case_count"] = 1
    with pytest.raises(ContractError, match="gates do not match"):
        validate_allocation_matrix_v2_document(missing_required_path)


def test_system_activity_uses_workload_cgroup_for_swap_and_oom_gates() -> None:
    activity, gates = _system_activity(
        {"swap_in_pages": 10, "swap_out_pages": 20},
        {"swap_in_pages": 11, "swap_out_pages": 20},
        {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()},
        {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()},
    )
    assert activity["host_vmstat"]["swap_in_pages_delta"] == 1
    assert gates == {
        "workload_cgroup_swap_disabled_passed": True,
        "no_swap_activity_passed": True,
        "no_oom_events_passed": True,
    }

    after = {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()}
    after["events"]["oom"] = 1
    _, failed = _system_activity(
        {"swap_in_pages": 10, "swap_out_pages": 20},
        {"swap_in_pages": 10, "swap_out_pages": 20},
        {"swap_current_bytes": 0, "swap_max_bytes": 0, "events": _events()},
        after,
    )
    assert failed["no_oom_events_passed"] is False


def test_proc_and_cgroup_parsers_reject_incomplete_evidence(tmp_path: Path) -> None:
    vmstat = tmp_path / "vmstat"
    vmstat.write_text("pswpin 2\npswpout 3\npgfault 4\n", encoding="utf-8")
    assert _read_vmstat(vmstat) == {"swap_in_pages": 2, "swap_out_pages": 3}

    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.swap.current").write_text("0\n", encoding="utf-8")
    (cgroup / "memory.swap.max").write_text("0\n", encoding="utf-8")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
        encoding="utf-8",
    )
    assert _read_cgroup_state(cgroup)["events"] == _events()

    (cgroup / "memory.events").write_text("oom 0\n", encoding="utf-8")
    with pytest.raises(ContractError, match="missing required counters"):
        _read_cgroup_state(cgroup)


def test_schema_file_is_valid_json() -> None:
    schema = Path("src/uma_qmoe/schemas/allocation_matrix_v2.schema.json")
    assert json.loads(schema.read_text(encoding="utf-8"))["title"].endswith("v2")


def test_cuda_compile_uses_nvcc_host_thread_flag(tmp_path: Path) -> None:
    command = _allocation_compile_command(
        compiler=str(tmp_path / "nvcc"),
        backend="cuda",
        architecture="sm_121",
        source=Path("allocation_matrix_v2.cu"),
        executable=Path("allocation-matrix-v2"),
    )
    assert "-Xcompiler=-pthread" in command
    assert "-pthread" not in command
    assert "-lcuda" in command


def test_native_read_kernel_avoids_global_atomic_contention() -> None:
    source = Path("benchmarks/native/allocation_matrix_v2.cu").read_text(
        encoding="utf-8"
    )
    read_kernel = source.split("__global__ void read_kernel", 1)[1].split(
        "template <typename Launch>", 1
    )[0]
    assert "atomicAdd" not in read_kernel
    assert "sink[blockIdx.x]" in read_kernel
    assert "constexpr int kThreads = 256;" in source
    assert "const int threads = kThreads;" in source
