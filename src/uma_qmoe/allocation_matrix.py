"""Diagnostic allocation-path matrix for unified-memory targets."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
import mmap
import platform
import resource
import statistics
import tempfile
import time
from typing import Any, Callable

from .contracts import ContractError


_MIN_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_BUFFER_BYTES = 2 * 1024 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _faults() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_minflt), int(usage.ru_majflt)


def _vmstat_swap() -> tuple[int, int] | None:
    try:
        text = open("/proc/vmstat", encoding="utf-8").read()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"pswpin", "pswpout"}:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError:
                return None
    if set(values) != {"pswpin", "pswpout"}:
        return None
    return values["pswpin"], values["pswpout"]


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _summarize_transfer(
    samples: list[float], payload_bytes: int, algorithmic_total_bytes: int
) -> dict[str, Any]:
    if len(samples) < 3:
        raise ContractError("at least three steady-state samples are required")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise ContractError("transfer samples must be finite and positive")
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "p95_seconds": _p95(samples),
        "mean_seconds": mean,
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
        "payload_gbps": payload_bytes / median / 1_000_000_000,
        "algorithmic_total_gbps": algorithmic_total_bytes
        / median
        / 1_000_000_000,
    }


def _measure_case(
    *,
    torch: Any,
    case_id: str,
    allocation_api: str,
    access_path: str,
    allocation_seconds: float,
    pretouch_seconds: float | None,
    operation: Callable[[], None],
    payload_bytes: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    first_faults_before = _faults()
    torch.cuda.synchronize()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - started
    first_faults_after = _faults()

    for _ in range(warmup_iterations):
        operation()
        torch.cuda.synchronize()
    steady_faults_before = _faults()
    samples: list[float] = []
    for _ in range(measured_iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ContractError(f"allocation case {case_id!r} produced invalid timing")
        samples.append(elapsed)
    steady_faults_after = _faults()
    algorithmic_total_bytes = payload_bytes * 2
    return {
        "id": case_id,
        "status": "measured",
        "allocation_api": allocation_api,
        "access_path": access_path,
        "measurement": {
            "allocation_seconds": allocation_seconds,
            "pretouch_seconds": pretouch_seconds,
            "payload_bytes": payload_bytes,
            "algorithmic_read_bytes": payload_bytes,
            "algorithmic_write_bytes": payload_bytes,
            "first_access": {
                "seconds": first_seconds,
                "payload_gbps": payload_bytes / first_seconds / 1_000_000_000,
                "algorithmic_total_gbps": algorithmic_total_bytes
                / first_seconds
                / 1_000_000_000,
            },
            "steady_state": _summarize_transfer(
                samples, payload_bytes, algorithmic_total_bytes
            ),
            "page_faults": {
                "first_access_minor_delta": first_faults_after[0]
                - first_faults_before[0],
                "first_access_major_delta": first_faults_after[1]
                - first_faults_before[1],
                "steady_state_minor_delta": steady_faults_after[0]
                - steady_faults_before[0],
                "steady_state_major_delta": steady_faults_after[1]
                - steady_faults_before[1],
            },
        },
    }


def _unmeasured(case_id: str, allocation_api: str, access_path: str, reason: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "status": "not_measured",
        "allocation_api": allocation_api,
        "access_path": access_path,
        "reason": reason,
    }


def benchmark_allocation_matrix(
    *,
    target_id: str,
    requested_buffer_bytes: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Measure the safe PyTorch-accessible slice of the M0 Allocation Matrix."""

    if not target_id.strip():
        raise ContractError("target_id must be non-empty")
    if not _MIN_BUFFER_BYTES <= requested_buffer_bytes <= _MAX_BUFFER_BYTES:
        raise ContractError(
            f"requested_buffer_bytes must be within [{_MIN_BUFFER_BYTES}, {_MAX_BUFFER_BYTES}]"
        )
    if not 1 <= warmup_iterations <= 100:
        raise ContractError("warmup_iterations must be within [1, 100]")
    if not 3 <= measured_iterations <= 100:
        raise ContractError("measured_iterations must be within [3, 100]")
    try:
        import torch
    except ImportError as exc:
        raise ContractError("Allocation Matrix requires torch") from exc
    if not torch.cuda.is_available():
        raise ContractError("no CUDA/HIP device is available to PyTorch")

    element_size = torch.empty((), dtype=torch.float32).element_size()
    element_count = requested_buffer_bytes // element_size
    actual_bytes = element_count * element_size
    free_before, total_memory = torch.cuda.mem_get_info()
    if actual_bytes * 2 > int(free_before * 0.5):
        raise ContractError("matrix buffers would consume more than 50% of free GPU memory")
    swap_before = _vmstat_swap()
    cases: list[dict[str, Any]] = []
    device = torch.device("cuda:0")

    allocation_started = time.perf_counter()
    device_source = torch.ones(element_count, dtype=torch.float32, device=device)
    device_destination = torch.empty_like(device_source)
    torch.cuda.synchronize()
    allocation_seconds = time.perf_counter() - allocation_started
    cases.append(
        _measure_case(
            torch=torch,
            case_id="runtime_device_copy",
            allocation_api="torch.empty(cuda)",
            access_path="device_to_device_copy",
            allocation_seconds=allocation_seconds,
            pretouch_seconds=None,
            operation=lambda destination=device_destination, source=device_source: (
                destination.copy_(source)
            ),
            payload_bytes=actual_bytes,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )
    )
    del device_source, device_destination
    torch.cuda.empty_cache()

    allocation_started = time.perf_counter()
    host_pageable = torch.ones(element_count, dtype=torch.float32)
    device_destination = torch.empty(element_count, dtype=torch.float32, device=device)
    allocation_seconds = time.perf_counter() - allocation_started
    cases.append(
        _measure_case(
            torch=torch,
            case_id="host_pageable_h2d",
            allocation_api="torch.empty(cpu,pageable)+torch.empty(cuda)",
            access_path="host_pageable_to_device_staging_copy",
            allocation_seconds=allocation_seconds,
            pretouch_seconds=0.0,
            operation=lambda destination=device_destination, source=host_pageable: (
                destination.copy_(source, non_blocking=False)
            ),
            payload_bytes=actual_bytes,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )
    )
    del host_pageable, device_destination
    torch.cuda.empty_cache()

    pinned_source = None
    pinned_destination = None
    try:
        allocation_started = time.perf_counter()
        pinned_source = torch.ones(element_count, dtype=torch.float32, pin_memory=True)
        device_destination = torch.empty(element_count, dtype=torch.float32, device=device)
        allocation_seconds = time.perf_counter() - allocation_started
        cases.append(
            _measure_case(
                torch=torch,
                case_id="host_pinned_h2d",
                allocation_api="torch.empty(cpu,pin_memory=True)+torch.empty(cuda)",
                access_path="host_pinned_to_device_nonblocking_copy",
                allocation_seconds=allocation_seconds,
                pretouch_seconds=0.0,
                operation=lambda destination=device_destination, source=pinned_source: (
                    destination.copy_(source, non_blocking=True)
                ),
                payload_bytes=actual_bytes,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
            )
        )
        del device_destination
        torch.cuda.empty_cache()

        allocation_started = time.perf_counter()
        device_source = torch.ones(element_count, dtype=torch.float32, device=device)
        pinned_destination = torch.empty(
            element_count, dtype=torch.float32, pin_memory=True
        )
        allocation_seconds = time.perf_counter() - allocation_started
        cases.append(
            _measure_case(
                torch=torch,
                case_id="host_pinned_d2h",
                allocation_api="torch.empty(cuda)+torch.empty(cpu,pin_memory=True)",
                access_path="device_to_host_pinned_nonblocking_copy",
                allocation_seconds=allocation_seconds,
                pretouch_seconds=0.0,
                operation=lambda destination=pinned_destination, source=device_source: (
                    destination.copy_(source, non_blocking=True)
                ),
                payload_bytes=actual_bytes,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
            )
        )
        del device_source
        torch.cuda.empty_cache()
    except (RuntimeError, OSError):
        measured_ids = {case["id"] for case in cases}
        for case_id, access_path in (
            ("host_pinned_h2d", "host_pinned_to_device_nonblocking_copy"),
            ("host_pinned_d2h", "device_to_host_pinned_nonblocking_copy"),
        ):
            if case_id not in measured_ids:
                cases.append(
                    {
                        "id": case_id,
                        "status": "unavailable",
                        "allocation_api": "torch.empty(pin_memory=True)",
                        "access_path": access_path,
                        "reason": "pinned_allocation_or_transfer_failed",
                    }
                )
    finally:
        del pinned_source, pinned_destination
        gc.collect()

    try:
        with tempfile.TemporaryFile() as backing_file:
            backing_file.truncate(actual_bytes)
            allocation_started = time.perf_counter()
            mapping = mmap.mmap(backing_file.fileno(), actual_bytes, access=mmap.ACCESS_WRITE)
            allocation_seconds = time.perf_counter() - allocation_started
            pretouch_started = time.perf_counter()
            page_size = mmap.PAGESIZE
            for offset in range(0, actual_bytes, page_size):
                mapping[offset] = 1
            mapping[actual_bytes - 1] = 1
            pretouch_seconds = time.perf_counter() - pretouch_started
            host_mmap = torch.frombuffer(mapping, dtype=torch.float32, count=element_count)
            device_destination = torch.empty(
                element_count, dtype=torch.float32, device=device
            )
            cases.append(
                _measure_case(
                    torch=torch,
                    case_id="file_mmap_pretouched_h2d",
                    allocation_api="mmap(MAP_SHARED)+torch.frombuffer+torch.empty(cuda)",
                    access_path="pretouched_file_mapping_to_device_staging_copy",
                    allocation_seconds=allocation_seconds,
                    pretouch_seconds=pretouch_seconds,
                    operation=lambda destination=device_destination, source=host_mmap: (
                        destination.copy_(source, non_blocking=False)
                    ),
                    payload_bytes=actual_bytes,
                    warmup_iterations=warmup_iterations,
                    measured_iterations=measured_iterations,
                )
            )
            del host_mmap, device_destination
            torch.cuda.empty_cache()
            mapping.close()
    except (BufferError, OSError, RuntimeError, ValueError):
        cases.append(
            {
                "id": "file_mmap_pretouched_h2d",
                "status": "unavailable",
                "allocation_api": "mmap(MAP_SHARED)+torch.frombuffer+torch.empty(cuda)",
                "access_path": "pretouched_file_mapping_to_device_staging_copy",
                "reason": "mmap_allocation_or_transfer_failed",
            }
        )

    cases.extend(
        [
            _unmeasured(
                "managed_unified",
                "cudaMallocManaged_or_hipMallocManaged",
                "direct_gpu_access_to_managed_allocation",
                "native_managed_allocator_probe_not_implemented_v1",
            ),
            _unmeasured(
                "platform_vmm_hmm",
                "platform_vmm_or_hmm_api",
                "direct_gpu_access_to_platform_virtual_memory",
                "native_vmm_hmm_probe_not_implemented_v1",
            ),
        ]
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    free_after, _ = torch.cuda.mem_get_info()
    swap_after = _vmstat_swap()
    swap_in_delta = None
    swap_out_delta = None
    if swap_before is not None and swap_after is not None:
        swap_in_delta = max(0, swap_after[0] - swap_before[0])
        swap_out_delta = max(0, swap_after[1] - swap_before[1])

    return {
        "schema_version": 1,
        "kind": "allocation_matrix",
        "generated_at": _utc_now(),
        "target_id": target_id.strip(),
        "status": "diagnostic",
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "limitations": [
                "PyTorch staging copies are not direct zero-copy host allocation access",
                "managed/unified and VMM/HMM require a target-native allocator probe",
                "process page faults do not expose all device-side migration events",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "accelerator_name": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "requested_buffer_bytes": requested_buffer_bytes,
            "actual_buffer_bytes": actual_bytes,
            "dtype": "float32",
            "element_count": element_count,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "host_page_size_bytes": mmap.PAGESIZE,
        },
        "memory": {
            "free_device_before_bytes": free_before,
            "total_device_bytes": total_memory,
            "free_device_after_bytes": free_after,
        },
        "system_activity": {
            "swap_in_pages_delta": swap_in_delta,
            "swap_out_pages_delta": swap_out_delta,
        },
        "cases": cases,
    }


__all__ = ["_summarize_transfer", "benchmark_allocation_matrix"]
