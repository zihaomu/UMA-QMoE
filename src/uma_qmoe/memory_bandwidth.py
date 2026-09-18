"""Provisional streaming memory microbenchmark for UMA accelerators.

The v1 probe deliberately reports algorithmic bytes divided by synchronized
elapsed time. It is not a DRAM counter and does not claim to isolate LPDDR from
cache, reduction overhead, page placement, or framework launch overhead.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import platform
import statistics
import time
from typing import Any, Callable

from .contracts import ContractError
from .telemetry import TelemetryProbe


_MIN_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_BUFFER_BYTES = 8 * 1024 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _summarize_samples(samples: list[float], traffic_bytes: int) -> dict[str, Any]:
    if len(samples) < 3:
        raise ContractError("at least three measured samples are required")
    if traffic_bytes <= 0:
        raise ContractError("traffic_bytes must be positive")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise ContractError("all elapsed-time samples must be finite and positive")
    median_seconds = statistics.median(samples)
    mean_seconds = statistics.mean(samples)
    coefficient_of_variation = statistics.pstdev(samples) / mean_seconds
    return {
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "mean_seconds": mean_seconds,
        "coefficient_of_variation": coefficient_of_variation,
        "effective_gbps": traffic_bytes / median_seconds / 1_000_000_000,
    }


def benchmark_memory_bandwidth(
    *,
    target_id: str,
    requested_buffer_bytes: int,
    warmup_iterations: int,
    measured_iterations: int,
    inner_iterations: int = 1,
    collect_telemetry: bool = False,
) -> dict[str, Any]:
    """Run synchronized PyTorch read, write, and copy probes on GPU memory."""

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
    if not 1 <= inner_iterations <= 10_000:
        raise ContractError("inner_iterations must be within [1, 10000]")

    try:
        import torch
    except ImportError as exc:
        raise ContractError("memory bandwidth benchmark requires torch") from exc
    if not torch.cuda.is_available():
        raise ContractError("no CUDA/HIP device is available to PyTorch")

    element_size = torch.empty((), dtype=torch.float32).element_size()
    element_count = requested_buffer_bytes // element_size
    actual_buffer_bytes = element_count * element_size
    if element_count < 1:
        raise ContractError("requested buffer is smaller than one float32 element")

    free_before, total_memory = torch.cuda.mem_get_info()
    required_bytes = actual_buffer_bytes * 2
    if required_bytes > int(free_before * 0.5):
        raise ContractError(
            "two benchmark buffers would consume more than 50% of currently free GPU memory"
        )

    device = torch.device("cuda:0")
    source = torch.ones(element_count, dtype=torch.float32, device=device)
    destination = torch.empty_like(source)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()

    read_sink: list[Any] = []

    def read_reduce() -> None:
        read_sink[:] = [torch.sum(source)]

    write_value = [0.0]

    def write_fill() -> None:
        write_value[0] = 1.0 if write_value[0] == 0.0 else 0.0
        destination.fill_(write_value[0])

    def copy() -> None:
        destination.copy_(source)

    telemetry_probe = TelemetryProbe.detect() if collect_telemetry else None
    telemetry_samples: list[dict[str, Any]] = []

    def measure(operation_id: str, operation: Callable[[], None]) -> list[float]:
        for _ in range(warmup_iterations):
            for _ in range(inner_iterations):
                operation()
            torch.cuda.synchronize()
        samples: list[float] = []
        for sample_index in range(measured_iterations):
            if telemetry_probe is not None:
                telemetry_samples.append(
                    telemetry_probe.capture(
                        operation_id=operation_id,
                        sample_index=sample_index,
                        boundary="before",
                    )
                )
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(inner_iterations):
                operation()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ContractError("observed a non-finite or non-positive elapsed time")
            samples.append(elapsed)
            if telemetry_probe is not None:
                telemetry_samples.append(
                    telemetry_probe.capture(
                        operation_id=operation_id,
                        sample_index=sample_index,
                        boundary="after",
                    )
                )
        return samples

    operation_specs = [
        ("read_reduce", "torch.sum(float32)", actual_buffer_bytes, 0, read_reduce),
        ("write_fill", "torch.Tensor.fill_(float32)", 0, actual_buffer_bytes, write_fill),
        ("copy", "torch.Tensor.copy_(float32)", actual_buffer_bytes, actual_buffer_bytes, copy),
    ]
    operations: list[dict[str, Any]] = []
    for operation_id, implementation, read_bytes, write_bytes, operation in operation_specs:
        read_bytes *= inner_iterations
        write_bytes *= inner_iterations
        total_bytes = read_bytes + write_bytes
        operations.append(
            {
                "id": operation_id,
                "implementation": implementation,
                "algorithmic_read_bytes": read_bytes,
                "algorithmic_write_bytes": write_bytes,
                "algorithmic_total_bytes": total_bytes,
                **_summarize_samples(measure(operation_id, operation), total_bytes),
            }
        )

    if not read_sink or not bool(torch.isfinite(read_sink[0]).item()):
        raise ContractError("read reduction did not produce a finite sink")
    capability = torch.cuda.get_device_capability(0)
    document = {
        "schema_version": 1,
        "kind": "memory_bandwidth_benchmark",
        "generated_at": _utc_now(),
        "target_id": target_id.strip(),
        "status": "passed",
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": [
                "read_reduce includes reduction arithmetic and framework overhead",
                "no hardware DRAM counter is sampled",
                "cache residency and UMA page placement are not isolated",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "accelerator_name": torch.cuda.get_device_name(0),
            "compute_capability": (
                list(capability) if torch.version.cuda is not None else None
            ),
        },
        "configuration": {
            "requested_buffer_bytes": requested_buffer_bytes,
            "actual_buffer_bytes": actual_buffer_bytes,
            "dtype": "float32",
            "element_count": element_count,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "inner_iterations": inner_iterations,
        },
        "memory": {
            "free_before_allocation_bytes": free_before,
            "total_bytes": total_memory,
            "free_after_allocation_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "operations": operations,
    }
    if telemetry_probe is not None:
        document["telemetry"] = {
            "status": "available",
            "provider": telemetry_probe.provider,
            "collection_mode": "sample_boundaries_outside_timed_region",
            "samples": telemetry_samples,
        }
        document["measurement_scope"]["limitations"].append(
            "management telemetry is sampled only at timed-sample boundaries"
        )
    return document


__all__ = ["_summarize_samples", "benchmark_memory_bandwidth"]
