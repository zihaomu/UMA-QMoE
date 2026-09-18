"""Long-running UMA bandwidth and thermal-stability soak benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable

from .contracts import ContractError
from .telemetry import TelemetryProbe


_MIN_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_BUFFER_BYTES = 8 * 1024 * 1024 * 1024
_MIN_FULL_SOAK_SECONDS = 30 * 60
_MAX_CV = 0.03
_MAX_ABSOLUTE_DRIFT = 0.05
_OPERATION_IDS = ("read_reduce", "write_fill", "copy")
_TELEMETRY_FIELDS = (
    "temperature_c",
    "socket_power_w",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "utilization_percent",
)
_CGROUP_EVENT_FIELDS = ("high", "max", "oom", "oom_kill", "oom_group_kill")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ContractError("cannot summarize an empty sample sequence")
    if not 0 <= quantile <= 1:
        raise ContractError("percentile quantile must be within [0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _summarize_bandwidth(samples_seconds: list[float], traffic_bytes: int) -> dict[str, Any]:
    if len(samples_seconds) < 3:
        raise ContractError("soak requires at least three operation samples")
    if traffic_bytes <= 0:
        raise ContractError("soak traffic_bytes must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in samples_seconds):
        raise ContractError("soak elapsed samples must be finite and positive")
    bandwidth = [traffic_bytes / value / 1_000_000_000 for value in samples_seconds]
    mean = statistics.mean(bandwidth)
    window = max(3, math.ceil(len(bandwidth) * 0.10))
    first_window = statistics.median(bandwidth[:window])
    last_window = statistics.median(bandwidth[-window:])
    return {
        "sample_count": len(bandwidth),
        "minimum_gbps": min(bandwidth),
        "p01_gbps": _percentile(bandwidth, 0.01),
        "p50_gbps": _percentile(bandwidth, 0.50),
        "p95_gbps": _percentile(bandwidth, 0.95),
        "p99_gbps": _percentile(bandwidth, 0.99),
        "maximum_gbps": max(bandwidth),
        "mean_gbps": mean,
        "coefficient_of_variation": statistics.pstdev(bandwidth) / mean,
        "drift_window_samples": window,
        "first_window_p50_gbps": first_window,
        "last_window_p50_gbps": last_window,
        "drift_fraction": (last_window - first_window) / first_window,
    }


def _summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in _TELEMETRY_FIELDS:
        values = [float(sample[field]) for sample in samples if sample[field] is not None]
        summary[field] = {
            "available_samples": len(values),
            "minimum": min(values) if values else None,
            "p50": _percentile(values, 0.50) if values else None,
            "p95": _percentile(values, 0.95) if values else None,
            "maximum": max(values) if values else None,
        }
    return summary


def _swap_pages() -> tuple[int, int]:
    try:
        with open("/proc/vmstat", encoding="utf-8") as stream:
            lines = stream.readlines()
    except OSError as exc:
        raise ContractError("cannot read /proc/vmstat for soak swap gate") from exc
    values: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"pswpin", "pswpout"}:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError as exc:
                raise ContractError("/proc/vmstat contains an invalid swap counter") from exc
    if set(values) != {"pswpin", "pswpout"}:
        raise ContractError("/proc/vmstat is missing swap counters")
    return values["pswpin"], values["pswpout"]


def _cgroup_memory_state(root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    """Read workload-local cgroup v2 swap and OOM state.

    ``/proc/vmstat`` is host-global inside Docker and can move because an unrelated
    process faults an old swapped page back in.  These counters belong to the
    benchmark cgroup and therefore form the fail-closed workload gate.
    """

    swap_current_path = root / "memory.swap.current"
    swap_max_path = root / "memory.swap.max"
    events_path = root / "memory.events"
    try:
        swap_current = int(swap_current_path.read_text(encoding="utf-8").strip())
        raw_swap_max = swap_max_path.read_text(encoding="utf-8").strip()
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        raise ContractError("bandwidth soak requires readable cgroup v2 memory state") from exc
    if swap_current < 0:
        raise ContractError("cgroup memory.swap.current must be non-negative")
    if raw_swap_max == "max":
        swap_max: int | None = None
    else:
        try:
            swap_max = int(raw_swap_max)
        except ValueError as exc:
            raise ContractError("cgroup memory.swap.max is invalid") from exc
        if swap_max < 0:
            raise ContractError("cgroup memory.swap.max must be non-negative")
    events: dict[str, int] = {}
    for line in event_lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in _CGROUP_EVENT_FIELDS:
            try:
                events[fields[0]] = int(fields[1])
            except ValueError as exc:
                raise ContractError("cgroup memory.events contains an invalid value") from exc
    if set(events) != set(_CGROUP_EVENT_FIELDS) or any(value < 0 for value in events.values()):
        raise ContractError("cgroup memory.events is missing required non-negative counters")
    return {
        "swap_current_bytes": swap_current,
        "swap_max_bytes": swap_max,
        "events": events,
    }


def benchmark_bandwidth_soak(
    *,
    target_id: str,
    requested_buffer_bytes: int,
    requested_duration_seconds: int,
    warmup_cycles: int,
    inner_iterations: int,
    telemetry_interval_seconds: float,
) -> dict[str, Any]:
    """Continuously cycle read/write/copy and retain the complete timing series."""

    if not target_id.strip():
        raise ContractError("target_id must be non-empty")
    if not _MIN_BUFFER_BYTES <= requested_buffer_bytes <= _MAX_BUFFER_BYTES:
        raise ContractError(
            f"requested_buffer_bytes must be within [{_MIN_BUFFER_BYTES}, {_MAX_BUFFER_BYTES}]"
        )
    if not 10 <= requested_duration_seconds <= 24 * 60 * 60:
        raise ContractError("requested_duration_seconds must be within [10, 86400]")
    if not 1 <= warmup_cycles <= 100:
        raise ContractError("warmup_cycles must be within [1, 100]")
    if not 1 <= inner_iterations <= 10_000:
        raise ContractError("inner_iterations must be within [1, 10000]")
    if not math.isfinite(telemetry_interval_seconds) or not 1 <= telemetry_interval_seconds <= 300:
        raise ContractError("telemetry_interval_seconds must be within [1, 300]")

    try:
        import torch
    except ImportError as exc:
        raise ContractError("bandwidth soak requires torch") from exc
    if not torch.cuda.is_available():
        raise ContractError("no CUDA/HIP device is available to PyTorch")

    telemetry_probe = TelemetryProbe.detect()
    swap_before = _swap_pages()
    cgroup_before = _cgroup_memory_state()
    element_size = torch.empty((), dtype=torch.float32).element_size()
    element_count = requested_buffer_bytes // element_size
    actual_buffer_bytes = element_count * element_size
    free_before, total_memory = torch.cuda.mem_get_info()
    if actual_buffer_bytes * 2 > int(free_before * 0.5):
        raise ContractError("soak buffers would consume more than 50% of free GPU memory")

    device = torch.device("cuda:0")
    source = torch.ones(element_count, dtype=torch.float32, device=device)
    destination = torch.empty_like(source)
    torch.cuda.synchronize()
    free_after_allocation, _ = torch.cuda.mem_get_info()
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

    operations: tuple[tuple[str, Callable[[], None]], ...] = (
        ("read_reduce", read_reduce),
        ("write_fill", write_fill),
        ("copy", copy),
    )

    def measure(operation: Callable[[], None]) -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(inner_iterations):
            operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ContractError("soak observed a non-finite or non-positive elapsed time")
        return elapsed

    for _ in range(warmup_cycles):
        for _, operation in operations:
            measure(operation)

    generated_at = _utc_now()
    measured_started_ns = time.monotonic_ns()
    telemetry_deadline_ns = measured_started_ns
    samples: list[dict[str, Any]] = []
    telemetry_samples: list[dict[str, Any]] = []
    while True:
        timings = {operation_id: measure(operation) for operation_id, operation in operations}
        now_ns = time.monotonic_ns()
        sample_index = len(samples)
        samples.append(
            {
                "sample_index": sample_index,
                "monotonic_ns": now_ns,
                "elapsed_from_start_seconds": (now_ns - measured_started_ns) / 1_000_000_000,
                "timings_seconds": timings,
            }
        )
        if now_ns >= telemetry_deadline_ns:
            telemetry_samples.append(
                telemetry_probe.capture(
                    operation_id="soak_cycle",
                    sample_index=sample_index,
                    boundary="after",
                )
            )
            telemetry_deadline_ns = now_ns + int(telemetry_interval_seconds * 1_000_000_000)
        if now_ns - measured_started_ns >= requested_duration_seconds * 1_000_000_000:
            break

    completed_at = _utc_now()
    actual_duration_seconds = samples[-1]["elapsed_from_start_seconds"]
    swap_after = _swap_pages()
    cgroup_after = _cgroup_memory_state()
    free_after_soak, _ = torch.cuda.mem_get_info()
    if not read_sink or not bool(torch.isfinite(read_sink[0]).item()):
        raise ContractError("soak read reduction did not produce a finite sink")

    traffic = {
        "read_reduce": (actual_buffer_bytes * inner_iterations, 0),
        "write_fill": (0, actual_buffer_bytes * inner_iterations),
        "copy": (
            actual_buffer_bytes * inner_iterations,
            actual_buffer_bytes * inner_iterations,
        ),
    }
    operation_summaries = []
    for operation_id in _OPERATION_IDS:
        read_bytes, write_bytes = traffic[operation_id]
        operation_summaries.append(
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

    duration_passed = (
        requested_duration_seconds >= _MIN_FULL_SOAK_SECONDS
        and actual_duration_seconds >= requested_duration_seconds
    )
    swap_in_delta = swap_after[0] - swap_before[0]
    swap_out_delta = swap_after[1] - swap_before[1]
    if cgroup_before["swap_max_bytes"] != cgroup_after["swap_max_bytes"]:
        raise ContractError("cgroup memory.swap.max changed during the soak")
    cgroup_swap_delta = (
        cgroup_after["swap_current_bytes"] - cgroup_before["swap_current_bytes"]
    )
    event_deltas = {
        field: cgroup_after["events"][field] - cgroup_before["events"][field]
        for field in _CGROUP_EVENT_FIELDS
    }
    if cgroup_swap_delta < 0 or any(value < 0 for value in event_deltas.values()):
        raise ContractError("cgroup memory counters decreased during the soak")
    swap_disabled_passed = cgroup_after["swap_max_bytes"] == 0
    swap_passed = (
        cgroup_before["swap_current_bytes"] == 0
        and cgroup_after["swap_current_bytes"] == 0
        and cgroup_swap_delta == 0
    )
    no_oom_events_passed = all(
        event_deltas[field] == 0 for field in ("oom", "oom_kill", "oom_group_kill")
    )
    cv_passed = all(
        operation["coefficient_of_variation"] <= _MAX_CV
        for operation in operation_summaries
    )
    drift_passed = all(
        abs(operation["drift_fraction"]) <= _MAX_ABSOLUTE_DRIFT
        for operation in operation_summaries
    )
    overall_passed = (
        duration_passed
        and swap_disabled_passed
        and swap_passed
        and no_oom_events_passed
        and cv_passed
        and drift_passed
    )
    status = (
        "diagnostic"
        if requested_duration_seconds < _MIN_FULL_SOAK_SECONDS
        else ("passed" if overall_passed else "failed")
    )
    capability = torch.cuda.get_device_capability(0)
    return {
        "schema_version": 1,
        "kind": "bandwidth_soak",
        "generated_at": generated_at,
        "completed_at": completed_at,
        "target_id": target_id.strip(),
        "status": status,
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": [
                "read_reduce includes reduction arithmetic and framework overhead",
                "management telemetry is sampled outside timed regions at a lower cadence",
                "hardware DRAM counters are not continuously sampled during the soak",
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
            "requested_duration_seconds": requested_duration_seconds,
            "minimum_full_soak_seconds": _MIN_FULL_SOAK_SECONDS,
            "actual_duration_seconds": actual_duration_seconds,
            "warmup_cycles": warmup_cycles,
            "inner_iterations": inner_iterations,
            "telemetry_interval_seconds": telemetry_interval_seconds,
            "maximum_coefficient_of_variation": _MAX_CV,
            "maximum_absolute_drift_fraction": _MAX_ABSOLUTE_DRIFT,
        },
        "memory": {
            "free_before_allocation_bytes": free_before,
            "total_bytes": total_memory,
            "free_after_allocation_bytes": free_after_allocation,
            "free_after_soak_bytes": free_after_soak,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "swap_in_pages_before": swap_before[0],
            "swap_in_pages_after": swap_after[0],
            "swap_in_pages_delta": swap_in_delta,
            "swap_out_pages_before": swap_before[1],
            "swap_out_pages_after": swap_after[1],
            "swap_out_pages_delta": swap_out_delta,
            "workload_cgroup": {
                "gate_scope": "workload_cgroup_v2",
                "swap_current_bytes_before": cgroup_before["swap_current_bytes"],
                "swap_current_bytes_after": cgroup_after["swap_current_bytes"],
                "swap_current_bytes_delta": cgroup_swap_delta,
                "swap_max_bytes": cgroup_after["swap_max_bytes"],
                "memory_events_before": cgroup_before["events"],
                "memory_events_after": cgroup_after["events"],
                "memory_events_delta": event_deltas,
            },
        },
        "samples": samples,
        "operations": operation_summaries,
        "telemetry": {
            "status": "available",
            "provider": telemetry_probe.provider,
            "collection_mode": "periodic_after_cycle_outside_timed_regions",
            "samples": telemetry_samples,
            "summary": _summarize_telemetry(telemetry_samples),
        },
        "gates": {
            "minimum_duration_passed": duration_passed,
            "workload_cgroup_swap_disabled_passed": swap_disabled_passed,
            "no_swap_activity_passed": swap_passed,
            "no_oom_events_passed": no_oom_events_passed,
            "coefficient_of_variation_passed": cv_passed,
            "drift_passed": drift_passed,
            "overall_passed": overall_passed,
        },
    }


__all__ = [
    "_percentile",
    "_summarize_bandwidth",
    "_summarize_telemetry",
    "benchmark_bandwidth_soak",
]
