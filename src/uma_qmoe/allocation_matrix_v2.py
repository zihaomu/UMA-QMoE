"""Target-native Allocation Matrix v2 for coherent UMA systems.

The v2 runner compiles one CUDA/HIP source and executes it at monotonically
increasing pressure points.  A capability bit may suppress an unsafe attempt,
but it is never promoted to a performance result: every allocation path is
reported as either ``measured`` with raw samples or ``unavailable`` with a
stable reason code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Literal, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from .contracts import ContractError
from .memory_bandwidth import _summarize_samples
from .native_stream import _native_compile_command, _run_checked


_MIN_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_BUFFER_BYTES = 64 * 1024**3
_MAX_OUTPUT_BYTES = 8_000_000
_TARGET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]+$")
_ARCHITECTURE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_REASON = re.compile(r"^[a-z0-9][a-z0-9_]{2,127}$")
_CASE_IDS = ("managed_unified", "system_pageable_direct", "platform_vmm")
_OPERATION_IDS = ("read", "write", "copy")
_TOUCH_IDS = ("cpu_first", "gpu_first")
_CGROUP_EVENT_FIELDS = ("high", "max", "oom", "oom_kill", "oom_group_kill")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ContractError(f"{context} returned an invalid field set")


def _positive_number(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ContractError(f"{context} must be finite and positive")
    return float(value)


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{context} must be a non-negative integer")
    return value


def _reason(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _REASON.fullmatch(value):
        raise ContractError(f"{context} must be a stable lower_snake_case reason")
    return value


def _samples(value: Any, measured_iterations: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != measured_iterations:
        raise ContractError(f"{context} must contain every raw measured sample")
    return [_positive_number(item, context) for item in value]


def _normalize_following_access(raw: Any, *, expected_actor: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("native following_access must be an object")
    status = raw.get("status")
    if status == "measured":
        _exact_keys(raw, {"actor", "status", "seconds"}, "native following_access")
        if raw["actor"] != expected_actor:
            raise ContractError("native following_access actor mismatch")
        return {
            "actor": expected_actor,
            "status": "measured",
            "seconds": _positive_number(
                raw["seconds"], "native following_access seconds"
            ),
        }
    if status == "unavailable":
        _exact_keys(raw, {"actor", "status", "reason"}, "native following_access")
        if raw["actor"] != expected_actor:
            raise ContractError("native following_access actor mismatch")
        return {
            "actor": expected_actor,
            "status": "unavailable",
            "reason": _reason(raw["reason"], "native following_access reason"),
        }
    raise ContractError("native following_access has an invalid status")


def _normalize_touch(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("id") not in _TOUCH_IDS:
        raise ContractError("native touch has an invalid identity")
    touch_id = raw["id"]
    expected_actor = "gpu" if touch_id == "cpu_first" else "cpu"
    status = raw.get("status")
    if status == "measured":
        _exact_keys(
            raw,
            {
                "id",
                "status",
                "seconds",
                "minor_faults_delta",
                "major_faults_delta",
                "following_access",
            },
            "native measured touch",
        )
        return {
            "id": touch_id,
            "status": "measured",
            "seconds": _positive_number(raw["seconds"], "native touch seconds"),
            "host_minor_faults_delta": _nonnegative_integer(
                raw["minor_faults_delta"], "native touch minor faults"
            ),
            "host_major_faults_delta": _nonnegative_integer(
                raw["major_faults_delta"], "native touch major faults"
            ),
            "following_access": _normalize_following_access(
                raw["following_access"], expected_actor=expected_actor
            ),
        }
    if status == "unavailable":
        _exact_keys(
            raw,
            {"id", "status", "reason", "following_access"},
            "native unavailable touch",
        )
        return {
            "id": touch_id,
            "status": "unavailable",
            "reason": _reason(raw["reason"], "native touch reason"),
            "following_access": _normalize_following_access(
                raw["following_access"], expected_actor=expected_actor
            ),
        }
    raise ContractError("native touch has an invalid status")


def _normalize_contention(
    raw: Any, *, buffer_bytes: int, measured_iterations: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("native contention must be an object")
    if raw.get("status") == "unavailable":
        _exact_keys(raw, {"status", "reason"}, "native unavailable contention")
        return {
            "status": "unavailable",
            "reason": _reason(raw["reason"], "native contention reason"),
        }
    if raw.get("status") != "measured":
        raise ContractError("native contention has an invalid status")
    _exact_keys(
        raw,
        {"status", "wall_samples_seconds", "gpu_samples_seconds"},
        "native measured contention",
    )
    wall = _samples(
        raw["wall_samples_seconds"], measured_iterations, "contention wall samples"
    )
    gpu = _samples(
        raw["gpu_samples_seconds"], measured_iterations, "contention GPU samples"
    )
    return {
        "status": "measured",
        "wall": _summarize_samples(wall, buffer_bytes),
        "gpu_half_buffer": _summarize_samples(gpu, buffer_bytes // 2),
    }


def _normalize_case(
    raw: Any, *, measured_iterations: int, inner_iterations: int
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("id") not in _CASE_IDS:
        raise ContractError("native allocation case has an invalid identity")
    common = {"id", "status", "allocation_api", "access_path"}
    if not isinstance(raw.get("allocation_api"), str) or not raw["allocation_api"]:
        raise ContractError("native allocation_api must be non-empty")
    if not isinstance(raw.get("access_path"), str) or not raw["access_path"]:
        raise ContractError("native access_path must be non-empty")
    result: dict[str, Any] = {
        "id": raw["id"],
        "status": raw.get("status"),
        "allocation_api": raw["allocation_api"],
        "access_path": raw["access_path"],
    }
    if raw.get("status") == "unavailable":
        _exact_keys(raw, common | {"reason"}, "native unavailable allocation case")
        result["reason"] = _reason(raw["reason"], "native allocation reason")
        return result
    if raw.get("status") != "measured":
        raise ContractError("native allocation case has an invalid status")
    _exact_keys(
        raw,
        common
        | {
            "buffer_bytes",
            "allocation_seconds",
            "touches",
            "operations",
            "contention",
        },
        "native measured allocation case",
    )
    buffer_bytes = _nonnegative_integer(raw["buffer_bytes"], "native case buffer")
    if buffer_bytes < _MIN_BUFFER_BYTES or buffer_bytes % 4:
        raise ContractError("native case buffer is invalid")
    touches = raw["touches"]
    if not isinstance(touches, list):
        raise ContractError("native case touches must be an array")
    normalized_touches = [_normalize_touch(touch) for touch in touches]
    touch_ids = [touch["id"] for touch in normalized_touches]
    if len(touch_ids) != len(_TOUCH_IDS) or set(touch_ids) != set(_TOUCH_IDS):
        raise ContractError("native case must contain each first-touch path exactly once")
    operations = raw["operations"]
    if not isinstance(operations, list):
        raise ContractError("native case operations must be an array")
    raw_by_id: dict[str, list[float]] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise ContractError("native operation must be an object")
        _exact_keys(operation, {"id", "samples_seconds"}, "native operation")
        operation_id = operation.get("id")
        if operation_id not in _OPERATION_IDS or operation_id in raw_by_id:
            raise ContractError("native operation identity is duplicate or unknown")
        raw_by_id[operation_id] = _samples(
            operation["samples_seconds"],
            measured_iterations,
            f"native {operation_id} samples",
        )
    if set(raw_by_id) != set(_OPERATION_IDS):
        raise ContractError("native case must contain read, write, and copy")
    traffic = {
        "read": (buffer_bytes * inner_iterations, 0),
        "write": (0, buffer_bytes * inner_iterations),
        "copy": (
            buffer_bytes * inner_iterations,
            buffer_bytes * inner_iterations,
        ),
    }
    normalized_operations: list[dict[str, Any]] = []
    for operation_id in _OPERATION_IDS:
        read_bytes, write_bytes = traffic[operation_id]
        normalized_operations.append(
            {
                "id": operation_id,
                "algorithmic_read_bytes": read_bytes,
                "algorithmic_write_bytes": write_bytes,
                "algorithmic_total_bytes": read_bytes + write_bytes,
                **_summarize_samples(
                    raw_by_id[operation_id], read_bytes + write_bytes
                ),
            }
        )
    result.update(
        {
            "buffer_bytes": buffer_bytes,
            "allocation_seconds": _positive_number(
                raw["allocation_seconds"], "native allocation seconds"
            ),
            "touches": normalized_touches,
            "operations": normalized_operations,
            "contention": _normalize_contention(
                raw["contention"],
                buffer_bytes=buffer_bytes,
                measured_iterations=measured_iterations,
            ),
        }
    )
    return result


def _normalize_native_output(
    raw: Any,
    *,
    backend: Literal["cuda", "hip"],
    requested_buffer_bytes: int,
    measured_iterations: int,
    inner_iterations: int,
) -> dict[str, Any]:
    """Strictly normalize one native pressure point without trusting claims."""

    if not isinstance(raw, dict):
        raise ContractError("native Allocation Matrix output must be an object")
    _exact_keys(
        raw,
        {
            "backend",
            "device_name",
            "total_global_memory_bytes",
            "multiprocessor_count",
            "threads_per_block",
            "blocks",
            "requested_buffer_bytes",
            "capabilities",
            "cases",
        },
        "native Allocation Matrix output",
    )
    if raw["backend"] != backend:
        raise ContractError("native Allocation Matrix backend identity mismatch")
    if raw["requested_buffer_bytes"] != requested_buffer_bytes:
        raise ContractError("native Allocation Matrix requested buffer mismatch")
    if not isinstance(raw["device_name"], str) or not raw["device_name"].strip():
        raise ContractError("native Allocation Matrix omitted device_name")
    capabilities = raw["capabilities"]
    if not isinstance(capabilities, dict):
        raise ContractError("native capabilities must be an object")
    capability_names = {
        "managed_memory",
        "concurrent_managed_access",
        "pageable_memory_access",
        "pageable_memory_access_uses_host_page_tables",
    }
    _exact_keys(capabilities, capability_names, "native capabilities")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}
        for value in capabilities.values()
    ):
        raise ContractError("native capability flags must be integer booleans")
    cases = raw["cases"]
    if not isinstance(cases, list):
        raise ContractError("native cases must be an array")
    normalized_cases = [
        _normalize_case(
            case,
            measured_iterations=measured_iterations,
            inner_iterations=inner_iterations,
        )
        for case in cases
    ]
    case_ids = [case["id"] for case in normalized_cases]
    if len(case_ids) != len(_CASE_IDS) or set(case_ids) != set(_CASE_IDS):
        raise ContractError("native output must contain each allocation path exactly once")
    runtime_integer_fields = (
        "total_global_memory_bytes",
        "multiprocessor_count",
        "threads_per_block",
        "blocks",
    )
    runtime_values = {
        name: _nonnegative_integer(raw[name], f"native {name}")
        for name in runtime_integer_fields
    }
    if any(value <= 0 for value in runtime_values.values()):
        raise ContractError("native runtime identity values must be positive")
    return {
        "requested_buffer_bytes": requested_buffer_bytes,
        "runtime": {
            "backend": backend,
            "device_name": raw["device_name"].strip(),
            **runtime_values,
            "capabilities": dict(capabilities),
        },
        "cases": normalized_cases,
    }


def _read_vmstat(path: Path = Path("/proc/vmstat")) -> dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError("Allocation Matrix v2 requires readable /proc/vmstat") from exc
    values: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"pswpin", "pswpout"}:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError as exc:
                raise ContractError("/proc/vmstat contains an invalid swap counter") from exc
    if set(values) != {"pswpin", "pswpout"} or any(value < 0 for value in values.values()):
        raise ContractError("/proc/vmstat is missing non-negative swap counters")
    return {"swap_in_pages": values["pswpin"], "swap_out_pages": values["pswpout"]}


def _current_cgroup_root(
    membership_path: Path = Path("/proc/self/cgroup"),
    mount_path: Path = Path("/sys/fs/cgroup"),
) -> Path:
    try:
        lines = membership_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError("Allocation Matrix v2 requires cgroup v2 membership") from exc
    unified = [line[3:] for line in lines if line.startswith("0::/")]
    if len(unified) != 1:
        raise ContractError("Allocation Matrix v2 requires one cgroup v2 membership")
    mount = mount_path.resolve()
    candidate = (mount / unified[0].lstrip("/")).resolve()
    if candidate != mount and mount not in candidate.parents:
        raise ContractError("current cgroup path escapes the cgroup v2 mount")
    return candidate


def _read_cgroup_state(root: Path) -> dict[str, Any]:
    try:
        swap_current = int((root / "memory.swap.current").read_text().strip())
        raw_swap_max = (root / "memory.swap.max").read_text().strip()
        event_lines = (root / "memory.events").read_text().splitlines()
    except (OSError, ValueError) as exc:
        raise ContractError("Allocation Matrix v2 requires readable cgroup v2 memory state") from exc
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
        raise ContractError("cgroup memory.events is missing required counters")
    return {
        "swap_current_bytes": swap_current,
        "swap_max_bytes": swap_max,
        "events": events,
    }


def _system_activity(
    vmstat_before: Mapping[str, int],
    vmstat_after: Mapping[str, int],
    cgroup_before: Mapping[str, Any],
    cgroup_after: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    host_deltas = {
        name: vmstat_after[name] - vmstat_before[name]
        for name in ("swap_in_pages", "swap_out_pages")
    }
    if any(value < 0 for value in host_deltas.values()):
        raise ContractError("host swap counters decreased during Allocation Matrix v2")
    if cgroup_before["swap_max_bytes"] != cgroup_after["swap_max_bytes"]:
        raise ContractError("cgroup memory.swap.max changed during Allocation Matrix v2")
    swap_delta = (
        cgroup_after["swap_current_bytes"] - cgroup_before["swap_current_bytes"]
    )
    event_deltas = {
        field: cgroup_after["events"][field] - cgroup_before["events"][field]
        for field in _CGROUP_EVENT_FIELDS
    }
    if swap_delta < 0 or any(value < 0 for value in event_deltas.values()):
        raise ContractError("workload cgroup counters decreased during Allocation Matrix v2")
    activity = {
        "host_vmstat": {
            "swap_in_pages_before": vmstat_before["swap_in_pages"],
            "swap_in_pages_after": vmstat_after["swap_in_pages"],
            "swap_in_pages_delta": host_deltas["swap_in_pages"],
            "swap_out_pages_before": vmstat_before["swap_out_pages"],
            "swap_out_pages_after": vmstat_after["swap_out_pages"],
            "swap_out_pages_delta": host_deltas["swap_out_pages"],
        },
        "workload_cgroup": {
            "swap_current_bytes_before": cgroup_before["swap_current_bytes"],
            "swap_current_bytes_after": cgroup_after["swap_current_bytes"],
            "swap_current_bytes_delta": swap_delta,
            "swap_max_bytes": cgroup_after["swap_max_bytes"],
            "memory_events_before": dict(cgroup_before["events"]),
            "memory_events_after": dict(cgroup_after["events"]),
            "memory_events_delta": event_deltas,
        },
    }
    gates = {
        "workload_cgroup_swap_disabled_passed": cgroup_after["swap_max_bytes"] == 0,
        "no_swap_activity_passed": (
            cgroup_before["swap_current_bytes"] == 0
            and cgroup_after["swap_current_bytes"] == 0
            and swap_delta == 0
        ),
        "no_oom_events_passed": all(
            event_deltas[field] == 0
            for field in ("oom", "oom_kill", "oom_group_kill")
        ),
    }
    return activity, gates


def _coverage_and_stability(
    pressure_points: Sequence[Mapping[str, Any]], maximum_cv: float
) -> tuple[bool, bool]:
    coverage = all(
        any(case["status"] == "measured" for case in point["cases"])
        for point in pressure_points
    )
    measured_operations = [
        operation
        for point in pressure_points
        for case in point["cases"]
        if case["status"] == "measured"
        for operation in case["operations"]
    ]
    stability = bool(measured_operations) and all(
        operation["coefficient_of_variation"] <= maximum_cv
        for operation in measured_operations
    )
    return coverage, stability


def _allocation_compile_command(
    *,
    compiler: str,
    backend: Literal["cuda", "hip"],
    architecture: str,
    source: Path,
    executable: Path,
) -> list[str]:
    command = _native_compile_command(
        compiler=compiler,
        backend=backend,
        architecture=architecture,
        source=source,
        executable=executable,
    )
    thread_flag = "-Xcompiler=-pthread" if backend == "cuda" else "-pthread"
    command.insert(-2, thread_flag)
    if backend == "cuda":
        command.append("-lcuda")
    return command


def benchmark_allocation_matrix_v2(
    source_path: str | Path,
    *,
    source_relative_path: str,
    source_file_sha256: str,
    target_id: str,
    backend: Literal["cuda", "hip"],
    architecture: str,
    buffer_sizes_bytes: Sequence[int],
    warmup_iterations: int,
    measured_iterations: int,
    inner_iterations: int = 1,
    maximum_coefficient_of_variation: float = 0.03,
    native_timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Compile and execute Allocation Matrix v2 at progressive pressure points."""

    source = Path(source_path)
    if not source.is_file():
        raise ContractError(f"Allocation Matrix v2 source does not exist: {source}")
    if not _TARGET_ID.fullmatch(target_id):
        raise ContractError("target_id has unsafe characters")
    if not _ARCHITECTURE.fullmatch(architecture):
        raise ContractError("architecture has unsafe characters")
    if not re.fullmatch(r"[0-9a-f]{64}", source_file_sha256):
        raise ContractError("source_file_sha256 must be lowercase SHA-256")
    try:
        observed_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f"cannot read Allocation Matrix v2 source: {source}") from exc
    if observed_source_sha256 != source_file_sha256:
        raise ContractError("source_file_sha256 does not match Allocation Matrix v2 source")
    if (
        not source_relative_path
        or source_relative_path.startswith("/")
        or "\\" in source_relative_path
        or any(part in {"", ".", ".."} for part in source_relative_path.split("/"))
    ):
        raise ContractError("source_relative_path must be project-relative POSIX")
    sizes = list(buffer_sizes_bytes)
    if not sizes or len(sizes) > 32:
        raise ContractError("buffer_sizes_bytes must contain between 1 and 32 points")
    if any(
        isinstance(size, bool)
        or not isinstance(size, int)
        or not _MIN_BUFFER_BYTES <= size <= _MAX_BUFFER_BYTES
        or size % 4
        for size in sizes
    ):
        raise ContractError("buffer sizes must be aligned integers within [4 MiB, 64 GiB]")
    if sizes != sorted(set(sizes)):
        raise ContractError("buffer_sizes_bytes must be strictly increasing and unique")
    if not 1 <= warmup_iterations <= 100:
        raise ContractError("warmup_iterations must be within [1, 100]")
    if not 3 <= measured_iterations <= 100:
        raise ContractError("measured_iterations must be within [3, 100]")
    if not 1 <= inner_iterations <= 10_000:
        raise ContractError("inner_iterations must be within [1, 10000]")
    if (
        isinstance(maximum_coefficient_of_variation, bool)
        or not isinstance(maximum_coefficient_of_variation, (int, float))
        or not math.isfinite(float(maximum_coefficient_of_variation))
        or not 0 < maximum_coefficient_of_variation <= 1
    ):
        raise ContractError("maximum_coefficient_of_variation must be within (0, 1]")
    if (
        isinstance(native_timeout_seconds, bool)
        or not isinstance(native_timeout_seconds, (int, float))
        or not math.isfinite(float(native_timeout_seconds))
        or not 10 <= native_timeout_seconds <= 7200
    ):
        raise ContractError("native_timeout_seconds must be within [10, 7200]")

    compiler_name = "nvcc" if backend == "cuda" else "hipcc"
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise ContractError(f"required native compiler {compiler_name!r} is unavailable")
    compiler_version = _run_checked([compiler, "--version"], timeout=15.0)
    version_line = next(
        (line.strip() for line in compiler_version.stdout.splitlines() if line.strip()),
        compiler_name,
    )[:512]

    with tempfile.TemporaryDirectory(prefix="uma-qmoe-allocation-v2-") as directory:
        executable = Path(directory) / "allocation-matrix-v2"
        compile_command = _allocation_compile_command(
            compiler=compiler,
            backend=backend,
            architecture=architecture,
            source=source,
            executable=executable,
        )
        _run_checked(compile_command, timeout=300.0)

        cgroup_root = _current_cgroup_root()
        vmstat_before = _read_vmstat()
        cgroup_before = _read_cgroup_state(cgroup_root)
        pressure_points: list[dict[str, Any]] = []
        runtime: dict[str, Any] | None = None
        for size in sizes:
            try:
                completed = subprocess.run(
                    [
                        str(executable),
                        target_id,
                        str(size),
                        str(warmup_iterations),
                        str(measured_iterations),
                        str(inner_iterations),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(native_timeout_seconds),
                    shell=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ContractError("Allocation Matrix v2 native process failed") from exc
            if completed.returncode != 0:
                raise ContractError(
                    f"Allocation Matrix v2 native process returned exit {completed.returncode}"
                )
            if len(completed.stdout.encode("utf-8")) > _MAX_OUTPUT_BYTES:
                raise ContractError("Allocation Matrix v2 output exceeds the evidence limit")
            try:
                raw = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise ContractError("Allocation Matrix v2 emitted malformed JSON") from exc
            point = _normalize_native_output(
                raw,
                backend=backend,
                requested_buffer_bytes=size,
                measured_iterations=measured_iterations,
                inner_iterations=inner_iterations,
            )
            observed_runtime = point.pop("runtime")
            if runtime is None:
                runtime = observed_runtime
            elif observed_runtime != runtime:
                raise ContractError("native runtime identity changed between pressure points")
            point["measured_case_count"] = sum(
                case["status"] == "measured" for case in point["cases"]
            )
            point["unavailable_case_count"] = sum(
                case["status"] == "unavailable" for case in point["cases"]
            )
            pressure_points.append(point)
        vmstat_after = _read_vmstat()
        cgroup_after = _read_cgroup_state(cgroup_root)

    assert runtime is not None
    system_activity, memory_gates = _system_activity(
        vmstat_before, vmstat_after, cgroup_before, cgroup_after
    )
    coverage_passed, stability_passed = _coverage_and_stability(
        pressure_points, float(maximum_coefficient_of_variation)
    )
    gates = {
        "minimum_coverage_passed": coverage_passed,
        "steady_state_stability_passed": stability_passed,
        **memory_gates,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 2,
        "kind": "allocation_matrix_v2",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "source": {"path": source_relative_path, "sha256": source_file_sha256},
        "build": {
            "backend": backend,
            "compiler": compiler_name,
            "compiler_version": version_line,
            "architecture": architecture,
            "optimization": "O3",
        },
        "runtime": runtime,
        "configuration": {
            "buffer_sizes_bytes": sizes,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "inner_iterations": inner_iterations,
            "maximum_coefficient_of_variation": float(
                maximum_coefficient_of_variation
            ),
        },
        "measurement_scope": {
            "elapsed_time": "gpu_events_and_host_monotonic",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "page_fault_scope": "calling_process_host_faults",
            "limitations": [
                "capability flags only suppress unsafe attempts and are not performance evidence",
                "process page faults do not expose every device-side migration event",
                "host vmstat is diagnostic; swap and OOM gates use the workload cgroup",
                "algorithmic bytes are not calibrated hardware DRAM traffic",
            ],
        },
        "pressure_points": pressure_points,
        "system_activity": system_activity,
        "gates": gates,
    }
    validate_allocation_matrix_v2_document(document)
    return document


def validate_allocation_matrix_v2_document(document: Mapping[str, Any]) -> None:
    """Validate the standalone v2 schema and recompute its safety gates."""

    schema = json.loads(
        resources.files("uma_qmoe.schemas")
        .joinpath("allocation_matrix_v2.schema.json")
        .read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(
            f"$.{'.'.join(str(part) for part in error.path)}: {error.message}"
            for error in errors
        )
        raise ContractError(detail)

    source_path = document["source"]["path"]
    if (
        source_path.startswith("/")
        or "\\" in source_path
        or any(part in {"", ".", ".."} for part in source_path.split("/"))
    ):
        raise ContractError("$.source.path must be a safe project-relative POSIX path")
    build = document["build"]
    expected_compiler = "nvcc" if build["backend"] == "cuda" else "hipcc"
    if build["compiler"] != expected_compiler:
        raise ContractError("Allocation Matrix v2 compiler/backend mismatch")
    if document["runtime"]["backend"] != build["backend"]:
        raise ContractError("Allocation Matrix v2 runtime/build backend mismatch")
    sizes = document["configuration"]["buffer_sizes_bytes"]
    if sizes != sorted(set(sizes)):
        raise ContractError("Allocation Matrix v2 buffer sizes must be strictly increasing")
    points = document["pressure_points"]
    if [point["requested_buffer_bytes"] for point in points] != sizes:
        raise ContractError("pressure points do not match configured buffer sizes")
    capabilities = document["runtime"]["capabilities"]
    for point in points:
        cases = point["cases"]
        case_ids = [case["id"] for case in cases]
        if len(case_ids) != len(_CASE_IDS) or set(case_ids) != set(_CASE_IDS):
            raise ContractError("pressure point must contain each allocation path once")
        measured = sum(case["status"] == "measured" for case in cases)
        if point["measured_case_count"] != measured:
            raise ContractError("pressure point measured_case_count is invalid")
        if point["unavailable_case_count"] != len(_CASE_IDS) - measured:
            raise ContractError("pressure point unavailable_case_count is invalid")
        cases_by_id = {case["id"]: case for case in cases}
        if (
            capabilities["managed_memory"] == 0
            and cases_by_id["managed_unified"]["status"] == "measured"
        ):
            raise ContractError("managed path contradicts its capability gate")
        if (
            (
                capabilities["pageable_memory_access"] == 0
                or capabilities["pageable_memory_access_uses_host_page_tables"] == 0
            )
            and cases_by_id["system_pageable_direct"]["status"] == "measured"
        ):
            raise ContractError("system-pageable path contradicts its capability gate")
        for case in cases:
            if case["status"] != "measured":
                continue
            inner = document["configuration"]["inner_iterations"]
            measured_iterations = document["configuration"]["measured_iterations"]
            expected_traffic = {
                "read": (case["buffer_bytes"] * inner, 0),
                "write": (0, case["buffer_bytes"] * inner),
                "copy": (case["buffer_bytes"] * inner, case["buffer_bytes"] * inner),
            }
            operations = case["operations"]
            if {operation["id"] for operation in operations} != set(_OPERATION_IDS):
                raise ContractError("measured case operations are incomplete")
            touch_ids = [touch["id"] for touch in case["touches"]]
            if len(touch_ids) != len(_TOUCH_IDS) or set(touch_ids) != set(_TOUCH_IDS):
                raise ContractError("measured case first-touch paths are incomplete")
            following_actor = {
                touch["id"]: touch["following_access"]["actor"]
                for touch in case["touches"]
            }
            if following_actor != {"cpu_first": "gpu", "gpu_first": "cpu"}:
                raise ContractError("measured case first-touch actor chain is invalid")
            for operation in operations:
                read_bytes, write_bytes = expected_traffic[operation["id"]]
                if (
                    operation["algorithmic_read_bytes"] != read_bytes
                    or operation["algorithmic_write_bytes"] != write_bytes
                    or operation["algorithmic_total_bytes"] != read_bytes + write_bytes
                ):
                    raise ContractError("measured case has invalid algorithmic traffic")
                if len(operation["samples_seconds"]) != measured_iterations:
                    raise ContractError("measured case omitted raw operation samples")
                recomputed = _summarize_samples(
                    list(operation["samples_seconds"]), read_bytes + write_bytes
                )
                for field in (
                    "median_seconds",
                    "mean_seconds",
                    "coefficient_of_variation",
                    "effective_gbps",
                ):
                    if not math.isclose(
                        operation[field],
                        recomputed[field],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        raise ContractError(
                            f"measured case operation {field} is inconsistent with raw samples"
                        )
            contention = case["contention"]
            if contention["status"] == "measured":
                expected_contention = (
                    ("wall", case["buffer_bytes"]),
                    ("gpu_half_buffer", case["buffer_bytes"] // 2),
                )
                for summary_name, traffic_bytes in expected_contention:
                    summary = contention[summary_name]
                    if len(summary["samples_seconds"]) != measured_iterations:
                        raise ContractError("contention omitted raw measured samples")
                    recomputed = _summarize_samples(
                        list(summary["samples_seconds"]), traffic_bytes
                    )
                    for field in (
                        "median_seconds",
                        "mean_seconds",
                        "coefficient_of_variation",
                        "effective_gbps",
                    ):
                        if not math.isclose(
                            summary[field],
                            recomputed[field],
                            rel_tol=1e-12,
                            abs_tol=1e-15,
                        ):
                            raise ContractError(
                                f"contention {field} is inconsistent with raw samples"
                            )

    activity = document["system_activity"]
    host = activity["host_vmstat"]
    if (
        host["swap_in_pages_delta"]
        != host["swap_in_pages_after"] - host["swap_in_pages_before"]
        or host["swap_out_pages_delta"]
        != host["swap_out_pages_after"] - host["swap_out_pages_before"]
    ):
        raise ContractError("host swap deltas are invalid")
    cgroup = activity["workload_cgroup"]
    if (
        cgroup["swap_current_bytes_delta"]
        != cgroup["swap_current_bytes_after"]
        - cgroup["swap_current_bytes_before"]
    ):
        raise ContractError("workload cgroup swap delta is invalid")
    expected_event_deltas = {
        field: cgroup["memory_events_after"][field]
        - cgroup["memory_events_before"][field]
        for field in _CGROUP_EVENT_FIELDS
    }
    if cgroup["memory_events_delta"] != expected_event_deltas:
        raise ContractError("workload cgroup event deltas are invalid")
    coverage, stability = _coverage_and_stability(
        points,
        document["configuration"]["maximum_coefficient_of_variation"],
    )
    expected_gates = {
        "minimum_coverage_passed": coverage,
        "steady_state_stability_passed": stability,
        "workload_cgroup_swap_disabled_passed": cgroup["swap_max_bytes"] == 0,
        "no_swap_activity_passed": (
            cgroup["swap_current_bytes_before"] == 0
            and cgroup["swap_current_bytes_after"] == 0
            and cgroup["swap_current_bytes_delta"] == 0
        ),
        "no_oom_events_passed": all(
            expected_event_deltas[field] == 0
            for field in ("oom", "oom_kill", "oom_group_kill")
        ),
    }
    expected_gates["overall_passed"] = all(expected_gates.values())
    if document["gates"] != expected_gates:
        raise ContractError("Allocation Matrix v2 gates do not match recomputed evidence")
    expected_status = "passed" if expected_gates["overall_passed"] else "failed"
    if document["status"] != expected_status:
        raise ContractError("Allocation Matrix v2 status does not match its gates")


__all__ = [
    "_allocation_compile_command",
    "_normalize_native_output",
    "_read_cgroup_state",
    "_read_vmstat",
    "_system_activity",
    "benchmark_allocation_matrix_v2",
    "validate_allocation_matrix_v2_document",
]
