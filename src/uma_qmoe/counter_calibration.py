"""Normalize rocprofv3 counter CSV into fail-closed calibration evidence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .contracts import ContractError


_TARGET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]+$")
_MAX_PROFILE_BYTES = 16 * 1024 * 1024
_REQUIRED_COLUMNS = {
    "Dispatch_Id",
    "Kernel_Name",
    "Counter_Name",
    "Counter_Value",
}
_CALIBRATION_SPECS = (
    ("read_reduce", "read", "GL2C_EA_RDREQ_DRAM_sum", 128),
    ("write", "write", "GCEA_WDRAM_SIZE_REQ_sum", 32),
    ("copy", "read", "GL2C_EA_RDREQ_DRAM_sum", 128),
    ("copy", "write", "GCEA_WDRAM_SIZE_REQ_sum", 32),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operation_id(kernel_name: str) -> str | None:
    if "read_reduce_kernel" in kernel_name:
        return "read_reduce"
    if "write_kernel" in kernel_name:
        return "write"
    if "copy_kernel" in kernel_name:
        return "copy"
    return None


def _parse_counter_value(value: str) -> int:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ContractError("rocprof counter value is not numeric") from exc
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        raise ContractError("rocprof counter values must be finite non-negative integers")
    return int(parsed)


def calibrate_rocprof_counters(
    profile_csv: str | Path,
    *,
    source_relative_path: str,
    source_file_sha256: str,
    target_id: str,
    architecture: str,
    profiler_version: str,
    known_bytes_per_dispatch: int,
    maximum_relative_error: float = 0.10,
) -> dict[str, Any]:
    """Calibrate selected gfx1151 counters against known native stream bytes."""

    path = Path(profile_csv)
    if not path.is_file():
        raise ContractError(f"rocprof counter CSV does not exist: {path}")
    profile_size = path.stat().st_size
    if profile_size <= 0 or profile_size > _MAX_PROFILE_BYTES:
        raise ContractError("rocprof counter CSV must be within (0, 16 MiB]")
    if not _TARGET_ID.fullmatch(target_id):
        raise ContractError("target_id has unsafe characters")
    if not architecture or len(architecture) > 128:
        raise ContractError("architecture is required and must be at most 128 characters")
    if not profiler_version or len(profiler_version) > 512:
        raise ContractError("profiler_version is required and must be at most 512 characters")
    if known_bytes_per_dispatch < 4 * 1024 * 1024:
        raise ContractError("known_bytes_per_dispatch must be at least 4 MiB")
    if not math.isfinite(maximum_relative_error) or not 0 < maximum_relative_error <= 1:
        raise ContractError("maximum_relative_error must be within (0, 1]")

    rows: dict[str, dict[int, dict[str, int]]] = {
        "read_reduce": {},
        "write": {},
        "copy": {},
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ContractError("rocprof counter CSV is missing required columns")
        for row in reader:
            operation_id = _operation_id(row["Kernel_Name"])
            if operation_id is None:
                raise ContractError("rocprof counter CSV contains an unexpected kernel")
            try:
                dispatch_id = int(row["Dispatch_Id"])
            except ValueError as exc:
                raise ContractError("rocprof Dispatch_Id must be an integer") from exc
            if dispatch_id < 0:
                raise ContractError("rocprof Dispatch_Id must be non-negative")
            counter_name = row["Counter_Name"]
            counters = rows[operation_id].setdefault(dispatch_id, {})
            if counter_name in counters:
                raise ContractError("rocprof CSV repeats a counter for one dispatch")
            counters[counter_name] = _parse_counter_value(row["Counter_Value"])

    if any(len(dispatches) < 3 for dispatches in rows.values()):
        raise ContractError("rocprof CSV needs at least three dispatches per operation")
    first_read_dispatch = min(rows["read_reduce"])
    initialization_dispatches = sorted(
        dispatch_id for dispatch_id in rows["write"] if dispatch_id < first_read_dispatch
    )
    if len(initialization_dispatches) != 1:
        raise ContractError("expected exactly one initialization write before read calibration")
    del rows["write"][initialization_dispatches[0]]
    dispatch_counts = {operation: len(dispatches) for operation, dispatches in rows.items()}
    if len(set(dispatch_counts.values())) != 1:
        raise ContractError("profiled operation dispatch counts do not match")

    calibrations = []
    overall_passed = True
    for operation, direction, counter_name, bytes_per_count in _CALIBRATION_SPECS:
        dispatch_ids = sorted(rows[operation])
        try:
            raw_values = [rows[operation][dispatch][counter_name] for dispatch in dispatch_ids]
        except KeyError as exc:
            raise ContractError(
                f"rocprof CSV is missing {counter_name!r} for {operation!r}"
            ) from exc
        measured_bytes = [value * bytes_per_count for value in raw_values]
        relative_errors = [
            abs(value - known_bytes_per_dispatch) / known_bytes_per_dispatch
            for value in measured_bytes
        ]
        maximum_error = max(relative_errors)
        passed = maximum_error <= maximum_relative_error
        overall_passed = overall_passed and passed
        calibrations.append(
            {
                "id": f"{operation}.{direction}",
                "operation": operation,
                "direction": direction,
                "counter_name": counter_name,
                "bytes_per_count": bytes_per_count,
                "scale_provenance": "empirical_known_byte_stream",
                "dispatch_ids": dispatch_ids,
                "raw_counter_values": raw_values,
                "measured_bytes": measured_bytes,
                "relative_errors": relative_errors,
                "median_relative_error": statistics.median(relative_errors),
                "maximum_relative_error": maximum_error,
                "status": "passed" if passed else "failed",
            }
        )

    return {
        "schema_version": 1,
        "kind": "hardware_counter_calibration",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if overall_passed else "failed",
        "native_source": {
            "path": source_relative_path,
            "sha256": source_file_sha256,
        },
        "profile": {
            "tool": "rocprofv3",
            "version": profiler_version,
            "backend": "hip",
            "architecture": architecture,
            "file_name": path.name,
            "file_bytes": profile_size,
            "file_sha256": _sha256(path),
        },
        "configuration": {
            "known_bytes_per_dispatch": known_bytes_per_dispatch,
            "maximum_relative_error": maximum_relative_error,
            "profiled_dispatches_per_operation": next(iter(dispatch_counts.values())),
        },
        "excluded_dispatches": [
            {
                "dispatch_id": initialization_dispatches[0],
                "operation": "write",
                "reason": "source buffer initialization before timed operations",
            }
        ],
        "calibrations": calibrations,
        "limitations": [
            "counter scale is empirically calibrated for gfx1151 with this pinned profiler stack",
            "calibration covers the native sequential stream access pattern, not arbitrary kernels",
            "counter collection perturbs timing, so unprofiled GPU-event runs remain the bandwidth source",
        ],
    }


__all__ = ["calibrate_rocprof_counters"]
