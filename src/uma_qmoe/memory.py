"""Read-only Linux UMA memory snapshots.

The collector only reads bounded, well-known procfs and cgroup v2 files.  It
does not inspect the process environment, alter kernel settings, or allocate
memory to estimate capacity.  Missing or malformed kernel fields are reported
as unavailable values instead of making snapshot collection fail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_MAX_KERNEL_FILE_CHARS = 256 * 1024
_MEMINFO_FIELDS = {
    "MemTotal": "mem_total_bytes",
    "MemAvailable": "mem_available_bytes",
    "SwapTotal": "swap_total_bytes",
    "SwapFree": "swap_free_bytes",
    "Committed_AS": "committed_as_bytes",
}


def _utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_small_text(path: Path) -> str | None:
    """Read a bounded kernel pseudo-file, returning ``None`` on failure."""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            contents = stream.read(_MAX_KERNEL_FILE_CHARS + 1)
    except OSError:
        return None
    if len(contents) > _MAX_KERNEL_FILE_CHARS:
        return None
    return contents


def _nonnegative_integer(value: str) -> int | None:
    try:
        parsed = int(value, 10)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _kib_to_bytes(value: str) -> int | None:
    """Parse a Linux ``<integer> kB`` value as bytes."""

    fields = value.split()
    if len(fields) != 2 or fields[1].casefold() != "kb":
        return None
    kibibytes = _nonnegative_integer(fields[0])
    return None if kibibytes is None else kibibytes * 1024


def _parse_meminfo(contents: str) -> dict[str, int | None]:
    """Extract the memory fields needed by a snapshot from ``/proc/meminfo``."""

    values = {field: None for field in _MEMINFO_FIELDS.values()}
    remaining = set(_MEMINFO_FIELDS)
    for line in contents.splitlines():
        key, separator, raw_value = line.partition(":")
        key = key.strip()
        if not separator or key not in remaining:
            continue
        parsed = _kib_to_bytes(raw_value)
        if parsed is not None:
            values[_MEMINFO_FIELDS[key]] = parsed
            remaining.remove(key)
        if not remaining:
            break
    return values


def _parse_pgmajfault(contents: str) -> int | None:
    """Parse the cumulative major-page-fault counter from ``/proc/vmstat``."""

    for line in contents.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "pgmajfault":
            return _nonnegative_integer(fields[1])
    return None


def _parse_process_rss(contents: str) -> int | None:
    """Parse current resident bytes from ``/proc/self/status``."""

    for line in contents.splitlines():
        key, separator, raw_value = line.partition(":")
        if separator and key.strip() == "VmRSS":
            return _kib_to_bytes(raw_value)
    return None


def _availability(present: Iterable[bool]) -> str:
    states = tuple(present)
    if states and all(states):
        return "available"
    if any(states):
        return "partial"
    return "unavailable"


def _parse_unified_cgroup_path(contents: str) -> tuple[str, ...] | None:
    """Return a safe relative component tuple for the process' v2 cgroup."""

    for line in contents.splitlines():
        hierarchy, separator, remainder = line.partition(":")
        controllers, second_separator, group_path = remainder.partition(":")
        if not separator or not second_separator:
            continue
        if hierarchy != "0" or controllers:
            continue
        parts = tuple(part for part in group_path.split("/") if part)
        if any(part in {".", ".."} or "\x00" in part for part in parts):
            return None
        return parts
    return None


def _read_cgroup_values(
    proc_root: Path, cgroup_root: Path
) -> tuple[str | None, str | None]:
    cgroup_text = _read_small_text(proc_root / "self" / "cgroup")
    parts = (
        _parse_unified_cgroup_path(cgroup_text)
        if cgroup_text is not None
        else None
    )
    candidates = [cgroup_root.joinpath(*parts)] if parts is not None else []
    if not candidates or candidates[0] != cgroup_root:
        candidates.append(cgroup_root)

    for directory in candidates:
        maximum = _read_small_text(directory / "memory.max")
        current = _read_small_text(directory / "memory.current")
        if maximum is not None or current is not None:
            return maximum, current
    return None, None


def _cgroup_snapshot(proc_root: Path, cgroup_root: Path) -> dict[str, Any]:
    maximum_text, current_text = _read_cgroup_values(proc_root, cgroup_root)

    maximum_bytes: int | None = None
    maximum_is_unlimited: bool | None = None
    maximum_available = False
    if maximum_text is not None:
        maximum_value = maximum_text.strip()
        if maximum_value == "max":
            maximum_available = True
            maximum_is_unlimited = True
        else:
            maximum_bytes = _nonnegative_integer(maximum_value)
            if maximum_bytes is not None:
                maximum_available = True
                maximum_is_unlimited = False

    current_bytes = (
        _nonnegative_integer(current_text.strip())
        if current_text is not None
        else None
    )
    current_available = current_bytes is not None

    return {
        "status": _availability((maximum_available, current_available)),
        "memory_max_bytes": maximum_bytes,
        "memory_max_is_unlimited": maximum_is_unlimited,
        "memory_current_bytes": current_bytes,
    }


def collect_memory_snapshot(
    target_id: str,
    *,
    proc_root: str | Path = Path("/proc"),
    cgroup_root: str | Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Collect a safe, best-effort memory snapshot for ``target_id``.

    ``proc_root`` and ``cgroup_root`` are injectable for deterministic tests.
    Production callers should use their defaults.  Collection performs only
    bounded reads and never probes capacity by allocating memory.

    Raises:
        ValueError: If ``target_id`` is not a non-empty string.
    """

    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be a non-empty string")

    proc_path = Path(proc_root)
    cgroup_path = Path(cgroup_root)

    meminfo_text = _read_small_text(proc_path / "meminfo")
    memory_values = _parse_meminfo(meminfo_text) if meminfo_text is not None else {
        field: None for field in _MEMINFO_FIELDS.values()
    }
    memory = {
        "status": _availability(value is not None for value in memory_values.values()),
        **memory_values,
    }

    vmstat_text = _read_small_text(proc_path / "vmstat")
    major_page_faults = (
        _parse_pgmajfault(vmstat_text) if vmstat_text is not None else None
    )

    status_text = _read_small_text(proc_path / "self" / "status")
    rss_bytes = _parse_process_rss(status_text) if status_text is not None else None

    return {
        "schema_version": 1,
        "kind": "memory_snapshot",
        "captured_at": _utc_now(),
        "target_id": target_id.strip(),
        "memory": memory,
        "page_faults": {
            "status": "available" if major_page_faults is not None else "unavailable",
            "major_total": major_page_faults,
        },
        "cgroup_v2": _cgroup_snapshot(proc_path, cgroup_path),
        "process": {
            "status": "available" if rss_bytes is not None else "unavailable",
            "rss_bytes": rss_bytes,
        },
    }


__all__ = ["collect_memory_snapshot"]
