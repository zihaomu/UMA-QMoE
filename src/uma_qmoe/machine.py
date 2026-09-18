"""Safe, best-effort collection of a host machine baseline.

The collector deliberately uses a small command allowlist and never records the
process environment.  Probe failures are represented in the returned document
instead of preventing callers from recording the rest of the machine state.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
from typing import Any, Sequence


_PROBE_TIMEOUT_SECONDS = 10.0
_MAX_ACCELERATORS = 64


def _utc_now() -> str:
    """Return an RFC 3339 timestamp with an explicit UTC designator."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _proc_value(path: Path, keys: Sequence[str]) -> str | None:
    """Read the first matching ``key: value`` entry from a procfs file."""

    try:
        contents = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    wanted = {key.casefold() for key in keys}
    values: dict[str, str] = {}
    for line in contents.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in wanted:
            cleaned = value.strip()
            if cleaned:
                values.setdefault(key.strip().casefold(), cleaned)
    for key in keys:
        value = values.get(key.casefold())
        if value:
            return value
    return None


def _cpu_model() -> str:
    model = _proc_value(
        Path("/proc/cpuinfo"),
        ("model name", "hardware"),
    )
    if model:
        return model

    processor = platform.processor().strip()
    return processor or platform.machine() or "unknown"


def _memory_bytes() -> dict[str, int | None]:
    total_bytes: int | None = None
    available_bytes: int | None = None

    try:
        contents = Path("/proc/meminfo").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        contents = ""

    for line in contents.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in {"MemTotal", "MemAvailable"}:
            continue
        fields = value.split()
        if not fields:
            continue
        try:
            # Linux exposes these two values in KiB.
            byte_value = int(fields[0]) * 1024
        except ValueError:
            continue
        if key == "MemTotal":
            total_bytes = byte_value
        else:
            available_bytes = byte_value

    # sysconf is a portable fallback for systems without Linux procfs.
    if total_bytes is None:
        total_bytes = _sysconf_bytes("SC_PHYS_PAGES")
    if available_bytes is None:
        available_bytes = _sysconf_bytes("SC_AVPHYS_PAGES")

    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
    }


def _sysconf_bytes(page_count_name: str) -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf(page_count_name))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return page_size * page_count


def _run_allowlisted(
    command: str,
    arguments: Sequence[str],
) -> tuple[str, subprocess.CompletedProcess[str] | None, str | None]:
    """Run one allowlisted executable without a shell.

    The returned tuple is ``(status, completed_process, reason)``.  Exception
    text and stderr are intentionally not returned because they may contain
    local paths or other data that do not belong in a baseline artifact.
    """

    executable = shutil.which(command)
    if executable is None:
        return "unavailable", None, "command_not_found"

    try:
        completed = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "error", None, "timeout"
    except (OSError, ValueError):
        return "error", None, "execution_failed"

    if completed.returncode != 0:
        return "error", completed, "nonzero_exit"
    return "available", completed, None


def _version_probe(command: str) -> dict[str, Any]:
    status, completed, reason = _run_allowlisted(command, ("--version",))
    result: dict[str, Any] = {"status": status, "version": None}
    if reason is not None:
        result["reason"] = reason
    if completed is not None and status == "available":
        result["version"] = next(
            (line.strip() for line in completed.stdout.splitlines() if line.strip()),
            "unknown",
        )[:512]
    return result


def _nvidia_probe() -> dict[str, Any]:
    status, completed, reason = _run_allowlisted(
        "nvidia-smi",
        (
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ),
    )
    result: dict[str, Any] = {"status": status, "devices": []}
    if reason is not None:
        result["reason"] = reason
    if completed is None or status != "available":
        return result

    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines()[:_MAX_ACCELERATORS]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            memory_total_mib: int | None = int(fields[2])
        except ValueError:
            memory_total_mib = None
        devices.append(
            {
                "name": fields[0][:256],
                "driver_version": fields[1][:128],
                "compute_capability": fields[3][:64],
                "memory_total_bytes": (
                    memory_total_mib * 1024 * 1024
                    if memory_total_mib is not None
                    else None
                ),
            }
        )
    result["devices"] = devices
    return result


def _rocminfo_probe() -> dict[str, Any]:
    status, completed, reason = _run_allowlisted("rocminfo", ())
    result: dict[str, Any] = {"status": status, "devices": []}
    if reason is not None:
        result["reason"] = reason
    if completed is None or status != "available":
        return result

    devices: list[dict[str, str]] = []
    current: dict[str, str] = {}

    def finish_agent() -> None:
        if current.get("device_type") != "GPU":
            return
        architecture = current.get("architecture")
        name = current.get("marketing_name") or architecture
        if not architecture or not name:
            return
        candidate = {"name": name[:256], "architecture": architecture[:128]}
        if candidate not in devices and len(devices) < _MAX_ACCELERATORS:
            devices.append(candidate)

    for line in completed.stdout.splitlines():
        if line.startswith("Agent "):
            finish_agent()
            current = {}
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().casefold()
        cleaned = value.strip()
        if normalized_key == "name" and "architecture" not in current:
            current["architecture"] = cleaned
        elif normalized_key == "marketing name":
            current["marketing_name"] = cleaned
        elif normalized_key == "device type":
            current["device_type"] = cleaned
    finish_agent()
    result["devices"] = devices
    return result


def collect_machine_baseline(target_id: str) -> dict[str, Any]:
    """Collect a non-sensitive machine baseline for ``target_id``.

    Args:
        target_id: Stable project-defined identifier for the target machine.

    Raises:
        ValueError: If ``target_id`` is empty.
    """

    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be a non-empty string")

    return {
        "schema_version": 1,
        "kind": "machine_baseline",
        "captured_at": _utc_now(),
        "target_id": target_id.strip(),
        "host": {
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "memory": _memory_bytes(),
        "software": {
            "python": {
                "status": "available",
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "docker": _version_probe("docker"),
            "git": _version_probe("git"),
            "hipconfig": _version_probe("hipconfig"),
            "nvcc": _version_probe("nvcc"),
        },
        "accelerators": {
            "nvidia_smi": _nvidia_probe(),
            "rocminfo": _rocminfo_probe(),
        },
    }


__all__ = ["collect_machine_baseline"]
