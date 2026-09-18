"""Compile and normalize target-native UMA allocation capability evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Literal

from .contracts import ContractError
from .native_stream import _native_compile_command, _run_checked


_TARGET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]+$")
_ARCHITECTURE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_COMMON_ATTRIBUTES = {
    "managed_memory",
    "concurrent_managed_access",
    "pageable_memory_access",
    "pageable_memory_access_uses_host_page_tables",
    "direct_managed_memory_access_from_host",
    "host_native_atomic_supported",
    "memory_pools_supported",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalize_capability_output(
    raw: Any, *, backend: Literal["cuda", "hip"]
) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("backend") != backend:
        raise ContractError("native allocation capability backend mismatch")
    if not isinstance(raw.get("device_name"), str) or not raw["device_name"].strip():
        raise ContractError("native allocation capability omitted device_name")
    total_memory = raw.get("total_global_memory_bytes")
    if not isinstance(total_memory, int) or isinstance(total_memory, bool) or total_memory <= 0:
        raise ContractError("native allocation capability returned invalid global memory")
    attributes = raw.get("attributes")
    expected = set(_COMMON_ATTRIBUTES)
    if backend == "hip":
        expected.add("virtual_memory_management_supported")
    if not isinstance(attributes, dict) or set(attributes) != expected:
        raise ContractError("native allocation capability returned an invalid attribute set")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1}
        for value in attributes.values()
    ):
        raise ContractError("native allocation attributes must be integer booleans")
    return {
        "device_name": raw["device_name"].strip(),
        "total_global_memory_bytes": total_memory,
        "attributes": dict(attributes),
    }


def probe_native_allocation_capabilities(
    source_path: str | Path,
    *,
    source_relative_path: str,
    source_file_sha256: str,
    target_id: str,
    backend: Literal["cuda", "hip"],
    architecture: str,
) -> dict[str, Any]:
    """Compile a pinned read-only capability probe on the target."""

    source = Path(source_path)
    if not source.is_file():
        raise ContractError(f"native allocation source does not exist: {source}")
    if not _TARGET_ID.fullmatch(target_id):
        raise ContractError("target_id has unsafe characters")
    if not _ARCHITECTURE.fullmatch(architecture):
        raise ContractError("architecture has unsafe characters")
    compiler_name = "nvcc" if backend == "cuda" else "hipcc"
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise ContractError(f"required native compiler {compiler_name!r} is unavailable")
    version = _run_checked([compiler, "--version"], timeout=15.0)
    version_line = next(
        (line.strip() for line in version.stdout.splitlines() if line.strip()),
        compiler_name,
    )[:512]
    with tempfile.TemporaryDirectory(prefix="uma-qmoe-native-allocation-") as directory:
        executable = Path(directory) / "allocation-capabilities"
        _run_checked(
            _native_compile_command(
                compiler=compiler,
                backend=backend,
                architecture=architecture,
                source=source,
                executable=executable,
            ),
            timeout=180.0,
        )
        completed = _run_checked([str(executable)], timeout=60.0)
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("native allocation capability emitted malformed JSON") from exc
    normalized = _normalize_capability_output(raw, backend=backend)
    return {
        "schema_version": 1,
        "kind": "native_allocation_capabilities",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed",
        "source": {"path": source_relative_path, "sha256": source_file_sha256},
        "build": {
            "backend": backend,
            "compiler": compiler_name,
            "compiler_version": version_line,
            "architecture": architecture,
            "optimization": "O3",
        },
        "runtime": normalized,
        "limitations": [
            "capability flags do not prove performance or migration behavior",
            "CUDA runtime attributes do not cover the driver VMM capability flag",
        ],
    }


__all__ = ["_normalize_capability_output", "probe_native_allocation_capabilities"]
