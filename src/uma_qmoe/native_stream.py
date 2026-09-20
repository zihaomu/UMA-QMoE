"""Compile and run the target-native known-byte streaming benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Literal

from .contracts import ContractError
from .memory_bandwidth import _summarize_samples


_TARGET_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]+$")
_ARCHITECTURE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_MAX_OUTPUT_BYTES = 1_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _run_checked(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"native command {Path(command[0]).name!r} failed to execute") from exc
    if completed.returncode != 0:
        raise ContractError(
            f"native command {Path(command[0]).name!r} returned exit {completed.returncode}"
        )
    if len(completed.stdout.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise ContractError("native benchmark output exceeds the evidence limit")
    return completed


def _native_compile_command(
    *,
    compiler: str,
    backend: Literal["cuda", "hip"],
    architecture: str,
    source: Path,
    executable: Path,
) -> list[str]:
    command = [compiler, "-O3", "-std=c++17"]
    if backend == "cuda":
        command.append(f"-arch={architecture}")
    else:
        command.append(f"--offload-arch={architecture}")
        # The Halo image ships ROCm as Python wheel packages. hipcc links
        # libamdhip64 but does not add the wheel's sibling lib directory to
        # the dynamic loader path, so an otherwise valid executable exits 127.
        runtime_candidates = [Path(compiler).resolve().parent.parent / "lib"]
        hipconfig = shutil.which("hipconfig")
        if hipconfig is not None:
            runtime_candidates.append(Path(hipconfig).resolve().parent.parent / "lib")
        rocm_library_directory = next(
            (
                candidate
                for candidate in runtime_candidates
                if (candidate / "libamdhip64.so").is_file()
            ),
            None,
        )
        if rocm_library_directory is None and hipconfig is not None:
            configured_root = Path(
                _run_checked([hipconfig, "--path"], timeout=15.0).stdout.strip()
            )
            configured_library_directory = configured_root / "lib"
            if (
                configured_root.is_absolute()
                and (configured_library_directory / "libamdhip64.so").is_file()
            ):
                rocm_library_directory = configured_library_directory
        if rocm_library_directory is None:
            raise ContractError(
                "ROCm compiler tools do not identify a runtime directory "
                "containing libamdhip64.so"
            )
        command.append(f"-Wl,-rpath,{rocm_library_directory}")
    command.extend((str(source), "-o", str(executable)))
    return command


def benchmark_native_stream(
    source_path: str | Path,
    *,
    source_relative_path: str,
    source_file_sha256: str,
    target_id: str,
    backend: Literal["cuda", "hip"],
    architecture: str,
    requested_buffer_bytes: int,
    warmup_iterations: int,
    measured_iterations: int,
    inner_iterations: int,
) -> dict[str, Any]:
    """Compile a pinned source file and return normalized benchmark evidence."""

    source = Path(source_path)
    if not source.is_file():
        raise ContractError(f"native stream source does not exist: {source}")
    if not _TARGET_ID.fullmatch(target_id):
        raise ContractError("target_id has unsafe characters")
    if not _ARCHITECTURE.fullmatch(architecture):
        raise ContractError("architecture has unsafe characters")
    if not 4 * 1024 * 1024 <= requested_buffer_bytes <= 8 * 1024**3:
        raise ContractError("requested native buffer must be within [4 MiB, 8 GiB]")
    if not 1 <= warmup_iterations <= 100:
        raise ContractError("warmup_iterations must be within [1, 100]")
    if not 3 <= measured_iterations <= 100:
        raise ContractError("measured_iterations must be within [3, 100]")
    if not 1 <= inner_iterations <= 10_000:
        raise ContractError("inner_iterations must be within [1, 10000]")

    compiler_name = "nvcc" if backend == "cuda" else "hipcc"
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise ContractError(f"required native compiler {compiler_name!r} is unavailable")
    version = _run_checked([compiler, "--version"], timeout=15.0)
    version_line = next(
        (line.strip() for line in version.stdout.splitlines() if line.strip()),
        compiler_name,
    )[:512]

    with tempfile.TemporaryDirectory(prefix="uma-qmoe-native-stream-") as directory:
        executable = Path(directory) / "native-stream"
        compile_command = _native_compile_command(
            compiler=compiler,
            backend=backend,
            architecture=architecture,
            source=source,
            executable=executable,
        )
        _run_checked(compile_command, timeout=180.0)
        completed = _run_checked(
            [
                str(executable),
                target_id,
                str(requested_buffer_bytes),
                str(warmup_iterations),
                str(measured_iterations),
                str(inner_iterations),
            ],
            timeout=600.0,
        )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError("native benchmark emitted malformed JSON") from exc
    if not isinstance(raw, dict) or raw.get("backend") != backend:
        raise ContractError("native benchmark backend identity mismatch")
    actual_bytes = raw.get("actual_buffer_bytes")
    if not isinstance(actual_bytes, int) or actual_bytes <= 0:
        raise ContractError("native benchmark returned invalid actual_buffer_bytes")
    if actual_bytes > requested_buffer_bytes or requested_buffer_bytes - actual_bytes >= 16:
        raise ContractError("native benchmark buffer rounding exceeded one float4")
    raw_operations = raw.get("operations")
    if not isinstance(raw_operations, list):
        raise ContractError("native benchmark did not return operations")
    by_id: dict[str, list[float]] = {}
    for operation in raw_operations:
        if not isinstance(operation, dict):
            raise ContractError("native operation must be an object")
        operation_id = operation.get("id")
        samples = operation.get("samples_seconds")
        if operation_id in by_id or operation_id not in {"read_reduce", "write", "copy"}:
            raise ContractError("native benchmark returned duplicate or unknown operation")
        if not isinstance(samples, list) or not all(isinstance(item, (int, float)) for item in samples):
            raise ContractError("native benchmark returned invalid samples")
        by_id[operation_id] = [float(item) for item in samples]
    if set(by_id) != {"read_reduce", "write", "copy"}:
        raise ContractError("native benchmark did not return all required operations")

    traffic = {
        "read_reduce": (actual_bytes * inner_iterations, 0),
        "write": (0, actual_bytes * inner_iterations),
        "copy": (actual_bytes * inner_iterations, actual_bytes * inner_iterations),
    }
    operations = []
    for operation_id in ("read_reduce", "write", "copy"):
        read_bytes, write_bytes = traffic[operation_id]
        total_bytes = read_bytes + write_bytes
        operations.append(
            {
                "id": operation_id,
                "algorithmic_read_bytes": read_bytes,
                "algorithmic_write_bytes": write_bytes,
                "algorithmic_total_bytes": total_bytes,
                **_summarize_samples(by_id[operation_id], total_bytes),
            }
        )

    runtime_keys = ("device_name", "multiprocessor_count", "threads_per_block", "blocks")
    if any(key not in raw for key in runtime_keys):
        raise ContractError("native benchmark omitted runtime identity")
    return {
        "schema_version": 1,
        "kind": "native_stream_benchmark",
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
        "runtime": {
            "backend": backend,
            **{key: raw[key] for key in runtime_keys},
        },
        "configuration": {
            "requested_buffer_bytes": requested_buffer_bytes,
            "actual_buffer_bytes": actual_bytes,
            "warmup_iterations": warmup_iterations,
            "measured_iterations": measured_iterations,
            "inner_iterations": inner_iterations,
        },
        "measurement_scope": {
            "elapsed_time": "gpu_events",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": [
                "read_reduce includes arithmetic and one atomic accumulation per thread",
                "algorithmic bytes are not yet calibrated against a hardware DRAM counter",
            ],
        },
        "operations": operations,
    }


__all__ = ["_native_compile_command", "benchmark_native_stream"]
