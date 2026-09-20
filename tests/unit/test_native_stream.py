from __future__ import annotations

import copy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from uma_qmoe import native_stream
from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.native_stream import _native_compile_command


def _evidence() -> dict:
    size = 512 * 1024 * 1024
    inner = 64
    operations = []
    for operation_id, read, write in (
        ("read_reduce", size * inner, 0),
        ("write", 0, size * inner),
        ("copy", size * inner, size * inner),
    ):
        operations.append(
            {
                "id": operation_id,
                "algorithmic_read_bytes": read,
                "algorithmic_write_bytes": write,
                "algorithmic_total_bytes": read + write,
                "samples_seconds": [0.1, 0.11, 0.09],
                "median_seconds": 0.1,
                "mean_seconds": 0.1,
                "coefficient_of_variation": 0.08,
                "effective_gbps": (read + write) / 0.1 / 1e9,
            }
        )
    return {
        "schema_version": 1,
        "kind": "native_stream_benchmark",
        "generated_at": "2026-09-18T02:00:00Z",
        "target_id": "spark1",
        "status": "passed",
        "source": {
            "path": "benchmarks/native/native_stream.cu",
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
            "multiprocessor_count": 48,
            "threads_per_block": 256,
            "blocks": 1536,
        },
        "configuration": {
            "requested_buffer_bytes": size,
            "actual_buffer_bytes": size,
            "warmup_iterations": 3,
            "measured_iterations": 3,
            "inner_iterations": inner,
        },
        "measurement_scope": {
            "elapsed_time": "gpu_events",
            "traffic": "algorithmic_bytes",
            "counter_calibrated": False,
            "limitations": ["not counter calibrated"],
        },
        "operations": operations,
    }


def test_native_stream_evidence_validates() -> None:
    validate_document(_evidence())


def test_native_stream_rejects_wrong_bytes_and_unsafe_source() -> None:
    wrong = copy.deepcopy(_evidence())
    wrong["operations"][2]["algorithmic_total_bytes"] -= 1
    with pytest.raises(ContractError, match="invalid total bytes"):
        validate_document(wrong)

    unsafe = copy.deepcopy(_evidence())
    unsafe["source"]["path"] = "../native_stream.cu"
    with pytest.raises(ContractError, match="safe project-relative"):
        validate_document(unsafe)


def test_hip_compile_command_embeds_wheel_sdk_runtime_path(tmp_path: Path) -> None:
    sdk = tmp_path / "_rocm_sdk_devel"
    compiler = sdk / "bin" / "hipcc"
    runtime = sdk / "lib" / "libamdhip64.so"
    compiler.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    compiler.touch()
    runtime.touch()

    command = _native_compile_command(
        compiler=str(compiler),
        backend="hip",
        architecture="gfx1151",
        source=Path("native_stream.cu"),
        executable=Path("native-stream"),
    )

    assert f"-Wl,-rpath,{sdk / 'lib'}" in command
    assert "--offload-arch=gfx1151" in command


def test_hip_compile_command_resolves_runtime_from_hipconfig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = tmp_path / "usr" / "local" / "bin" / "hipcc"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()
    sdk = tmp_path / "site-packages" / "_rocm_sdk_devel"
    hipconfig = sdk / "bin" / "hipconfig"
    runtime = sdk / "lib" / "libamdhip64.so"
    hipconfig.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    hipconfig.touch()
    runtime.touch()
    original_which = shutil.which

    def fake_which(command: str) -> str | None:
        if command == "hipconfig":
            return str(hipconfig)
        return original_which(command)

    monkeypatch.setattr(shutil, "which", fake_which)
    command = _native_compile_command(
        compiler=str(wrapper),
        backend="hip",
        architecture="gfx1151",
        source=Path("native_stream.cu"),
        executable=Path("native-stream"),
    )

    assert f"-Wl,-rpath,{sdk / 'lib'}" in command


def test_hip_compile_command_uses_hipconfig_reported_sdk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrappers = tmp_path / "usr" / "local" / "bin"
    compiler = wrappers / "hipcc"
    hipconfig = wrappers / "hipconfig"
    wrappers.mkdir(parents=True)
    compiler.touch()
    hipconfig.touch()
    sdk = tmp_path / "site-packages" / "_rocm_sdk_devel"
    runtime = sdk / "lib" / "libamdhip64.so"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(shutil, "which", lambda command: str(hipconfig))
    monkeypatch.setattr(
        native_stream,
        "_run_checked",
        lambda command, timeout: SimpleNamespace(stdout=str(sdk)),
    )

    command = _native_compile_command(
        compiler=str(compiler),
        backend="hip",
        architecture="gfx1151",
        source=Path("native_stream.cu"),
        executable=Path("native-stream"),
    )

    assert f"-Wl,-rpath,{sdk / 'lib'}" in command


def test_cuda_compile_command_does_not_add_rocm_rpath(tmp_path: Path) -> None:
    compiler = tmp_path / "nvcc"
    compiler.touch()
    command = _native_compile_command(
        compiler=str(compiler),
        backend="cuda",
        architecture="sm_121",
        source=Path("native_stream.cu"),
        executable=Path("native-stream"),
    )
    assert "-arch=sm_121" in command
    assert not any(argument.startswith("-Wl,-rpath,") for argument in command)
