#!/usr/bin/env python3
"""Run a reproducible local vLLM serving baseline without network access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def _memory_available() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def _telemetry() -> dict[str, Any] | None:
    try:
        from uma_qmoe.telemetry import TelemetryProbe

        return TelemetryProbe.detect().capture(
            operation_id="public_baseline", sample_index=0, boundary="before"
        )
    except Exception:
        return None


class _Sampler:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._accelerator_telemetry = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def enable_accelerator_telemetry(self) -> None:
        self._accelerator_telemetry.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "cgroup_memory_current_bytes": _read_integer(
                        Path("/sys/fs/cgroup/memory.current")
                    ),
                    "memory_available_bytes": _memory_available(),
                    "telemetry": (
                        _telemetry()
                        if self._accelerator_telemetry.is_set()
                        else None
                    ),
                }
            )
            self._stop.wait(self.interval_seconds)


def _wait_ready(url: str, process: subprocess.Popen[Any], timeout: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited before readiness: {process.returncode}")
        try:
            with urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return time.monotonic() - started
        except (OSError, URLError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"vLLM server did not become ready within {timeout:.0f}s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--served-model-name", default="umaq-public-baseline")
    parser.add_argument("--vllm", default="vllm")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if args.input_len < 1 or args.output_len < 2:
        raise SystemExit("input-len must be positive and output-len must be at least 2")
    if args.num_warmups < 1 or args.num_prompts < 3 or args.max_concurrency < 1:
        raise SystemExit("warmups, prompts, and concurrency are below the evidence floor")
    if not 0 < args.gpu_memory_utilization < 1:
        raise SystemExit("gpu-memory-utilization must be within (0, 1)")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_result = output / "vllm-result.json"
    metadata_path = output / "runner-metadata.json"
    server_log = output / "server.log"
    benchmark_log = output / "benchmark.log"
    for path in (raw_result, metadata_path, server_log, benchmark_log):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing evidence: {path}")

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    max_model_len = args.input_len + args.output_len
    server_argv = [
        args.vllm,
        "serve",
        str(args.model.resolve()),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    benchmark_argv = [
        args.vllm,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        str(args.model.resolve()),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0.0",
        "--num-warmups",
        str(args.num_warmups),
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "0",
        "--save-detailed",
        "--save-result",
        "--result-dir",
        str(output),
        "--result-filename",
        raw_result.name,
        "--metadata",
        f"target_id={args.target_id}",
    ]

    started_at = _utc_now()
    sampler = _Sampler(args.sample_interval_seconds)
    server: subprocess.Popen[Any] | None = None
    ready_seconds: float | None = None
    benchmark_returncode: int | None = None
    failure: str | None = None
    with server_log.open("x", encoding="utf-8") as server_stream, benchmark_log.open(
        "x", encoding="utf-8"
    ) as benchmark_stream:
        sampler.start()
        try:
            server = subprocess.Popen(
                server_argv,
                stdout=server_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
            ready_seconds = _wait_ready(
                f"http://127.0.0.1:{args.port}/health",
                server,
                args.ready_timeout_seconds,
            )
            # On ROCm, invoking amd-smi concurrently with vLLM's platform
            # discovery can make device detection fail. Memory sampling starts
            # immediately, but management telemetry is enabled only after the
            # server has completed device discovery and reports healthy.
            sampler.enable_accelerator_telemetry()
            completed = subprocess.run(
                benchmark_argv,
                stdout=benchmark_stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )
            benchmark_returncode = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"vLLM serving benchmark returned {completed.returncode}"
                )
            if not raw_result.is_file():
                raise RuntimeError("vLLM benchmark did not create its result JSON")
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            sampler.stop()
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10.0)

    memory_values = [
        sample["cgroup_memory_current_bytes"]
        for sample in sampler.samples
        if sample["cgroup_memory_current_bytes"] is not None
    ]
    metadata = {
        "schema_version": 1,
        "kind": "vllm_public_baseline_runner_metadata",
        "target_id": args.target_id,
        "status": "succeeded" if failure is None else "failed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "server_ready_seconds": ready_seconds,
        "server_argv": server_argv,
        "benchmark_argv": benchmark_argv,
        "runner_argv": [sys.executable, *sys.argv],
        "benchmark_returncode": benchmark_returncode,
        "failure": failure,
        "cgroup_memory_peak_bytes": max(memory_values) if memory_values else None,
        "samples": sampler.samples,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failure is not None:
        raise SystemExit(failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
