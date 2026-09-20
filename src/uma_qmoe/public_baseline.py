"""Normalize vLLM serving output into fail-closed public baseline evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .contracts import ContractError, validate_document


_RAW_FILES = ("vllm-result.json", "runner-metadata.json", "server.log", "benchmark.log")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read public baseline input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"public baseline input {path} must be an object")
    return value


def _finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ContractError(f"{field} must be finite and at least {minimum}")
    return float(value)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    return value


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ContractError("latency samples must be non-empty, finite, and non-negative")
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _identity(value: str, field: str, length: int) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{field} must be a lowercase {length}-character hex identity")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(
            f"public baseline artifact must be a regular non-symlink file: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot hash public baseline artifact {path}: {exc}") from exc
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def build_public_baseline(
    run_directory: str | Path,
    *,
    target_id: str,
    implementation_version: str,
    source_commit: str,
    backend: str,
    container_image: str,
    model_id: str,
    model_revision: str,
    derivation_semantic_sha256: str,
    input_tokens: int,
    output_tokens: int,
    warmup_requests: int,
    measured_requests: int,
    max_concurrency: int,
) -> dict[str, Any]:
    """Build validated evidence from one runner output directory."""

    directory = Path(run_directory)
    paths = {name: directory / name for name in _RAW_FILES}
    if any(not path.is_file() for path in paths.values()):
        raise ContractError("public baseline run directory is missing required raw files")
    raw = _load_mapping(paths["vllm-result.json"])
    metadata = _load_mapping(paths["runner-metadata.json"])
    if metadata.get("target_id") != target_id:
        raise ContractError("runner target_id does not match requested target")

    duration = _finite(raw.get("duration"), "duration", minimum=1e-12)
    completed = _integer(raw.get("completed"), "completed")
    failed = _integer(raw.get("failed"), "failed")
    total_input = _integer(raw.get("total_input_tokens"), "total_input_tokens")
    total_output = _integer(raw.get("total_output_tokens"), "total_output_tokens")
    request_throughput = _finite(raw.get("request_throughput"), "request_throughput")
    output_throughput = _finite(raw.get("output_throughput"), "output_throughput")
    total_throughput = _finite(raw.get("total_token_throughput"), "total_token_throughput")
    input_lens = raw.get("input_lens")
    output_lens = raw.get("output_lens")
    ttfts = raw.get("ttfts")
    itls = raw.get("itls")
    errors = raw.get("errors")
    if not all(isinstance(value, list) for value in (input_lens, output_lens, ttfts, itls, errors)):
        raise ContractError("vLLM result must retain detailed request arrays")
    if not all(len(value) == completed for value in (input_lens, output_lens, ttfts, itls, errors)):
        raise ContractError("vLLM detailed request arrays do not match completed count")

    normalized_input = [_integer(value, "input_lens") for value in input_lens]
    normalized_output = [_integer(value, "output_lens") for value in output_lens]
    ttft_ms = [_finite(value, "ttfts") * 1000.0 for value in ttfts]
    tpot_ms: list[float] = []
    e2el_ms: list[float] = []
    for index, raw_intervals in enumerate(itls):
        if not isinstance(raw_intervals, list):
            raise ContractError("each itls entry must be an array")
        intervals = [_finite(value, "itls") for value in raw_intervals]
        if normalized_output[index] > 1 and len(intervals) != normalized_output[index] - 1:
            raise ContractError("inter-token latency count does not match output length")
        tpot_ms.append(
            (sum(intervals) / (normalized_output[index] - 1)) * 1000.0
        )
        e2el_ms.append(ttft_ms[index] + sum(intervals) * 1000.0)

    throughput_consistent = all(
        math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-9)
        for observed, expected in (
            (request_throughput, completed / duration),
            (output_throughput, total_output / duration),
            (total_throughput, (total_input + total_output) / duration),
        )
    )
    gates = {
        "runner_succeeded": metadata.get("status") == "succeeded"
        and metadata.get("benchmark_returncode") == 0,
        "all_requests_completed": completed == measured_requests and failed == 0,
        "no_request_errors": all(error in (None, "") for error in errors),
        "exact_input_length": normalized_input == [input_tokens] * completed,
        "exact_output_length": normalized_output == [output_tokens] * completed,
        "throughput_consistent": throughput_consistent,
    }
    gates["overall_passed"] = all(gates.values())

    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("runner metadata must contain resource samples")
    temperatures: list[float] = []
    powers: list[float] = []
    available: list[int] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ContractError("runner resource sample must be an object")
        value = sample.get("memory_available_bytes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            available.append(value)
        telemetry = sample.get("telemetry")
        if isinstance(telemetry, dict):
            temperature = telemetry.get("temperature_c")
            power = telemetry.get("socket_power_w")
            if isinstance(temperature, (int, float)) and math.isfinite(float(temperature)):
                temperatures.append(float(temperature))
            if isinstance(power, (int, float)) and math.isfinite(float(power)):
                powers.append(float(power))
    peak = _integer(metadata.get("cgroup_memory_peak_bytes"), "cgroup_memory_peak_bytes")
    if peak == 0:
        raise ContractError("cgroup memory peak must be positive")

    document = {
        "schema_version": 1,
        "kind": "public_baseline",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "implementation": {
            "name": "vllm",
            "version": implementation_version,
            "source_commit": _identity(source_commit, "source_commit", 40),
            "backend": backend,
            "container_image": container_image,
        },
        "model": {
            "model_id": model_id,
            "revision": _identity(model_revision, "model_revision", 40),
            "weight_dtype": "BF16",
            "derivation_semantic_sha256": _identity(
                derivation_semantic_sha256, "derivation_semantic_sha256", 64
            ),
        },
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "warmup_requests": warmup_requests,
            "measured_requests": measured_requests,
            "max_concurrency": max_concurrency,
            "request_rate": "inf",
            "ignore_eos": True,
            "seed": 0,
        },
        "results": {
            "duration_seconds": duration,
            "completed_requests": completed,
            "failed_requests": failed,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "request_throughput": request_throughput,
            "output_token_throughput": output_throughput,
            "total_token_throughput": total_throughput,
            "ttft_ms": _summary(ttft_ms),
            "tpot_ms": _summary(tpot_ms),
            "e2el_ms": _summary(e2el_ms),
        },
        "memory": {"scope": "workload_cgroup", "peak_bytes": peak},
        "telemetry": {
            "sample_count": len(samples),
            "maximum_temperature_c": max(temperatures) if temperatures else None,
            "maximum_socket_power_w": max(powers) if powers else None,
            "minimum_memory_available_bytes": min(available) if available else None,
        },
        "raw_artifacts": [_artifact(paths[name]) for name in _RAW_FILES],
        "gates": gates,
    }
    return document


def build_public_baseline_run_manifest(
    baseline_path: str | Path,
    run_directory: str | Path,
    *,
    run_id: str,
    git_commit: str,
    git_dirty: bool,
    dirty_patch_sha256: str | None,
    machine_baseline_sha256: str,
    benchmark_contract_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind one normalized baseline and its raw artifacts into RunManifest v1."""

    baseline_file = Path(baseline_path)
    baseline = _load_mapping(baseline_file)
    validate_document(baseline)
    if baseline.get("kind") != "public_baseline":
        raise ContractError("run manifest builder requires Public Baseline evidence")
    directory = Path(run_directory)
    metadata = _load_mapping(directory / "runner-metadata.json")
    runner_argv = metadata.get("runner_argv")
    if (
        not isinstance(runner_argv, list)
        or not runner_argv
        or any(not isinstance(item, str) for item in runner_argv)
    ):
        raise ContractError("runner metadata has no complete runner argv")
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("runner metadata has no resource samples")

    def boundary(index: int, key: str) -> Any:
        sample = samples[index]
        if not isinstance(sample, dict):
            return None
        telemetry = sample.get("telemetry")
        if key in {"temperature_c", "socket_power_w"}:
            return telemetry.get(key) if isinstance(telemetry, dict) else None
        return sample.get(key)

    git_commit = _identity(git_commit, "git_commit", 40)
    machine_baseline_sha256 = _identity(
        machine_baseline_sha256, "machine_baseline_sha256", 64
    )
    benchmark_contract_sha256 = _identity(
        benchmark_contract_sha256, "benchmark_contract_sha256", 64
    )
    model_manifest_sha256 = _identity(
        model_manifest_sha256, "model_manifest_sha256", 64
    )
    if git_dirty:
        if dirty_patch_sha256 is None:
            raise ContractError("dirty RunManifest requires dirty_patch_sha256")
        dirty_patch_sha256 = _identity(
            dirty_patch_sha256, "dirty_patch_sha256", 64
        )
    elif dirty_patch_sha256 is not None:
        raise ContractError("clean RunManifest must not set dirty_patch_sha256")

    artifacts = [_artifact(directory / name) for name in _RAW_FILES]
    artifacts.append(_artifact(baseline_file))
    manifest = {
        "schema_version": 1,
        "kind": "run_manifest",
        "run_id": run_id,
        "status": "succeeded" if baseline["status"] == "passed" else "failed",
        "created_at": metadata["started_at"],
        "completed_at": metadata["completed_at"],
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
            "dirty_patch_sha256": dirty_patch_sha256,
        },
        "target": {
            "id": baseline["target_id"],
            "machine_baseline_sha256": machine_baseline_sha256,
            "container_image": baseline["implementation"]["container_image"],
        },
        "inputs": {
            "benchmark_contract_sha256": benchmark_contract_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "route_trace_sha256": None,
            "quant_policy_sha256": None,
            "expert_pack_sha256": None,
        },
        "command": {
            "argv": runner_argv,
            "environment_allowlist": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        },
        "conditions": {
            "power_mode": None,
            "temperature_celsius": {
                "start": boundary(0, "temperature_c"),
                "end": boundary(-1, "temperature_c"),
            },
            "memory_available_bytes": {
                "start": boundary(0, "memory_available_bytes"),
                "end": boundary(-1, "memory_available_bytes"),
            },
        },
        "raw_artifacts": artifacts,
        "aggregation": {
            "method": "median",
            "sample_count": baseline["workload"]["measured_requests"],
            "report_quantiles": ["p50", "p95", "p99"],
        },
        "result_class": "pending",
    }
    validate_document(manifest)
    return manifest


__all__ = ["build_public_baseline", "build_public_baseline_run_manifest"]
