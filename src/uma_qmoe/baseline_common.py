"""Runtime-neutral helpers for measured inference baseline evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from .contracts import ContractError, validate_document


REQUIRED_RAW_FILES = ("result.json", "runner-metadata.json")


def load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read baseline input {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"baseline input {source} must be an object")
    return value


def finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ContractError(f"{field} must be finite and at least {minimum}")
    return float(value)


def integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} must be an integer of at least {minimum}")
    return value


def identity(value: str, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{field} must be a lowercase {length}-character hex identity")
    return value


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summary(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ContractError("latency samples must be non-empty, finite, and non-negative")
    return {
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def require_safe_relative_path(path: str, field: str) -> None:
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ContractError(f"{field} must be a safe relative POSIX path")


def artifact(directory: Path, relative_path: str) -> dict[str, Any]:
    require_safe_relative_path(relative_path, "artifact path")
    path = directory
    try:
        root = directory.resolve(strict=True)
        for part in relative_path.split("/"):
            path = path / part
            if path.is_symlink():
                raise ContractError(
                    f"baseline artifact {relative_path!r} must not use symbolic links"
                )
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise ContractError(
                f"baseline artifact {relative_path!r} must be a regular file"
            )
        payload = resolved.read_bytes()
    except ContractError:
        raise
    except ValueError as exc:
        raise ContractError(
            f"baseline artifact {relative_path!r} escapes its run directory"
        ) from exc
    except OSError as exc:
        raise ContractError(f"cannot hash baseline artifact {path}: {exc}") from exc
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def raw_artifact_paths(metadata: Mapping[str, Any]) -> list[str]:
    configured = metadata.get("artifact_paths", list(REQUIRED_RAW_FILES))
    if (
        not isinstance(configured, list)
        or not configured
        or any(not isinstance(path, str) for path in configured)
    ):
        raise ContractError("runner metadata artifact_paths must be a non-empty string array")
    paths = list(configured)
    for required in REQUIRED_RAW_FILES:
        if required not in paths:
            raise ContractError(f"runner metadata artifact_paths must include {required}")
    if len(paths) != len(set(paths)):
        raise ContractError("runner metadata artifact_paths contains duplicates")
    for path in paths:
        require_safe_relative_path(path, "runner metadata artifact path")
    return paths


def normalize_requests(
    raw: Mapping[str, Any],
    *,
    expected_input_tokens: int,
    expected_output_tokens: int,
    measured_requests: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    duration = finite(raw.get("duration_seconds"), "duration_seconds", minimum=1e-12)
    requests = raw.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ContractError("result requests must be a non-empty array")

    input_lengths: list[int] = []
    output_lengths: list[int] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    e2els: list[float] = []
    errors: list[str | None] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ContractError(f"request {index} must be an object")
        input_length = integer(request.get("input_tokens"), f"requests[{index}].input_tokens")
        output_length = integer(
            request.get("output_tokens"), f"requests[{index}].output_tokens", minimum=1
        )
        ttft = finite(request.get("ttft_ms"), f"requests[{index}].ttft_ms")
        intervals = request.get("inter_token_latencies_ms")
        if not isinstance(intervals, list):
            raise ContractError(
                f"requests[{index}].inter_token_latencies_ms must be an array"
            )
        normalized_intervals = [
            finite(value, f"requests[{index}].inter_token_latencies_ms")
            for value in intervals
        ]
        if len(normalized_intervals) != output_length - 1:
            raise ContractError(
                f"request {index} inter-token latency count does not match output length"
            )
        error = request.get("error")
        if error is not None and not isinstance(error, str):
            raise ContractError(f"requests[{index}].error must be a string or null")
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        ttfts.append(ttft)
        tpots.append(
            statistics.fmean(normalized_intervals) if normalized_intervals else 0.0
        )
        e2els.append(ttft + sum(normalized_intervals))
        errors.append(error)

    completed = len(requests)
    failed = sum(error not in (None, "") for error in errors)
    total_input = sum(input_lengths)
    total_output = sum(output_lengths)
    results = {
        "duration_seconds": duration,
        "completed_requests": completed,
        "failed_requests": failed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / duration,
        "output_token_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "ttft_ms": summary(ttfts),
        "tpot_ms": summary(tpots),
        "e2el_ms": summary(e2els),
    }
    gates = {
        "all_requests_completed": completed == measured_requests and failed == 0,
        "no_request_errors": failed == 0,
        "exact_input_length": input_lengths == [expected_input_tokens] * completed,
        "exact_output_length": output_lengths == [expected_output_tokens] * completed,
        "throughput_consistent": True,
    }
    return results, gates


def resource_evidence(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("runner metadata must contain resource samples")
    temperatures: list[float] = []
    powers: list[float] = []
    available: list[int] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ContractError("runner resource sample must be an object")
        memory_available = sample.get("memory_available_bytes")
        if isinstance(memory_available, int) and not isinstance(memory_available, bool):
            if memory_available >= 0:
                available.append(memory_available)
        telemetry = sample.get("telemetry")
        if isinstance(telemetry, dict):
            temperature = telemetry.get("temperature_c")
            power = telemetry.get("socket_power_w")
            if isinstance(temperature, (int, float)) and math.isfinite(float(temperature)):
                temperatures.append(float(temperature))
            if isinstance(power, (int, float)) and math.isfinite(float(power)):
                powers.append(float(power))

    cgroup_peak = integer(
        metadata.get("cgroup_memory_peak_bytes"), "cgroup_memory_peak_bytes", minimum=1
    )
    memory = {
        "scope": "workload_cgroup",
        "peak_bytes": cgroup_peak,
        "torch_peak_allocated_bytes": integer(
            metadata.get("torch_peak_allocated_bytes", 0), "torch_peak_allocated_bytes"
        ),
        "torch_peak_reserved_bytes": integer(
            metadata.get("torch_peak_reserved_bytes", 0), "torch_peak_reserved_bytes"
        ),
    }
    telemetry_summary = {
        "sample_count": len(samples),
        "maximum_temperature_c": max(temperatures) if temperatures else None,
        "maximum_socket_power_w": max(powers) if powers else None,
        "minimum_memory_available_bytes": min(available) if available else None,
    }
    return memory, telemetry_summary


def validate_baseline_semantics(document: Mapping[str, Any], label: str) -> None:
    results = document["results"]
    duration = results["duration_seconds"]
    expected = {
        "request_throughput": results["completed_requests"] / duration,
        "output_token_throughput": results["total_output_tokens"] / duration,
        "total_token_throughput": (
            results["total_input_tokens"] + results["total_output_tokens"]
        )
        / duration,
    }
    if any(
        not math.isclose(results[name], value, rel_tol=1e-6, abs_tol=1e-9)
        for name, value in expected.items()
    ):
        raise ContractError(f"{label} throughput does not match counts/duration")
    for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
        item = results[name]
        if not (
            item["p50"] <= item["p95"] <= item["p99"] <= item["max"]
        ) or item["mean"] > item["max"]:
            raise ContractError(f"{label} {name} summary is invalid")
    paths = [item["path"] for item in document["raw_artifacts"]]
    if len(paths) != len(set(paths)):
        raise ContractError(f"{label} raw artifacts contain duplicate paths")
    for path in paths:
        require_safe_relative_path(path, f"{label} raw artifact path")
    gates = document["gates"]
    expected_overall = all(
        value for name, value in gates.items() if name != "overall_passed"
    )
    if gates["overall_passed"] != expected_overall:
        raise ContractError(f"{label} overall gate is inconsistent")
    expected_status = "passed" if expected_overall else "failed"
    if document["status"] != expected_status:
        raise ContractError(f"{label} status does not match its gates")


def build_run_manifest(
    baseline_path: str | Path,
    run_directory: str | Path,
    *,
    expected_kind: str,
    run_id: str,
    git_commit: str,
    git_dirty: bool,
    dirty_patch_sha256: str | None,
    machine_baseline_sha256: str,
    benchmark_contract_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, Any]:
    baseline_file = Path(baseline_path)
    baseline = load_mapping(baseline_file)
    validate_document(baseline)
    if baseline.get("kind") != expected_kind:
        raise ContractError(f"run manifest builder requires {expected_kind} evidence")
    directory = Path(run_directory)
    metadata = load_mapping(directory / "runner-metadata.json")
    argv = metadata.get("runner_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) for item in argv)
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
        if key == "temperature_c":
            return telemetry.get(key) if isinstance(telemetry, dict) else None
        return sample.get(key)

    git_commit = identity(git_commit, "git_commit", 40)
    machine_baseline_sha256 = identity(
        machine_baseline_sha256, "machine_baseline_sha256", 64
    )
    benchmark_contract_sha256 = identity(
        benchmark_contract_sha256, "benchmark_contract_sha256", 64
    )
    model_manifest_sha256 = identity(
        model_manifest_sha256, "model_manifest_sha256", 64
    )
    if git_dirty:
        if dirty_patch_sha256 is None:
            raise ContractError("dirty RunManifest requires dirty_patch_sha256")
        dirty_patch_sha256 = identity(
            dirty_patch_sha256, "dirty_patch_sha256", 64
        )
    elif dirty_patch_sha256 is not None:
        raise ContractError("clean RunManifest must not set dirty_patch_sha256")

    raw_paths = raw_artifact_paths(metadata)
    raw_artifacts = [artifact(directory, path) for path in raw_paths]
    if baseline_file.name in raw_paths:
        raise ContractError("baseline evidence path duplicates a runner artifact")
    if baseline_file.is_symlink() or not baseline_file.is_file():
        raise ContractError("baseline evidence must be a regular non-symlink file")
    baseline_payload = baseline_file.read_bytes()
    raw_artifacts.append(
        {
            "path": baseline_file.name,
            "sha256": hashlib.sha256(baseline_payload).hexdigest(),
            "size_bytes": len(baseline_payload),
        }
    )
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
            "argv": argv,
            "environment_allowlist": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
        },
        "conditions": {
            "power_mode": metadata.get("power_mode"),
            "temperature_celsius": {
                "start": boundary(0, "temperature_c"),
                "end": boundary(-1, "temperature_c"),
            },
            "memory_available_bytes": {
                "start": boundary(0, "memory_available_bytes"),
                "end": boundary(-1, "memory_available_bytes"),
            },
        },
        "raw_artifacts": raw_artifacts,
        "aggregation": {
            "method": "median",
            "sample_count": baseline["workload"]["measured_requests"],
            "report_quantiles": ["p50", "p95", "p99"],
        },
        "result_class": "pending",
    }
    validate_document(manifest)
    return manifest


__all__ = [
    "artifact",
    "build_run_manifest",
    "finite",
    "identity",
    "integer",
    "load_mapping",
    "normalize_requests",
    "raw_artifact_paths",
    "resource_evidence",
    "summary",
    "validate_baseline_semantics",
]
