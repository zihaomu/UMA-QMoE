"""Normalize an isolated third-party runner into ExternalBaseline v1 evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline_common import (
    artifact,
    build_run_manifest,
    identity,
    load_mapping,
    normalize_requests,
    raw_artifact_paths,
    resource_evidence,
)
from .contracts import ContractError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_external_baseline(
    run_directory: str | Path,
    *,
    target_id: str,
    implementation_name: str,
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
    """Build runtime-neutral evidence from ``result.json`` and runner metadata."""

    directory = Path(run_directory)
    raw = load_mapping(directory / "result.json")
    metadata = load_mapping(directory / "runner-metadata.json")
    if metadata.get("target_id") != target_id:
        raise ContractError("runner target_id does not match requested target")
    isolation = metadata.get("isolation")
    if not isinstance(isolation, dict):
        raise ContractError("external runner metadata must declare isolation")
    boundary = isolation.get("execution_boundary")
    if boundary not in {"separate_process", "separate_container"}:
        raise ContractError("external runner must use a separate process or container")
    imports_core = isolation.get("imports_uma_qmoe_runtime")
    loads_pack = isolation.get("loads_uma_qmoe_expert_pack")
    if not isinstance(imports_core, bool) or not isinstance(loads_pack, bool):
        raise ContractError("external runner isolation flags must be boolean")
    if imports_core or loads_pack:
        raise ContractError(
            "external runner must not import UMA-QMoE runtime or load ExpertPack"
        )
    runtime = metadata.get("runtime")
    expected_runtime = {
        "implementation_name": implementation_name,
        "implementation_version": implementation_version,
        "source_commit": source_commit,
        "backend": backend,
        "container_image": container_image,
    }
    if not isinstance(runtime, dict) or any(
        runtime.get(name) != value for name, value in expected_runtime.items()
    ):
        raise ContractError("external runner identity does not match normalization")
    runner_model = metadata.get("model")
    if not isinstance(runner_model, dict) or (
        runner_model.get("model_id") != model_id
        or runner_model.get("model_revision") != model_revision
    ):
        raise ContractError("external runner model identity does not match normalization")
    runner_workload = metadata.get("workload")
    expected_workload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "max_concurrency": max_concurrency,
    }
    if not isinstance(runner_workload, dict) or any(
        runner_workload.get(name) != value
        for name, value in expected_workload.items()
    ):
        raise ContractError("external runner workload does not match normalization")

    results, request_gates = normalize_requests(
        raw,
        expected_input_tokens=input_tokens,
        expected_output_tokens=output_tokens,
        measured_requests=measured_requests,
    )
    memory, telemetry = resource_evidence(metadata)
    gates = {
        "runner_succeeded": metadata.get("status") == "succeeded"
        and metadata.get("returncode") == 0,
        **request_gates,
        "isolation_respected": not imports_core and not loads_pack,
    }
    gates["overall_passed"] = all(gates.values())
    raw_paths = raw_artifact_paths(metadata)
    document = {
        "schema_version": 1,
        "kind": "external_baseline",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "implementation": {
            "name": implementation_name,
            "version": implementation_version,
            "source_commit": identity(source_commit, "source_commit", 40),
            "backend": backend,
            "container_image": container_image,
        },
        "isolation": {
            "execution_boundary": boundary,
            "imports_uma_qmoe_runtime": imports_core,
            "loads_uma_qmoe_expert_pack": loads_pack,
        },
        "model": {
            "model_id": model_id,
            "revision": identity(model_revision, "model_revision", 40),
            "weight_dtype": "BF16",
            "derivation_semantic_sha256": identity(
                derivation_semantic_sha256, "derivation_semantic_sha256", 64
            ),
        },
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "warmup_requests": warmup_requests,
            "measured_requests": measured_requests,
            "max_concurrency": max_concurrency,
            "decoding": "greedy",
            "ignore_eos": True,
            "seed": 0,
        },
        "results": results,
        "memory": memory,
        "telemetry": telemetry,
        "raw_artifacts": [artifact(directory, path) for path in raw_paths],
        "gates": gates,
    }
    return document


def build_external_baseline_run_manifest(
    baseline_path: str | Path,
    run_directory: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_run_manifest(
        baseline_path,
        run_directory,
        expected_kind="external_baseline",
        **kwargs,
    )


__all__ = ["build_external_baseline", "build_external_baseline_run_manifest"]
