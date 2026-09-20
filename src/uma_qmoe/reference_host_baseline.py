"""Normalize the fixed PyTorch/HF host into ReferenceHostBaseline v1."""

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


def build_reference_host_baseline(
    run_directory: str | Path,
    *,
    target_id: str,
    source_commit: str,
    backend: str,
    container_image: str,
    model_id: str,
    model_revision: str,
    derivation_semantic_sha256: str | None = None,
    weight_source_kind: str | None = None,
    weight_source_semantic_sha256: str | None = None,
    input_tokens: int,
    output_tokens: int,
    warmup_requests: int,
    measured_requests: int,
) -> dict[str, Any]:
    """Build evidence for the fixed manual-cache greedy generation host."""

    if weight_source_kind is None and weight_source_semantic_sha256 is None:
        if derivation_semantic_sha256 is None:
            raise ContractError(
                "reference host requires a derivation or weight-source identity"
            )
        schema_version = 1
        model_source = {
            "derivation_semantic_sha256": identity(
                derivation_semantic_sha256, "derivation_semantic_sha256", 64
            )
        }
    else:
        if derivation_semantic_sha256 is not None:
            raise ContractError(
                "reference host weight source cannot also claim a derivation"
            )
        if weight_source_kind not in {"model_derivation", "model_manifest"}:
            raise ContractError("unsupported reference host weight source kind")
        if weight_source_semantic_sha256 is None:
            raise ContractError("reference host weight source SHA-256 is required")
        schema_version = 2
        model_source = {
            "weight_source": {
                "kind": weight_source_kind,
                "semantic_sha256": identity(
                    weight_source_semantic_sha256,
                    "weight_source_semantic_sha256",
                    64,
                ),
            }
        }

    directory = Path(run_directory)
    raw = load_mapping(directory / "result.json")
    metadata = load_mapping(directory / "runner-metadata.json")
    if metadata.get("target_id") != target_id:
        raise ContractError("runner target_id does not match requested target")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractError("reference host metadata must contain runtime identity")
    torch_version = runtime.get("torch_version")
    transformers_version = runtime.get("transformers_version")
    backend_version = runtime.get("backend_version")
    if not all(
        isinstance(value, str) and value
        for value in (torch_version, transformers_version, backend_version)
    ):
        raise ContractError("reference host runtime versions must be non-empty strings")
    expected_runtime = {
        "source_commit": source_commit,
        "backend": backend,
        "container_image": container_image,
    }
    if any(runtime.get(name) != value for name, value in expected_runtime.items()):
        raise ContractError("reference host runtime identity does not match normalization")
    runner_model = metadata.get("model")
    if not isinstance(runner_model, dict) or (
        runner_model.get("model_id") != model_id
        or runner_model.get("model_revision") != model_revision
    ):
        raise ContractError("reference host model identity does not match normalization")
    runner_workload = metadata.get("workload")
    expected_workload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "max_concurrency": 1,
    }
    if not isinstance(runner_workload, dict) or any(
        runner_workload.get(name) != value
        for name, value in expected_workload.items()
    ):
        raise ContractError("reference host workload does not match normalization")
    if metadata.get("generation_loop") != "manual_past_key_values_greedy":
        raise ContractError("reference host must use the manual past_key_values greedy loop")

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
        "manual_cache_decode": True,
        "offline_local_model": metadata.get("offline_local_model") is True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": schema_version,
        "kind": "reference_host_baseline",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "implementation": {
            "name": "pytorch_hf_reference_host",
            "source_commit": identity(source_commit, "source_commit", 40),
            "backend": backend,
            "container_image": container_image,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "backend_version": backend_version,
        },
        "model": {
            "model_id": model_id,
            "revision": identity(model_revision, "model_revision", 40),
            "weight_dtype": "BF16",
            **model_source,
        },
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "warmup_requests": warmup_requests,
            "measured_requests": measured_requests,
            "max_concurrency": 1,
            "decoding": "greedy",
            "ignore_eos": True,
            "seed": 0,
        },
        "results": results,
        "memory": memory,
        "telemetry": telemetry,
        "raw_artifacts": [
            artifact(directory, path) for path in raw_artifact_paths(metadata)
        ],
        "gates": gates,
    }
    return document


def build_reference_host_run_manifest(
    baseline_path: str | Path,
    run_directory: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_run_manifest(
        baseline_path,
        run_directory,
        expected_kind="reference_host_baseline",
        **kwargs,
    )


__all__ = ["build_reference_host_baseline", "build_reference_host_run_manifest"]
