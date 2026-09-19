"""Normalize the packed native OLMoE host into CompressedHostBaseline v1/v2."""

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
from .contracts import ContractError, validate_document


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_compressed_host_baseline(
    run_directory: str | Path,
    *,
    target_id: str,
    source_commit: str,
    backend: str,
    container_image: str,
    model_id: str,
    model_revision: str,
    expert_pack_sha256: str,
    target_policy_id: str | None = None,
    input_tokens: int,
    output_tokens: int,
    warmup_requests: int,
    measured_requests: int,
    safe_uma_budget_bytes: int = 32 * 1024**3,
) -> dict[str, Any]:
    directory = Path(run_directory)
    raw = load_mapping(directory / "result.json")
    metadata = load_mapping(directory / "runner-metadata.json")
    if metadata.get("target_id") != target_id:
        raise ContractError("compressed runner target_id does not match requested target")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict):
        raise ContractError("compressed runner metadata must contain runtime identity")
    expected_runtime = {
        "source_commit": source_commit,
        "backend": backend,
        "container_image": container_image,
    }
    if any(runtime.get(name) != value for name, value in expected_runtime.items()):
        raise ContractError("compressed runner runtime identity does not match normalization")
    for name in (
        "torch_version",
        "transformers_version",
        "backend_version",
        "platform",
        "native_source_sha256",
    ):
        if not isinstance(runtime.get(name), str) or not runtime[name]:
            raise ContractError(f"compressed runner runtime {name} is missing")
    expected_platform = {"halo3": "hip_gfx1151", "spark1": "cuda_sm121"}[target_id]
    if runtime["platform"] != expected_platform:
        raise ContractError("compressed runner native platform does not match target")
    runner_model = metadata.get("model")
    if not isinstance(runner_model, dict) or any(
        runner_model.get(name) != value
        for name, value in {
            "model_id": model_id,
            "model_revision": model_revision,
            "expert_pack_sha256": expert_pack_sha256,
        }.items()
    ):
        raise ContractError("compressed runner model identity does not match normalization")
    if runner_model.get("target_policy_id") != target_policy_id:
        raise ContractError("compressed runner TargetPack policy does not match normalization")
    expected_workload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "max_concurrency": 1,
    }
    runner_workload = metadata.get("workload")
    if not isinstance(runner_workload, dict) or any(
        runner_workload.get(name) != value
        for name, value in expected_workload.items()
    ):
        raise ContractError("compressed runner workload does not match normalization")
    if metadata.get("generation_loop") != "manual_past_key_values_greedy":
        raise ContractError("compressed host must use the manual cache decode loop")
    loader = metadata.get("loader")
    backend_cache = metadata.get("backend_cache")
    if not isinstance(loader, dict) or not isinstance(backend_cache, dict):
        raise ContractError("compressed runner loader/cache evidence is missing")

    results, request_gates = normalize_requests(
        raw,
        expected_input_tokens=input_tokens,
        expected_output_tokens=output_tokens,
        measured_requests=measured_requests,
    )
    memory, telemetry = resource_evidence(metadata)
    after_warmup = backend_cache.get("after_warmup")
    after_measurement = backend_cache.get("after_measurement")
    if not isinstance(after_warmup, dict) or not isinstance(after_measurement, dict):
        raise ContractError("compressed runner cache snapshots are missing")
    gates = {
        "runner_succeeded": metadata.get("status") == "succeeded"
        and metadata.get("returncode") == 0,
        **request_gates,
        "manual_cache_decode": True,
        "offline_local_model": metadata.get("offline_local_model") is True,
        "packed_backend_registered": runtime["platform"] == expected_platform,
        "performance_mode_executed": loader.get("performance_mode") is True,
        "no_silent_fallback": runtime.get("fallback_count") == 0,
        "cache_stable_after_warmup": after_warmup == after_measurement,
        "no_dequantized_weight_cache": after_measurement.get(
            "dequantized_weight_bytes"
        )
        == 0,
        "dense_only_checkpoint_load": loader.get("loaded_expert_tensor_count") == 0,
        "no_expert_parameters": loader.get("expert_parameter_count") == 0,
        "single_pack_mapping": loader.get("expert_pack_mapping_count") == 1,
        "within_safe_uma_budget": memory["peak_bytes"] <= safe_uma_budget_bytes,
    }
    gates["overall_passed"] = all(gates.values())
    model = {
        "model_id": model_id,
        "revision": identity(model_revision, "model_revision", 40),
        "dense_weight_dtype": "BF16",
        "expert_quantization": (
            "mixed_target_pack_q4_q8_bf16"
            if target_policy_id
            else "canonical_q4_group128"
        ),
        "expert_pack_sha256": identity(
            expert_pack_sha256, "expert_pack_sha256", 64
        ),
    }
    if target_policy_id:
        model["target_policy_id"] = target_policy_id
    document = {
        "schema_version": 2 if target_policy_id else 1,
        "kind": "compressed_host_baseline",
        "generated_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "implementation": {
            "name": "uma_qmoe_compressed_host",
            "source_commit": identity(source_commit, "source_commit", 40),
            "backend": backend,
            "platform": runtime["platform"],
            "container_image": container_image,
            "torch_version": runtime["torch_version"],
            "transformers_version": runtime["transformers_version"],
            "backend_version": runtime["backend_version"],
            "native_source_sha256": identity(
                runtime["native_source_sha256"], "native_source_sha256", 64
            ),
            "fallback_count": runtime["fallback_count"],
        },
        "model": model,
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
        "loader": loader,
        "backend_cache": backend_cache,
        "memory": memory,
        "resource_policy": {"safe_uma_budget_bytes": safe_uma_budget_bytes},
        "telemetry": telemetry,
        "raw_artifacts": [
            artifact(directory, path) for path in raw_artifact_paths(metadata)
        ],
        "gates": gates,
    }
    validate_document(document)
    return document


def build_compressed_host_run_manifest(
    baseline_path: str | Path,
    run_directory: str | Path,
    *,
    route_trace_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    baseline = load_mapping(baseline_path)
    document = build_run_manifest(
        baseline_path,
        run_directory,
        expected_kind="compressed_host_baseline",
        **kwargs,
    )
    document["inputs"]["route_trace_sha256"] = identity(
        route_trace_sha256, "route_trace_sha256", 64
    )
    document["inputs"]["expert_pack_sha256"] = baseline["model"][
        "expert_pack_sha256"
    ]
    validate_document(document)
    return document


__all__ = [
    "build_compressed_host_baseline",
    "build_compressed_host_run_manifest",
]
