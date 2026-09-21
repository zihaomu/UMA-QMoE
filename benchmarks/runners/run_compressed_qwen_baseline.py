#!/usr/bin/env python3
"""Run the fixed Qwen packed-Q4 host with deterministic greedy decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from pytorch_reference_host import (
    _cgroup_memory_peak,
    _exact_token_ids,
    _generate_once,
    _prompt,
    _resource_sample,
    _sha256,
    _telemetry_probe,
    _utc_now,
)
from uma_qmoe.fixed_models import QWEN1_5_MOE
from uma_qmoe.native_backend import (
    install_mixed_target_backend,
    install_packed_q4_backend,
    native_kernel_source_sha256,
)
from uma_qmoe.qwen_compressed_loader import load_fixed_qwen


def _cgroup_metric(name: str) -> int | None:
    try:
        value = (Path("/sys/fs/cgroup") / name).read_text(encoding="utf-8").strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _runtime_source_identity() -> tuple[str, list[dict[str, object]]]:
    project_root = Path(__file__).resolve().parents[2]
    names = (
        "benchmarks/runners/pytorch_reference_host.py",
        "benchmarks/runners/run_compressed_qwen_baseline.py",
        "src/uma_qmoe/custom_op.py",
        "src/uma_qmoe/expert_pack.py",
        "src/uma_qmoe/native/packed_q4_binding.cpp",
        "src/uma_qmoe/native/packed_q4_kernel.cu",
        "src/uma_qmoe/native_backend.py",
        "src/uma_qmoe/qwen_compressed_loader.py",
        "src/uma_qmoe/target_pack.py",
    )
    combined = hashlib.sha256()
    records: list[dict[str, object]] = []
    for name in names:
        path = project_root / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        records.append(
            {"path": name, "size_bytes": len(payload), "sha256": digest}
        )
    return combined.hexdigest(), records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("local-halo", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--target-policy-id")
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=10)
    parser.add_argument("--native-build-dir", type=Path)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if min(
        args.input_tokens,
        args.output_tokens,
        args.warmup_requests,
        args.measured_requests,
    ) <= 0:
        raise SystemExit("all workload counts must be positive")
    if args.output_tokens < 2:
        raise SystemExit("output-tokens must be at least 2")
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(
        (output_directory / name).exists()
        for name in ("result.json", "runner-metadata.json")
    ):
        raise SystemExit("refusing to overwrite Qwen compressed host evidence")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started_at = _utc_now()
    source_state_sha256, source_files = _runtime_source_identity()
    swap_before = _cgroup_metric("memory.swap.current")
    major_faults_before = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    probe = _telemetry_probe()
    samples = [_resource_sample(probe, "before")]

    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen compressed host requires a BF16 device")
    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"compressed host backend mismatch: requested {args.backend}, "
            f"observed {observed_backend}"
        )
    expected_platform = {
        "local-halo": "hip_gfx1151",
        "spark1": "cuda_sm121",
    }[args.target_id]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    token_ids = _exact_token_ids(tokenizer, prompt, args.input_tokens)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    backend_installer = (
        install_mixed_target_backend
        if args.target_policy_id
        else install_packed_q4_backend
    )
    native = backend_installer(
        build_directory=args.native_build_dir, verbose=args.verbose_build
    )
    if native.platform != expected_platform:
        raise RuntimeError(
            f"compressed host expected {expected_platform}, got {native.platform}"
        )
    torch.cuda.reset_peak_memory_stats()
    pack_sha256 = _sha256(args.expert_pack)
    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
        target_policy_id=args.target_policy_id,
    ) as host:
        for _ in range(args.warmup_requests):
            _generate_once(
                host.model,
                input_ids,
                output_tokens=args.output_tokens,
                torch=torch,
            )
        torch.cuda.synchronize()
        cache_after_warmup = native.cache_summary()
        measurement_faults_before = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_majflt
        measurement_swap_before = _cgroup_metric("memory.swap.current")
        measured_started = time.perf_counter_ns()
        requests = [
            _generate_once(
                host.model,
                input_ids,
                output_tokens=args.output_tokens,
                torch=torch,
            )
            for _ in range(args.measured_requests)
        ]
        torch.cuda.synchronize()
        duration_seconds = (
            time.perf_counter_ns() - measured_started
        ) / 1_000_000_000.0
        measurement_faults_after = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_majflt
        measurement_swap_after = _cgroup_metric("memory.swap.current")
        cache_after_measurement = native.cache_summary()
        loader = dict(host.evidence)
        torch_peak_allocated = int(torch.cuda.max_memory_allocated())
        torch_peak_reserved = int(torch.cuda.max_memory_reserved())
        cgroup_peak = _cgroup_memory_peak()
    samples.append(_resource_sample(probe, "after"))
    swap_after = _cgroup_metric("memory.swap.current")
    swap_peak = _cgroup_metric("memory.swap.peak")
    major_faults_after = resource.getrusage(resource.RUSAGE_SELF).ru_majflt

    request_durations_ms = [
        request["ttft_ms"] + sum(request["inter_token_latencies_ms"])
        for request in requests
    ]
    ttft_ms = [request["ttft_ms"] for request in requests]
    tpot_ms = [
        latency
        for request in requests
        for latency in request["inter_token_latencies_ms"]
    ]
    generated_tokens = sum(len(request["generated_token_ids"]) for request in requests)
    unique_outputs = len(
        {tuple(request["generated_token_ids"]) for request in requests}
    )
    error_count = sum(request["error"] is not None for request in requests)
    result = {
        "schema_version": 1,
        "duration_seconds": duration_seconds,
        "summary": {
            "generated_tokens": generated_tokens,
            "aggregate_tokens_per_second": generated_tokens / duration_seconds,
            "request_duration_ms": {
                "mean": statistics.mean(request_durations_ms),
                "p50": _quantile(request_durations_ms, 0.50),
                "p95": _quantile(request_durations_ms, 0.95),
                "p99": _quantile(request_durations_ms, 0.99),
                "coefficient_of_variation": (
                    statistics.stdev(request_durations_ms)
                    / statistics.mean(request_durations_ms)
                    if len(request_durations_ms) > 1
                    else 0.0
                ),
            },
            "ttft_ms": {
                "p50": _quantile(ttft_ms, 0.50),
                "p95": _quantile(ttft_ms, 0.95),
                "p99": _quantile(ttft_ms, 0.99),
            },
            "tpot_ms": {
                "p50": _quantile(tpot_ms, 0.50),
                "p95": _quantile(tpot_ms, 0.95),
                "p99": _quantile(tpot_ms, 0.99),
            },
            "error_count": error_count,
            "unique_output_count": unique_outputs,
        },
        "requests": requests,
        "prompt": {
            "fixture_sha256": _sha256(args.prompt_fixture),
            "prompt_id": args.prompt_id,
            "construction": "repeat_tokenized_fixture_then_truncate",
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "token_ids": token_ids,
        },
    }
    (output_directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "kind": "qwen_compressed_host_evidence",
        "target_id": args.target_id,
        "status": "succeeded",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "runner_argv": [sys.executable, *sys.argv],
        "offline_local_model": True,
        "generation_loop": "manual_past_key_values_greedy",
        "runtime": {
            "source_commit": args.source_commit,
            "source_state_sha256": source_state_sha256,
            "source_files": source_files,
            "backend": observed_backend,
            "platform": native.platform,
            "container_image": args.container_image,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "backend_version": torch.version.hip or torch.version.cuda,
            "native_source_sha256": native_kernel_source_sha256(),
            "fallback_count": 0,
        },
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "path": str(args.model.resolve()),
            "expert_pack_sha256": pack_sha256,
            "target_policy_id": args.target_policy_id,
        },
        "workload": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.measured_requests,
            "max_concurrency": 1,
        },
        "loader": loader,
        "backend_cache": {
            "after_warmup": cache_after_warmup,
            "after_measurement": cache_after_measurement,
        },
        "cgroup_memory_peak_bytes": cgroup_peak,
        "cgroup_swap_current_before_bytes": swap_before,
        "cgroup_swap_current_after_bytes": swap_after,
        "cgroup_swap_peak_bytes": swap_peak,
        "major_page_faults_delta": major_faults_after - major_faults_before,
        "measurement_major_page_faults_delta": (
            measurement_faults_after - measurement_faults_before
        ),
        "measurement_swap_current_before_bytes": measurement_swap_before,
        "measurement_swap_current_after_bytes": measurement_swap_after,
        "torch_peak_allocated_bytes": torch_peak_allocated,
        "torch_peak_reserved_bytes": torch_peak_reserved,
        "samples": samples,
        "gates": {
            "requests_succeeded": error_count == 0,
            "deterministic_outputs": unique_outputs == 1,
            "statistics_contract_satisfied": (
                args.warmup_requests >= 3 and args.measured_requests >= 10
            ),
            "cache_stable_during_measurement": (
                cache_after_warmup == cache_after_measurement
            ),
            "no_dequantized_weight_cache": (
                cache_after_measurement["dequantized_weight_bytes"] == 0
            ),
            "no_measurement_swap": (
                measurement_swap_before == 0 and measurement_swap_after == 0
            ),
            "no_measurement_major_page_faults": (
                measurement_faults_after == measurement_faults_before
            ),
            "end_to_end_cv_at_most_0_05": (
                result["summary"]["request_duration_ms"][
                    "coefficient_of_variation"
                ]
                <= 0.05
            ),
        },
    }
    metadata["gates"]["overall_passed"] = all(metadata["gates"].values())
    (output_directory / "runner-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
