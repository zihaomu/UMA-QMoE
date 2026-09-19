#!/usr/bin/env python3
"""Run the packed native UMA-QMoE host with fixed 128+32 greedy decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from pytorch_reference_host import (
    MODEL_ID,
    MODEL_REVISION,
    _cgroup_memory_peak,
    _exact_token_ids,
    _generate_once,
    _prompt,
    _resource_sample,
    _sha256,
    _telemetry_probe,
    _utc_now,
)
from uma_qmoe.compressed_loader import load_fixed_olmoe
from uma_qmoe.native_backend import (
    install_packed_q4_backend,
    native_kernel_source_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
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
        raise SystemExit("output-tokens must be at least 2 for TPOT evidence")
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_files = ("result.json", "runner-metadata.json")
    if any((output_directory / name).exists() for name in output_files):
        raise SystemExit("refusing to overwrite compressed host evidence")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started_at = _utc_now()
    probe = _telemetry_probe()
    samples = [_resource_sample(probe, "before")]

    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("compressed host requires a BF16 CUDA/HIP device")
    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"compressed host backend mismatch: requested {args.backend}, "
            f"observed {observed_backend}"
        )
    expected_platform = {
        "halo3": "hip_gfx1151",
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
    native = install_packed_q4_backend(
        build_directory=args.native_build_dir,
        verbose=args.verbose_build,
    )
    if native.platform != expected_platform:
        raise RuntimeError(
            f"compressed host expected {expected_platform}, got {native.platform}"
        )
    torch.cuda.reset_peak_memory_stats()
    pack_sha256 = _sha256(args.expert_pack)
    with load_fixed_olmoe(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
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
        cache_after_measurement = native.cache_summary()
        loader = {
            key: host.evidence[key]
            for key in (
                "performance_mode",
                "dense_tensor_count",
                "dense_tensor_bytes",
                "skipped_expert_tensor_count",
                "loaded_expert_tensor_count",
                "expert_parameter_count",
                "quantized_moe_block_count",
                "model_parameter_bytes",
                "expert_pack_size_bytes",
                "expert_pack_mapping_count",
                "expert_pack_vma_count",
            )
        }
        torch_peak_allocated = int(torch.cuda.max_memory_allocated())
        torch_peak_reserved = int(torch.cuda.max_memory_reserved())
        cgroup_peak = _cgroup_memory_peak()
    samples.append(_resource_sample(probe, "after"))

    result = {
        "schema_version": 1,
        "duration_seconds": duration_seconds,
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
    backend_version = torch.version.hip or torch.version.cuda
    metadata = {
        "schema_version": 1,
        "target_id": args.target_id,
        "status": "succeeded",
        "returncode": 0,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "runner_argv": [sys.executable, *sys.argv],
        "artifact_paths": ["result.json", "runner-metadata.json"],
        "offline_local_model": True,
        "generation_loop": "manual_past_key_values_greedy",
        "runtime": {
            "source_commit": args.source_commit,
            "backend": observed_backend,
            "platform": native.platform,
            "container_image": args.container_image,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "backend_version": backend_version,
            "native_source_sha256": native_kernel_source_sha256(),
            "fallback_count": 0,
        },
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "path": str(args.model.resolve()),
            "expert_pack_sha256": pack_sha256,
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
        "torch_peak_allocated_bytes": torch_peak_allocated,
        "torch_peak_reserved_bytes": torch_peak_reserved,
        "samples": samples,
    }
    (output_directory / "runner-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
