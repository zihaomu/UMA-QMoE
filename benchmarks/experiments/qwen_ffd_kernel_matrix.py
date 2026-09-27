#!/usr/bin/env python3
"""Run gfx1151 FFD microbenchmarks with real Qwen attention shapes."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable

from uma_qmoe.ffd import (
    FFDCache,
    FFDPolicy,
    Gfx1151TritonBackend,
    backend_source_sha256,
)


def _numbers(value: str, cast: Callable[[str], Any]) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _measure(
    torch: Any, function: Callable[[], Any], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "cv": statistics.pstdev(samples) / statistics.fmean(samples),
    }


def _git_identity(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        commit = os.environ.get("UMA_QMOE_PROJECT_COMMIT", "")
        dirty_hash = os.environ.get("UMA_QMOE_DIRTY_STATE_SHA256", "")
        if len(commit) != 40 or len(dirty_hash) != 64:
            raise RuntimeError(
                "containerized runs require UMA_QMOE_PROJECT_COMMIT and "
                "UMA_QMOE_DIRTY_STATE_SHA256 when Git worktree metadata is not mounted"
            )
        return {"commit": commit, "dirty": True, "dirty_state_sha256": dirty_hash}
    digest = hashlib.sha256(status.encode("utf-8") + b"\0" + diff)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    )
    for raw in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(raw)
        digest.update(b"\0")
        digest.update((root / raw.decode("utf-8")).read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_state_sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-lens", default="128,1024,4096,7936")
    parser.add_argument("--deltas", default="5,7")
    parser.add_argument("--block-sizes", default="64,128,256")
    parser.add_argument("--key-bits", default="2,4")
    parser.add_argument("--residual-dtypes", default="fp8_e4m3fn,none")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--verify-oracle", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 2:
        raise SystemExit("warmup must be >=1 and iterations must be >=2")
    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Qwen FFD kernel matrix requires PyTorch HIP")
    properties = torch.cuda.get_device_properties(0)
    if str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] != "gfx1151":
        raise RuntimeError("Qwen FFD kernel matrix is restricted to gfx1151")
    root = Path(__file__).resolve().parents[2]
    identity = _git_identity(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sequence_lengths = _numbers(args.seq_lens, int)
    deltas = _numbers(args.deltas, float)
    block_sizes = _numbers(args.block_sizes, int)
    key_bits_values = _numbers(args.key_bits, int)
    residual_dtypes = _numbers(args.residual_dtypes, str)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with args.output.open("a", encoding="utf-8") as stream:
        for sequence_length in sequence_lengths:
            if sequence_length <= 0 or sequence_length > 8192:
                raise RuntimeError("sequence lengths must be in [1, 8192]")
            q = torch.randn(
                (1, 16, 1, 128),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            key = torch.randn(
                (1, 16, sequence_length, 128),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            value = torch.randn(
                (1, 16, sequence_length, 128),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )

            def dense() -> Any:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, key, value, dropout_p=0.0, is_causal=False
                )

            dense_samples = _measure(torch, dense, args.warmup, args.iterations)
            dense_output = dense()
            torch.cuda.synchronize()
            for block_size, delta, key_bits, residual_dtype in product(
                block_sizes, deltas, key_bits_values, residual_dtypes
            ):
                policy_fields = {
                    "delta": delta,
                    "key_bits": key_bits,
                    "residual_dtype": residual_dtype,
                    "block_size": block_size,
                }
                common = {
                    "schema_version": 1,
                    "kind": "ffd_kernel_trial",
                    "evidence_level": "L0",
                    "workload_source": "synthetic_random_qwen_shape",
                    "observed_a1_keep_distribution": False,
                    "project": identity,
                    "backend_source_sha256": backend_source_sha256(),
                    "runtime": {
                        "torch": torch.__version__,
                        "hip": torch.version.hip,
                        "triton": __import__("triton").__version__,
                        "device": torch.cuda.get_device_name(0),
                        "architecture": "gfx1151",
                    },
                    "shape": {
                        "batch": 1,
                        "tokens": sequence_length,
                        "query_heads": 16,
                        "kv_heads": 16,
                        "head_dim": 128,
                        "value_dim": 128,
                    },
                    "policy": policy_fields,
                    "dense": _summary(dense_samples),
                }
                if block_size not in (64, 128):
                    row = {
                        **common,
                        "status": "rejected",
                        "performance_claim_allowed": False,
                        "reason": "gfx1151 fused v1 supports block size 64 or 128",
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    continue
                if key_bits != 2:
                    row = {
                        **common,
                        "status": "rejected",
                        "performance_claim_allowed": False,
                        "reason": "gfx1151 fused v1 supports Q2 scan; Q4 remains an oracle control",
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    continue
                policy = FFDPolicy(
                    policy_id=(
                        f"qwen-ffd-q{key_bits}-{residual_dtype}-delta{delta:g}-bs{block_size}-matrix"
                    ),
                    delta=delta,
                    key_bits=key_bits,
                    residual_dtype=residual_dtype,
                    block_size=block_size,
                    sink_tokens=block_size,
                    local_tokens=block_size,
                    max_seq_len=8192,
                    num_layers=1,
                    require_fused_backend=True,
                )
                cache = FFDCache(policy)
                torch.cuda.synchronize()
                build_started = time.perf_counter_ns()
                cache.update(
                    key,
                    value,
                    0,
                    {"cache_position": torch.arange(sequence_length, device="cuda")},
                )
                torch.cuda.synchronize()
                cache_build_ms = (time.perf_counter_ns() - build_started) / 1_000_000
                backend = Gfx1151TritonBackend.build()

                def sparse() -> Any:
                    return backend.decode(q, cache.layer(0), policy, layer_index=0)

                try:
                    sparse_samples = _measure(
                        torch, sparse, args.warmup, args.iterations
                    )
                    sparse_output = sparse().unsqueeze(2)
                    torch.cuda.synchronize()
                    difference = (sparse_output - dense_output).float()
                    sparse_summary = _summary(sparse_samples)
                    oracle_comparison = None
                    if args.verify_oracle:
                        from uma_qmoe.ffd.oracle import q2_sparse_attention

                        layer_cache = cache.layer(0)
                        if layer_cache.full_token_count:
                            oracle_output, selector = q2_sparse_attention(
                                q,
                                layer_cache.quantized_blocks(),
                                layer_cache.full_values(),
                                delta=policy.delta,
                                sink_tokens=policy.sink_tokens,
                                local_tokens=policy.local_tokens,
                                tail_keys=layer_cache.tail_keys(),
                                tail_values=layer_cache.tail_values(),
                            )
                        else:
                            from uma_qmoe.ffd.oracle import dense_attention

                            oracle_output = dense_attention(
                                q,
                                layer_cache.tail_keys(),
                                layer_cache.tail_values(),
                            )
                            selector = {"tail_only": True}
                        oracle_difference = (
                            sparse_output[:, :, 0].float() - oracle_output.float()
                        )
                        oracle_comparison = {
                            "max_abs_error": float(
                                oracle_difference.abs().max().item()
                            ),
                            "mean_abs_error": float(
                                oracle_difference.abs().mean().item()
                            ),
                            "cosine": float(
                                torch.nn.functional.cosine_similarity(
                                    sparse_output.float().flatten(),
                                    oracle_output.float().flatten(),
                                    dim=0,
                                ).item()
                            ),
                            "selector": selector,
                        }
                    row = {
                        **common,
                        "status": "passed",
                        "performance_claim_allowed": False,
                        "performance_claim_rejection_reason": (
                            "synthetic smoke does not use an A1-passed observed keep distribution"
                        ),
                        "cache_build_ms": cache_build_ms,
                        "ffd": sparse_summary,
                        "diagnostic_dense_ratio": (
                            common["dense"]["median_ms"] / sparse_summary["median_ms"]
                        ),
                        "dense_comparison": {
                            "max_abs_error": float(difference.abs().max().item()),
                            "mean_abs_error": float(difference.abs().mean().item()),
                            "cosine": float(
                                torch.nn.functional.cosine_similarity(
                                    sparse_output.float().flatten(),
                                    dense_output.float().flatten(),
                                    dim=0,
                                ).item()
                            ),
                        },
                        "policy_oracle_comparison": oracle_comparison,
                        "memory": cache.memory_breakdown()["totals"],
                    }
                except Exception as exc:
                    row = {
                        **common,
                        "status": "error",
                        "performance_claim_allowed": False,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
            torch.cuda.empty_cache()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
