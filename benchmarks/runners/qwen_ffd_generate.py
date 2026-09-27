#!/usr/bin/env python3
"""Diagnostic greedy generation through the fail-closed Qwen FFD backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from uma_qmoe.ffd import FFDPolicy, Gfx1151TritonBackend, install_qwen_ffd
from uma_qmoe.native_backend import install_mixed_target_backend
from uma_qmoe.qwen_compressed_loader import load_fixed_qwen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--target-policy-id", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--native-build-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--delta", type=float, default=7)
    parser.add_argument("--block-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--collect-keep-stats", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("max-new-tokens must be positive")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from transformers import AutoTokenizer

    if not torch.version.hip:
        raise RuntimeError("Qwen FFD generation requires PyTorch HIP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to("cuda:0")
    if input_ids.numel() == 0:
        raise RuntimeError("prompt tokenized to zero tokens")
    if input_ids.shape[1] + args.max_new_tokens > 8192:
        raise RuntimeError("prompt + generation exceeds native 8192 positions")
    policy = FFDPolicy(
        policy_id=(
            f"qwen-ffd-q2-fp8-delta{args.delta:g}-bs{args.block_size}-diagnostic"
        ),
        delta=args.delta,
        block_size=args.block_size,
        sink_tokens=args.block_size,
        local_tokens=args.block_size,
        max_seq_len=8192,
        num_layers=24,
        require_fused_backend=True,
    )
    native = install_mixed_target_backend(build_directory=args.native_build_dir)
    backend = Gfx1151TritonBackend.build(collect_keep_stats=args.collect_keep_stats)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
        target_policy_id=args.target_policy_id,
    ) as host:
        installation = install_qwen_ffd(host.model, backend, policy)
        cache = installation.cache()
        try:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with torch.inference_mode():
                output = host.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter_ns() - started) / 1_000_000
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [int(next_token.item())]
            decode_times = []
            while len(generated) < args.max_new_tokens:
                started = time.perf_counter_ns()
                with torch.inference_mode():
                    output = host.model(
                        input_ids=next_token,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                torch.cuda.synchronize()
                decode_times.append((time.perf_counter_ns() - started) / 1_000_000)
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(int(next_token.item()))
                if generated[-1] == tokenizer.eos_token_id:
                    break
            evidence = {
                "schema_version": 1,
                "kind": "qwen_ffd_diagnostic_generation",
                "evidence_level": "L0",
                "deployment_claim_allowed": False,
                "deployment_rejection_reason": (
                    "FFD-A1 selector gate must pass before Host promotion"
                ),
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "ttft_ms": ttft_ms,
                "decode_ms": decode_times,
                "policy": policy.to_dict(),
                "ffd_audit": installation.audit(),
                "cache_memory": cache.memory_breakdown(),
                "native_platform": native.platform,
            }
        finally:
            installation.uninstall()
    rendered = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
