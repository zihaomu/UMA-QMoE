#!/usr/bin/env python3
"""Run one user prompt through the validated local-halo Qwen MVP host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

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
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
        raise RuntimeError("local-halo generation requires the HIP runtime")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if encoded["input_ids"].numel() == 0:
        raise RuntimeError("prompt tokenized to zero tokens")
    input_ids = encoded["input_ids"].to("cuda:0")
    native = install_mixed_target_backend(build_directory=args.native_build_dir)
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
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = host.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        generated = [int(next_token.item())]
        eos_token_id = tokenizer.eos_token_id
        decode_started = time.perf_counter_ns()
        while len(generated) < args.max_new_tokens and generated[-1] != eos_token_id:
            with torch.inference_mode():
                output = host.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = output.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                past_key_values = output.past_key_values
            generated.append(int(next_token.item()))
        torch.cuda.synchronize()
        decode_seconds = (time.perf_counter_ns() - decode_started) / 1_000_000_000.0
        cache = native.cache_summary()
        loader = dict(host.evidence)
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_token_ids": generated,
                "input_tokens": int(input_ids.shape[-1]),
                "output_tokens": len(generated),
                "ttft_ms": ttft_ms,
                "decode_tokens_per_second": (
                    max(0, len(generated) - 1) / decode_seconds
                    if decode_seconds > 0
                    else 0.0
                ),
                "native_platform": native.platform,
                "expert_pack_mapping_count": loader["expert_pack_mapping_count"],
                "compressed_cache_bytes": cache["device_storage_bytes"],
                "dequantized_weight_bytes": cache["dequantized_weight_bytes"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
