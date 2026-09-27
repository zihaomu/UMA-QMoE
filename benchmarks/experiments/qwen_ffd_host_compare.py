#!/usr/bin/env python3
"""Matched-process dense versus FFD decode benchmark on the local Qwen Host."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

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
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--delta", type=float, default=7.0)
    parser.add_argument("--block-size", type=int, choices=(64, 128), default=128)
    return parser


def _fixture_text(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("prompt") or row.get("text")
        if isinstance(value, str) and value:
            values.append(value)
    if not values:
        raise RuntimeError("prompt fixture has no text")
    return "\n\n".join(values)


def _input_ids(tokenizer: Any, text: str, length: int, torch: Any) -> Any:
    source = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not source:
        raise RuntimeError("prompt fixture tokenized to zero tokens")
    repeats = (length + len(source) - 1) // len(source)
    return torch.tensor(
        [(source * repeats)[:length]], dtype=torch.long, device="cuda:0"
    )


def _measure_call(torch: Any, function: Any) -> tuple[Any, float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start_event.record()
    result = function()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
    return result, float(start_event.elapsed_time(end_event)), wall_ms


def _decode(model: Any, token: Any, cache: Any, torch: Any) -> Any:
    with torch.inference_mode():
        return model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )


def _run(
    model: Any,
    input_ids: Any,
    decode_tokens: int,
    torch: Any,
    *,
    cache: Any | None = None,
    timed: bool,
) -> dict[str, Any]:
    kwargs = {"use_cache": True, "return_dict": True}
    if cache is not None:
        kwargs["past_key_values"] = cache

    def prefill() -> Any:
        with torch.inference_mode():
            return model(input_ids=input_ids, **kwargs)

    if timed:
        output, prefill_device_ms, prefill_wall_ms = _measure_call(torch, prefill)
    else:
        output = prefill()
        torch.cuda.synchronize()
        prefill_device_ms = prefill_wall_ms = 0.0
    active_cache = cache if cache is not None else output.past_key_values
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    device_ms = []
    wall_ms = []
    for _ in range(decode_tokens):
        if timed:
            output, step_device_ms, step_wall_ms = _measure_call(
                torch,
                lambda: _decode(model, next_token, active_cache, torch),
            )
            device_ms.append(step_device_ms)
            wall_ms.append(step_wall_ms)
        else:
            output = _decode(model, next_token, active_cache, torch)
            torch.cuda.synchronize()
        if cache is None:
            active_cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    result = {
        "generated_token_ids": generated,
        "prefill_device_ms": prefill_device_ms,
        "prefill_wall_ms": prefill_wall_ms,
        "decode_device_ms": device_ms,
        "decode_wall_ms": wall_ms,
    }
    if timed:
        result.update(
            {
                "median_device_tpot_ms": statistics.median(device_ms),
                "mean_device_tpot_ms": statistics.fmean(device_ms),
                "median_wall_tpot_ms": statistics.median(wall_ms),
                "mean_wall_tpot_ms": statistics.fmean(wall_ms),
            }
        )
    return result


def _release(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:
    args = _parser().parse_args()
    if args.context_length <= 0 or args.decode_tokens <= 0:
        raise RuntimeError("context-length and decode-tokens must be positive")
    if args.context_length + args.decode_tokens > 8192:
        raise RuntimeError("context + decode must not exceed native 8192 positions")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    import transformers
    from transformers import AutoTokenizer

    if not torch.version.hip:
        raise RuntimeError("matched FFD benchmark requires PyTorch HIP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    input_ids = _input_ids(
        tokenizer, _fixture_text(args.prompt_fixture), args.context_length, torch
    )
    policy = FFDPolicy(
        policy_id=f"qwen-ffd-relaxed-d{args.delta:g}-bs{args.block_size}",
        delta=args.delta,
        block_size=args.block_size,
        sink_tokens=args.block_size,
        local_tokens=args.block_size,
        max_seq_len=8192,
        num_layers=24,
        require_fused_backend=True,
    )
    native = install_mixed_target_backend(build_directory=args.native_build_dir)
    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
        target_policy_id=args.target_policy_id,
    ) as host:
        # Warm both paths at the measured shape. Triton compilation is therefore
        # outside the reported steady-state decode samples.
        _run(host.model, input_ids, 1, torch, timed=False)
        _release(torch)
        warm_backend = Gfx1151TritonBackend.build()
        warm_installation = install_qwen_ffd(host.model, warm_backend, policy)
        try:
            warm_cache = warm_installation.cache()
            _run(host.model, input_ids, 1, torch, cache=warm_cache, timed=False)
            del warm_cache
        finally:
            warm_installation.uninstall()
        del warm_backend
        _release(torch)

        torch.cuda.reset_peak_memory_stats()
        dense = _run(host.model, input_ids, args.decode_tokens, torch, timed=True)
        dense_peak = int(torch.cuda.max_memory_allocated())
        _release(torch)

        backend = Gfx1151TritonBackend.build()
        installation = install_qwen_ffd(host.model, backend, policy)
        try:
            cache = installation.cache()
            torch.cuda.reset_peak_memory_stats()
            ffd = _run(
                host.model,
                input_ids,
                args.decode_tokens,
                torch,
                cache=cache,
                timed=True,
            )
            ffd_peak = int(torch.cuda.max_memory_allocated())
            ffd_cache_memory = cache.memory_breakdown()
            audit = installation.audit()
            del cache
        finally:
            installation.uninstall()

    dense_median = dense["median_wall_tpot_ms"]
    ffd_median = ffd["median_wall_tpot_ms"]
    token_pairs = zip(
        dense["generated_token_ids"], ffd["generated_token_ids"], strict=True
    )
    agreement = [left == right for left, right in token_pairs]
    result = {
        "schema_version": 1,
        "kind": "qwen_ffd_relaxed_accuracy_host_comparison",
        "evidence_level": "L0",
        "deployment_claim_allowed": False,
        "reason": "performance exploration after explicitly relaxing accuracy",
        "context_tokens": args.context_length,
        "decode_tokens": args.decode_tokens,
        "policy": policy.to_dict(),
        "dense": dense,
        "ffd": ffd,
        "speedup_median_wall_tpot": dense_median / ffd_median,
        "tpot_reduction_fraction": 1.0 - ffd_median / dense_median,
        "token_agreement": agreement,
        "token_agreement_fraction": sum(agreement) / len(agreement),
        "first_token_divergence_index": next(
            (index for index, matches in enumerate(agreement) if not matches), None
        ),
        "dense_peak_allocated_bytes": dense_peak,
        "ffd_peak_allocated_bytes": ffd_peak,
        "ffd_cache_memory": ffd_cache_memory,
        "ffd_audit": audit,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "attention_backend": host.model.config._attn_implementation,
            "native_platform": native.platform,
        },
        "target_policy_id": args.target_policy_id,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
