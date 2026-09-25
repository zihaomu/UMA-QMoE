#!/usr/bin/env python3
"""Measure the real Qwen dense attention baseline on the local gfx1151 Host."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
from typing import Any

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
    parser.add_argument("--context-lengths", default="128,1024,4096,7168")
    parser.add_argument("--decode-tokens", type=int, default=8)
    return parser


def _read_text(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("prompt") or row.get("text")
        if isinstance(text, str) and text:
            values.append(text)
    if not values:
        raise RuntimeError("prompt fixture has no text")
    return "\n\n".join(values)


def _ids(tokenizer: Any, text: str, length: int, torch: Any) -> Any:
    source = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not source:
        raise RuntimeError("prompt fixture tokenized to zero tokens")
    repeats = (length + len(source) - 1) // len(source)
    return torch.tensor(
        [(source * repeats)[:length]], dtype=torch.long, device="cuda:0"
    )


def _proc_memory() -> dict[str, int]:
    fields = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"VmRSS", "VmSwap", "VmHWM"}:
            fields[name] = int(value.strip().split()[0]) * 1024
    cgroup = Path("/sys/fs/cgroup/memory.current")
    fields["cgroup_current"] = (
        int(cgroup.read_text().strip()) if cgroup.is_file() else -1
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    fields["minor_faults"] = int(usage.ru_minflt)
    fields["major_faults"] = int(usage.ru_majflt)
    return fields


def _elapsed(event_pairs: list[tuple[Any, Any]]) -> float:
    return sum(float(start.elapsed_time(end)) for start, end in event_pairs)


def main() -> int:
    args = _parser().parse_args()
    lengths = [int(item) for item in args.context_lengths.split(",") if item]
    if not lengths or any(length <= 0 for length in lengths):
        raise RuntimeError("context lengths must be positive")
    if args.decode_tokens <= 0:
        raise RuntimeError("decode-tokens must be positive")
    if any(length + args.decode_tokens > 8192 for length in lengths):
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
        raise RuntimeError("Qwen dense baseline requires PyTorch HIP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    fixture_text = _read_text(args.prompt_fixture)
    native = install_mixed_target_backend(build_directory=args.native_build_dir)
    rows = []
    phase = {"name": "idle"}
    events: dict[str, dict[str, list[tuple[Any, Any]]]] = {}
    pending: dict[int, Any] = {}

    def before(module: Any, _inputs: Any) -> None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        pending[id(module)] = start

    def after(kind: str):
        def hook(module: Any, _inputs: Any, _output: Any) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = pending.pop(id(module))
            events.setdefault(phase["name"], {}).setdefault(kind, []).append(
                (start, end)
            )

        return hook

    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=True,
        target_policy_id=args.target_policy_id,
    ) as host:
        handles = []
        for layer in host.model.model.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(before))
            handles.append(layer.self_attn.register_forward_hook(after("attention")))
            handles.append(layer.mlp.register_forward_pre_hook(before))
            handles.append(layer.mlp.register_forward_hook(after("mlp")))
        try:
            for context_length in lengths:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                events.clear()
                memory_before = _proc_memory()
                input_ids = _ids(tokenizer, fixture_text, context_length, torch)
                phase["name"] = "prefill"
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    output = host.model(
                        input_ids=input_ids, use_cache=True, return_dict=True
                    )
                end.record()
                end.synchronize()
                prefill_ms = float(start.elapsed_time(end))
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                decode_ms = []
                for index in range(args.decode_tokens):
                    phase["name"] = f"decode-{index}"
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.inference_mode():
                        output = host.model(
                            input_ids=next_token,
                            past_key_values=cache,
                            use_cache=True,
                            return_dict=True,
                        )
                    end.record()
                    end.synchronize()
                    decode_ms.append(float(start.elapsed_time(end)))
                    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    cache = output.past_key_values
                torch.cuda.synchronize()
                attention_decode = []
                mlp_decode = []
                other_decode = []
                for index, total in enumerate(decode_ms):
                    values = events[f"decode-{index}"]
                    attention = _elapsed(values.get("attention", []))
                    mlp = _elapsed(values.get("mlp", []))
                    attention_decode.append(attention)
                    mlp_decode.append(mlp)
                    other_decode.append(max(0.0, total - attention - mlp))
                memory_after = _proc_memory()
                rows.append(
                    {
                        "schema_version": 1,
                        "kind": "ffd_dense_attention_baseline",
                        "evidence_level": "L0",
                        "context_tokens": context_length,
                        "decode_tokens": args.decode_tokens,
                        "attention_backend": host.model.config._attn_implementation,
                        "shape": {
                            "layers": int(host.model.config.num_hidden_layers),
                            "query_heads": int(host.model.config.num_attention_heads),
                            "kv_heads": int(host.model.config.num_key_value_heads),
                            "head_dim": int(
                                host.model.config.hidden_size
                                // host.model.config.num_attention_heads
                            ),
                        },
                        "prefill_ms": prefill_ms,
                        "prefill_attention_ms": _elapsed(
                            events["prefill"].get("attention", [])
                        ),
                        "prefill_mlp_ms": _elapsed(events["prefill"].get("mlp", [])),
                        "decode": {
                            "per_token_ms": decode_ms,
                            "median_tpot_ms": statistics.median(decode_ms),
                            "mean_tpot_ms": statistics.fmean(decode_ms),
                            "attention_ms": attention_decode,
                            "attention_fraction": (
                                sum(attention_decode) / sum(decode_ms)
                            ),
                            "mlp_ms": mlp_decode,
                            "mlp_fraction": sum(mlp_decode) / sum(decode_ms),
                            "other_ms": other_decode,
                        },
                        "memory_before": memory_before,
                        "memory_after": memory_after,
                        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
                        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
                        "torch_peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated()
                        ),
                        "torch_peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved()
                        ),
                        "runtime": {
                            "torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "hip": torch.version.hip,
                            "device": torch.cuda.get_device_name(0),
                            "native_platform": native.platform,
                        },
                        "target_policy_id": args.target_policy_id,
                    }
                )
                del output, cache, input_ids, next_token
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            for handle in handles:
                handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
