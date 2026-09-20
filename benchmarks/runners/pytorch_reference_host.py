#!/usr/bin/env python3
"""Run the fixed offline BF16 PyTorch/HF reference host.

This runner deliberately implements generation itself: one prefill followed by
single-token greedy decode calls carrying ``past_key_values``.  It is not a
service runtime and has no dependency on any external serving framework.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt(path: Path, prompt_id: str) -> str:
    matches: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("id") == prompt_id:
                matches.append(item.get("prompt"))
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise RuntimeError(f"prompt fixture must contain exactly one {prompt_id!r}")
    return matches[0]


def _exact_token_ids(tokenizer: Any, prompt: str, length: int) -> list[int]:
    seed = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not isinstance(seed, list) or not seed:
        raise RuntimeError("fixed prompt encoded to zero tokens")
    repeats = (length + len(seed) - 1) // len(seed)
    tokens = (seed * repeats)[:length]
    if len(tokens) != length or any(not isinstance(item, int) for item in tokens):
        raise RuntimeError("could not construct exact deterministic token fixture")
    return tokens


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def _cgroup_memory_peak() -> int:
    for path in (
        Path("/sys/fs/cgroup/memory.peak"),
        Path("/sys/fs/cgroup/memory.current"),
    ):
        value = _read_nonnegative_integer(path)
        if value is not None and value > 0:
            return value
    raise RuntimeError("cgroup v2 memory peak/current is unavailable")


def _memory_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _telemetry_probe() -> Any | None:
    try:
        from uma_qmoe.telemetry import TelemetryProbe

        return TelemetryProbe.detect()
    except Exception:
        return None


def _resource_sample(probe: Any | None, boundary: str) -> dict[str, Any]:
    normalized = {
        "temperature_c": None,
        "socket_power_w": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
        "utilization_percent": None,
    }
    provider = None
    if probe is not None:
        try:
            captured = probe.capture(
                operation_id="reference_host", sample_index=0, boundary=boundary
            )
            normalized.update(
                {key: captured.get(key) for key in tuple(normalized)}
            )
            provider = probe.provider
        except Exception:
            provider = f"{probe.provider}_capture_failed"
    return {
        "boundary": boundary,
        "monotonic_ns": time.monotonic_ns(),
        "memory_available_bytes": _memory_available(),
        "telemetry_provider": provider,
        "telemetry": normalized,
    }


def _generate_once(
    model: Any,
    input_ids: Any,
    *,
    output_tokens: int,
    torch: Any,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = output.past_key_values
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    generated = [int(next_token.item())]
    intervals_ms: list[float] = []
    for _ in range(output_tokens - 1):
        torch.cuda.synchronize()
        token_started = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
        torch.cuda.synchronize()
        intervals_ms.append(
            (time.perf_counter_ns() - token_started) / 1_000_000.0
        )
        generated.append(int(next_token.item()))
    if len(generated) != output_tokens:
        raise RuntimeError("manual generation returned an unexpected token count")
    return {
        "input_tokens": int(input_ids.shape[-1]),
        "output_tokens": len(generated),
        "ttft_ms": ttft_ms,
        "inter_token_latencies_ms": intervals_ms,
        "generated_token_ids": generated,
        "error": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    from uma_qmoe.contracts import ContractError
    from uma_qmoe.fixed_models import fixed_model_spec

    try:
        spec = fixed_model_spec(args.model_id, args.model_revision)
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
    if min(
        args.input_tokens,
        args.output_tokens,
        args.warmup_requests,
        args.measured_requests,
    ) <= 0:
        raise SystemExit("all workload counts must be positive")
    if args.output_tokens < 2:
        raise SystemExit("output-tokens must be at least 2 for TPOT evidence")
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_files = ("result.json", "runner-metadata.json")
    if any((output_directory / name).exists() for name in output_files):
        raise SystemExit("refusing to overwrite existing reference host evidence")

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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("reference host requires a BF16-capable CUDA/HIP device")
    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"reference host backend mismatch: requested {args.backend}, "
            f"observed {observed_backend}"
        )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    token_ids = _exact_token_ids(tokenizer, prompt, args.input_tokens)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    observed = (
        getattr(model.config, "model_type", None),
        getattr(model.config, "num_hidden_layers", None),
        getattr(model.config, "num_experts", None),
        getattr(model.config, "num_experts_per_tok", None),
    )
    expected = (spec.model_type, spec.num_layers, spec.num_experts, spec.top_k)
    if observed != expected:
        raise RuntimeError(
            f"unexpected fixed-model architecture: {observed!r}, expected {expected!r}"
        )

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_requests):
        _generate_once(
            model, input_ids, output_tokens=args.output_tokens, torch=torch
        )
    torch.cuda.synchronize()
    measured_started = time.perf_counter_ns()
    requests = [
        _generate_once(
            model, input_ids, output_tokens=args.output_tokens, torch=torch
        )
        for _ in range(args.measured_requests)
    ]
    torch.cuda.synchronize()
    duration_seconds = (time.perf_counter_ns() - measured_started) / 1_000_000_000.0
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
    result_path = output_directory / "result.json"
    result_path.write_text(
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
        "model": {
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "path": str(args.model.resolve()),
        },
        "workload": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.measured_requests,
            "max_concurrency": 1,
        },
        "runtime": {
            "source_commit": args.source_commit,
            "backend": observed_backend,
            "container_image": args.container_image,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "backend_version": str(backend_version),
            "accelerator": torch.cuda.get_device_name(0),
        },
        "cgroup_memory_peak_bytes": _cgroup_memory_peak(),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "samples": samples,
    }
    (output_directory / "runner-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
