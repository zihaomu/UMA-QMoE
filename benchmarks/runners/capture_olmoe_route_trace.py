#!/usr/bin/env python3
"""Capture every OLMoE Top-8 route for a fixed offline greedy generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise RuntimeError("fixed RouteTrace prompt encoded to zero tokens")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.input_tokens <= 0 or args.output_tokens < 2:
        raise SystemExit("RouteTrace requires positive input and at least two output tokens")
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit("refusing to overwrite existing RouteTrace capture")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RouteTrace capture requires a BF16 CUDA/HIP device")
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
    if observed != ("olmoe", NUM_LAYERS, NUM_EXPERTS, TOP_K):
        raise RuntimeError(f"unexpected OLMoE architecture: {observed!r}")

    current_event: dict[str, int] = {}
    layer_events: list[list[dict[str, Any]]] = [[] for _ in range(NUM_LAYERS)]
    handles = []

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE gate returned an unexpected value")
            weights = result[1].detach().float().cpu().reshape(-1).tolist()
            experts = result[2].detach().cpu().reshape(-1).tolist()
            token_count = int(result[2].shape[0])
            if len(experts) != token_count * TOP_K or len(weights) != len(experts):
                raise RuntimeError("OLMoE gate Top-K capture shape is inconsistent")
            layer_events[layer_index].append(
                {
                    "event_index": current_event["index"],
                    "expert_indices": [int(value) for value in experts],
                    "routing_weights": [float(value) for value in weights],
                }
            )

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(make_hook(layer_index)))

    events: list[dict[str, Any]] = []

    def run_event(token_input: Any, phase: str, decode_step: int | None, past: Any):
        event_index = len(events)
        current_event["index"] = event_index
        events.append(
            {
                "event_index": event_index,
                "phase": phase,
                "decode_step": decode_step,
                "batch_size": int(token_input.shape[0]),
                "tokens_per_sequence": int(token_input.shape[1]),
            }
        )
        with torch.inference_mode():
            result = model(
                input_ids=token_input,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            next_token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token, result.past_key_values

    try:
        next_token, past_key_values = run_event(input_ids, "prefill", None, None)
        generated = [int(next_token.item())]
        for decode_step in range(args.output_tokens - 1):
            next_token, past_key_values = run_event(
                next_token, "decode", decode_step, past_key_values
            )
            generated.append(int(next_token.item()))
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    if len(events) != args.output_tokens or any(
        len(observed_events) != len(events) for observed_events in layer_events
    ):
        raise RuntimeError("RouteTrace capture did not cover every layer/event")
    layers = [
        {
            "layer_index": layer_index,
            "events": observed_events,
        }
        for layer_index, observed_events in enumerate(layer_events)
    ]
    token_ids_sha256 = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    capture = {
        "schema_version": 1,
        "captured_at": _utc_now(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": {
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
        },
        "fixture": {
            "id": args.prompt_id,
            "sha256": _sha256(args.prompt_fixture),
            "token_ids_sha256": token_ids_sha256,
        },
        "workload": {
            "batch_size": 1,
            "prompt_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "decoding": "greedy",
            "ignore_eos": True,
        },
        "capture": {
            "target_id": args.target_id,
            "backend": args.backend,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "generated_token_ids": generated,
        "events": events,
        "layers": layers,
    }
    destination.write_text(
        json.dumps(capture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
