#!/usr/bin/env python3
"""Capture every Qwen1.5-MoE Top-4 route for fixed greedy generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from uma_qmoe.fixed_models import QWEN1_5_MOE


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


def _router_topk(
    result: Any,
    functional: Any,
    torch: Any,
    top_k: int,
    normalize_top_k: bool,
) -> tuple[Any, Any]:
    """Read exact router outputs from current or legacy Transformers APIs."""

    if isinstance(result, tuple) and len(result) == 3:
        _router_probabilities, routing_weights, selected_experts = result
        return routing_weights, selected_experts
    if not isinstance(result, torch.Tensor) or result.ndim != 2:
        raise RuntimeError("Qwen gate returned an unexpected router value")
    routing = functional.softmax(result, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing, top_k, dim=-1)
    if normalize_top_k:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
    return routing_weights, selected_experts


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
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RouteTrace capture requires a BF16 CUDA/HIP device")
    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"RouteTrace backend mismatch: requested {args.backend}, "
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
    if observed != QWEN1_5_MOE.architecture:
        raise RuntimeError(f"unexpected Qwen architecture: {observed!r}")
    if getattr(model.config, "moe_intermediate_size", None) != (
        QWEN1_5_MOE.expert_intermediate_size
    ) or getattr(model.config, "shared_expert_intermediate_size", None) != (
        QWEN1_5_MOE.shared_expert_intermediate_size
    ):
        raise RuntimeError("Qwen routed/shared expert widths do not match the fixed ABI")

    current_event: dict[str, int] = {}
    layer_events: list[list[dict[str, Any]]] = [
        [] for _ in range(QWEN1_5_MOE.num_layers)
    ]
    handles = []
    normalize_top_k = bool(getattr(model.config, "norm_topk_prob", False))

    def make_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            weights, experts = _router_topk(
                result,
                functional,
                torch,
                QWEN1_5_MOE.top_k,
                normalize_top_k,
            )
            flat_weights = weights.detach().cpu().reshape(-1).tolist()
            flat_experts = experts.detach().cpu().reshape(-1).tolist()
            token_count = int(experts.shape[0])
            expected = token_count * QWEN1_5_MOE.top_k
            if len(flat_experts) != expected or len(flat_weights) != expected:
                raise RuntimeError("Qwen gate Top-K capture shape is inconsistent")
            layer_events[layer_index].append(
                {
                    "event_index": current_event["index"],
                    "expert_indices": [int(value) for value in flat_experts],
                    "routing_weights": [float(value) for value in flat_weights],
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
    token_ids_sha256 = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    capture = {
        "schema_version": 1,
        "captured_at": _utc_now(),
        "model_id": QWEN1_5_MOE.model_id,
        "model_revision": QWEN1_5_MOE.model_revision,
        "architecture": {
            "num_layers": QWEN1_5_MOE.num_layers,
            "num_experts": QWEN1_5_MOE.num_experts,
            "top_k": QWEN1_5_MOE.top_k,
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
        "layers": [
            {"layer_index": layer_index, "events": observed_events}
            for layer_index, observed_events in enumerate(layer_events)
        ],
    }
    destination.write_text(
        json.dumps(capture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
