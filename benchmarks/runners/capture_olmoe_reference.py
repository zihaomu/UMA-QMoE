#!/usr/bin/env python3
"""Capture the fixed three-level OLMoE numerical Oracle tensor streams."""

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
LAYER_INDEX = 2
EXPERT_INDEX = 7


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(path: Path, prompt_id: str) -> str:
    matches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("id") == prompt_id:
            matches.append(item.get("prompt"))
    if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
        raise RuntimeError(f"prompt fixture must contain exactly one {prompt_id!r}")
    return matches[0]


def _record(identity: str, role: str, tensor: Any, dtype: str) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "identity": identity,
        "role": role,
        "dtype": dtype,
        "shape": list(value.shape),
        "values": (
            value.tolist()
            if dtype.startswith("int") or dtype.startswith("uint")
            else value.float().tolist()
        ),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "single-expert.jsonl",
        "single-moe-layer.jsonl",
        "full-model.jsonl",
        "capture-metadata.json",
    )
    if any((output / name).exists() for name in names):
        raise SystemExit("refusing to overwrite existing Oracle capture evidence")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Oracle capture requires a BF16-capable CUDA/HIP device")
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0].tolist()
    if not token_ids:
        raise RuntimeError("fixed Oracle prompt encoded to zero tokens")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    config = model.config
    observed = (
        getattr(config, "model_type", None),
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_experts", None),
        getattr(config, "num_experts_per_tok", None),
    )
    if observed != ("olmoe", 16, 64, 8):
        raise RuntimeError(f"unexpected OLMoE architecture: {observed!r}")

    hidden_size = int(config.hidden_size)
    expert_fixture = (
        (torch.arange(4 * hidden_size, dtype=torch.float32) % 257) / 128.0 - 1.0
    ).reshape(4, hidden_size)
    expert_fixture_sha256 = hashlib.sha256(
        expert_fixture.numpy().tobytes()
    ).hexdigest()
    expert_input = expert_fixture.to(device="cuda:0", dtype=torch.bfloat16)
    experts = model.model.layers[LAYER_INDEX].mlp.experts
    with torch.inference_mode():
        gate, up = functional.linear(
            expert_input, experts.gate_up_proj[EXPERT_INDEX]
        ).chunk(2, dim=-1)
        expert_output = functional.linear(
            experts.act_fn(gate) * up, experts.down_proj[EXPERT_INDEX]
        )

    router_logits: dict[int, Any] = {}
    router_topk: dict[int, Any] = {}
    moe_output: dict[str, Any] = {}
    handles = []

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE router hook returned an unexpected value")
            router_logits[layer_index] = result[0].detach().cpu()
            router_topk[layer_index] = result[2].detach().cpu()

        return hook

    def moe_hook(_module: Any, _inputs: Any, result: Any) -> None:
        moe_output["value"] = result.detach().cpu()

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(gate_hook(layer_index)))
    handles.append(model.model.layers[LAYER_INDEX].mlp.register_forward_hook(moe_hook))
    device_inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
    try:
        with torch.inference_mode():
            result = model(**device_inputs, use_cache=False)
            final_logits = result.logits[:, -1, :].detach().cpu()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if set(router_logits) != set(range(16)) or set(router_topk) != set(range(16)):
        raise RuntimeError("Oracle capture did not observe every OLMoE router")
    if "value" not in moe_output:
        raise RuntimeError("Oracle capture did not observe the selected MoE layer")
    if not all(
        torch.isfinite(value).all().item()
        for value in [expert_output, *router_logits.values(), moe_output["value"], final_logits]
    ):
        raise RuntimeError("Oracle capture produced NaN or Inf")

    expert_records = [
        _record(
            f"model.layers.{LAYER_INDEX}.mlp.experts.{EXPERT_INDEX}.output",
            "expert_output",
            expert_output,
            "bfloat16",
        )
    ]
    moe_records = [
        _record(
            f"model.layers.{LAYER_INDEX}.mlp.router_logits",
            "router_logits",
            router_logits[LAYER_INDEX],
            "bfloat16",
        ),
        _record(
            f"model.layers.{LAYER_INDEX}.mlp.output",
            "moe_layer_output",
            moe_output["value"],
            "bfloat16",
        ),
        _record(
            f"model.layers.{LAYER_INDEX}.mlp.router_topk_indices",
            "router_topk_indices",
            router_topk[LAYER_INDEX],
            "int64",
        ),
    ]
    full_records = [
        _record("model.final_logits", "final_logits", final_logits, "float32"),
        *[
            _record(
                f"model.layers.{layer_index}.mlp.router_topk_indices",
                "router_topk_indices",
                router_topk[layer_index],
                "int64",
            )
            for layer_index in range(16)
        ],
    ]
    files = {
        "single_expert": output / "single-expert.jsonl",
        "single_moe_layer": output / "single-moe-layer.jsonl",
        "full_model": output / "full-model.jsonl",
    }
    _write_jsonl(files["single_expert"], expert_records)
    _write_jsonl(files["single_moe_layer"], moe_records)
    _write_jsonl(files["full_model"], full_records)
    metadata = {
        "schema_version": 1,
        "kind": "olmoe_reference_capture_metadata",
        "captured_at": _utc_now(),
        "target_id": args.target_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_dtype": "BF16",
        "scope": {"layer_index": LAYER_INDEX, "expert_index": EXPERT_INDEX},
        "prompt": {
            "fixture_sha256": _sha256(args.prompt_fixture),
            "prompt_id": args.prompt_id,
            "token_ids": token_ids,
        },
        "expert_fixture_sha256": expert_fixture_sha256,
        "streams": {
            name: {
                "path": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    (output / "capture-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
