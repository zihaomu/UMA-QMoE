#!/usr/bin/env python3
"""Capture single-expert and single-layer outputs from the real Q4 ExpertPack."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from uma_qmoe.expert_pack import ExpertPackReader
from uma_qmoe.q4 import dequantize_q4


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
LAYER_INDEX = 2
EXPERT_INDEX = 7


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


def _record(identity: str, role: str, tensor: Any, dtype: str) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "identity": identity,
        "role": role,
        "dtype": dtype,
        "shape": list(value.shape),
        "values": (
            value.reshape(-1).tolist()
            if dtype.startswith(("int", "uint"))
            else value.float().reshape(-1).tolist()
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
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_names = (
        "single-expert-q4.jsonl",
        "single-moe-layer-q4.jsonl",
        "capture-metadata.json",
    )
    if any((output / name).exists() for name in output_names):
        raise SystemExit("refusing to overwrite existing Q4 Oracle evidence")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q4 Oracle capture requires a BF16 CUDA/HIP device")
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    experts = model.model.layers[LAYER_INDEX].mlp.experts

    with ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    ) as pack:
        if pack.mapping_count != 1:
            raise RuntimeError("ExpertPack must be mapped exactly once")
        with torch.no_grad():
            for expert_index in range(64):
                prefix = f"model.layers.{LAYER_INDEX}.mlp.experts.{expert_index}"
                gate = dequantize_q4(pack.tensor_q4(f"{prefix}.gate_proj.weight"))
                up = dequantize_q4(pack.tensor_q4(f"{prefix}.up_proj.weight"))
                down = dequantize_q4(pack.tensor_q4(f"{prefix}.down_proj.weight"))
                gate_up = np.concatenate((gate, up), axis=0)
                experts.gate_up_proj[expert_index].copy_(
                    torch.from_numpy(gate_up).to(
                        device="cuda:0", dtype=torch.bfloat16
                    )
                )
                experts.down_proj[expert_index].copy_(
                    torch.from_numpy(down).to(
                        device="cuda:0", dtype=torch.bfloat16
                    )
                )
                del gate, up, down, gate_up

        hidden_size = int(model.config.hidden_size)
        fixture = (
            (torch.arange(4 * hidden_size, dtype=torch.float32) % 257) / 128.0
            - 1.0
        ).reshape(4, hidden_size)
        expert_input = fixture.to(device="cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            gate_value, up_value = functional.linear(
                expert_input, experts.gate_up_proj[EXPERT_INDEX]
            ).chunk(2, dim=-1)
            expert_output = functional.linear(
                experts.act_fn(gate_value) * up_value,
                experts.down_proj[EXPERT_INDEX],
            )

        observed: dict[str, Any] = {}

        def gate_hook(_module: Any, _inputs: Any, result: Any) -> None:
            observed["router_logits"] = result[0].detach().cpu()
            observed["router_topk"] = result[2].detach().cpu()

        def moe_hook(_module: Any, _inputs: Any, result: Any) -> None:
            observed["moe_output"] = result.detach().cpu()

        handles = [
            model.model.layers[LAYER_INDEX].mlp.gate.register_forward_hook(gate_hook),
            model.model.layers[LAYER_INDEX].mlp.register_forward_hook(moe_hook),
        ]
        try:
            device_inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
            with torch.inference_mode():
                model(**device_inputs, use_cache=False)
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        if set(observed) != {"router_logits", "router_topk", "moe_output"}:
            raise RuntimeError("Q4 Oracle layer hooks did not capture all tensors")
        if not all(
            torch.isfinite(value).all().item()
            for value in (expert_output, observed["router_logits"], observed["moe_output"])
        ):
            raise RuntimeError("Q4 Oracle capture produced NaN or Inf")

    single_expert = output / "single-expert-q4.jsonl"
    single_layer = output / "single-moe-layer-q4.jsonl"
    _write_jsonl(
        single_expert,
        [
            _record(
                f"model.layers.{LAYER_INDEX}.mlp.experts.{EXPERT_INDEX}.output",
                "expert_output",
                expert_output,
                "bfloat16",
            )
        ],
    )
    _write_jsonl(
        single_layer,
        [
            _record(
                f"model.layers.{LAYER_INDEX}.mlp.router_logits",
                "router_logits",
                observed["router_logits"],
                "bfloat16",
            ),
            _record(
                f"model.layers.{LAYER_INDEX}.mlp.output",
                "moe_layer_output",
                observed["moe_output"],
                "bfloat16",
            ),
            _record(
                f"model.layers.{LAYER_INDEX}.mlp.router_topk_indices",
                "router_topk_indices",
                observed["router_topk"],
                "int64",
            ),
        ],
    )
    metadata = {
        "schema_version": 1,
        "captured_at": _utc_now(),
        "target_id": args.target_id,
        "backend": args.backend,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "expert_index": EXPERT_INDEX,
        "quantization": "canonical_q4_group128",
        "expert_pack": {
            "path": str(args.expert_pack.resolve()),
            "sha256": _sha256(args.expert_pack),
            "mapping_count": 1,
        },
        "prompt": {
            "fixture_sha256": _sha256(args.prompt_fixture),
            "prompt_id": args.prompt_id,
            "token_ids": encoded["input_ids"][0].tolist(),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "streams": {
            "single_expert": {
                "path": single_expert.name,
                "sha256": _sha256(single_expert),
            },
            "single_moe_layer": {
                "path": single_layer.name,
                "sha256": _sha256(single_layer),
            },
        },
    }
    (output / "capture-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
