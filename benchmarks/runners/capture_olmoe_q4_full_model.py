#!/usr/bin/env python3
"""Capture full-model Q4 logits and all OLMoE Top-8 routes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from uma_qmoe.compressed_loader import load_fixed_olmoe


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


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
    values = (
        value.reshape(-1).tolist()
        if dtype.startswith(("int", "uint"))
        else value.float().reshape(-1).tolist()
    )
    return {
        "identity": identity,
        "role": role,
        "dtype": dtype,
        "shape": list(value.shape),
        "values": values,
    }


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
    stream_path = output / "full-model-q4.jsonl"
    metadata_path = output / "capture-metadata.json"
    if stream_path.exists() or metadata_path.exists():
        raise SystemExit("refusing to overwrite full-model Q4 capture evidence")
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

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("full-model Q4 capture requires a BF16 CUDA/HIP device")
    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    routes: dict[int, Any] = {}
    handles: list[Any] = []

    with load_fixed_olmoe(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=False,
    ) as host:

        def gate_hook(layer_index: int):
            def hook(_module: Any, _inputs: Any, result: Any) -> None:
                if not isinstance(result, tuple) or len(result) != 3:
                    raise RuntimeError("OLMoE Q4 router hook returned an unexpected value")
                routes[layer_index] = result[2].detach().cpu()

            return hook

        for layer_index, layer in enumerate(host.model.model.layers):
            handles.append(layer.mlp.gate.register_forward_hook(gate_hook(layer_index)))
        try:
            device_inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
            with torch.inference_mode():
                result = host.model(**device_inputs, use_cache=False)
                final_logits = result.logits[:, -1, :].detach().float().cpu()
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        if set(routes) != set(range(16)):
            raise RuntimeError("full-model Q4 capture did not observe all 16 routers")
        if not torch.isfinite(final_logits).all().item():
            raise RuntimeError("full-model Q4 capture produced NaN or Inf")
        loader = dict(host.evidence)

    records = [
        _record("model.final_logits", "final_logits", final_logits, "float32"),
        *[
            _record(
                f"model.layers.{layer_index}.mlp.router_topk_indices",
                "router_topk_indices",
                routes[layer_index],
                "int64",
            )
            for layer_index in range(16)
        ],
    ]
    with stream_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
    metadata = {
        "schema_version": 1,
        "kind": "olmoe_q4_full_model_capture_metadata",
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "target_id": args.target_id,
        "backend": args.backend,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": "canonical_q4_group128",
        "expert_pack": {
            "path": str(args.expert_pack.resolve()),
            "sha256": _sha256(args.expert_pack),
            "mapping_count": loader["expert_pack_mapping_count"],
        },
        "loader": {
            "dense_tensor_bytes": loader["dense_tensor_bytes"],
            "loaded_expert_tensor_count": loader["loaded_expert_tensor_count"],
            "expert_parameter_count": loader["expert_parameter_count"],
            "quantized_moe_block_count": loader["quantized_moe_block_count"],
        },
        "prompt": {
            "fixture_sha256": _sha256(args.prompt_fixture),
            "prompt_id": args.prompt_id,
            "token_ids": encoded["input_ids"][0].tolist(),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(0),
        },
        "stream": {
            "path": stream_path.name,
            "sha256": _sha256(stream_path),
            "size_bytes": stream_path.stat().st_size,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
