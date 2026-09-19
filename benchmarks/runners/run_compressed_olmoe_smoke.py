#!/usr/bin/env python3
"""Load dense-only OLMoE plus one ExpertPack mmap and run a full Q4 forward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from uma_qmoe.compressed_loader import load_fixed_olmoe


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--prompt-id", default="general-001")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite compressed loader evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from transformers import AutoTokenizer

    prompt = _prompt(args.prompt_fixture, args.prompt_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with load_fixed_olmoe(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=False,
    ) as host:
        device_inputs = {name: value.to("cuda:0") for name, value in encoded.items()}
        with torch.inference_mode():
            result = host.model(**device_inputs, use_cache=False)
            logits = result.logits[:, -1, :].detach().float().cpu().contiguous()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(logits).all().item())
        loader = {
            key: host.evidence[key]
            for key in (
                "device",
                "performance_mode",
                "dense_tensor_count",
                "dense_tensor_bytes",
                "skipped_expert_tensor_count",
                "loaded_expert_tensor_count",
                "expert_parameter_count",
                "quantized_moe_block_count",
                "model_parameter_bytes",
                "expert_pack_size_bytes",
                "expert_pack_mapping_count",
                "expert_pack_vma_count",
            )
        }
        memory = {
            "cgroup_current_bytes": host.evidence["cgroup_memory_current_bytes"],
            "cgroup_peak_bytes": host.evidence["cgroup_memory_peak_bytes"],
            "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        smoke = {
            "prompt_id": args.prompt_id,
            "logits_shape": list(logits.shape),
            "finite": finite,
            "top1_token_id": int(logits.argmax(dim=-1).item()),
            "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
        }
    gates = {
        "dense_only_checkpoint_load": loader["loaded_expert_tensor_count"] == 0,
        "no_expert_parameters": loader["expert_parameter_count"] == 0,
        "single_pack_mapping": loader["expert_pack_mapping_count"] == 1,
        "no_full_dequantized_copy": (
            loader["model_parameter_bytes"] < loader["expert_pack_size_bytes"]
        ),
        "full_model_forward": finite,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "compressed_loader_evidence",
        "captured_at": _utc_now(),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": _sha256(args.expert_pack),
        },
        "loader": loader,
        "memory": memory,
        "smoke": smoke,
        "gates": gates,
    }
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
