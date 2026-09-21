#!/usr/bin/env python3
"""Build the fixed Qwen1.5-MoE canonical Q4 ExpertPack locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from uma_qmoe.contracts import canonical_sha256, validate_file
from uma_qmoe.expert_pack import (
    _ordered_qwen_experts,
    _validate_qwen_expert_tensor,
    write_expert_pack,
)
from uma_qmoe.fixed_models import QWEN1_5_MOE
from uma_qmoe.model_store import manifest_file_identities, verify_model_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def _tensor_stream(
    model: Path, ordered: list[tuple[str, str]]
) -> Iterator[tuple[str, Any]]:
    from safetensors import safe_open

    shard_for_name = {name: shard for name, shard in ordered}
    handles: dict[str, Any] = {}
    try:
        for name, _ in ordered:
            shard = shard_for_name[name]
            handle = handles.get(shard)
            if handle is None:
                handle = safe_open(model / shard, framework="pt", device="cpu")
                handles[shard] = handle
            tensor = handle.get_tensor(name)
            _validate_qwen_expert_tensor(name, tensor)
            yield name, tensor.float().numpy()
            del tensor
    finally:
        handles.clear()


def main() -> int:
    args = _parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if args.manifest_output.exists():
        raise SystemExit("refusing to overwrite existing ExpertPack manifest")
    model_manifest = validate_file(args.model_manifest)
    if (
        model_manifest.get("model_id") != QWEN1_5_MOE.model_id
        or model_manifest.get("model_revision") != QWEN1_5_MOE.model_revision
        or model_manifest.get("dtypes", {}).get("uploaded_weights") != "bfloat16"
    ):
        raise RuntimeError("ExpertPack requires the fixed BF16 Qwen model manifest")
    manifest_sha256 = canonical_sha256(model_manifest)
    verification = validate_file(args.source_verification)
    if (
        verification.get("kind") != "model_acquisition"
        or verification.get("model_manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError("Qwen source verification does not match the manifest")
    expected_files = {
        item["path"]: (item["size_bytes"], item["sha256"])
        for item in manifest_file_identities(model_manifest)
    }
    observed_files = {
        item["path"]: (item["size_bytes"], item["sha256"])
        for item in verification.get("files", [])
    }
    if observed_files != expected_files:
        raise RuntimeError("Qwen source verification file identities are incomplete")
    current_verification = verify_model_files(model_manifest, args.model)
    if {
        item["path"]: (item["size_bytes"], item["sha256"])
        for item in current_verification["files"]
    } != expected_files:
        raise RuntimeError("Qwen model changed after source verification")

    index_path = args.model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise RuntimeError("model Safetensors index has no valid weight_map")
    ordered = _ordered_qwen_experts(weight_map)
    manifest = write_expert_pack(
        args.output,
        _tensor_stream(args.model, ordered),
        model_id=QWEN1_5_MOE.model_id,
        model_revision=QWEN1_5_MOE.model_revision,
        model_manifest_sha256=manifest_sha256,
        group_size=128,
        tensor_alignment_bytes=64,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
