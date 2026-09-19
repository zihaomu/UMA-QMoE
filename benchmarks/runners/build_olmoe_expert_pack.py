#!/usr/bin/env python3
"""Build the full OLMoE ExpertPack locally from a verified BF16 derivation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from uma_qmoe.contracts import file_sha256, find_project_root, validate_file
from uma_qmoe.expert_pack import (
    _ordered_olmoe_experts,
    _validate_olmoe_expert_tensor,
    write_expert_pack,
)
from uma_qmoe.model_store import verify_derivation_files


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--derivation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def _tensor_stream(model: Path, ordered: list[tuple[str, str]]) -> Iterator[tuple[str, Any]]:
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for name, shard in ordered:
        by_shard.setdefault(shard, []).append(name)
    ordered_names = {name: index for index, (name, _) in enumerate(ordered)}
    # Shards are opened once, but tensor order in the pack remains semantic
    # layer/expert/projection order via a small name-to-array staging window.
    # OLMoE places an expert's projections in the same shard, so at most one
    # tensor is materialized when the generator yields.
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
            _validate_olmoe_expert_tensor(name, tensor)
            yield name, tensor.float().numpy()
            del tensor
    finally:
        handles.clear()
    assert len(ordered_names) == len(ordered) and len(by_shard) > 0


def main() -> int:
    args = _parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if args.manifest_output.exists():
        raise SystemExit("refusing to overwrite existing ExpertPack manifest")
    derivation_path = args.derivation.resolve()
    derivation = validate_file(derivation_path)
    if (
        derivation.get("kind") != "model_derivation"
        or derivation["source"]["model_id"] != MODEL_ID
        or derivation["source"]["model_revision"] != MODEL_REVISION
        or derivation["transform"]["target_dtype"] != "BF16"
        or derivation["weights"]["observed_dtypes"] != ["BF16"]
    ):
        raise RuntimeError("ExpertPack requires the fixed verified OLMoE BF16 derivation")
    project_root = find_project_root(derivation_path)
    derivation_relative = derivation_path.relative_to(project_root).as_posix()
    verify_derivation_files(
        derivation,
        args.model,
        target_id="local-expert-pack-builder",
        source_contract_path=derivation_relative,
        source_contract_sha256=file_sha256(derivation_path),
    )
    index_identity = next(
        (
            item
            for item in derivation["weights"]["artifacts"]
            if item["path"] == "model.safetensors.index.json"
        ),
        None,
    )
    if index_identity is None:
        raise RuntimeError("verified OLMoE derivation has no Safetensors index")
    index_path = args.model / index_identity["path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise RuntimeError("model Safetensors index has no valid weight_map")
    ordered = _ordered_olmoe_experts(weight_map)
    manifest = write_expert_pack(
        args.output,
        _tensor_stream(args.model, ordered),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_manifest_sha256=derivation["source"]["model_manifest_sha256"],
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
