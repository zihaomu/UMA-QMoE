#!/usr/bin/env python3
"""Compare the fixed Qwen Q4 Host against a captured BF16 Oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from uma_qmoe.fixed_models import QWEN1_5_MOE
from uma_qmoe.qwen_compressed_loader import load_fixed_qwen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identity = record.get("identity")
        if not isinstance(identity, str) or identity in records:
            raise RuntimeError(f"invalid or duplicate Oracle identity in {path}")
        records[identity] = record
    return records


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
    parser.add_argument("--target-id", choices=("local-halo", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--target-policy-id")
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--performance-mode", action="store_true")
    parser.add_argument("--backend", choices=("cuda", "hip"))
    parser.add_argument("--native-build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite Qwen Host quality evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    metadata_path = args.oracle_dir / "capture-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("model_id") != QWEN1_5_MOE.model_id
        or metadata.get("model_revision") != QWEN1_5_MOE.model_revision
        or metadata.get("prompt", {}).get("fixture_sha256")
        != _sha256(args.prompt_fixture)
    ):
        raise RuntimeError("Qwen Oracle identity or prompt fixture mismatch")
    for stream in metadata.get("streams", {}).values():
        stream_path = args.oracle_dir / stream["path"]
        if (
            stream_path.stat().st_size != stream["size_bytes"]
            or _sha256(stream_path) != stream["sha256"]
        ):
            raise RuntimeError(f"Qwen Oracle stream changed: {stream_path}")
    records = _load_jsonl(args.oracle_dir / "full-model.jsonl")
    moe_records = _load_jsonl(args.oracle_dir / "single-moe-layer.jsonl")

    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer

    native = None
    if args.performance_mode:
        if args.backend is None:
            raise RuntimeError("performance-mode quality requires --backend")
        observed_backend = "hip" if torch.version.hip else "cuda"
        if observed_backend != args.backend:
            raise RuntimeError(
                f"quality backend mismatch: requested {args.backend}, "
                f"observed {observed_backend}"
            )
        from uma_qmoe.native_backend import (
            install_mixed_target_backend,
            install_packed_q4_backend,
        )

        installer = (
            install_mixed_target_backend
            if args.target_policy_id
            else install_packed_q4_backend
        )
        native = installer(build_directory=args.native_build_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prompt_id = metadata["prompt"]["prompt_id"]
    encoded = tokenizer(
        _prompt(args.prompt_fixture, prompt_id),
        return_tensors="pt",
        add_special_tokens=False,
    )
    token_ids = encoded["input_ids"][0].tolist()
    if token_ids != metadata["prompt"]["token_ids"]:
        raise RuntimeError("Qwen tokenizer no longer reproduces Oracle token IDs")

    reference_logits_record = records["model.final_logits"]
    reference_logits = torch.tensor(
        reference_logits_record["values"], dtype=torch.float32
    ).reshape(reference_logits_record["shape"])
    reference_routes = {
        layer: torch.tensor(
            records[
                f"model.layers.{layer}.mlp.router_topk_indices"
            ]["values"],
            dtype=torch.int64,
        ).reshape(
            records[
                f"model.layers.{layer}.mlp.router_topk_indices"
            ]["shape"]
        )
        for layer in range(QWEN1_5_MOE.num_layers)
    }
    candidate_routes: dict[int, Any] = {}
    candidate_moe_output: dict[str, Any] = {}

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("compressed Qwen gate returned an unexpected value")
            candidate_routes[layer_index] = result[2].detach().cpu()

        return hook

    def moe_hook(_module: Any, _inputs: Any, result: Any) -> None:
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("compressed Qwen MoE block returned an unexpected value")
        candidate_moe_output["value"] = result.detach().float().cpu()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    with load_fixed_qwen(
        args.model,
        args.expert_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=args.performance_mode,
        target_policy_id=args.target_policy_id,
    ) as host:
        handles = [
            layer.mlp.gate.register_forward_hook(gate_hook(layer_index))
            for layer_index, layer in enumerate(host.model.model.layers)
        ]
        handles.append(host.model.model.layers[2].mlp.register_forward_hook(moe_hook))
        try:
            with torch.inference_mode():
                result = host.model(
                    **{name: value.to("cuda:0") for name, value in encoded.items()},
                    use_cache=False,
                )
                candidate_logits = result.logits[:, -1, :].detach().float().cpu()
            torch.cuda.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        loader = dict(host.evidence)
        backend_cache = native.cache_summary() if native is not None else None
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    if set(candidate_routes) != set(range(QWEN1_5_MOE.num_layers)):
        raise RuntimeError("compressed Qwen Host did not execute every router")
    if candidate_logits.shape != reference_logits.shape:
        raise RuntimeError("compressed Qwen logits shape differs from Oracle")
    reference_moe_record = moe_records["model.layers.2.mlp.output"]
    reference_moe_output = torch.tensor(
        reference_moe_record["values"], dtype=torch.float32
    ).reshape(reference_moe_record["shape"])
    observed_moe_output = candidate_moe_output.get("value")
    if (
        observed_moe_output is None
        or observed_moe_output.shape != reference_moe_output.shape
    ):
        raise RuntimeError("compressed Qwen layer-2 output differs in shape")
    finite = bool(torch.isfinite(candidate_logits).all().item())
    cosine = float(
        functional.cosine_similarity(
            candidate_logits.reshape(1, -1),
            reference_logits.reshape(1, -1),
        ).item()
    )
    reference_log_prob = functional.log_softmax(reference_logits, dim=-1)
    candidate_log_prob = functional.log_softmax(candidate_logits, dim=-1)
    reference_prob = reference_log_prob.exp()
    logit_kl = float(
        (reference_prob * (reference_log_prob - candidate_log_prob)).sum().item()
    )
    max_abs_error = float((candidate_logits - reference_logits).abs().max().item())
    moe_cosine = float(
        functional.cosine_similarity(
            observed_moe_output.reshape(1, -1),
            reference_moe_output.reshape(1, -1),
        ).item()
    )
    moe_max_abs_error = float(
        (observed_moe_output - reference_moe_output).abs().max().item()
    )
    per_layer_route_agreement: list[float] = []
    for layer in range(QWEN1_5_MOE.num_layers):
        reference = reference_routes[layer]
        candidate = candidate_routes[layer]
        if reference.shape != candidate.shape:
            raise RuntimeError(f"Qwen layer {layer} route shape differs from Oracle")
        exact = [
            set(reference[row].tolist()) == set(candidate[row].tolist())
            for row in range(reference.shape[0])
        ]
        per_layer_route_agreement.append(sum(exact) / len(exact))
    route_agreement = sum(per_layer_route_agreement) / len(
        per_layer_route_agreement
    )
    reference_top1 = int(reference_logits.argmax(dim=-1).item())
    candidate_top1 = int(candidate_logits.argmax(dim=-1).item())
    gates = {
        "oracle_stream_identity": True,
        "finite": finite,
        "top1_token_match": candidate_top1 == reference_top1,
        "logit_kl_at_most_0_01": logit_kl <= 0.01,
        "router_top4_set_agreement_at_least_0_99": route_agreement >= 0.99,
        "dense_only_checkpoint_load": (
            loader["skipped_expert_tensor_count"] == 4320
            and loader["loaded_expert_tensor_count"] == 0
        ),
        "no_expert_parameters": loader["expert_parameter_count"] == 0,
        "single_pack_mapping": (
            loader["expert_pack_mapping_count"] == 1
            and loader["expert_pack_vma_count"] in {1, -1}
        ),
    }
    gates["overall_passed"] = all(gates.values())
    evidence = {
        "schema_version": 1,
        "kind": "qwen_expert_pack_host_quality",
        "captured_at": _utc_now(),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "model_manifest_sha256": args.model_manifest_sha256,
        },
        "oracle": {
            "metadata_sha256": _sha256(metadata_path),
            "full_model_stream_sha256": metadata["streams"]["full_model"][
                "sha256"
            ],
            "prompt_id": prompt_id,
            "token_ids": token_ids,
        },
        "expert_pack": {
            "sha256": _sha256(args.expert_pack),
            "size_bytes": args.expert_pack.stat().st_size,
            "encoding": (
                "target_pack" if args.target_policy_id else "q4_group128"
            ),
            "target_policy_id": args.target_policy_id,
        },
        "execution": {
            "performance_mode": args.performance_mode,
            "native_platform": None if native is None else native.platform,
            "backend_cache": backend_cache,
        },
        "quality": {
            "reference_top1_token_id": reference_top1,
            "candidate_top1_token_id": candidate_top1,
            "logit_cosine_similarity": cosine,
            "logit_kl": logit_kl,
            "maximum_absolute_logit_error": max_abs_error,
            "layer2_moe_output_cosine_similarity": moe_cosine,
            "layer2_moe_output_maximum_absolute_error": moe_max_abs_error,
            "router_top4_exact_set_agreement": route_agreement,
            "per_layer_router_top4_exact_set_agreement": per_layer_route_agreement,
        },
        "loader": loader,
        "memory": {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
        },
        "gates": gates,
    }
    if not math.isfinite(cosine) or not math.isfinite(logit_kl):
        raise RuntimeError("Qwen Host quality metrics are not finite")
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
