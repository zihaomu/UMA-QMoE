#!/usr/bin/env python3
"""Validate mixed TargetPack through the fixed OLMoE Host correctness path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from run_mixed_precision_policy_search import (
    _capture_quality,
    _fixture_semantic_sha256,
    _load_samples,
    _metrics,
    _prepare_samples,
)
from run_mixed_precision_sensitivity import _routes_sha256, _tensor_sha256
from uma_qmoe.compressed_loader import load_fixed_olmoe
from uma_qmoe.contracts import canonical_sha256, validate_document


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
POLICY_ID = "olmoe-layer15-awq-q4-q8-bf16-v1"
Q4_LAYERS = [15]
Q8_LAYERS = [8, 11, 12, 13, 14]
BF16_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-pack", type=Path, required=True)
    parser.add_argument("--target-pack-manifest", type=Path, required=True)
    parser.add_argument("--policy-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_lists(layer_encodings: dict[str, str]) -> dict[str, list[int]]:
    return {
        "q4_layers": sorted(
            int(layer)
            for layer, encoding in layer_encodings.items()
            if encoding == "q4_group128"
        ),
        "q8_layers": sorted(
            int(layer)
            for layer, encoding in layer_encodings.items()
            if encoding == "q8_group128"
        ),
        "bf16_layers": sorted(
            int(layer)
            for layer, encoding in layer_encodings.items()
            if encoding == "bf16_le"
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite TargetPack Host quality evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    manifest = json.loads(args.target_pack_manifest.read_text(encoding="utf-8"))
    policy_evidence = json.loads(args.policy_evidence.read_text(encoding="utf-8"))
    validate_document(manifest)
    validate_document(policy_evidence)
    policy_semantic_sha256 = canonical_sha256(policy_evidence)
    manifest_semantic_sha256 = canonical_sha256(manifest)
    pack_header = manifest["header"]
    pack_policy = pack_header["policy"]
    layer_lists = _layer_lists(pack_policy["layer_encodings"])
    manifest_identity = (
        manifest.get("kind") == "target_pack_manifest"
        and manifest.get("status") == "verified"
        and pack_header["model"]["model_id"] == MODEL_ID
        and pack_header["model"]["model_revision"] == MODEL_REVISION
        and pack_header["model"]["model_manifest_sha256"]
        == args.model_manifest_sha256
        and manifest["artifact"]["size_bytes"] == args.target_pack.stat().st_size
        and manifest["artifact"]["sha256"] == _sha256_file(args.target_pack)
    )
    policy_identity = (
        policy_evidence.get("kind") == "activation_aware_mixed_policy"
        and policy_evidence.get("target_id") == args.target_id
        and policy_evidence.get("status") == "passed"
        and pack_policy["policy_id"] == POLICY_ID
        and pack_policy["policy_evidence_sha256"] == policy_semantic_sha256
        and layer_lists
        == {
            "q4_layers": Q4_LAYERS,
            "q8_layers": Q8_LAYERS,
            "bf16_layers": BF16_LAYERS,
        }
    )
    if not manifest_identity or not policy_identity:
        raise RuntimeError("TargetPack manifest or policy identity is incompatible")

    samples = _load_samples(args.prompt_fixture)
    evaluation_samples = samples[len(samples) // 2 :]
    evaluation_ids = [sample["id"] for sample in evaluation_samples]
    dataset_identity = evaluation_ids == policy_evidence["dataset"][
        "evaluation_sample_ids"
    ]
    if not dataset_identity:
        raise RuntimeError("TargetPack Host fixture differs from policy evidence")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("TargetPack Host validation requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    evaluation = _prepare_samples(evaluation_samples, tokenizer, torch)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    reference_model.eval()
    reference = _capture_quality(reference_model, evaluation, torch)
    del reference_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with load_fixed_olmoe(
        args.model,
        args.target_pack,
        model_manifest_sha256=args.model_manifest_sha256,
        device="cuda:0",
        performance_mode=False,
        target_policy_id=POLICY_ID,
    ) as host:
        candidate_capture = _capture_quality(host.model, evaluation, torch)
        loader = dict(host.evidence)
    metrics = _metrics(candidate_capture, reference, torch)
    quality_passed = (
        metrics["relative_perplexity_change"] <= 0.01
        and metrics["router_exact_set_agreement"] >= 0.99
    )
    gates = {
        "manifest_identity": manifest_identity,
        "policy_evidence_identity": policy_identity,
        "dataset_identity": dataset_identity,
        "reference_finite": bool(reference["finite"]),
        "candidate_finite": bool(metrics["finite"]),
        "quality_passed": quality_passed,
        "dense_only_checkpoint_load": (
            loader["skipped_expert_tensor_count"] == 3072
            and loader["loaded_expert_tensor_count"] == 0
        ),
        "no_expert_parameters": loader["expert_parameter_count"] == 0,
        "single_pack_mapping": (
            loader["expert_pack_mapping_count"] == 1
            and loader["expert_pack_vma_count"] in {1, -1}
        ),
        "full_model_executed": len(
            metrics["per_layer_router_exact_set_agreement"]
        )
        == 16,
    }
    gates["overall_passed"] = all(gates.values())
    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "target_pack_host_quality",
        "captured_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "target_id": args.target_id,
        "status": "passed" if gates["overall_passed"] else "failed",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_manifest_sha256": args.model_manifest_sha256,
        },
        "target_pack": {
            "artifact_sha256": manifest["artifact"]["sha256"],
            "artifact_size_bytes": manifest["artifact"]["size_bytes"],
            "manifest_file_sha256": _sha256_file(args.target_pack_manifest),
            "manifest_semantic_sha256": manifest_semantic_sha256,
            "policy_id": POLICY_ID,
            "policy_evidence_semantic_sha256": policy_semantic_sha256,
            "layer_encodings": layer_lists,
        },
        "dataset": {
            "fixture_file_sha256": _sha256_file(args.prompt_fixture),
            "fixture_semantic_sha256": _fixture_semantic_sha256(samples),
            "evaluation_sample_ids": evaluation_ids,
            "evaluation_prompt_token_count": reference["prompt_token_count"],
            "evaluation_target_token_count": reference["target_token_count"],
        },
        "reference": {
            "finite": bool(reference["finite"]),
            "nll": float(reference["nll"]),
            "perplexity": math.exp(float(reference["nll"])),
            "logits_sha256": _tensor_sha256(reference["logits"]),
            "routes_sha256": _routes_sha256(reference["routes"]),
        },
        "candidate": metrics,
        "loader": {
            name: loader[name]
            for name in (
                "performance_mode",
                "dense_tensor_count",
                "dense_tensor_bytes",
                "skipped_expert_tensor_count",
                "loaded_expert_tensor_count",
                "expert_parameter_count",
                "quantized_moe_block_count",
                "expert_pack_mapping_count",
                "expert_pack_vma_count",
            )
        },
        "quality_gate": {
            "maximum_relative_perplexity_increase": 0.01,
            "minimum_router_exact_set_agreement": 0.99,
        },
        "gates": gates,
    }
    validate_document(document)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not gates["overall_passed"]:
        raise RuntimeError("TargetPack Fixed Host quality gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
