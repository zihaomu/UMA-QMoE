#!/usr/bin/env python3
"""Capture and compare fixed Qwen completion-quality evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from uma_qmoe.experiments.evaluate import (
    aggregate_quality as _aggregate,
    capture_moe_causal_lm_quality,
    load_completion_samples,
    route_agreement,
)
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


def _samples(path: Path) -> list[dict[str, str]]:
    try:
        return load_completion_samples(path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "candidate"), required=True)
    parser.add_argument("--target-id", choices=("local-halo", "spark1"), required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--expert-pack", type=Path)
    parser.add_argument("--target-policy-id")
    parser.add_argument("--performance-mode", action="store_true")
    parser.add_argument("--native-build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _capture(model: Any, samples: list[dict[str, str]], tokenizer: Any, torch: Any):
    return capture_moe_causal_lm_quality(
        model,
        samples,
        tokenizer,
        num_layers=QWEN1_5_MOE.num_layers,
        device="cuda:0",
        torch_module=torch,
    )


def _route_agreement(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> tuple[float, list[float]]:
    try:
        exact, per_layer, _overlap = route_agreement(
            candidate, reference, num_layers=QWEN1_5_MOE.num_layers
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return exact, per_layer


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite Qwen completion-quality evidence")
    if args.mode == "reference" and (
        args.reference is not None or args.expert_pack is not None
    ):
        raise RuntimeError("reference mode does not accept reference or pack inputs")
    if args.mode == "candidate" and (
        args.reference is None or args.expert_pack is None
    ):
        raise RuntimeError("candidate mode requires --reference and --expert-pack")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"completion backend mismatch: requested {args.backend}, "
            f"observed {observed_backend}"
        )
    samples = _samples(args.prompt_fixture)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    native = None
    native_cache = None
    loader = None
    if args.mode == "reference":
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
            raise RuntimeError(f"unexpected fixed Qwen architecture {observed!r}")
        records, peak_allocated, peak_reserved = _capture(
            model, samples, tokenizer, torch
        )
    else:
        if args.performance_mode:
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
        with load_fixed_qwen(
            args.model,
            args.expert_pack,
            model_manifest_sha256=args.model_manifest_sha256,
            device="cuda:0",
            performance_mode=args.performance_mode,
            target_policy_id=args.target_policy_id,
        ) as host:
            records, peak_allocated, peak_reserved = _capture(
                host.model, samples, tokenizer, torch
            )
            loader = dict(host.evidence)
            native_cache = None if native is None else native.cache_summary()

    aggregate = _aggregate(records)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "kind": "qwen_completion_quality",
        "captured_at": _utc_now(),
        "target_id": args.target_id,
        "mode": args.mode,
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "model_manifest_sha256": args.model_manifest_sha256,
        },
        "fixture": {
            "path": str(args.prompt_fixture),
            "sha256": _sha256(args.prompt_fixture),
            "sample_ids": [sample["id"] for sample in samples],
        },
        "execution": {
            "backend": observed_backend,
            "performance_mode": args.performance_mode,
            "native_platform": None if native is None else native.platform,
        },
        "aggregate": aggregate,
        "records": records,
        "memory": {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_reserved_bytes": peak_reserved,
        },
    }
    if args.mode == "reference":
        evidence["status"] = "passed" if aggregate["finite"] else "failed"
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if (
            reference.get("kind") != "qwen_completion_quality"
            or reference.get("mode") != "reference"
            or reference.get("status") != "passed"
            or reference.get("model") != evidence["model"]
            or reference.get("fixture", {}).get("sha256")
            != evidence["fixture"]["sha256"]
        ):
            raise RuntimeError("Qwen completion reference identity mismatch")
        reference_aggregate = reference["aggregate"]
        route_agreement, per_layer = _route_agreement(
            records, reference["records"]
        )
        relative_ppl = (
            aggregate["perplexity"] / reference_aggregate["perplexity"] - 1.0
        )
        score_drop_points = 100.0 * (
            reference_aggregate["completion_token_accuracy"]
            - aggregate["completion_token_accuracy"]
        )
        gates = {
            "finite": aggregate["finite"],
            "relative_perplexity_increase_at_most_0_01": relative_ppl <= 0.01,
            "completion_score_drop_at_most_0_5_points": score_drop_points <= 0.5,
            "router_top4_set_agreement_at_least_0_99": route_agreement >= 0.99,
            "dense_only_checkpoint_load": (
                loader is not None
                and loader["skipped_expert_tensor_count"] == 4320
                and loader["loaded_expert_tensor_count"] == 0
            ),
            "no_dequantized_weight_cache": (
                native_cache is None
                or native_cache["dequantized_weight_bytes"] == 0
            ),
        }
        gates["overall_passed"] = all(gates.values())
        evidence.update(
            {
                "status": "passed" if gates["overall_passed"] else "failed",
                "reference": {
                    "path": str(args.reference),
                    "sha256": _sha256(args.reference),
                    "aggregate": reference_aggregate,
                },
                "comparison": {
                    "relative_perplexity_change": relative_ppl,
                    "completion_score_drop_points": score_drop_points,
                    "router_top4_exact_set_agreement": route_agreement,
                    "per_layer_router_top4_exact_set_agreement": per_layer,
                },
                "expert_pack": {
                    "path": str(args.expert_pack),
                    "sha256": _sha256(args.expert_pack),
                    "target_policy_id": args.target_policy_id,
                },
                "loader": loader,
                "backend_cache": native_cache,
                "gates": gates,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
