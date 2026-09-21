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
    values = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if set(value) != {"id", "category", "prompt", "completion"}:
            raise RuntimeError(
                f"invalid completion fixture fields at line {line_number}"
            )
        if not all(isinstance(item, str) and item for item in value.values()):
            raise RuntimeError(
                f"invalid completion fixture value at line {line_number}"
            )
        if value["id"] in seen:
            raise RuntimeError(f"duplicate completion id {value['id']!r}")
        seen.add(value["id"])
        values.append(value)
    if len(values) < 2:
        raise RuntimeError("completion fixture must contain at least two samples")
    return values


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
    current_routes: dict[int, Any] = {}

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("Qwen gate returned an unexpected value")
            current_routes[layer_index] = result[2].detach().cpu()

        return hook

    handles = [
        layer.mlp.gate.register_forward_hook(gate_hook(layer_index))
        for layer_index, layer in enumerate(model.model.layers)
    ]
    records = []
    peak_allocated = 0
    peak_reserved = 0
    try:
        for sample in samples:
            prompt_ids = tokenizer(
                sample["prompt"], add_special_tokens=False
            ).input_ids
            completion_ids = tokenizer(
                sample["completion"], add_special_tokens=False
            ).input_ids
            if not prompt_ids or not completion_ids:
                raise RuntimeError(f"sample {sample['id']!r} tokenized empty")
            token_ids = prompt_ids + completion_ids
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device="cuda:0"
            )
            current_routes.clear()
            with torch.inference_mode():
                result = model(input_ids=input_ids, use_cache=False)
                logits = result.logits.float()
                selected = logits[:, len(prompt_ids) - 1 : -1, :]
                targets = input_ids[:, len(prompt_ids) :]
                losses = torch.nn.functional.cross_entropy(
                    selected.reshape(-1, selected.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                )
                predictions = selected.argmax(dim=-1)
            torch.cuda.synchronize()
            if set(current_routes) != set(range(QWEN1_5_MOE.num_layers)):
                raise RuntimeError("completion capture missed a Qwen router")
            records.append(
                {
                    "id": sample["id"],
                    "category": sample["category"],
                    "prompt_token_count": len(prompt_ids),
                    "target_token_ids": completion_ids,
                    "per_token_nll": losses.detach().cpu().tolist(),
                    "completion_top1_token_ids": predictions[0].detach().cpu().tolist(),
                    "routes": {
                        str(layer): current_routes[layer].tolist()
                        for layer in range(QWEN1_5_MOE.num_layers)
                    },
                    "finite": bool(torch.isfinite(logits).all().item()),
                }
            )
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated()))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
    finally:
        for handle in handles:
            handle.remove()
    return records, peak_allocated, peak_reserved


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    nll_values = [value for record in records for value in record["per_token_nll"]]
    targets = [value for record in records for value in record["target_token_ids"]]
    predictions = [
        value for record in records for value in record["completion_top1_token_ids"]
    ]
    nll = sum(nll_values) / len(nll_values)
    return {
        "finite": all(record["finite"] for record in records),
        "sample_count": len(records),
        "target_token_count": len(targets),
        "nll": nll,
        "perplexity": math.exp(nll),
        "completion_token_accuracy": (
            sum(left == right for left, right in zip(predictions, targets, strict=True))
            / len(targets)
        ),
    }


def _route_agreement(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> tuple[float, list[float]]:
    if [item["id"] for item in candidate] != [item["id"] for item in reference]:
        raise RuntimeError("candidate and reference completion IDs differ")
    exact = [0] * QWEN1_5_MOE.num_layers
    totals = [0] * QWEN1_5_MOE.num_layers
    for candidate_sample, reference_sample in zip(candidate, reference, strict=True):
        for layer in range(QWEN1_5_MOE.num_layers):
            candidate_rows = candidate_sample["routes"][str(layer)]
            reference_rows = reference_sample["routes"][str(layer)]
            if len(candidate_rows) != len(reference_rows):
                raise RuntimeError("candidate and reference route lengths differ")
            for candidate_row, reference_row in zip(
                candidate_rows, reference_rows, strict=True
            ):
                exact[layer] += set(candidate_row) == set(reference_row)
                totals[layer] += 1
    per_layer = [matches / count for matches, count in zip(exact, totals, strict=True)]
    return sum(exact) / sum(totals), per_layer


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
