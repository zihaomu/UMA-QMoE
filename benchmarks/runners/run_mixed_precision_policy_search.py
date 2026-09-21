#!/usr/bin/env python3
"""Evaluate cumulative layer-2 BF16 expert prefixes across fixed completions."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from run_mixed_precision_refinement import (
    _copy_bf16_expert,
    _storage_by_prefix,
)
from run_mixed_precision_sensitivity import (
    EXPERT_COUNT,
    LAYER_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    _copy_q4_layer,
    _routes_sha256,
    _sha256_file,
    _tensor_sha256,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.expert_pack import ExpertPackReader
from uma_qmoe.numeric import clamp_cosine_similarity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-id", choices=("halo3", "local-halo", "spark1"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--refinement-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--expert-order", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _expert_order(value: str) -> list[int]:
    try:
        order = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expert order must be comma-separated ints"
        ) from exc
    if (
        len(order) < 2
        or order[:2] != [30, 26]
        or len(order) != len(set(order))
        or any(expert < 0 or expert >= EXPERT_COUNT for expert in order)
    ):
        raise argparse.ArgumentTypeError(
            "expert order must be unique layer-2 ids beginning with 30,26"
        )
    return order


def _load_samples(path: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if set(value) != {"id", "category", "prompt", "completion"}:
            raise RuntimeError(f"invalid quality fixture fields at line {line_number}")
        if not all(isinstance(value[name], str) and value[name] for name in value):
            raise RuntimeError(f"invalid quality fixture value at line {line_number}")
        if value["id"] in seen:
            raise RuntimeError(f"duplicate quality fixture id {value['id']!r}")
        seen.add(value["id"])
        samples.append(value)
    if len(samples) < 2:
        raise RuntimeError("quality fixture must contain at least two samples")
    return samples


def _fixture_semantic_sha256(samples: list[dict[str, str]]) -> str:
    payload = json.dumps(
        samples, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_samples(
    samples: list[dict[str, str]], tokenizer: Any, torch: Any
) -> list[dict[str, Any]]:
    prepared = []
    for sample in samples:
        prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False).input_ids
        completion_ids = tokenizer(
            sample["completion"], add_special_tokens=False
        ).input_ids
        if not prompt_ids or not completion_ids:
            raise RuntimeError(f"quality sample {sample['id']!r} tokenized empty")
        input_ids = torch.tensor(
            [prompt_ids + completion_ids], dtype=torch.long, device="cuda:0"
        )
        prepared.append(
            {
                "id": sample["id"],
                "input_ids": input_ids,
                "prompt_tokens": len(prompt_ids),
                "target_tokens": len(completion_ids),
            }
        )
    return prepared


def _capture_quality(
    model: Any, prepared: list[dict[str, Any]], torch: Any
) -> dict[str, Any]:
    routes: dict[int, list[Any]] = {layer: [] for layer in range(LAYER_COUNT)}
    last_logits = []
    total_nll = 0.0
    target_token_count = 0
    prompt_token_count = 0
    finite = True
    current_routes: dict[int, Any] = {}

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("OLMoE router hook returned an unexpected value")
            current_routes[layer_index] = result[2].detach().cpu().reshape(-1, 8)

        return hook

    handles = [
        layer.mlp.gate.register_forward_hook(gate_hook(layer_index))
        for layer_index, layer in enumerate(model.model.layers)
    ]
    try:
        for sample in prepared:
            current_routes.clear()
            input_ids = sample["input_ids"]
            prompt_tokens = sample["prompt_tokens"]
            target_tokens = sample["target_tokens"]
            with torch.inference_mode():
                result = model(input_ids=input_ids, use_cache=False)
                logits = result.logits.float()
                selected = logits[:, prompt_tokens - 1 : -1, :]
                targets = input_ids[:, prompt_tokens:]
                loss = torch.nn.functional.cross_entropy(
                    selected.reshape(-1, selected.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
            torch.cuda.synchronize()
            if set(current_routes) != set(range(LAYER_COUNT)):
                raise RuntimeError("quality capture did not observe all OLMoE routers")
            finite = finite and bool(torch.isfinite(logits).all().item())
            total_nll += float(loss.item())
            target_token_count += target_tokens
            prompt_token_count += prompt_tokens
            last_logits.append(logits[:, -1, :].detach().cpu())
            for layer_index in range(LAYER_COUNT):
                routes[layer_index].append(current_routes[layer_index])
    finally:
        for handle in handles:
            handle.remove()
    return {
        "finite": finite,
        "nll": total_nll / target_token_count,
        "target_token_count": target_token_count,
        "prompt_token_count": prompt_token_count,
        "logits": torch.cat(last_logits, dim=0),
        "routes": {
            layer: torch.cat(layer_routes, dim=0)
            for layer, layer_routes in routes.items()
        },
    }


def _metrics(
    candidate: dict[str, Any], reference: dict[str, Any], torch: Any
) -> dict[str, Any]:
    candidate_logits = candidate["logits"]
    reference_logits = reference["logits"]
    absolute = (candidate_logits - reference_logits).abs().reshape(-1)
    p99_index = max(0, math.ceil(0.99 * absolute.numel()) - 1)
    cosine = clamp_cosine_similarity(
        float(
            torch.nn.functional.cosine_similarity(
                candidate_logits.reshape(1, -1), reference_logits.reshape(1, -1)
            ).item()
        )
    )
    exact_count = 0
    decision_count = 0
    overlap_total = 0
    per_layer = []
    for layer_index in range(LAYER_COUNT):
        candidate_routes = candidate["routes"][layer_index]
        reference_routes = reference["routes"][layer_index]
        exact = (
            candidate_routes.sort(dim=-1).values == reference_routes.sort(dim=-1).values
        ).all(dim=-1)
        intersection = (
            (candidate_routes[:, :, None] == reference_routes[:, None, :])
            .any(dim=-1)
            .sum(dim=-1)
        )
        layer_decisions = int(exact.numel())
        layer_exact = int(exact.sum().item())
        per_layer.append(layer_exact / layer_decisions)
        exact_count += layer_exact
        decision_count += layer_decisions
        overlap_total += int(intersection.sum().item())
    nll = float(candidate["nll"])
    perplexity = math.exp(nll)
    reference_nll = float(reference["nll"])
    reference_perplexity = math.exp(reference_nll)
    top1 = candidate_logits.argmax(dim=-1)
    reference_top1 = reference_logits.argmax(dim=-1)
    return {
        "finite": bool(candidate["finite"]),
        "nll": nll,
        "perplexity": perplexity,
        "relative_nll_change": (nll / reference_nll) - 1.0,
        "relative_perplexity_change": (perplexity / reference_perplexity) - 1.0,
        "top1_agreement": float((top1 == reference_top1).float().mean().item()),
        "logit_max_absolute_error": float(absolute.max().item()),
        "logit_p99_absolute_error": float(
            torch.sort(absolute).values[p99_index].item()
        ),
        "logit_cosine_similarity": cosine,
        "router_exact_set_agreement": exact_count / decision_count,
        "router_mean_set_overlap": overlap_total / (decision_count * 8),
        "per_layer_router_exact_set_agreement": per_layer,
    }


def main() -> int:
    args = _parser().parse_args()
    order = _expert_order(args.expert_order)
    if args.output.exists():
        raise SystemExit("refusing to overwrite mixed-precision policy evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.refinement_evidence.read_text(encoding="utf-8"))
    validate_document(source)
    if (
        source.get("kind") != "mixed_precision_refinement"
        or source.get("target_id") != args.target_id
        or source.get("status") != "passed"
    ):
        raise RuntimeError("refinement evidence identity is incompatible")
    samples = _load_samples(args.prompt_fixture)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("mixed-precision policy search requires a BF16 device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prepared = _prepare_samples(samples, tokenizer, torch)
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
    if observed != ("olmoe", LAYER_COUNT, EXPERT_COUNT, 8):
        raise RuntimeError(f"unexpected fixed OLMoE architecture {observed!r}")

    reference = _capture_quality(model, prepared, torch)
    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    pack_sha256 = _sha256_file(args.expert_pack)
    if source["model"]["expert_pack_sha256"] != pack_sha256:
        raise RuntimeError("refinement evidence used a different ExpertPack")
    headers = reader.header["tensors"]
    total_weight_count = sum(item["element_count"] for item in headers)
    expert_storage = {
        expert: _storage_by_prefix(headers, f"model.layers.2.mlp.experts.{expert}.")
        for expert in order
    }
    extra_values = {bf16 - q4 for q4, bf16, _weights in expert_storage.values()}
    if len(extra_values) != 1:
        raise RuntimeError("selected experts do not have identical storage shape")
    single_expert_extra = extra_values.pop()
    all_q4_bytes = args.expert_pack.stat().st_size
    try:
        with ExitStack() as stack:
            sources = {
                shard: stack.enter_context(
                    safe_open(args.model / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(weight_map.values()))
            }
            for layer_index in range(LAYER_COUNT):
                print(f"quantizing layer {layer_index}/15", flush=True)
                _copy_q4_layer(model, reader, layer_index, torch)
            all_q4_capture = _capture_quality(model, prepared, torch)
            all_q4 = _metrics(all_q4_capture, reference, torch)
            rows = []
            restored: list[int] = []
            for expert_index in order:
                print(f"adding BF16 layer2 expert {expert_index}", flush=True)
                _copy_bf16_expert(model, sources, weight_map, 2, expert_index, torch)
                restored.append(expert_index)
                capture = _capture_quality(model, prepared, torch)
                metrics = _metrics(capture, reference, torch)
                extra_bytes = len(restored) * single_expert_extra
                mixed_bytes = all_q4_bytes + extra_bytes
                rows.append(
                    {
                        "policy_id": "layer2-"
                        + "-".join(f"e{expert}" for expert in restored),
                        "added_expert": expert_index,
                        "restored_experts": list(restored),
                        "metrics": metrics,
                        "extra_bytes": extra_bytes,
                        "mixed_bytes": mixed_bytes,
                        "effective_bpw": mixed_bytes * 8 / total_weight_count,
                        "finite": bool(metrics["finite"]),
                    }
                )
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"] <= 0.01
            and row["metrics"]["router_exact_set_agreement"] >= 0.99
        ]
        gates = {
            "reference_finite": bool(reference["finite"]),
            "all_q4_finite": bool(all_q4["finite"]),
            "dataset_complete": len(samples) >= 2
            and reference["target_token_count"] >= 2,
            "prefix_progression": len(rows) == len(order),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": True,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "mixed_precision_policy_search",
            "captured_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "target_id": args.target_id,
            "status": "passed" if gates["overall_passed"] else "failed",
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expert_pack_sha256": pack_sha256,
            },
            "method": {
                "id": "cumulative-layer2-expert-prefix-quality-v1",
                "diagnostic_only": True,
                "materializes_expert_parameters": True,
                "performance_evidence": False,
                "expert_layer": 2,
                "expert_order": order,
                "source_refinement_evidence": {
                    "file_sha256": _sha256_file(args.refinement_evidence),
                    "semantic_sha256": canonical_sha256(source),
                },
                "prompt_fixture": {
                    "file_sha256": _sha256_file(args.prompt_fixture),
                    "semantic_sha256": _fixture_semantic_sha256(samples),
                },
            },
            "dataset": {
                "sample_ids": [sample["id"] for sample in samples],
                "sample_count": len(samples),
                "prompt_token_count": reference["prompt_token_count"],
                "target_token_count": reference["target_token_count"],
            },
            "reference": {
                "finite": bool(reference["finite"]),
                "nll": float(reference["nll"]),
                "perplexity": math.exp(float(reference["nll"])),
                "logits_sha256": _tensor_sha256(reference["logits"]),
                "routes_sha256": _routes_sha256(reference["routes"]),
            },
            "all_q4_baseline": all_q4,
            "storage": {
                "all_q4_bytes": all_q4_bytes,
                "total_expert_weight_count": total_weight_count,
                "single_expert_extra_bytes": single_expert_extra,
            },
            "candidate_rows": rows,
            "quality_gate": {
                "maximum_relative_perplexity_increase": 0.01,
                "minimum_router_exact_set_agreement": 0.99,
                "first_passing_policy_id": (
                    passing[0]["policy_id"] if passing else None
                ),
            },
            "gates": gates,
        }
        validate_document(document)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not gates["overall_passed"]:
            raise RuntimeError("mixed-precision policy integrity gate failed")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
