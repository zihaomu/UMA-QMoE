"""Shared causal-LM quality capture for Qwen exploration and formal runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_completion_samples(path: str | Path) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "id",
            "category",
            "prompt",
            "completion",
        }:
            raise ValueError(f"invalid completion fixture fields at line {line_number}")
        if not all(isinstance(item, str) and item for item in value.values()):
            raise ValueError(f"invalid completion fixture value at line {line_number}")
        if value["id"] in seen:
            raise ValueError(f"duplicate completion id {value['id']!r}")
        seen.add(value["id"])
        values.append(value)
    if len(values) < 2:
        raise ValueError("completion fixture must contain at least two samples")
    return values


def capture_moe_causal_lm_quality(
    model: Any,
    samples: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    num_layers: int,
    device: str,
    torch_module: Any | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Capture token NLL, top-1 predictions and per-layer routed expert sets."""

    if torch_module is None:
        import torch as torch_module

    torch = torch_module
    current_routes: dict[int, Any] = {}

    def gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("MoE gate returned an unexpected value")
            current_routes[layer_index] = result[2].detach().cpu()

        return hook

    handles = [
        layer.mlp.gate.register_forward_hook(gate_hook(layer_index))
        for layer_index, layer in enumerate(model.model.layers)
    ]
    records: list[dict[str, Any]] = []
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
            input_ids = torch.tensor(
                [prompt_ids + completion_ids], dtype=torch.long, device=device
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
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            if set(current_routes) != set(range(num_layers)):
                raise RuntimeError("quality capture missed an MoE router")
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
                        for layer in range(num_layers)
                    },
                    "finite": bool(torch.isfinite(logits).all().item()),
                }
            )
            if str(device).startswith("cuda"):
                peak_allocated = max(
                    peak_allocated, int(torch.cuda.max_memory_allocated(device))
                )
                peak_reserved = max(
                    peak_reserved, int(torch.cuda.max_memory_reserved(device))
                )
    finally:
        for handle in handles:
            handle.remove()
    return records, peak_allocated, peak_reserved


def aggregate_quality(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nll_values = [value for record in records for value in record["per_token_nll"]]
    targets = [value for record in records for value in record["target_token_ids"]]
    predictions = [
        value for record in records for value in record["completion_top1_token_ids"]
    ]
    if not nll_values or len(targets) != len(predictions):
        raise ValueError("quality records contain no aligned completion tokens")
    nll = sum(nll_values) / len(nll_values)
    return {
        "finite": all(record["finite"] for record in records),
        "sample_count": len(records),
        "target_token_count": len(targets),
        "nll": nll,
        "perplexity": math.exp(nll),
        "completion_token_accuracy": sum(
            left == right for left, right in zip(predictions, targets, strict=True)
        )
        / len(targets),
    }


def route_agreement(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
) -> tuple[float, list[float], float]:
    if [item["id"] for item in candidate] != [item["id"] for item in reference]:
        raise ValueError("candidate and reference completion IDs differ")
    exact = [0] * num_layers
    totals = [0] * num_layers
    overlap_total = 0.0
    for candidate_sample, reference_sample in zip(candidate, reference, strict=True):
        for layer in range(num_layers):
            candidate_rows = candidate_sample["routes"][str(layer)]
            reference_rows = reference_sample["routes"][str(layer)]
            if len(candidate_rows) != len(reference_rows):
                raise ValueError("candidate and reference route lengths differ")
            for candidate_row, reference_row in zip(
                candidate_rows, reference_rows, strict=True
            ):
                candidate_set = set(candidate_row)
                reference_set = set(reference_row)
                if not reference_set:
                    raise ValueError("reference route set must not be empty")
                exact[layer] += candidate_set == reference_set
                overlap_total += len(candidate_set & reference_set) / len(reference_set)
                totals[layer] += 1
    if not all(totals):
        raise ValueError("route records contain an empty layer")
    per_layer = [matches / count for matches, count in zip(exact, totals, strict=True)]
    decisions = sum(totals)
    return sum(exact) / decisions, per_layer, overlap_total / decisions


def compare_quality(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
) -> dict[str, Any]:
    aggregate = aggregate_quality(candidate)
    reference_aggregate = aggregate_quality(reference)
    exact, per_layer, overlap = route_agreement(
        candidate, reference, num_layers=num_layers
    )
    return {
        **aggregate,
        "relative_perplexity_change": (
            aggregate["perplexity"] / reference_aggregate["perplexity"] - 1.0
        ),
        "completion_score_drop_points": 100.0
        * (
            reference_aggregate["completion_token_accuracy"]
            - aggregate["completion_token_accuracy"]
        ),
        "router_exact_set_agreement": exact,
        "router_mean_set_overlap": overlap,
        "per_layer_router_exact_set_agreement": per_layer,
    }


@dataclass(frozen=True)
class QualityThresholds:
    maximum_relative_perplexity_change: float = 0.01
    maximum_completion_score_drop_points: float = 0.5
    minimum_router_exact_set_agreement: float = 0.99

    def to_dict(self) -> dict[str, float | str]:
        return {
            "id": "moe-causal-lm-quality-v1",
            "maximum_relative_perplexity_change": self.maximum_relative_perplexity_change,
            "maximum_completion_score_drop_points": self.maximum_completion_score_drop_points,
            "minimum_router_exact_set_agreement": self.minimum_router_exact_set_agreement,
        }


class RelativeQualityEvaluator:
    """Capture one reference, then compare every restored session candidate to it."""

    def __init__(
        self,
        model: Any,
        samples: Sequence[Mapping[str, str]],
        tokenizer: Any,
        *,
        num_layers: int,
        device: str,
        torch_module: Any,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        self.model = model
        self.samples = samples
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.device = device
        self.torch = torch_module
        self.thresholds = thresholds or QualityThresholds()
        self.reference, _allocated, _reserved = capture_moe_causal_lm_quality(
            model,
            samples,
            tokenizer,
            num_layers=num_layers,
            device=device,
            torch_module=torch_module,
        )

    def evaluate(self) -> dict[str, Any]:
        records, _allocated, _reserved = capture_moe_causal_lm_quality(
            self.model,
            self.samples,
            self.tokenizer,
            num_layers=self.num_layers,
            device=self.device,
            torch_module=self.torch,
        )
        return compare_quality(records, self.reference, num_layers=self.num_layers)

    def gate(self, _offline: Mapping[str, Any], quality: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "finite": bool(quality["finite"]),
            "relative_perplexity_change": quality["relative_perplexity_change"]
            <= self.thresholds.maximum_relative_perplexity_change,
            "completion_score_drop": quality["completion_score_drop_points"]
            <= self.thresholds.maximum_completion_score_drop_points,
            "router_exact_set_agreement": quality["router_exact_set_agreement"]
            >= self.thresholds.minimum_router_exact_set_agreement,
        }


__all__ = [
    "QualityThresholds",
    "RelativeQualityEvaluator",
    "aggregate_quality",
    "capture_moe_causal_lm_quality",
    "compare_quality",
    "file_sha256",
    "load_completion_samples",
    "route_agreement",
]
