"""Deterministic dataset, route-coverage, and EBSS research evidence.

This module deliberately contains no Torch dependency.  GPU runners capture routing
observations and activation summaries; the control plane below validates and reduces
those observations into portable contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_sha256, validate_document
from .fixed_models import QWEN1_5_MOE


MIN_PROMPT_TOKENS = 32_768
MIN_TARGET_TOKENS = 4_096
MIN_EFFECTIVE_SAMPLES = 256.0
TARGET_ROUTED_TOKENS_P5 = 512
_HISTOGRAM_BINS = 2_048


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash dataset source {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read dataset source {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid JSON on {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise ContractError("dataset source contains no samples")
    return rows


def build_quantization_dataset_manifest(
    source_path: str | Path,
    *,
    partition: str,
    source_name: str,
    source_uri: str,
    source_revision: str,
    license_id: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_artifacts_sha256: str,
    encode: Callable[[str], Sequence[int]],
    frozen: bool = False,
) -> dict[str, Any]:
    """Build a token-exact dataset manifest from a JSONL prompt fixture.

    The source rows must contain ``id``, ``category``, and ``prompt``.  ``completion``
    is optional for calibration and mandatory for search/held-out partitions.
    Exact token IDs are intentionally retained so tokenizer drift cannot silently
    change calibration evidence.
    """

    if partition not in {"calibration", "search", "held_out"}:
        raise ContractError("partition must be calibration, search, or held_out")
    source = Path(source_path)
    rows = _load_jsonl(source)
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    duplicates_removed = 0
    for row_index, row in enumerate(rows):
        sample_id = row.get("id")
        category = row.get("category")
        prompt = row.get("prompt")
        completion = row.get("completion", "")
        if not isinstance(sample_id, str) or not sample_id:
            raise ContractError(f"dataset row {row_index} has no non-empty id")
        if sample_id in seen_ids:
            raise ContractError(f"duplicate dataset sample id {sample_id!r}")
        seen_ids.add(sample_id)
        if not isinstance(category, str) or not category:
            raise ContractError(f"dataset row {sample_id!r} has no category")
        if not isinstance(prompt, str) or not prompt:
            raise ContractError(f"dataset row {sample_id!r} has no prompt")
        if not isinstance(completion, str):
            raise ContractError(f"dataset row {sample_id!r} has invalid completion")
        if partition != "calibration" and not completion:
            raise ContractError(
                f"{partition} dataset row {sample_id!r} requires completion"
            )
        prompt_ids = [int(token) for token in encode(prompt)]
        target_ids = [int(token) for token in encode(completion)] if completion else []
        if not prompt_ids or any(token < 0 for token in prompt_ids + target_ids):
            raise ContractError(f"dataset row {sample_id!r} has invalid token IDs")
        text_hash = _sha256_text(prompt + "\0" + completion)
        if text_hash in seen_text:
            duplicates_removed += 1
            continue
        seen_text.add(text_hash)
        samples.append(
            {
                "sample_id": sample_id,
                "category": category,
                "source_text_sha256": text_hash,
                "prompt_sha256": _sha256_text(prompt),
                "target_sha256": _sha256_text(completion),
                "prompt_token_ids": prompt_ids,
                "target_token_ids": target_ids,
                "prompt_token_count": len(prompt_ids),
                "target_token_count": len(target_ids),
            }
        )

    prompt_tokens = sum(row["prompt_token_count"] for row in samples)
    target_tokens = sum(row["target_token_count"] for row in samples)
    required_prompt = MIN_PROMPT_TOKENS if partition == "calibration" else 0
    required_target = MIN_TARGET_TOKENS if partition != "calibration" else 0
    threshold_passed = (
        prompt_tokens >= required_prompt and target_tokens >= required_target
    )
    if frozen and not threshold_passed:
        raise ContractError("cannot freeze a dataset below the plan's token floor")
    document = {
        "schema_version": 1,
        "kind": "quantization_dataset_manifest",
        "created_at": _utc_now(),
        "status": "frozen" if frozen else "draft",
        "partition": partition,
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
        },
        "tokenizer": {
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "artifacts_sha256": tokenizer_artifacts_sha256,
            "add_special_tokens": False,
        },
        "source": {
            "name": source_name,
            "uri": source_uri,
            "revision": source_revision,
            "license": license_id,
            "file_sha256": _sha256_file(source),
        },
        "deduplication": {
            "algorithm": "sha256(prompt + NUL + completion)",
            "input_samples": len(rows),
            "duplicates_removed": duplicates_removed,
        },
        "samples": samples,
        "totals": {
            "samples": len(samples),
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "categories": sorted({row["category"] for row in samples}),
        },
        "token_floor": {
            "minimum_prompt_tokens": required_prompt,
            "minimum_target_tokens": required_target,
            "passed": threshold_passed,
        },
    }
    validate_document(document)
    return document


class _Histogram:
    def __init__(self, maximum: float) -> None:
        self.maximum = maximum
        self.bins = [0] * _HISTOGRAM_BINS
        self.count = 0
        self.total = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= self.maximum:
            raise ContractError("route statistic is outside its finite range")
        index = min(
            _HISTOGRAM_BINS - 1,
            int(value / self.maximum * _HISTOGRAM_BINS) if self.maximum else 0,
        )
        self.bins[index] += 1
        self.count += 1
        self.total += value

    def summary(self) -> dict[str, float]:
        if not self.count:
            return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}

        def percentile(q: float) -> float:
            rank = max(1, math.ceil(q * self.count))
            cumulative = 0
            for index, count in enumerate(self.bins):
                cumulative += count
                if cumulative >= rank:
                    return min(
                        self.maximum,
                        (index + 0.5) * self.maximum / _HISTOGRAM_BINS,
                    )
            raise AssertionError("histogram rank is unreachable")

        return {
            "mean": self.total / self.count,
            "p05": percentile(0.05),
            "p50": percentile(0.5),
            "p95": percentile(0.95),
        }


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _validate_observations(
    manifest: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> None:
    expected = {
        (sample["sample_id"], token_index): token_id
        for sample in manifest["samples"]
        for token_index, token_id in enumerate(sample["prompt_token_ids"])
    }
    observed: dict[tuple[str, int], int] = {}
    for row in observations:
        key = (row.get("sample_id"), row.get("token_index"))
        if key not in expected:
            raise ContractError(f"unexpected calibration token reference {key!r}")
        if key in observed:
            raise ContractError(f"duplicate calibration token reference {key!r}")
        token_id = row.get("token_id")
        if token_id != expected[key]:
            raise ContractError(f"token ID drift for calibration token {key!r}")
        layers = row.get("layers")
        if not isinstance(layers, list) or len(layers) != QWEN1_5_MOE.num_layers:
            raise ContractError(f"calibration token {key!r} has incomplete layers")
        for layer_index, layer in enumerate(layers):
            if layer.get("layer_index") != layer_index:
                raise ContractError("route observation layers are not canonical")
            experts = layer.get("expert_indices")
            weights = layer.get("routing_weights")
            if (
                not isinstance(experts, list)
                or not isinstance(weights, list)
                or len(experts) != QWEN1_5_MOE.top_k
                or len(weights) != QWEN1_5_MOE.top_k
                or len(set(experts)) != QWEN1_5_MOE.top_k
                or any(
                    isinstance(expert, bool)
                    or not isinstance(expert, int)
                    or not 0 <= expert < QWEN1_5_MOE.num_experts
                    for expert in experts
                )
                or any(
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or not 0.0 <= float(weight) <= 1.0
                    for weight in weights
                )
            ):
                raise ContractError("route observation Top-K payload is invalid")
            entropy = layer.get("router_entropy")
            margin = layer.get("route_margin")
            if (
                not isinstance(entropy, (int, float))
                or not math.isfinite(float(entropy))
                or not 0.0 <= float(entropy) <= math.log(QWEN1_5_MOE.num_experts)
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or not 0.0 <= float(margin) <= 1.0
            ):
                raise ContractError("route entropy or margin is invalid")
        observed[key] = token_id
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ContractError(f"calibration capture is missing {len(missing)} prompt tokens")


def route_capture_sha256(observations: Sequence[Mapping[str, Any]]) -> str:
    """Return the stable identity used to bind aggregate evidence to raw routes."""

    return _canonical_hash(observations)


def build_expert_calibration_coverage(
    dataset_manifest: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
    activation_statistics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce exact natural routes into all 24x60 explicit coverage states."""

    validate_document(dataset_manifest)
    if dataset_manifest["kind"] != "quantization_dataset_manifest":
        raise ContractError("coverage input must be a quantization dataset manifest")
    if dataset_manifest["partition"] != "calibration":
        raise ContractError("coverage input must use the calibration partition")
    _validate_observations(dataset_manifest, observations)

    counts = [
        [0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    sum_weights = [
        [0.0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    sum_squared_weights = [
        [0.0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    positions = [
        [[0] * QWEN1_5_MOE.top_k for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    weight_histograms = [
        [_Histogram(1.0) for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    margin_histograms = [
        [_Histogram(1.0) for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    entropy_histograms = [
        [_Histogram(math.log(QWEN1_5_MOE.num_experts)) for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]

    for observation in observations:
        for layer in observation["layers"]:
            layer_index = layer["layer_index"]
            for position, (expert, weight) in enumerate(
                zip(layer["expert_indices"], layer["routing_weights"], strict=True)
            ):
                value = float(weight)
                counts[layer_index][expert] += 1
                sum_weights[layer_index][expert] += value
                sum_squared_weights[layer_index][expert] += value * value
                positions[layer_index][expert][position] += 1
                weight_histograms[layer_index][expert].add(value)
                margin_histograms[layer_index][expert].add(float(layer["route_margin"]))
                entropy_histograms[layer_index][expert].add(
                    float(layer["router_entropy"])
                )

    activation_lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
    if activation_statistics is not None:
        for layer in activation_statistics:
            layer_index = layer.get("layer_index")
            for expert in layer.get("experts", []):
                key = (layer_index, expert.get("expert_index"))
                if (
                    isinstance(layer_index, bool)
                    or not isinstance(layer_index, int)
                    or not 0 <= layer_index < QWEN1_5_MOE.num_layers
                    or isinstance(key[1], bool)
                    or not isinstance(key[1], int)
                    or not 0 <= key[1] < QWEN1_5_MOE.num_experts
                ):
                    raise ContractError("activation statistic index is invalid")
                if key in activation_lookup:
                    raise ContractError("duplicate activation statistic")
                activation_lookup[key] = expert

    layer_rows = []
    eligible_units = 0
    empty_units = 0
    for layer_index in range(QWEN1_5_MOE.num_layers):
        expert_rows = []
        total_assignments = sum(counts[layer_index])
        for expert_index in range(QWEN1_5_MOE.num_experts):
            count = counts[layer_index][expert_index]
            sum_weight = sum_weights[layer_index][expert_index]
            sum_squared = sum_squared_weights[layer_index][expert_index]
            n_eff = (sum_weight * sum_weight / sum_squared) if sum_squared else 0.0
            eligible = count >= TARGET_ROUTED_TOKENS_P5 and n_eff >= MIN_EFFECTIVE_SAMPLES
            eligible_units += int(eligible)
            empty_units += int(count == 0)
            expert_rows.append(
                {
                    "expert_index": expert_index,
                    "status": "eligible" if eligible else "insufficient_coverage",
                    "routed_token_count": count,
                    "assignment_fraction": (
                        count / total_assignments if total_assignments else 0.0
                    ),
                    "sum_route_weight": sum_weight,
                    "sum_squared_route_weight": sum_squared,
                    "effective_samples": n_eff,
                    "route_probability": weight_histograms[layer_index][
                        expert_index
                    ].summary(),
                    "route_margin": margin_histograms[layer_index][
                        expert_index
                    ].summary(),
                    "router_entropy": entropy_histograms[layer_index][
                        expert_index
                    ].summary(),
                    "top_k_position_counts": positions[layer_index][expert_index],
                    "activations": activation_lookup.get(
                        (layer_index, expert_index)
                    ),
                }
            )
        layer_counts = counts[layer_index]
        layer_rows.append(
            {
                "layer_index": layer_index,
                "total_assignments": total_assignments,
                "routed_token_count_p05": _nearest_rank(layer_counts, 0.05),
                "routed_token_count_p50": _nearest_rank(layer_counts, 0.5),
                "routed_token_count_p95": _nearest_rank(layer_counts, 0.95),
                "experts": expert_rows,
            }
        )

    all_units = QWEN1_5_MOE.num_layers * QWEN1_5_MOE.num_experts
    token_floor_passed = dataset_manifest["token_floor"]["passed"]
    activation_complete = len(activation_lookup) == all_units
    overall_passed = (
        token_floor_passed
        and activation_complete
        and eligible_units == all_units
        and empty_units == 0
    )
    document = {
        "schema_version": 1,
        "kind": "expert_calibration_coverage",
        "captured_at": _utc_now(),
        "target_id": target_id,
        "status": "passed" if overall_passed else "insufficient_coverage",
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "num_layers": QWEN1_5_MOE.num_layers,
            "num_experts": QWEN1_5_MOE.num_experts,
            "top_k": QWEN1_5_MOE.top_k,
        },
        "inputs": {
            "dataset_manifest_sha256": canonical_sha256(dataset_manifest),
            "route_capture_sha256": route_capture_sha256(observations),
        },
        "method": {
            "id": "qwen-natural-route-coverage-v1",
            "natural_routes_only": True,
            "probability_quantiles": "deterministic-2048-bin-histogram-v1",
            "activation_statistics": (
                "fp32-raw-moments-plus-deterministic-strided-reservoir-8192-v1"
            ),
            "minimum_effective_samples": MIN_EFFECTIVE_SAMPLES,
            "target_routed_tokens_p05": TARGET_ROUTED_TOKENS_P5,
        },
        "summary": {
            "prompt_tokens": len(observations),
            "layer_expert_units": all_units,
            "eligible_units": eligible_units,
            "insufficient_units": all_units - eligible_units,
            "empty_units": empty_units,
        },
        "layers": layer_rows,
        "gates": {
            "dataset_token_floor_passed": token_floor_passed,
            "all_layer_experts_explicit": len(layer_rows) == QWEN1_5_MOE.num_layers
            and all(
                len(layer["experts"]) == QWEN1_5_MOE.num_experts
                for layer in layer_rows
            ),
            "no_empty_experts": empty_units == 0,
            "all_units_eligible": eligible_units == all_units,
            "activation_statistics_complete": activation_complete,
            "overall_passed": overall_passed,
        },
    }
    validate_document(document)
    return document


def _token_identity(observation: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "sample_id": observation["sample_id"],
            "token_index": observation["token_index"],
            "token_id": observation["token_id"],
        }
    )


def _contributions(observation: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    return [
        (layer["layer_index"], expert, float(weight))
        for layer in observation["layers"]
        for expert, weight in zip(
            layer["expert_indices"], layer["routing_weights"], strict=True
        )
    ]


def build_expert_balanced_sample_manifest(
    dataset_manifest: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    coverage_report: Mapping[str, Any],
    *,
    maximum_selected_tokens: int | None = None,
) -> dict[str, Any]:
    """Greedily select unique tokens using deficit-weighted router affinity.

    Lazy-greedy heap updates keep the selection practical for the formal 32K-token
    floor.  Scores monotonically decrease as expert deficits are filled; token hash
    is the deterministic tie-breaker.
    """

    validate_document(dataset_manifest)
    validate_document(coverage_report)
    _validate_observations(dataset_manifest, observations)
    if coverage_report["inputs"]["dataset_manifest_sha256"] != canonical_sha256(
        dataset_manifest
    ) or coverage_report["inputs"]["route_capture_sha256"] != route_capture_sha256(
        observations
    ):
        raise ContractError("coverage report is not bound to the EBSS inputs")
    if maximum_selected_tokens is None:
        maximum_selected_tokens = len(observations)
    if maximum_selected_tokens <= 0:
        raise ContractError("maximum_selected_tokens must be positive")

    contributions = [_contributions(row) for row in observations]
    identities = [_token_identity(row) for row in observations]
    counts = [
        [0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    sum_weights = [
        [0.0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]
    sum_squared = [
        [0.0 for _ in range(QWEN1_5_MOE.num_experts)]
        for _ in range(QWEN1_5_MOE.num_layers)
    ]

    def score(index: int) -> tuple[float, int]:
        affinity = 0.0
        gain = 0
        for layer, expert, weight in contributions[index]:
            deficit = max(0, TARGET_ROUTED_TOKENS_P5 - counts[layer][expert])
            if deficit:
                affinity += weight * deficit / TARGET_ROUTED_TOKENS_P5
                gain += 1
        return affinity, gain

    heap: list[tuple[float, str, int]] = []
    for index, identity in enumerate(identities):
        affinity, _ = score(index)
        heapq.heappush(heap, (-affinity, identity, index))

    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    epsilon = 1e-15
    while heap and len(selected) < min(maximum_selected_tokens, len(observations)):
        _negative_bound, identity, index = heapq.heappop(heap)
        if index in selected_indices:
            continue
        affinity, gain = score(index)
        next_bound = -heap[0][0] if heap else -1.0
        if affinity + epsilon < next_bound:
            heapq.heappush(heap, (-affinity, identity, index))
            continue
        if gain == 0:
            break
        selected_indices.add(index)
        observation = observations[index]
        for layer, expert, weight in contributions[index]:
            counts[layer][expert] += 1
            sum_weights[layer][expert] += weight
            sum_squared[layer][expert] += weight * weight
        selected.append(
            {
                "rank": len(selected),
                "sample_id": observation["sample_id"],
                "token_index": observation["token_index"],
                "token_id": observation["token_id"],
                "token_sha256": identity,
                "affinity_score_at_selection": affinity,
                "coverage_gain": gain,
            }
        )

    layers = []
    eligible_units = 0
    for layer_index in range(QWEN1_5_MOE.num_layers):
        expert_status = []
        for expert_index in range(QWEN1_5_MOE.num_experts):
            weight = sum_weights[layer_index][expert_index]
            squared = sum_squared[layer_index][expert_index]
            n_eff = weight * weight / squared if squared else 0.0
            eligible = (
                counts[layer_index][expert_index] >= TARGET_ROUTED_TOKENS_P5
                and n_eff >= MIN_EFFECTIVE_SAMPLES
            )
            eligible_units += int(eligible)
            expert_status.append(
                {
                    "expert_index": expert_index,
                    "routed_token_count": counts[layer_index][expert_index],
                    "effective_samples": n_eff,
                    "status": "eligible" if eligible else "insufficient_coverage",
                }
            )
        layers.append(
            {
                "layer_index": layer_index,
                "routed_token_count_p05": _nearest_rank(
                    counts[layer_index], 0.05
                ),
                "eligible_experts": sum(
                    row["status"] == "eligible" for row in expert_status
                ),
                "experts": expert_status,
            }
        )

    all_units = QWEN1_5_MOE.num_layers * QWEN1_5_MOE.num_experts
    passed = (
        dataset_manifest["token_floor"]["passed"] and eligible_units == all_units
    )
    algorithm = {
        "id": "deficit-weighted-router-affinity-lazy-greedy-v1",
        "minimum_effective_samples": MIN_EFFECTIVE_SAMPLES,
        "target_routed_tokens_p05": TARGET_ROUTED_TOKENS_P5,
        "maximum_selected_tokens": maximum_selected_tokens,
        "tie_breaker": "ascending token_sha256",
        "duplicate_tokens_allowed": False,
    }
    document = {
        "schema_version": 1,
        "kind": "expert_balanced_sample_manifest",
        "created_at": _utc_now(),
        "status": "passed" if passed else "insufficient_coverage",
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
        },
        "inputs": {
            "dataset_manifest_sha256": canonical_sha256(dataset_manifest),
            "coverage_report_sha256": canonical_sha256(coverage_report),
            "route_capture_sha256": route_capture_sha256(observations),
        },
        "algorithm": algorithm,
        "algorithm_sha256": _canonical_hash(algorithm),
        "selection": {
            "candidate_tokens": len(observations),
            "selected_tokens": len(selected),
            "selected_order_sha256": _canonical_hash(
                [row["token_sha256"] for row in selected]
            ),
            "tokens": selected,
        },
        "summary": {
            "layer_expert_units": all_units,
            "eligible_units": eligible_units,
            "insufficient_units": all_units - eligible_units,
        },
        "layers": layers,
        "gates": {
            "dataset_token_floor_passed": dataset_manifest["token_floor"]["passed"],
            "selection_is_unique": len(selected)
            == len({row["token_sha256"] for row in selected}),
            "all_layer_experts_explicit": len(layers) == QWEN1_5_MOE.num_layers
            and all(
                len(layer["experts"]) == QWEN1_5_MOE.num_experts
                for layer in layers
            ),
            "all_units_eligible": eligible_units == all_units,
            "overall_passed": passed,
        },
    }
    validate_document(document)
    return document


def split_disjointness_report(
    manifests: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a compact overlap audit for calibration/search/held-out manifests."""

    by_partition: dict[str, set[str]] = {}
    identities: dict[str, str] = {}
    for manifest in manifests:
        validate_document(manifest)
        if manifest["kind"] != "quantization_dataset_manifest":
            raise ContractError("split audit accepts dataset manifests only")
        partition = manifest["partition"]
        if partition in by_partition:
            raise ContractError(f"duplicate manifest partition {partition!r}")
        by_partition[partition] = {
            row["source_text_sha256"] for row in manifest["samples"]
        }
        identities[partition] = canonical_sha256(manifest)
    overlaps = []
    names = sorted(by_partition)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(by_partition[left] & by_partition[right])
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": len(shared),
                    "overlap_sha256": _canonical_hash(shared),
                }
            )
    return {
        "manifest_sha256": identities,
        "pairwise": overlaps,
        "passed": all(row["overlap_count"] == 0 for row in overlaps),
    }


def validate_quantization_dataset_manifest(
    document: Mapping[str, Any], *, require_frozen: bool = False
) -> None:
    """Enforce token-count, deduplication, and freeze invariants."""

    samples = document["samples"]
    sample_ids = [row["sample_id"] for row in samples]
    text_hashes = [row["source_text_sha256"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ContractError("QuantizationDatasetManifest sample IDs are not unique")
    if len(text_hashes) != len(set(text_hashes)):
        raise ContractError("QuantizationDatasetManifest still contains duplicate text")
    for row in samples:
        if row["prompt_token_count"] != len(row["prompt_token_ids"]):
            raise ContractError("dataset prompt token count is inconsistent")
        if row["target_token_count"] != len(row["target_token_ids"]):
            raise ContractError("dataset target token count is inconsistent")
    expected_totals = {
        "samples": len(samples),
        "prompt_tokens": sum(row["prompt_token_count"] for row in samples),
        "target_tokens": sum(row["target_token_count"] for row in samples),
        "categories": sorted({row["category"] for row in samples}),
    }
    if document["totals"] != expected_totals:
        raise ContractError("dataset totals do not match sample records")
    deduplication = document["deduplication"]
    if deduplication["input_samples"] != (
        len(samples) + deduplication["duplicates_removed"]
    ):
        raise ContractError("dataset deduplication counts are inconsistent")
    partition = document["partition"]
    expected_floor = {
        "minimum_prompt_tokens": MIN_PROMPT_TOKENS
        if partition == "calibration"
        else 0,
        "minimum_target_tokens": MIN_TARGET_TOKENS
        if partition != "calibration"
        else 0,
    }
    floor = document["token_floor"]
    if any(floor[name] != value for name, value in expected_floor.items()):
        raise ContractError("dataset token floors do not match the frozen plan")
    expected_passed = (
        expected_totals["prompt_tokens"] >= expected_floor["minimum_prompt_tokens"]
        and expected_totals["target_tokens"]
        >= expected_floor["minimum_target_tokens"]
    )
    if floor["passed"] != expected_passed:
        raise ContractError("dataset token-floor result is inconsistent")
    if partition != "calibration" and any(
        not row["target_token_ids"] for row in samples
    ):
        raise ContractError("search/held-out samples require target token IDs")
    if document["status"] == "frozen" and not expected_passed:
        raise ContractError("a below-floor dataset cannot be frozen")
    if require_frozen and document["status"] != "frozen":
        raise ContractError("QuantizationDatasetManifest is not frozen")


def validate_expert_calibration_coverage(document: Mapping[str, Any]) -> None:
    """Recompute coverage gates and all layer/expert arithmetic."""

    eligible_units = 0
    empty_units = 0
    activation_complete = True
    for layer_index, layer in enumerate(document["layers"]):
        if layer["layer_index"] != layer_index:
            raise ContractError("coverage layers are not in canonical order")
        counts: list[int] = []
        for expert_index, expert in enumerate(layer["experts"]):
            if expert["expert_index"] != expert_index:
                raise ContractError("coverage experts are not in canonical order")
            count = expert["routed_token_count"]
            counts.append(count)
            if sum(expert["top_k_position_counts"]) != count:
                raise ContractError("coverage Top-K position counts are inconsistent")
            expected_fraction = count / layer["total_assignments"]
            if not math.isclose(
                expert["assignment_fraction"],
                expected_fraction,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ContractError("coverage assignment fraction is inconsistent")
            squared = expert["sum_squared_route_weight"]
            weight = expert["sum_route_weight"]
            expected_n_eff = weight * weight / squared if squared else 0.0
            if not math.isclose(
                expert["effective_samples"],
                expected_n_eff,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ContractError("coverage effective sample count is inconsistent")
            eligible = (
                count >= TARGET_ROUTED_TOKENS_P5
                and expected_n_eff >= MIN_EFFECTIVE_SAMPLES
            )
            if expert["status"] != (
                "eligible" if eligible else "insufficient_coverage"
            ):
                raise ContractError("coverage expert status is inconsistent")
            eligible_units += int(eligible)
            empty_units += int(count == 0)
            activation_complete &= expert["activations"] is not None
            if expert["activations"] is not None and (
                expert["activations"]["expert_index"] != expert_index
            ):
                raise ContractError("activation statistic expert index is inconsistent")
        if sum(counts) != layer["total_assignments"]:
            raise ContractError("coverage layer assignment count is inconsistent")
        expected_quantiles = (
            _nearest_rank(counts, 0.05),
            _nearest_rank(counts, 0.5),
            _nearest_rank(counts, 0.95),
        )
        observed_quantiles = (
            layer["routed_token_count_p05"],
            layer["routed_token_count_p50"],
            layer["routed_token_count_p95"],
        )
        if observed_quantiles != expected_quantiles:
            raise ContractError("coverage layer count quantiles are inconsistent")
    all_units = QWEN1_5_MOE.num_layers * QWEN1_5_MOE.num_experts
    summary = document["summary"]
    if (
        summary["layer_expert_units"] != all_units
        or summary["eligible_units"] != eligible_units
        or summary["insufficient_units"] != all_units - eligible_units
        or summary["empty_units"] != empty_units
        or any(
            layer["total_assignments"]
            != summary["prompt_tokens"] * QWEN1_5_MOE.top_k
            for layer in document["layers"]
        )
    ):
        raise ContractError("coverage summary is inconsistent")
    gates = document["gates"]
    expected_overall = (
        gates["dataset_token_floor_passed"]
        and activation_complete
        and eligible_units == all_units
        and empty_units == 0
    )
    expected_gates = {
        "all_layer_experts_explicit": True,
        "no_empty_experts": empty_units == 0,
        "all_units_eligible": eligible_units == all_units,
        "activation_statistics_complete": activation_complete,
        "overall_passed": expected_overall,
    }
    if any(gates[name] != value for name, value in expected_gates.items()):
        raise ContractError("coverage gate result is inconsistent")
    if document["status"] != (
        "passed" if expected_overall else "insufficient_coverage"
    ):
        raise ContractError("coverage status is inconsistent")


def validate_expert_balanced_sample_manifest(document: Mapping[str, Any]) -> None:
    """Recompute EBSS ordering, coverage, and status invariants."""

    tokens = document["selection"]["tokens"]
    if document["selection"]["selected_tokens"] != len(tokens):
        raise ContractError("EBSS selected token count is inconsistent")
    if [row["rank"] for row in tokens] != list(range(len(tokens))):
        raise ContractError("EBSS ranks are not canonical")
    identities = [row["token_sha256"] for row in tokens]
    unique = len(identities) == len(set(identities))
    if document["selection"]["selected_order_sha256"] != _canonical_hash(
        identities
    ):
        raise ContractError("EBSS selected order hash is inconsistent")
    if document["algorithm_sha256"] != _canonical_hash(document["algorithm"]):
        raise ContractError("EBSS algorithm hash is inconsistent")
    eligible_units = 0
    for layer_index, layer in enumerate(document["layers"]):
        if layer["layer_index"] != layer_index:
            raise ContractError("EBSS layers are not in canonical order")
        counts = []
        layer_eligible = 0
        for expert_index, expert in enumerate(layer["experts"]):
            if expert["expert_index"] != expert_index:
                raise ContractError("EBSS experts are not in canonical order")
            counts.append(expert["routed_token_count"])
            eligible = (
                expert["routed_token_count"] >= TARGET_ROUTED_TOKENS_P5
                and expert["effective_samples"] >= MIN_EFFECTIVE_SAMPLES
            )
            if expert["status"] != (
                "eligible" if eligible else "insufficient_coverage"
            ):
                raise ContractError("EBSS expert status is inconsistent")
            layer_eligible += int(eligible)
        eligible_units += layer_eligible
        if (
            layer["eligible_experts"] != layer_eligible
            or layer["routed_token_count_p05"] != _nearest_rank(counts, 0.05)
        ):
            raise ContractError("EBSS layer summary is inconsistent")
    all_units = QWEN1_5_MOE.num_layers * QWEN1_5_MOE.num_experts
    summary = document["summary"]
    if (
        summary["layer_expert_units"] != all_units
        or summary["eligible_units"] != eligible_units
        or summary["insufficient_units"] != all_units - eligible_units
    ):
        raise ContractError("EBSS summary is inconsistent")
    gates = document["gates"]
    passed = gates["dataset_token_floor_passed"] and eligible_units == all_units
    expected_gates = {
        "selection_is_unique": unique,
        "all_layer_experts_explicit": True,
        "all_units_eligible": eligible_units == all_units,
        "overall_passed": passed,
    }
    if any(gates[name] != value for name, value in expected_gates.items()):
        raise ContractError("EBSS gate result is inconsistent")
    if document["status"] != (
        "passed" if passed else "insufficient_coverage"
    ):
        raise ContractError("EBSS status is inconsistent")


__all__ = [
    "MIN_EFFECTIVE_SAMPLES",
    "MIN_PROMPT_TOKENS",
    "MIN_TARGET_TOKENS",
    "TARGET_ROUTED_TOKENS_P5",
    "build_expert_balanced_sample_manifest",
    "build_expert_calibration_coverage",
    "build_quantization_dataset_manifest",
    "route_capture_sha256",
    "split_disjointness_report",
    "validate_expert_balanced_sample_manifest",
    "validate_expert_calibration_coverage",
    "validate_quantization_dataset_manifest",
]
