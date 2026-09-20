"""Build, validate, hash, and replay fixed-model RouteTrace evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from .contracts import ContractError, canonical_sha256, validate_document
from .fixed_models import OLMOE, FixedModelSpec, fixed_model_spec


# Backward-compatible constants for callers that still name the canonical
# OLMoE v1 trace dimensions directly.
MODEL_ID = OLMOE.model_id
MODEL_REVISION = OLMOE.model_revision
NUM_LAYERS = OLMOE.num_layers
NUM_EXPERTS = OLMOE.num_experts
TOP_K = OLMOE.top_k


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read RouteTrace capture {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("RouteTrace capture root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash RouteTrace capture {path}: {exc}") from exc
    return digest.hexdigest()


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} must be an integer of at least {minimum}")
    return value


def _identity(value: Any, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{field} must be a lowercase {length}-character hex identity")
    return value


def _finite_weight(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ContractError(f"{field} must be finite and in [0, 1]")
    return float(value)


def _payload_hash(
    layers: list[dict[str, Any]], *, schema_version: int
) -> str:
    digest = hashlib.sha256()
    digest.update(f"UMA-QMoE.RouteTrace.v{schema_version}\0".encode())
    for layer in layers:
        digest.update(struct.pack("<H", layer["layer_index"]))
        for event in layer["events"]:
            digest.update(struct.pack("<II", event["event_index"], event["token_count"]))
            for expert in event["expert_indices"]:
                digest.update(struct.pack("<B", expert))
            for weight in event["routing_weights"]:
                digest.update(struct.pack("<d", weight))
    return digest.hexdigest()


def _nearest_rank(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _statistics(counts: list[int], *, top_k: int) -> dict[str, Any]:
    num_experts = len(counts)
    ranked_hot = sorted(range(num_experts), key=lambda expert: (-counts[expert], expert))
    nonempty = [expert for expert in range(num_experts) if counts[expert] > 0]
    ranked_tail = sorted(nonempty, key=lambda expert: (counts[expert], expert))
    empty = [expert for expert in range(num_experts) if counts[expert] == 0]
    return {
        "tokens_per_expert": counts,
        "hot_experts": ranked_hot[:top_k],
        "long_tail_experts": ranked_tail[:top_k],
        "empty_experts": empty,
        "minimum_tokens": min(counts),
        "maximum_tokens": max(counts),
        "mean_tokens": sum(counts) / num_experts,
        "p95_tokens": _nearest_rank(counts, 0.95),
    }


def _normalize_events(raw_events: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_events, list) or len(raw_events) < 2:
        raise ContractError("RouteTrace capture requires prefill and decode events")
    events: list[dict[str, Any]] = []
    for index, item in enumerate(raw_events):
        if not isinstance(item, dict):
            raise ContractError(f"RouteTrace event {index} must be an object")
        event_index = _integer(item.get("event_index"), "event_index")
        if event_index != index:
            raise ContractError("RouteTrace event indices must be contiguous from zero")
        phase = item.get("phase")
        if phase not in {"prefill", "decode"}:
            raise ContractError("RouteTrace event phase must be prefill or decode")
        decode_step = item.get("decode_step")
        if phase == "prefill" and decode_step is not None:
            raise ContractError("prefill RouteTrace event must have null decode_step")
        if phase == "decode" and _integer(decode_step, "decode_step") != index - 1:
            raise ContractError("decode steps must be contiguous after prefill")
        batch_size = _integer(item.get("batch_size"), "batch_size", 1)
        tokens_per_sequence = _integer(
            item.get("tokens_per_sequence"), "tokens_per_sequence", 1
        )
        events.append(
            {
                "event_index": event_index,
                "phase": phase,
                "decode_step": decode_step,
                "batch_size": batch_size,
                "tokens_per_sequence": tokens_per_sequence,
                "token_count": batch_size * tokens_per_sequence,
            }
        )
    if events[0]["phase"] != "prefill" or any(
        event["phase"] != "decode" for event in events[1:]
    ):
        raise ContractError("RouteTrace must contain one prefill followed by decode events")
    return events


def _normalize_layers(
    raw_layers: Any,
    events: list[dict[str, Any]],
    spec: FixedModelSpec,
) -> list[dict[str, Any]]:
    if not isinstance(raw_layers, list) or len(raw_layers) != spec.num_layers:
        raise ContractError(
            f"RouteTrace capture must contain exactly {spec.num_layers} layers"
        )
    layers: list[dict[str, Any]] = []
    for layer_index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, dict) or raw_layer.get("layer_index") != layer_index:
            raise ContractError("RouteTrace layers must be ordered contiguously")
        raw_layer_events = raw_layer.get("events")
        if not isinstance(raw_layer_events, list) or len(raw_layer_events) != len(events):
            raise ContractError("each RouteTrace layer must cover every event")
        counts = [0] * spec.num_experts
        normalized_layer_events: list[dict[str, Any]] = []
        total_tokens = 0
        for event, raw_event in zip(events, raw_layer_events, strict=True):
            if not isinstance(raw_event, dict):
                raise ContractError("RouteTrace layer event must be an object")
            if raw_event.get("event_index") != event["event_index"]:
                raise ContractError("RouteTrace layer event index mismatch")
            token_count = event["token_count"]
            experts = raw_event.get("expert_indices")
            weights = raw_event.get("routing_weights")
            expected_values = token_count * spec.top_k
            if (
                not isinstance(experts, list)
                or not isinstance(weights, list)
                or len(experts) != expected_values
                or len(weights) != expected_values
            ):
                raise ContractError("RouteTrace route arrays do not match token_count × Top-K")
            normalized_experts = [
                _integer(value, "expert index") for value in experts
            ]
            if any(value >= spec.num_experts for value in normalized_experts):
                raise ContractError("RouteTrace expert index exceeds model expert count")
            normalized_weights = [
                _finite_weight(value, "routing weight") for value in weights
            ]
            for offset in range(0, expected_values, spec.top_k):
                row = normalized_experts[offset : offset + spec.top_k]
                if len(set(row)) != spec.top_k:
                    raise ContractError("RouteTrace Top-K row contains duplicate experts")
                for expert in row:
                    counts[expert] += 1
            normalized_layer_events.append(
                {
                    "event_index": event["event_index"],
                    "token_count": token_count,
                    "expert_indices": normalized_experts,
                    "routing_weights": normalized_weights,
                }
            )
            total_tokens += token_count
        layers.append(
            {
                "layer_index": layer_index,
                "top_k": spec.top_k,
                "token_count": total_tokens,
                "events": normalized_layer_events,
                "expert_statistics": _statistics(counts, top_k=spec.top_k),
            }
        )
    return layers


def build_route_trace(capture_path: str | Path, *, trace_id: str) -> dict[str, Any]:
    """Normalize a runner capture into an immutable RouteTrace document."""

    source = Path(capture_path)
    raw = _load(source)
    spec = fixed_model_spec(raw.get("model_id"), raw.get("model_revision"))
    architecture = raw.get("architecture")
    if architecture != {
        "num_layers": spec.num_layers,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
    }:
        raise ContractError("RouteTrace capture architecture does not match fixed model")
    events = _normalize_events(raw.get("events"))
    layers = _normalize_layers(raw.get("layers"), events, spec)
    workload = raw.get("workload")
    fixture = raw.get("fixture")
    capture = raw.get("capture")
    if not all(isinstance(item, dict) for item in (workload, fixture, capture)):
        raise ContractError("RouteTrace workload, fixture, and capture must be objects")
    if workload.get("batch_size") != events[0]["batch_size"]:
        raise ContractError("RouteTrace workload batch size does not match events")
    if workload.get("prompt_tokens") != events[0]["tokens_per_sequence"]:
        raise ContractError("RouteTrace prompt length does not match prefill")
    # Greedy generation produces the first output token from the prefill and
    # then executes output_tokens - 1 cached decode calls.
    if workload.get("output_tokens") != len(events):
        raise ContractError("RouteTrace output length does not match generation events")

    tokens_per_layer = sum(event["token_count"] for event in events)
    document = {
        "schema_version": spec.trace_schema_version,
        "kind": "route_trace",
        "trace_id": trace_id,
        "status": "frozen",
        "captured_at": raw.get("captured_at", _utc_now()),
        "source_capture_sha256": _sha256(source),
        "model": {
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            **architecture,
        },
        "fixture": {
            "id": fixture.get("id"),
            "sha256": _identity(fixture.get("sha256"), "fixture.sha256", 64),
            "token_ids_sha256": _identity(
                fixture.get("token_ids_sha256"), "fixture.token_ids_sha256", 64
            ),
        },
        "workload": {
            "batch_size": _integer(workload.get("batch_size"), "batch_size", 1),
            "prompt_tokens": _integer(workload.get("prompt_tokens"), "prompt_tokens", 1),
            "output_tokens": _integer(workload.get("output_tokens"), "output_tokens", 1),
            "decoding": "greedy",
            "ignore_eos": True,
        },
        "capture": {
            "target_id": capture.get("target_id"),
            "backend": capture.get("backend"),
            "torch_version": capture.get("torch_version"),
            "transformers_version": capture.get("transformers_version"),
        },
        "events": events,
        "layers": layers,
        "summary": {
            "event_count": len(events),
            "layer_count": spec.num_layers,
            "tokens_per_layer": tokens_per_layer,
            "routed_token_decisions": tokens_per_layer * spec.num_layers,
            "expert_assignments": tokens_per_layer * spec.num_layers * spec.top_k,
            "batch_shapes": [
                {
                    "phase": "prefill",
                    "batch_size": events[0]["batch_size"],
                    "tokens_per_sequence": events[0]["tokens_per_sequence"],
                    "occurrences": 1,
                },
                {
                    "phase": "decode",
                    "batch_size": events[1]["batch_size"],
                    "tokens_per_sequence": events[1]["tokens_per_sequence"],
                    "occurrences": len(events) - 1,
                },
            ],
        },
        "route_payload_sha256": _payload_hash(
            layers, schema_version=spec.trace_schema_version
        ),
    }
    validate_document(document)
    return document


def validate_route_trace_document(document: Mapping[str, Any]) -> None:
    """Recompute every derived statistic and the binary route payload hash."""

    events = _normalize_events(document["events"])
    model = document["model"]
    spec = fixed_model_spec(model["model_id"], model["model_revision"])
    if document["schema_version"] != spec.trace_schema_version:
        raise ContractError("RouteTrace schema version does not match fixed model")
    if model != {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "num_layers": spec.num_layers,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
    }:
        raise ContractError("RouteTrace model architecture is inconsistent")
    raw_layers = [
        {
            "layer_index": layer["layer_index"],
            "events": layer["events"],
        }
        for layer in document["layers"]
    ]
    layers = _normalize_layers(raw_layers, events, spec)
    for observed, expected in zip(document["layers"], layers, strict=True):
        if observed != expected:
            raise ContractError(
                f"RouteTrace layer {observed['layer_index']} derived fields are inconsistent"
            )
    if document["route_payload_sha256"] != _payload_hash(
        layers, schema_version=spec.trace_schema_version
    ):
        raise ContractError("RouteTrace route payload SHA-256 is inconsistent")
    tokens_per_layer = sum(event["token_count"] for event in events)
    expected_summary = {
        "event_count": len(events),
        "layer_count": spec.num_layers,
        "tokens_per_layer": tokens_per_layer,
        "routed_token_decisions": tokens_per_layer * spec.num_layers,
        "expert_assignments": tokens_per_layer * spec.num_layers * spec.top_k,
        "batch_shapes": [
            {
                "phase": "prefill",
                "batch_size": events[0]["batch_size"],
                "tokens_per_sequence": events[0]["tokens_per_sequence"],
                "occurrences": 1,
            },
            {
                "phase": "decode",
                "batch_size": events[1]["batch_size"],
                "tokens_per_sequence": events[1]["tokens_per_sequence"],
                "occurrences": len(events) - 1,
            },
        ],
    }
    if document["summary"] != expected_summary:
        raise ContractError("RouteTrace summary is inconsistent")


def iter_replay_events(document: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield immutable event-major/layer-major route tensors for kernel replay."""

    validate_document(document)
    model = document["model"]
    spec = fixed_model_spec(model["model_id"], model["model_revision"])
    layer_by_index = {layer["layer_index"]: layer for layer in document["layers"]}
    for event in document["events"]:
        event_index = event["event_index"]
        for layer_index in range(spec.num_layers):
            layer_event = layer_by_index[layer_index]["events"][event_index]
            yield {
                "event_index": event_index,
                "phase": event["phase"],
                "decode_step": event["decode_step"],
                "layer_index": layer_index,
                "batch_size": event["batch_size"],
                "tokens_per_sequence": event["tokens_per_sequence"],
                "token_count": event["token_count"],
                "top_k": spec.top_k,
                "expert_indices": tuple(layer_event["expert_indices"]),
                "routing_weights": tuple(layer_event["routing_weights"]),
            }


def build_replay_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Build compact evidence that a trace was fully traversed in replay order."""

    records = list(iter_replay_events(document))
    return {
        "schema_version": 1,
        "kind": "route_trace_replay",
        "generated_at": _utc_now(),
        "trace_semantic_sha256": canonical_sha256(document),
        "route_payload_sha256": document["route_payload_sha256"],
        "event_layer_records": len(records),
        "event_count": document["summary"]["event_count"],
        "layer_count": document["model"]["num_layers"],
        "tokens_per_layer": document["summary"]["tokens_per_layer"],
        "expert_assignments": document["summary"]["expert_assignments"],
    }


__all__ = [
    "build_replay_summary",
    "build_route_trace",
    "iter_replay_events",
    "validate_route_trace_document",
]
