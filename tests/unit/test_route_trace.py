from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.route_trace import (
    build_replay_summary,
    build_route_trace,
    iter_replay_events,
)


def _raw_capture() -> dict:
    events = [
        {
            "event_index": 0,
            "phase": "prefill",
            "decode_step": None,
            "batch_size": 1,
            "tokens_per_sequence": 2,
        },
        {
            "event_index": 1,
            "phase": "decode",
            "decode_step": 0,
            "batch_size": 1,
            "tokens_per_sequence": 1,
        },
        {
            "event_index": 2,
            "phase": "decode",
            "decode_step": 1,
            "batch_size": 1,
            "tokens_per_sequence": 1,
        },
    ]
    layers = []
    for layer_index in range(16):
        layer_events = []
        for event in events:
            token_count = event["batch_size"] * event["tokens_per_sequence"]
            experts = [
                (expert + layer_index) % 64
                for _ in range(token_count)
                for expert in range(8)
            ]
            layer_events.append(
                {
                    "event_index": event["event_index"],
                    "expert_indices": experts,
                    "routing_weights": [0.125] * len(experts),
                }
            )
        layers.append({"layer_index": layer_index, "events": layer_events})
    return {
        "schema_version": 1,
        "captured_at": "2026-09-18T07:00:00Z",
        "model_id": "allenai/OLMoE-1B-7B-0125",
        "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
        "architecture": {"num_layers": 16, "num_experts": 64, "top_k": 8},
        "fixture": {
            "id": "general-001",
            "sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
        },
        "workload": {
            "batch_size": 1,
            "prompt_tokens": 2,
            "output_tokens": 3,
            "decoding": "greedy",
            "ignore_eos": True,
        },
        "capture": {
            "target_id": "halo3",
            "backend": "hip",
            "torch_version": "2.14.0",
            "transformers_version": "5.10.2",
        },
        "events": events,
        "layers": layers,
    }


def _raw_qwen_capture() -> dict:
    capture = _raw_capture()
    capture["model_id"] = "Qwen/Qwen1.5-MoE-A2.7B"
    capture["model_revision"] = "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    capture["architecture"] = {"num_layers": 24, "num_experts": 60, "top_k": 4}
    events = capture["events"]
    layers = []
    for layer_index in range(24):
        layer_events = []
        for event in events:
            token_count = event["batch_size"] * event["tokens_per_sequence"]
            experts = [
                (expert + layer_index) % 60
                for _ in range(token_count)
                for expert in range(4)
            ]
            layer_events.append(
                {
                    "event_index": event["event_index"],
                    "expert_indices": experts,
                    "routing_weights": [0.25] * len(experts),
                }
            )
        layers.append({"layer_index": layer_index, "events": layer_events})
    capture["layers"] = layers
    return capture


def test_route_trace_recomputes_statistics_hash_and_replay(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(_raw_capture()), encoding="utf-8")

    trace = build_route_trace(capture, trace_id="olmoe_route_trace_test_v1")

    assert trace["summary"]["tokens_per_layer"] == 4
    assert trace["summary"]["expert_assignments"] == 4 * 16 * 8
    assert trace["layers"][0]["expert_statistics"]["tokens_per_expert"][:8] == [4] * 8
    assert len(trace["layers"][0]["expert_statistics"]["empty_experts"]) == 56
    assert len(list(iter_replay_events(trace))) == 3 * 16
    replay = build_replay_summary(trace)
    assert replay["event_layer_records"] == 48
    validate_document(trace)
    validate_document(replay)


def test_route_trace_rejects_statistical_or_payload_tampering(tmp_path: Path) -> None:
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(_raw_capture()), encoding="utf-8")
    trace = build_route_trace(capture, trace_id="olmoe_route_trace_test_v1")

    tampered = copy.deepcopy(trace)
    tampered["layers"][0]["expert_statistics"]["maximum_tokens"] += 1
    with pytest.raises(ContractError, match="derived fields"):
        validate_document(tampered)

    tampered = copy.deepcopy(trace)
    tampered["route_payload_sha256"] = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(ContractError, match="payload SHA"):
        validate_document(tampered)


def test_route_trace_rejects_duplicate_top_k_expert(tmp_path: Path) -> None:
    raw = _raw_capture()
    raw["layers"][0]["events"][0]["expert_indices"][1] = 0
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="duplicate experts"):
        build_route_trace(capture, trace_id="olmoe_route_trace_test_v1")


def test_qwen_route_trace_v2_recomputes_dynamic_shapes(tmp_path: Path) -> None:
    capture = tmp_path / "qwen-capture.json"
    capture.write_text(json.dumps(_raw_qwen_capture()), encoding="utf-8")

    trace = build_route_trace(capture, trace_id="qwen1_5_moe_route_trace_test_v2")

    assert trace["schema_version"] == 2
    assert trace["summary"]["layer_count"] == 24
    assert trace["summary"]["expert_assignments"] == 4 * 24 * 4
    assert len(trace["layers"]) == 24
    assert len(trace["layers"][0]["expert_statistics"]["tokens_per_expert"]) == 60
    assert len(list(iter_replay_events(trace))) == 3 * 24
    replay = build_replay_summary(trace)
    assert replay["event_layer_records"] == 3 * 24
    assert replay["layer_count"] == 24
    validate_document(trace)
    validate_document(replay)


def test_qwen_route_trace_rejects_olmoe_shape_aliasing(tmp_path: Path) -> None:
    raw = _raw_qwen_capture()
    raw["architecture"]["top_k"] = 8
    capture = tmp_path / "qwen-capture.json"
    capture.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ContractError, match="architecture"):
        build_route_trace(capture, trace_id="qwen1_5_moe_route_trace_test_v2")
