from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.traffic_model import build_spark_traffic_model


def _ledger() -> dict:
    def artifact(path: str) -> dict:
        return {
            "path": path,
            "file_sha256": "a" * 64,
            "semantic_sha256": "b" * 64,
        }

    return {
        "schema_version": 1,
        "kind": "traffic_source_ledger",
        "ledger_id": "spark-test",
        "captured_at": "2026-09-19T00:00:00Z",
        "target_id": "spark1",
        "status": "frozen",
        "sources": [
            {
                "id": "official",
                "source_type": "official_specification",
                "publisher": "NVIDIA",
                "title": "hardware",
                "url": "https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
                "accessed_on": "2026-09-19",
                "artifact": None,
                "claims": [
                    {
                        "id": "dgx_spark_memory_bandwidth_spec_gbps",
                        "metric": "bandwidth",
                        "value": 273.0,
                        "unit": "GB/s",
                        "evidence_class": "specified",
                    },
                    {
                        "id": "dgx_spark_unified_memory_capacity_gb",
                        "metric": "capacity",
                        "value": 128.0,
                        "unit": "GB",
                        "evidence_class": "specified",
                    },
                ],
                "limitations": ["not a workload measurement"],
            },
            {
                "id": "soak",
                "source_type": "local_measurement",
                "publisher": "UMA-QMoE",
                "title": "soak",
                "url": None,
                "accessed_on": "2026-09-19",
                "artifact": artifact("lab-private/soak.json"),
                "claims": [
                    {
                        "id": "spark1_read_reduce_p50_algorithmic_gbps",
                        "metric": "algorithmic read",
                        "value": 243.56989275164187,
                        "unit": "GB/s",
                        "evidence_class": "measured_algorithmic",
                    }
                ],
                "limitations": ["not DRAM counters"],
            },
            {
                "id": "weights",
                "source_type": "derived_contract",
                "publisher": "UMA-QMoE",
                "title": "weights",
                "url": None,
                "accessed_on": "2026-09-19",
                "artifact": artifact("models/weight.json"),
                "claims": [
                    {
                        "id": "weight-bytes",
                        "metric": "weights",
                        "value": 1162616832,
                        "unit": "bytes/token",
                        "evidence_class": "derived",
                    }
                ],
                "limitations": ["cold-read bound"],
            },
            {
                "id": "trace",
                "source_type": "derived_contract",
                "publisher": "UMA-QMoE",
                "title": "trace",
                "url": None,
                "accessed_on": "2026-09-19",
                "artifact": artifact("benchmarks/trace.json"),
                "claims": [
                    {
                        "id": "assignments",
                        "metric": "assignments",
                        "value": 20352,
                        "unit": "assignments",
                        "evidence_class": "derived",
                    }
                ],
                "limitations": ["logical routing only"],
            },
        ],
        "decision": {
            "privileged_counters_used": False,
            "measured_dram_bytes_available": False,
            "allowed_result_labels": ["modeled", "estimated"],
            "prohibited_claims": ["measured DRAM bytes", "measured amplification"],
        },
    }


def _inputs() -> tuple[dict, dict, dict, dict]:
    weight = {
        "kind": "weight_traffic_estimate",
        "scenario": "decode_batch1_cold_weight_read",
        "quantization": {
            "dense_bits_per_element": 16,
            "expert_bits_per_element": 4,
            "group_size": 128,
            "scale_bytes_per_group": 2,
            "zero_point_bytes_per_group": 0,
            "tensor_alignment_bytes": 128,
        },
        "per_token": {
            "dense_weight_bytes": 747380736,
            "active_expert_payload_bytes": 402653184,
            "active_expert_metadata_bytes": 12582912,
            "active_expert_padding_bytes": 0,
        },
    }
    trace = {
        "kind": "route_trace",
        "status": "frozen",
        "events": [
            {
                "phase": "prefill",
                "batch_size": 1,
                "tokens_per_sequence": 128,
            },
            *[
                {
                    "phase": "decode",
                    "batch_size": 1,
                    "tokens_per_sequence": 1,
                }
                for _ in range(31)
            ],
        ],
    }
    soak = {
        "kind": "bandwidth_soak",
        "target_id": "spark1",
        "status": "passed",
        "measurement_scope": {"counter_calibrated": False},
        "operations": [
            {"id": "read_reduce", "p50_gbps": 243.56989275164187}
        ],
    }
    config = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_hidden_layers": 16,
        "torch_dtype": "bfloat16",
    }
    return weight, trace, soak, config


def _build() -> dict:
    weight, trace, soak, config = _inputs()
    return build_spark_traffic_model(
        _ledger(),
        weight,
        trace,
        soak,
        config,
        source_ledger_path="benchmarks/sources/ledger.json",
        source_ledger_file_sha256="1" * 64,
        weight_traffic_path="models/weight.json",
        weight_traffic_file_sha256="2" * 64,
        route_trace_path="benchmarks/trace.json",
        route_trace_file_sha256="3" * 64,
        bandwidth_soak_path="lab-private/soak.json",
        bandwidth_soak_file_sha256="4" * 64,
        model_config_path="models/config.json",
        model_config_file_sha256="5" * 64,
        amplification_factors=[1.0, 2.0, 4.0],
    )


def test_source_ledger_is_counter_free_and_frozen() -> None:
    validate_document(_ledger())
    changed = _ledger()
    changed["decision"]["privileged_counters_used"] = True
    with pytest.raises(ContractError):
        validate_document(changed)


def test_builds_hash_bound_spark_traffic_sensitivity() -> None:
    document = _build()
    assert document["result_class"] == "modeled_estimated"
    assert document["components"]["algorithmic_minimum_bytes_per_token"] == 1181491200
    assert document["workload"]["mean_past_tokens"] == 143.0
    assert [row["amplification_factor"] for row in document["sensitivity"]] == [
        1.0,
        2.0,
        4.0,
    ]
    validate_document(document)


def test_rejects_tampered_modeled_byte_formula() -> None:
    document = _build()
    tampered = copy.deepcopy(document)
    tampered["sensitivity"][0]["modeled_bytes_per_token"] += 1
    with pytest.raises(ContractError, match="gate does not match|formula"):
        validate_document(tampered)
