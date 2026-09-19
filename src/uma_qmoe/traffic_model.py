"""Auditable modeled/estimated traffic accounting for NVIDIA DGX Spark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any

from .contracts import canonical_sha256


class TrafficModelError(ValueError):
    """Raised when traffic inputs cannot support a fail-closed model."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _claim(ledger: Mapping[str, Any], claim_id: str) -> Mapping[str, Any]:
    matches = [
        claim
        for source in ledger["sources"]
        for claim in source["claims"]
        if claim["id"] == claim_id
    ]
    if len(matches) != 1:
        raise TrafficModelError(f"SourceLedger must contain one {claim_id!r} claim")
    return matches[0]


def validate_traffic_source_ledger(document: Mapping[str, Any]) -> None:
    source_ids = [source["id"] for source in document["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise TrafficModelError("TrafficSourceLedger contains duplicate source ids")
    claim_ids = [
        claim["id"]
        for source in document["sources"]
        for claim in source["claims"]
    ]
    if len(claim_ids) != len(set(claim_ids)):
        raise TrafficModelError("TrafficSourceLedger contains duplicate claim ids")
    for source in document["sources"]:
        if source["source_type"] == "official_specification":
            if source["url"] is None or source["artifact"] is not None:
                raise TrafficModelError(
                    "official TrafficSourceLedger sources require URL and no artifact"
                )
        else:
            if source["url"] is not None or source["artifact"] is None:
                raise TrafficModelError(
                    "local/derived TrafficSourceLedger sources require artifact and no URL"
                )
    required = {
        "dgx_spark_memory_bandwidth_spec_gbps": (273.0, "GB/s", "specified"),
        "dgx_spark_unified_memory_capacity_gb": (128.0, "GB", "specified"),
        "spark1_read_reduce_p50_algorithmic_gbps": (
            243.56989275164187,
            "GB/s",
            "measured_algorithmic",
        ),
    }
    for claim_id, expected in required.items():
        claim = _claim(document, claim_id)
        if (
            not math.isclose(claim["value"], expected[0], rel_tol=1e-12)
            or claim["unit"] != expected[1]
            or claim["evidence_class"] != expected[2]
        ):
            raise TrafficModelError(f"TrafficSourceLedger claim {claim_id!r} changed")
    decision = document["decision"]
    if decision["privileged_counters_used"] or decision["measured_dram_bytes_available"]:
        raise TrafficModelError("Spark TrafficSourceLedger must remain counter-free")


def _identity(
    *, path: str, file_sha256: str, document: Mapping[str, Any] | None = None
) -> dict[str, str]:
    value = {"path": path, "file_sha256": file_sha256}
    if document is not None:
        value["semantic_sha256"] = canonical_sha256(document)
    return value


def build_spark_traffic_model(
    source_ledger: Mapping[str, Any],
    weight_traffic: Mapping[str, Any],
    route_trace: Mapping[str, Any],
    bandwidth_soak: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    source_ledger_path: str,
    source_ledger_file_sha256: str,
    weight_traffic_path: str,
    weight_traffic_file_sha256: str,
    route_trace_path: str,
    route_trace_file_sha256: str,
    bandwidth_soak_path: str,
    bandwidth_soak_file_sha256: str,
    model_config_path: str,
    model_config_file_sha256: str,
    amplification_factors: Sequence[float],
) -> dict[str, Any]:
    validate_traffic_source_ledger(source_ledger)
    if source_ledger.get("status") != "frozen":
        raise TrafficModelError("Spark TrafficSourceLedger must be frozen")
    if weight_traffic.get("kind") != "weight_traffic_estimate":
        raise TrafficModelError("weight input must be WeightTrafficEstimate")
    if weight_traffic.get("scenario") != "decode_batch1_cold_weight_read":
        raise TrafficModelError("weight input has the wrong scenario")
    if weight_traffic["quantization"] != {
        "dense_bits_per_element": 16,
        "expert_bits_per_element": 4,
        "group_size": 128,
        "scale_bytes_per_group": 2,
        "zero_point_bytes_per_group": 0,
        "tensor_alignment_bytes": 128,
    }:
        raise TrafficModelError("weight input is not canonical Q4 group-128")
    if route_trace.get("kind") != "route_trace" or route_trace.get("status") != "frozen":
        raise TrafficModelError("route input must be a frozen RouteTrace")
    if bandwidth_soak.get("kind") != "bandwidth_soak" or bandwidth_soak.get("target_id") != "spark1":
        raise TrafficModelError("bandwidth input must be the Spark soak")
    if bandwidth_soak.get("status") != "passed":
        raise TrafficModelError("Spark bandwidth soak did not pass")
    if bandwidth_soak["measurement_scope"]["counter_calibrated"]:
        raise TrafficModelError("Spark traffic model must not use privileged counters")
    factors = [float(value) for value in amplification_factors]
    if len(factors) < 2 or factors != sorted(set(factors)) or factors[0] != 1.0:
        raise TrafficModelError(
            "amplification factors must be unique ascending values starting at 1.0"
        )
    if any(not math.isfinite(value) or value < 1.0 for value in factors):
        raise TrafficModelError("amplification factors must be finite and at least 1")

    events = route_trace["events"]
    prefill = [event for event in events if event["phase"] == "prefill"]
    decode = [event for event in events if event["phase"] == "decode"]
    if len(prefill) != 1 or not decode or any(event["batch_size"] != 1 for event in events):
        raise TrafficModelError("RouteTrace does not describe fixed batch-1 generation")
    prompt_tokens = prefill[0]["tokens_per_sequence"]
    output_tokens = len(decode) + 1
    minimum_past = prompt_tokens
    maximum_past = prompt_tokens + len(decode) - 1
    mean_past = (minimum_past + maximum_past) / 2

    required_config = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_hidden_layers": 16,
        "torch_dtype": "bfloat16",
    }
    if any(model_config.get(name) != value for name, value in required_config.items()):
        raise TrafficModelError("OLMoE model config changed from the fixed traffic ABI")
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    kv_bytes_per_sequence_token = (
        2
        * model_config["num_hidden_layers"]
        * model_config["num_key_value_heads"]
        * head_dim
        * 2
    )
    kv_read = int(mean_past * kv_bytes_per_sequence_token)
    kv_write = kv_bytes_per_sequence_token
    per_token = weight_traffic["per_token"]
    components = {
        "dense_weight_bytes_per_token": per_token["dense_weight_bytes"],
        "expert_payload_bytes_per_token": per_token["active_expert_payload_bytes"],
        "expert_metadata_bytes_per_token": per_token["active_expert_metadata_bytes"],
        "expert_alignment_bytes_per_token": per_token["active_expert_padding_bytes"],
        "kv_read_bytes_per_token": kv_read,
        "kv_write_bytes_per_token": kv_write,
        "activation_workspace_bytes_per_token": 0,
        "activation_workspace_treatment": "unmeasured_excluded_from_minimum_and_covered_by_amplification_sensitivity",
    }
    minimum = sum(
        value
        for name, value in components.items()
        if name.endswith("_bytes_per_token") and isinstance(value, int)
    )
    components["algorithmic_minimum_bytes_per_token"] = minimum
    local_gbps = next(
        operation["p50_gbps"]
        for operation in bandwidth_soak["operations"]
        if operation["id"] == "read_reduce"
    )
    specified_gbps = float(
        _claim(source_ledger, "dgx_spark_memory_bandwidth_spec_gbps")["value"]
    )
    rows = [
        {
            "amplification_factor": factor,
            "modeled_bytes_per_token": minimum * factor,
            "local_bandwidth_ceiling_tokens_per_second": local_gbps
            * 1_000_000_000
            / (minimum * factor),
            "official_spec_ceiling_tokens_per_second": specified_gbps
            * 1_000_000_000
            / (minimum * factor),
        }
        for factor in factors
    ]
    gates = {
        "source_ledger_frozen": source_ledger["status"] == "frozen",
        "inputs_hash_bound": True,
        "route_workload_bound": prompt_tokens == 128 and output_tokens == 32,
        "component_formula_consistent": True,
        "sensitivity_formula_consistent": True,
        "privileged_counter_not_used": True,
        "no_measured_dram_claim": True,
    }
    gates["overall_passed"] = all(gates.values())
    document = {
        "schema_version": 1,
        "kind": "spark_traffic_model",
        "generated_at": _now(),
        "target_id": "spark1",
        "status": "passed" if gates["overall_passed"] else "failed",
        "result_class": "modeled_estimated",
        "inputs": {
            "source_ledger": _identity(
                path=source_ledger_path,
                file_sha256=source_ledger_file_sha256,
                document=source_ledger,
            ),
            "weight_traffic_estimate": _identity(
                path=weight_traffic_path,
                file_sha256=weight_traffic_file_sha256,
                document=weight_traffic,
            ),
            "route_trace": _identity(
                path=route_trace_path,
                file_sha256=route_trace_file_sha256,
                document=route_trace,
            ),
            "bandwidth_soak": _identity(
                path=bandwidth_soak_path,
                file_sha256=bandwidth_soak_file_sha256,
                document=bandwidth_soak,
            ),
            "model_config": _identity(
                path=model_config_path, file_sha256=model_config_file_sha256
            ),
        },
        "workload": {
            "batch_size": 1,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "decode_steps": len(decode),
            "minimum_past_tokens": minimum_past,
            "maximum_past_tokens": maximum_past,
            "mean_past_tokens": mean_past,
        },
        "components": components,
        "bandwidth": {
            "official_spec_gbps": specified_gbps,
            "local_read_p50_algorithmic_gbps": local_gbps,
            "local_measurement_is_dram_counter": False,
        },
        "sensitivity": rows,
        "assumptions": [
            "weight bytes are the canonical batch-1 cold-read upper workload bound from WeightTrafficEstimate",
            "KV reads cover the mean past length across 31 decode steps and BF16 K/V writes cover one new token",
            "activation and workspace traffic are excluded from the algorithmic minimum",
            "amplification factors represent unknown cache, replay, page migration, system contention, activation, and workspace effects",
            "bandwidth-derived token rates are ceilings, not measured or predicted end-to-end throughput",
            "273 GB/s is an official physical specification and 243.57 GB/s is an unprivileged algorithmic stream measurement",
        ],
        "gates": gates,
    }
    validate_spark_traffic_model(document)
    return document


def validate_spark_traffic_model(document: Mapping[str, Any]) -> None:
    workload = document["workload"]
    if workload["decode_steps"] != workload["output_tokens"] - 1:
        raise TrafficModelError("SparkTrafficModel decode step count is inconsistent")
    expected_mean = (
        workload["minimum_past_tokens"] + workload["maximum_past_tokens"]
    ) / 2
    if not math.isclose(workload["mean_past_tokens"], expected_mean, rel_tol=1e-12):
        raise TrafficModelError("SparkTrafficModel mean past length is inconsistent")
    components = document["components"]
    expected_minimum = sum(
        value
        for name, value in components.items()
        if name.endswith("_bytes_per_token")
        and name != "algorithmic_minimum_bytes_per_token"
        and isinstance(value, int)
    )
    component_ok = components["algorithmic_minimum_bytes_per_token"] == expected_minimum
    local = document["bandwidth"]["local_read_p50_algorithmic_gbps"]
    specified = document["bandwidth"]["official_spec_gbps"]
    factors = [row["amplification_factor"] for row in document["sensitivity"]]
    sensitivity_ok = factors == sorted(set(factors)) and factors[0] == 1.0
    for row in document["sensitivity"]:
        expected_bytes = expected_minimum * row["amplification_factor"]
        sensitivity_ok &= math.isclose(
            row["modeled_bytes_per_token"], expected_bytes, rel_tol=1e-12
        )
        sensitivity_ok &= math.isclose(
            row["local_bandwidth_ceiling_tokens_per_second"],
            local * 1_000_000_000 / expected_bytes,
            rel_tol=1e-12,
        )
        sensitivity_ok &= math.isclose(
            row["official_spec_ceiling_tokens_per_second"],
            specified * 1_000_000_000 / expected_bytes,
            rel_tol=1e-12,
        )
    expected_gates = {
        "source_ledger_frozen": True,
        "inputs_hash_bound": all(
            len(identity["file_sha256"]) == 64
            for identity in document["inputs"].values()
        ),
        "route_workload_bound": workload["prompt_tokens"] == 128
        and workload["output_tokens"] == 32,
        "component_formula_consistent": component_ok,
        "sensitivity_formula_consistent": bool(sensitivity_ok),
        "privileged_counter_not_used": True,
        "no_measured_dram_claim": document["result_class"] == "modeled_estimated"
        and not document["bandwidth"]["local_measurement_is_dram_counter"],
    }
    gates = document["gates"]
    if any(gates[name] != value for name, value in expected_gates.items()):
        raise TrafficModelError("SparkTrafficModel gate does not match its evidence")
    overall = all(expected_gates.values())
    if gates["overall_passed"] != overall:
        raise TrafficModelError("SparkTrafficModel overall gate is inconsistent")
    if document["status"] != ("passed" if overall else "failed"):
        raise TrafficModelError("SparkTrafficModel status is inconsistent")


__all__ = [
    "TrafficModelError",
    "build_spark_traffic_model",
    "validate_spark_traffic_model",
    "validate_traffic_source_ledger",
]
