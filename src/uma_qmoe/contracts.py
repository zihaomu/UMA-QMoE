"""Load, validate, and hash UMA-QMoE machine-readable contracts."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_BY_KIND = {
    "allocation_matrix": "allocation_matrix.schema.json",
    "allocation_matrix_v2": "allocation_matrix_v2.schema.json",
    "activation_aware_mixed_policy": "activation_aware_mixed_policy.schema.json",
    "artifact_verification": "artifact_verification.schema.json",
    "bandwidth_soak": "bandwidth_soak.schema.json",
    "benchmark_contract": "benchmark_contract.schema.json",
    "compressed_host_baseline": "compressed_host_baseline.schema.json",
    "compressed_loader_evidence": "compressed_loader_evidence.schema.json",
    "custom_operator_evidence": "custom_operator_evidence.schema.json",
    "external_baseline": "external_baseline.schema.json",
    "expert_pack_manifest": "expert_pack_manifest.schema.json",
    "expert_balanced_sample_manifest": "expert_balanced_sample_manifest.schema.json",
    "expert_calibration_coverage": "expert_calibration_coverage.schema.json",
    "hardware_counter_calibration": "hardware_counter_calibration.schema.json",
    "layer_precision_search": "layer_precision_search.schema.json",
    "machine_baseline": "machine_baseline.schema.json",
    "memory_bandwidth_benchmark": "memory_bandwidth_benchmark.schema.json",
    "native_allocation_capabilities": "native_allocation_capabilities.schema.json",
    "model_acquisition": "model_acquisition.schema.json",
    "model_derivation": "model_derivation.schema.json",
    "memory_snapshot": "memory_snapshot.schema.json",
    "mixed_precision_refinement": "mixed_precision_refinement.schema.json",
    "mixed_precision_policy_search": "mixed_precision_policy_search.schema.json",
    "mixed_precision_sensitivity": "mixed_precision_sensitivity.schema.json",
    "native_stream_benchmark": "native_stream_benchmark.schema.json",
    "model_manifest": "model_manifest.schema.json",
    "oracle_smoke": "oracle_smoke.schema.json",
    "packed_q4_kernel_evidence": "packed_q4_kernel_evidence.schema.json",
    "public_baseline": "public_baseline.schema.json",
    "quantization_compensation_search": "quantization_compensation_search.schema.json",
    "quantization_dataset_manifest": "quantization_dataset_manifest.schema.json",
    "router_logit_compensation_search": "router_logit_compensation_search.schema.json",
    "reference_host_baseline": "reference_host_baseline.schema.json",
    "reference_oracle_comparison": "reference_oracle_comparison.schema.json",
    "reference_oracle_policy": "reference_oracle_policy.schema.json",
    "reverse_layer_quantization_search": "reverse_layer_quantization_search.schema.json",
    "run_manifest": "run_manifest.schema.json",
    "route_trace": "route_trace.schema.json",
    "route_trace_replay": "route_trace_replay.schema.json",
    "route_coverage_policy_search": "route_coverage_policy_search.schema.json",
    "safe_uma_budget": "safe_uma_budget.schema.json",
    "target_inventory": "target_inventory.schema.json",
    "target_pack_manifest": "target_pack_manifest.schema.json",
    "target_pack_host_quality": "target_pack_host_quality.schema.json",
    "tensor_inventory": "tensor_inventory.schema.json",
    "traffic_source_ledger": "traffic_source_ledger.schema.json",
    "spark_traffic_model": "spark_traffic_model.schema.json",
    "weight_traffic_estimate": "weight_traffic_estimate.schema.json",
}

# Operational timestamps do not change the identity of an otherwise identical
# contract. Status deliberately remains part of the identity.
NON_IDENTITY_KEYS = frozenset(
    {"captured_at", "created_at", "generated_at", "resolved_at"}
)


class ContractError(ValueError):
    """Raised when a contract is malformed or violates a semantic invariant."""


def load_document(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping without applying defaults."""

    document_path = Path(path)
    try:
        text = document_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read {document_path}: {exc}") from exc

    try:
        if document_path.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            value = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot parse {document_path}: {exc}") from exc

    if not isinstance(value, dict):
        raise ContractError(f"{document_path} must contain a mapping at its root")
    return value


def _schema_for_kind(kind: str) -> dict[str, Any]:
    try:
        schema_name = SCHEMA_BY_KIND[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(SCHEMA_BY_KIND))
        raise ContractError(
            f"unsupported contract kind {kind!r}; expected one of: {supported}"
        ) from exc

    schema_resource = resources.files("uma_qmoe.schemas").joinpath(schema_name)
    return json.loads(schema_resource.read_text(encoding="utf-8"))


def _format_path(parts: list[Any]) -> str:
    if not parts:
        return "$"
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _require_project_relative_path(path: str, field: str) -> None:
    """Reject absolute, platform-dependent, or traversal-bearing references."""

    segments = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise ContractError(f"{field} must be a safe project-relative POSIX path")


def _validate_refinement_delta(
    row: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    metrics = row["metrics"]
    expected = {
        "router_exact_set_agreement": metrics["router_exact_set_agreement"]
        - baseline["router_exact_set_agreement"],
        "router_mean_set_overlap": metrics["router_mean_set_overlap"]
        - baseline["router_mean_set_overlap"],
        "logit_cosine_similarity": metrics["logit_cosine_similarity"]
        - baseline["logit_cosine_similarity"],
    }
    if any(
        not math.isclose(
            row["delta_vs_all_q4"][name], value, rel_tol=1e-9, abs_tol=1e-12
        )
        for name, value in expected.items()
    ):
        raise ContractError("Mixed-precision refinement delta is inconsistent")


def _validate_quality_metrics(
    metrics: Mapping[str, Any], reference: Mapping[str, Any], label: str
) -> None:
    expected_ppl = math.exp(metrics["nll"])
    if not math.isclose(
        metrics["perplexity"], expected_ppl, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ContractError(f"{label} perplexity is inconsistent")
    expected_nll_change = (metrics["nll"] / reference["nll"]) - 1.0
    expected_ppl_change = (metrics["perplexity"] / reference["perplexity"]) - 1.0
    if not math.isclose(
        metrics["relative_nll_change"],
        expected_nll_change,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) or not math.isclose(
        metrics["relative_perplexity_change"],
        expected_ppl_change,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ContractError(f"{label} relative quality is inconsistent")


def _experts_for_route_coverage(
    counts: list[int], threshold: float
) -> tuple[list[int], float]:
    total = sum(counts)
    if total <= 0:
        raise ContractError("Route coverage calibration layer has no assignments")
    ranking = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
    selected: list[int] = []
    covered = 0
    for expert in ranking:
        if counts[expert] == 0:
            break
        selected.append(expert)
        covered += counts[expert]
        if covered / total >= threshold:
            break
    return selected, covered / total


def validate_document(
    document: Mapping[str, Any], *, require_frozen: bool = False
) -> None:
    """Validate schema and invariants that do not require external files."""

    kind = document.get("kind")
    if not isinstance(kind, str):
        raise ContractError("$.kind must be a string")

    validator = Draft202012Validator(
        _schema_for_kind(kind), format_checker=FormatChecker()
    )
    errors = sorted(
        validator.iter_errors(document), key=lambda error: list(error.absolute_path)
    )
    if errors:
        details = "; ".join(
            f"{_format_path(list(error.absolute_path))}: {error.message}"
            for error in errors
        )
        raise ContractError(details)

    if kind == "model_manifest":
        architecture = document["architecture"]
        if architecture["top_k"] > architecture["num_experts"]:
            raise ContractError("$.architecture.top_k cannot exceed num_experts")
        identities = [document["config"]]
        identities.extend(document["tokenizer"].get("files", []))
        identities.extend(document["weights"]["artifacts"])
        seen_paths: set[str] = set()
        for identity in identities:
            path = identity["path"]
            _require_project_relative_path(path, f"file identity path {path!r}")
            if path in seen_paths:
                raise ContractError(f"duplicate file identity path {path!r}")
            seen_paths.add(path)
        if not any(
            artifact["path"].endswith(".safetensors")
            for artifact in document["weights"]["artifacts"]
        ):
            raise ContractError(
                "$.weights.artifacts must contain at least one Safetensors shard"
            )
        tensor_inventory = document["weights"].get("tensor_inventory")
        if tensor_inventory is not None:
            _require_project_relative_path(
                tensor_inventory["path"], "$.weights.tensor_inventory.path"
            )
        if require_frozen:
            _require_model_frozen(document)
    elif kind == "benchmark_contract":
        target_ids = [target["id"] for target in document["targets"]]
        if len(target_ids) != len(set(target_ids)):
            raise ContractError("$.targets contains duplicate target ids")
        for target in document["targets"]:
            baseline_ids = [baseline["id"] for baseline in target["external_baselines"]]
            if len(baseline_ids) != len(set(baseline_ids)):
                raise ContractError(
                    f"target {target['id']!r} contains duplicate external baseline ids"
                )
        workload_ids = [workload["id"] for workload in document["workloads"]]
        if len(workload_ids) != len(set(workload_ids)):
            raise ContractError("$.workloads contains duplicate workload ids")
        gate_ids = [gate["id"] for gate in document["quality_gates"]]
        if len(gate_ids) != len(set(gate_ids)):
            raise ContractError("$.quality_gates contains duplicate gate ids")
        _require_project_relative_path(
            document["model"]["manifest_path"], "$.model.manifest_path"
        )
        _require_project_relative_path(
            document["oracle"]["prompt_fixture"]["path"],
            "$.oracle.prompt_fixture.path",
        )
        oracle_derivation = document["oracle"].get("derivation")
        if oracle_derivation is not None:
            _require_project_relative_path(
                oracle_derivation["path"], "$.oracle.derivation.path"
            )
        oracle_weight_source = document["oracle"].get("weight_source")
        if oracle_weight_source is not None:
            _require_project_relative_path(
                oracle_weight_source["path"], "$.oracle.weight_source.path"
            )
        for gate in document["quality_gates"]:
            dataset = gate.get("dataset")
            if dataset is not None:
                _require_project_relative_path(
                    dataset["sample_ids_path"],
                    f"quality gate {gate['id']!r} dataset.sample_ids_path",
                )
            trace_reference = gate.get("trace_reference")
            if trace_reference is not None:
                _require_project_relative_path(
                    trace_reference["path"],
                    f"quality gate {gate['id']!r} trace_reference.path",
                )
        budget_references = document["resource_gates"]["safe_uma_budget"]["references"]
        budget_target_ids = [reference["target_id"] for reference in budget_references]
        if len(budget_target_ids) != len(set(budget_target_ids)):
            raise ContractError(
                "Safe UMA Budget references contain duplicate target ids"
            )
        for reference in budget_references:
            _require_project_relative_path(
                reference["path"],
                f"Safe UMA Budget reference for {reference['target_id']!r}",
            )
        if require_frozen:
            _require_benchmark_frozen(document)
    elif kind == "compressed_loader_evidence":
        loader = document["loader"]
        gates = document["gates"]
        expected = {
            "dense_only_checkpoint_load": loader["loaded_expert_tensor_count"] == 0,
            "no_expert_parameters": loader["expert_parameter_count"] == 0,
            "single_pack_mapping": loader["expert_pack_mapping_count"] == 1,
            "no_full_dequantized_copy": (
                loader["model_parameter_bytes"] < loader["expert_pack_size_bytes"]
            ),
            "full_model_forward": document["smoke"]["finite"],
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError(
                "Compressed loader gate does not match measured evidence"
            )
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Compressed loader overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Compressed loader status is inconsistent")
    elif kind == "custom_operator_evidence":
        operator = document["operator"]
        reference = document["reference"]
        performance = document["performance_mode"]
        gates = document["gates"]
        expected_platform = {
            "halo3": "hip_gfx1151",
            "local-halo": "hip_gfx1151",
            "spark1": "cuda_sm121",
        }[document["target_id"]]
        expected = {
            "operator_registered": operator["registered"],
            "architecture_dispatch": operator["platform"] == expected_platform,
            "cpu_reference_forward": reference["cpu_finite"],
            "target_reference_forward": reference["target_finite"],
            "cpu_target_agreement": (
                reference["max_abs_error"] <= reference["acceptance"]["max_abs_error"]
                and reference["cosine_similarity"]
                >= reference["acceptance"]["minimum_cosine_similarity"]
            ),
            "performance_mode_fail_closed": (
                performance["requested"]
                and not performance["backend_registered"]
                and performance["rejected"]
                and "silent reference fallback is forbidden" in performance["error"]
            ),
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Custom operator gate does not match measured evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Custom operator overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Custom operator status is inconsistent")
    elif kind == "packed_q4_kernel_evidence":
        kernel = document["kernel"]
        strategy_by_abi = {
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-fused-moe-v2": (
                "two-launch-gate-up-swiglu-down-route"
            ),
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-specialized-v3": (
                "decode-two-launch-prefill-expert-sorted-three-stage"
            ),
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-tiled-v4": (
                "decode-two-launch-prefill-expert-tiled-three-stage"
            ),
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-gfx11-wmma-v5": (
                "decode-two-launch-prefill-gfx11-wmma-three-stage"
            ),
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-pruned-tiled-v5": (
                "decode-two-launch-prefill-expert-tiled-pruned-three-stage"
            ),
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-pruned-vector32-k64-v7": (
                "decode-two-launch-prefill-expert-vector32-k64-pruned-three-stage"
            ),
        }
        expected_strategy = strategy_by_abi.get(kernel["abi"])
        if expected_strategy is not None and (
            kernel.get("execution_strategy") != expected_strategy
        ):
            raise ContractError(
                "Packed Q4 fused evidence must declare its execution strategy"
            )
        if (
            kernel["abi"]
            == "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-gfx11-wmma-v5"
            and kernel["platform"] != "hip_gfx1151"
        ):
            raise ContractError("Packed Q4 gfx11 WMMA evidence requires hip_gfx1151")
        correctness = document["correctness"]
        performance = document["performance"]
        gates = document["gates"]
        expected_platform = {
            "halo3": "hip_gfx1151",
            "local-halo": "hip_gfx1151",
            "spark1": "cuda_sm121",
        }[document["target_id"]]
        acceptance = correctness["acceptance"]
        projection_results = correctness["projection_results"]
        names = [result["name"] for result in projection_results]
        if sorted(names) != ["down_proj", "gate_proj", "up_proj"]:
            raise ContractError("Packed Q4 evidence must cover each projection once")
        projection_correctness = all(
            result["finite"]
            and result["max_absolute_error"] <= acceptance["max_absolute_error"]
            and result["cosine_similarity"] >= acceptance["minimum_cosine_similarity"]
            for result in projection_results
        )
        moe = correctness["moe_forward"]
        moe_correctness = (
            moe["finite"]
            and moe["max_absolute_error"] <= acceptance["max_absolute_error"]
            and moe["cosine_similarity"] >= acceptance["minimum_cosine_similarity"]
        )
        samples = performance["samples_milliseconds"]
        expected_median = statistics.median(samples)
        expected_p95 = _nearest_rank_percentile(samples, 0.95)
        if not math.isclose(
            performance["median_milliseconds"], expected_median, rel_tol=1e-9
        ) or not math.isclose(
            performance["p95_milliseconds"], expected_p95, rel_tol=1e-9
        ):
            raise ContractError("Packed Q4 timing summary does not match samples")
        prefill_expected: dict[str, bool] = {}
        if kernel["abi"] in {
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-specialized-v3",
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-tiled-v4",
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-gfx11-wmma-v5",
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-pruned-tiled-v5",
            "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-pruned-vector32-k64-v7",
        }:
            prefill = correctness.get("prefill_moe_forward")
            prefill_samples = performance.get("prefill_samples_milliseconds")
            if (
                document["workload"].get("prefill_tokens", 0) < 2
                or not isinstance(prefill, dict)
                or not isinstance(prefill_samples, list)
            ):
                raise ContractError(
                    "Packed Q4 v3 evidence must include a prefill workload"
                )
            prefill_correctness = (
                prefill["finite"]
                and prefill["max_absolute_error"]
                <= acceptance["max_absolute_error"]
                and prefill["cosine_similarity"]
                >= acceptance["minimum_cosine_similarity"]
            )
            prefill_median = statistics.median(prefill_samples)
            prefill_p95 = _nearest_rank_percentile(prefill_samples, 0.95)
            if not math.isclose(
                performance.get("prefill_median_milliseconds", math.nan),
                prefill_median,
                rel_tol=1e-9,
            ) or not math.isclose(
                performance.get("prefill_p95_milliseconds", math.nan),
                prefill_p95,
                rel_tol=1e-9,
            ):
                raise ContractError(
                    "Packed Q4 prefill timing summary does not match samples"
                )
            prefill_expected = {
                "prefill_moe_correctness": prefill_correctness,
                "prefill_performance_mode_executed": len(prefill_samples)
                == document["workload"]["measured_iterations"],
            }
        expected = {
            "target_architecture": kernel["platform"] == expected_platform,
            "target_compilation": kernel["compiled_for_target"],
            "direct_packed_input": kernel["reads_packed_weights_directly"],
            "no_dequantized_weight_cache": (
                not kernel["full_dequantized_weight_cache"]
                and performance["dequantized_weight_cache_bytes"] == 0
            ),
            "projection_correctness": projection_correctness,
            "moe_correctness": moe_correctness,
            "performance_mode_executed": len(samples)
            == document["workload"]["measured_iterations"],
            **prefill_expected,
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Packed Q4 gate does not match measured evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Packed Q4 overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Packed Q4 status is inconsistent")
    elif kind == "mixed_precision_sensitivity":
        rows = document["rows"]
        baseline = document["all_q4_baseline"]
        layers = [row["restored_layer"] for row in rows]
        if sorted(layers) != list(range(16)):
            raise ContractError("Mixed-precision matrix must cover layers 0 through 15")
        for row in rows:
            metrics = row["metrics"]
            expected_delta = {
                "router_exact_set_agreement": (
                    metrics["router_exact_set_agreement"]
                    - baseline["router_exact_set_agreement"]
                ),
                "router_mean_set_overlap": (
                    metrics["router_mean_set_overlap"]
                    - baseline["router_mean_set_overlap"]
                ),
                "logit_cosine_similarity": (
                    metrics["logit_cosine_similarity"]
                    - baseline["logit_cosine_similarity"]
                ),
            }
            if any(
                not math.isclose(
                    row["delta_vs_all_q4"][name], value, rel_tol=1e-9, abs_tol=1e-12
                )
                for name, value in expected_delta.items()
            ):
                raise ContractError("Mixed-precision row delta is inconsistent")
        expected_ranking = [
            row["restored_layer"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["delta_vs_all_q4"]["router_exact_set_agreement"],
                    -row["delta_vs_all_q4"]["logit_cosine_similarity"],
                    row["restored_layer"],
                ),
            )
        ]
        if document["ranking"] != expected_ranking:
            raise ContractError("Mixed-precision ranking is inconsistent")
        storage = document["storage"]
        expected_mixed_bytes = (
            storage["all_q4_bytes"]
            - storage["replaced_q4_layer_bytes"]
            + storage["single_bf16_layer_bytes"]
        )
        if storage["single_layer_mixed_bytes"] != expected_mixed_bytes:
            raise ContractError("Mixed-precision storage accounting is inconsistent")
        gates = document["gates"]
        expected = {
            "reference_finite": document["reference"]["finite"],
            "all_q4_finite": baseline["finite"],
            "all_layers_covered": sorted(layers) == list(range(16)),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": True,
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Mixed-precision gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Mixed-precision overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Mixed-precision status is inconsistent")
    elif kind == "mixed_precision_refinement":
        baseline = document["all_q4_baseline"]
        order = document["method"]["layer_order"]
        cumulative = document["cumulative_layer_rows"]
        experts = document["layer2_expert_rows"]
        if order[0] != 2 or sorted(order) != list(range(16)):
            raise ContractError(
                "Mixed-precision refinement must cover layers 0-15 from layer 2"
            )
        expected_layers: list[int] = []
        for index, row in enumerate(cumulative):
            expected_layers.append(order[index])
            if (
                row["added_layer"] != order[index]
                or row["restored_layers"] != expected_layers
            ):
                raise ContractError(
                    "Mixed-precision cumulative layer progression is inconsistent"
                )
            _validate_refinement_delta(row, baseline)
        expert_ids = [row["restored_expert"] for row in experts]
        if sorted(expert_ids) != list(range(64)):
            raise ContractError(
                "Mixed-precision refinement must cover layer-2 experts 0-63"
            )
        for row in experts:
            _validate_refinement_delta(row, baseline)
        expected_ranking = [
            row["restored_expert"]
            for row in sorted(
                experts,
                key=lambda row: (
                    -row["delta_vs_all_q4"]["router_exact_set_agreement"],
                    -row["delta_vs_all_q4"]["logit_cosine_similarity"],
                    -row["reference_route_count"],
                    row["restored_expert"],
                ),
            )
        ]
        if document["layer2_expert_ranking"] != expected_ranking:
            raise ContractError(
                "Mixed-precision layer-2 expert ranking is inconsistent"
            )
        quality_gate = document["quality_gate"]
        qualifying = [
            row
            for row in cumulative
            if row["metrics"]["router_exact_set_agreement"]
            >= quality_gate["minimum_router_exact_set_agreement"]
        ]
        expected_first = qualifying[0]["restored_layers"] if qualifying else None
        if quality_gate["first_passing_restored_layers"] != expected_first:
            raise ContractError("Mixed-precision quality gate result is inconsistent")
        gates = document["gates"]
        expected = {
            "reference_finite": document["reference"]["finite"],
            "all_q4_finite": baseline["finite"],
            "layer2_first": order[0] == 2,
            "all_layers_covered": len(cumulative) == 16,
            "all_layer2_experts_covered": len(experts) == 64,
            "matrix_finite": all(row["finite"] for row in cumulative + experts),
            "quality_gate_unchanged": math.isclose(
                quality_gate["minimum_router_exact_set_agreement"], 0.99
            ),
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError(
                "Mixed-precision refinement gate does not match evidence"
            )
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError(
                "Mixed-precision refinement overall gate is inconsistent"
            )
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Mixed-precision refinement status is inconsistent")
    elif kind == "mixed_precision_policy_search":
        dataset = document["dataset"]
        reference = document["reference"]
        baseline = document["all_q4_baseline"]
        storage = document["storage"]
        order = document["method"]["expert_order"]
        rows = document["candidate_rows"]
        if dataset["sample_count"] != len(dataset["sample_ids"]):
            raise ContractError("Mixed-precision policy dataset count is inconsistent")
        if len(rows) != len(order):
            raise ContractError("Mixed-precision policy row count is inconsistent")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("Mixed-precision reference perplexity is inconsistent")
        for metrics in [baseline, *(row["metrics"] for row in rows)]:
            expected_ppl = math.exp(metrics["nll"])
            if not math.isclose(
                metrics["perplexity"], expected_ppl, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ContractError("Mixed-precision policy perplexity is inconsistent")
            expected_nll_change = (metrics["nll"] / reference["nll"]) - 1.0
            expected_ppl_change = (
                metrics["perplexity"] / reference["perplexity"]
            ) - 1.0
            if not math.isclose(
                metrics["relative_nll_change"],
                expected_nll_change,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ) or not math.isclose(
                metrics["relative_perplexity_change"],
                expected_ppl_change,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ContractError(
                    "Mixed-precision policy relative quality is inconsistent"
                )
        expected_prefix: list[int] = []
        for index, row in enumerate(rows):
            expected_prefix.append(order[index])
            expected_policy = "layer2-" + "-".join(
                f"e{expert}" for expert in expected_prefix
            )
            expected_extra = len(expected_prefix) * storage["single_expert_extra_bytes"]
            expected_mixed = storage["all_q4_bytes"] + expected_extra
            expected_bpw = expected_mixed * 8 / storage["total_expert_weight_count"]
            if (
                row["added_expert"] != order[index]
                or row["restored_experts"] != expected_prefix
                or row["policy_id"] != expected_policy
                or row["extra_bytes"] != expected_extra
                or row["mixed_bytes"] != expected_mixed
                or not math.isclose(row["effective_bpw"], expected_bpw, rel_tol=1e-9)
            ):
                raise ContractError(
                    "Mixed-precision policy prefix or storage is inconsistent"
                )
        quality = document["quality_gate"]
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and row["metrics"]["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        ]
        expected_first = passing[0]["policy_id"] if passing else None
        if quality["first_passing_policy_id"] != expected_first:
            raise ContractError("Mixed-precision policy gate result is inconsistent")
        gates = document["gates"]
        expected = {
            "reference_finite": reference["finite"],
            "all_q4_finite": baseline["finite"],
            "dataset_complete": dataset["sample_count"] >= 2
            and dataset["target_token_count"] >= 2,
            "prefix_progression": len(rows) == len(order),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Mixed-precision policy gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Mixed-precision policy overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Mixed-precision policy status is inconsistent")
    elif kind == "route_coverage_policy_search":
        dataset = document["dataset"]
        reference = document["reference"]
        baseline = document["all_q4_baseline"]
        storage = document["storage"]
        thresholds = document["method"]["coverage_thresholds"]
        calibration = document["calibration"]["route_counts_by_layer"]
        rows = document["candidate_rows"]
        calibration_ids = dataset["calibration_sample_ids"]
        evaluation_ids = dataset["evaluation_sample_ids"]
        if set(calibration_ids) & set(evaluation_ids):
            raise ContractError("Route coverage dataset splits must be disjoint")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("Route coverage reference perplexity is inconsistent")
        expected_layers = list(range(16))
        if [row["layer_index"] for row in calibration] != expected_layers:
            raise ContractError("Route coverage calibration layers are inconsistent")
        for row in calibration:
            if sum(row["expert_counts"]) != row["total_assignments"]:
                raise ContractError("Route coverage assignment count is inconsistent")
        if len(rows) != len(thresholds) + 1:
            raise ContractError("Route coverage candidate count is inconsistent")
        for metrics in [baseline, *(row["metrics"] for row in rows)]:
            _validate_quality_metrics(metrics, reference, "Route coverage policy")
        previous_sets = [set() for _ in range(16)]
        for index, row in enumerate(rows):
            all_experts = index == len(thresholds)
            expected_mode = "all_experts" if all_experts else "route_coverage"
            expected_threshold = None if all_experts else thresholds[index]
            expected_id = (
                "all-experts-bf16"
                if all_experts
                else f"coverage-{round(thresholds[index] * 100)}-bf16"
            )
            if (
                row["mode"] != expected_mode
                or row["coverage_threshold"] != expected_threshold
                or row["policy_id"] != expected_id
                or [layer["layer_index"] for layer in row["layers"]] != expected_layers
            ):
                raise ContractError("Route coverage candidate identity is inconsistent")
            restored_count = 0
            for layer_index, layer in enumerate(row["layers"]):
                counts = calibration[layer_index]["expert_counts"]
                if all_experts:
                    expected_experts = list(range(64))
                    expected_coverage = 1.0
                else:
                    expected_experts, expected_coverage = _experts_for_route_coverage(
                        counts, thresholds[index]
                    )
                if (
                    layer["expert_ids"] != expected_experts
                    or not math.isclose(
                        layer["assignment_coverage"],
                        expected_coverage,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                    or not previous_sets[layer_index].issubset(expected_experts)
                ):
                    raise ContractError(
                        "Route coverage expert selection is inconsistent"
                    )
                previous_sets[layer_index] = set(expected_experts)
                restored_count += len(expected_experts)
            expected_extra = restored_count * storage["single_expert_extra_bytes"]
            expected_mixed = storage["all_q4_bytes"] + expected_extra
            expected_bpw = expected_mixed * 8 / storage["total_expert_weight_count"]
            if (
                row["restored_expert_count"] != restored_count
                or row["extra_bytes"] != expected_extra
                or row["mixed_bytes"] != expected_mixed
                or not math.isclose(row["effective_bpw"], expected_bpw, rel_tol=1e-9)
            ):
                raise ContractError("Route coverage storage is inconsistent")
        quality = document["quality_gate"]
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and row["metrics"]["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        ]
        expected_first = passing[0]["policy_id"] if passing else None
        if quality["first_passing_policy_id"] != expected_first:
            raise ContractError("Route coverage quality gate result is inconsistent")
        upper = rows[-1]["metrics"]
        upper_bound = (
            math.isclose(upper["relative_perplexity_change"], 0.0, abs_tol=1e-12)
            and math.isclose(upper["router_exact_set_agreement"], 1.0)
            and math.isclose(upper["logit_max_absolute_error"], 0.0, abs_tol=1e-12)
        )
        gates = document["gates"]
        expected = {
            "reference_finite": reference["finite"],
            "all_q4_finite": baseline["finite"],
            "dataset_split_disjoint": not bool(
                set(calibration_ids) & set(evaluation_ids)
            ),
            "calibration_complete": len(calibration) == 16,
            "candidate_progression": len(rows) == len(thresholds) + 1,
            "matrix_finite": all(row["finite"] for row in rows),
            "all_experts_upper_bound": upper_bound,
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Route coverage gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Route coverage overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Route coverage status is inconsistent")
    elif kind == "quantization_compensation_search":
        source_ids = document["method"]["candidate_ids"]
        rows = document["candidate_rows"]
        reference = document["reference"]
        source_baseline = document["source_all_q4_baseline"]
        storage = document["storage"]
        quality = document["quality_gate"]
        expected_specs = [
            ("q4-g128", 4, 128, 0),
            ("q4-g128-r1", 4, 128, 1),
            ("q4-g64", 4, 64, 0),
            ("q4-g128-r2", 4, 128, 2),
            ("q4-g32", 4, 32, 0),
            ("q4-g128-r4", 4, 128, 4),
            ("q5-g128", 5, 128, 0),
            ("q4-g128-r8", 4, 128, 8),
            ("q4-g16", 4, 16, 0),
            ("q6-g128", 6, 128, 0),
            ("q4-g128-r16", 4, 128, 16),
            ("q8-g128", 8, 128, 0),
            ("q9-g128", 9, 128, 0),
            ("q10-g128", 10, 128, 0),
            ("q12-g128", 12, 128, 0),
            ("bf16-upper-bound", 16, None, 0),
        ]
        if source_ids != [spec[0] for spec in expected_specs] or len(rows) != len(
            expected_specs
        ):
            raise ContractError("Compensation candidate progression is inconsistent")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("Compensation reference perplexity is inconsistent")
        _validate_quality_metrics(source_baseline, reference, "Compensation source")
        total_weights = storage["total_expert_weight_count"]
        expected_pack_bpw = storage["all_q4_pack_bytes"] * 8 / total_weights
        if not math.isclose(
            storage["all_q4_pack_effective_bpw"], expected_pack_bpw, rel_tol=1e-12
        ):
            raise ContractError("Compensation Q4 pack storage is inconsistent")
        for row, (candidate_id, bits, group_size, residual_count) in zip(
            rows, expected_specs, strict=True
        ):
            _validate_quality_metrics(row["metrics"], reference, "Compensation")
            if (
                row["candidate_id"] != candidate_id
                or row["bits"] != bits
                or row["group_size"] != group_size
                or row["residual_values_per_group"] != residual_count
                or row["finite"] != row["metrics"]["finite"]
            ):
                raise ContractError("Compensation candidate identity is inconsistent")
            primary_bpw = float(bits)
            scale_bpw = 0.0 if group_size is None else 32.0 / group_size
            residual_bpw = (
                0.0 if group_size is None else residual_count * 24.0 / group_size
            )
            if candidate_id == "q4-g128":
                effective_bpw = expected_pack_bpw
                payload_bytes = storage["all_q4_pack_bytes"]
            elif candidate_id.startswith("q4-g128-r"):
                effective_bpw = expected_pack_bpw + residual_bpw
                payload_bytes = math.ceil(effective_bpw * total_weights / 8)
            else:
                effective_bpw = primary_bpw + scale_bpw + residual_bpw
                payload_bytes = math.ceil(effective_bpw * total_weights / 8)
            if (
                not math.isclose(row["primary_bpw"], primary_bpw, rel_tol=1e-12)
                or not math.isclose(row["scale_bpw"], scale_bpw, rel_tol=1e-12)
                or not math.isclose(row["residual_bpw"], residual_bpw, rel_tol=1e-12)
                or not math.isclose(row["effective_bpw"], effective_bpw, rel_tol=1e-12)
                or row["projected_payload_bytes"] != payload_bytes
            ):
                raise ContractError("Compensation candidate storage is inconsistent")
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and row["metrics"]["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        ]
        expected_first = passing[0]["candidate_id"] if passing else None
        if quality["first_passing_candidate_id"] != expected_first:
            raise ContractError("Compensation quality gate result is inconsistent")
        observed_baseline = rows[0]["metrics"]
        baseline_reproduced = math.isclose(
            observed_baseline["nll"], source_baseline["nll"], abs_tol=1e-6
        ) and math.isclose(
            observed_baseline["router_exact_set_agreement"],
            source_baseline["router_exact_set_agreement"],
            abs_tol=1e-12,
        )
        upper = rows[-1]["metrics"]
        upper_bound = (
            math.isclose(upper["relative_perplexity_change"], 0.0, abs_tol=1e-12)
            and math.isclose(upper["router_exact_set_agreement"], 1.0)
            and math.isclose(upper["logit_max_absolute_error"], 0.0, abs_tol=1e-12)
        )
        expected = {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": reference["finite"],
            "candidate_progression": len(rows) == len(expected_specs),
            "matrix_finite": all(row["finite"] for row in rows),
            "q4_baseline_reproduced": baseline_reproduced,
            "bf16_upper_bound": upper_bound,
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        gates = document["gates"]
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Compensation gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Compensation overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Compensation status is inconsistent")
    elif kind == "router_logit_compensation_search":
        reference = document["reference"]
        baseline = document["all_q4_baseline"]
        source_baseline = document["source_all_q4_baseline"]
        storage = document["storage"]
        rows = document["candidate_rows"]
        quality = document["quality_gate"]
        dataset = document["dataset"]
        expected_specs = [
            ("q4-baseline", "identity", None, None),
            ("bias", "bias", None, None),
            ("diagonal-affine", "diagonal_affine", None, None),
            ("ridge-delta-r8-l1e-4", "ridge_delta", 8, 1e-4),
            ("ridge-delta-r16-l1e-4", "ridge_delta", 16, 1e-4),
            ("ridge-delta-r32-l1e-4", "ridge_delta", 32, 1e-4),
            ("ridge-delta-full-l1e-4", "ridge_delta_full", None, 1e-4),
        ]
        if document["method"]["candidate_ids"] != [
            spec[0] for spec in expected_specs
        ] or len(rows) != len(expected_specs):
            raise ContractError(
                "Router compensation candidate progression is inconsistent"
            )
        calibration_ids = dataset["calibration_sample_ids"]
        evaluation_ids = dataset["evaluation_sample_ids"]
        if set(calibration_ids) & set(evaluation_ids):
            raise ContractError("Router compensation dataset splits must be disjoint")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError(
                "Router compensation reference perplexity is inconsistent"
            )
        _validate_quality_metrics(baseline, reference, "Router compensation baseline")
        _validate_quality_metrics(
            source_baseline, reference, "Router compensation source baseline"
        )
        expected_counts = {
            "identity": 0,
            "bias": 16 * 64,
            "diagonal_affine": 16 * 64 * 2,
            "ridge_delta": None,
            "ridge_delta_full": 16 * (64 * 64 + 64),
        }
        for row, (candidate_id, transform, rank, regularization) in zip(
            rows, expected_specs, strict=True
        ):
            _validate_quality_metrics(
                row["metrics"], reference, "Router compensation candidate"
            )
            expected_count = expected_counts[transform]
            if transform == "ridge_delta":
                expected_count = 16 * (64 * rank + rank * 64 + 64)
            expected_bytes = expected_count * 2
            expected_effective_bpw = (
                (storage["all_q4_pack_bytes"] + expected_bytes)
                * 8
                / storage["total_expert_weight_count"]
            )
            if (
                row["candidate_id"] != candidate_id
                or row["transform"] != transform
                or row["rank"] != rank
                or row["regularization_relative"] != regularization
                or row["parameter_count"] != expected_count
                or row["parameter_bytes_bf16"] != expected_bytes
                or not math.isclose(
                    row["projected_effective_bpw"],
                    expected_effective_bpw,
                    rel_tol=1e-12,
                )
                or row["finite"] != row["metrics"]["finite"]
            ):
                raise ContractError(
                    "Router compensation candidate identity or storage is inconsistent"
                )
        baseline_reproduced = (
            rows[0]["metrics"] == baseline
            and rows[0]["calibration_router_exact_set_agreement"]
            == document["calibration"]["all_q4_router_exact_set_agreement"]
            and math.isclose(baseline["nll"], source_baseline["nll"], abs_tol=1e-6)
            and math.isclose(
                baseline["router_exact_set_agreement"],
                source_baseline["router_exact_set_agreement"],
                abs_tol=1e-12,
            )
        )
        passing = [
            row
            for row in rows
            if row["metrics"]["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and row["metrics"]["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        ]
        expected_first = passing[0]["candidate_id"] if passing else None
        if quality["first_passing_candidate_id"] != expected_first:
            raise ContractError(
                "Router compensation quality gate result is inconsistent"
            )
        gates = document["gates"]
        expected = {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "dataset_split_disjoint": not bool(
                set(calibration_ids) & set(evaluation_ids)
            ),
            "reference_finite": reference["finite"],
            "all_q4_finite": baseline["finite"],
            "q4_baseline_reproduced": baseline_reproduced,
            "candidate_progression": len(rows) == len(expected_specs),
            "matrix_finite": all(row["finite"] for row in rows),
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Router compensation gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Router compensation overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Router compensation status is inconsistent")
    elif kind == "activation_aware_mixed_policy":
        reference = document["reference"]
        q8_base = document["q8_base_metrics"]
        source_q8_base = document["source_q8_base_metrics"]
        policy = document["policy"]
        metrics = policy["metrics"]
        storage = document["storage"]
        quality = document["quality_gate"]
        dataset = document["dataset"]
        q4_layers = policy["q4_layers"]
        q8_layers = policy["q8_layers"]
        bf16_layers = policy["bf16_layers"]
        policy_version = document["schema_version"]
        method = document["method"]
        source_target_id = method.get("source_evidence_target_id")
        cross_target_reproduction = method.get("cross_target_reproduction")
        if (source_target_id is None) != (cross_target_reproduction is None):
            raise ContractError(
                "Activation-aware mixed policy source target provenance is incomplete"
            )
        if source_target_id is not None and cross_target_reproduction != (
            source_target_id != document["target_id"]
        ):
            raise ContractError(
                "Activation-aware mixed policy cross-target provenance is inconsistent"
            )
        expected_layers = {
            1: (
                [15],
                [8, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            ),
            2: (
                [15],
                [11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
        }[policy_version]
        expected_method_id = f"layer15-awq-q4-q8-bf16-mixed-v{policy_version}"
        expected_policy_id = f"olmoe-layer15-awq-q4-q8-bf16-v{policy_version}"
        if (
            document["method"]["id"] != expected_method_id
            or policy["policy_id"] != expected_policy_id
        ):
            raise ContractError(
                "Activation-aware mixed policy version identity is inconsistent"
            )
        if (
            q4_layers != expected_layers[0]
            or q8_layers != expected_layers[1]
            or bf16_layers != expected_layers[2]
        ):
            raise ContractError("Activation-aware mixed policy layers are inconsistent")
        if sorted(q4_layers + q8_layers + bf16_layers) != list(range(16)):
            raise ContractError(
                "Activation-aware mixed policy must partition all layers"
            )
        if set(dataset["calibration_sample_ids"]) & set(
            dataset["evaluation_sample_ids"]
        ):
            raise ContractError("Activation-aware mixed policy splits must be disjoint")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError(
                "Activation-aware mixed policy reference perplexity is inconsistent"
            )
        _validate_quality_metrics(q8_base, reference, "Q8 base policy")
        _validate_quality_metrics(source_q8_base, reference, "Source Q8 base policy")
        _validate_quality_metrics(metrics, reference, "Activation-aware mixed policy")
        q8_reproduced = math.isclose(
            q8_base["nll"], source_q8_base["nll"], abs_tol=1e-6
        ) and math.isclose(
            q8_base["router_exact_set_agreement"],
            source_q8_base["router_exact_set_agreement"],
            abs_tol=1e-12,
        )
        if document["q8_base_reproduced"] != q8_reproduced:
            raise ContractError(
                "Activation-aware mixed policy Q8 reproduction is inconsistent"
            )
        expected_bpw = (
            len(q4_layers) * storage["q4_effective_bpw"]
            + len(q8_layers) * storage["q8_effective_bpw"]
            + len(bf16_layers) * storage["bf16_effective_bpw"]
        ) / 16
        expected_bytes = math.ceil(
            expected_bpw * storage["total_expert_weight_count"] / 8
        )
        if (
            not math.isclose(storage["policy_effective_bpw"], expected_bpw)
            or storage["projected_payload_bytes"] != expected_bytes
            or not math.isclose(policy["effective_bpw"], expected_bpw)
            or policy["projected_payload_bytes"] != expected_bytes
        ):
            raise ContractError("Activation-aware mixed policy storage is inconsistent")
        q8_passed = (
            q8_base["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and q8_base["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        )
        policy_passed = (
            metrics["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and metrics["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        )
        gates = document["gates"]
        expected = {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "dataset_split_disjoint": not bool(
                set(dataset["calibration_sample_ids"])
                & set(dataset["evaluation_sample_ids"])
            ),
            "reference_finite": reference["finite"],
            "q8_base_reproduced": q8_reproduced,
            "q8_base_quality_passed": q8_passed,
            "policy_finite": metrics["finite"],
            "policy_quality_passed": policy_passed,
            "storage_accounting": True,
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        if source_target_id is not None:
            expected["source_target_compatible"] = True
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError(
                "Activation-aware mixed policy gate does not match evidence"
            )
        overall = all(
            value
            for name, value in expected.items()
            if policy_version == 1 or name != "q8_base_reproduced"
        )
        if gates["overall_passed"] != overall:
            raise ContractError(
                "Activation-aware mixed policy overall gate is inconsistent"
            )
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Activation-aware mixed policy status is inconsistent")
    elif kind == "target_pack_host_quality":
        reference = document["reference"]
        candidate = document["candidate"]
        quality = document["quality_gate"]
        loader = document["loader"]
        target_pack = document["target_pack"]
        layer_encodings = target_pack["layer_encodings"]
        expected_policy_layers = {
            "olmoe-layer15-awq-q4-q8-bf16-v1": {
                "q4_layers": [15],
                "q8_layers": [8, 11, 12, 13, 14],
                "bf16_layers": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            },
            "olmoe-layer15-awq-q4-q8-bf16-v2": {
                "q4_layers": [15],
                "q8_layers": [11, 12, 13, 14],
                "bf16_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
        }
        if layer_encodings != expected_policy_layers[target_pack["policy_id"]]:
            raise ContractError("TargetPack Host policy layers are inconsistent")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("TargetPack Host reference perplexity is inconsistent")
        _validate_quality_metrics(candidate, reference, "TargetPack Host")
        quality_passed = (
            candidate["relative_perplexity_change"]
            <= quality["maximum_relative_perplexity_increase"]
            and candidate["router_exact_set_agreement"]
            >= quality["minimum_router_exact_set_agreement"]
        )
        expected = {
            "manifest_identity": True,
            "policy_evidence_identity": True,
            "dataset_identity": True,
            "reference_finite": reference["finite"],
            "candidate_finite": candidate["finite"],
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
                candidate["per_layer_router_exact_set_agreement"]
            )
            == 16,
        }
        gates = document["gates"]
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("TargetPack Host gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("TargetPack Host overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("TargetPack Host status is inconsistent")
    elif kind == "layer_precision_search":
        reference = document["reference"]
        sources = document["source_uniform_metrics"]
        searches = document["searches"]
        storage = document["storage"]
        quality = document["quality_gate"]
        expected_bits = [8, 9, 12]
        if (
            storage["expert_weight_count_per_layer"] * 16
            != storage["total_expert_weight_count"]
        ):
            raise ContractError("Layer precision storage partition is inconsistent")
        if [row["base_bits"] for row in sources] != expected_bits or [
            row["candidate_id"] for row in sources
        ] != ["q8-g128", "q9-g128", "q12-g128"]:
            raise ContractError("Layer precision source rows are inconsistent")
        if [search["base_bits"] for search in searches] != expected_bits:
            raise ContractError("Layer precision search order is inconsistent")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("Layer precision reference perplexity is inconsistent")
        for source in sources:
            _validate_quality_metrics(
                source["metrics"], reference, "Layer precision source"
            )
        passing_policies: list[dict[str, Any]] = []
        base_reproduced = True
        all_layers_covered = True
        matrix_finite = True
        upper_bounds = True
        for source, search in zip(sources, searches, strict=True):
            bits = search["base_bits"]
            base_bpw = bits + 0.25
            if not math.isclose(search["base_effective_bpw"], base_bpw):
                raise ContractError("Layer precision base storage is inconsistent")
            _validate_quality_metrics(
                search["base_metrics"], reference, "Layer precision base"
            )
            base_reproduced = (
                base_reproduced
                and math.isclose(
                    search["base_metrics"]["nll"],
                    source["metrics"]["nll"],
                    abs_tol=1e-6,
                )
                and math.isclose(
                    search["base_metrics"]["router_exact_set_agreement"],
                    source["metrics"]["router_exact_set_agreement"],
                    abs_tol=1e-12,
                )
            )
            singles = search["single_layer_rows"]
            cumulative = search["cumulative_rows"]
            for row in singles:
                _validate_quality_metrics(
                    row["metrics"], reference, "Layer precision single"
                )
                if row["finite"] != row["metrics"]["finite"]:
                    raise ContractError("Layer precision finite flag is inconsistent")
            expected_ranking = [
                row["restored_layer"]
                for row in sorted(
                    singles,
                    key=lambda row: (
                        -row["metrics"]["router_exact_set_agreement"],
                        row["metrics"]["relative_perplexity_change"],
                        row["restored_layer"],
                    ),
                )
            ]
            if search["ranking"] != expected_ranking:
                raise ContractError("Layer precision ranking is inconsistent")
            restored: list[int] = []
            for index, row in enumerate(cumulative):
                restored.append(expected_ranking[index])
                _validate_quality_metrics(
                    row["metrics"], reference, "Layer precision cumulative"
                )
                effective_bpw = base_bpw + len(restored) * (16.0 - base_bpw) / 16
                payload_bytes = math.ceil(
                    effective_bpw * storage["total_expert_weight_count"] / 8
                )
                if (
                    row["added_layer"] != expected_ranking[index]
                    or row["restored_layers"] != restored
                    or not math.isclose(row["effective_bpw"], effective_bpw)
                    or row["projected_payload_bytes"] != payload_bytes
                    or row["finite"] != row["metrics"]["finite"]
                ):
                    raise ContractError(
                        "Layer precision cumulative progression is inconsistent"
                    )
            qualifying = [
                row
                for row in cumulative
                if row["metrics"]["relative_perplexity_change"]
                <= quality["maximum_relative_perplexity_increase"]
                and row["metrics"]["router_exact_set_agreement"]
                >= quality["minimum_router_exact_set_agreement"]
            ]
            expected_first = qualifying[0]["restored_layers"] if qualifying else None
            if search["first_passing_restored_layers"] != expected_first:
                raise ContractError("Layer precision passing prefix is inconsistent")
            if (
                search["base_metrics"]["relative_perplexity_change"]
                <= quality["maximum_relative_perplexity_increase"]
                and search["base_metrics"]["router_exact_set_agreement"]
                >= quality["minimum_router_exact_set_agreement"]
            ):
                passing_policies.append(
                    {
                        "base_bits": bits,
                        "restored_layers": [],
                        "effective_bpw": base_bpw,
                    }
                )
            passing_policies.extend(
                {
                    "base_bits": bits,
                    "restored_layers": row["restored_layers"],
                    "effective_bpw": row["effective_bpw"],
                }
                for row in qualifying
            )
            all_layers_covered = all_layers_covered and (
                [row["restored_layer"] for row in singles] == list(range(16))
                and sorted(search["ranking"]) == list(range(16))
                and cumulative[-1]["restored_layers"] == search["ranking"]
            )
            matrix_finite = (
                matrix_finite
                and search["base_metrics"]["finite"]
                and all(row["finite"] for row in singles + cumulative)
            )
            upper = cumulative[-1]["metrics"]
            upper_bounds = upper_bounds and (
                math.isclose(upper["relative_perplexity_change"], 0.0, abs_tol=1e-12)
                and math.isclose(upper["router_exact_set_agreement"], 1.0)
                and math.isclose(upper["logit_max_absolute_error"], 0.0, abs_tol=1e-12)
            )
        expected_lowest = (
            min(
                passing_policies,
                key=lambda row: (
                    row["effective_bpw"],
                    row["base_bits"],
                    len(row["restored_layers"]),
                ),
            )
            if passing_policies
            else None
        )
        if quality["lowest_bpw_passing_policy"] != expected_lowest:
            raise ContractError("Layer precision lowest-bpw policy is inconsistent")
        expected = {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": reference["finite"],
            "base_metrics_reproduced": base_reproduced,
            "all_layers_covered": all_layers_covered,
            "matrix_finite": matrix_finite,
            "bf16_upper_bounds": upper_bounds,
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        gates = document["gates"]
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Layer precision gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Layer precision overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Layer precision status is inconsistent")
    elif kind == "reverse_layer_quantization_search":
        reference = document["reference"]
        sources = document["source_uniform_metrics"]
        searches = document["searches"]
        storage = document["storage"]
        quality = document["quality_gate"]
        expected_bits = [4, 8, 12]
        if (
            storage["expert_weight_count_per_layer"] * 16
            != storage["total_expert_weight_count"]
        ):
            raise ContractError("Reverse layer storage partition is inconsistent")
        if [row["quantized_bits"] for row in sources] != expected_bits or [
            row["candidate_id"] for row in sources
        ] != ["q4-g128", "q8-g128", "q12-g128"]:
            raise ContractError("Reverse layer source rows are inconsistent")
        if [search["quantized_bits"] for search in searches] != expected_bits:
            raise ContractError("Reverse layer search order is inconsistent")
        if not math.isclose(
            reference["perplexity"],
            math.exp(reference["nll"]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ContractError("Reverse layer reference perplexity is inconsistent")
        for source in sources:
            _validate_quality_metrics(source["metrics"], reference, "Reverse source")
        passing_policies: list[dict[str, Any]] = []
        endpoints_reproduced = True
        all_layers_covered = True
        matrix_finite = True
        for source, search in zip(sources, searches, strict=True):
            bits = search["quantized_bits"]
            quantized_bpw = (
                storage["q4_pack_effective_bpw"] if bits == 4 else bits + 0.25
            )
            if not math.isclose(search["quantized_effective_bpw"], quantized_bpw):
                raise ContractError("Reverse layer base storage is inconsistent")
            singles = search["single_layer_rows"]
            cumulative = search["cumulative_rows"]
            for row in singles:
                _validate_quality_metrics(row["metrics"], reference, "Reverse single")
                if row["finite"] != row["metrics"]["finite"]:
                    raise ContractError("Reverse layer finite flag is inconsistent")
            expected_ranking = [
                row["quantized_layer"]
                for row in sorted(
                    singles,
                    key=lambda row: (
                        -row["metrics"]["router_exact_set_agreement"],
                        row["metrics"]["relative_perplexity_change"],
                        row["quantized_layer"],
                    ),
                )
            ]
            if search["ranking"] != expected_ranking:
                raise ContractError("Reverse layer ranking is inconsistent")
            quantized_layers: list[int] = []
            qualifying: list[Mapping[str, Any]] = []
            for index, row in enumerate(cumulative):
                quantized_layers.append(expected_ranking[index])
                _validate_quality_metrics(
                    row["metrics"], reference, "Reverse cumulative"
                )
                effective_bpw = (
                    16.0 - len(quantized_layers) * (16.0 - quantized_bpw) / 16
                )
                payload_bytes = math.ceil(
                    effective_bpw * storage["total_expert_weight_count"] / 8
                )
                if (
                    row["added_quantized_layer"] != expected_ranking[index]
                    or row["quantized_layers"] != quantized_layers
                    or not math.isclose(row["effective_bpw"], effective_bpw)
                    or row["projected_payload_bytes"] != payload_bytes
                    or row["finite"] != row["metrics"]["finite"]
                ):
                    raise ContractError(
                        "Reverse layer cumulative progression is inconsistent"
                    )
                if (
                    row["metrics"]["relative_perplexity_change"]
                    <= quality["maximum_relative_perplexity_increase"]
                    and row["metrics"]["router_exact_set_agreement"]
                    >= quality["minimum_router_exact_set_agreement"]
                ):
                    qualifying.append(row)
            expected_lowest_layers = (
                min(qualifying, key=lambda row: row["effective_bpw"])[
                    "quantized_layers"
                ]
                if qualifying
                else None
            )
            if search["lowest_bpw_passing_quantized_layers"] != expected_lowest_layers:
                raise ContractError("Reverse layer passing subset is inconsistent")
            passing_policies.extend(
                {
                    "quantized_bits": bits,
                    "quantized_layers": row["quantized_layers"],
                    "effective_bpw": row["effective_bpw"],
                }
                for row in qualifying
            )
            endpoint = cumulative[-1]["metrics"]
            endpoints_reproduced = (
                endpoints_reproduced
                and math.isclose(
                    endpoint["nll"], source["metrics"]["nll"], abs_tol=1e-6
                )
                and math.isclose(
                    endpoint["router_exact_set_agreement"],
                    source["metrics"]["router_exact_set_agreement"],
                    abs_tol=1e-12,
                )
            )
            all_layers_covered = all_layers_covered and (
                [row["quantized_layer"] for row in singles] == list(range(16))
                and sorted(search["ranking"]) == list(range(16))
                and cumulative[-1]["quantized_layers"] == search["ranking"]
            )
            matrix_finite = matrix_finite and all(
                row["finite"] for row in singles + cumulative
            )
        expected_lowest = (
            min(
                passing_policies,
                key=lambda row: (
                    row["effective_bpw"],
                    row["quantized_bits"],
                    -len(row["quantized_layers"]),
                ),
            )
            if passing_policies
            else None
        )
        if quality["lowest_bpw_passing_policy"] != expected_lowest:
            raise ContractError("Reverse layer lowest-bpw policy is inconsistent")
        expected = {
            "source_evidence_compatible": True,
            "dataset_identity": True,
            "reference_finite": reference["finite"],
            "uniform_endpoints_reproduced": endpoints_reproduced,
            "all_layers_covered": all_layers_covered,
            "matrix_finite": matrix_finite,
            "quality_gate_unchanged": math.isclose(
                quality["maximum_relative_perplexity_increase"], 0.01
            )
            and math.isclose(quality["minimum_router_exact_set_agreement"], 0.99),
        }
        gates = document["gates"]
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Reverse layer gate does not match evidence")
        overall = all(expected.values())
        if gates["overall_passed"] != overall:
            raise ContractError("Reverse layer overall gate is inconsistent")
        if document["status"] != ("passed" if overall else "failed"):
            raise ContractError("Reverse layer status is inconsistent")
    elif kind == "traffic_source_ledger":
        from .traffic_model import (
            TrafficModelError,
            validate_traffic_source_ledger,
        )

        try:
            validate_traffic_source_ledger(document)
        except TrafficModelError as exc:
            raise ContractError(str(exc)) from exc
    elif kind == "spark_traffic_model":
        from .traffic_model import TrafficModelError, validate_spark_traffic_model

        try:
            validate_spark_traffic_model(document)
        except TrafficModelError as exc:
            raise ContractError(str(exc)) from exc
    elif kind == "run_manifest":
        forbidden_fragments = (
            "SECRET",
            "TOKEN",
            "PASSWORD",
            "CREDENTIAL",
            "PRIVATE_KEY",
        )
        names = document["command"]["environment_allowlist"]
        unsafe_names = [
            name
            for name in names
            if any(fragment in name.upper() for fragment in forbidden_fragments)
        ]
        if unsafe_names:
            raise ContractError(
                "$.command.environment_allowlist contains secret-like names: "
                + ", ".join(sorted(unsafe_names))
            )
        artifact_paths = [item["path"] for item in document["raw_artifacts"]]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ContractError("RunManifest raw artifacts contain duplicate paths")
        for path in artifact_paths:
            _require_project_relative_path(path, "RunManifest raw artifact path")
    elif kind == "allocation_matrix_v2":
        from .allocation_matrix_v2 import validate_allocation_matrix_v2_document

        validate_allocation_matrix_v2_document(document)
    elif kind == "reference_oracle_comparison":
        from .reference_oracle import (
            ReferenceOracleError,
            validate_reference_oracle_comparison,
        )

        try:
            validate_reference_oracle_comparison(document)
        except ReferenceOracleError as exc:
            raise ContractError(str(exc)) from exc
    elif kind == "reference_oracle_policy":
        from .fixed_models import fixed_model_spec

        spec = fixed_model_spec(document["model_id"], document["model_revision"])
        scope = document["scope"]
        level = scope["level"]
        layer = scope["layer_index"]
        expert = scope["expert_index"]
        if level == "single_expert":
            if layer is None or expert is None:
                raise ContractError("single_expert policy requires layer and expert")
            if not 0 <= layer < spec.num_layers or not 0 <= expert < spec.num_experts:
                raise ContractError(
                    "single_expert policy layer or expert exceeds fixed model shape"
                )
            if document["policy"]["min_router_top_k_set_agreement"] is not None:
                raise ContractError(
                    "single_expert policy must not set router agreement"
                )
        elif level == "single_moe_layer":
            if layer is None or expert is not None:
                raise ContractError("single_moe_layer policy requires only layer")
            if not 0 <= layer < spec.num_layers:
                raise ContractError(
                    "single_moe_layer policy layer exceeds fixed model shape"
                )
            if document["policy"]["min_router_top_k_set_agreement"] is None:
                raise ContractError("single_moe_layer policy requires router agreement")
        else:
            if layer is not None or expert is not None:
                raise ContractError("full_model policy forbids layer and expert")
            if document["policy"]["min_router_top_k_set_agreement"] is None:
                raise ContractError("full_model policy requires router agreement")
    elif kind == "quantization_dataset_manifest":
        from .quantization_calibration import validate_quantization_dataset_manifest

        validate_quantization_dataset_manifest(document, require_frozen=require_frozen)
    elif kind == "expert_calibration_coverage":
        from .quantization_calibration import validate_expert_calibration_coverage

        validate_expert_calibration_coverage(document)
    elif kind == "expert_balanced_sample_manifest":
        from .quantization_calibration import (
            validate_expert_balanced_sample_manifest,
        )

        validate_expert_balanced_sample_manifest(document)
    elif kind == "route_trace":
        from .route_trace import validate_route_trace_document

        validate_route_trace_document(document)
    elif kind == "route_trace_replay":
        if document["event_layer_records"] != (
            document["event_count"] * document["layer_count"]
        ):
            raise ContractError("RouteTrace replay record count is inconsistent")
    elif kind == "public_baseline":
        results = document["results"]
        duration = results["duration_seconds"]
        expected_throughput = {
            "request_throughput": results["completed_requests"] / duration,
            "output_token_throughput": results["total_output_tokens"] / duration,
            "total_token_throughput": (
                results["total_input_tokens"] + results["total_output_tokens"]
            )
            / duration,
        }
        if any(
            not math.isclose(results[field], expected, rel_tol=1e-6, abs_tol=1e-9)
            for field, expected in expected_throughput.items()
        ):
            raise ContractError(
                "Public Baseline throughput does not match counts/duration"
            )
        for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
            summary = results[name]
            if (
                not (
                    summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]
                )
                or summary["mean"] > summary["max"]
            ):
                raise ContractError(f"Public Baseline {name} summary is invalid")
        paths = [artifact["path"] for artifact in document["raw_artifacts"]]
        if len(paths) != len(set(paths)):
            raise ContractError("Public Baseline raw artifacts contain duplicate paths")
        gates = document["gates"]
        expected_overall = all(
            value for name, value in gates.items() if name != "overall_passed"
        )
        if gates["overall_passed"] != expected_overall:
            raise ContractError("Public Baseline overall gate is inconsistent")
        expected_status = "passed" if expected_overall else "failed"
        if document["status"] != expected_status:
            raise ContractError("Public Baseline status does not match its gates")
    elif kind == "external_baseline":
        from .baseline_common import validate_baseline_semantics

        validate_baseline_semantics(document, "External Baseline")
    elif kind == "expert_pack_manifest":
        header = document["header"]
        if header["tensor_count"] != len(header["tensors"]):
            raise ContractError("ExpertPack manifest tensor count is inconsistent")
        names = [tensor["name"] for tensor in header["tensors"]]
        if len(names) != len(set(names)):
            raise ContractError("ExpertPack manifest contains duplicate tensor names")
        if document["artifact"]["size_bytes"] != (
            document["payload_offset"] + document["payload_length"]
        ):
            raise ContractError("ExpertPack manifest artifact size is inconsistent")
    elif kind == "target_pack_manifest":
        header = document["header"]
        if header["tensor_count"] != len(header["tensors"]):
            raise ContractError("TargetPack manifest tensor count is inconsistent")
        names = [tensor["name"] for tensor in header["tensors"]]
        if len(names) != len(set(names)):
            raise ContractError("TargetPack manifest contains duplicate tensor names")
        observed_counts = {
            encoding: sum(
                tensor["encoding"] == encoding for tensor in header["tensors"]
            )
            for encoding in ("bf16_le", "q4_group128", "q8_group128")
        }
        if header["encoding_tensor_counts"] != observed_counts:
            raise ContractError(
                "TargetPack manifest encoding tensor counts are inconsistent"
            )
        if document["artifact"]["size_bytes"] != (
            document["payload_offset"] + document["payload_length"]
        ):
            raise ContractError("TargetPack manifest artifact size is inconsistent")
    elif kind == "reference_host_baseline":
        from .baseline_common import validate_baseline_semantics

        validate_baseline_semantics(document, "Reference Host Baseline")
    elif kind == "compressed_host_baseline":
        from .baseline_common import validate_baseline_semantics

        validate_baseline_semantics(document, "Compressed Host Baseline")
        if document["schema_version"] == 1:
            expected_quantization = "canonical_q4_group128"
            expected_policy = None
        else:
            expected_quantization = "mixed_target_pack_q4_q8_bf16"
            expected_policy = "olmoe-layer15-awq-q4-q8-bf16-v2"
        model = document["model"]
        if (
            model["expert_quantization"] != expected_quantization
            or model.get("target_policy_id") != expected_policy
        ):
            raise ContractError(
                "Compressed Host schema version does not match its expert policy"
            )
        loader = document["loader"]
        cache = document["backend_cache"]
        gates = document["gates"]
        expected = {
            "packed_backend_registered": document["implementation"]["platform"]
            in {"cuda_sm121", "hip_gfx1151"},
            "performance_mode_executed": loader["performance_mode"],
            "no_silent_fallback": document["implementation"]["fallback_count"] == 0,
            "cache_stable_after_warmup": cache["after_warmup"]
            == cache["after_measurement"],
            "no_dequantized_weight_cache": cache["after_measurement"][
                "dequantized_weight_bytes"
            ]
            == 0,
            "dense_only_checkpoint_load": loader["loaded_expert_tensor_count"] == 0,
            "no_expert_parameters": loader["expert_parameter_count"] == 0,
            "single_pack_mapping": loader["expert_pack_mapping_count"] == 1,
            "within_safe_uma_budget": document["memory"]["peak_bytes"]
            <= document["resource_policy"]["safe_uma_budget_bytes"],
        }
        if any(gates[name] != value for name, value in expected.items()):
            raise ContractError("Compressed Host gate does not match measured evidence")
    elif kind == "target_inventory":
        target_ids = [target["id"] for target in document["targets"]]
        ssh_hosts = [target["ssh_host"] for target in document["targets"]]
        if len(target_ids) != len(set(target_ids)):
            raise ContractError("$.targets contains duplicate target ids")
        if len(ssh_hosts) != len(set(ssh_hosts)):
            raise ContractError("$.targets contains duplicate ssh_host aliases")
    elif kind == "safe_uma_budget":
        component_names = (
            "os_daemon_reserve_bytes",
            "runtime_reserve_bytes",
            "kv_budget_bytes",
            "workspace_budget_bytes",
            "safety_margin_bytes",
        )
        observed_reserved = sum(document[name] for name in component_names)
        if observed_reserved != document["total_reserved_bytes"]:
            raise ContractError(
                "$.total_reserved_bytes does not equal the reserve components"
            )
        cgroup_limit = document["cgroup_memory_max_bytes"]
        expected_physical_limit = (
            document["mem_total_bytes"]
            if cgroup_limit is None
            else min(document["mem_total_bytes"], cgroup_limit)
        )
        if document["physical_limit_bytes"] != expected_physical_limit:
            raise ContractError(
                "$.physical_limit_bytes does not equal min(MemTotal, finite cgroup limit)"
            )
        expected_budget = document["physical_limit_bytes"] - observed_reserved
        if expected_budget < 0 or expected_budget != document["safe_budget_bytes"]:
            raise ContractError(
                "$.safe_budget_bytes does not match physical limit minus reserves"
            )
        if require_frozen and document["status"] != "frozen":
            raise ContractError("Safe UMA Budget is draft; frozen validation requested")
        if document["status"] == "frozen":
            if not document["policy_provenance"].strip():
                raise ContractError("frozen Safe UMA Budget requires policy_provenance")
            if not document["decision_record"].strip():
                raise ContractError("frozen Safe UMA Budget requires decision_record")
    elif kind == "tensor_inventory":
        shard_paths = [shard["path"] for shard in document["shards"]]
        if len(shard_paths) != len(set(shard_paths)):
            raise ContractError("$.shards contains duplicate shard paths")
        for path in shard_paths:
            _require_project_relative_path(path, f"TensorInventory shard path {path!r}")
        tensor_names = [
            tensor["name"]
            for shard in document["shards"]
            for tensor in shard["tensors"]
        ]
        if len(tensor_names) != len(set(tensor_names)):
            raise ContractError("TensorInventory contains duplicate tensor names")
        observed_tensor_count = sum(
            len(shard["tensors"]) for shard in document["shards"]
        )
        if observed_tensor_count != document["tensor_count"]:
            raise ContractError(
                "$.tensor_count does not equal the shard tensor entries"
            )
        observed_dtypes = sorted(
            {
                tensor["dtype"]
                for shard in document["shards"]
                for tensor in shard["tensors"]
            }
        )
        if observed_dtypes != document["observed_dtypes"]:
            raise ContractError(
                "$.observed_dtypes does not equal the shard tensor dtypes"
            )
    elif kind == "model_acquisition":
        file_paths = [entry["path"] for entry in document["files"]]
        if len(file_paths) != len(set(file_paths)):
            raise ContractError("$.files contains duplicate paths")
        for path in file_paths:
            _require_project_relative_path(path, f"model acquisition path {path!r}")
        if document["file_count"] != len(document["files"]):
            raise ContractError(
                "$.file_count does not equal the number of file entries"
            )
        total_bytes = sum(entry["size_bytes"] for entry in document["files"])
        if document["total_bytes"] != total_bytes:
            raise ContractError("$.total_bytes does not equal the file entry sizes")
    elif kind == "artifact_verification":
        _require_project_relative_path(
            document["source_contract_path"], "$.source_contract_path"
        )
        file_paths = [entry["path"] for entry in document["files"]]
        if len(file_paths) != len(set(file_paths)):
            raise ContractError("$.files contains duplicate paths")
        for path in file_paths:
            _require_project_relative_path(path, f"artifact verification path {path!r}")
        if document["file_count"] != len(document["files"]):
            raise ContractError(
                "$.file_count does not equal the number of file entries"
            )
        total_bytes = sum(entry["size_bytes"] for entry in document["files"])
        if document["total_bytes"] != total_bytes:
            raise ContractError("$.total_bytes does not equal the file entry sizes")
    elif kind == "oracle_smoke":
        _require_project_relative_path(
            document["model"]["derivation_path"], "$.model.derivation_path"
        )
        _require_project_relative_path(
            document["model"]["artifact_root"], "$.model.artifact_root"
        )
        _require_project_relative_path(
            document["input"]["fixture_path"], "$.input.fixture_path"
        )
    elif kind == "memory_bandwidth_benchmark":
        expected_operations = {"read_reduce", "write_fill", "copy"}
        operations = document["operations"]
        operation_ids = {operation["id"] for operation in operations}
        if operation_ids != expected_operations or len(operations) != 3:
            raise ContractError(
                "$.operations must contain read_reduce, write_fill, and copy exactly once"
            )
        actual_bytes = document["configuration"]["actual_buffer_bytes"]
        inner_iterations = document["configuration"].get("inner_iterations", 1)
        expected_traffic = {
            "read_reduce": (actual_bytes * inner_iterations, 0),
            "write_fill": (0, actual_bytes * inner_iterations),
            "copy": (actual_bytes * inner_iterations, actual_bytes * inner_iterations),
        }
        for operation in operations:
            expected_read, expected_write = expected_traffic[operation["id"]]
            if operation["algorithmic_read_bytes"] != expected_read:
                raise ContractError(
                    f"operation {operation['id']!r} has invalid algorithmic_read_bytes"
                )
            if operation["algorithmic_write_bytes"] != expected_write:
                raise ContractError(
                    f"operation {operation['id']!r} has invalid algorithmic_write_bytes"
                )
            if operation["algorithmic_total_bytes"] != expected_read + expected_write:
                raise ContractError(
                    f"operation {operation['id']!r} has invalid algorithmic_total_bytes"
                )
        telemetry = document.get("telemetry")
        if telemetry is not None:
            measured_iterations = document["configuration"]["measured_iterations"]
            observed = [
                (sample["operation_id"], sample["sample_index"], sample["boundary"])
                for sample in telemetry["samples"]
            ]
            expected = [
                (operation_id, sample_index, boundary)
                for operation_id in ("read_reduce", "write_fill", "copy")
                for sample_index in range(measured_iterations)
                for boundary in ("before", "after")
            ]
            if observed != expected:
                raise ContractError(
                    "$.telemetry.samples must contain ordered before/after boundaries "
                    "for every measured operation sample"
                )
    elif kind == "bandwidth_soak":
        operation_ids = ("read_reduce", "write_fill", "copy")
        samples = document["samples"]
        sample_count = len(samples)
        if [sample["sample_index"] for sample in samples] != list(range(sample_count)):
            raise ContractError(
                "$.samples must use consecutive zero-based sample_index values"
            )
        monotonic = [sample["monotonic_ns"] for sample in samples]
        elapsed = [sample["elapsed_from_start_seconds"] for sample in samples]
        if any(
            current <= previous for previous, current in zip(monotonic, monotonic[1:])
        ):
            raise ContractError(
                "$.samples monotonic_ns values must be strictly increasing"
            )
        if any(current <= previous for previous, current in zip(elapsed, elapsed[1:])):
            raise ContractError("$.samples elapsed values must be strictly increasing")
        configuration = document["configuration"]
        if not math.isclose(
            configuration["actual_duration_seconds"],
            elapsed[-1],
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ContractError(
                "$.configuration.actual_duration_seconds must match the last sample"
            )
        actual_bytes = configuration["actual_buffer_bytes"]
        inner_iterations = configuration["inner_iterations"]
        expected_traffic = {
            "read_reduce": (actual_bytes * inner_iterations, 0),
            "write_fill": (0, actual_bytes * inner_iterations),
            "copy": (actual_bytes * inner_iterations, actual_bytes * inner_iterations),
        }
        operations = document["operations"]
        observed_ids = [operation["id"] for operation in operations]
        if len(observed_ids) != 3 or set(observed_ids) != set(operation_ids):
            raise ContractError(
                "$.operations must contain every soak operation exactly once"
            )
        by_id = {operation["id"]: operation for operation in operations}
        for operation_id in operation_ids:
            operation = by_id[operation_id]
            expected_read, expected_write = expected_traffic[operation_id]
            total_bytes = expected_read + expected_write
            if operation["algorithmic_read_bytes"] != expected_read:
                raise ContractError(
                    f"soak operation {operation_id!r} has invalid read bytes"
                )
            if operation["algorithmic_write_bytes"] != expected_write:
                raise ContractError(
                    f"soak operation {operation_id!r} has invalid write bytes"
                )
            if operation["algorithmic_total_bytes"] != total_bytes:
                raise ContractError(
                    f"soak operation {operation_id!r} has invalid total bytes"
                )
            if operation["sample_count"] != sample_count:
                raise ContractError(
                    f"soak operation {operation_id!r} has invalid sample count"
                )
            bandwidth = [
                total_bytes / sample["timings_seconds"][operation_id] / 1_000_000_000
                for sample in samples
            ]
            mean = statistics.mean(bandwidth)
            window = max(3, math.ceil(sample_count * 0.10))
            first_window = statistics.median(bandwidth[:window])
            last_window = statistics.median(bandwidth[-window:])
            expected_summary = {
                "minimum_gbps": min(bandwidth),
                "p01_gbps": _nearest_rank_percentile(bandwidth, 0.01),
                "p50_gbps": _nearest_rank_percentile(bandwidth, 0.50),
                "p95_gbps": _nearest_rank_percentile(bandwidth, 0.95),
                "p99_gbps": _nearest_rank_percentile(bandwidth, 0.99),
                "maximum_gbps": max(bandwidth),
                "mean_gbps": mean,
                "coefficient_of_variation": statistics.pstdev(bandwidth) / mean,
                "first_window_p50_gbps": first_window,
                "last_window_p50_gbps": last_window,
                "drift_fraction": (last_window - first_window) / first_window,
            }
            if operation["drift_window_samples"] != window:
                raise ContractError(
                    f"soak operation {operation_id!r} has invalid drift window"
                )
            for field, expected_value in expected_summary.items():
                if not math.isclose(
                    operation[field], expected_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ContractError(
                        f"soak operation {operation_id!r} has invalid {field}"
                    )
        memory = document["memory"]
        if memory["swap_in_pages_delta"] != (
            memory["swap_in_pages_after"] - memory["swap_in_pages_before"]
        ) or memory["swap_out_pages_delta"] != (
            memory["swap_out_pages_after"] - memory["swap_out_pages_before"]
        ):
            raise ContractError(
                "$.memory swap deltas do not match before/after counters"
            )
        telemetry = document["telemetry"]
        telemetry_samples = telemetry["samples"]
        telemetry_monotonic = [sample["monotonic_ns"] for sample in telemetry_samples]
        if any(
            current <= previous
            for previous, current in zip(telemetry_monotonic, telemetry_monotonic[1:])
        ):
            raise ContractError("$.telemetry.samples must be strictly monotonic")
        if any(sample["sample_index"] >= sample_count for sample in telemetry_samples):
            raise ContractError("$.telemetry.samples references an unknown soak sample")
        for field, summary in telemetry["summary"].items():
            values = [
                float(sample[field])
                for sample in telemetry_samples
                if sample[field] is not None
            ]
            expected_values = {
                "available_samples": len(values),
                "minimum": min(values) if values else None,
                "p50": _nearest_rank_percentile(values, 0.50) if values else None,
                "p95": _nearest_rank_percentile(values, 0.95) if values else None,
                "maximum": max(values) if values else None,
            }
            for summary_field, expected_value in expected_values.items():
                observed_value = summary[summary_field]
                if expected_value is None:
                    if observed_value is not None:
                        raise ContractError(
                            f"telemetry summary {field!r} should be unavailable"
                        )
                elif summary_field == "available_samples":
                    if observed_value != expected_value:
                        raise ContractError(
                            f"telemetry summary {field!r} has invalid count"
                        )
                elif not math.isclose(
                    observed_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ContractError(f"telemetry summary {field!r} is invalid")
        duration_passed = (
            configuration["requested_duration_seconds"]
            >= configuration["minimum_full_soak_seconds"]
            and configuration["actual_duration_seconds"]
            >= configuration["requested_duration_seconds"]
        )
        workload_cgroup = memory.get("workload_cgroup")
        if workload_cgroup is None:
            no_swap_passed = (
                memory["swap_in_pages_delta"] == 0
                and memory["swap_out_pages_delta"] == 0
            )
            swap_disabled_passed = None
            no_oom_events_passed = None
        else:
            if workload_cgroup["swap_current_bytes_delta"] != (
                workload_cgroup["swap_current_bytes_after"]
                - workload_cgroup["swap_current_bytes_before"]
            ):
                raise ContractError("$.memory.workload_cgroup swap delta is invalid")
            expected_event_deltas = {
                field: workload_cgroup["memory_events_after"][field]
                - workload_cgroup["memory_events_before"][field]
                for field in ("high", "max", "oom", "oom_kill", "oom_group_kill")
            }
            if workload_cgroup["memory_events_delta"] != expected_event_deltas:
                raise ContractError("$.memory.workload_cgroup event deltas are invalid")
            swap_disabled_passed = workload_cgroup["swap_max_bytes"] == 0
            no_swap_passed = (
                workload_cgroup["swap_current_bytes_before"] == 0
                and workload_cgroup["swap_current_bytes_after"] == 0
                and workload_cgroup["swap_current_bytes_delta"] == 0
            )
            no_oom_events_passed = all(
                expected_event_deltas[field] == 0
                for field in ("oom", "oom_kill", "oom_group_kill")
            )
        cv_passed = all(
            operation["coefficient_of_variation"]
            <= configuration["maximum_coefficient_of_variation"]
            for operation in operations
        )
        drift_passed = all(
            abs(operation["drift_fraction"])
            <= configuration["maximum_absolute_drift_fraction"]
            for operation in operations
        )
        expected_gates: dict[str, bool] = {
            "minimum_duration_passed": duration_passed,
            "no_swap_activity_passed": no_swap_passed,
            "coefficient_of_variation_passed": cv_passed,
            "drift_passed": drift_passed,
        }
        if workload_cgroup is not None:
            assert swap_disabled_passed is not None
            assert no_oom_events_passed is not None
            expected_gates["workload_cgroup_swap_disabled_passed"] = (
                swap_disabled_passed
            )
            expected_gates["no_oom_events_passed"] = no_oom_events_passed
        expected_gates["overall_passed"] = (
            duration_passed
            and no_swap_passed
            and cv_passed
            and drift_passed
            and (swap_disabled_passed if swap_disabled_passed is not None else True)
            and (no_oom_events_passed if no_oom_events_passed is not None else True)
        )
        if document["gates"] != expected_gates:
            raise ContractError("$.gates does not match recomputed soak gates")
        expected_status = (
            "diagnostic"
            if configuration["requested_duration_seconds"]
            < configuration["minimum_full_soak_seconds"]
            else ("passed" if expected_gates["overall_passed"] else "failed")
        )
        if document["status"] != expected_status:
            raise ContractError("bandwidth soak status does not match its gates")
    elif kind == "allocation_matrix":
        expected_ids = {
            "runtime_device_copy",
            "host_pageable_h2d",
            "host_pinned_h2d",
            "host_pinned_d2h",
            "file_mmap_pretouched_h2d",
            "managed_unified",
            "platform_vmm_hmm",
        }
        cases = document["cases"]
        case_ids = [case["id"] for case in cases]
        if len(case_ids) != len(set(case_ids)) or set(case_ids) != expected_ids:
            raise ContractError(
                "$.cases must contain every Allocation Matrix case exactly once"
            )
        buffer_bytes = document["configuration"]["actual_buffer_bytes"]
        for case in cases:
            if case["status"] == "measured":
                if "measurement" not in case or "reason" in case:
                    raise ContractError(
                        f"measured allocation case {case['id']!r} requires measurement only"
                    )
                measurement = case["measurement"]
                if measurement["payload_bytes"] != buffer_bytes:
                    raise ContractError(
                        f"allocation case {case['id']!r} payload_bytes does not match buffer"
                    )
                if measurement["algorithmic_read_bytes"] != buffer_bytes:
                    raise ContractError(
                        f"allocation case {case['id']!r} must account one source read"
                    )
                if measurement["algorithmic_write_bytes"] != buffer_bytes:
                    raise ContractError(
                        f"allocation case {case['id']!r} must account one destination write"
                    )
            elif "reason" not in case or "measurement" in case:
                raise ContractError(
                    f"unmeasured allocation case {case['id']!r} requires reason only"
                )
    elif kind == "native_allocation_capabilities":
        _require_project_relative_path(document["source"]["path"], "$.source.path")
        build = document["build"]
        expected_compiler = "nvcc" if build["backend"] == "cuda" else "hipcc"
        if build["compiler"] != expected_compiler:
            raise ContractError(
                "native allocation capability compiler/backend mismatch"
            )
        attributes = document["runtime"]["attributes"]
        expected_attributes = {
            "managed_memory",
            "concurrent_managed_access",
            "pageable_memory_access",
            "pageable_memory_access_uses_host_page_tables",
            "direct_managed_memory_access_from_host",
            "host_native_atomic_supported",
            "memory_pools_supported",
        }
        if build["backend"] == "hip":
            expected_attributes.add("virtual_memory_management_supported")
        if set(attributes) != expected_attributes:
            raise ContractError("native allocation capability attribute set is invalid")
    elif kind == "native_stream_benchmark":
        _require_project_relative_path(document["source"]["path"], "$.source.path")
        expected_ids = {"read_reduce", "write", "copy"}
        operations = document["operations"]
        operation_ids = [operation["id"] for operation in operations]
        if len(operation_ids) != 3 or set(operation_ids) != expected_ids:
            raise ContractError(
                "$.operations must contain read_reduce, write, and copy exactly once"
            )
        buffer_bytes = document["configuration"]["actual_buffer_bytes"]
        inner_iterations = document["configuration"]["inner_iterations"]
        expected = {
            "read_reduce": (buffer_bytes * inner_iterations, 0),
            "write": (0, buffer_bytes * inner_iterations),
            "copy": (
                buffer_bytes * inner_iterations,
                buffer_bytes * inner_iterations,
            ),
        }
        for operation in operations:
            read_bytes, write_bytes = expected[operation["id"]]
            if operation["algorithmic_read_bytes"] != read_bytes:
                raise ContractError(
                    f"native operation {operation['id']!r} has invalid read bytes"
                )
            if operation["algorithmic_write_bytes"] != write_bytes:
                raise ContractError(
                    f"native operation {operation['id']!r} has invalid write bytes"
                )
            if operation["algorithmic_total_bytes"] != read_bytes + write_bytes:
                raise ContractError(
                    f"native operation {operation['id']!r} has invalid total bytes"
                )
    elif kind == "hardware_counter_calibration":
        _require_project_relative_path(
            document["native_source"]["path"], "$.native_source.path"
        )
        expected = {
            "read_reduce.read": ("read_reduce", "read", "GL2C_EA_RDREQ_DRAM_sum", 128),
            "write.write": ("write", "write", "GCEA_WDRAM_SIZE_REQ_sum", 32),
            "copy.read": ("copy", "read", "GL2C_EA_RDREQ_DRAM_sum", 128),
            "copy.write": ("copy", "write", "GCEA_WDRAM_SIZE_REQ_sum", 32),
        }
        calibrations = document["calibrations"]
        ids = [item["id"] for item in calibrations]
        if len(ids) != 4 or set(ids) != set(expected):
            raise ContractError(
                "$.calibrations must contain every required mapping exactly once"
            )
        known_bytes = document["configuration"]["known_bytes_per_dispatch"]
        threshold = document["configuration"]["maximum_relative_error"]
        sample_count = document["configuration"]["profiled_dispatches_per_operation"]
        all_passed = True
        for item in calibrations:
            operation, direction, counter, scale = expected[item["id"]]
            if (
                item["operation"] != operation
                or item["direction"] != direction
                or item["counter_name"] != counter
                or item["bytes_per_count"] != scale
            ):
                raise ContractError(
                    f"counter calibration {item['id']!r} has an invalid mapping"
                )
            arrays = (
                item["dispatch_ids"],
                item["raw_counter_values"],
                item["measured_bytes"],
                item["relative_errors"],
            )
            if any(len(values) != sample_count for values in arrays):
                raise ContractError(
                    f"counter calibration {item['id']!r} sample counts differ"
                )
            if item["dispatch_ids"] != sorted(set(item["dispatch_ids"])):
                raise ContractError(
                    f"counter calibration {item['id']!r} dispatch IDs are invalid"
                )
            measured = [value * scale for value in item["raw_counter_values"]]
            if item["measured_bytes"] != measured:
                raise ContractError(
                    f"counter calibration {item['id']!r} has invalid measured bytes"
                )
            errors = [abs(value - known_bytes) / known_bytes for value in measured]
            if any(
                not math.isclose(observed, calculated, rel_tol=1e-12, abs_tol=1e-15)
                for observed, calculated in zip(
                    item["relative_errors"], errors, strict=True
                )
            ):
                raise ContractError(
                    f"counter calibration {item['id']!r} has invalid relative errors"
                )
            median_error = statistics.median(errors)
            maximum_error = max(errors)
            if not math.isclose(
                item["median_relative_error"],
                median_error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ) or not math.isclose(
                item["maximum_relative_error"],
                maximum_error,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ContractError(
                    f"counter calibration {item['id']!r} has invalid error summary"
                )
            passed = maximum_error <= threshold
            if item["status"] != ("passed" if passed else "failed"):
                raise ContractError(
                    f"counter calibration {item['id']!r} has invalid status"
                )
            all_passed = all_passed and passed
        if document["status"] != ("passed" if all_passed else "failed"):
            raise ContractError(
                "hardware counter calibration has invalid aggregate status"
            )
    elif kind == "model_derivation":
        _require_project_relative_path(
            document["source"]["model_manifest_path"],
            "$.source.model_manifest_path",
        )
        _require_project_relative_path(
            document["source"]["tensor_inventory_path"],
            "$.source.tensor_inventory_path",
        )
        _require_project_relative_path(document["artifact_root"], "$.artifact_root")
        identities = [document["config"]]
        identities.extend(document["tokenizer_files"])
        identities.extend(document["weights"]["artifacts"])
        paths = [identity["path"] for identity in identities]
        if len(paths) != len(set(paths)):
            raise ContractError("model derivation contains duplicate artifact paths")
        for path in paths:
            _require_project_relative_path(path, f"model derivation artifact {path!r}")
        if not any(
            artifact["path"].endswith(".safetensors")
            for artifact in document["weights"]["artifacts"]
        ):
            raise ContractError("model derivation must contain a Safetensors artifact")
        total_artifact_bytes = sum(
            artifact["size_bytes"] for artifact in document["weights"]["artifacts"]
        )
        if total_artifact_bytes != document["weights"]["total_artifact_bytes"]:
            raise ContractError(
                "$.weights.total_artifact_bytes does not equal the artifact sizes"
            )
    elif kind == "weight_traffic_estimate":
        _require_project_relative_path(
            document["model_manifest_path"], "$.model_manifest_path"
        )
        _require_project_relative_path(
            document["tensor_inventory_path"], "$.tensor_inventory_path"
        )
        if document["model"]["top_k"] > document["model"]["num_experts"]:
            raise ContractError("$.model.top_k cannot exceed num_experts")
        storage = document["storage"]
        storage_total = sum(
            storage[field]
            for field in (
                "dense_weight_bytes",
                "expert_payload_bytes",
                "expert_metadata_bytes",
                "expert_padding_bytes",
            )
        )
        if storage_total != storage["total_weight_bytes"]:
            raise ContractError(
                "$.storage.total_weight_bytes does not match components"
            )
        per_token = document["per_token"]
        per_token_total = sum(
            per_token[field]
            for field in (
                "dense_weight_bytes",
                "active_expert_payload_bytes",
                "active_expert_metadata_bytes",
                "active_expert_padding_bytes",
            )
        )
        if per_token_total != per_token["total_weight_bytes"]:
            raise ContractError(
                "$.per_token.total_weight_bytes does not match components"
            )


def _require_model_frozen(document: Mapping[str, Any]) -> None:
    if document["status"] != "frozen":
        raise ContractError("model manifest is draft; frozen validation requested")
    if document["weights"]["local_verification"] != "verified":
        raise ContractError("model weights have not been locally SHA-256 verified")
    if document["dtypes"]["local_tensor_scan"] != "verified":
        raise ContractError("model tensor dtypes have not been scanned locally")
    if document["weights"].get("tensor_hashes_status") != "verified":
        raise ContractError("model tensor payload hashes have not been verified")
    if "tensor_inventory" not in document["weights"]:
        raise ContractError("frozen model manifest requires a bound TensorInventory")
    observed = document["dtypes"].get("observed_tensor_dtypes", [])
    if not observed:
        raise ContractError("frozen model manifest requires observed_tensor_dtypes")


def _require_benchmark_frozen(document: Mapping[str, Any]) -> None:
    if document["status"] != "frozen":
        raise ContractError("benchmark contract is draft; frozen validation requested")
    if document["blocking_items"]:
        raise ContractError("frozen benchmark contract cannot contain blocking_items")
    if not {"derivation", "weight_source"}.intersection(document["oracle"]):
        raise ContractError(
            "frozen benchmark contract requires a hash-bound Oracle weight source"
        )
    budget_gate = document["resource_gates"]["safe_uma_budget"]
    if budget_gate["status"] != "frozen":
        raise ContractError(
            "frozen benchmark contract requires a frozen Safe UMA Budget"
        )
    target_ids = {target["id"] for target in document["targets"]}
    budget_target_ids = {
        reference["target_id"] for reference in budget_gate["references"]
    }
    if budget_target_ids != target_ids:
        raise ContractError(
            "frozen benchmark contract requires one Safe UMA Budget reference per target"
        )
    pending = [
        gate["id"]
        for gate in document["quality_gates"]
        if gate["applicability"] == "pending"
    ]
    if pending:
        raise ContractError(
            f"frozen benchmark contract has pending quality gates: {', '.join(pending)}"
        )
    threshold_fields = {
        "max_relative_nll_ppl_increase",
        "max_normalized_task_score_drop_points",
        "min_router_top_k_set_agreement",
        "max_logit_kl",
    }
    dataset_thresholds = {
        "max_relative_nll_ppl_increase",
        "max_normalized_task_score_drop_points",
    }
    trace_thresholds = {"min_router_top_k_set_agreement", "max_logit_kl"}
    for gate in document["quality_gates"]:
        if gate["applicability"] != "required":
            continue
        present_thresholds = threshold_fields.intersection(gate)
        if not present_thresholds:
            raise ContractError(
                f"required quality gate {gate['id']!r} has no quantitative threshold"
            )
        if (
            present_thresholds.intersection(dataset_thresholds)
            and "dataset" not in gate
        ):
            raise ContractError(
                f"required quality gate {gate['id']!r} requires a pinned dataset"
            )
        if (
            present_thresholds.intersection(trace_thresholds)
            and "trace_reference" not in gate
        ):
            raise ContractError(
                f"required quality gate {gate['id']!r} requires a pinned trace reference"
            )


def _identity_projection(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _identity_projection(item)
            for key, item in value.items()
            if key not in NON_IDENTITY_KEYS
        }
    if isinstance(value, list):
        return [_identity_projection(item) for item in value]
    return value


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON for the contract identity projection."""

    projected = _identity_projection(dict(document))
    try:
        text = json.dumps(
            projected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"document cannot be canonicalized: {exc}") from exc
    return text.encode("utf-8")


def canonical_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ContractError(f"cannot locate project root from {start}")


def validate_file(
    path: str | Path, *, require_frozen: bool = False, check_references: bool = True
) -> dict[str, Any]:
    """Validate a document and, for benchmark contracts, its bound local inputs."""

    document_path = Path(path)
    document = load_document(document_path)
    validate_document(document, require_frozen=require_frozen)
    if check_references:
        if document.get("kind") == "model_manifest":
            _validate_model_references(document, document_path)
        elif document.get("kind") == "model_derivation":
            _validate_model_derivation_references(document, document_path)
        elif document.get("kind") == "weight_traffic_estimate":
            _validate_weight_traffic_references(document, document_path)
        elif document.get("kind") == "benchmark_contract":
            _validate_benchmark_references(
                document, document_path, require_frozen=require_frozen
            )
    return document


def _referenced_path(root: Path, relative_path: str, field: str) -> Path:
    _require_project_relative_path(relative_path, field)
    resolved_root = root.resolve()
    candidate = root.joinpath(*relative_path.split("/")).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{field} resolves outside the project root") from exc
    return candidate


def _validate_file_reference(
    root: Path, relative_path: str, expected_sha256: str, field: str
) -> Path:
    path = _referenced_path(root, relative_path, field)
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ContractError(
            f"{field} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return path


def _validate_model_references(
    manifest: Mapping[str, Any], manifest_path: Path
) -> None:
    tensor_reference = manifest["weights"].get("tensor_inventory")
    if tensor_reference is None:
        return

    root = find_project_root(manifest_path)
    inventory_path = _validate_file_reference(
        root,
        tensor_reference["path"],
        tensor_reference["sha256"],
        "$.weights.tensor_inventory.path",
    )
    inventory = load_document(inventory_path)
    validate_document(inventory)
    if inventory["kind"] != "tensor_inventory":
        raise ContractError(
            "$.weights.tensor_inventory must reference a TensorInventory"
        )
    if inventory["model_manifest_sha256"] != tensor_reference["source_manifest_sha256"]:
        raise ContractError(
            "$.weights.tensor_inventory.source_manifest_sha256 does not match "
            "the referenced TensorInventory"
        )

    expected_shards = {
        artifact["path"]: (artifact["size_bytes"], artifact["sha256"])
        for artifact in manifest["weights"]["artifacts"]
        if artifact["path"].endswith(".safetensors")
    }
    observed_shards = {
        shard["path"]: (shard["size_bytes"], shard["file_sha256"])
        for shard in inventory["shards"]
    }
    if observed_shards != expected_shards:
        raise ContractError(
            "referenced TensorInventory shard identities do not match ModelManifest artifacts"
        )


def _validate_model_derivation_references(
    derivation: Mapping[str, Any], derivation_path: Path
) -> None:
    root = find_project_root(derivation_path)
    source = derivation["source"]
    manifest_path = _validate_file_reference(
        root,
        source["model_manifest_path"],
        source["model_manifest_sha256"],
        "$.source.model_manifest_path",
    )
    manifest = validate_file(manifest_path, require_frozen=True)
    if manifest["kind"] != "model_manifest":
        raise ContractError(
            "$.source.model_manifest_path must reference a ModelManifest"
        )
    if manifest["model_id"] != source["model_id"]:
        raise ContractError(
            "model derivation source model_id does not match ModelManifest"
        )
    if manifest["model_revision"] != source["model_revision"]:
        raise ContractError(
            "model derivation source revision does not match ModelManifest"
        )

    inventory_path = _validate_file_reference(
        root,
        source["tensor_inventory_path"],
        source["tensor_inventory_sha256"],
        "$.source.tensor_inventory_path",
    )
    inventory = validate_file(inventory_path)
    if inventory["kind"] != "tensor_inventory":
        raise ContractError(
            "$.source.tensor_inventory_path must reference a TensorInventory"
        )
    manifest_inventory = manifest["weights"]["tensor_inventory"]
    if (
        manifest_inventory["path"] != source["tensor_inventory_path"]
        or manifest_inventory["sha256"] != source["tensor_inventory_sha256"]
    ):
        raise ContractError(
            "model derivation TensorInventory does not match the frozen ModelManifest"
        )


def _validate_weight_traffic_references(
    estimate: Mapping[str, Any], estimate_path: Path
) -> None:
    root = find_project_root(estimate_path)
    manifest_path = _referenced_path(
        root, estimate["model_manifest_path"], "$.model_manifest_path"
    )
    manifest = validate_file(manifest_path, require_frozen=True)
    observed_manifest_sha256 = canonical_sha256(manifest)
    if observed_manifest_sha256 != estimate["model_manifest_sha256"]:
        raise ContractError(
            "$.model_manifest_sha256 does not match the referenced ModelManifest"
        )
    inventory_path = _validate_file_reference(
        root,
        estimate["tensor_inventory_path"],
        estimate["tensor_inventory_sha256"],
        "$.tensor_inventory_path",
    )
    inventory = validate_file(inventory_path)
    if inventory["kind"] != "tensor_inventory":
        raise ContractError("$.tensor_inventory_path must reference a TensorInventory")
    inventory_reference = manifest["weights"]["tensor_inventory"]
    if (
        inventory_reference["path"] != estimate["tensor_inventory_path"]
        or inventory_reference["sha256"] != estimate["tensor_inventory_sha256"]
    ):
        raise ContractError(
            "WeightTrafficEstimate TensorInventory does not match the frozen ModelManifest"
        )


def _validate_benchmark_references(
    contract: Mapping[str, Any], contract_path: Path, *, require_frozen: bool
) -> None:
    root = find_project_root(contract_path)
    model_reference = contract["model"]
    model_path = _referenced_path(
        root, model_reference["manifest_path"], "$.model.manifest_path"
    )
    model_manifest = validate_file(model_path, require_frozen=require_frozen)
    observed_model_hash = canonical_sha256(model_manifest)
    if observed_model_hash != model_reference["manifest_sha256"]:
        raise ContractError(
            "$.model.manifest_sha256 does not match the referenced model manifest: "
            f"expected {model_reference['manifest_sha256']}, observed {observed_model_hash}"
        )

    max_context = model_manifest["architecture"]["max_position_embeddings"]
    for workload in contract["workloads"]:
        total_tokens = workload["prompt_tokens"] + workload["max_new_tokens"]
        if workload["status"] == "active" and total_tokens > max_context:
            raise ContractError(
                f"workload {workload['id']!r} requests {total_tokens} tokens, "
                f"exceeding model-native context {max_context}"
            )

    fixture_reference = contract["oracle"]["prompt_fixture"]
    _validate_file_reference(
        root,
        fixture_reference["path"],
        fixture_reference["sha256"],
        "$.oracle.prompt_fixture.path",
    )
    derivation_reference = contract["oracle"].get("derivation")
    if derivation_reference is not None:
        derivation_path = _validate_file_reference(
            root,
            derivation_reference["path"],
            derivation_reference["sha256"],
            "$.oracle.derivation.path",
        )
        derivation = validate_file(derivation_path)
        if derivation["kind"] != "model_derivation":
            raise ContractError("$.oracle.derivation must reference a ModelDerivation")
        if (
            derivation["source"]["model_manifest_path"]
            != model_reference["manifest_path"]
        ):
            raise ContractError(
                "Oracle derivation source does not match the benchmark ModelManifest"
            )
        dtype_names = {"bf16": "bfloat16", "f32": "float32"}
        observed_dtype = derivation["weights"]["observed_dtypes"][0].lower()
        normalized_observed = dtype_names.get(observed_dtype, observed_dtype)
        if normalized_observed != contract["oracle"]["weight_dtype"].lower():
            raise ContractError(
                "Oracle weight_dtype does not match the ModelDerivation artifacts"
            )
    weight_source = contract["oracle"].get("weight_source")
    if weight_source is not None:
        source_path = _validate_file_reference(
            root,
            weight_source["path"],
            weight_source["sha256"],
            "$.oracle.weight_source.path",
        )
        source = validate_file(source_path, require_frozen=True)
        if weight_source["kind"] == "model_manifest":
            if source["kind"] != "model_manifest":
                raise ContractError(
                    "$.oracle.weight_source must reference a ModelManifest"
                )
            if weight_source["path"] != model_reference["manifest_path"]:
                raise ContractError(
                    "Oracle ModelManifest source does not match benchmark model"
                )
            observed_dtypes = {
                dtype.lower() for dtype in source["dtypes"]["observed_tensor_dtypes"]
            }
            dtype_names = {"bf16": "bfloat16", "f32": "float32"}
            normalized_dtypes = {
                dtype_names.get(dtype, dtype) for dtype in observed_dtypes
            }
            if normalized_dtypes != {contract["oracle"]["weight_dtype"].lower()}:
                raise ContractError(
                    "Oracle weight_dtype does not match ModelManifest artifacts"
                )
        elif weight_source["kind"] == "model_derivation":
            if source["kind"] != "model_derivation":
                raise ContractError(
                    "$.oracle.weight_source must reference a ModelDerivation"
                )
            if source["source"]["model_manifest_path"] != model_reference["manifest_path"]:
                raise ContractError(
                    "Oracle derivation source does not match benchmark ModelManifest"
                )
            dtype_names = {"bf16": "bfloat16", "f32": "float32"}
            observed_dtype = source["weights"]["observed_dtypes"][0].lower()
            normalized_observed = dtype_names.get(observed_dtype, observed_dtype)
            if normalized_observed != contract["oracle"]["weight_dtype"].lower():
                raise ContractError(
                    "Oracle weight_dtype does not match the ModelDerivation artifacts"
                )

    for gate in contract["quality_gates"]:
        dataset = gate.get("dataset")
        if dataset is not None:
            _validate_file_reference(
                root,
                dataset["sample_ids_path"],
                dataset["sample_ids_sha256"],
                f"quality gate {gate['id']!r} dataset.sample_ids_path",
            )
        trace_reference = gate.get("trace_reference")
        if trace_reference is not None:
            _validate_file_reference(
                root,
                trace_reference["path"],
                trace_reference["sha256"],
                f"quality gate {gate['id']!r} trace_reference.path",
            )

    budget_gate = contract["resource_gates"]["safe_uma_budget"]
    for reference in budget_gate["references"]:
        budget_path = _validate_file_reference(
            root,
            reference["path"],
            reference["sha256"],
            f"Safe UMA Budget reference for {reference['target_id']!r}",
        )
        budget = validate_file(budget_path, require_frozen=True)
        if budget["kind"] != "safe_uma_budget":
            raise ContractError("Safe UMA Budget reference has the wrong document kind")
        if budget["target_id"] != reference["target_id"]:
            raise ContractError(
                f"Safe UMA Budget target mismatch: reference names "
                f"{reference['target_id']!r}, document names {budget['target_id']!r}"
            )
