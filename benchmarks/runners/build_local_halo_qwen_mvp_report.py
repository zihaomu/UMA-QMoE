#!/usr/bin/env python3
"""Assemble the auditable local-halo Qwen MVP decision report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from uma_qmoe.fixed_models import QWEN1_5_MOE


MANIFEST_SHA256 = "87667d3fb147eb692748ce33d0e570919281b264a3cc950b2e2e9df4dc12b13f"


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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence must be a JSON object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--oracle-metadata", type=Path, required=True)
    parser.add_argument("--route-trace", type=Path, required=True)
    parser.add_argument("--single-prompt-quality", type=Path, required=True)
    parser.add_argument("--completion-reference", type=Path, required=True)
    parser.add_argument("--completion-quality", type=Path, required=True)
    parser.add_argument("--q4-quality", type=Path, required=True)
    parser.add_argument("--mixed-run-dir", type=Path, required=True)
    parser.add_argument("--q4-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _evidence_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _model_matches(value: dict[str, Any]) -> bool:
    model = value.get("model", value)
    return (
        model.get("model_id") == QWEN1_5_MOE.model_id
        and model.get("model_revision") == QWEN1_5_MOE.model_revision
        and model.get("model_manifest_sha256", MANIFEST_SHA256) == MANIFEST_SHA256
    )


def _build(args: argparse.Namespace) -> dict[str, Any]:
    source = _json(args.source_verification)
    oracle = _json(args.oracle_metadata)
    route_trace = _json(args.route_trace)
    single_quality = _json(args.single_prompt_quality)
    completion_reference = _json(args.completion_reference)
    completion_quality = _json(args.completion_quality)
    q4_quality = _json(args.q4_quality)
    mixed_result_path = args.mixed_run_dir / "result.json"
    mixed_metadata_path = args.mixed_run_dir / "runner-metadata.json"
    q4_result_path = args.q4_run_dir / "result.json"
    q4_metadata_path = args.q4_run_dir / "runner-metadata.json"
    mixed_result = _json(mixed_result_path)
    mixed_metadata = _json(mixed_metadata_path)
    q4_result = _json(q4_result_path)
    q4_metadata = _json(q4_metadata_path)

    for name, value in (
        ("route trace", route_trace),
        ("single-prompt quality", single_quality),
        ("completion reference", completion_reference),
        ("completion quality", completion_quality),
        ("Q4 quality", q4_quality),
        ("mixed run", mixed_metadata),
        ("Q4 run", q4_metadata),
    ):
        if not _model_matches(value):
            raise RuntimeError(f"{name} uses a different fixed Qwen identity")
    if (
        source.get("kind") != "model_acquisition"
        or source.get("model_manifest_sha256") != MANIFEST_SHA256
        or source.get("file_count") != len(source.get("files", []))
        or source.get("file_count", 0) < 1
    ):
        raise RuntimeError("source verification identity is incomplete")
    if (
        oracle.get("model_id") != QWEN1_5_MOE.model_id
        or oracle.get("model_revision") != QWEN1_5_MOE.model_revision
    ):
        raise RuntimeError("Oracle metadata uses a different fixed Qwen identity")
    for stream in oracle.get("streams", {}).values():
        path = args.oracle_metadata.parent / stream["path"]
        if path.stat().st_size != stream["size_bytes"] or _sha256(path) != stream["sha256"]:
            raise RuntimeError(f"Oracle stream identity changed: {path}")

    mixed_tps = mixed_result["summary"]["aggregate_tokens_per_second"]
    q4_tps = q4_result["summary"]["aggregate_tokens_per_second"]
    speed_gain = mixed_tps / q4_tps - 1.0
    reference_peak = completion_reference["memory"]["torch_peak_reserved_bytes"]
    candidate_peak = completion_quality["memory"]["torch_peak_reserved_bytes"]
    memory_reduction = 1.0 - candidate_peak / reference_peak
    same_runtime_source = (
        mixed_metadata["runtime"].get("source_state_sha256")
        == q4_metadata["runtime"].get("source_state_sha256")
        and mixed_metadata["runtime"].get("source_state_sha256") is not None
    )
    q4_failed_expected_gates = (
        q4_quality.get("status") == "failed"
        and not q4_quality.get("gates", {}).get("logit_kl_at_most_0_01", True)
        and not q4_quality.get("gates", {}).get(
            "router_top4_set_agreement_at_least_0_99", True
        )
    )
    q4_performance_gates = q4_metadata.get("gates", {})
    q4_comparison_valid = all(
        q4_performance_gates.get(name) is True
        for name in (
            "requests_succeeded",
            "deterministic_outputs",
            "statistics_contract_satisfied",
            "cache_stable_during_measurement",
            "no_dequantized_weight_cache",
            "no_measurement_major_page_faults",
            "end_to_end_cv_at_most_0_05",
        )
    )
    gates = {
        "source_verified": True,
        "oracle_streams_verified": len(oracle.get("streams", {})) == 3,
        "route_trace_frozen": (
            route_trace.get("status") == "frozen"
            and len(route_trace.get("layers", [])) == QWEN1_5_MOE.num_layers
            and len(route_trace.get("events", [])) == 32
        ),
        "native_single_prompt_quality_passed": (
            single_quality.get("status") == "passed"
            and single_quality.get("execution", {}).get("performance_mode") is True
            and single_quality.get("execution", {}).get("native_platform")
            == "hip_gfx1151"
            and single_quality.get("gates", {}).get("overall_passed") is True
        ),
        "completion_quality_passed": (
            completion_quality.get("status") == "passed"
            and completion_quality.get("gates", {}).get("overall_passed") is True
        ),
        "all_q4_negative_control_failed_quality": q4_failed_expected_gates,
        "formal_mixed_run_passed": mixed_metadata.get("gates", {}).get(
            "overall_passed"
        )
        is True,
        # Q4 is a deliberately failing quality negative control, not the
        # release candidate.  Its timing comparison requires stable execution
        # and no timed major faults, but does not inherit the candidate's
        # no-swap release gate.
        "formal_q4_comparison_valid": q4_comparison_valid,
        "same_runtime_source_for_performance_comparison": same_runtime_source,
        "mixed_faster_than_all_q4": speed_gain > 0.0,
        "peak_memory_reduction_at_least_15_percent": memory_reduction >= 0.15,
        "single_pack_mapping": (
            completion_quality.get("loader", {}).get("expert_pack_mapping_count") == 1
        ),
        "no_global_dequantized_expert_cache": (
            completion_quality.get("backend_cache", {}).get(
                "dequantized_weight_bytes"
            )
            == 0
            and mixed_metadata.get("backend_cache", {})
            .get("after_measurement", {})
            .get("dequantized_weight_bytes")
            == 0
        ),
    }
    gates["overall_passed"] = all(gates.values())
    evidence_paths = (
        args.source_verification,
        args.oracle_metadata,
        args.route_trace,
        args.single_prompt_quality,
        args.completion_reference,
        args.completion_quality,
        args.q4_quality,
        mixed_result_path,
        mixed_metadata_path,
        q4_result_path,
        q4_metadata_path,
    )
    return {
        "schema_version": 1,
        "kind": "local_halo_qwen_mvp_report",
        "captured_at": _utc_now(),
        "target_id": "local-halo",
        "status": "passed" if gates["overall_passed"] else "failed",
        "scope": (
            "single-user batch-1 Qwen1.5-MoE-A2.7B base-model inference MVP "
            "on AMD Strix Halo gfx1151; not Spark release evidence"
        ),
        "model": {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "model_manifest_sha256": MANIFEST_SHA256,
        },
        "candidate": {
            "policy_id": single_quality["expert_pack"]["target_policy_id"],
            "expert_pack_sha256": single_quality["expert_pack"]["sha256"],
            "expert_pack_size_bytes": single_quality["expert_pack"]["size_bytes"],
            "encoding": "BF16 layers 0-15 plus Q8 group-128 layers 16-23",
        },
        "quality": {
            "single_prompt_logit_kl": single_quality["quality"]["logit_kl"],
            "single_prompt_router_top4_exact_set_agreement": single_quality[
                "quality"
            ]["router_top4_exact_set_agreement"],
            "completion_sample_count": completion_quality["aggregate"][
                "sample_count"
            ],
            "completion_target_token_count": completion_quality["aggregate"][
                "target_token_count"
            ],
            "relative_perplexity_change": completion_quality["comparison"][
                "relative_perplexity_change"
            ],
            "completion_score_drop_points": completion_quality["comparison"][
                "completion_score_drop_points"
            ],
            "router_top4_exact_set_agreement": completion_quality["comparison"][
                "router_top4_exact_set_agreement"
            ],
        },
        "performance": {
            "workload": mixed_metadata["workload"],
            "mixed_tokens_per_second": mixed_tps,
            "all_q4_tokens_per_second": q4_tps,
            "relative_speed_gain_vs_all_q4": speed_gain,
            "mixed_ttft_p50_ms": mixed_result["summary"]["ttft_ms"]["p50"],
            "mixed_tpot_p50_ms": mixed_result["summary"]["tpot_ms"]["p50"],
            "mixed_request_cv": mixed_result["summary"]["request_duration_ms"][
                "coefficient_of_variation"
            ],
            "runtime_source_state_sha256": mixed_metadata["runtime"][
                "source_state_sha256"
            ],
        },
        "memory": {
            "bf16_reference_peak_reserved_bytes": reference_peak,
            "mixed_peak_reserved_bytes": candidate_peak,
            "relative_peak_memory_reduction": memory_reduction,
            "formal_128x32_peak_reserved_bytes": mixed_metadata[
                "torch_peak_reserved_bytes"
            ],
            "measurement_swap_bytes": mixed_metadata[
                "measurement_swap_current_after_bytes"
            ],
            "measurement_major_page_faults": mixed_metadata[
                "measurement_major_page_faults_delta"
            ],
        },
        "negative_control": {
            "all_q4_status": q4_quality["status"],
            "all_q4_logit_kl": q4_quality["quality"]["logit_kl"],
            "all_q4_router_top4_exact_set_agreement": q4_quality["quality"][
                "router_top4_exact_set_agreement"
            ],
        },
        "evidence": [_evidence_record(path) for path in evidence_paths],
        "gates": gates,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite local-halo Qwen MVP report")
    report = _build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
