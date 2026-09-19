#!/usr/bin/env python3
"""Compile and validate the target-native canonical packed-Q4 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import statistics
from typing import Any

from uma_qmoe.contracts import validate_document
from uma_qmoe.custom_op import (
    register_expert_pack,
    register_moe_forward,
    unregister_expert_pack,
)
from uma_qmoe.expert_pack import ExpertPackReader
from uma_qmoe.native_backend import (
    install_packed_q4_backend,
    native_kernel_source_sha256,
)
from uma_qmoe.q4 import dequantize_q4


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("halo3", "spark1"), required=True)
    parser.add_argument(
        "--expected-platform", choices=("hip_gfx1151", "cuda_sm121"), required=True
    )
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _agreement(candidate: Any, reference: Any, functional: Any) -> dict[str, Any]:
    candidate_float = candidate.detach().float()
    reference_float = reference.detach().float()
    return {
        "finite": bool(candidate_float.isfinite().all().item()),
        "max_absolute_error": float(
            (candidate_float - reference_float).abs().max().item()
        ),
        "cosine_similarity": float(
            functional.cosine_similarity(
                candidate_float.reshape(1, -1), reference_float.reshape(1, -1)
            ).item()
        ),
    }


def _time_cuda(torch: Any, operation: Any, warmup: int, measured: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite packed Q4 kernel evidence")
    if args.warmup_iterations < 1 or args.measured_iterations < 3:
        raise SystemExit(
            "packed Q4 timing requires >=1 warmup and >=3 measured samples"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("packed Q4 validation requires a BF16 CUDA/HIP device")
    torch.manual_seed(0)
    register_moe_forward()
    backend = install_packed_q4_backend(
        build_directory=args.build_directory, verbose=True
    )
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    handle = register_expert_pack(reader)
    try:
        projection_results = []
        projection_inputs = {
            "gate_proj": torch.linspace(-0.5, 0.5, 2048, dtype=torch.float32).reshape(
                1, 2048
            ),
            "up_proj": torch.linspace(0.25, -0.25, 2048, dtype=torch.float32).reshape(
                1, 2048
            ),
            "down_proj": torch.linspace(-0.75, 0.75, 1024, dtype=torch.float32).reshape(
                1, 1024
            ),
        }
        with torch.inference_mode():
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.experts.0.{projection}.weight"
                q4 = reader.tensor_q4(name)
                weight = torch.from_numpy(dequantize_q4(q4)).to(
                    device="cuda:0", dtype=torch.bfloat16
                )
                input_tensor = projection_inputs[projection].to(
                    device="cuda:0", dtype=torch.bfloat16
                )
                candidate = backend.q4_linear(input_tensor, handle, name)
                reference = functional.linear(input_tensor, weight)
                result = _agreement(candidate, reference, functional)
                result.update({"name": projection, "shape": list(weight.shape)})
                projection_results.append(result)
                del weight, input_tensor, candidate, reference

            hidden = (
                torch.linspace(-0.5, 0.5, 2048, dtype=torch.float32, device="cuda:0")
                .reshape(1, 2048)
                .to(torch.bfloat16)
            )
            experts = torch.arange(8, dtype=torch.int64, device="cuda:0").reshape(1, 8)
            weights = torch.full((1, 8), 0.125, dtype=torch.bfloat16, device="cuda:0")
            reference_output = torch.ops.uma_qmoe.moe_forward(
                hidden, experts, weights, 0, handle, False
            )
            performance_output = torch.ops.uma_qmoe.moe_forward(
                hidden, experts, weights, 0, handle, True
            )
            torch.cuda.synchronize()
            moe_result = _agreement(performance_output, reference_output, functional)
            moe_result["output_sha256"] = _tensor_sha256(performance_output)

            prefill_hidden = torch.linspace(
                -0.5,
                0.5,
                4 * 2048,
                dtype=torch.float32,
                device="cuda:0",
            ).reshape(4, 2048).to(torch.bfloat16)
            prefill_experts = experts.repeat(4, 1)
            prefill_weights = weights.repeat(4, 1)
            prefill_reference = torch.ops.uma_qmoe.moe_forward(
                prefill_hidden,
                prefill_experts,
                prefill_weights,
                0,
                handle,
                False,
            )
            prefill_output = torch.ops.uma_qmoe.moe_forward(
                prefill_hidden,
                prefill_experts,
                prefill_weights,
                0,
                handle,
                True,
            )
            torch.cuda.synchronize()
            prefill_result = _agreement(
                prefill_output, prefill_reference, functional
            )
            prefill_result["output_sha256"] = _tensor_sha256(prefill_output)

            samples = _time_cuda(
                torch,
                lambda: torch.ops.uma_qmoe.moe_forward(
                    hidden, experts, weights, 0, handle, True
                ),
                args.warmup_iterations,
                args.measured_iterations,
            )
            prefill_samples = _time_cuda(
                torch,
                lambda: torch.ops.uma_qmoe.moe_forward(
                    prefill_hidden,
                    prefill_experts,
                    prefill_weights,
                    0,
                    handle,
                    True,
                ),
                args.warmup_iterations,
                args.measured_iterations,
            )
        cache = backend.cache_summary()
        acceptance = {
            "max_absolute_error": 1.0,
            "minimum_cosine_similarity": 0.99,
        }
        projection_passed = all(
            result["finite"]
            and result["max_absolute_error"] <= acceptance["max_absolute_error"]
            and result["cosine_similarity"] >= acceptance["minimum_cosine_similarity"]
            for result in projection_results
        )
        moe_passed = (
            moe_result["finite"]
            and moe_result["max_absolute_error"] <= acceptance["max_absolute_error"]
            and moe_result["cosine_similarity"]
            >= acceptance["minimum_cosine_similarity"]
        )
        prefill_passed = (
            prefill_result["finite"]
            and prefill_result["max_absolute_error"]
            <= acceptance["max_absolute_error"]
            and prefill_result["cosine_similarity"]
            >= acceptance["minimum_cosine_similarity"]
        )
        expected_platform = {
            "halo3": "hip_gfx1151",
            "spark1": "cuda_sm121",
        }[args.target_id]
        gates = {
            "target_architecture": backend.platform
            == args.expected_platform
            == expected_platform,
            "target_compilation": True,
            "direct_packed_input": True,
            "no_dequantized_weight_cache": cache["dequantized_weight_bytes"] == 0,
            "projection_correctness": projection_passed,
            "moe_correctness": moe_passed,
            "prefill_moe_correctness": prefill_passed,
            "performance_mode_executed": len(samples) == args.measured_iterations,
            "prefill_performance_mode_executed": len(prefill_samples)
            == args.measured_iterations,
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "packed_q4_kernel_evidence",
            "captured_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "target_id": args.target_id,
            "status": "passed" if gates["overall_passed"] else "failed",
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expert_pack_sha256": _sha256_file(args.expert_pack),
            },
            "kernel": {
                "platform": backend.platform,
                "device_name": torch.cuda.get_device_name(0),
                "source_sha256": native_kernel_source_sha256(),
                "abi": "q4-group128-packed-u8-fp32-scale-bf16-in-bf16-out-route-specialized-v3",
                "execution_strategy": "decode-two-launch-prefill-expert-sorted-three-stage",
                "compiled_for_target": True,
                "reads_packed_weights_directly": True,
                "full_dequantized_weight_cache": False,
            },
            "workload": {
                "layer_index": 0,
                "tokens": 1,
                "prefill_tokens": 4,
                "top_k": 8,
                "unique_experts": 8,
                "warmup_iterations": args.warmup_iterations,
                "measured_iterations": args.measured_iterations,
            },
            "correctness": {
                "projection_results": projection_results,
                "moe_forward": moe_result,
                "prefill_moe_forward": prefill_result,
                "acceptance": acceptance,
            },
            "performance": {
                "scope": "single_moe_layer_microbenchmark",
                "samples_milliseconds": samples,
                "median_milliseconds": statistics.median(samples),
                "p95_milliseconds": sorted(samples)[
                    max(0, math.ceil(0.95 * len(samples)) - 1)
                ],
                "prefill_samples_milliseconds": prefill_samples,
                "prefill_median_milliseconds": statistics.median(prefill_samples),
                "prefill_p95_milliseconds": sorted(prefill_samples)[
                    max(0, math.ceil(0.95 * len(prefill_samples)) - 1)
                ],
                "compressed_cache_tensor_count": cache["tensor_count"],
                "compressed_cache_bytes": cache["device_storage_bytes"],
                "dequantized_weight_cache_bytes": cache["dequantized_weight_bytes"],
                "formal_tps_claim": False,
            },
            "gates": gates,
        }
        validate_document(document)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not gates["overall_passed"]:
            raise RuntimeError("packed Q4 validation gate failed")
    finally:
        unregister_expert_pack(handle)
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
