#!/usr/bin/env python3
"""Validate CPU-reference and target dispatch for uma_qmoe::moe_forward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from uma_qmoe.custom_op import (
    register_expert_pack,
    register_moe_forward,
    unregister_expert_pack,
)
from uma_qmoe.expert_pack import ExpertPackReader


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-id", choices=("halo3", "local-halo", "spark1"), required=True
    )
    parser.add_argument("--expected-platform", choices=("hip_gfx1151", "cuda_sm121"), required=True)
    parser.add_argument("--expert-pack", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: object) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _observed_platform(torch: object) -> tuple[str, str]:
    properties = torch.cuda.get_device_properties(0)
    if torch.version.hip:
        architecture = str(getattr(properties, "gcnArchName", ""))
        return f"hip_{architecture.split(':', 1)[0]}", str(properties.name)
    major, minor = torch.cuda.get_device_capability(0)
    return f"cuda_sm{major}{minor}", str(properties.name)


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite custom operator evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("custom operator validation requires a CUDA/HIP target")
    platform, device_name = _observed_platform(torch)
    register_moe_forward()
    registered = hasattr(torch.ops.uma_qmoe, "moe_forward")
    reader = ExpertPackReader(
        args.expert_pack,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_model_manifest_sha256=args.model_manifest_sha256,
    )
    handle = register_expert_pack(reader)
    try:
        hidden_cpu = torch.linspace(-0.5, 0.5, 2048, dtype=torch.float32).reshape(1, 2048).to(torch.bfloat16)
        # Repeating one valid expert keeps this cross-device contract test small;
        # full-model evidence covers real Top-8 expert diversity on every layer.
        experts_cpu = torch.zeros((1, 8), dtype=torch.int64)
        weights_cpu = torch.full((1, 8), 0.125, dtype=torch.bfloat16)
        with torch.inference_mode():
            cpu_output = torch.ops.uma_qmoe.moe_forward(
                hidden_cpu, experts_cpu, weights_cpu, 0, handle, False
            )
            target_output = torch.ops.uma_qmoe.moe_forward(
                hidden_cpu.to("cuda:0"),
                experts_cpu.to("cuda:0"),
                weights_cpu.to("cuda:0"),
                0,
                handle,
                False,
            )
            torch.cuda.synchronize()
        cpu_float = cpu_output.detach().float().cpu()
        target_float = target_output.detach().float().cpu()
        max_abs = float((cpu_float - target_float).abs().max().item())
        cosine = float(functional.cosine_similarity(cpu_float, target_float).item())

        rejected = False
        error = ""
        try:
            torch.ops.uma_qmoe.moe_forward(
                hidden_cpu.to("cuda:0"),
                experts_cpu.to("cuda:0"),
                weights_cpu.to("cuda:0"),
                0,
                handle,
                True,
            )
        except RuntimeError as exc:
            rejected = True
            error = str(exc)

        acceptance = {"max_abs_error": 1.0, "minimum_cosine_similarity": 0.99}
        reference = {
            "layer_index": 0,
            "input_shape": [1, 2048],
            "routes_shape": [1, 8],
            "unique_experts": 1,
            "cpu_finite": bool(torch.isfinite(cpu_float).all().item()),
            "target_finite": bool(torch.isfinite(target_float).all().item()),
            "cpu_output_sha256": _tensor_sha256(cpu_float),
            "target_output_sha256": _tensor_sha256(target_float),
            "max_abs_error": max_abs,
            "cosine_similarity": cosine,
            "acceptance": acceptance,
        }
        performance = {
            "requested": True,
            "backend_registered": False,
            "rejected": rejected,
            "error": error,
        }
        gates = {
            "operator_registered": registered,
            "architecture_dispatch": platform == args.expected_platform,
            "cpu_reference_forward": reference["cpu_finite"],
            "target_reference_forward": reference["target_finite"],
            "cpu_target_agreement": (
                max_abs <= acceptance["max_abs_error"]
                and cosine >= acceptance["minimum_cosine_similarity"]
            ),
            "performance_mode_fail_closed": (
                rejected and "silent reference fallback is forbidden" in error
            ),
        }
        gates["overall_passed"] = all(gates.values())
        document = {
            "schema_version": 1,
            "kind": "custom_operator_evidence",
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "target_id": args.target_id,
            "status": "passed" if gates["overall_passed"] else "failed",
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "expert_pack_sha256": _sha256_file(args.expert_pack),
            },
            "operator": {
                "qualified_name": "uma_qmoe::moe_forward",
                "registration": "torch.library",
                "registered": registered,
                "platform": platform,
                "device_name": device_name,
            },
            "reference": reference,
            "performance_mode": performance,
            "gates": gates,
        }
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if not gates["overall_passed"]:
            raise RuntimeError("custom operator validation gate failed")
    finally:
        unregister_expert_pack(handle)
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
