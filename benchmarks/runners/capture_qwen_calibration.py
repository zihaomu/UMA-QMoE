#!/usr/bin/env python3
"""Capture Qwen natural routes and emit deterministic A0/A1 calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from uma_qmoe.contracts import canonical_sha256
from uma_qmoe.fixed_models import QWEN1_5_MOE
from uma_qmoe.quantization_calibration import (
    build_expert_balanced_sample_manifest,
    build_expert_calibration_coverage,
    build_quantization_dataset_manifest,
    route_capture_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", choices=("local-halo", "spark1"), required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-selected-tokens", type=int)
    parser.add_argument("--skip-activation-statistics", action="store_true")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_artifacts_sha256(model: Path) -> str:
    candidates = {
        path
        for pattern in ("tokenizer*", "vocab*", "merges.txt", "added_tokens.json")
        for path in model.glob(pattern)
        if path.is_file()
    }
    if not candidates:
        raise RuntimeError("model directory contains no tokenizer artifacts")
    digest = hashlib.sha256()
    digest.update(b"UMA-QMoE.TokenizerArtifacts.v1\0")
    for path in sorted(candidates, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


class _ActivationAccumulator:
    """Bounded deterministic sampler plus streamed FP32 raw scalar moments."""

    _sample_per_update = 256
    _reservoir_limit = 8192

    def __init__(self, phase: int) -> None:
        self.phase = phase
        self.updates = 0
        self.count = 0
        self.sum1 = 0.0
        self.sum2 = 0.0
        self.sum3 = 0.0
        self.sum4 = 0.0
        self.max_abs = 0.0
        self.sample: list[float] = []

    def add(self, tensor: Any, torch: Any) -> None:
        values = tensor.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.sum1 += float(values.sum(dtype=torch.float32).item())
        squared = values * values
        self.sum2 += float(squared.sum(dtype=torch.float32).item())
        self.sum3 += float((squared * values).sum(dtype=torch.float32).item())
        self.sum4 += float((squared * squared).sum(dtype=torch.float32).item())
        self.max_abs = max(self.max_abs, float(values.abs().max().item()))
        stride = max(1, math.ceil(values.numel() / self._sample_per_update))
        offset = (self.phase + self.updates * 131) % stride
        sampled = values[offset::stride][: self._sample_per_update].cpu().tolist()
        self.sample.extend(float(value) for value in sampled)
        while len(self.sample) > self._reservoir_limit:
            parity = (self.phase + self.updates) & 1
            self.sample = self.sample[parity::2]
        self.updates += 1

    def summary(self) -> dict[str, Any]:
        if not self.count or not self.sample:
            raise RuntimeError("cannot summarize an empty activation accumulator")
        mean = self.sum1 / self.count
        second = self.sum2 / self.count
        third = self.sum3 / self.count
        fourth = self.sum4 / self.count
        variance = max(0.0, second - mean * mean)
        central_fourth = max(
            0.0,
            fourth - 4 * mean * third + 6 * mean * mean * second - 3 * mean**4,
        )
        kurtosis = central_fourth / (variance * variance) if variance else 0.0
        ordered = sorted(self.sample)

        def quantile(value: float) -> float:
            return ordered[max(0, math.ceil(value * len(ordered)) - 1)]

        result = {
            "element_count": self.count,
            "mean": mean,
            "second_moment": second,
            "max_abs": self.max_abs,
            "kurtosis": kurtosis,
            "p05": quantile(0.05),
            "p50": quantile(0.5),
            "p95": quantile(0.95),
        }
        if not all(math.isfinite(float(value)) for value in result.values()):
            raise RuntimeError("non-finite activation statistic")
        return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if not args.fixture.is_file():
        raise SystemExit(f"fixture does not exist: {args.fixture}")
    if args.maximum_selected_tokens is not None and args.maximum_selected_tokens <= 0:
        raise SystemExit("--maximum-selected-tokens must be positive")
    destinations = {
        "dataset": args.output_directory / "calibration-dataset-manifest.json",
        "routes": args.output_directory / "natural-route-capture.json",
        "coverage": args.output_directory / "expert-calibration-coverage.json",
        "ebss": args.output_directory / "ebss-manifest.json",
        "summary": args.output_directory / "smoke-summary.json",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite calibration evidence: {existing}")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("calibration capture requires a BF16 CUDA/HIP device")
    observed_backend = "hip" if torch.version.hip else "cuda"
    if observed_backend != args.backend:
        raise RuntimeError(
            f"calibration backend mismatch: requested {args.backend}, "
            f"observed {observed_backend}"
        )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    dataset = build_quantization_dataset_manifest(
        args.fixture,
        partition="calibration",
        source_name=args.source_name,
        source_uri=args.source_uri,
        source_revision=args.source_revision,
        license_id=args.license,
        tokenizer_id=QWEN1_5_MOE.model_id,
        tokenizer_revision=QWEN1_5_MOE.model_revision,
        tokenizer_artifacts_sha256=_tokenizer_artifacts_sha256(args.model),
        encode=lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    if (
        getattr(model.config, "model_type", None),
        getattr(model.config, "num_hidden_layers", None),
        getattr(model.config, "num_experts", None),
        getattr(model.config, "num_experts_per_tok", None),
    ) != QWEN1_5_MOE.architecture:
        raise RuntimeError("loaded model does not match the fixed Qwen architecture")

    current: dict[str, Any] = {}
    handles = []
    input_accumulators: dict[tuple[int, int], _ActivationAccumulator] = {}
    down_accumulators: dict[tuple[int, int], _ActivationAccumulator] = {}

    def make_gate_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("Qwen router did not return full probabilities")
            probabilities, weights, experts = result
            probabilities = probabilities.detach().float()
            weights = weights.detach().float()
            experts = experts.detach()
            entropy = -(
                probabilities * probabilities.clamp_min(1e-30).log()
            ).sum(dim=-1)
            top_two = probabilities.topk(2, dim=-1).values
            margin = top_two[:, 0] - top_two[:, 1]
            expert_rows = experts.cpu().tolist()
            weight_rows = weights.cpu().tolist()
            entropy_rows = entropy.cpu().tolist()
            margin_rows = margin.cpu().tolist()
            observations = current["observations"]
            if len(expert_rows) != len(observations):
                raise RuntimeError("router token count does not match dataset sample")
            for token_index, observation in enumerate(observations):
                observation["layers"].append(
                    {
                        "layer_index": layer_index,
                        "expert_indices": [
                            int(value) for value in expert_rows[token_index]
                        ],
                        "routing_weights": [
                            float(value) for value in weight_rows[token_index]
                        ],
                        "router_entropy": float(entropy_rows[token_index]),
                        "route_margin": float(margin_rows[token_index]),
                    }
                )

        return hook

    def make_expert_pre_hook(layer_index: int):
        def hook(module: Any, inputs: Any) -> None:
            if args.skip_activation_statistics:
                return
            hidden_states, selected_experts, _routing_weights = inputs
            for expert_index in selected_experts.unique().tolist():
                expert_index = int(expert_index)
                selected_tokens = (selected_experts == expert_index).any(dim=-1)
                routed_hidden = hidden_states[selected_tokens]
                key = (layer_index, expert_index)
                input_accumulator = input_accumulators.setdefault(
                    key,
                    _ActivationAccumulator(layer_index * 61 + expert_index),
                )
                down_accumulator = down_accumulators.setdefault(
                    key,
                    _ActivationAccumulator(100_003 + layer_index * 61 + expert_index),
                )
                input_accumulator.add(routed_hidden, torch)
                gate, up = functional.linear(
                    routed_hidden, module.gate_up_proj[expert_index]
                ).chunk(2, dim=-1)
                down_input = module.act_fn(gate) * up
                down_accumulator.add(down_input, torch)

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(make_gate_hook(layer_index)))
        handles.append(
            layer.mlp.experts.register_forward_pre_hook(
                make_expert_pre_hook(layer_index)
            )
        )

    observations: list[dict[str, Any]] = []
    try:
        for sample in dataset["samples"]:
            sample_observations = [
                {
                    "sample_id": sample["sample_id"],
                    "token_index": token_index,
                    "token_id": token_id,
                    "layers": [],
                }
                for token_index, token_id in enumerate(sample["prompt_token_ids"])
            ]
            current["observations"] = sample_observations
            input_ids = torch.tensor(
                [sample["prompt_token_ids"]], dtype=torch.long, device="cuda:0"
            )
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False, return_dict=True)
            if any(
                len(row["layers"]) != QWEN1_5_MOE.num_layers
                for row in sample_observations
            ):
                raise RuntimeError("route capture missed a Qwen layer")
            observations.extend(sample_observations)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    activation_statistics = None
    if not args.skip_activation_statistics:
        activation_statistics = []
        for layer_index in range(QWEN1_5_MOE.num_layers):
            experts = []
            for expert_index in range(QWEN1_5_MOE.num_experts):
                key = (layer_index, expert_index)
                if key not in input_accumulators:
                    continue
                input_summary = input_accumulators[key].summary()
                experts.append(
                    {
                        "expert_index": expert_index,
                        "projections": {
                            "gate_proj": dict(input_summary),
                            "up_proj": dict(input_summary),
                            "down_proj": down_accumulators[key].summary(),
                        },
                    }
                )
            activation_statistics.append(
                {"layer_index": layer_index, "experts": experts}
            )

    coverage = build_expert_calibration_coverage(
        dataset,
        observations,
        target_id=args.target_id,
        activation_statistics=activation_statistics,
    )
    ebss = build_expert_balanced_sample_manifest(
        dataset,
        observations,
        coverage,
        maximum_selected_tokens=args.maximum_selected_tokens,
    )
    raw_capture = {
        "schema_version": 1,
        "kind": "qwen_calibration_route_capture",
        "model_id": QWEN1_5_MOE.model_id,
        "model_revision": QWEN1_5_MOE.model_revision,
        "dataset_manifest_sha256": canonical_sha256(dataset),
        "capture_sha256": route_capture_sha256(observations),
        "runtime": {
            "backend": observed_backend,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "observations": observations,
    }
    summary = {
        "schema_version": 1,
        "kind": "qwen_calibration_smoke_summary",
        "dataset_manifest_sha256": canonical_sha256(dataset),
        "route_capture_sha256": raw_capture["capture_sha256"],
        "coverage_report_sha256": canonical_sha256(coverage),
        "ebss_manifest_sha256": canonical_sha256(ebss),
        "prompt_tokens": dataset["totals"]["prompt_tokens"],
        "minimum_prompt_tokens": dataset["token_floor"]["minimum_prompt_tokens"],
        "coverage_status": coverage["status"],
        "eligible_units": coverage["summary"]["eligible_units"],
        "empty_units": coverage["summary"]["empty_units"],
        "ebss_status": ebss["status"],
        "ebss_selected_tokens": ebss["selection"]["selected_tokens"],
        "formal_a1_exit_gate_passed": coverage["gates"]["overall_passed"]
        and ebss["gates"]["overall_passed"],
    }
    _write_json(destinations["dataset"], dataset)
    _write_json(destinations["routes"], raw_capture)
    _write_json(destinations["coverage"], coverage)
    _write_json(destinations["ebss"], ebss)
    _write_json(destinations["summary"], summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
