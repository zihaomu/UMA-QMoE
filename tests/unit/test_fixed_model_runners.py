from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runner(name: str) -> ModuleType:
    path = PROJECT_ROOT / "benchmarks" / "runners" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


HOST = _load_runner("pytorch_reference_host")
ORACLE = _load_runner("capture_qwen_reference")
TRACE = _load_runner("capture_qwen_route_trace")
QWEN_HOST = _load_runner("run_compressed_qwen_baseline")
QWEN_MVP_REPORT = _load_runner("build_local_halo_qwen_mvp_report")
QWEN_COMPLETION = _load_runner("qwen_completion_quality")


def test_reference_host_defaults_preserve_olmoe_cli() -> None:
    arguments = HOST._parser().parse_args(
        [
            "--target-id",
            "halo4",
            "--model",
            "/model",
            "--prompt-fixture",
            "/fixture",
            "--source-commit",
            "a" * 40,
            "--backend",
            "hip",
            "--container-image",
            "image@sha256:" + "b" * 64,
            "--output-dir",
            "/output",
        ]
    )

    assert arguments.model_id == HOST.MODEL_ID
    assert arguments.model_revision == HOST.MODEL_REVISION


def test_reference_host_accepts_frozen_qwen_identity() -> None:
    arguments = HOST._parser().parse_args(
        [
            "--target-id",
            "spark1",
            "--model",
            "/model",
            "--model-id",
            "Qwen/Qwen1.5-MoE-A2.7B",
            "--model-revision",
            "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
            "--prompt-fixture",
            "/fixture",
            "--source-commit",
            "a" * 40,
            "--backend",
            "cuda",
            "--container-image",
            "image@sha256:" + "b" * 64,
            "--output-dir",
            "/output",
        ]
    )

    assert arguments.model_id == "Qwen/Qwen1.5-MoE-A2.7B"
    assert arguments.backend == "cuda"


def test_qwen_capture_runners_require_explicit_target_backend_and_paths() -> None:
    common = [
        "--target-id",
        "halo4",
        "--backend",
        "hip",
        "--model",
        "/model",
        "--prompt-fixture",
        "/fixture",
    ]
    trace = TRACE._parser().parse_args([*common, "--output", "/trace.json"])
    oracle = ORACLE._parser().parse_args([*common, "--output-dir", "/oracle"])

    assert trace.input_tokens == 128
    assert trace.output_tokens == 32
    assert oracle.prompt_id == "general-001"


def test_qwen_compressed_host_has_fixed_workload_defaults() -> None:
    arguments = QWEN_HOST._parser().parse_args(
        [
            "--target-id",
            "local-halo",
            "--model",
            "/model",
            "--expert-pack",
            "/pack",
            "--model-manifest-sha256",
            "a" * 64,
            "--prompt-fixture",
            "/fixture",
            "--source-commit",
            "b" * 40,
            "--backend",
            "hip",
            "--container-image",
            "image@sha256:" + "c" * 64,
            "--output-dir",
            "/output",
        ]
    )

    assert arguments.input_tokens == 128
    assert arguments.output_tokens == 32
    assert arguments.warmup_requests == 3
    assert arguments.measured_requests == 10


def test_qwen_compressed_host_binds_runtime_source_files() -> None:
    digest, files = QWEN_HOST._runtime_source_identity()

    assert len(digest) == 64
    assert len(files) == 9
    assert all(len(item["sha256"]) == 64 for item in files)
    assert files == sorted(files, key=lambda item: item["path"])


def test_local_halo_qwen_mvp_report_requires_fixed_model_identity() -> None:
    model = {
        "model": {
            "model_id": "Qwen/Qwen1.5-MoE-A2.7B",
            "model_revision": "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
            "model_manifest_sha256": QWEN_MVP_REPORT.MANIFEST_SHA256,
        }
    }

    assert QWEN_MVP_REPORT._model_matches(model)
    model["model"]["model_revision"] = "different"
    assert not QWEN_MVP_REPORT._model_matches(model)


def test_qwen_completion_metrics_and_route_agreement() -> None:
    reference = [
        {
            "id": "sample",
            "finite": True,
            "per_token_nll": [1.0, 2.0],
            "target_token_ids": [4, 5],
            "completion_top1_token_ids": [4, 7],
            "routes": {str(layer): [[0, 1, 2, 3]] for layer in range(24)},
        }
    ]
    candidate = copy.deepcopy(reference)
    candidate[0]["routes"]["23"] = [[0, 1, 2, 4]]

    metrics = QWEN_COMPLETION._aggregate(reference)
    agreement, per_layer = QWEN_COMPLETION._route_agreement(
        candidate, reference
    )

    assert metrics["nll"] == 1.5
    assert metrics["completion_token_accuracy"] == 0.5
    assert agreement == 23 / 24
    assert per_layer == [1.0] * 23 + [0.0]


def test_qwen_oracle_runs_one_stacked_expert() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    gate_up = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    experts = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=functional.silu,
    )
    expert_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    observed = ORACLE._single_expert_forward(
        experts, 1, expert_input, functional
    )
    gate, up = functional.linear(expert_input, gate_up[1]).chunk(2, dim=-1)
    expected = functional.linear(functional.silu(gate) * up, down[1])

    assert torch.equal(observed, expected)


def test_qwen_runners_accept_structured_router_output() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    module = SimpleNamespace(weight=torch.eye(3, dtype=torch.float32))
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    probabilities = functional.softmax(hidden, dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(probabilities, 2, dim=-1)
    result = (probabilities, weights, experts)

    logits, oracle_weights, oracle_experts = ORACLE._router_observation(
        module, (hidden,), result, functional, 2, False
    )
    trace_weights, trace_experts = TRACE._router_topk(
        result, functional, torch, 2, False
    )

    assert torch.equal(logits, hidden)
    assert torch.equal(oracle_weights, weights)
    assert torch.equal(oracle_experts, experts)
    assert torch.equal(trace_weights, weights)
    assert torch.equal(trace_experts, experts)
