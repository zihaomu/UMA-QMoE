from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runner(name: str) -> ModuleType:
    path = PROJECT_ROOT / "benchmarks" / "runners" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HOST = _load_runner("pytorch_reference_host")
ORACLE = _load_runner("capture_qwen_reference")
TRACE = _load_runner("capture_qwen_route_trace")


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
