from __future__ import annotations

from pathlib import Path

from uma_qmoe.dependency_boundary import check_dependency_boundaries


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_core_has_no_external_runtime_imports() -> None:
    assert check_dependency_boundaries(PROJECT_ROOT) == []


def test_dependency_boundary_rejects_python_import(tmp_path: Path) -> None:
    source = tmp_path / "src" / "uma_qmoe"
    source.mkdir(parents=True)
    (source / "bad.py").write_text("from vllm import LLM\n", encoding="utf-8")

    violations = check_dependency_boundaries(tmp_path)

    assert len(violations) == 1
    assert "forbidden import vllm" in violations[0]


def test_external_boundary_rejects_core_runtime_import(tmp_path: Path) -> None:
    source = tmp_path / "benchmarks" / "external" / "example"
    source.mkdir(parents=True)
    (source / "bad.py").write_text(
        "from uma_qmoe.telemetry import TelemetryProbe\n", encoding="utf-8"
    )

    violations = check_dependency_boundaries(tmp_path)

    assert len(violations) == 1
    assert "forbidden import uma_qmoe.telemetry" in violations[0]


def test_vllm_runner_lives_only_at_external_boundary() -> None:
    old_path = PROJECT_ROOT / "benchmarks" / "runners" / "vllm_public_baseline.py"
    external_path = (
        PROJECT_ROOT
        / "benchmarks"
        / "external"
        / "vllm"
        / "vllm_public_baseline.py"
    )

    assert not old_path.exists()
    assert external_path.is_file()
