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


def test_experiment_script_cannot_import_sibling(tmp_path: Path) -> None:
    experiments = tmp_path / "benchmarks" / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "first.py").write_text(
        "from second import private_helper\n", encoding="utf-8"
    )
    (experiments / "second.py").write_text(
        "def private_helper(): pass\n", encoding="utf-8"
    )

    violations = check_dependency_boundaries(tmp_path)

    assert len(violations) == 1
    assert "forbidden sibling import second" in violations[0]


def test_torch_free_experiment_control_plane_is_enforced(tmp_path: Path) -> None:
    package = tmp_path / "src" / "uma_qmoe" / "experiments"
    package.mkdir(parents=True)
    (package / "ledger.py").write_text("import torch\n", encoding="utf-8")

    violations = check_dependency_boundaries(tmp_path)

    assert len(violations) == 1
    assert "control-plane module imports torch" in violations[0]


def test_quantizer_cannot_import_formal_pack_writer(tmp_path: Path) -> None:
    package = tmp_path / "src" / "uma_qmoe" / "experiments"
    package.mkdir(parents=True)
    (package / "quantizers.py").write_text(
        "from uma_qmoe import target_pack\n", encoding="utf-8"
    )

    violations = check_dependency_boundaries(tmp_path)

    assert len(violations) == 1
    assert "quantizer crosses framework boundary" in violations[0]
