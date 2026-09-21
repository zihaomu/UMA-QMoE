from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from uma_qmoe.experiments.ledger import JsonlLedger
from uma_qmoe.experiments.quantizers import RTNQuantizer
from uma_qmoe.experiments.search import pareto_front, run_trial
from uma_qmoe.experiments.types import CalibrationView, Encoding, TrialSpec, Unit


class _Transaction:
    def __init__(self, value: np.ndarray) -> None:
        self.target = value
        self.snapshot = value.copy()
        self.weight = self.snapshot
        self.restoration = None

    def apply(self, candidate) -> None:
        self.target[...] = candidate.restored_weight

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.target[...] = self.snapshot
        self.restoration = {"exact": bool(np.array_equal(self.target, self.snapshot))}


class _Session:
    def __init__(self) -> None:
        self.value = np.linspace(-1, 1, 128, dtype=np.float32)

    def transaction(self, _unit):
        return _Transaction(self.value)


def _spec() -> TrialSpec:
    return TrialSpec(
        model_identity={"id": "model"},
        data_identity={"sha256": "a" * 64},
        unit=Unit(0, "gate"),
        encoding=Encoding("q4", 128, "rtn"),
        quantizer_version="rtn-v1",
        evaluation={"id": "smoke"},
    )


def test_run_trial_restores_records_and_skips_completed(tmp_path: Path) -> None:
    session = _Session()
    original = session.value.copy()
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    result = run_trial(
        spec=_spec(),
        ledger=ledger,
        session=session,
        quantizer=RTNQuantizer(),
        calibration=CalibrationView(identity={"kind": "none"}),
        evaluate=lambda: {"quality": 1.0},
        gate=lambda _offline, quality: {"quality": quality["quality"] == 1.0},
        provenance={"code_commit": "test"},
    )

    assert result is not None and result.status.value == "passed"
    assert np.array_equal(session.value, original)
    assert result.diagnostics["restoration"]["exact"] is True
    assert run_trial(
        spec=_spec(),
        ledger=ledger,
        session=session,
        quantizer=RTNQuantizer(),
        calibration=CalibrationView(identity={"kind": "none"}),
        evaluate=lambda: pytest.fail("completed trial must be skipped"),
        gate=lambda _offline, _quality: {},
    ) is None


def test_run_trial_records_interrupt_and_restores(tmp_path: Path) -> None:
    session = _Session()
    original = session.value.copy()
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")

    with pytest.raises(KeyboardInterrupt):
        run_trial(
            spec=_spec(),
            ledger=ledger,
            session=session,
            quantizer=RTNQuantizer(),
            calibration=CalibrationView(identity={"kind": "none"}),
            evaluate=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
            gate=lambda _offline, _quality: {},
            provenance={"code_commit": "test"},
        )

    assert np.array_equal(session.value, original)
    assert ledger.records()[0]["status"] == "interrupted"
    assert not ledger.is_complete(_spec())


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (RuntimeError("broken evaluator"), "error"),
        (type("OutOfMemoryError", (RuntimeError,), {})("allocation failed"), "oom"),
    ],
)
def test_run_trial_records_failures_as_terminal_results(
    tmp_path: Path, error: BaseException, expected_status: str
) -> None:
    session = _Session()
    original = session.value.copy()
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")

    result = run_trial(
        spec=_spec(),
        ledger=ledger,
        session=session,
        quantizer=RTNQuantizer(),
        calibration=CalibrationView(identity={"kind": "none"}),
        evaluate=lambda: (_ for _ in ()).throw(error),
        gate=lambda _offline, _quality: {},
        provenance={"code_commit": "test"},
    )

    assert result is not None and result.status.value == expected_status
    assert result.error["type"] == type(error).__name__
    assert np.array_equal(session.value, original)
    assert ledger.is_complete(_spec())


def test_non_json_plugin_metrics_become_an_error_record(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    result = run_trial(
        spec=_spec(),
        ledger=ledger,
        session=_Session(),
        quantizer=RTNQuantizer(),
        calibration=CalibrationView(identity={"kind": "none"}),
        evaluate=lambda: {"bad": object()},
        gate=lambda _offline, _quality: {"overall_passed": True},
        provenance={"code_commit": "test"},
    )

    assert result is not None and result.status.value == "error"
    assert result.diagnostics["result_validation_error"]["type"] == "ExperimentTypeError"
    assert ledger.records()[0]["status"] == "error"


def test_pareto_front_keeps_only_non_dominated_rows() -> None:
    rows = [
        {"quality": 1.0, "bytes": 10},
        {"quality": 0.9, "bytes": 8},
        {"quality": 0.8, "bytes": 12},
    ]
    front = pareto_front(
        rows,
        ((lambda row: row["quality"], "max"), (lambda row: row["bytes"], "min")),
    )

    assert front == rows[:2]
