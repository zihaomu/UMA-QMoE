from __future__ import annotations

import json
from pathlib import Path

from uma_qmoe.experiments.ledger import JsonlLedger, trial_identity
from uma_qmoe.experiments.types import (
    Encoding,
    TrialResult,
    TrialSpec,
    TrialStatus,
    Unit,
)


def _spec(layer: int = 0) -> TrialSpec:
    return TrialSpec(
        model_identity={"id": "model"},
        data_identity={"sha256": "a" * 64},
        unit=Unit(layer, "gate"),
        encoding=Encoding("q4", 128, "rtn"),
        quantizer_version="rtn-v1",
        evaluation={"id": "smoke"},
    )


def _result(spec: TrialSpec, status: TrialStatus) -> TrialResult:
    return TrialResult(
        trial_id=trial_identity(spec),
        spec=spec,
        status=status,
        started_at="2026-09-21T00:00:00Z",
        finished_at="2026-09-21T00:00:01Z",
        wall_time_seconds=1.0,
    )


def test_ledger_flushes_terminal_result_and_resumes(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "results.jsonl")
    spec = _spec()

    ledger.append(_result(spec, TrialStatus.PASSED))

    assert ledger.is_complete(spec)
    rows = ledger.records()
    assert rows[0]["evidence_level"] == "L0"
    assert rows[0]["spec"]["diagnostic_only"] is True
    assert json.loads(ledger.path.read_text(encoding="utf-8"))["status"] == "passed"


def test_interrupted_attempt_is_retained_but_retryable(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "results.jsonl")
    spec = _spec()
    ledger.append(_result(spec, TrialStatus.INTERRUPTED))

    assert not ledger.is_complete(spec)
    assert ledger.records()[0]["status"] == "interrupted"


def test_incomplete_final_append_is_ignored_and_replaced(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"kind":"exploratory_trial')
    ledger = JsonlLedger(path)
    assert ledger.records() == []

    ledger.append(_result(_spec(1), TrialStatus.REJECTED))

    assert len(ledger.records()) == 1
    assert ledger.records()[0]["status"] == "rejected"
