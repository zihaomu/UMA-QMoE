"""Trial lifecycle, constraints and small search helpers for L0 experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any

from .ledger import JsonlLedger, trial_identity
from .quantizers import Quantizer
from .types import CalibrationView, TrialResult, TrialSpec, TrialStatus


Metrics = Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _failure(error: BaseException, limit: int = 4096) -> dict[str, str]:
    diagnostic = "".join(traceback.format_exception(error))
    if len(diagnostic) > limit:
        diagnostic = diagnostic[: limit - 15] + "\n...[truncated]"
    return {"type": type(error).__name__, "diagnostic": diagnostic}


def _is_oom(error: BaseException) -> bool:
    return type(error).__name__.endswith("OutOfMemoryError") or "out of memory" in str(
        error
    ).lower()


def collect_provenance(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {
        "code_commit": commit,
        "code_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def run_trial(
    *,
    spec: TrialSpec,
    ledger: JsonlLedger,
    session: Any,
    quantizer: Quantizer,
    calibration: CalibrationView,
    evaluate: Callable[[], Metrics],
    gate: Callable[[Metrics, Metrics], Metrics],
    offline_gate: Callable[[Metrics], bool] | None = None,
    memory_reset: Callable[[], None] | None = None,
    memory_read: Callable[[], Metrics] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> TrialResult | None:
    """Run one trial, always restore it, then durably append one L0 result.

    A completed identity is skipped.  ``interrupted`` attempts are appended and the
    ``KeyboardInterrupt`` is re-raised so a subsequent process retries that unit.
    """

    identity = trial_identity(spec)
    if ledger.is_complete(identity):
        return None
    started_at = _utc_now()
    started = time.perf_counter()
    offline: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    transaction = None
    status = TrialStatus.ERROR
    failure: dict[str, str] | None = None
    interrupted: KeyboardInterrupt | None = None
    fatal: BaseException | None = None
    try:
        if memory_reset is not None:
            memory_reset()
        with session.transaction(spec.unit) as transaction:
            candidate = quantizer.quantize(
                transaction.weight, calibration, spec.encoding
            )
            offline = dict(candidate.offline_metrics)
            diagnostics["quantizer"] = dict(candidate.diagnostics)
            if offline_gate is not None and not offline_gate(offline):
                gates = {"offline_passed": False, "overall_passed": False}
                status = TrialStatus.REJECTED
            else:
                transaction.apply(candidate)
                quality = dict(evaluate())
                gates = dict(gate(offline, quality))
                if "overall_passed" not in gates:
                    gates["overall_passed"] = all(bool(value) for value in gates.values())
                status = (
                    TrialStatus.PASSED
                    if gates["overall_passed"]
                    else TrialStatus.REJECTED
                )
        if transaction is not None and transaction.restoration is not None:
            diagnostics["restoration"] = transaction.restoration
    except KeyboardInterrupt as error:
        status = TrialStatus.INTERRUPTED
        failure = _failure(error)
        interrupted = error
        if transaction is not None and transaction.restoration is not None:
            diagnostics["restoration"] = transaction.restoration
    except BaseException as error:
        status = TrialStatus.OOM if _is_oom(error) else TrialStatus.ERROR
        failure = _failure(error)
        if bool(getattr(error, "fatal", False)):
            fatal = error
        if transaction is not None and transaction.restoration is not None:
            diagnostics["restoration"] = transaction.restoration
    finally:
        if memory_read is not None:
            try:
                memory = dict(memory_read())
            except BaseException as error:
                diagnostics["memory_read_error"] = _failure(error, limit=1024)
    finished_at = _utc_now()
    wall_time = time.perf_counter() - started
    try:
        result = TrialResult(
            trial_id=identity,
            spec=spec,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time,
            offline_metrics=offline,
            quality_metrics=quality,
            gate=gates,
            memory=memory,
            diagnostics=diagnostics,
            error=failure,
            provenance=dict(provenance or collect_provenance()),
        )
    except (TypeError, ValueError) as error:
        # A plugin returning a tensor/NaN as a metric is itself an experimental
        # failure.  Keep the attempt resumable instead of losing the ledger row.
        result = TrialResult(
            trial_id=identity,
            spec=spec,
            status=status if interrupted is not None else TrialStatus.ERROR,
            started_at=started_at,
            finished_at=finished_at,
            wall_time_seconds=wall_time,
            diagnostics={"result_validation_error": _failure(error, limit=1024)},
            error=failure or _failure(error),
            provenance={},
        )
    ledger.append(result)
    if interrupted is not None:
        raise interrupted
    if fatal is not None:
        raise fatal
    return result


def constraint_filter(
    rows: Sequence[Any], predicates: Sequence[Callable[[Any], bool]]
) -> list[Any]:
    return [row for row in rows if all(predicate(row) for predicate in predicates)]


def pareto_front(
    rows: Sequence[Any], objectives: Sequence[tuple[Callable[[Any], float], str]]
) -> list[Any]:
    """Return non-dominated rows for explicit ``min``/``max`` objectives."""

    if any(direction not in {"min", "max"} for _key, direction in objectives):
        raise ValueError("Pareto objective direction must be 'min' or 'max'")

    def dominates(left: Any, right: Any) -> bool:
        comparisons = []
        for key, direction in objectives:
            left_value, right_value = key(left), key(right)
            comparisons.append(
                left_value <= right_value if direction == "min" else left_value >= right_value
            )
        strict = any(
            key(left) != key(right) for key, _direction in objectives
        )
        return all(comparisons) and strict

    return [
        row
        for index, row in enumerate(rows)
        if not any(
            dominates(other, row)
            for other_index, other in enumerate(rows)
            if other_index != index
        )
    ]


def beam_select(
    rows: Sequence[Any], *, score: Callable[[Any], float], width: int
) -> list[Any]:
    if width <= 0:
        raise ValueError("beam width must be positive")
    return sorted(rows, key=score)[:width]


__all__ = [
    "beam_select",
    "collect_provenance",
    "constraint_filter",
    "pareto_front",
    "run_trial",
]
