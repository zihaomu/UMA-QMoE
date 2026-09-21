"""Append-only JSONL ledger with stable trial identities and safe resume."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .types import COMPLETED_STATUSES, TrialResult, TrialSpec, TrialStatus


class LedgerError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"value is not canonical JSON: {exc}") from exc


def trial_identity(spec: TrialSpec) -> str:
    """SHA-256 of the complete, normalized semantic trial specification."""

    return hashlib.sha256(canonical_json_bytes(spec.to_dict())).hexdigest()


class JsonlLedger:
    """One durable JSON object per attempt; interrupted attempts are retryable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise LedgerError(f"cannot read ledger {self.path}: {exc}") from exc
        lines = payload.splitlines(keepends=True)
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(lines, start=1):
            complete_line = raw.endswith((b"\n", b"\r"))
            if not complete_line and index == len(lines):
                # A killed append may leave an arbitrary suffix.  It is not a result.
                continue
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LedgerError(f"invalid ledger row {index}: {exc}") from exc
            if not isinstance(value, dict):
                raise LedgerError(f"ledger row {index} is not an object")
            self._validate_row(value, index)
            rows.append(value)
        return rows

    @staticmethod
    def _validate_row(row: Mapping[str, Any], line_number: int | None = None) -> None:
        location = "ledger row" if line_number is None else f"ledger row {line_number}"
        try:
            canonical_json_bytes(row)
        except LedgerError as exc:
            raise LedgerError(f"{location} contains invalid JSON values: {exc}") from exc
        if row.get("kind") != "exploratory_trial_result":
            raise LedgerError(f"{location} is not exploratory trial evidence")
        if row.get("schema_version") != 1:
            raise LedgerError(f"{location} has an unsupported schema version")
        if row.get("evidence_level") != "L0":
            raise LedgerError(f"{location} must be marked as L0 evidence")
        try:
            spec = TrialSpec.from_dict(row["spec"])
            status = TrialStatus(row["status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"{location} has invalid trial fields: {exc}") from exc
        del status
        if row.get("trial_id") != trial_identity(spec):
            raise LedgerError(f"{location} trial identity does not match its spec")

    def records(self) -> list[dict[str, Any]]:
        return self._rows()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows())

    def completed_ids(self) -> frozenset[str]:
        return frozenset(
            row["trial_id"]
            for row in self._rows()
            if TrialStatus(row["status"]) in COMPLETED_STATUSES
        )

    def is_complete(self, spec_or_id: TrialSpec | str) -> bool:
        identity = (
            trial_identity(spec_or_id)
            if isinstance(spec_or_id, TrialSpec)
            else spec_or_id
        )
        return identity in self.completed_ids()

    def append(self, result: TrialResult) -> None:
        if result.trial_id != trial_identity(result.spec):
            raise LedgerError("result trial identity does not match its spec")
        row = result.to_dict()
        self._validate_row(row)
        encoded = canonical_json_bytes(row) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a+b", buffering=0) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.seek(0)
                existing = stream.read()
                if existing and not existing.endswith(b"\n"):
                    # The suffix was never a complete record.  Remove it before the
                    # next atomic append so it cannot become a corrupt interior row.
                    last_newline = existing.rfind(b"\n")
                    stream.seek(last_newline + 1)
                    stream.truncate()
                stream.seek(0, os.SEEK_END)
                stream.write(encoded)
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise LedgerError(f"cannot append ledger {self.path}: {exc}") from exc


__all__ = [
    "JsonlLedger",
    "LedgerError",
    "canonical_json_bytes",
    "trial_identity",
]
