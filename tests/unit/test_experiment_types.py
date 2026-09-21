from __future__ import annotations

from dataclasses import replace

import pytest

from uma_qmoe.experiments.ledger import trial_identity
from uma_qmoe.experiments.types import (
    Encoding,
    ExperimentTypeError,
    TrialSpec,
    Unit,
)


def _spec(**evaluation: object) -> TrialSpec:
    return TrialSpec(
        model_identity={"id": "model", "revision": "abc"},
        data_identity={"sha256": "d" * 64},
        unit=Unit(layer=3, projection="gate"),
        encoding=Encoding(
            storage="Q4",
            group_size=128,
            method="RTN",
            parameters={"clip": 1.0},
        ),
        quantizer_version="rtn-v1",
        evaluation=evaluation,
        seed=7,
    )


def test_trial_identity_is_canonical_and_covers_semantics() -> None:
    left = _spec(b=2, a=1)
    right = _spec(a=1, b=2)
    changed = _spec(a=1, b=3)

    assert trial_identity(left) == trial_identity(right)
    assert trial_identity(left) != trial_identity(changed)
    assert left.encoding.storage == "q4"
    assert left.encoding.method == "rtn"


@pytest.mark.parametrize(
    "change",
    [
        {"model_identity": {"id": "other"}},
        {"data_identity": {"sha256": "e" * 64}},
        {"unit": Unit(4, "gate")},
        {"encoding": Encoding("q8", 128, "rtn")},
        {"quantizer_version": "rtn-v2"},
        {"evaluation": {"a": 2, "b": 2}},
        {"seed": 8},
        {"diagnostic_only": False},
    ],
)
def test_every_trial_identity_field_invalidates_reuse(change: dict) -> None:
    spec = _spec(a=1, b=2)

    assert trial_identity(replace(spec, **change)) != trial_identity(spec)


def test_identity_values_are_frozen_after_construction() -> None:
    source = {"nested": [1, 2]}
    spec = TrialSpec(
        model_identity={"id": "model"},
        data_identity={"sha256": "d" * 64},
        unit=Unit(0, "down"),
        encoding=Encoding("q8", 128, "rtn", source),
        quantizer_version="v1",
        evaluation={},
    )
    identity = trial_identity(spec)
    source["nested"].append(3)

    assert trial_identity(spec) == identity
    assert spec.encoding.to_dict()["parameters"] == {"nested": [1, 2]}


def test_identity_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises(ExperimentTypeError, match="NaN"):
        _spec(metric=float("nan"))
    with pytest.raises(ExperimentTypeError, match="non-JSON"):
        _spec(metric=object())
