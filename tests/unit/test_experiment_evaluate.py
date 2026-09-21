from __future__ import annotations

import copy

import pytest

from uma_qmoe.experiments.evaluate import (
    aggregate_quality,
    compare_quality,
    route_agreement,
)


def _records() -> list[dict]:
    return [
        {
            "id": "sample",
            "finite": True,
            "per_token_nll": [1.0, 2.0],
            "target_token_ids": [4, 5],
            "completion_top1_token_ids": [4, 7],
            "routes": {"0": [[0, 1, 2, 3]], "1": [[4, 5, 6, 7]]},
        }
    ]


def test_shared_quality_comparison_reports_route_overlap() -> None:
    reference = _records()
    candidate = copy.deepcopy(reference)
    candidate[0]["routes"]["1"] = [[4, 5, 6, 8]]

    aggregate = aggregate_quality(reference)
    exact, per_layer, overlap = route_agreement(candidate, reference, num_layers=2)
    metrics = compare_quality(candidate, reference, num_layers=2)

    assert aggregate["nll"] == 1.5
    assert aggregate["completion_token_accuracy"] == 0.5
    assert exact == 0.5
    assert per_layer == [1.0, 0.0]
    assert overlap == pytest.approx(0.875)
    assert metrics["relative_perplexity_change"] == 0.0
