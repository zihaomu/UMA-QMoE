from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from uma_qmoe.ffd.oracle import (  # noqa: E402
    dense_attention,
    exact_attention_scores,
    selector_fidelity_report,
)


def test_exact_scores_support_grouped_query_attention() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    keys = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
    scores = exact_attention_scores(query, keys, scale=1.0)
    assert scores.shape == (1, 4, 1)
    assert scores.flatten().tolist() == [2.0, 0.0, 0.0, 3.0]


def test_dense_attention_returns_expected_single_token_value() -> None:
    query = torch.ones(1, 1, 2)
    keys = torch.ones(1, 1, 1, 2)
    values = torch.tensor([[[[3.0, 4.0]]]])
    output = dense_attention(query, keys, values)
    assert torch.equal(output, torch.tensor([[[3.0, 4.0]]]))


def test_selector_report_exposes_all_promotion_metrics() -> None:
    torch.manual_seed(9)
    query = torch.randn(1, 2, 8)
    keys = torch.randn(1, 16, 2, 8)
    values = torch.randn(1, 16, 2, 8)
    report = selector_fidelity_report(
        query,
        keys,
        values,
        delta=7,
        block_size=4,
        sink_tokens=4,
        local_tokens=4,
    )
    assert report["shape"] == {
        "batch": 1,
        "tokens": 16,
        "query_heads": 2,
        "kv_heads": 2,
        "head_dim": 8,
    }
    for name in (
        "pseudo_max_gap",
        "block_recall",
        "block_precision",
        "salient_token_false_negative_rate",
        "selected_attention_mass",
        "keep_ratio",
        "attention_output_cosine",
    ):
        assert "mean" in report[name]
    assert 0.0 <= report["keep_ratio"]["mean"] <= 1.0
