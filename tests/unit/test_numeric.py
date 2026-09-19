from __future__ import annotations

import math

from uma_qmoe.numeric import clamp_cosine_similarity


def test_clamp_cosine_similarity_handles_reduction_roundoff() -> None:
    assert clamp_cosine_similarity(1.0000090599060059) == 1.0
    assert clamp_cosine_similarity(-1.0000090599060059) == -1.0


def test_clamp_cosine_similarity_preserves_valid_and_non_finite_values() -> None:
    assert clamp_cosine_similarity(0.875) == 0.875
    assert math.isnan(clamp_cosine_similarity(float("nan")))
    assert clamp_cosine_similarity(float("inf")) == float("inf")
