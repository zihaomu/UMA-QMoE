"""Numerically safe helpers for evidence metrics."""

from __future__ import annotations

import math


def clamp_cosine_similarity(value: float) -> float:
    """Clamp finite cosine output to its mathematical range.

    Floating-point reductions can produce values a few ulps outside [-1, 1]
    even though cosine similarity is mathematically bounded by that interval.
    Non-finite values are left untouched so evidence validation still rejects
    them instead of disguising a failed computation.
    """

    if not math.isfinite(value):
        return value
    return max(-1.0, min(1.0, value))
