from __future__ import annotations

import numpy as np
import pytest

from uma_qmoe.experiments.quantizers import QuantizationError, RTNQuantizer
from uma_qmoe.experiments.types import CalibrationView, Encoding


CALIBRATION = CalibrationView(identity={"kind": "none"})


def test_rtn_q4_returns_restored_weight_payload_and_metrics() -> None:
    source = np.linspace(-2.0, 2.0, 256, dtype=np.float32).reshape(2, 128)

    candidate = RTNQuantizer().quantize(
        source, CALIBRATION, Encoding("q4", 128, "rtn")
    )

    assert candidate.restored_weight.shape == source.shape
    assert candidate.payload.dtype == np.int8
    assert candidate.scales.shape == (2,)
    assert candidate.offline_metrics["mse"] > 0
    assert np.max(np.abs(candidate.payload)) <= 7


def test_rtn_identity_control_is_exact() -> None:
    source = np.arange(128, dtype=np.float32)
    candidate = RTNQuantizer().quantize(
        source, CALIBRATION, Encoding("bf16", None, "rtn")
    )

    assert np.array_equal(candidate.restored_weight, source)
    assert candidate.offline_metrics["mse"] == 0.0


def test_rtn_rejects_misaligned_group() -> None:
    with pytest.raises(QuantizationError, match="divisible"):
        RTNQuantizer().quantize(
            np.ones(129, dtype=np.float32),
            CALIBRATION,
            Encoding("q4", 128, "rtn"),
        )
