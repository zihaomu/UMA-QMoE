from __future__ import annotations

import struct

import numpy as np
import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.q4 import (
    dequantize_q4,
    pack_int4,
    q4_layout,
    quantize_q4,
    Q4Tensor,
    unpack_int4,
)


def test_canonical_nibble_order_and_signed_codes() -> None:
    packed = pack_int4(np.array([-7, -1, 0, 1, 7], dtype=np.int8))

    assert packed == bytes((0xF9, 0x10, 0x07))
    assert unpack_int4(packed, 5).tolist() == [-7, -1, 0, 1, 7]


def test_q4_group_128_tail_scale_and_effective_bpw() -> None:
    source = np.linspace(-3.0, 3.0, 129, dtype=np.float32)

    tensor = quantize_q4(source, group_size=128)
    restored = dequantize_q4(tensor)
    layout = q4_layout(129, 128, 64)

    assert layout.group_count == 2
    assert layout.packed_bytes == 65
    assert layout.scale_bytes == 8
    assert layout.storage_bytes == 128
    assert layout.effective_bits_per_weight == pytest.approx(128 * 8 / 129)
    scales = np.frombuffer(tensor.scales, dtype="<f4")
    assert np.max(np.abs(restored[:128] - source[:128])) <= scales[0] / 2 + 1e-6
    assert np.max(np.abs(restored[128:] - source[128:])) <= scales[1] / 2 + 1e-6


def test_q4_rejects_reserved_minus_eight_and_nonzero_tail_padding() -> None:
    with pytest.raises(ContractError, match="reserved -8"):
        unpack_int4(bytes((0x08,)), 1)
    with pytest.raises(ContractError, match="tail padding"):
        unpack_int4(bytes((0x10,)), 1)


def test_q4_zero_group_round_trips_exactly() -> None:
    source = np.zeros((3, 43), dtype=np.float32)
    tensor = quantize_q4(source)

    assert np.array_equal(dequantize_q4(tensor), source)
    assert np.frombuffer(tensor.scales, dtype="<f4").tolist() == [1.0, 1.0]


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_q4_rejects_invalid_scales(scale: float) -> None:
    tensor = Q4Tensor(
        shape=(2,),
        group_size=128,
        packed=b"\x00",
        scales=struct.pack("<f", scale),
    )
    with pytest.raises(ContractError, match="finite and positive"):
        dequantize_q4(tensor)
