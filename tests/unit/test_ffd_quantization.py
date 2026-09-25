from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from uma_qmoe.contracts import ContractError  # noqa: E402
from uma_qmoe.ffd.quantization import (  # noqa: E402
    dequantize_key_blocks,
    pack_quantized,
    quantize_key_blocks,
    unpack_quantized,
)


@pytest.mark.parametrize("bits", [2, 4])
def test_pack_round_trip_with_odd_head_dimension(bits: int) -> None:
    maximum = (1 << bits) - 1
    values = torch.arange(2 * 3 * 5).reshape(2, 3, 5) % (maximum + 1)
    packed = pack_quantized(values, bits)
    restored = unpack_quantized(packed, bits, 5)
    assert torch.equal(restored, values.to(torch.int16))


@pytest.mark.parametrize("bits", [2, 4])
def test_block_quantization_shapes_and_residual_reconstruct(bits: int) -> None:
    torch.manual_seed(4)
    keys = torch.randn(1, 8, 2, 7, dtype=torch.float32)
    value = quantize_key_blocks(
        keys,
        block_size=4,
        bits=bits,
        residual_dtype="bf16",
    )
    assert value.scales.shape == (1, 2, 2, 7)
    assert value.packed.shape[-1] == (7 + (8 // bits) - 1) // (8 // bits)
    restored = dequantize_key_blocks(value, include_residual=True)
    assert torch.allclose(restored, keys, atol=0.02, rtol=0.02)


def test_quantization_requires_complete_blocks() -> None:
    with pytest.raises(ContractError, match="complete key blocks"):
        quantize_key_blocks(torch.ones(1, 7, 1, 8), block_size=4)
