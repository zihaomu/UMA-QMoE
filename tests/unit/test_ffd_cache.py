from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from uma_qmoe.contracts import ContractError  # noqa: E402
from uma_qmoe.ffd.cache import FFDCache  # noqa: E402
from uma_qmoe.ffd.policy import FFDPolicy  # noqa: E402


def _policy(**kwargs) -> FFDPolicy:
    return FFDPolicy(
        policy_id="unit",
        block_size=64,
        sink_tokens=64,
        local_tokens=64,
        max_seq_len=256,
        num_layers=2,
        residual_dtype="bf16",
        require_fused_backend=False,
        **kwargs,
    )


def test_cache_quantizes_full_blocks_and_preserves_tail_without_key_shadow() -> None:
    cache = FFDCache(_policy())
    key = torch.randn(1, 2, 70, 8)
    value = torch.randn(1, 2, 70, 8)
    returned_key, returned_value = cache.update(
        key,
        value,
        0,
        {"cache_position": torch.arange(70)},
    )
    layer = cache.layer(0)
    assert returned_key.shape == (1, 2, 0, 8)
    assert returned_value.shape == (1, 2, 0, 8)
    assert layer.keys is None
    assert layer.full_token_count == 64
    assert layer.current_len == 6
    assert layer.get_seq_length() == 70
    assert layer.quantized_blocks().packed.shape == (1, 64, 2, 2)
    assert torch.equal(layer.tail_keys(), key.transpose(1, 2)[:, 64:])


def test_cache_crosses_block_boundary_on_decode_append() -> None:
    cache = FFDCache(_policy())
    cache.update(torch.randn(1, 2, 63, 8), torch.randn(1, 2, 63, 8), 0)
    cache.update(
        torch.randn(1, 2, 1, 8),
        torch.randn(1, 2, 1, 8),
        0,
        {"cache_position": torch.tensor([63])},
    )
    layer = cache.layer(0)
    assert layer.full_token_count == 64
    assert layer.current_len == 0
    assert layer.get_seq_length() == 64


def test_cache_non_selected_layer_remains_dense() -> None:
    cache = FFDCache(_policy(layer_indices=(1,)))
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    dense_key, dense_value = cache.update(key, value, 0)
    assert torch.equal(dense_key, key)
    assert torch.equal(dense_value, value)


def test_cache_rejects_non_sequential_positions_and_overflow() -> None:
    cache = FFDCache(_policy())
    with pytest.raises(ContractError, match="sequential"):
        cache.update(
            torch.randn(1, 2, 2, 8),
            torch.randn(1, 2, 2, 8),
            0,
            {"cache_position": torch.tensor([1, 2])},
        )
    with pytest.raises(ContractError, match="overflow"):
        cache.update(torch.randn(1, 2, 257, 8), torch.randn(1, 2, 257, 8), 0)


def test_cache_reorder_and_repeat_cover_every_compressed_buffer() -> None:
    cache = FFDCache(_policy())
    key = torch.randn(2, 2, 70, 8)
    value = torch.randn(2, 2, 70, 8)
    cache.update(key, value, 0)
    original_tail = cache.layer(0).tail_keys().clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    assert torch.equal(cache.layer(0).tail_keys()[0], original_tail[1])
    cache.batch_repeat_interleave(2)
    assert cache.layer(0).batch_size == 4
    assert torch.equal(cache.layer(0).tail_keys()[0], cache.layer(0).tail_keys()[1])
