"""Compressed FFD KV cache with a high-precision incomplete tail."""

from __future__ import annotations

import math
from typing import Any

from ..contracts import ContractError
from .policy import FFDPolicy
from .quantization import QuantizedKeyBlocks, quantize_key_blocks

try:  # Transformers is a runtime integration dependency, not a core dependency.
    from transformers.cache_utils import Cache as _TransformersCache
    from transformers.cache_utils import CacheLayerMixin as _CacheLayerMixin
except ImportError:  # pragma: no cover - exercised by core-only CI imports.

    class _CacheLayerMixin:
        def __init__(self) -> None:
            self.keys = None
            self.values = None
            self.is_initialized = False

    class _TransformersCache:
        def __init__(self, layers=None, **_: Any) -> None:
            self.layers = [] if layers is None else layers


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("FFD cache requires PyTorch") from exc
    return torch


def _tensor_bytes(value: Any | None) -> int:
    if value is None:
        return 0
    return int(value.numel() * value.element_size())


class FFDLayerCache(_CacheLayerMixin):
    """One layer of compressed Keys and BF16/FP16 Values.

    Public ``update`` accepts the Transformers cache layout ``[B, H, T, D]``.
    Internal tensors use ``[B, T, H, D]`` so token blocks are contiguous.
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, policy: FFDPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.block_size = policy.block_size
        self.max_seq_len = policy.max_seq_len
        self.max_full_tokens = self.max_seq_len // self.block_size * self.block_size
        self.k_q = None
        self.k_scale = None
        self.k_residual = None
        self.k_tail = None
        self.v_tail = None
        self.value = None
        self.value_len = 0
        self.full_token_count = 0
        self.current_len = 0
        self.dtype = None
        self.device = None
        self.batch_size = 0
        self.num_heads = 0
        self.head_dim = 0
        # Deliberately never points at a dequantized Key shadow.
        self.keys = None
        self.values = None

    def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
        torch = _torch()
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ContractError("FFD cache K/V must use [B, H, T, D] layout")
        batch, heads, _, head_dim = key_states.shape
        if value_states.shape[:3] != key_states.shape[:3]:
            raise ContractError("FFD cache K/V leading dimensions must match")
        if key_states.device != value_states.device:
            raise ContractError("FFD cache K/V must be on the same device")
        values_per_byte = 8 // self.policy.key_bits
        packed_dim = math.ceil(head_dim / values_per_byte)
        full_blocks = self.max_full_tokens // self.block_size
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = int(batch)
        self.num_heads = int(heads)
        self.head_dim = int(head_dim)
        value_dim = int(value_states.shape[-1])
        self.k_q = torch.empty(
            (batch, self.max_full_tokens, heads, packed_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self.k_scale = torch.empty(
            (batch, full_blocks, heads, head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.policy.residual_dtype == "fp8_e4m3fn":
            if not hasattr(torch, "float8_e4m3fn"):
                raise ContractError("the FFD runtime has no float8_e4m3fn dtype")
            residual_dtype = torch.float8_e4m3fn
        elif self.policy.residual_dtype == "bf16":
            residual_dtype = torch.bfloat16
        else:
            residual_dtype = None
        if residual_dtype is not None:
            try:
                self.k_residual = torch.empty(
                    (batch, self.max_full_tokens, heads, head_dim),
                    dtype=residual_dtype,
                    device=self.device,
                )
            except (RuntimeError, TypeError) as exc:
                raise ContractError(
                    f"FFD residual dtype {self.policy.residual_dtype!r} is unavailable"
                ) from exc
        self.k_tail = torch.empty(
            (batch, self.block_size, heads, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_tail = torch.empty(
            (batch, self.block_size, heads, value_dim),
            dtype=value_states.dtype,
            device=self.device,
        )
        self.value = torch.empty(
            (batch, self.max_seq_len, heads, value_dim),
            dtype=value_states.dtype,
            device=self.device,
        )
        self.is_initialized = True

    def _validate_position(
        self, cache_kwargs: dict[str, Any] | None, new_len: int
    ) -> None:
        if not cache_kwargs or cache_kwargs.get("cache_position") is None:
            return
        torch = _torch()
        position = cache_kwargs["cache_position"]
        expected = torch.arange(
            self.value_len,
            self.value_len + new_len,
            device=position.device,
            dtype=position.dtype,
        )
        if position.shape != expected.shape or not bool(
            torch.equal(position, expected)
        ):
            raise ContractError(
                "FFD cache only accepts sequential cache_position updates"
            )

    def update(
        self,
        key_states: Any,
        value_states: Any,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        torch = _torch()
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ContractError("FFD cache K/V must use [B, H, T, D] layout")
        if (
            key_states.shape[0] != self.batch_size
            or key_states.shape[1] != self.num_heads
        ):
            raise ContractError(
                "FFD cache batch/head shape cannot change after allocation"
            )
        if key_states.shape[-1] != self.head_dim:
            raise ContractError(
                "FFD cache head dimension cannot change after allocation"
            )
        new_len = int(key_states.shape[2])
        if new_len <= 0:
            raise ContractError("FFD cache cannot append zero tokens")
        if self.value_len + new_len > self.max_seq_len:
            raise ContractError(
                f"FFD cache overflow: {self.value_len + new_len} > {self.max_seq_len}"
            )
        self._validate_position(cache_kwargs, new_len)
        canonical_k = key_states.transpose(1, 2).contiguous()
        canonical_v = value_states.transpose(1, 2).contiguous()
        self.value[:, self.value_len : self.value_len + new_len].copy_(canonical_v)

        if self.current_len:
            combined = torch.cat(
                (self.k_tail[:, : self.current_len], canonical_k), dim=1
            )
        else:
            combined = canonical_k
        complete_tokens = combined.shape[1] // self.block_size * self.block_size
        if complete_tokens:
            quantized = quantize_key_blocks(
                combined[:, :complete_tokens],
                block_size=self.block_size,
                bits=self.policy.key_bits,
                residual_dtype=self.policy.residual_dtype,
            )
            start = self.full_token_count
            end = start + complete_tokens
            if end > self.max_full_tokens:
                raise ContractError(
                    "FFD complete-block storage exceeded its allocation"
                )
            start_block = start // self.block_size
            end_block = end // self.block_size
            self.k_q[:, start:end].copy_(quantized.packed)
            self.k_scale[:, start_block:end_block].copy_(quantized.scales)
            if self.k_residual is not None:
                if quantized.residual is None:
                    raise ContractError("FFD residual allocation/source mismatch")
                self.k_residual[:, start:end].copy_(quantized.residual)
            self.full_token_count = end
        remainder = combined[:, complete_tokens:]
        self.current_len = int(remainder.shape[1])
        if self.current_len:
            self.k_tail[:, : self.current_len].copy_(remainder)
            value_tail_start = self.value_len + new_len - self.current_len
            self.v_tail[:, : self.current_len].copy_(
                self.value[:, value_tail_start : self.value_len + new_len]
            )
        self.value_len += new_len
        self.values = None
        empty_k = torch.empty(
            (self.batch_size, self.num_heads, 0, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        empty_v = torch.empty(
            (self.batch_size, self.num_heads, 0, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=self.device,
        )
        return empty_k, empty_v

    def quantized_blocks(self) -> QuantizedKeyBlocks:
        if not self.is_initialized or self.full_token_count <= 0:
            raise ContractError("FFD layer has no complete quantized block")
        block_count = self.full_token_count // self.block_size
        return QuantizedKeyBlocks(
            packed=self.k_q[:, : self.full_token_count],
            scales=self.k_scale[:, :block_count],
            residual=(
                None
                if self.k_residual is None
                else self.k_residual[:, : self.full_token_count]
            ),
            bits=self.policy.key_bits,
            block_size=self.block_size,
            head_dim=self.head_dim,
            token_count=self.full_token_count,
        )

    def full_values(self) -> Any:
        return self.value[:, : self.full_token_count]

    def tail_keys(self) -> Any | None:
        return None if self.current_len == 0 else self.k_tail[:, : self.current_len]

    def tail_values(self) -> Any | None:
        return None if self.current_len == 0 else self.v_tail[:, : self.current_len]

    def get_seq_length(self) -> int:
        return self.value_len

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        query_length = int(cache_position.shape[0])
        return self.value_len + query_length, 0

    def reset(self) -> None:
        self.value_len = 0
        self.full_token_count = 0
        self.current_len = 0

    def reorder_cache(self, beam_idx: Any) -> None:
        if not self.is_initialized or self.value_len == 0:
            return
        index = beam_idx.to(self.device)
        for name in (
            "k_q",
            "k_scale",
            "k_residual",
            "k_tail",
            "v_tail",
            "value",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, index))
        self.batch_size = int(index.numel())

    def batch_select_indices(self, indices: Any) -> None:
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        torch = _torch()
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ContractError("FFD cache repeats must be a positive integer")
        if not self.is_initialized or self.value_len == 0:
            return
        indices = torch.arange(self.batch_size, device=self.device).repeat_interleave(
            repeats
        )
        self.reorder_cache(indices)

    def memory_breakdown(self) -> dict[str, int]:
        full_blocks = self.full_token_count // self.block_size
        scale = self.k_scale[:, :full_blocks] if self.is_initialized else None
        packed = self.k_q[:, : self.full_token_count] if self.is_initialized else None
        residual = (
            self.k_residual[:, : self.full_token_count]
            if self.k_residual is not None
            else None
        )
        values = self.value[:, : self.value_len] if self.is_initialized else None
        tail = self.k_tail[:, : self.current_len] if self.is_initialized else None
        tail_values = (
            self.v_tail[:, : self.current_len] if self.is_initialized else None
        )
        payload = {
            "q2_or_q4_key_bytes": _tensor_bytes(packed),
            "scale_bytes": _tensor_bytes(scale),
            "residual_bytes": _tensor_bytes(residual),
            "value_bytes": _tensor_bytes(values),
            "tail_key_bytes": _tensor_bytes(tail),
            "tail_value_duplicate_bytes": _tensor_bytes(tail_values),
        }
        payload["payload_bytes"] = sum(payload.values())
        payload["allocated_bytes"] = sum(
            _tensor_bytes(getattr(self, name))
            for name in ("k_q", "k_scale", "k_residual", "k_tail", "v_tail", "value")
        )
        return payload


class DenseLayerCache(_CacheLayerMixin):
    """Preallocated dense layer used for a policy's non-FFD layers."""

    is_sliding = False
    is_compileable = False

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.length = 0

    def lazy_initialization(self, key_states: Any, value_states: Any) -> None:
        torch = _torch()
        batch, heads, _, head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.empty(
            (batch, heads, self.max_seq_len, head_dim),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.empty(
            (batch, heads, self.max_seq_len, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.is_initialized = True

    def update(
        self, key_states: Any, value_states: Any, cache_kwargs=None
    ) -> tuple[Any, Any]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        new_len = int(key_states.shape[2])
        if self.length + new_len > self.max_seq_len:
            raise ContractError("dense companion cache exceeded max_seq_len")
        self.keys[:, :, self.length : self.length + new_len].copy_(key_states)
        self.values[:, :, self.length : self.length + new_len].copy_(value_states)
        self.length += new_len
        return self.keys[:, :, : self.length], self.values[:, :, : self.length]

    def get_seq_length(self) -> int:
        return self.length

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        return self.length + int(cache_position.shape[0]), 0

    def reset(self) -> None:
        self.length = 0

    def reorder_cache(self, beam_idx: Any) -> None:
        if not self.is_initialized or self.length == 0:
            return
        index = beam_idx.to(self.device)
        self.keys = self.keys.index_select(0, index)
        self.values = self.values.index_select(0, index)

    def batch_select_indices(self, indices: Any) -> None:
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        torch = _torch()
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ContractError("dense cache repeats must be a positive integer")
        if not self.is_initialized or self.length == 0:
            return
        indices = torch.arange(
            self.keys.shape[0], device=self.device
        ).repeat_interleave(repeats)
        self.reorder_cache(indices)


class FFDCache(_TransformersCache):
    """Transformers-compatible mixed compressed/dense cache."""

    def __init__(self, policy: FFDPolicy) -> None:
        try:
            super().__init__(layers=[])
        except TypeError:  # Compatibility with older Transformers Cache.
            super().__init__()
            self.layers = []
        self.policy = policy

    def _ensure_layer(self, layer_index: int) -> None:
        if layer_index < 0 or layer_index >= self.policy.num_layers:
            raise ContractError("FFD cache layer index is outside the policy")
        while len(self.layers) <= layer_index:
            index = len(self.layers)
            layer = (
                FFDLayerCache(self.policy)
                if self.policy.applies_to(index)
                else DenseLayerCache(self.policy.max_seq_len)
            )
            self.layers.append(layer)

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        self._ensure_layer(layer_idx)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def layer(self, layer_index: int) -> FFDLayerCache:
        self._ensure_layer(layer_index)
        layer = self.layers[layer_index]
        if not isinstance(layer, FFDLayerCache):
            raise ContractError(f"layer {layer_index} is not selected for FFD")
        return layer

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers or layer_idx >= len(self.layers):
            return 0
        return int(self.layers[layer_idx].get_seq_length())

    def get_max_cache_shape(self) -> int:
        return self.policy.max_seq_len

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: Any) -> None:
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    def batch_select_indices(self, indices: Any) -> None:
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def memory_breakdown(self) -> dict[str, Any]:
        layers = {}
        totals: dict[str, int] = {}
        for index, layer in enumerate(self.layers):
            if isinstance(layer, FFDLayerCache):
                row = layer.memory_breakdown()
                layers[str(index)] = row
                for key, value in row.items():
                    totals[key] = totals.get(key, 0) + value
        return {"layers": layers, "totals": totals}


__all__ = ["DenseLayerCache", "FFDCache", "FFDLayerCache"]
