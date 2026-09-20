"""Canonical group-wise signed Q4 pack/unpack primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .contracts import ContractError


Q4_BITS = 4
Q4_MIN = -7
Q4_MAX = 7
DEFAULT_GROUP_SIZE = 128
DEFAULT_ALIGNMENT = 64


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractError("Canonical Q4 conversion requires the conversion extra") from exc
    return np


def align_up(value: int, alignment: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("alignment input must be a non-negative integer")
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment & (alignment - 1)
    ):
        raise ContractError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class Q4Layout:
    element_count: int
    group_size: int
    group_count: int
    packed_bytes: int
    scale_bytes: int
    unpadded_bytes: int
    storage_bytes: int
    alignment_bytes: int
    effective_bits_per_weight: float


@dataclass(frozen=True)
class Q4Tensor:
    shape: tuple[int, ...]
    group_size: int
    packed: bytes
    scales: bytes

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def layout(self) -> Q4Layout:
        return q4_layout(self.element_count, self.group_size)


def q4_layout(
    element_count: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    alignment_bytes: int = DEFAULT_ALIGNMENT,
) -> Q4Layout:
    if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
        raise ContractError("Q4 element_count must be a positive integer")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ContractError("Q4 group_size must be a positive integer")
    group_count = math.ceil(element_count / group_size)
    packed_bytes = math.ceil(element_count / 2)
    scale_bytes = group_count * 4
    unpadded = packed_bytes + scale_bytes
    storage = align_up(unpadded, alignment_bytes)
    return Q4Layout(
        element_count=element_count,
        group_size=group_size,
        group_count=group_count,
        packed_bytes=packed_bytes,
        scale_bytes=scale_bytes,
        unpadded_bytes=unpadded,
        storage_bytes=storage,
        alignment_bytes=alignment_bytes,
        effective_bits_per_weight=storage * 8 / element_count,
    )


def pack_int4(values: Any) -> bytes:
    """Pack signed values in [-7, 7], low nibble first, two's complement."""

    np = _numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ContractError("cannot pack an empty Q4 array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ContractError("Q4 nibble input must be integer")
    normalized = array.astype(np.int16, copy=False)
    if np.any(normalized < Q4_MIN) or np.any(normalized > Q4_MAX):
        raise ContractError("Q4 nibble input is outside [-7, 7]")
    nibbles = (normalized & 0xF).astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed.tobytes()


def unpack_int4(payload: bytes | bytearray | memoryview, element_count: int) -> Any:
    np = _numpy()
    layout = q4_layout(element_count)
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size != layout.packed_bytes:
        raise ContractError("Q4 packed byte count does not match element count")
    nibbles = np.empty(raw.size * 2, dtype=np.uint8)
    nibbles[0::2] = raw & 0xF
    nibbles[1::2] = raw >> 4
    signed = nibbles.astype(np.int8)
    signed[signed >= 8] -= 16
    values = signed[:element_count]
    if np.any(values < Q4_MIN) or np.any(values > Q4_MAX):
        raise ContractError("Q4 payload contains the reserved -8 code")
    if element_count % 2 and int(nibbles[-1]) != 0:
        raise ContractError("Q4 odd tail padding nibble must be zero")
    return values


def quantize_q4(values: Any, *, group_size: int = DEFAULT_GROUP_SIZE) -> Q4Tensor:
    """Symmetric per-group Q4 using float32 scales and round-to-nearest-even."""

    np = _numpy()
    source = np.asarray(values)
    if source.size == 0:
        raise ContractError("cannot quantize an empty tensor")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise ContractError("Q4 source must contain only finite numeric values")
    shape = tuple(int(item) for item in source.shape)
    flat = source.astype(np.float32, copy=False).reshape(-1)
    layout = q4_layout(int(flat.size), group_size)
    padded_count = layout.group_count * group_size
    if padded_count == flat.size:
        padded = flat
    else:
        padded = np.zeros(padded_count, dtype=np.float32)
        padded[: flat.size] = flat
    groups = padded.reshape(layout.group_count, group_size)
    scales = (np.max(np.abs(groups), axis=1) / Q4_MAX).astype("<f4")
    scales[scales == 0] = 1.0
    quantized = np.clip(
        np.rint(groups / scales[:, None]), Q4_MIN, Q4_MAX
    ).astype(np.int8).reshape(-1)[: flat.size]
    return Q4Tensor(
        shape=shape,
        group_size=group_size,
        packed=pack_int4(quantized),
        scales=scales.tobytes(),
    )


def dequantize_q4(tensor: Q4Tensor) -> Any:
    np = _numpy()
    layout = q4_layout(tensor.element_count, tensor.group_size)
    if len(tensor.packed) != layout.packed_bytes or len(tensor.scales) != layout.scale_bytes:
        raise ContractError("Q4 tensor payload lengths do not match layout")
    quantized = unpack_int4(tensor.packed, tensor.element_count).astype(np.float32)
    scales = np.frombuffer(tensor.scales, dtype="<f4")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ContractError("Q4 scales must be finite and positive")
    expanded_scales = np.repeat(scales, tensor.group_size)[: tensor.element_count]
    result = quantized * expanded_scales
    return result.reshape(tensor.shape)


__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_GROUP_SIZE",
    "Q4Layout",
    "Q4Tensor",
    "align_up",
    "dequantize_q4",
    "pack_int4",
    "q4_layout",
    "quantize_q4",
    "unpack_int4",
]
