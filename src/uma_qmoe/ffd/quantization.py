"""Page-wise packed Key quantization shared by FFD cache and oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import ContractError


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("FFD tensor operations require PyTorch") from exc
    return torch


def _validate_bits(bits: int) -> None:
    if bits not in (2, 4):
        raise ContractError("FFD packed keys support only 2-bit or 4-bit values")


@dataclass(frozen=True)
class QuantizedKeyBlocks:
    """Packed full blocks in canonical ``[B, T, H, D]`` layout."""

    packed: Any
    scales: Any
    residual: Any | None
    bits: int
    block_size: int
    head_dim: int
    token_count: int

    @property
    def block_count(self) -> int:
        return self.token_count // self.block_size

    @property
    def values_per_byte(self) -> int:
        return 8 // self.bits

    @property
    def packed_dim(self) -> int:
        return (self.head_dim + self.values_per_byte - 1) // self.values_per_byte


def pack_quantized(values: Any, bits: int) -> Any:
    """Pack unsigned quantization codes along the last dimension."""

    torch = _torch()
    _validate_bits(bits)
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ContractError("FFD quantization codes need a non-empty last dimension")
    qmax = (1 << bits) - 1
    if values.dtype == torch.bool or values.is_floating_point():
        raise ContractError("FFD quantization codes must use an integer dtype")
    if bool(torch.any(values < 0).item()) or bool(torch.any(values > qmax).item()):
        raise ContractError(f"FFD {bits}-bit code is outside [0, {qmax}]")
    values_per_byte = 8 // bits
    pad = (-values.shape[-1]) % values_per_byte
    if pad:
        values = torch.nn.functional.pad(values, (0, pad), value=0)
    grouped = values.to(torch.int32).reshape(
        *values.shape[:-1], values.shape[-1] // values_per_byte, values_per_byte
    )
    packed = torch.zeros(grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for index in range(values_per_byte):
        packed = packed | (grouped[..., index] << (index * bits))
    return packed.to(torch.uint8).contiguous()


def unpack_quantized(packed: Any, bits: int, head_dim: int) -> Any:
    """Unpack uint8 payload to unsigned int16 quantization codes."""

    torch = _torch()
    _validate_bits(bits)
    if packed.dtype != torch.uint8:
        raise ContractError("FFD packed key payload must have dtype uint8")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ContractError("FFD head_dim must be a positive integer")
    values_per_byte = 8 // bits
    expected = (head_dim + values_per_byte - 1) // values_per_byte
    if packed.shape[-1] != expected:
        raise ContractError(
            f"FFD packed dimension {packed.shape[-1]} does not match {expected}"
        )
    raw = packed.to(torch.int32)
    codes = torch.stack(
        [
            (raw >> (index * bits)) & ((1 << bits) - 1)
            for index in range(values_per_byte)
        ],
        dim=-1,
    ).reshape(*packed.shape[:-1], expected * values_per_byte)
    return codes[..., :head_dim].to(torch.int16)


def quantize_key_blocks(
    keys: Any,
    *,
    block_size: int,
    bits: int = 2,
    residual_dtype: str = "fp8_e4m3fn",
    eps: float = 1e-8,
) -> QuantizedKeyBlocks:
    """Quantize complete blocks with per-block/head/dimension scales.

    ``keys`` must be ``[batch, tokens, kv_heads, head_dim]`` and the token
    count must be divisible by ``block_size``.  No BF16 shadow copy is kept.
    """

    torch = _torch()
    _validate_bits(bits)
    if keys.ndim != 4:
        raise ContractError("FFD keys must have shape [B, T, H, D]")
    batch, token_count, heads, head_dim = keys.shape
    if not all(int(item) > 0 for item in (batch, token_count, heads, head_dim)):
        raise ContractError("FFD keys cannot contain an empty dimension")
    if token_count % block_size:
        raise ContractError("FFD only quantizes complete key blocks")
    if not keys.is_floating_point() or not bool(torch.isfinite(keys).all().item()):
        raise ContractError("FFD keys must contain finite floating-point values")
    if residual_dtype not in ("fp8_e4m3fn", "bf16", "none"):
        raise ContractError("unsupported FFD residual dtype")
    block_count = token_count // block_size
    source = keys.reshape(batch, block_count, block_size, heads, head_dim)
    qmax = (1 << bits) - 1
    qzero = qmax / 2.0
    scales = (source.float().abs().amax(dim=2) / qzero).clamp_min(eps)
    codes = torch.round(source.float() / scales[:, :, None] + qzero).clamp(0, qmax)
    dequantized = (codes - qzero) * scales[:, :, None]
    residual = None
    if residual_dtype != "none":
        residual_values = source.float() - dequantized
        if residual_dtype == "fp8_e4m3fn":
            if not hasattr(torch, "float8_e4m3fn"):
                raise ContractError("this PyTorch runtime has no float8_e4m3fn dtype")
            try:
                residual = residual_values.to(torch.float8_e4m3fn)
            except (RuntimeError, TypeError) as exc:
                raise ContractError(
                    "FP8 residual is unavailable on the selected FFD device"
                ) from exc
        else:
            residual = residual_values.to(torch.bfloat16)
        residual = residual.reshape(batch, token_count, heads, head_dim).contiguous()
    packed = pack_quantized(
        codes.to(torch.uint8).reshape(batch, token_count, heads, head_dim), bits
    )
    return QuantizedKeyBlocks(
        packed=packed,
        scales=scales.contiguous(),
        residual=residual,
        bits=bits,
        block_size=block_size,
        head_dim=head_dim,
        token_count=token_count,
    )


def dequantize_key_blocks(value: QuantizedKeyBlocks, *, include_residual: bool) -> Any:
    """Reference-only materialization used by correctness tests and oracles."""

    torch = _torch()
    codes = unpack_quantized(value.packed, value.bits, value.head_dim).float()
    batch, token_count, heads, head_dim = codes.shape
    if token_count != value.token_count:
        raise ContractError("FFD packed token count disagrees with metadata")
    if value.scales.shape != (
        batch,
        value.block_count,
        heads,
        head_dim,
    ):
        raise ContractError("FFD scale shape disagrees with packed keys")
    expanded_scales = (
        value.scales[:, :, None]
        .expand(batch, value.block_count, value.block_size, heads, head_dim)
        .reshape(batch, token_count, heads, head_dim)
    )
    qzero = ((1 << value.bits) - 1) / 2.0
    result = (codes - qzero) * expanded_scales.float()
    if include_residual:
        if value.residual is None:
            raise ContractError("FFD residual was requested but is not present")
        result = result + value.residual.float()
    if not bool(torch.isfinite(result).all().item()):
        raise ContractError("FFD dequantization produced NaN or Inf")
    return result


__all__ = [
    "QuantizedKeyBlocks",
    "dequantize_key_blocks",
    "pack_quantized",
    "quantize_key_blocks",
    "unpack_quantized",
]
