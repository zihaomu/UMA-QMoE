"""Unified quantizer protocol and the round-to-nearest baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, runtime_checkable

from .types import CalibrationView, Encoding, FrozenJsonValue


class QuantizationError(ValueError):
    pass


@dataclass
class QuantizedCandidate:
    """Transient candidate data; quantizers never write artifacts themselves."""

    restored_weight: Any
    payload: Any | None = None
    scales: Any | None = None
    zero_points: Any | None = None
    offline_metrics: dict[str, float | int] = field(default_factory=dict)
    diagnostics: dict[str, FrozenJsonValue] = field(default_factory=dict)


@runtime_checkable
class Quantizer(Protocol):
    version: str

    def quantize(
        self,
        weight: Any,
        calibration: CalibrationView,
        encoding: Encoding,
    ) -> QuantizedCandidate: ...


def _torch_rtn(weight: Any, bits: int, group_size: int) -> QuantizedCandidate:
    import torch

    if weight.numel() == 0 or weight.numel() % group_size:
        raise QuantizationError("weight count must be divisible by group_size")
    source = weight.detach().float()
    if not bool(torch.isfinite(source).all().item()):
        raise QuantizationError("weight must contain only finite values")
    groups = source.reshape(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    maximum = groups.abs().amax(dim=1, keepdim=True)
    scales = maximum / qmax
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    payload = torch.round(groups / safe_scales).clamp(-qmax, qmax).to(torch.int8)
    restored_float = payload.float() * safe_scales
    difference = restored_float - groups
    squared_error = difference.square().sum()
    source_squared = groups.square().sum()
    restored = restored_float.reshape(weight.shape).to(dtype=weight.dtype)
    return QuantizedCandidate(
        restored_weight=restored,
        payload=payload,
        scales=safe_scales.squeeze(1),
        zero_points=None,
        offline_metrics={
            "element_count": int(weight.numel()),
            "mse": float((squared_error / weight.numel()).item()),
            "max_abs_error": float(difference.abs().max().item()),
            "relative_l2_error": float(
                torch.sqrt(squared_error / source_squared.clamp_min(1e-30)).item()
            ),
        },
        diagnostics={
            "bits": bits,
            "group_count": int(groups.shape[0]),
            "zero_groups": int((maximum == 0).sum().item()),
            "rounding": "nearest_even",
            "symmetric": True,
        },
    )


def _numpy_rtn(weight: Any, bits: int, group_size: int) -> QuantizedCandidate:
    import numpy as np

    source_array = np.asarray(weight)
    if source_array.size == 0 or source_array.size % group_size:
        raise QuantizationError("weight count must be divisible by group_size")
    source = source_array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(source)):
        raise QuantizationError("weight must contain only finite values")
    groups = source.reshape(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(groups), axis=1, keepdims=True)
    scales = maximum / qmax
    safe_scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    payload = np.clip(np.rint(groups / safe_scales), -qmax, qmax).astype(np.int8)
    restored_float = payload.astype(np.float32) * safe_scales
    difference = restored_float - groups
    squared_error = float(np.square(difference).sum())
    source_squared = max(float(np.square(groups).sum()), 1e-30)
    return QuantizedCandidate(
        restored_weight=restored_float.reshape(source_array.shape).astype(
            source_array.dtype, copy=False
        ),
        payload=payload,
        scales=safe_scales[:, 0],
        offline_metrics={
            "element_count": int(source_array.size),
            "mse": squared_error / source_array.size,
            "max_abs_error": float(np.max(np.abs(difference))),
            "relative_l2_error": math.sqrt(squared_error / source_squared),
        },
        diagnostics={
            "bits": bits,
            "group_count": int(groups.shape[0]),
            "zero_groups": int(np.count_nonzero(maximum == 0)),
            "rounding": "nearest_even",
            "symmetric": True,
        },
    )


class RTNQuantizer:
    """Symmetric, per-group RTN for Q3-Q8 plus a BF16 identity control."""

    version = "rtn-v1"

    def quantize(
        self,
        weight: Any,
        calibration: CalibrationView,
        encoding: Encoding,
    ) -> QuantizedCandidate:
        del calibration  # RTN deliberately does not consume calibration statistics.
        if encoding.method != "rtn":
            raise QuantizationError("RTNQuantizer requires method='rtn'")
        if encoding.storage == "bf16":
            restored = weight.detach().clone() if hasattr(weight, "detach") else weight.copy()
            element_count = int(weight.numel() if hasattr(weight, "numel") else weight.size)
            return QuantizedCandidate(
                restored_weight=restored,
                offline_metrics={
                    "element_count": element_count,
                    "mse": 0.0,
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                },
                diagnostics={"bits": 16, "identity_control": True},
            )
        if not encoding.storage.startswith("q") or not encoding.storage[1:].isdigit():
            raise QuantizationError(f"unsupported RTN storage {encoding.storage!r}")
        bits = int(encoding.storage[1:])
        if not 3 <= bits <= 8:
            raise QuantizationError("RTN integer storage must be between Q3 and Q8")
        if encoding.group_size is None:
            raise QuantizationError("integer RTN requires group_size")
        module = type(weight).__module__.split(".", 1)[0]
        if module == "torch":
            return _torch_rtn(weight, bits, encoding.group_size)
        return _numpy_rtn(weight, bits, encoding.group_size)


__all__ = [
    "QuantizationError",
    "QuantizedCandidate",
    "Quantizer",
    "RTNQuantizer",
]
