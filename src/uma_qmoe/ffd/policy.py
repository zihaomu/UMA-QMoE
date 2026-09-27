"""Validated policy contract for Faster Flash Decoding experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from ..contracts import ContractError


SUPPORTED_KEY_BITS = (2, 4)
SUPPORTED_BLOCK_SIZES = (64, 128, 256)
SUPPORTED_RESIDUAL_DTYPES = ("fp8_e4m3fn", "bf16", "none")
SUPPORTED_GRAPH_MODES = ("eager", "attention_only")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"FFD {name} must be a positive integer")
    return value


def _normalize_layers(values: Iterable[int] | None, num_layers: int) -> tuple[int, ...]:
    if values is None:
        return tuple(range(num_layers))
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("FFD layer indices must be integers")
        if value < 0 or value >= num_layers:
            raise ContractError(
                f"FFD layer index {value} is outside [0, {num_layers - 1}]"
            )
        result.append(value)
    if len(result) != len(set(result)):
        raise ContractError("FFD layer indices must be unique")
    if not result:
        raise ContractError("FFD layer selection cannot be empty")
    return tuple(sorted(result))


@dataclass(frozen=True)
class FFDPolicy:
    """Static, auditable FFD selection and cache policy.

    The policy deliberately excludes prompt- or answer-dependent knobs.  A run
    may select a fixed layer set, but it may not tune delta from live outputs.
    """

    policy_id: str = "qwen-ffd-q2-fp8-delta7-bs128-v1"
    delta: float = 7.0
    key_bits: int = 2
    residual_dtype: str = "fp8_e4m3fn"
    block_size: int = 128
    sink_tokens: int = 128
    local_tokens: int = 128
    max_seq_len: int = 8192
    num_layers: int = 24
    layer_indices: tuple[int, ...] | None = None
    graph_mode: str = "eager"
    require_fused_backend: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ContractError("FFD policy_id must be a non-empty string")
        if isinstance(self.delta, bool) or not isinstance(self.delta, (int, float)):
            raise ContractError("FFD delta must be numeric")
        if not 0.0 < float(self.delta) <= 32.0:
            raise ContractError("FFD delta must be in (0, 32]")
        if self.key_bits not in SUPPORTED_KEY_BITS:
            raise ContractError(f"FFD key_bits must be one of {SUPPORTED_KEY_BITS}")
        if self.residual_dtype not in SUPPORTED_RESIDUAL_DTYPES:
            raise ContractError(
                f"FFD residual_dtype must be one of {SUPPORTED_RESIDUAL_DTYPES}"
            )
        if self.block_size not in SUPPORTED_BLOCK_SIZES:
            raise ContractError(
                f"FFD block_size must be one of {SUPPORTED_BLOCK_SIZES}"
            )
        sink_tokens = _positive_int("sink_tokens", self.sink_tokens)
        local_tokens = _positive_int("local_tokens", self.local_tokens)
        max_seq_len = _positive_int("max_seq_len", self.max_seq_len)
        num_layers = _positive_int("num_layers", self.num_layers)
        if max_seq_len > 8192:
            raise ContractError(
                "Qwen1.5-MoE FFD max_seq_len cannot exceed the native 8192 limit"
            )
        if sink_tokens > max_seq_len or local_tokens > max_seq_len:
            raise ContractError("FFD sink/local windows cannot exceed max_seq_len")
        if self.graph_mode not in SUPPORTED_GRAPH_MODES:
            raise ContractError(
                f"FFD graph_mode must be one of {SUPPORTED_GRAPH_MODES}"
            )
        if not isinstance(self.require_fused_backend, bool):
            raise ContractError("FFD require_fused_backend must be boolean")
        normalized = _normalize_layers(self.layer_indices, num_layers)
        object.__setattr__(self, "layer_indices", normalized)
        object.__setattr__(self, "delta", float(self.delta))

    def applies_to(self, layer_index: int) -> bool:
        return layer_index in self.layer_indices

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer_indices"] = list(self.layer_indices)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FFDPolicy":
        if not isinstance(value, Mapping):
            raise ContractError("FFD policy must be an object")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ContractError(f"unknown FFD policy fields: {unknown!r}")
        payload = dict(value)
        if "layer_indices" in payload and payload["layer_indices"] is not None:
            layers = payload["layer_indices"]
            if isinstance(layers, (str, bytes)) or not isinstance(layers, Iterable):
                raise ContractError("FFD layer_indices must be an array")
            payload["layer_indices"] = tuple(layers)
        return cls(**payload)


__all__ = [
    "FFDPolicy",
    "SUPPORTED_BLOCK_SIZES",
    "SUPPORTED_GRAPH_MODES",
    "SUPPORTED_KEY_BITS",
    "SUPPORTED_RESIDUAL_DTYPES",
]
