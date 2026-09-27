"""Fail-closed backend contract and reference implementation for FFD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import threading
from typing import Any, Protocol

from ..contracts import ContractError
from .cache import FFDLayerCache
from .oracle import q2_sparse_attention
from .policy import FFDPolicy


@dataclass(frozen=True)
class BackendCapability:
    backend_id: str
    available: bool
    platform: str
    fused_selector_computer: bool
    performance_claim_allowed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FFDBackend(Protocol):
    capability: BackendCapability

    def decode(
        self,
        query_states: Any,
        layer_cache: FFDLayerCache,
        policy: FFDPolicy,
        *,
        layer_index: int,
    ) -> Any: ...

    def audit(self) -> dict[str, Any]: ...


class _AuditedBackend:
    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def _record(self, layer_index: int) -> None:
        with self._lock:
            self._counts[layer_index] = self._counts.get(layer_index, 0) + 1

    def audit(self) -> dict[str, Any]:
        with self._lock:
            counts = {str(key): value for key, value in sorted(self._counts.items())}
        return {
            "capability": self.capability.to_dict(),
            "decode_calls_by_layer": counts,
            "total_decode_calls": sum(counts.values()),
        }


class TorchOracleBackend(_AuditedBackend):
    """Slow correctness path; never valid for a performance claim."""

    capability = BackendCapability(
        backend_id="torch-oracle",
        available=True,
        platform="portable_torch",
        fused_selector_computer=False,
        performance_claim_allowed=False,
        reason="materializes dequantized keys and is intended only for correctness",
    )

    def __init__(self, *, explicit_reference_mode: bool = False) -> None:
        super().__init__()
        if not explicit_reference_mode:
            raise ContractError(
                "TorchOracleBackend requires explicit_reference_mode=True; "
                "it is not a silent FFD fallback"
            )

    def decode(
        self,
        query_states: Any,
        layer_cache: FFDLayerCache,
        policy: FFDPolicy,
        *,
        layer_index: int,
    ) -> Any:
        if policy.require_fused_backend:
            raise ContractError(
                "FFD policy requires a fused backend; torch oracle is diagnostic only"
            )
        if layer_cache.full_token_count == 0:
            from .oracle import dense_attention

            tail_keys = layer_cache.tail_keys()
            tail_values = layer_cache.tail_values()
            if tail_keys is None or tail_values is None:
                raise ContractError("FFD cache has no Keys for decode")
            output = dense_attention(query_states, tail_keys, tail_values)
        else:
            output, _ = q2_sparse_attention(
                query_states,
                layer_cache.quantized_blocks(),
                layer_cache.full_values(),
                delta=policy.delta,
                sink_tokens=policy.sink_tokens,
                local_tokens=policy.local_tokens,
                tail_keys=layer_cache.tail_keys(),
                tail_values=layer_cache.tail_values(),
            )
        self._record(layer_index)
        return output.to(query_states.dtype)


class UnavailableBackend(_AuditedBackend):
    """Explicit failure object used when a requested native backend is unavailable."""

    def __init__(self, backend_id: str, reason: str) -> None:
        super().__init__()
        self.capability = BackendCapability(
            backend_id=backend_id,
            available=False,
            platform="unavailable",
            fused_selector_computer=False,
            performance_claim_allowed=False,
            reason=reason,
        )

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ContractError(
            f"FFD backend {self.capability.backend_id!r} is unavailable: "
            f"{self.capability.reason}"
        )


def observed_amd_platform(torch: Any, device: Any = 0) -> str:
    if not torch.cuda.is_available():
        raise ContractError("FFD gfx1151 backend requires a HIP device")
    if not torch.version.hip:
        raise ContractError("FFD gfx1151 backend requires PyTorch HIP")
    properties = torch.cuda.get_device_properties(device)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture != "gfx1151":
        raise ContractError(f"FFD HIP backend requires gfx1151, got {architecture!r}")
    return "hip_gfx1151"


def backend_source_sha256() -> str:
    """Hash project-owned FFD backend sources for evidence binding."""

    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("backend.py", "triton_backend.py"):
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


__all__ = [
    "BackendCapability",
    "FFDBackend",
    "TorchOracleBackend",
    "UnavailableBackend",
    "backend_source_sha256",
    "observed_amd_platform",
]
