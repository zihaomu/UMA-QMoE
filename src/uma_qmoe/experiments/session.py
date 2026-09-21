"""Transactional access to Qwen layer-by-projection research units."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..fixed_models import QWEN1_5_MOE
from .quantizers import QuantizedCandidate
from .types import Unit


class SessionError(RuntimeError):
    pass


class SessionRestoreError(SessionError):
    """Fatal error: the live model can no longer be trusted for another trial."""

    fatal = True


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    source = tensor.detach()
    values = source.float()
    return {
        "shape": [int(value) for value in source.shape],
        "dtype": str(source.dtype),
        "device": str(source.device),
        "element_count": int(source.numel()),
        "sum": float(values.sum().item()),
        "sum_abs": float(values.abs().sum().item()),
    }


@dataclass
class ProjectionTransaction(AbstractContextManager["ProjectionTransaction"]):
    session: "QwenProjectionSession"
    unit: Unit
    target: Any
    snapshot: Any
    original_summary: dict[str, Any]
    applied: bool = False
    restoration: dict[str, Any] | None = None

    @property
    def weight(self) -> Any:
        """An isolated source supplied to quantizers; the restore copy stays private."""

        return self.snapshot.detach().clone()

    def apply(self, candidate: QuantizedCandidate) -> None:
        if self.applied:
            raise SessionError("a transaction candidate may only be applied once")
        restored = candidate.restored_weight
        if tuple(restored.shape) != tuple(self.target.shape):
            raise SessionError("candidate shape does not match the target unit")
        if not bool(self.session.torch.isfinite(restored).all().item()):
            raise SessionError("candidate contains non-finite values")
        with self.session.torch.no_grad():
            self.target.copy_(restored.to(device=self.target.device, dtype=self.target.dtype))
        self.applied = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        restore_error: BaseException | None = None
        try:
            with self.session.torch.no_grad():
                self.target.copy_(self.snapshot)
            equal = bool(self.session.torch.equal(self.target, self.snapshot))
            restored_summary = _tensor_summary(self.target)
            self.restoration = {
                "exact": equal,
                "original": self.original_summary,
                "restored": restored_summary,
            }
            if not equal or restored_summary != self.original_summary:
                raise SessionRestoreError(
                    f"failed to restore unit {self.unit!r} exactly"
                )
        except BaseException as error:  # restoration must be visible over trial failure
            restore_error = error
            self.session._poisoned = f"{type(error).__name__}: {error}"
        finally:
            self.session._active = None
        if restore_error is not None:
            raise restore_error
        return False


class QwenProjectionSession:
    """Discover and transactionally modify 24 x gate/up/down expert projections."""

    projections = ("gate", "up", "down")

    def __init__(self, model: Any, *, torch_module: Any | None = None) -> None:
        if torch_module is None:
            import torch as torch_module

        self.model = model
        self.torch = torch_module
        self._active: Unit | None = None
        self._poisoned: str | None = None
        observed = (
            getattr(model.config, "model_type", None),
            getattr(model.config, "num_hidden_layers", None),
            getattr(model.config, "num_experts", None),
            getattr(model.config, "num_experts_per_tok", None),
        )
        if observed != QWEN1_5_MOE.architecture:
            raise SessionError(f"unexpected Qwen architecture {observed!r}")
        units = self.discover_units()
        if len(units) != 72:
            raise SessionError(f"expected 72 Qwen projection units, found {len(units)}")
        for unit in units:
            self._validate_weight(unit, self._weight(unit))

    def discover_units(self) -> tuple[Unit, ...]:
        return tuple(
            Unit(layer=layer, projection=projection)
            for layer in range(QWEN1_5_MOE.num_layers)
            for projection in self.projections
        )

    def _weight(self, unit: Unit) -> Any:
        if unit.layer >= QWEN1_5_MOE.num_layers:
            raise SessionError(f"layer {unit.layer} is outside the fixed Qwen model")
        try:
            experts = self.model.model.layers[unit.layer].mlp.experts
            if unit.projection == "down":
                value = experts.down_proj
            elif unit.projection in {"gate", "up"}:
                gate_up = experts.gate_up_proj
                size = QWEN1_5_MOE.expert_intermediate_size
                value = gate_up[:, :size, :] if unit.projection == "gate" else gate_up[:, size:, :]
            else:
                raise SessionError(f"unsupported Qwen projection {unit.projection!r}")
            return value if unit.expert is None else value[unit.expert]
        except (AttributeError, IndexError, TypeError) as exc:
            raise SessionError(f"cannot resolve Qwen unit {unit!r}") from exc

    @staticmethod
    def _validate_weight(unit: Unit, weight: Any) -> None:
        expected_rank = 3 if unit.expert is None else 2
        if getattr(weight, "ndim", None) != expected_rank:
            raise SessionError(f"unit {unit!r} has an unexpected tensor rank")
        if unit.expert is None and int(weight.shape[0]) != QWEN1_5_MOE.num_experts:
            raise SessionError(f"unit {unit!r} does not contain all Qwen experts")
        if int(weight.shape[-1]) % 128:
            raise SessionError(f"unit {unit!r} is not group-128 aligned")

    def transaction(self, unit: Unit) -> ProjectionTransaction:
        if self._poisoned is not None:
            raise SessionRestoreError(
                f"Qwen session is poisoned by an earlier restore failure: {self._poisoned}"
            )
        if self._active is not None:
            raise SessionError(f"nested transaction while {self._active!r} is active")
        target = self._weight(unit)
        self._validate_weight(unit, target)
        snapshot = target.detach().clone()
        transaction = ProjectionTransaction(
            session=self,
            unit=unit,
            target=target,
            snapshot=snapshot,
            original_summary=_tensor_summary(snapshot),
        )
        self._active = unit
        return transaction


@dataclass(frozen=True)
class LoadedQwenExperiment:
    model: Any
    tokenizer: Any
    session: QwenProjectionSession
    torch: Any
    backend: str
    device: str

    def reset_peak_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self) -> dict[str, int]:
        return {
            "torch_peak_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
            "torch_peak_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved(self.device)
            ),
        }


def load_qwen_experiment(
    model_path: str | Path,
    *,
    backend: str,
    device: str = "cuda:0",
    seed: int = 0,
) -> LoadedQwenExperiment:
    """Load the fixed BF16 Qwen model and tokenizer exactly once for a scan."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    observed_backend = "hip" if torch.version.hip else "cuda"
    if (
        observed_backend != backend
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
    ):
        raise SessionError(
            f"requested {backend} BF16 device, observed {observed_backend}"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    session = QwenProjectionSession(model, torch_module=torch)
    return LoadedQwenExperiment(
        model=model,
        tokenizer=tokenizer,
        session=session,
        torch=torch,
        backend=observed_backend,
        device=device,
    )


__all__ = [
    "LoadedQwenExperiment",
    "ProjectionTransaction",
    "QwenProjectionSession",
    "SessionError",
    "SessionRestoreError",
    "load_qwen_experiment",
]
