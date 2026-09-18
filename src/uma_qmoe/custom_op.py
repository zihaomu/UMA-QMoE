"""PyTorch custom operator boundary for compressed OLMoE expert execution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import threading
from typing import Any, Callable, Iterator

from .contracts import ContractError
from .expert_pack import ExpertPackReader
from .q4 import dequantize_q4


@dataclass
class _PackState:
    reader: ExpertPackReader
    active_calls: int = 0
    closing: bool = False


_PACKS: dict[int, _PackState] = {}
_PACK_CONDITION = threading.Condition()
_PACK_IDS = itertools.count(1)
_PACK_LEASES = threading.local()
_LIBRARIES: list[Any] = []
_REGISTERED = False
_PERFORMANCE_BACKENDS: dict[str, Callable[..., Any]] = {}


def register_expert_pack(reader: ExpertPackReader) -> int:
    if reader.mapping_count != 1:
        raise ContractError("custom operator requires exactly one ExpertPack mapping")
    with _PACK_CONDITION:
        handle = next(_PACK_IDS)
        _PACKS[handle] = _PackState(reader=reader)
    return handle


def unregister_expert_pack(handle: int) -> None:
    leases = getattr(_PACK_LEASES, "counts", {})
    if leases.get(handle, 0):
        raise ContractError("cannot unregister an ExpertPack from its active call")
    with _PACK_CONDITION:
        state = _PACKS.get(handle)
        if state is None:
            return
        if state.closing:
            while handle in _PACKS:
                _PACK_CONDITION.wait()
            return
        state.closing = True
        while state.active_calls:
            _PACK_CONDITION.wait()
        _PACKS.pop(handle, None)
        implementations = tuple(_PERFORMANCE_BACKENDS.values())
        _PACK_CONDITION.notify_all()
    for implementation in implementations:
        release = getattr(implementation, "release_pack", None)
        if callable(release):
            release(handle)


def register_performance_backend(
    platform: str, implementation: Callable[..., Any]
) -> None:
    """Install an explicit optimized backend; never used as an implicit fallback."""

    if platform not in {"cpu", "cuda_sm121", "hip_gfx1151"}:
        raise ContractError(f"unsupported custom operator platform {platform!r}")
    if not callable(implementation):
        raise ContractError("custom operator backend must be callable")
    with _PACK_CONDITION:
        _PERFORMANCE_BACKENDS[platform] = implementation


def _lease_counts() -> dict[int, int]:
    counts = getattr(_PACK_LEASES, "counts", None)
    if counts is None:
        counts = {}
        _PACK_LEASES.counts = counts
    return counts


@contextmanager
def acquire_expert_pack(handle: int) -> Iterator[ExpertPackReader]:
    """Hold one pack open across staging and asynchronous kernel submission."""

    counts = _lease_counts()
    nested = counts.get(handle, 0) > 0
    with _PACK_CONDITION:
        state = _PACKS.get(handle)
        if state is None or (state.closing and not nested):
            raise RuntimeError(f"unknown or closed ExpertPack handle {handle}")
        if not nested:
            state.active_calls += 1
        counts[handle] = counts.get(handle, 0) + 1
    try:
        yield state.reader
    finally:
        with _PACK_CONDITION:
            remaining = counts[handle] - 1
            if remaining:
                counts[handle] = remaining
            else:
                del counts[handle]
                state.active_calls -= 1
                _PACK_CONDITION.notify_all()


def expert_pack_for_handle(handle: int) -> ExpertPackReader:
    """Resolve a registered pack for an explicitly installed native backend."""

    if _lease_counts().get(handle, 0) == 0:
        raise RuntimeError("ExpertPack access requires an active operator lease")
    with _PACK_CONDITION:
        state = _PACKS.get(handle)
        if state is None:
            raise RuntimeError(f"unknown or closed ExpertPack handle {handle}")
        return state.reader


def _platform(torch: Any, hidden_states: Any) -> str:
    if hidden_states.device.type == "cpu":
        return "cpu"
    if hidden_states.device.type != "cuda":
        raise RuntimeError(
            f"uma_qmoe::moe_forward does not support {hidden_states.device}"
        )
    if torch.version.hip:
        properties = torch.cuda.get_device_properties(hidden_states.device)
        architecture = str(getattr(properties, "gcnArchName", ""))
        if "gfx1151" not in architecture:
            raise RuntimeError(
                f"HIP implementation requires gfx1151, observed {architecture!r}"
            )
        return "hip_gfx1151"
    capability = torch.cuda.get_device_capability(hidden_states.device)
    if capability != (12, 1):
        raise RuntimeError(
            f"CUDA implementation requires SM121, observed sm_{capability[0]}{capability[1]}"
        )
    return "cuda_sm121"


def _streaming_reference(
    hidden_states: Any,
    expert_indices: Any,
    routing_weights: Any,
    layer_index: int,
    reader: ExpertPackReader,
) -> Any:
    """Correctness backend with per-expert transient dequantization.

    This path never caches a global dequantized expert.  It is intentionally
    simple and forms the executable Oracle for native CUDA/HIP kernels.
    """

    import torch
    import torch.nn.functional as functional

    if not 0 <= layer_index < 16:
        raise RuntimeError("OLMoE layer_index must be in [0, 15]")
    if hidden_states.shape[-1] != 2048:
        raise RuntimeError("OLMoE hidden size must be 2048")
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    if expert_indices.shape != routing_weights.shape:
        raise RuntimeError("expert_indices and routing_weights shapes must match")
    if expert_indices.ndim != 2 or expert_indices.shape != (flattened.shape[0], 8):
        raise RuntimeError("OLMoE routes must have shape [tokens, 8]")
    if expert_indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("expert_indices must use int32 or int64")
    if torch.any(expert_indices < 0).item() or torch.any(expert_indices >= 64).item():
        raise RuntimeError("expert_indices contains an invalid expert")

    output_dtype = hidden_states.dtype
    compute_dtype = (
        torch.float32 if hidden_states.device.type == "cpu" else output_dtype
    )
    output = torch.zeros_like(flattened, dtype=compute_dtype)
    unique_experts = torch.unique(expert_indices).detach().cpu().tolist()
    for expert_index in unique_experts:
        locations = torch.nonzero(expert_indices == expert_index, as_tuple=False)
        rows = locations[:, 0]
        slots = locations[:, 1]
        prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
        gate_array = dequantize_q4(reader.tensor_q4(f"{prefix}.gate_proj.weight"))
        up_array = dequantize_q4(reader.tensor_q4(f"{prefix}.up_proj.weight"))
        down_array = dequantize_q4(reader.tensor_q4(f"{prefix}.down_proj.weight"))
        gate = torch.from_numpy(gate_array).to(
            device=hidden_states.device, dtype=compute_dtype
        )
        up = torch.from_numpy(up_array).to(
            device=hidden_states.device, dtype=compute_dtype
        )
        down = torch.from_numpy(down_array).to(
            device=hidden_states.device, dtype=compute_dtype
        )
        selected = flattened.index_select(0, rows).to(compute_dtype)
        intermediate = functional.silu(functional.linear(selected, gate))
        intermediate.mul_(functional.linear(selected, up))
        expert_output = functional.linear(intermediate, down)
        weights = routing_weights[rows, slots].to(compute_dtype).unsqueeze(-1)
        output.index_add_(0, rows, expert_output * weights)
        del gate, up, down, gate_array, up_array, down_array
    return output.to(output_dtype).reshape(hidden_states.shape)


def _dispatch(
    hidden_states: Any,
    expert_indices: Any,
    routing_weights: Any,
    layer_index: int,
    pack_handle: int,
    performance_mode: bool = False,
) -> Any:
    import torch

    with acquire_expert_pack(pack_handle) as reader:
        platform = _platform(torch, hidden_states)
        if performance_mode:
            with _PACK_CONDITION:
                implementation = _PERFORMANCE_BACKENDS.get(platform)
            if implementation is None:
                raise RuntimeError(
                    f"performance mode has no registered {platform} backend; "
                    "silent reference fallback is forbidden"
                )
            return implementation(
                hidden_states,
                expert_indices,
                routing_weights,
                layer_index,
                pack_handle,
            )
        return _streaming_reference(
            hidden_states,
            expert_indices,
            routing_weights,
            layer_index,
            reader,
        )


def register_moe_forward() -> None:
    """Register ``uma_qmoe::moe_forward`` once with CPU and CUDA dispatch."""

    global _REGISTERED
    with _PACK_CONDITION:
        if _REGISTERED:
            return
        import torch

        definition = torch.library.Library("uma_qmoe", "DEF")
        definition.define(
            "moe_forward(Tensor hidden_states, Tensor expert_indices, "
            "Tensor routing_weights, int layer_index, int pack_handle, "
            "bool performance_mode=False) -> Tensor"
        )
        cpu = torch.library.Library("uma_qmoe", "IMPL", "CPU")
        cpu.impl("moe_forward", _dispatch)
        cuda = torch.library.Library("uma_qmoe", "IMPL", "CUDA")
        cuda.impl("moe_forward", _dispatch)
        meta = torch.library.Library("uma_qmoe", "IMPL", "Meta")
        meta.impl(
            "moe_forward",
            lambda hidden_states,
            _indices,
            _weights,
            _layer,
            _pack,
            _performance=False: hidden_states.new_empty(hidden_states.shape),
        )
        _LIBRARIES.extend((definition, cpu, cuda, meta))
        _REGISTERED = True


__all__ = [
    "acquire_expert_pack",
    "expert_pack_for_handle",
    "register_expert_pack",
    "register_moe_forward",
    "register_performance_backend",
    "unregister_expert_pack",
]
