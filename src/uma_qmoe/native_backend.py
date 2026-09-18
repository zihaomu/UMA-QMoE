"""Explicit CUDA/HIP performance backend for canonical packed Q4 experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

from .contracts import ContractError
from .custom_op import acquire_expert_pack, register_performance_backend


_SOURCE_NAMES = ("packed_q4_binding.cpp", "packed_q4_kernel.cu")
_BUILD_LOCK = threading.Lock()
_EXTENSIONS: dict[str, Any] = {}


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).with_name("native")
    paths = tuple(root / name for name in _SOURCE_NAMES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ContractError(f"packed Q4 native sources are missing: {missing!r}")
    return paths


def native_kernel_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in _source_paths():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def observed_native_platform(torch: Any, device: Any = 0) -> str:
    if not torch.cuda.is_available():
        raise ContractError("packed Q4 native backend requires a CUDA/HIP device")
    properties = torch.cuda.get_device_properties(device)
    if torch.version.hip:
        architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
        if architecture != "gfx1151":
            raise ContractError(
                f"packed Q4 HIP backend requires gfx1151, got {architecture!r}"
            )
        return "hip_gfx1151"
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise ContractError(
            f"packed Q4 CUDA backend requires SM121, got sm_{capability[0]}{capability[1]}"
        )
    return "cuda_sm121"


def load_native_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> tuple[str, Any]:
    """Compile/load the target-specific extension and return its platform id."""

    import torch
    from torch.utils.cpp_extension import load

    platform = observed_native_platform(torch)
    with _BUILD_LOCK:
        cached = _EXTENSIONS.get(platform)
        if cached is not None:
            return platform, cached
        if build_directory is None:
            configured = os.environ.get("UMA_QMOE_NATIVE_BUILD_ROOT")
            build_root = (
                Path(configured)
                if configured
                else Path.home() / ".cache" / "uma-qmoe" / "native"
            )
            target_build = build_root / platform
        else:
            target_build = Path(build_directory)
        target_build.mkdir(parents=True, exist_ok=True)

        arch_name = "PYTORCH_ROCM_ARCH" if torch.version.hip else "TORCH_CUDA_ARCH_LIST"
        arch_value = "gfx1151" if torch.version.hip else "12.1a"
        previous = os.environ.get(arch_name)
        os.environ[arch_name] = arch_value
        try:
            extension = load(
                name=f"uma_qmoe_packed_q4_{platform}_v1",
                sources=[str(path) for path in _source_paths()],
                build_directory=str(target_build),
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17"],
                with_cuda=True,
                verbose=verbose,
            )
        finally:
            if previous is None:
                os.environ.pop(arch_name, None)
            else:
                os.environ[arch_name] = previous
        _EXTENSIONS[platform] = extension
        return platform, extension


@dataclass(frozen=True)
class _DeviceQ4Tensor:
    packed: Any
    scales: Any
    output_features: int
    input_features: int
    group_size: int

    @property
    def device_storage_bytes(self) -> int:
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


class PackedQ4NativeBackend:
    """Stage-1 W4A16 backend with a compressed-only device tensor cache."""

    def __init__(self, platform: str, extension: Any) -> None:
        if platform not in {"cuda_sm121", "hip_gfx1151"}:
            raise ContractError(f"unsupported packed Q4 platform {platform!r}")
        if not hasattr(extension, "q4_linear"):
            raise ContractError("packed Q4 extension does not expose q4_linear")
        self.platform = platform
        self.extension = extension
        self._cache: dict[tuple[int, str, str], _DeviceQ4Tensor] = {}
        self._lock = threading.Lock()

    @classmethod
    def build_and_register(
        cls,
        *,
        build_directory: str | Path | None = None,
        verbose: bool = False,
    ) -> "PackedQ4NativeBackend":
        platform, extension = load_native_extension(
            build_directory=build_directory, verbose=verbose
        )
        backend = cls(platform, extension)
        register_performance_backend(platform, backend)
        return backend

    def _device_tensor(
        self, pack_handle: int, name: str, device: Any, reader: Any
    ) -> _DeviceQ4Tensor:
        import torch

        key = (pack_handle, name, str(device))
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        packed_view, scale_view, metadata = reader.tensor_views(name)
        try:
            shape = tuple(metadata["shape"])
            if len(shape) != 2:
                raise RuntimeError(f"packed Q4 tensor {name!r} is not a matrix")
            packed_host = torch.frombuffer(packed_view, dtype=torch.uint8).clone()
            scales_host = torch.frombuffer(scale_view, dtype=torch.float32).clone()
        finally:
            packed_view.release()
            scale_view.release()
        value = _DeviceQ4Tensor(
            packed=packed_host.to(device=device, non_blocking=False),
            scales=scales_host.to(device=device, non_blocking=False),
            output_features=int(shape[0]),
            input_features=int(shape[1]),
            group_size=int(reader.header["quantization"]["group_size"]),
        )
        with self._lock:
            existing = self._cache.setdefault(key, value)
        return existing

    def q4_linear(self, input_tensor: Any, pack_handle: int, name: str) -> Any:
        import torch

        with acquire_expert_pack(pack_handle) as reader:
            weight = self._device_tensor(
                pack_handle, name, input_tensor.device, reader
            )
            stream = torch.cuda.current_stream(input_tensor.device)
            weight.packed.record_stream(stream)
            weight.scales.record_stream(stream)
            return self.extension.q4_linear(
                input_tensor.contiguous(),
                weight.packed,
                weight.scales,
                weight.output_features,
                weight.input_features,
                weight.group_size,
            )

    def cache_summary(self) -> dict[str, int]:
        with self._lock:
            values = list(self._cache.values())
        return {
            "tensor_count": len(values),
            "device_storage_bytes": sum(item.device_storage_bytes for item in values),
            "dequantized_weight_bytes": 0,
        }

    def release_pack(self, pack_handle: int) -> None:
        """Release every compressed device tensor owned by a closed pack handle."""

        with self._lock:
            stale = [key for key in self._cache if key[0] == pack_handle]
            for key in stale:
                del self._cache[key]

    def __call__(
        self,
        hidden_states: Any,
        expert_indices: Any,
        routing_weights: Any,
        layer_index: int,
        pack_handle: int,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "packed Q4 performance backend requires BF16 hidden states"
            )
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flattened.shape[-1] != 2048:
            raise RuntimeError(
                "packed Q4 performance backend requires OLMoE hidden size 2048"
            )
        if expert_indices.shape != (flattened.shape[0], 8):
            raise RuntimeError(
                "packed Q4 performance backend requires [tokens, 8] routes"
            )
        output = torch.zeros_like(flattened)
        for expert_index in torch.unique(expert_indices).detach().cpu().tolist():
            locations = torch.nonzero(expert_indices == expert_index, as_tuple=False)
            rows = locations[:, 0]
            slots = locations[:, 1]
            prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
            selected = flattened.index_select(0, rows).contiguous()
            gate = self.q4_linear(selected, pack_handle, f"{prefix}.gate_proj.weight")
            up = self.q4_linear(selected, pack_handle, f"{prefix}.up_proj.weight")
            intermediate = functional.silu(gate)
            intermediate.mul_(up)
            expert_output = self.q4_linear(
                intermediate, pack_handle, f"{prefix}.down_proj.weight"
            )
            weights = routing_weights[rows, slots].to(torch.bfloat16).unsqueeze(-1)
            output.index_add_(0, rows, expert_output * weights)
        return output.reshape(hidden_states.shape)


def install_packed_q4_backend(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> PackedQ4NativeBackend:
    return PackedQ4NativeBackend.build_and_register(
        build_directory=build_directory, verbose=verbose
    )


__all__ = [
    "PackedQ4NativeBackend",
    "install_packed_q4_backend",
    "load_native_extension",
    "native_kernel_source_sha256",
    "observed_native_platform",
]
