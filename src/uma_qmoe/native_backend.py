"""Explicit CUDA/HIP performance backend for canonical packed Q4 experts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import threading
from typing import Any

from .contracts import ContractError
from .custom_op import acquire_expert_pack, register_performance_backend
from .target_pack import TargetPackReader


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


def _stage_native_sources(build_directory: Path) -> tuple[Path, ...]:
    """Copy immutable package sources before Torch/HIP may rewrite ``.cu`` files."""

    staged = []
    for source in _source_paths():
        destination = build_directory / source.name
        shutil.copyfile(source, destination)
        staged.append(destination)
    return tuple(staged)


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
        staged_sources = _stage_native_sources(target_build)

        arch_name = "PYTORCH_ROCM_ARCH" if torch.version.hip else "TORCH_CUDA_ARCH_LIST"
        arch_value = "gfx1151" if torch.version.hip else "12.1a"
        previous = os.environ.get(arch_name)
        os.environ[arch_name] = arch_value
        try:
            extension = load(
                name=f"uma_qmoe_packed_q4_{platform}_v7",
                sources=[str(path) for path in staged_sources],
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


@dataclass(frozen=True)
class _DeviceQ4Layer:
    gate_packed: Any
    gate_scales: Any
    up_packed: Any
    up_scales: Any
    down_packed: Any
    down_scales: Any
    group_size: int

    @property
    def tensors(self) -> tuple[Any, ...]:
        return (
            self.gate_packed,
            self.gate_scales,
            self.up_packed,
            self.up_scales,
            self.down_packed,
            self.down_scales,
        )

    @property
    def device_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


class PackedQ4NativeBackend:
    """Fused W4A16 backend with compressed-only tensor and layer caches."""

    def __init__(self, platform: str, extension: Any) -> None:
        if platform not in {"cuda_sm121", "hip_gfx1151"}:
            raise ContractError(f"unsupported packed Q4 platform {platform!r}")
        missing = [
            name
            for name in ("q4_linear", "q4_moe_forward", "q4_moe_prefill")
            if not hasattr(extension, name)
        ]
        if missing:
            raise ContractError(
                f"packed Q4 extension does not expose {', '.join(missing)}"
            )
        self.platform = platform
        self.extension = extension
        self._cache: dict[tuple[int, str, str], _DeviceQ4Tensor] = {}
        self._layer_cache: dict[tuple[int, int, str], _DeviceQ4Layer] = {}
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
            group_size=(
                128
                if isinstance(reader, TargetPackReader)
                else int(reader.header["quantization"]["group_size"])
            ),
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

    @staticmethod
    def _host_projection(
        reader: Any,
        layer_index: int,
        projection: str,
        expected_shape: tuple[int, int],
        device: Any,
    ) -> tuple[Any, Any]:
        import torch

        packed_chunks = []
        scale_chunks = []
        for expert_index in range(64):
            name = (
                f"model.layers.{layer_index}.mlp.experts.{expert_index}."
                f"{projection}.weight"
            )
            packed_view, scale_view, metadata = reader.tensor_views(name)
            try:
                if tuple(metadata["shape"]) != expected_shape:
                    raise RuntimeError(
                        f"packed Q4 tensor {name!r} has unexpected shape "
                        f"{tuple(metadata['shape'])!r}"
                    )
                packed_chunks.append(
                    torch.frombuffer(packed_view, dtype=torch.uint8).clone()
                )
                scale_chunks.append(
                    torch.frombuffer(scale_view, dtype=torch.float32).clone()
                )
            finally:
                packed_view.release()
                scale_view.release()
        packed = torch.cat(packed_chunks).to(device=device, non_blocking=False)
        scales = torch.cat(scale_chunks).to(device=device, non_blocking=False)
        return packed, scales

    def _device_layer(
        self, pack_handle: int, layer_index: int, device: Any, reader: Any
    ) -> _DeviceQ4Layer:
        key = (pack_handle, layer_index, str(device))
        with self._lock:
            cached = self._layer_cache.get(key)
        if cached is not None:
            return cached
        if not 0 <= layer_index < 16:
            raise RuntimeError("OLMoE layer_index must be in [0, 15]")
        if isinstance(reader, TargetPackReader):
            encoding = reader.layer_encoding(layer_index)
            if encoding != "q4_group128":
                raise RuntimeError(
                    "packed Q4 backend cannot execute TargetPack layer "
                    f"{layer_index} encoded as {encoding}; install the mixed backend"
                )
            group_size = 128
        else:
            group_size = int(reader.header["quantization"]["group_size"])
        if group_size != 128:
            raise RuntimeError(
                f"fused packed Q4 backend requires group size 128, got {group_size}"
            )
        gate_packed, gate_scales = self._host_projection(
            reader, layer_index, "gate_proj", (1024, 2048), device
        )
        up_packed, up_scales = self._host_projection(
            reader, layer_index, "up_proj", (1024, 2048), device
        )
        down_packed, down_scales = self._host_projection(
            reader, layer_index, "down_proj", (2048, 1024), device
        )
        value = _DeviceQ4Layer(
            gate_packed=gate_packed,
            gate_scales=gate_scales,
            up_packed=up_packed,
            up_scales=up_scales,
            down_packed=down_packed,
            down_scales=down_scales,
            group_size=group_size,
        )
        with self._lock:
            existing = self._layer_cache.setdefault(key, value)
        return existing

    def cache_summary(self) -> dict[str, int]:
        with self._lock:
            values = list(self._cache.values())
            layers = list(self._layer_cache.values())
        return {
            "tensor_count": len(values) + 64 * 3 * len(layers),
            "device_storage_bytes": sum(
                item.device_storage_bytes for item in (*values, *layers)
            ),
            "dequantized_weight_bytes": 0,
        }

    def release_pack(self, pack_handle: int) -> None:
        """Release every compressed device tensor owned by a closed pack handle."""

        with self._lock:
            stale = [key for key in self._cache if key[0] == pack_handle]
            for key in stale:
                del self._cache[key]
            stale_layers = [
                key for key in self._layer_cache if key[0] == pack_handle
            ]
            for key in stale_layers:
                del self._layer_cache[key]

    def __call__(
        self,
        hidden_states: Any,
        expert_indices: Any,
        routing_weights: Any,
        layer_index: int,
        pack_handle: int,
    ) -> Any:
        import torch

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
        if expert_indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("expert_indices must use int32 or int64")
        with acquire_expert_pack(pack_handle) as reader:
            layer = self._device_layer(
                pack_handle, layer_index, hidden_states.device, reader
            )
        stream = torch.cuda.current_stream(hidden_states.device)
        for tensor in layer.tensors:
            tensor.record_stream(stream)
        indices = expert_indices.to(dtype=torch.int64).contiguous()
        weights = routing_weights.to(dtype=torch.bfloat16).contiguous()
        arguments = (
            flattened.contiguous(),
            indices,
            weights,
            layer.gate_packed,
            layer.gate_scales,
            layer.up_packed,
            layer.up_scales,
            layer.down_packed,
            layer.down_scales,
            layer.group_size,
        )
        if flattened.shape[0] == 1:
            output = self.extension.q4_moe_forward(*arguments)
        else:
            flattened_indices = indices.reshape(-1)
            route_order = torch.argsort(flattened_indices, stable=True)
            expert_counts = torch.bincount(flattened_indices, minlength=64)
            expert_offsets = torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int64,
                        device=hidden_states.device,
                    ),
                    torch.cumsum(expert_counts, dim=0),
                )
            )
            output = self.extension.q4_moe_prefill(
                arguments[0],
                arguments[1],
                arguments[2],
                route_order,
                expert_offsets,
                *arguments[3:],
            )
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
