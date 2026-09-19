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
                name=f"uma_qmoe_packed_q4_{platform}_v8",
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


@dataclass(frozen=True)
class _DeviceQ8Layer:
    gate_quantized: Any
    gate_scales: Any
    up_quantized: Any
    up_scales: Any
    down_quantized: Any
    down_scales: Any
    group_size: int

    @property
    def tensors(self) -> tuple[Any, ...]:
        return (
            self.gate_quantized,
            self.gate_scales,
            self.up_quantized,
            self.up_scales,
            self.down_quantized,
            self.down_scales,
        )

    @property
    def device_storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


@dataclass(frozen=True)
class _DeviceBF16Layer:
    gate_up: Any
    down: Any

    @property
    def tensors(self) -> tuple[Any, ...]:
        return self.gate_up, self.down

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


class MixedTargetNativeBackend:
    """Native mixed Q4/Q8/BF16 backend for one validated TargetPack policy.

    Q4 and Q8 layers execute directly from their quantized bytes and FP32
    group scales. BF16 layers keep their source representation and use the
    same stable expert/route traversal as the correctness Oracle. No Q8/Q4
    layer is expanded into a persistent BF16/F32 matrix.
    """

    def __init__(self, platform: str, extension: Any) -> None:
        missing = [
            name
            for name in ("q8_moe_forward", "q8_moe_prefill")
            if not hasattr(extension, name)
        ]
        if missing:
            raise ContractError(
                f"mixed TargetPack extension does not expose {', '.join(missing)}"
            )
        self.platform = platform
        self.extension = extension
        self._q4 = PackedQ4NativeBackend(platform, extension)
        self._q8_layers: dict[tuple[int, int, str], _DeviceQ8Layer] = {}
        self._bf16_layers: dict[tuple[int, int, str], _DeviceBF16Layer] = {}
        self._lock = threading.Lock()

    @classmethod
    def build_and_register(
        cls,
        *,
        build_directory: str | Path | None = None,
        verbose: bool = False,
    ) -> "MixedTargetNativeBackend":
        platform, extension = load_native_extension(
            build_directory=build_directory, verbose=verbose
        )
        backend = cls(platform, extension)
        register_performance_backend(platform, backend)
        return backend

    @staticmethod
    def _host_q8_projection(
        reader: TargetPackReader,
        layer_index: int,
        projection: str,
        expected_shape: tuple[int, int],
        device: Any,
    ) -> tuple[Any, Any]:
        import torch

        quantized_chunks = []
        scale_chunks = []
        for expert_index in range(64):
            name = (
                f"model.layers.{layer_index}.mlp.experts.{expert_index}."
                f"{projection}.weight"
            )
            data_view, scale_view, metadata = reader.tensor_views(name)
            try:
                if metadata["encoding"] != "q8_group128":
                    raise RuntimeError(f"TargetPack tensor {name!r} is not Q8")
                if tuple(metadata["shape"]) != expected_shape:
                    raise RuntimeError(
                        f"TargetPack tensor {name!r} has unexpected shape "
                        f"{tuple(metadata['shape'])!r}"
                    )
                quantized_chunks.append(
                    torch.frombuffer(data_view, dtype=torch.int8).clone()
                )
                scale_chunks.append(
                    torch.frombuffer(scale_view, dtype=torch.float32).clone()
                )
            finally:
                data_view.release()
                scale_view.release()
        quantized = torch.cat(quantized_chunks).to(device=device, non_blocking=False)
        scales = torch.cat(scale_chunks).to(device=device, non_blocking=False)
        return quantized, scales

    def _device_q8_layer(
        self,
        pack_handle: int,
        layer_index: int,
        device: Any,
        reader: TargetPackReader,
    ) -> _DeviceQ8Layer:
        key = (pack_handle, layer_index, str(device))
        with self._lock:
            cached = self._q8_layers.get(key)
        if cached is not None:
            return cached
        if reader.layer_encoding(layer_index) != "q8_group128":
            raise RuntimeError(f"TargetPack layer {layer_index} is not Q8")
        gate_quantized, gate_scales = self._host_q8_projection(
            reader, layer_index, "gate_proj", (1024, 2048), device
        )
        up_quantized, up_scales = self._host_q8_projection(
            reader, layer_index, "up_proj", (1024, 2048), device
        )
        down_quantized, down_scales = self._host_q8_projection(
            reader, layer_index, "down_proj", (2048, 1024), device
        )
        value = _DeviceQ8Layer(
            gate_quantized=gate_quantized,
            gate_scales=gate_scales,
            up_quantized=up_quantized,
            up_scales=up_scales,
            down_quantized=down_quantized,
            down_scales=down_scales,
            group_size=128,
        )
        with self._lock:
            existing = self._q8_layers.setdefault(key, value)
        return existing

    @staticmethod
    def _host_bf16_matrix(
        reader: TargetPackReader,
        layer_index: int,
        expert_index: int,
        projection: str,
        expected_shape: tuple[int, int],
    ) -> Any:
        import torch

        name = (
            f"model.layers.{layer_index}.mlp.experts.{expert_index}."
            f"{projection}.weight"
        )
        data_view, scale_view, metadata = reader.tensor_views(name)
        try:
            if metadata["encoding"] != "bf16_le":
                raise RuntimeError(f"TargetPack tensor {name!r} is not BF16")
            if tuple(metadata["shape"]) != expected_shape:
                raise RuntimeError(
                    f"TargetPack tensor {name!r} has unexpected shape "
                    f"{tuple(metadata['shape'])!r}"
                )
            if len(scale_view):
                raise RuntimeError(f"BF16 TargetPack tensor {name!r} has scales")
            return torch.frombuffer(data_view, dtype=torch.bfloat16).clone().reshape(
                expected_shape
            )
        finally:
            data_view.release()
            scale_view.release()

    def _device_bf16_layer(
        self,
        pack_handle: int,
        layer_index: int,
        device: Any,
        reader: TargetPackReader,
    ) -> _DeviceBF16Layer:
        import torch

        key = (pack_handle, layer_index, str(device))
        with self._lock:
            cached = self._bf16_layers.get(key)
        if cached is not None:
            return cached
        if reader.layer_encoding(layer_index) != "bf16_le":
            raise RuntimeError(f"TargetPack layer {layer_index} is not BF16")
        gate_up_experts = []
        down_experts = []
        for expert_index in range(64):
            gate = self._host_bf16_matrix(
                reader, layer_index, expert_index, "gate_proj", (1024, 2048)
            )
            up = self._host_bf16_matrix(
                reader, layer_index, expert_index, "up_proj", (1024, 2048)
            )
            down = self._host_bf16_matrix(
                reader, layer_index, expert_index, "down_proj", (2048, 1024)
            )
            gate_up_experts.append(torch.cat((gate, up), dim=0))
            down_experts.append(down)
        value = _DeviceBF16Layer(
            gate_up=torch.stack(gate_up_experts).to(
                device=device, non_blocking=False
            ),
            down=torch.stack(down_experts).to(device=device, non_blocking=False),
        )
        with self._lock:
            existing = self._bf16_layers.setdefault(key, value)
        return existing

    @staticmethod
    def _grouped_bf16_forward(
        hidden_states: Any,
        expert_indices: Any,
        routing_weights: Any,
        gate_up_weights: Any,
        down_weights: Any,
    ) -> Any:
        import torch
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError(
                "mixed BF16 performance path requires torch._grouped_mm"
            )
        top_k = expert_indices.shape[1]
        sample_weights = routing_weights.reshape(-1)
        expert_ids = expert_indices.reshape(-1)
        expert_ids_grouped, permutation = torch.sort(expert_ids)
        selected_hidden = flattened[permutation // top_k]
        grouped_weights = sample_weights[permutation]
        tokens_per_expert = torch.histc(
            expert_ids_grouped.int(), bins=64, min=0, max=63
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

        gate_up = torch._grouped_mm(
            selected_hidden, gate_up_weights.transpose(-2, -1), offs=offsets
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = torch.nn.functional.silu(gate)
        intermediate.mul_(up)
        projected = torch._grouped_mm(
            intermediate, down_weights.transpose(-2, -1), offs=offsets
        )
        projected.mul_(grouped_weights.unsqueeze(-1))
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(
            permutation.numel(), device=hidden_states.device
        )
        ordered = projected[inverse_permutation]
        return ordered.view(flattened.shape[0], top_k, 2048).sum(dim=1).reshape(
            hidden_states.shape
        )

    @staticmethod
    def _q8_to_bf16(
        quantized: Any,
        scales: Any,
        *,
        output_features: int,
        input_features: int,
    ) -> Any:
        import torch

        grouped = quantized.view(64, output_features, input_features // 128, 128)
        dequantized = grouped.to(dtype=torch.float32)
        dequantized.mul_(
            scales.view(64, output_features, input_features // 128, 1).to(
                dtype=torch.float32
            )
        )
        return dequantized.to(dtype=torch.bfloat16).reshape(
            64, output_features, input_features
        )

    @staticmethod
    def _q4_values(packed: Any) -> Any:
        import torch

        nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(-1)
        values = nibbles.to(dtype=torch.int8)
        values.sub_((values >= 8).to(dtype=torch.int8) * 16)
        return values

    @classmethod
    def _q4_to_bf16(
        cls,
        packed: Any,
        scales: Any,
        *,
        output_features: int,
        input_features: int,
    ) -> Any:
        import torch

        quantized = cls._q4_values(packed)
        grouped = quantized.view(64, output_features, input_features // 128, 128)
        dequantized = grouped.to(dtype=torch.float32)
        dequantized.mul_(
            scales.view(64, output_features, input_features // 128, 1).to(
                dtype=torch.float32
            )
        )
        return dequantized.to(dtype=torch.bfloat16).reshape(
            64, output_features, input_features
        )

    @classmethod
    def _transient_grouped_weights(
        cls,
        layer: _DeviceQ4Layer | _DeviceQ8Layer,
        *,
        encoding: str,
        device: Any,
    ) -> tuple[Any, Any]:
        import torch

        convert = cls._q4_to_bf16 if encoding == "q4_group128" else cls._q8_to_bf16
        gate_up = torch.empty(
            (64, 2048, 2048), dtype=torch.bfloat16, device=device
        )
        gate_up[:, :1024].copy_(
            convert(
                layer.gate_packed
                if isinstance(layer, _DeviceQ4Layer)
                else layer.gate_quantized,
                layer.gate_scales,
                output_features=1024,
                input_features=2048,
            )
        )
        gate_up[:, 1024:].copy_(
            convert(
                layer.up_packed
                if isinstance(layer, _DeviceQ4Layer)
                else layer.up_quantized,
                layer.up_scales,
                output_features=1024,
                input_features=2048,
            )
        )
        down = convert(
            layer.down_packed
            if isinstance(layer, _DeviceQ4Layer)
            else layer.down_quantized,
            layer.down_scales,
            output_features=2048,
            input_features=1024,
        )
        return gate_up, down

    @staticmethod
    def _sorted_routes(
        indices: Any, hidden_states: Any, torch: Any
    ) -> tuple[Any, Any]:
        flattened_indices = indices.reshape(-1)
        route_order = torch.argsort(flattened_indices, stable=True)
        expert_counts = torch.bincount(flattened_indices, minlength=64)
        expert_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=hidden_states.device),
                torch.cumsum(expert_counts, dim=0),
            )
        )
        return route_order, expert_offsets

    def cache_summary(self) -> dict[str, int]:
        q4 = self._q4.cache_summary()
        with self._lock:
            q8_layers = list(self._q8_layers.values())
            bf16_layers = list(self._bf16_layers.values())
        return {
            "tensor_count": (
                q4["tensor_count"] + 192 * (len(q8_layers) + len(bf16_layers))
            ),
            "device_storage_bytes": (
                q4["device_storage_bytes"]
                + sum(layer.device_storage_bytes for layer in q8_layers)
                + sum(layer.device_storage_bytes for layer in bf16_layers)
            ),
            "dequantized_weight_bytes": 0,
        }

    def release_pack(self, pack_handle: int) -> None:
        self._q4.release_pack(pack_handle)
        with self._lock:
            for cache in (self._q8_layers, self._bf16_layers):
                stale = [key for key in cache if key[0] == pack_handle]
                for key in stale:
                    del cache[key]

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
            raise RuntimeError("mixed TargetPack backend requires BF16 hidden states")
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flattened.shape[-1] != 2048:
            raise RuntimeError("mixed TargetPack backend requires hidden size 2048")
        if expert_indices.shape != (flattened.shape[0], 8):
            raise RuntimeError("mixed TargetPack backend requires [tokens, 8] routes")
        if expert_indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("expert_indices must use int32 or int64")
        with acquire_expert_pack(pack_handle) as reader:
            if not isinstance(reader, TargetPackReader):
                raise RuntimeError("mixed backend requires a TargetPack")
            encoding = reader.layer_encoding(layer_index)
            if encoding == "q8_group128":
                layer: _DeviceQ8Layer | _DeviceBF16Layer = self._device_q8_layer(
                    pack_handle, layer_index, hidden_states.device, reader
                )
            elif encoding == "bf16_le":
                layer = self._device_bf16_layer(
                    pack_handle, layer_index, hidden_states.device, reader
                )
            elif encoding == "q4_group128" and flattened.shape[0] > 1:
                layer = self._q4._device_layer(
                    pack_handle, layer_index, hidden_states.device, reader
                )
            elif encoding != "q4_group128":
                raise RuntimeError(
                    f"unsupported TargetPack layer encoding {encoding!r}"
                )
        if encoding == "q4_group128" and flattened.shape[0] == 1:
            return self._q4(
                hidden_states,
                expert_indices,
                routing_weights,
                layer_index,
                pack_handle,
            )
        stream = torch.cuda.current_stream(hidden_states.device)
        for tensor in layer.tensors:
            tensor.record_stream(stream)
        indices = expert_indices.to(dtype=torch.int64).contiguous()
        weights = routing_weights.to(dtype=torch.bfloat16).contiguous()
        if encoding == "bf16_le":
            return self._grouped_bf16_forward(
                hidden_states, indices, weights, layer.gate_up, layer.down
            ).reshape(hidden_states.shape)
        if flattened.shape[0] > 1:
            gate_up, down = self._transient_grouped_weights(
                layer, encoding=encoding, device=hidden_states.device
            )
            return self._grouped_bf16_forward(
                hidden_states, indices, weights, gate_up, down
            ).reshape(hidden_states.shape)
        arguments = (
            flattened.contiguous(),
            indices,
            weights,
            layer.gate_quantized,
            layer.gate_scales,
            layer.up_quantized,
            layer.up_scales,
            layer.down_quantized,
            layer.down_scales,
            layer.group_size,
        )
        if flattened.shape[0] == 1:
            output = self.extension.q8_moe_forward(*arguments)
        else:
            route_order, expert_offsets = self._sorted_routes(
                indices, hidden_states, torch
            )
            output = self.extension.q8_moe_prefill(
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


def install_mixed_target_backend(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> MixedTargetNativeBackend:
    return MixedTargetNativeBackend.build_and_register(
        build_directory=build_directory, verbose=verbose
    )


__all__ = [
    "PackedQ4NativeBackend",
    "MixedTargetNativeBackend",
    "install_mixed_target_backend",
    "install_packed_q4_backend",
    "load_native_extension",
    "native_kernel_source_sha256",
    "observed_native_platform",
]
