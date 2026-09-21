"""Fixed Qwen1.5-MoE loader that never materializes BF16 expert parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .compressed_loader import (
    _cgroup_value,
    _pack_vma_count,
    _resolve_parent,
    _restore_fixed_nonpersistent_buffers,
)
from .contracts import ContractError
from .custom_op import (
    register_expert_pack,
    register_moe_forward,
    unregister_expert_pack,
)
from .expert_pack import ExpertPackReader
from .fixed_models import QWEN1_5_MOE
from .target_pack import MAGIC as TARGET_PACK_MAGIC
from .target_pack import TargetPackReader


PackReader = ExpertPackReader | TargetPackReader


_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:down_proj|gate_proj|up_proj)\.weight$"
)


try:
    import torch
except ImportError:  # pragma: no cover - exercised only without runtime extra
    torch = None


if torch is not None:

    class QuantizedQwenMoEBlock(torch.nn.Module):
        """Qwen router/shared expert plus packed routed experts."""

        def __init__(
            self,
            gate: Any,
            shared_expert: Any,
            shared_expert_gate: Any,
            *,
            layer_index: int,
            pack_handle: int,
            performance_mode: bool,
        ) -> None:
            super().__init__()
            self.gate = gate
            self.shared_expert = shared_expert
            self.shared_expert_gate = shared_expert_gate
            self.layer_index = layer_index
            self.pack_handle = pack_handle
            self.performance_mode = performance_mode

        def forward(self, hidden_states: Any) -> Any:
            import torch.nn.functional as functional

            batch_size, sequence_length, hidden_size = hidden_states.shape
            flattened = hidden_states.reshape(-1, hidden_size)
            shared_output = self.shared_expert(flattened)
            _router_logits, routing_weights, selected_experts = self.gate(flattened)
            expert_output = torch.ops.uma_qmoe.qwen_moe_forward(
                flattened,
                selected_experts,
                routing_weights,
                self.layer_index,
                self.pack_handle,
                self.performance_mode,
            )
            shared_output = functional.sigmoid(
                self.shared_expert_gate(flattened)
            ) * shared_output
            expert_output.add_(shared_output)
            return expert_output.reshape(batch_size, sequence_length, hidden_size)

else:

    class QuantizedQwenMoEBlock:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ContractError("compressed Qwen loading requires PyTorch")


def is_qwen_expert_tensor_name(name: str) -> bool:
    return _EXPERT_WEIGHT.fullmatch(name) is not None


def _partition_qwen_checkpoint_tensors(
    weight_map: Any, required_dense_names: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    if not isinstance(weight_map, dict):
        raise ContractError("fixed Qwen weight_map must be an object")
    by_shard: dict[str, list[str]] = {}
    skipped_expert_names: list[str] = []
    checkpoint_dense_names: set[str] = set()
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str) or not shard:
            raise ContractError("fixed Qwen weight_map entries must be strings")
        if is_qwen_expert_tensor_name(name):
            skipped_expert_names.append(name)
            continue
        checkpoint_dense_names.add(name)
        by_shard.setdefault(shard, []).append(name)
    if checkpoint_dense_names != required_dense_names:
        missing = sorted(required_dense_names - checkpoint_dense_names)
        unexpected = sorted(checkpoint_dense_names - required_dense_names)
        raise ContractError(
            "fixed Qwen dense checkpoint set mismatch: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}"
        )
    return by_shard, skipped_expert_names


@dataclass
class FixedQwenHost:
    model: Any
    expert_pack: PackReader
    pack_handle: int
    evidence: dict[str, Any]

    def close(self) -> None:
        unregister_expert_pack(self.pack_handle)
        self.expert_pack.close()

    def __enter__(self) -> "FixedQwenHost":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def load_fixed_qwen(
    model_directory: str | Path,
    expert_pack_path: str | Path,
    *,
    model_manifest_sha256: str,
    device: str = "cuda:0",
    performance_mode: bool = False,
    target_policy_id: str | None = None,
) -> FixedQwenHost:
    """Load dense Qwen BF16 tensors plus one canonical Q4 ExpertPack mapping."""

    if torch is None:
        raise ContractError("compressed Qwen loading requires PyTorch")
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    model_root = Path(model_directory)
    pack_path = Path(expert_pack_path)
    config = AutoConfig.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    observed = (
        getattr(config, "model_type", None),
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_experts", None),
        getattr(config, "num_experts_per_tok", None),
    )
    if observed != QWEN1_5_MOE.architecture:
        raise ContractError(f"unexpected fixed Qwen architecture {observed!r}")
    try:
        with pack_path.open("rb") as pack_stream:
            magic = pack_stream.read(8)
    except OSError as exc:
        raise ContractError(f"cannot read compressed Qwen pack {pack_path}: {exc}") from exc
    if magic == TARGET_PACK_MAGIC:
        if not target_policy_id:
            raise ContractError("Qwen TargetPack loading requires a policy id")
        reader: PackReader = TargetPackReader(
            pack_path,
            expected_model_id=QWEN1_5_MOE.model_id,
            expected_model_revision=QWEN1_5_MOE.model_revision,
            expected_model_manifest_sha256=model_manifest_sha256,
            expected_policy_id=target_policy_id,
        )
        reader.validate_fixed_qwen_complete()
    else:
        if target_policy_id is not None:
            raise ContractError("Qwen ExpertPack cannot satisfy a TargetPack policy id")
        reader = ExpertPackReader(
            pack_path,
            expected_model_id=QWEN1_5_MOE.model_id,
            expected_model_revision=QWEN1_5_MOE.model_revision,
            expected_model_manifest_sha256=model_manifest_sha256,
        )
    register_moe_forward()
    pack_handle = register_expert_pack(reader)
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, dtype=torch.bfloat16, trust_remote_code=False
            )
        for layer_index, layer in enumerate(model.model.layers):
            original = layer.mlp
            layer.mlp = QuantizedQwenMoEBlock(
                original.gate,
                original.shared_expert,
                original.shared_expert_gate,
                layer_index=layer_index,
                pack_handle=pack_handle,
                performance_mode=performance_mode,
            )
        model.to_empty(device=device)
        _restore_fixed_nonpersistent_buffers(model, config, device)

        index_path = model_root / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ContractError("cannot load fixed Qwen Safetensors index") from exc
        required_dense_names = set(model.state_dict())
        by_shard, skipped_expert_names = _partition_qwen_checkpoint_tensors(
            weight_map, required_dense_names
        )
        loaded: set[str] = set()
        dense_tensor_bytes = 0
        with torch.no_grad():
            for shard in sorted(by_shard):
                with safe_open(
                    model_root / shard, framework="pt", device="cpu"
                ) as source:
                    for name in sorted(by_shard[shard]):
                        value = source.get_tensor(name)
                        if value.dtype != torch.bfloat16:
                            raise ContractError(
                                f"dense tensor {name!r} is not BF16"
                            )
                        parent, leaf = _resolve_parent(model, name)
                        destination = getattr(parent, leaf)
                        if tuple(destination.shape) != tuple(value.shape):
                            raise ContractError(f"dense tensor {name!r} shape mismatch")
                        destination.copy_(value.to(device=device, non_blocking=False))
                        dense_tensor_bytes += value.numel() * value.element_size()
                        loaded.add(name)
                        del value
        if loaded != required_dense_names:
            raise ContractError("not every Qwen dense checkpoint tensor was loaded")
        if len(skipped_expert_names) != 24 * 60 * 3:
            raise ContractError(
                "expected to skip 4320 Qwen expert tensors, skipped "
                f"{len(skipped_expert_names)}"
            )
        expert_parameters = [
            name for name, _ in model.named_parameters() if ".mlp.experts." in name
        ]
        if expert_parameters:
            raise ContractError("compressed Qwen host still contains expert Parameters")
        if any(parameter.is_meta for parameter in model.parameters()):
            raise ContractError("compressed Qwen host contains meta Parameters")
        if any(buffer.is_meta for buffer in model.buffers()):
            raise ContractError("compressed Qwen host contains meta buffers")
        model.eval()

        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        evidence = {
            "model_id": QWEN1_5_MOE.model_id,
            "model_revision": QWEN1_5_MOE.model_revision,
            "device": device,
            "performance_mode": performance_mode,
            "dense_tensor_count": len(loaded),
            "dense_tensor_bytes": dense_tensor_bytes,
            "skipped_expert_tensor_count": len(skipped_expert_names),
            "loaded_expert_tensor_count": 0,
            "expert_parameter_count": 0,
            "quantized_moe_block_count": sum(
                isinstance(layer.mlp, QuantizedQwenMoEBlock)
                for layer in model.model.layers
            ),
            "model_parameter_bytes": parameter_bytes,
            "expert_pack_size_bytes": pack_path.stat().st_size,
            "expert_pack_mapping_count": reader.mapping_count,
            "expert_pack_vma_count": _pack_vma_count(pack_path),
            "cgroup_memory_current_bytes": _cgroup_value("memory.current"),
            "cgroup_memory_peak_bytes": _cgroup_value("memory.peak"),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
        if evidence["quantized_moe_block_count"] != 24:
            raise ContractError("compressed Qwen host did not replace every MoE block")
        if evidence["expert_pack_mapping_count"] != 1:
            raise ContractError("compressed Qwen host mapped ExpertPack more than once")
        if evidence["expert_pack_vma_count"] not in {1, -1}:
            raise ContractError("compressed Qwen host observed multiple ExpertPack VMAs")
        return FixedQwenHost(model, reader, pack_handle, evidence)
    except BaseException:
        unregister_expert_pack(pack_handle)
        reader.close()
        raise


__all__ = [
    "FixedQwenHost",
    "QuantizedQwenMoEBlock",
    "is_qwen_expert_tensor_name",
    "load_fixed_qwen",
]
