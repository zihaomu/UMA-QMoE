"""Fixed OLMoE loader that never materializes BF16/F32 expert parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .contracts import ContractError
from .custom_op import (
    register_expert_pack,
    register_moe_forward,
    unregister_expert_pack,
)
from .expert_pack import ExpertPackReader


MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
_EXPERT_WEIGHT = re.compile(
    r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:down_proj|gate_proj|up_proj)\.weight$"
)


try:
    import torch
except ImportError:  # pragma: no cover - exercised only without runtime extra
    torch = None


if torch is not None:

    class QuantizedMoEBlock(torch.nn.Module):
        """OLMoE router plus a compressed custom-op expert implementation."""

        def __init__(
            self,
            gate: Any,
            *,
            layer_index: int,
            pack_handle: int,
            performance_mode: bool,
        ) -> None:
            super().__init__()
            self.gate = gate
            self.layer_index = layer_index
            self.pack_handle = pack_handle
            self.performance_mode = performance_mode

        def forward(self, hidden_states: Any) -> Any:
            _router_logits, routing_weights, selected_experts = self.gate(hidden_states)
            return torch.ops.uma_qmoe.moe_forward(
                hidden_states,
                selected_experts,
                routing_weights,
                self.layer_index,
                self.pack_handle,
                self.performance_mode,
            )

else:

    class QuantizedMoEBlock:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ContractError("compressed OLMoE loading requires PyTorch")


def is_expert_tensor_name(name: str) -> bool:
    return _EXPERT_WEIGHT.fullmatch(name) is not None


def _resolve_parent(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _partition_checkpoint_tensors(
    weight_map: Any, required_dense_names: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Require an exact dense state and partition it without opening experts."""

    if not isinstance(weight_map, dict):
        raise ContractError("fixed OLMoE weight_map must be an object")
    by_shard: dict[str, list[str]] = {}
    skipped_expert_names: list[str] = []
    checkpoint_dense_names: set[str] = set()
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str) or not shard:
            raise ContractError("fixed OLMoE weight_map entries must be strings")
        if is_expert_tensor_name(name):
            skipped_expert_names.append(name)
            continue
        checkpoint_dense_names.add(name)
        by_shard.setdefault(shard, []).append(name)
    if checkpoint_dense_names != required_dense_names:
        missing = sorted(required_dense_names - checkpoint_dense_names)
        unexpected = sorted(checkpoint_dense_names - required_dense_names)
        raise ContractError(
            "fixed OLMoE dense checkpoint set mismatch: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}"
        )
    return by_shard, skipped_expert_names


def _cgroup_value(name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / name
    try:
        text = path.read_text(encoding="utf-8").strip()
        return None if text == "max" else int(text)
    except (OSError, ValueError):
        return None


def _pack_vma_count(path: Path) -> int:
    resolved = str(path.resolve())
    try:
        return sum(
            resolved in line
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return -1


def _restore_fixed_nonpersistent_buffers(model: Any, config: Any, device: str) -> None:
    """Recreate deterministic buffers that ``to_empty`` cannot initialize.

    OLMoE's rotary frequencies are intentionally absent from ``state_dict``.
    Creating the model on ``meta`` and then calling ``to_empty`` therefore
    leaves them as uninitialized storage unless the tiny rotary module is
    reconstructed after materialization.
    """

    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config).to(device=device)
    persistent_names = set(model.state_dict())
    nonpersistent = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name not in persistent_names
    }
    expected = {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    if set(nonpersistent) != expected:
        raise ContractError(
            "fixed OLMoE non-persistent buffer set changed: "
            f"{sorted(nonpersistent)!r}"
        )
    if any(buffer.is_meta or not torch.isfinite(buffer).all().item() for buffer in nonpersistent.values()):
        raise ContractError("fixed OLMoE non-persistent buffers are not initialized")
    if not torch.equal(
        nonpersistent["model.rotary_emb.inv_freq"],
        nonpersistent["model.rotary_emb.original_inv_freq"],
    ):
        raise ContractError("fixed OLMoE rotary frequency buffers disagree at load time")


@dataclass
class FixedOlmoeHost:
    model: Any
    expert_pack: ExpertPackReader
    pack_handle: int
    evidence: dict[str, Any]

    def close(self) -> None:
        unregister_expert_pack(self.pack_handle)
        self.expert_pack.close()

    def __enter__(self) -> "FixedOlmoeHost":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def load_fixed_olmoe(
    model_directory: str | Path,
    expert_pack_path: str | Path,
    *,
    model_manifest_sha256: str,
    device: str = "cuda:0",
    performance_mode: bool = False,
) -> FixedOlmoeHost:
    """Load dense BF16 tensors and one ExpertPack mapping, skipping experts pre-read."""

    if torch is None:
        raise ContractError("compressed OLMoE loading requires PyTorch")
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
    if observed != ("olmoe", 16, 64, 8):
        raise ContractError(f"unexpected fixed OLMoE architecture {observed!r}")

    reader = ExpertPackReader(
        pack_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
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
            layer.mlp = QuantizedMoEBlock(
                original.gate,
                layer_index=layer_index,
                pack_handle=pack_handle,
                performance_mode=performance_mode,
            )
        # Only dense modules and the small routers exist at this point.  This
        # materializes their storage; expert Parameters have already vanished.
        model.to_empty(device=device)
        _restore_fixed_nonpersistent_buffers(model, config, device)

        index_path = model_root / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ContractError("cannot load fixed OLMoE Safetensors index") from exc
        parameter_names = set(dict(model.named_parameters()))
        required_dense_names = set(model.state_dict())
        loaded: set[str] = set()
        dense_tensor_bytes = 0
        by_shard, skipped_expert_names = _partition_checkpoint_tensors(
            weight_map, required_dense_names
        )

        with torch.no_grad():
            for shard in sorted(by_shard):
                with safe_open(model_root / shard, framework="pt", device="cpu") as source:
                    for name in sorted(by_shard[shard]):
                        value = source.get_tensor(name)
                        if value.dtype != torch.bfloat16:
                            raise ContractError(
                                f"dense tensor {name!r} is not BF16 in the Oracle derivation"
                            )
                        parent, leaf = _resolve_parent(model, name)
                        if name in parameter_names:
                            parameter = getattr(parent, leaf)
                            if tuple(parameter.shape) != tuple(value.shape):
                                raise ContractError(f"dense tensor {name!r} shape mismatch")
                            parameter.copy_(value.to(device=device, non_blocking=False))
                        else:
                            buffer = getattr(parent, leaf)
                            if tuple(buffer.shape) != tuple(value.shape):
                                raise ContractError(f"dense buffer {name!r} shape mismatch")
                            buffer.copy_(value.to(device=device, non_blocking=False))
                        dense_tensor_bytes += value.numel() * value.element_size()
                        loaded.add(name)
                        del value
        if loaded != required_dense_names:
            raise ContractError("not every dense checkpoint tensor was loaded")
        if len(skipped_expert_names) != 16 * 64 * 3:
            raise ContractError(
                f"expected to skip 3072 expert tensors, skipped {len(skipped_expert_names)}"
            )
        expert_parameters = [
            name for name, _ in model.named_parameters() if ".mlp.experts." in name
        ]
        if expert_parameters:
            raise ContractError("compressed host still contains expert Parameters")
        meta_parameters = [
            name for name, parameter in model.named_parameters() if parameter.is_meta
        ]
        if meta_parameters:
            raise ContractError("compressed host contains unmaterialized dense Parameters")
        meta_buffers = [
            name for name, buffer in model.named_buffers() if buffer.is_meta
        ]
        if meta_buffers:
            raise ContractError("compressed host contains unmaterialized dense buffers")
        model.eval()

        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        evidence = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": device,
            "performance_mode": performance_mode,
            "dense_tensor_count": len(loaded),
            "dense_tensor_bytes": dense_tensor_bytes,
            "skipped_expert_tensor_count": len(skipped_expert_names),
            "loaded_expert_tensor_count": 0,
            "expert_parameter_count": 0,
            "quantized_moe_block_count": sum(
                isinstance(layer.mlp, QuantizedMoEBlock)
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
        if evidence["quantized_moe_block_count"] != 16:
            raise ContractError("compressed host did not replace every MoE block")
        if evidence["expert_pack_mapping_count"] != 1:
            raise ContractError("compressed host mapped ExpertPack more than once")
        if evidence["expert_pack_vma_count"] not in {1, -1}:
            raise ContractError("compressed host observed multiple ExpertPack VMAs")
        return FixedOlmoeHost(model, reader, pack_handle, evidence)
    except BaseException:
        unregister_expert_pack(pack_handle)
        reader.close()
        raise


__all__ = [
    "FixedOlmoeHost",
    "QuantizedMoEBlock",
    "is_expert_tensor_name",
    "load_fixed_olmoe",
]
