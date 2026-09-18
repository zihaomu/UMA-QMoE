"""Format-aware model storage and decode weight-traffic estimates."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .contracts import canonical_sha256


_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\."
)


class WeightTrafficError(ValueError):
    """Raised when a tensor inventory cannot support a trustworthy estimate."""


def _positive_integer(value: int, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WeightTrafficError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise WeightTrafficError(f"{name} must not exceed {maximum}")
    return value


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WeightTrafficError(f"{name} must be a non-negative integer")
    return value


def _element_count(tensor: Mapping[str, Any]) -> int:
    try:
        shape = tensor["shape"]
        recorded_size = tensor["size_bytes"]
    except KeyError as exc:
        raise WeightTrafficError("TensorInventory entry is incomplete") from exc
    if not isinstance(shape, list) or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        raise WeightTrafficError(f"tensor {tensor.get('name')!r} has an invalid shape")
    elements = math.prod(shape)
    if tensor.get("dtype") == "F32" and recorded_size != elements * 4:
        raise WeightTrafficError(
            f"tensor {tensor.get('name')!r} F32 size does not match its shape"
        )
    return elements


def _packed_bytes(
    elements: int,
    *,
    bits: int,
    group_size: int,
    metadata_bytes_per_group: int,
    alignment: int,
) -> tuple[int, int, int]:
    payload = (elements * bits + 7) // 8
    groups = (elements + group_size - 1) // group_size
    metadata = groups * metadata_bytes_per_group
    unaligned = payload + metadata
    aligned = ((unaligned + alignment - 1) // alignment) * alignment
    return payload, metadata, aligned - unaligned


def build_weight_traffic_estimate(
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    model_manifest_path: str,
    tensor_inventory_path: str,
    tensor_inventory_sha256: str,
    dense_bits_per_element: int = 16,
    expert_bits_per_element: int = 4,
    group_size: int = 128,
    scale_bytes_per_group: int = 2,
    zero_point_bytes_per_group: int = 0,
    tensor_alignment_bytes: int = 128,
) -> dict[str, Any]:
    """Estimate packed storage and batch-1 cold weight bytes per decode token.

    Dense tensors use a simple fixed-width representation. Expert tensors use
    bit-packed payloads plus explicit per-group metadata and per-tensor
    alignment. For non-uniform experts, decode traffic selects the largest
    ``top_k`` packed experts in every layer, making the bound conservative.
    """

    dense_bits = _positive_integer(
        dense_bits_per_element, "dense_bits_per_element", maximum=64
    )
    expert_bits = _positive_integer(
        expert_bits_per_element, "expert_bits_per_element", maximum=16
    )
    group_size = _positive_integer(group_size, "group_size")
    scale_bytes = _non_negative_integer(
        scale_bytes_per_group, "scale_bytes_per_group"
    )
    zero_point_bytes = _non_negative_integer(
        zero_point_bytes_per_group, "zero_point_bytes_per_group"
    )
    alignment = _positive_integer(tensor_alignment_bytes, "tensor_alignment_bytes")

    if inventory.get("kind") != "tensor_inventory":
        raise WeightTrafficError("inventory kind must be 'tensor_inventory'")
    if inventory.get("observed_dtypes") != ["F32"]:
        raise WeightTrafficError("initial traffic estimator requires an all-F32 inventory")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise WeightTrafficError("ModelManifest architecture is missing")
    try:
        num_layers = architecture["num_layers"]
        num_experts = architecture["num_experts"]
        top_k = architecture["top_k"]
    except KeyError as exc:
        raise WeightTrafficError("ModelManifest MoE dimensions are incomplete") from exc
    for value, name in (
        (num_layers, "num_layers"),
        (num_experts, "num_experts"),
        (top_k, "top_k"),
    ):
        _positive_integer(value, name)
    if top_k > num_experts:
        raise WeightTrafficError("top_k cannot exceed num_experts")

    expert_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    dense_tensors: list[tuple[str, int, list[int]]] = []
    expert_tensor_count = 0
    for shard in inventory.get("shards", []):
        for tensor in shard.get("tensors", []):
            name = tensor.get("name")
            if not isinstance(name, str) or not name:
                raise WeightTrafficError("TensorInventory tensor name is invalid")
            elements = _element_count(tensor)
            match = _EXPERT_PATTERN.match(name)
            if match is None:
                dense_tensors.append((name, elements, tensor["shape"]))
                continue
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            expert_groups[(layer, expert)].append(elements)
            expert_tensor_count += 1

    expected_groups = {
        (layer, expert)
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    observed_groups = set(expert_groups)
    if observed_groups != expected_groups:
        missing = len(expected_groups.difference(observed_groups))
        unexpected = len(observed_groups.difference(expected_groups))
        raise WeightTrafficError(
            "expert tensor coverage does not match architecture: "
            f"missing_groups={missing}, unexpected_groups={unexpected}"
        )
    if not dense_tensors:
        raise WeightTrafficError("TensorInventory contains no dense tensors")

    metadata_bytes = scale_bytes + zero_point_bytes
    expert_breakdowns: dict[tuple[int, int], tuple[int, int, int]] = {}
    expert_parameter_count = 0
    for key, tensors in expert_groups.items():
        payload = metadata = padding = 0
        for elements in tensors:
            part = _packed_bytes(
                elements,
                bits=expert_bits,
                group_size=group_size,
                metadata_bytes_per_group=metadata_bytes,
                alignment=alignment,
            )
            payload += part[0]
            metadata += part[1]
            padding += part[2]
            expert_parameter_count += elements
        expert_breakdowns[key] = (payload, metadata, padding)

    storage_expert = tuple(
        sum(breakdown[position] for breakdown in expert_breakdowns.values())
        for position in range(3)
    )
    active_expert = [0, 0, 0]
    for layer in range(num_layers):
        layer_experts = sorted(
            (
                expert_breakdowns[(layer, expert)]
                for expert in range(num_experts)
            ),
            key=sum,
            reverse=True,
        )
        for breakdown in layer_experts[:top_k]:
            for position in range(3):
                active_expert[position] += breakdown[position]

    dense_parameter_count = sum(elements for _name, elements, _shape in dense_tensors)
    dense_storage = sum(
        (elements * dense_bits + 7) // 8
        for _name, elements, _shape in dense_tensors
    )
    dense_per_token = 0
    embedding_count = 0
    for name, elements, shape in dense_tensors:
        if name == "model.embed_tokens.weight":
            if len(shape) < 2:
                raise WeightTrafficError("embedding tensor must have at least two dimensions")
            elements = math.prod(shape[1:])
            embedding_count += 1
        dense_per_token += (elements * dense_bits + 7) // 8
    if embedding_count != 1:
        raise WeightTrafficError(
            "exactly one model.embed_tokens.weight tensor is required"
        )

    storage_total = dense_storage + sum(storage_expert)
    per_token_total = dense_per_token + sum(active_expert)
    return {
        "schema_version": 1,
        "kind": "weight_traffic_estimate",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model_manifest_path": model_manifest_path,
        "model_manifest_sha256": canonical_sha256(manifest),
        "tensor_inventory_path": tensor_inventory_path,
        "tensor_inventory_sha256": tensor_inventory_sha256,
        "scenario": "decode_batch1_cold_weight_read",
        "quantization": {
            "dense_bits_per_element": dense_bits,
            "expert_bits_per_element": expert_bits,
            "group_size": group_size,
            "scale_bytes_per_group": scale_bytes,
            "zero_point_bytes_per_group": zero_point_bytes,
            "tensor_alignment_bytes": alignment,
        },
        "model": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "counts": {
            "expert_tensor_count": expert_tensor_count,
            "expert_parameter_count": expert_parameter_count,
            "dense_tensor_count": len(dense_tensors),
            "dense_parameter_count": dense_parameter_count,
        },
        "storage": {
            "dense_weight_bytes": dense_storage,
            "expert_payload_bytes": storage_expert[0],
            "expert_metadata_bytes": storage_expert[1],
            "expert_padding_bytes": storage_expert[2],
            "total_weight_bytes": storage_total,
        },
        "per_token": {
            "expert_selection": "worst_case_top_k_per_layer",
            "embedding_lookup": "one_row_per_token",
            "dense_weight_bytes": dense_per_token,
            "active_expert_payload_bytes": active_expert[0],
            "active_expert_metadata_bytes": active_expert[1],
            "active_expert_padding_bytes": active_expert[2],
            "total_weight_bytes": per_token_total,
        },
        "exclusions": [
            "activation and workspace traffic",
            "attention KV-cache reads and writes",
            "cache reuse and hardware traffic amplification",
            "ExpertPack container headers",
        ],
    }


__all__ = ["WeightTrafficError", "build_weight_traffic_estimate"]
