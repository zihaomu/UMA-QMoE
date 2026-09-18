from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.weight_traffic import WeightTrafficError, build_weight_traffic_estimate


def _tensor(name: str, shape: list[int]) -> dict[str, object]:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return {
        "name": name,
        "dtype": "F32",
        "shape": shape,
        "size_bytes": elements * 4,
    }


def _inputs() -> tuple[dict, dict]:
    manifest = {
        "kind": "model_manifest",
        "architecture": {"num_layers": 2, "num_experts": 2, "top_k": 1},
    }
    tensors = [
        _tensor("model.embed_tokens.weight", [10, 4]),
        _tensor("lm_head.weight", [10, 4]),
    ]
    for layer in range(2):
        for expert in range(2):
            tensors.extend(
                [
                    _tensor(
                        f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                        [2, 2],
                    ),
                    _tensor(
                        f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                        [2, 2],
                    ),
                ]
            )
    inventory = {
        "kind": "tensor_inventory",
        "observed_dtypes": ["F32"],
        "shards": [{"tensors": tensors}],
    }
    return manifest, inventory


def test_estimates_storage_and_worst_case_active_expert_traffic() -> None:
    manifest, inventory = _inputs()

    estimate = build_weight_traffic_estimate(
        manifest,
        inventory,
        model_manifest_path="models/manifests/fixture.yaml",
        tensor_inventory_path="models/inventories/fixture.json",
        tensor_inventory_sha256="a" * 64,
        dense_bits_per_element=16,
        expert_bits_per_element=4,
        group_size=4,
        scale_bytes_per_group=2,
        zero_point_bytes_per_group=0,
        tensor_alignment_bytes=4,
    )

    assert estimate["counts"] == {
        "expert_tensor_count": 8,
        "expert_parameter_count": 32,
        "dense_tensor_count": 2,
        "dense_parameter_count": 80,
    }
    assert estimate["storage"] == {
        "dense_weight_bytes": 160,
        "expert_payload_bytes": 16,
        "expert_metadata_bytes": 16,
        "expert_padding_bytes": 0,
        "total_weight_bytes": 192,
    }
    assert estimate["per_token"]["dense_weight_bytes"] == 88
    assert estimate["per_token"]["active_expert_payload_bytes"] == 8
    assert estimate["per_token"]["active_expert_metadata_bytes"] == 8
    assert estimate["per_token"]["total_weight_bytes"] == 104
    validate_document(estimate)


def test_rejects_missing_expert_group_and_non_f32_inventory() -> None:
    manifest, inventory = _inputs()
    del inventory["shards"][0]["tensors"][-2:]
    with pytest.raises(WeightTrafficError, match="coverage"):
        build_weight_traffic_estimate(
            manifest,
            inventory,
            model_manifest_path="models/manifests/fixture.yaml",
            tensor_inventory_path="models/inventories/fixture.json",
            tensor_inventory_sha256="a" * 64,
        )

    _manifest, inventory = _inputs()
    inventory["observed_dtypes"] = ["BF16"]
    with pytest.raises(WeightTrafficError, match="all-F32"):
        build_weight_traffic_estimate(
            manifest,
            inventory,
            model_manifest_path="models/manifests/fixture.yaml",
            tensor_inventory_path="models/inventories/fixture.json",
            tensor_inventory_sha256="a" * 64,
        )


def test_contract_rejects_inconsistent_weight_byte_totals() -> None:
    manifest, inventory = _inputs()
    estimate = build_weight_traffic_estimate(
        manifest,
        inventory,
        model_manifest_path="models/manifests/fixture.yaml",
        tensor_inventory_path="models/inventories/fixture.json",
        tensor_inventory_sha256="a" * 64,
    )
    tampered = copy.deepcopy(estimate)
    tampered["per_token"]["total_weight_bytes"] += 1

    with pytest.raises(ContractError, match="does not match components"):
        validate_document(tampered)
