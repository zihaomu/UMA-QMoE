from __future__ import annotations

from types import SimpleNamespace

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.expert_pack import (
    _ordered_olmoe_experts,
    _validate_olmoe_expert_tensor,
)


def _weight_map() -> dict[str, str]:
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": "model.safetensors"
        for layer in range(16)
        for expert in range(64)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def test_expert_pack_builder_requires_exact_olmoe_tensor_set() -> None:
    weights = _weight_map()
    ordered = _ordered_olmoe_experts(weights)
    assert len(ordered) == 3072
    assert ordered[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"
    assert ordered[-1][0] == "model.layers.15.mlp.experts.63.down_proj.weight"

    weights.pop(ordered[0][0])
    weights["model.layers.16.mlp.experts.0.gate_proj.weight"] = "model.safetensors"
    with pytest.raises(ContractError, match="tensor set mismatch"):
        _ordered_olmoe_experts(weights)


@pytest.mark.parametrize(
    ("projection", "shape"),
    [
        ("gate_proj", (1024, 2048)),
        ("up_proj", (1024, 2048)),
        ("down_proj", (2048, 1024)),
    ],
)
def test_expert_pack_builder_requires_fixed_shape_and_bf16(
    projection: str, shape: tuple[int, int]
) -> None:
    name = f"model.layers.0.mlp.experts.0.{projection}.weight"
    _validate_olmoe_expert_tensor(
        name, SimpleNamespace(shape=shape, dtype="torch.bfloat16")
    )
    with pytest.raises(ContractError, match="shape mismatch"):
        _validate_olmoe_expert_tensor(
            name, SimpleNamespace(shape=tuple(reversed(shape)), dtype="torch.bfloat16")
        )
    with pytest.raises(ContractError, match="must be BF16"):
        _validate_olmoe_expert_tensor(
            name, SimpleNamespace(shape=shape, dtype="torch.float32")
        )
