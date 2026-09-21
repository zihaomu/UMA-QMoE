from __future__ import annotations

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.qwen_compressed_loader import (
    _partition_qwen_checkpoint_tensors,
    is_qwen_expert_tensor_name,
)


def test_qwen_expert_tensor_filter_is_exact() -> None:
    assert is_qwen_expert_tensor_name(
        "model.layers.23.mlp.experts.59.down_proj.weight"
    )
    assert is_qwen_expert_tensor_name(
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert not is_qwen_expert_tensor_name("model.layers.0.mlp.gate.weight")
    assert not is_qwen_expert_tensor_name("model.layers.0.mlp.experts.gate_up_proj")


def test_qwen_dense_checkpoint_partition_is_exact() -> None:
    expert = "model.layers.0.mlp.experts.0.gate_proj.weight"
    by_shard, skipped = _partition_qwen_checkpoint_tensors(
        {"model.embed_tokens.weight": "model-1.safetensors", expert: "model-2.safetensors"},
        {"model.embed_tokens.weight"},
    )
    assert by_shard == {"model-1.safetensors": ["model.embed_tokens.weight"]}
    assert skipped == [expert]

    with pytest.raises(ContractError, match="dense checkpoint set mismatch"):
        _partition_qwen_checkpoint_tensors(
            {expert: "model-2.safetensors"}, {"model.embed_tokens.weight"}
        )
