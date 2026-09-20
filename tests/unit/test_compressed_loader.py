from __future__ import annotations

import copy

import pytest

import uma_qmoe.compressed_loader as compressed_loader
from uma_qmoe.compressed_loader import (
    _partition_checkpoint_tensors,
    is_expert_tensor_name,
)
from uma_qmoe.contracts import ContractError, validate_document


def _evidence() -> dict:
    gates = {
        "dense_only_checkpoint_load": True,
        "no_expert_parameters": True,
        "single_pack_mapping": True,
        "no_full_dequantized_copy": True,
        "full_model_forward": True,
        "overall_passed": True,
    }
    return {
        "schema_version": 1,
        "kind": "compressed_loader_evidence",
        "captured_at": "2026-09-18T08:00:00Z",
        "target_id": "halo3",
        "status": "passed",
        "model": {
            "model_id": "allenai/OLMoE-1B-7B-0125",
            "model_revision": "9b0c1aa87e34a20052389dce1f0cf01da783f654",
            "expert_pack_sha256": "a" * 64,
        },
        "loader": {
            "device": "cuda:0",
            "performance_mode": False,
            "dense_tensor_count": 100,
            "dense_tensor_bytes": 1_000_000,
            "skipped_expert_tensor_count": 3072,
            "loaded_expert_tensor_count": 0,
            "expert_parameter_count": 0,
            "quantized_moe_block_count": 16,
            "model_parameter_bytes": 1_000_000,
            "expert_pack_size_bytes": 3_000_000,
            "expert_pack_mapping_count": 1,
            "expert_pack_vma_count": 1,
        },
        "memory": {
            "cgroup_current_bytes": 2_000_000,
            "cgroup_peak_bytes": 3_000_000,
            "torch_allocated_bytes": 1_000_000,
            "torch_reserved_bytes": 1_500_000,
        },
        "smoke": {
            "prompt_id": "general-001",
            "logits_shape": [1, 50304],
            "finite": True,
            "top1_token_id": 7785,
            "logits_sha256": "b" * 64,
        },
        "gates": gates,
    }


def test_expert_tensor_filter_is_exact() -> None:
    assert is_expert_tensor_name(
        "model.layers.15.mlp.experts.63.down_proj.weight"
    )
    assert is_expert_tensor_name(
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert not is_expert_tensor_name("model.layers.0.mlp.gate.weight")
    assert not is_expert_tensor_name("model.layers.0.self_attn.q_proj.weight")
    assert not is_expert_tensor_name("model.layers.0.mlp.experts.0.bias")


def test_dense_checkpoint_partition_requires_exact_model_state() -> None:
    expert = "model.layers.0.mlp.experts.0.gate_proj.weight"
    by_shard, skipped = _partition_checkpoint_tensors(
        {"model.embed_tokens.weight": "model-1.safetensors", expert: "model-2.safetensors"},
        {"model.embed_tokens.weight"},
    )
    assert by_shard == {"model-1.safetensors": ["model.embed_tokens.weight"]}
    assert skipped == [expert]

    with pytest.raises(ContractError, match="dense checkpoint set mismatch"):
        _partition_checkpoint_tensors(
            {expert: "model-2.safetensors"}, {"model.embed_tokens.weight"}
        )

    with pytest.raises(ContractError, match="dense checkpoint set mismatch"):
        _partition_checkpoint_tensors(
            {
                "model.embed_tokens.weight": "model-1.safetensors",
                "unexpected.weight": "model-1.safetensors",
            },
            {"model.embed_tokens.weight"},
        )


def test_compressed_loader_evidence_recomputes_single_copy_gates() -> None:
    document = _evidence()
    validate_document(document)

    tampered = copy.deepcopy(document)
    tampered["loader"]["model_parameter_bytes"] = 4_000_000
    with pytest.raises(ContractError, match="gate"):
        validate_document(tampered)


def test_fixed_loader_rebuilds_nonpersistent_rotary_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    class Scalar:
        def __init__(self, value: bool) -> None:
            self.value = value

        def all(self) -> "Scalar":
            return self

        def item(self) -> bool:
            return self.value

    class Buffer:
        is_meta = False

        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values

    class Rotary:
        def __init__(self, _config: object) -> None:
            self.inv_freq = Buffer((1.0, 0.5))
            self.original_inv_freq = Buffer((1.0, 0.5))

        def to(self, *, device: str) -> "Rotary":
            assert device == "cuda:0"
            return self

    class Inner:
        def __init__(self) -> None:
            self.rotary_emb = Rotary(object())

    class Model:
        def __init__(self) -> None:
            self.model = Inner()

        def state_dict(self) -> dict:
            return {}

        def named_buffers(self):
            rotary = self.model.rotary_emb
            return iter(
                (
                    ("model.rotary_emb.inv_freq", rotary.inv_freq),
                    (
                        "model.rotary_emb.original_inv_freq",
                        rotary.original_inv_freq,
                    ),
                )
            )

    class FakeTorch:
        @staticmethod
        def isfinite(_buffer: Buffer) -> Scalar:
            return Scalar(True)

        @staticmethod
        def equal(left: Buffer, right: Buffer) -> bool:
            return left.values == right.values

    model = Model()
    original = model.model.rotary_emb
    monkeypatch.setattr(compressed_loader, "torch", FakeTorch)

    compressed_loader._restore_fixed_nonpersistent_buffers(
        model, object(), "cuda:0"
    )

    assert model.model.rotary_emb is not original
