from __future__ import annotations

from types import SimpleNamespace

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.ffd.backend import BackendCapability
from uma_qmoe.ffd.integration import install_qwen_ffd
from uma_qmoe.ffd.policy import FFDPolicy


class _Backend:
    capability = BackendCapability(
        backend_id="unit",
        available=True,
        platform="unit",
        fused_selector_computer=True,
        performance_claim_allowed=False,
    )

    def audit(self):
        return {"unit": True}


def _model(model_type: str = "qwen2_moe"):
    layers = []
    for _ in range(2):
        attention = SimpleNamespace()
        attention.forward = lambda *_args, **_kwargs: None
        layers.append(SimpleNamespace(self_attn=attention))
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            max_position_embeddings=8192,
            num_attention_heads=16,
            num_key_value_heads=16,
        ),
        model=SimpleNamespace(layers=layers),
    )


def test_installation_patches_only_static_layer_set_and_uninstalls() -> None:
    model = _model()
    original = [layer.self_attn.forward for layer in model.model.layers]
    policy = FFDPolicy(
        num_layers=2,
        layer_indices=(1,),
        require_fused_backend=True,
    )
    installation = install_qwen_ffd(model, _Backend(), policy)
    assert installation.audit()["patched_layers"] == [1]
    assert model.model.layers[0].self_attn.forward is original[0]
    assert model.model.layers[1].self_attn.forward is not original[1]
    installation.uninstall()
    assert model.model.layers[1].self_attn.forward is original[1]
    second = install_qwen_ffd(model, _Backend(), policy)
    second.uninstall()


def test_installation_rejects_wrong_model_identity() -> None:
    with pytest.raises(ContractError, match="restricted to Qwen2-MoE"):
        install_qwen_ffd(
            _model("llama"),
            _Backend(),
            FFDPolicy(num_layers=2, layer_indices=(0,)),
        )
