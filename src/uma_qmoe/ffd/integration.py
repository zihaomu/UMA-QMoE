"""Qwen2-MoE attention integration without forking Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

from ..contracts import ContractError
from .backend import FFDBackend
from .cache import FFDCache
from .policy import FFDPolicy


@dataclass
class QwenFFDInstallation:
    model: Any
    backend: FFDBackend
    policy: FFDPolicy
    original_forwards: dict[int, Any]

    def cache(self) -> FFDCache:
        return FFDCache(self.policy)

    def audit(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "backend": self.backend.audit(),
            "patched_layers": sorted(self.original_forwards),
        }

    def uninstall(self) -> None:
        for layer_index, original in self.original_forwards.items():
            attention = self.model.model.layers[layer_index].self_attn
            attention.forward = original
            if getattr(attention, "_uma_ffd_installation", None) is self:
                delattr(attention, "_uma_ffd_installation")
        self.original_forwards.clear()


def _ffd_forward(
    attention: Any,
    hidden_states: Any,
    position_embeddings: tuple[Any, Any] | None = None,
    attention_mask: Any | None = None,
    past_key_values: Any | None = None,
    cache_position: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except ImportError as exc:
        raise ContractError("Qwen FFD integration requires Transformers") from exc

    installation: QwenFFDInstallation = attention._uma_ffd_installation
    policy = installation.policy
    if attention.training:
        raise ContractError("Qwen FFD integration is inference-only")
    if not isinstance(past_key_values, FFDCache):
        raise ContractError(
            "patched Qwen FFD attention requires an explicit FFDCache; "
            "dense fallback is disabled"
        )
    if not policy.applies_to(attention.layer_idx):
        raise ContractError("FFD forward was installed on an unselected layer")
    if position_embeddings is None:
        raise ContractError("Qwen FFD attention requires rotary position embeddings")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    prior_length = past_key_values.get_seq_length(attention.layer_idx)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    query_length = int(query_states.shape[2])
    if query_length > 1:
        if prior_length != 0:
            raise ContractError(
                "FFD does not support chunked prefill after cached tokens"
            )
        # Prefill stays on the configured dense attention interface.  Cache
        # compression happens after projections and does not retain BF16 K.
        past_key_values.update(
            key_states, value_states, attention.layer_idx, cache_kwargs
        )
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation, eager_attention_forward
        )
        output, _ = attention_interface(
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attention.training else attention.attention_dropout,
            scaling=attention.scaling,
            **kwargs,
        )
    elif query_length == 1:
        if prior_length == 0:
            raise ContractError("FFD single-token decode requires a populated cache")
        past_key_values.update(
            key_states, value_states, attention.layer_idx, cache_kwargs
        )
        layer_cache = past_key_values.layer(attention.layer_idx)
        if attention_mask is not None:
            import torch

            valid_mask = attention_mask[..., : layer_cache.get_seq_length()]
            if valid_mask.dtype == torch.bool:
                has_masked_history = not bool(valid_mask.all().item())
            else:
                has_masked_history = bool((valid_mask < 0).any().item())
            if has_masked_history:
                raise ContractError(
                    "Qwen FFD decode does not support padded/masked history"
                )
        output = installation.backend.decode(
            query_states,
            layer_cache,
            policy,
            layer_index=attention.layer_idx,
        ).unsqueeze(1)
    else:
        raise ContractError("Qwen FFD attention received an empty query")

    output = output.reshape(*input_shape, -1).contiguous()
    output = attention.o_proj(output)
    return output, None


def install_qwen_ffd(
    model: Any,
    backend: FFDBackend,
    policy: FFDPolicy,
) -> QwenFFDInstallation:
    """Patch only selected Qwen attention modules and return an audit handle."""

    capability = backend.capability
    if not capability.available:
        raise ContractError(
            f"cannot install unavailable FFD backend: {capability.reason}"
        )
    if policy.require_fused_backend and not capability.fused_selector_computer:
        raise ContractError("FFD policy requires a fused selector-computer backend")
    try:
        layers = model.model.layers
        config = model.config
    except AttributeError as exc:
        raise ContractError("FFD integration expects Qwen2MoeForCausalLM") from exc
    if getattr(config, "model_type", None) != "qwen2_moe":
        raise ContractError("FFD integration is restricted to Qwen2-MoE models")
    if len(layers) != policy.num_layers:
        raise ContractError(
            f"FFD policy expects {policy.num_layers} layers, model has {len(layers)}"
        )
    if int(config.max_position_embeddings) != 8192:
        raise ContractError("FFD model identity requires native 8192 positions")
    if int(config.num_attention_heads) != 16 or int(config.num_key_value_heads) != 16:
        raise ContractError("FFD Qwen identity requires 16 attention and 16 KV heads")
    for layer_index in policy.layer_indices:
        if hasattr(layers[layer_index].self_attn, "_uma_ffd_installation"):
            raise ContractError(f"Qwen layer {layer_index} already has FFD installed")
    installation = QwenFFDInstallation(model, backend, policy, {})
    for layer_index in policy.layer_indices:
        attention = layers[layer_index].self_attn
        installation.original_forwards[layer_index] = attention.forward
        attention._uma_ffd_installation = installation
        attention.forward = MethodType(_ffd_forward, attention)
    return installation


__all__ = ["QwenFFDInstallation", "install_qwen_ffd"]
