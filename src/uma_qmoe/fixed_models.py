"""Canonical identities and shapes for the two fixed UMA-QMoE research models."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractError


@dataclass(frozen=True)
class FixedModelSpec:
    model_id: str
    model_revision: str
    model_type: str
    num_layers: int
    hidden_size: int
    expert_intermediate_size: int
    num_experts: int
    top_k: int
    trace_schema_version: int
    shared_expert_intermediate_size: int | None = None

    @property
    def identity(self) -> tuple[str, str]:
        return self.model_id, self.model_revision

    @property
    def architecture(self) -> tuple[str, int, int, int]:
        return self.model_type, self.num_layers, self.num_experts, self.top_k


OLMOE = FixedModelSpec(
    model_id="allenai/OLMoE-1B-7B-0125",
    model_revision="9b0c1aa87e34a20052389dce1f0cf01da783f654",
    model_type="olmoe",
    num_layers=16,
    hidden_size=2048,
    expert_intermediate_size=1024,
    num_experts=64,
    top_k=8,
    trace_schema_version=1,
)

QWEN1_5_MOE = FixedModelSpec(
    model_id="Qwen/Qwen1.5-MoE-A2.7B",
    model_revision="1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
    model_type="qwen2_moe",
    num_layers=24,
    hidden_size=2048,
    expert_intermediate_size=1408,
    shared_expert_intermediate_size=5632,
    num_experts=60,
    top_k=4,
    trace_schema_version=2,
)

FIXED_MODELS = {spec.identity: spec for spec in (OLMOE, QWEN1_5_MOE)}


def fixed_model_spec(model_id: str, model_revision: str) -> FixedModelSpec:
    try:
        return FIXED_MODELS[(model_id, model_revision)]
    except KeyError as exc:
        raise ContractError(
            "unsupported fixed model identity: "
            f"{model_id!r} at revision {model_revision!r}"
        ) from exc


__all__ = [
    "FIXED_MODELS",
    "OLMOE",
    "QWEN1_5_MOE",
    "FixedModelSpec",
    "fixed_model_spec",
]
