from __future__ import annotations

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.fixed_models import OLMOE, QWEN1_5_MOE, fixed_model_spec


def test_fixed_model_registry_freezes_olmoe_and_qwen_shapes() -> None:
    assert fixed_model_spec(*OLMOE.identity) is OLMOE
    assert fixed_model_spec(*QWEN1_5_MOE.identity) is QWEN1_5_MOE
    assert QWEN1_5_MOE.expert_intermediate_size == 1408
    assert QWEN1_5_MOE.shared_expert_intermediate_size == 5632
    assert QWEN1_5_MOE.architecture == ("qwen2_moe", 24, 60, 4)


def test_fixed_model_registry_rejects_floating_or_unknown_identity() -> None:
    with pytest.raises(ContractError, match="unsupported fixed model identity"):
        fixed_model_spec("Qwen/Qwen1.5-MoE-A2.7B", "main")
