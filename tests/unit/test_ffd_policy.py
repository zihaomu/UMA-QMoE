from __future__ import annotations

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.ffd.policy import FFDPolicy


def test_ffd_policy_defaults_are_frozen_qwen_contract() -> None:
    policy = FFDPolicy()
    assert policy.delta == 7.0
    assert policy.key_bits == 2
    assert policy.block_size == 128
    assert policy.max_seq_len == 8192
    assert policy.layer_indices == tuple(range(24))
    assert policy.require_fused_backend is True


def test_ffd_policy_round_trips_and_sorts_static_layers() -> None:
    policy = FFDPolicy.from_dict(
        {
            "policy_id": "test",
            "layer_indices": [23, 12, 13],
            "require_fused_backend": False,
        }
    )
    assert policy.layer_indices == (12, 13, 23)
    assert FFDPolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.parametrize(
    "override",
    [
        {"delta": 0},
        {"key_bits": 3},
        {"block_size": 32},
        {"residual_dtype": "fp16"},
        {"max_seq_len": 8193},
        {"layer_indices": []},
        {"layer_indices": [1, 1]},
        {"layer_indices": [24]},
        {"graph_mode": "full_chain"},
    ],
)
def test_ffd_policy_rejects_out_of_contract_values(override: dict) -> None:
    with pytest.raises(ContractError):
        FFDPolicy(**override)


def test_ffd_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ContractError, match="unknown FFD policy fields"):
        FFDPolicy.from_dict({"delta": 7, "answer_dependent_delta": True})
