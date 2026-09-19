from __future__ import annotations

import threading

import numpy as np
import pytest

import uma_qmoe.custom_op as custom_op
from uma_qmoe.contracts import ContractError
from uma_qmoe.custom_op import (
    acquire_expert_pack,
    expert_pack_for_handle,
    register_expert_pack,
    unregister_expert_pack,
)


class _Reader:
    mapping_count = 1


def test_pack_access_requires_active_lease() -> None:
    reader = _Reader()
    handle = register_expert_pack(reader)  # type: ignore[arg-type]
    try:
        with pytest.raises(RuntimeError, match="active operator lease"):
            expert_pack_for_handle(handle)
        with acquire_expert_pack(handle) as leased:
            assert leased is reader
            assert expert_pack_for_handle(handle) is reader
            with pytest.raises(ContractError, match="active call"):
                unregister_expert_pack(handle)
    finally:
        unregister_expert_pack(handle)


def test_unregister_waits_for_active_call_and_rejects_new_calls() -> None:
    reader = _Reader()
    handle = register_expert_pack(reader)  # type: ignore[arg-type]
    entered = threading.Event()
    release = threading.Event()
    unregistered = threading.Event()

    def active_call() -> None:
        with acquire_expert_pack(handle):
            entered.set()
            assert release.wait(timeout=5)

    def close_pack() -> None:
        unregister_expert_pack(handle)
        unregistered.set()

    worker = threading.Thread(target=active_call)
    closer = threading.Thread(target=close_pack)
    worker.start()
    assert entered.wait(timeout=5)
    closer.start()
    assert not unregistered.wait(timeout=0.1)
    with pytest.raises(RuntimeError, match="unknown or closed"):
        with acquire_expert_pack(handle):
            pass
    release.set()
    worker.join(timeout=5)
    closer.join(timeout=5)
    assert not worker.is_alive()
    assert not closer.is_alive()
    assert unregistered.is_set()


def test_reference_preserves_transformers_fused_gate_up_execution() -> None:
    gate = np.full((2, 3), 1.0, dtype=np.float32)
    up = np.full((2, 3), 2.0, dtype=np.float32)
    combined = custom_op._combine_gate_up_arrays(gate, up)
    assert combined.shape == (4, 3)
    assert np.array_equal(combined[:2], gate)
    assert np.array_equal(combined[2:], up)

    class Value:
        def __init__(self, name: str) -> None:
            self.name = name

        def chunk(self, count: int, *, dim: int) -> tuple["Value", "Value"]:
            assert count == 2
            assert dim == -1
            return Value("gate"), Value("up")

        def mul_(self, other: "Value") -> "Value":
            assert self.name == "silu(gate)"
            assert other.name == "up"
            self.name = "silu(gate)*up"
            return self

    class Functional:
        linear_calls = 0

        @classmethod
        def linear(cls, selected: Value, weight: Value) -> Value:
            assert selected.name == "selected"
            assert weight.name == "gate_up"
            cls.linear_calls += 1
            return Value("gate_up_output")

        @staticmethod
        def silu(value: Value) -> Value:
            assert value.name == "gate"
            return Value("silu(gate)")

    result = custom_op._fused_gate_up(
        Value("selected"), Value("gate_up"), Functional
    )
    assert Functional.linear_calls == 1
    assert result.name == "silu(gate)*up"
