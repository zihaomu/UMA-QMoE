from __future__ import annotations

import pytest

from uma_qmoe.contracts import ContractError
from uma_qmoe.ffd.backend import (
    TorchOracleBackend,
    UnavailableBackend,
    backend_source_sha256,
)


def test_reference_backend_requires_explicit_opt_in() -> None:
    with pytest.raises(ContractError, match="explicit_reference_mode"):
        TorchOracleBackend()
    backend = TorchOracleBackend(explicit_reference_mode=True)
    assert backend.capability.performance_claim_allowed is False
    assert backend.audit()["total_decode_calls"] == 0


def test_unavailable_backend_fails_closed() -> None:
    backend = UnavailableBackend("test-native", "compiler missing")
    with pytest.raises(ContractError, match="compiler missing"):
        backend.decode(None)


def test_backend_source_hash_is_stable_digest() -> None:
    digest = backend_source_sha256()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
