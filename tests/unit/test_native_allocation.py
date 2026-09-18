from __future__ import annotations

import copy

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.native_allocation import _normalize_capability_output


def _attributes() -> dict[str, int]:
    return {
        "managed_memory": 1,
        "concurrent_managed_access": 1,
        "pageable_memory_access": 1,
        "pageable_memory_access_uses_host_page_tables": 1,
        "direct_managed_memory_access_from_host": 0,
        "host_native_atomic_supported": 1,
        "memory_pools_supported": 1,
    }


def _evidence() -> dict:
    return {
        "schema_version": 1,
        "kind": "native_allocation_capabilities",
        "generated_at": "2026-09-18T03:00:00Z",
        "target_id": "spark1-shanghai-zihaomu",
        "status": "passed",
        "source": {
            "path": "benchmarks/native/allocation_capabilities.cu",
            "sha256": "a" * 64,
        },
        "build": {
            "backend": "cuda",
            "compiler": "nvcc",
            "compiler_version": "Cuda compilation tools, release 13.0",
            "architecture": "sm_121",
            "optimization": "O3",
        },
        "runtime": {
            "device_name": "NVIDIA GB10",
            "total_global_memory_bytes": 128_495_218_688,
            "attributes": _attributes(),
        },
        "limitations": ["capability flags are not performance measurements"],
    }


def test_native_allocation_capability_validates_and_binds_backend() -> None:
    validate_document(_evidence())
    wrong = copy.deepcopy(_evidence())
    wrong["build"]["compiler"] = "hipcc"
    with pytest.raises(ContractError, match="compiler/backend mismatch"):
        validate_document(wrong)


def test_normalize_native_allocation_capability_is_fail_closed() -> None:
    raw = {
        "backend": "cuda",
        "device_name": "NVIDIA GB10",
        "total_global_memory_bytes": 128_495_218_688,
        "attributes": _attributes(),
    }
    normalized = _normalize_capability_output(raw, backend="cuda")
    assert normalized["attributes"]["pageable_memory_access"] == 1

    invalid = copy.deepcopy(raw)
    invalid["attributes"]["managed_memory"] = True
    with pytest.raises(ContractError, match="integer booleans"):
        _normalize_capability_output(invalid, backend="cuda")


def test_hip_capability_requires_vmm_attribute() -> None:
    hip = copy.deepcopy(_evidence())
    hip["build"].update({"backend": "hip", "compiler": "hipcc", "architecture": "gfx1151"})
    with pytest.raises(ContractError, match="required property"):
        validate_document(hip)
