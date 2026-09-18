from __future__ import annotations

import copy

import pytest

from uma_qmoe.allocation_matrix import _summarize_transfer
from uma_qmoe.contracts import ContractError, validate_document


def _measurement(size: int) -> dict:
    return {
        "allocation_seconds": 0.01,
        "pretouch_seconds": None,
        "payload_bytes": size,
        "algorithmic_read_bytes": size,
        "algorithmic_write_bytes": size,
        "first_access": {
            "seconds": 0.02,
            "payload_gbps": 1.0,
            "algorithmic_total_gbps": 2.0,
        },
        "steady_state": {
            "samples_seconds": [0.01, 0.011, 0.009],
            "median_seconds": 0.01,
            "p95_seconds": 0.011,
            "mean_seconds": 0.01,
            "coefficient_of_variation": 0.08,
            "payload_gbps": 2.0,
            "algorithmic_total_gbps": 4.0,
        },
        "page_faults": {
            "first_access_minor_delta": 1,
            "first_access_major_delta": 0,
            "steady_state_minor_delta": 0,
            "steady_state_major_delta": 0,
        },
    }


def _evidence() -> dict:
    size = 8 * 1024 * 1024
    measured_ids = (
        "runtime_device_copy",
        "host_pageable_h2d",
        "host_pinned_h2d",
        "host_pinned_d2h",
        "file_mmap_pretouched_h2d",
    )
    cases = [
        {
            "id": case_id,
            "status": "measured",
            "allocation_api": "test",
            "access_path": "test_copy",
            "measurement": _measurement(size),
        }
        for case_id in measured_ids
    ]
    cases.extend(
        {
            "id": case_id,
            "status": "not_measured",
            "allocation_api": "native",
            "access_path": "direct",
            "reason": "not_implemented_v1",
        }
        for case_id in ("managed_unified", "platform_vmm_hmm")
    )
    return {
        "schema_version": 1,
        "kind": "allocation_matrix",
        "generated_at": "2026-09-18T01:00:00Z",
        "target_id": "halo3",
        "status": "diagnostic",
        "measurement_scope": {
            "elapsed_time": "host_monotonic_with_device_synchronize",
            "traffic": "algorithmic_bytes",
            "limitations": ["diagnostic"],
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.14.0",
            "cuda": None,
            "hip": "7.15.0",
            "accelerator_name": "AMD Radeon 8060S Graphics",
        },
        "configuration": {
            "requested_buffer_bytes": size,
            "actual_buffer_bytes": size,
            "dtype": "float32",
            "element_count": size // 4,
            "warmup_iterations": 1,
            "measured_iterations": 3,
            "host_page_size_bytes": 4096,
        },
        "memory": {
            "free_device_before_bytes": size * 10,
            "total_device_bytes": size * 20,
            "free_device_after_bytes": size * 10,
        },
        "system_activity": {"swap_in_pages_delta": 0, "swap_out_pages_delta": 0},
        "cases": cases,
    }


def test_allocation_matrix_evidence_validates() -> None:
    validate_document(_evidence())


def test_allocation_matrix_rejects_missing_case_and_wrong_payload() -> None:
    missing = _evidence()
    missing["cases"].pop()
    with pytest.raises(ContractError):
        validate_document(missing)

    wrong = copy.deepcopy(_evidence())
    wrong["cases"][0]["measurement"]["payload_bytes"] -= 1
    with pytest.raises(ContractError, match="payload_bytes"):
        validate_document(wrong)


def test_transfer_summary_reports_p95_and_both_bandwidth_conventions() -> None:
    summary = _summarize_transfer([1.0, 2.0, 3.0, 4.0], 1_000_000_000, 2_000_000_000)
    assert summary["median_seconds"] == 2.5
    assert summary["p95_seconds"] == 4.0
    assert summary["payload_gbps"] == 0.4
    assert summary["algorithmic_total_gbps"] == 0.8
