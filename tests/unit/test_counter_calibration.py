from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.counter_calibration import calibrate_rocprof_counters


_FIELDNAMES = ("Dispatch_Id", "Kernel_Name", "Counter_Name", "Counter_Value")


def _write_profile(path: Path, *, write_scale: float = 1.0) -> None:
    known = 512 * 1024 * 1024
    rows: list[dict[str, object]] = []

    def add(dispatch: int, kernel: str, read: int, write: int) -> None:
        rows.extend(
            (
                {
                    "Dispatch_Id": dispatch,
                    "Kernel_Name": kernel,
                    "Counter_Name": "GL2C_EA_RDREQ_DRAM_sum",
                    "Counter_Value": f"{read}.000000",
                },
                {
                    "Dispatch_Id": dispatch,
                    "Kernel_Name": kernel,
                    "Counter_Name": "GCEA_WDRAM_SIZE_REQ_sum",
                    "Counter_Value": f"{write}.000000",
                },
            )
        )

    read_count = known // 128
    write_count = round((known // 32) * write_scale)
    add(1, "write_kernel(float4*, unsigned long, float)", 0, write_count)
    for dispatch in (3, 5, 7):
        add(dispatch, "read_reduce_kernel(float4 const*, unsigned long, float*)", read_count, 0)
    for dispatch in (9, 10, 11):
        add(dispatch, "write_kernel(float4*, unsigned long, float)", 0, write_count)
    for dispatch in (12, 13, 14):
        add(dispatch, "copy_kernel(float4 const*, float4*, unsigned long)", read_count, write_count)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _calibrate(path: Path, *, write_scale: float = 1.0) -> dict:
    _write_profile(path, write_scale=write_scale)
    return calibrate_rocprof_counters(
        path,
        source_relative_path="benchmarks/native/native_stream.cu",
        source_file_sha256="a" * 64,
        target_id="halo3",
        architecture="gfx1151",
        profiler_version="rocprofv3 1.3.3",
        known_bytes_per_dispatch=512 * 1024 * 1024,
    )


def test_counter_calibration_normalizes_and_validates(tmp_path: Path) -> None:
    evidence = _calibrate(tmp_path / "counters.csv")
    validate_document(evidence)
    assert evidence["status"] == "passed"
    assert evidence["excluded_dispatches"][0]["dispatch_id"] == 1
    assert all(item["maximum_relative_error"] == 0 for item in evidence["calibrations"])


def test_counter_calibration_records_failed_threshold(tmp_path: Path) -> None:
    evidence = _calibrate(tmp_path / "counters.csv", write_scale=0.75)
    validate_document(evidence)
    assert evidence["status"] == "failed"
    assert [item["status"] for item in evidence["calibrations"]] == [
        "passed",
        "failed",
        "passed",
        "failed",
    ]


def test_counter_calibration_contract_rejects_tampering(tmp_path: Path) -> None:
    evidence = _calibrate(tmp_path / "counters.csv")
    tampered = copy.deepcopy(evidence)
    tampered["calibrations"][0]["measured_bytes"][0] -= 1
    with pytest.raises(ContractError, match="measured bytes"):
        validate_document(tampered)
