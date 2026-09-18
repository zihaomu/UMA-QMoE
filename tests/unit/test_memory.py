from __future__ import annotations

from pathlib import Path

import pytest

from uma_qmoe import memory
from uma_qmoe.contracts import validate_document


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_meminfo_kib_values_are_converted_to_bytes() -> None:
    parsed = memory._parse_meminfo(
        """\
MemTotal:       1024 kB
MemAvailable:    768 kB
SwapTotal:       512 kB
SwapFree:        256 kB
Committed_AS:   2048 kB
"""
    )

    assert parsed == {
        "mem_total_bytes": 1024 * 1024,
        "mem_available_bytes": 768 * 1024,
        "swap_total_bytes": 512 * 1024,
        "swap_free_bytes": 256 * 1024,
        "committed_as_bytes": 2048 * 1024,
    }


def test_snapshot_reports_unlimited_cgroup_without_losing_availability(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    _write(
        proc_root / "meminfo",
        "MemTotal: 100 kB\nMemAvailable: 80 kB\n"
        "SwapTotal: 40 kB\nSwapFree: 30 kB\nCommitted_AS: 50 kB\n",
    )
    _write(proc_root / "vmstat", "pgfault 123\npgmajfault 7\n")
    _write(proc_root / "self" / "status", "Name:\ttest\nVmRSS:\t12 kB\n")
    _write(proc_root / "self" / "cgroup", "0::/workloads/qmoe\n")
    _write(cgroup_root / "workloads" / "qmoe" / "memory.max", "max\n")
    _write(cgroup_root / "workloads" / "qmoe" / "memory.current", "8192\n")

    snapshot = memory.collect_memory_snapshot(
        " halo-test ", proc_root=proc_root, cgroup_root=cgroup_root
    )

    assert snapshot["schema_version"] == 1
    assert snapshot["kind"] == "memory_snapshot"
    assert snapshot["captured_at"].endswith("Z")
    assert snapshot["target_id"] == "halo-test"
    assert snapshot["memory"] == {
        "status": "available",
        "mem_total_bytes": 100 * 1024,
        "mem_available_bytes": 80 * 1024,
        "swap_total_bytes": 40 * 1024,
        "swap_free_bytes": 30 * 1024,
        "committed_as_bytes": 50 * 1024,
    }
    assert snapshot["page_faults"] == {
        "status": "available",
        "major_total": 7,
    }
    assert snapshot["cgroup_v2"] == {
        "status": "available",
        "memory_max_bytes": None,
        "memory_max_is_unlimited": True,
        "memory_current_bytes": 8192,
    }
    assert snapshot["process"] == {
        "status": "available",
        "rss_bytes": 12 * 1024,
    }
    validate_document(snapshot)


def test_missing_fields_are_explicitly_unavailable_or_null(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    _write(
        proc_root / "meminfo",
        "MemTotal: 64 kB\nCommitted_AS: malformed kB\n",
    )
    _write(proc_root / "vmstat", "pgfault 3\n")

    snapshot = memory.collect_memory_snapshot(
        "missing-test", proc_root=proc_root, cgroup_root=cgroup_root
    )

    assert snapshot["memory"] == {
        "status": "partial",
        "mem_total_bytes": 64 * 1024,
        "mem_available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
        "committed_as_bytes": None,
    }
    assert snapshot["page_faults"] == {
        "status": "unavailable",
        "major_total": None,
    }
    assert snapshot["cgroup_v2"] == {
        "status": "unavailable",
        "memory_max_bytes": None,
        "memory_max_is_unlimited": None,
        "memory_current_bytes": None,
    }
    assert snapshot["process"] == {"status": "unavailable", "rss_bytes": None}


@pytest.mark.parametrize("target_id", ["", "   ", None])
def test_target_id_must_be_nonempty(target_id: str | None) -> None:
    with pytest.raises(ValueError, match="target_id"):
        memory.collect_memory_snapshot(target_id)  # type: ignore[arg-type]
