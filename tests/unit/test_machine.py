from __future__ import annotations

import json
import subprocess

from uma_qmoe import machine


def test_missing_commands_are_reported_as_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(machine.shutil, "which", lambda _command: None)

    baseline = machine.collect_machine_baseline("test-target")

    assert baseline["schema_version"] == 1
    assert baseline["kind"] == "machine_baseline"
    assert baseline["target_id"] == "test-target"
    assert baseline["captured_at"].endswith("Z")
    assert baseline["software"]["python"]["status"] == "available"
    assert baseline["software"]["docker"] == {
        "status": "unavailable",
        "version": None,
        "reason": "command_not_found",
    }
    assert baseline["software"]["git"] == {
        "status": "unavailable",
        "version": None,
        "reason": "command_not_found",
    }
    assert baseline["software"]["hipconfig"]["status"] == "unavailable"
    assert baseline["software"]["nvcc"]["status"] == "unavailable"
    assert baseline["accelerators"]["nvidia_smi"] == {
        "status": "unavailable",
        "devices": [],
        "reason": "command_not_found",
    }
    assert baseline["accelerators"]["rocminfo"] == {
        "status": "unavailable",
        "devices": [],
        "reason": "command_not_found",
    }


def test_collection_never_serializes_arbitrary_environment(monkeypatch) -> None:
    secret_name = "UMA_QMOE_TEST_SECRET"
    secret_value = "do-not-record-this-token"
    monkeypatch.setenv(secret_name, secret_value)
    monkeypatch.setattr(machine.shutil, "which", lambda _command: None)

    encoded = json.dumps(machine.collect_machine_baseline("safe-target"), sort_keys=True)

    assert secret_name not in encoded
    assert secret_value not in encoded
    assert "environment" not in encoded.casefold()


def test_external_probes_use_argument_lists_without_a_shell(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(
        machine.shutil,
        "which",
        lambda command: f"/probe-bin/{command}",
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        executable = command[0]
        if executable.endswith("nvidia-smi"):
            stdout = "NVIDIA Test GPU, 999.1, 1024, 12.1\n"
        elif executable.endswith("rocminfo"):
            stdout = "Agent 2\n  Name: gfx1151\n  Marketing Name: AMD Test GPU\n  Device Type: GPU\n"
        else:
            stdout = f"{executable.rsplit('/', 1)[-1]} version 1.0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(machine.subprocess, "run", fake_run)

    baseline = machine.collect_machine_baseline("probe-target")

    assert len(calls) == 6
    assert all(call[1]["shell"] is False for call in calls)
    assert all(isinstance(call[0], list) for call in calls)
    assert baseline["accelerators"]["nvidia_smi"]["devices"] == [
        {
            "name": "NVIDIA Test GPU",
            "driver_version": "999.1",
            "compute_capability": "12.1",
            "memory_total_bytes": 1024 * 1024 * 1024,
        }
    ]
    assert baseline["accelerators"]["rocminfo"]["devices"] == [
        {"name": "AMD Test GPU", "architecture": "gfx1151"}
    ]
