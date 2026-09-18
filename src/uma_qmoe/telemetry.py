"""Normalized, allowlisted target telemetry for diagnostic benchmarks."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from typing import Any, Literal

from .contracts import ContractError


_PROBE_TIMEOUT_SECONDS = 5.0


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned or cleaned.casefold() in {"n/a", "na", "none"}:
            return None
        try:
            result = float(cleaned)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _metric_value(container: object, key: str) -> float | None:
    if not isinstance(container, dict):
        return None
    item = container.get(key)
    if isinstance(item, dict):
        return _number(item.get("value"))
    return _number(item)


def _parse_nvidia_smi(text: str) -> dict[str, float | None]:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise ContractError("nvidia-smi telemetry returned an unexpected field count")
    return {
        "temperature_c": _number(fields[0]),
        "socket_power_w": _number(fields[1]),
        "graphics_clock_mhz": _number(fields[2]),
        "memory_clock_mhz": _number(fields[3]),
        "utilization_percent": _number(fields[4]),
    }


def _parse_amd_smi(text: str) -> dict[str, float | None]:
    try:
        document = json.loads(text)
        device = document["gpu_data"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ContractError("amd-smi telemetry returned malformed JSON") from exc
    if not isinstance(device, dict):
        raise ContractError("amd-smi telemetry device must be an object")
    power = device.get("power", {})
    clock = device.get("clock", {})
    temperature = device.get("temperature", {})
    usage = device.get("usage", {})
    return {
        "temperature_c": (
            _metric_value(temperature, "apu_temperature_gfx")
            if _metric_value(temperature, "apu_temperature_gfx") is not None
            else _metric_value(temperature, "edge")
        ),
        "socket_power_w": (
            _metric_value(power, "apu_average_socket_power")
            if _metric_value(power, "apu_average_socket_power") is not None
            else _metric_value(power, "socket_power")
        ),
        "graphics_clock_mhz": _metric_value(
            clock, "apu_average_gfxclk_frequency"
        ),
        "memory_clock_mhz": _metric_value(clock, "apu_average_uclk_frequency"),
        "utilization_percent": (
            _metric_value(usage, "apu_average_gfx_activity")
            if _metric_value(usage, "apu_average_gfx_activity") is not None
            else _metric_value(usage, "gfx_activity")
        ),
    }


class TelemetryProbe:
    """One detected management CLI with a normalized capture method."""

    def __init__(self, provider: Literal["nvidia_smi", "amd_smi"], executable: str):
        self.provider = provider
        self.executable = executable

    @classmethod
    def detect(cls) -> "TelemetryProbe":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is not None:
            probe = cls("nvidia_smi", nvidia_smi)
            probe._capture_values()
            return probe
        amd_smi = shutil.which("amd-smi")
        if amd_smi is not None:
            probe = cls("amd_smi", amd_smi)
            probe._capture_values()
            return probe
        raise ContractError("telemetry requested but neither nvidia-smi nor amd-smi is available")

    def _capture_values(self) -> dict[str, float | None]:
        if self.provider == "nvidia_smi":
            command = [
                self.executable,
                "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            parser = _parse_nvidia_smi
        else:
            command = [
                self.executable,
                "metric",
                "--power",
                "--temperature",
                "--clock",
                "--usage",
                "--json",
            ]
            parser = _parse_amd_smi
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError(f"{self.provider} telemetry execution failed") from exc
        if completed.returncode != 0:
            raise ContractError(f"{self.provider} telemetry returned nonzero exit")
        values = parser(completed.stdout)
        if all(value is None for value in values.values()):
            raise ContractError(f"{self.provider} returned no usable telemetry fields")
        return values

    def capture(
        self,
        *,
        operation_id: str,
        sample_index: int,
        boundary: Literal["before", "after"],
    ) -> dict[str, Any]:
        values = self._capture_values()
        return {
            "operation_id": operation_id,
            "sample_index": sample_index,
            "boundary": boundary,
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }


__all__ = ["TelemetryProbe", "_parse_amd_smi", "_parse_nvidia_smi"]
