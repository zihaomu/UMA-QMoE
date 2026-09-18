from __future__ import annotations

from uma_qmoe.telemetry import _parse_amd_smi, _parse_nvidia_smi


def test_parse_nvidia_smi_normalizes_unavailable_fields() -> None:
    assert _parse_nvidia_smi("42, 4.44, 208, [N/A], 0\n") == {
        "temperature_c": 42.0,
        "socket_power_w": 4.44,
        "graphics_clock_mhz": 208.0,
        "memory_clock_mhz": None,
        "utilization_percent": 0.0,
    }


def test_parse_amd_smi_prefers_apu_metrics() -> None:
    document = """
    {
      "gpu_data": [{
        "power": {
          "socket_power": {"value": 8, "unit": "W"},
          "apu_average_socket_power": {"value": 11.5, "unit": "W"}
        },
        "clock": {
          "apu_average_gfxclk_frequency": {"value": 618, "unit": "MHz"},
          "apu_average_uclk_frequency": {"value": 698, "unit": "MHz"}
        },
        "temperature": {
          "edge": {"value": 35, "unit": "C"},
          "apu_temperature_gfx": {"value": 32.0, "unit": "C"}
        },
        "usage": {
          "gfx_activity": {"value": 1, "unit": "%"},
          "apu_average_gfx_activity": {"value": 7, "unit": "%"}
        }
      }]
    }
    """
    assert _parse_amd_smi(document) == {
        "temperature_c": 32.0,
        "socket_power_w": 11.5,
        "graphics_clock_mhz": 618.0,
        "memory_clock_mhz": 698.0,
        "utilization_percent": 7.0,
    }
