from __future__ import annotations

import json

from probe.hardware import capture_host, parse_system_profiler


def test_system_profiler_parser_allowlists_performance_fields() -> None:
    payload = json.dumps(
        {
            "SPHardwareDataType": [
                {
                    "machine_name": "MacBook Pro",
                    "machine_model": "Mac16,7",
                    "chip_type": "Apple M4 Pro",
                    "number_processors": "proc 14:10:4:0",
                    "serial_number": "must-not-leak",
                    "platform_UUID": "must-not-leak-either",
                }
            ],
            "SPDisplaysDataType": [
                {
                    "sppci_cores": "20",
                    "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
                }
            ],
        }
    )

    parsed = parse_system_profiler(payload)

    assert parsed == {
        "machine_name": "MacBook Pro",
        "model_identifier": "Mac16,7",
        "chip": "Apple M4 Pro",
        "logical_cpu_cores": 14,
        "performance_cpu_cores": 10,
        "efficiency_cpu_cores": 4,
        "gpu_cores": 20,
        "metal_support": "Metal 4",
    }
    assert "must-not-leak" not in json.dumps(parsed)


def test_capture_host_uses_injected_command_reader() -> None:
    profiler = json.dumps(
        {
            "SPHardwareDataType": [
                {
                    "machine_name": "MacBook Pro",
                    "machine_model": "Mac16,7",
                    "chip_type": "Apple M4 Pro",
                    "number_processors": "proc 14:10:4:0",
                }
            ],
            "SPDisplaysDataType": [{"sppci_cores": "20"}],
        }
    )

    def reader(command: list[str]) -> str | None:
        if command[0] == "system_profiler":
            return profiler
        if command[-1] == "hw.memsize":
            return str(48 * 1024**3)
        if command[-1] == "hw.logicalcpu":
            return "14"
        if command[-1] == "hw.perflevel0.physicalcpu":
            return "10"
        if command[-1] == "hw.perflevel1.physicalcpu":
            return "4"
        return None

    host = capture_host(reader=reader)

    assert host.chip == "Apple M4 Pro"
    assert host.memory_bytes == 48 * 1024**3
    assert host.logical_cpu_cores == 14
    assert host.performance_cpu_cores == 10
    assert host.efficiency_cpu_cores == 4
    assert host.gpu_cores == 20


def test_malformed_profiler_payload_returns_no_fields() -> None:
    assert parse_system_profiler("not-json") == {}
    assert parse_system_profiler(None) == {}


def test_system_profiler_parser_handles_current_processor_layout() -> None:
    payload = json.dumps(
        {
            "SPHardwareDataType": [
                {
                    "number_processors": "proc 14:0:10:4",
                }
            ]
        }
    )

    parsed = parse_system_profiler(payload)

    assert parsed["logical_cpu_cores"] == 14
    assert parsed["performance_cpu_cores"] == 10
    assert parsed["efficiency_cpu_cores"] == 4
