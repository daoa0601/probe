from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CommandReader = Callable[[list[str]], str | None]


@dataclass(frozen=True)
class HostSnapshot:
    """Privacy-safe hardware facts needed to interpret a benchmark."""

    captured_at: str
    operating_system: str
    os_version: str
    kernel_release: str
    architecture: str
    machine_name: str | None
    model_identifier: str | None
    chip: str | None
    memory_bytes: int | None
    logical_cpu_cores: int
    performance_cpu_cores: int | None
    efficiency_cpu_cores: int | None
    gpu_cores: int | None
    metal_support: str | None
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeTool:
    name: str
    path: str | None
    version: str | None

    @property
    def available(self) -> bool:
        return self.path is not None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "version": self.version}


def capture_host(*, reader: CommandReader | None = None) -> HostSnapshot:
    """Capture stable performance-relevant facts without retaining machine identifiers."""

    read = reader or _read_command
    profiler = parse_system_profiler(
        read(["system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType"])
    )
    memory_bytes = _parse_int(read(["sysctl", "-n", "hw.memsize"]))
    if memory_bytes is None:
        memory_bytes = _fallback_memory_bytes()
    logical_cores = _parse_int(read(["sysctl", "-n", "hw.logicalcpu"])) or os.cpu_count() or 1
    mac_version = platform.mac_ver()[0]
    operating_system = platform.system()

    return HostSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        operating_system="macOS" if operating_system == "Darwin" else operating_system,
        os_version=mac_version or platform.version(),
        kernel_release=platform.release(),
        architecture=platform.machine(),
        machine_name=_as_string(profiler.get("machine_name")),
        model_identifier=_as_string(profiler.get("model_identifier")),
        chip=_as_string(profiler.get("chip")),
        memory_bytes=memory_bytes,
        logical_cpu_cores=logical_cores,
        performance_cpu_cores=_as_int(profiler.get("performance_cpu_cores")),
        efficiency_cpu_cores=_as_int(profiler.get("efficiency_cpu_cores")),
        gpu_cores=_as_int(profiler.get("gpu_cores")),
        metal_support=_as_string(profiler.get("metal_support")),
        python_version=platform.python_version(),
    )


def parse_system_profiler(payload: str | None) -> dict[str, str | int | None]:
    """Extract an allowlist of fields from system_profiler JSON.

    system_profiler also returns serial numbers and device UUIDs. They are deliberately ignored.
    """

    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}

    hardware = _first_mapping(data.get("SPHardwareDataType"))
    display = _first_mapping(data.get("SPDisplaysDataType"))
    total_cores, performance_cores, efficiency_cores = _parse_processor_counts(
        _as_string(hardware.get("number_processors"))
    )
    metal_raw = _as_string(display.get("spdisplays_mtlgpufamilysupport"))
    metal = metal_raw.replace("spdisplays_metal", "Metal ") if metal_raw else None

    return {
        "machine_name": _as_string(hardware.get("machine_name")),
        "model_identifier": _as_string(hardware.get("machine_model")),
        "chip": _as_string(hardware.get("chip_type")) or _as_string(display.get("sppci_model")),
        "logical_cpu_cores": total_cores,
        "performance_cpu_cores": performance_cores,
        "efficiency_cpu_cores": efficiency_cores,
        "gpu_cores": _parse_int(_as_string(display.get("sppci_cores"))),
        "metal_support": metal,
    }


def discover_runtime_tools(project_root: Path) -> list[RuntimeTool]:
    """Find supported runtimes without starting a server or loading a model."""

    local_bin = project_root / ".cache" / "llama.cpp" / "build" / "bin"
    specifications = (
        ("llama-bench", "PROBE_LLAMA_BENCH", local_bin / "llama-bench"),
        ("llama-server", "PROBE_LLAMA_SERVER", local_bin / "llama-server"),
        ("llama-cli", "PROBE_LLAMA_CLI", local_bin / "llama-cli"),
        ("ollama", "PROBE_OLLAMA", None),
        ("mlx_lm.generate", "PROBE_MLX_GENERATE", None),
    )
    tools: list[RuntimeTool] = []
    for name, environment_key, local_candidate in specifications:
        path = _resolve_executable(name, environment_key, local_candidate)
        tools.append(RuntimeTool(name=name, path=path, version=_tool_version(name, path)))
    return tools


def _resolve_executable(
    name: str, environment_key: str, local_candidate: Path | None
) -> str | None:
    override = os.environ.get(environment_key)
    if override:
        candidate = Path(override).expanduser()
        return (
            str(candidate.resolve())
            if candidate.is_file() and os.access(candidate, os.X_OK)
            else None
        )
    on_path = shutil.which(name)
    if on_path:
        return str(Path(on_path).resolve())
    if (
        local_candidate is not None
        and local_candidate.is_file()
        and os.access(local_candidate, os.X_OK)
    ):
        return str(local_candidate.resolve())
    return None


def _tool_version(name: str, path: str | None) -> str | None:
    if path is None:
        return None
    try:
        completed = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if not output:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    version_lines = [line for line in lines if "version" in line.lower()]
    first_line = (version_lines or lines)[-1]
    first_line = re.sub(r"^warning:\s*(?:client\s+)?", "", first_line, flags=re.IGNORECASE)
    return first_line[:240] if first_line else None


def _read_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _parse_processor_counts(value: str | None) -> tuple[int | None, int | None, int | None]:
    if not value:
        return None, None, None
    match = re.fullmatch(r"proc\s+(\d+):(\d+):(\d+):\d+", value.strip())
    if match is None:
        return None, None, None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _as_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _fallback_memory_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    return pages * page_size


def project_root() -> Path:
    """Return the source checkout root when running editable or from a wheel."""

    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd().resolve()


def python_executable() -> str:
    return sys.executable
