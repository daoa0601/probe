from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from probe.config import ProbeConfig, build_command, display_command
from probe.hardware import capture_host, project_root


class BenchmarkError(RuntimeError):
    """Raised after a failed benchmark has been preserved as an artifact."""

    def __init__(self, message: str, artifact_dir: Path | None = None) -> None:
        super().__init__(message)
        self.artifact_dir = artifact_dir


@dataclass(frozen=True)
class RunArtifact:
    run_id: str
    directory: Path
    records: int
    wall_seconds: float
    peak_rss_bytes: int | None


def resolve_binary(config: ProbeConfig, *, root: Path | None = None) -> str:
    """Resolve llama-bench from an override, PATH, or Probe's pinned local build."""

    repository_root = (root or project_root()).resolve()
    configured = os.environ.get("PROBE_LLAMA_BENCH") or config.runtime.binary
    explicit = Path(configured).expanduser()
    if explicit.parent != Path("."):
        if explicit.is_file() and os.access(explicit, os.X_OK):
            return str(explicit.resolve())
        raise BenchmarkError(f"llama-bench is not executable: {explicit}")

    on_path = shutil.which(configured)
    if on_path:
        return str(Path(on_path).resolve())
    local = repository_root / ".cache" / "llama.cpp" / "build" / "bin" / configured
    if local.is_file() and os.access(local, os.X_OK):
        return str(local.resolve())
    raise BenchmarkError(
        f"could not find {configured!r}; run ./scripts/bootstrap-llama-cpp.sh or set "
        "PROBE_LLAMA_BENCH"
    )


def run_sweep(
    config: ProbeConfig,
    *,
    output_root: Path | None = None,
    root: Path | None = None,
) -> RunArtifact:
    """Run a complete llama-bench matrix and preserve raw and normalized output."""

    repository_root = (root or project_root()).resolve()
    binary = resolve_binary(config, root=repository_root)
    _validate_local_model(config, repository_root)

    started_at = datetime.now(UTC)
    run_id = _run_id(config.experiment.name, started_at)
    configured_output = config.experiment.output_dir
    base_output = output_root or (
        configured_output
        if configured_output.is_absolute()
        else repository_root / configured_output
    )
    artifact_dir = base_output.expanduser().resolve() / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    host = capture_host().to_dict()
    command = build_command(config, binary=binary)
    execution_command = _with_memory_capture(command, config.runtime.capture_peak_memory)
    _write_text_atomic(artifact_dir / "command.txt", display_command(command) + "\n")
    _write_text_atomic(artifact_dir / "config.toml", config.source.read_text(encoding="utf-8"))
    _write_json_atomic(artifact_dir / "host.json", host)

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    monotonic_start = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            execution_command,
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.runtime.timeout_seconds,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        stderr += f"\nprobe: benchmark timed out after {config.runtime.timeout_seconds}s\n"
    wall_seconds = time.perf_counter() - monotonic_start
    peak_rss_bytes = parse_peak_rss(stderr)

    _write_text_atomic(artifact_dir / "stdout.json", stdout)
    _write_text_atomic(artifact_dir / "stderr.log", stderr)

    rows: list[dict[str, Any]] = []
    parse_error: str | None = None
    if return_code == 0:
        try:
            decoded = json.loads(stdout)
            if not isinstance(decoded, list) or any(not isinstance(row, dict) for row in decoded):
                raise ValueError("llama-bench JSON must be an array of objects")
            rows = decoded
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)

    status = "succeeded" if return_code == 0 and parse_error is None else "failed"
    model_source = config.model.path or config.model.hf_repo
    execution = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "command": command,
        "binary": binary,
        "return_code": return_code,
    }
    records = [
        {
            "schema_version": 1,
            "experiment": {
                "name": config.experiment.name,
                "description": config.experiment.description,
                "tags": list(config.experiment.tags),
            },
            "model": {
                "label": config.model.label,
                "source": model_source,
                "file": config.model.hf_file,
                "quantization": config.model.quantization,
            },
            "host": host,
            "execution": execution,
            "benchmark": row,
        }
        for row in rows
    ]
    _write_jsonl_atomic(artifact_dir / "results.jsonl", records)
    manifest = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "return_code": return_code,
        "timed_out": timed_out,
        "parse_error": parse_error,
        "result_count": len(records),
        "files": {
            "config": "config.toml",
            "host": "host.json",
            "command": "command.txt",
            "stdout": "stdout.json",
            "stderr": "stderr.log",
            "results": "results.jsonl",
        },
    }
    _write_json_atomic(artifact_dir / "manifest.json", manifest)

    if status != "succeeded":
        reason = parse_error or f"llama-bench exited with status {return_code}"
        raise BenchmarkError(f"benchmark failed: {reason}", artifact_dir)
    return RunArtifact(
        run_id=run_id,
        directory=artifact_dir,
        records=len(records),
        wall_seconds=wall_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )


def parse_peak_rss(stderr: str) -> int | None:
    """Parse macOS `/usr/bin/time -l` peak resident memory output (already in bytes)."""

    match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", stderr)
    return int(match.group(1)) if match else None


def _with_memory_capture(command: list[str], enabled: bool) -> list[str]:
    time_binary = Path("/usr/bin/time")
    if enabled and platform.system() == "Darwin" and time_binary.is_file():
        return [str(time_binary), "-l", *command]
    return command


def _validate_local_model(config: ProbeConfig, root: Path) -> None:
    if config.model.path is None:
        return
    path = Path(config.model.path)
    resolved = path.expanduser() if path.is_absolute() else root / path
    if not resolved.is_file():
        raise BenchmarkError(f"model file does not exist: {resolved}")


def _run_id(name: str, started_at: datetime) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.") or "experiment"
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{slug}-{uuid.uuid4().hex[:8]}"


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _write_text_atomic(path, content)
