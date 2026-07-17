from __future__ import annotations

import hashlib
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
from probe.evidence import (
    EvidenceError,
    capture_environment,
    capture_model_identity,
    capture_repository_state,
    seal_artifact,
)
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
    try:
        if config.promotion.max_load_1m is not None or config.promotion.require_ac_power:
            _validate_environment_for_promotion(config, capture_environment())
        model_identity = capture_model_identity(config.model, repository_root)
        promotion_evidence, correctness_contract = _prepare_promotion_evidence(
            config,
            repository_root,
        )
        source_state = capture_repository_state(repository_root)
        _validate_source_for_promotion(config, source_state)
        environment_start = capture_environment()
        _validate_environment_for_promotion(config, environment_start)
    except EvidenceError as exc:
        raise BenchmarkError(str(exc)) from exc

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
    trace_path = artifact_dir / "trace.jsonl"
    _append_trace(
        trace_path,
        "artifact_created",
        run_id=run_id,
        config_source=str(config.source),
    )

    _write_json_atomic(artifact_dir / "source.json", source_state)
    _append_trace(
        trace_path,
        "source_captured",
        head=source_state.get("head"),
        dirty=source_state.get("dirty"),
        available=source_state.get("available"),
    )
    _write_json_atomic(artifact_dir / "model.json", model_identity)
    _append_trace(
        trace_path,
        "model_verified",
        verified_local_bytes=model_identity.get("verified_local_bytes"),
        size_bytes=model_identity.get("size_bytes"),
        sha256=model_identity.get("sha256"),
    )
    if promotion_evidence is not None and correctness_contract is not None:
        _write_json_atomic(artifact_dir / "promotion.json", promotion_evidence)
        _write_bytes_atomic(artifact_dir / "correctness-contract.json", correctness_contract)
        _append_trace(
            trace_path,
            "promotion_contract_captured",
            replication_tolerance_percent=config.promotion.replication_tolerance_percent,
            correctness_sha256=promotion_evidence["correctness_contract"]["sha256"],
        )

    host = capture_host().to_dict()
    _write_json_atomic(artifact_dir / "host.json", host)
    _append_trace(trace_path, "host_captured", host_file="host.json")
    _write_json_atomic(artifact_dir / "environment-start.json", environment_start)
    _append_trace(
        trace_path,
        "environment_captured",
        phase="start",
        environment_file="environment-start.json",
        load_average=environment_start.get("load_average"),
    )
    command = build_command(config, binary=binary)
    execution_command = _with_memory_capture(command, config.runtime.capture_peak_memory)
    _append_trace(
        trace_path,
        "command_compiled",
        command=command,
        memory_capture=execution_command != command,
    )
    _write_text_atomic(artifact_dir / "command.txt", display_command(command) + "\n")
    _write_text_atomic(artifact_dir / "config.toml", config.source.read_text(encoding="utf-8"))

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    monotonic_start = time.perf_counter()
    timed_out = False
    _append_trace(
        trace_path,
        "process_started",
        cwd=str(repository_root),
        timeout_seconds=config.runtime.timeout_seconds,
    )
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
    _append_trace(
        trace_path,
        "process_finished",
        return_code=return_code,
        timed_out=timed_out,
        wall_seconds=wall_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )
    environment_end = capture_environment()
    _write_json_atomic(artifact_dir / "environment-end.json", environment_end)
    _append_trace(
        trace_path,
        "environment_captured",
        phase="end",
        environment_file="environment-end.json",
        load_average=environment_end.get("load_average"),
    )

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
    _append_trace(
        trace_path,
        "output_parsed",
        status=status,
        result_count=len(rows),
        parse_error=parse_error,
    )
    model_source = config.model.path or config.model.hf_repo
    execution = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "command": command,
        "binary": binary,
        "return_code": return_code,
        "source_head": source_state.get("head"),
        "source_dirty": source_state.get("dirty"),
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
                "size_bytes": model_identity.get("size_bytes"),
                "sha256": model_identity.get("sha256"),
            },
            "host": host,
            "execution": execution,
            "benchmark": row,
        }
        for row in rows
    ]
    _write_jsonl_atomic(artifact_dir / "results.jsonl", records)
    files = {
        "config": "config.toml",
        "source": "source.json",
        "model": "model.json",
        "host": "host.json",
        "environment_start": "environment-start.json",
        "environment_end": "environment-end.json",
        "command": "command.txt",
        "stdout": "stdout.json",
        "stderr": "stderr.log",
        "results": "results.jsonl",
        "trace": "trace.jsonl",
        "seal": "seal.json",
    }
    if promotion_evidence is not None:
        files["promotion"] = "promotion.json"
        files["correctness_contract"] = "correctness-contract.json"
    manifest = {
        "schema_version": 2,
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
        "promotion": promotion_evidence,
        "files": files,
    }
    _write_json_atomic(artifact_dir / "manifest.json", manifest)
    _append_trace(
        trace_path,
        "run_finalized",
        status=status,
        manifest_file="manifest.json",
        results_file="results.jsonl",
        seal_file="seal.json",
    )
    _write_json_atomic(artifact_dir / "seal.json", seal_artifact(artifact_dir))

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


def _prepare_promotion_evidence(
    config: ProbeConfig,
    root: Path,
) -> tuple[dict[str, Any] | None, bytes | None]:
    contract = config.promotion.correctness_contract
    if contract is None:
        return None, None

    candidate = contract.expanduser()
    resolved = candidate if candidate.is_absolute() else root / candidate
    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"could not read correctness contract {resolved}: {exc}") from exc
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"correctness contract is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(decoded, dict):
        raise EvidenceError(f"correctness contract must be a JSON object: {resolved}")

    return (
        {
            "require_clean_repository": config.promotion.require_clean_repository,
            "replication_tolerance_percent": (config.promotion.replication_tolerance_percent),
            "max_load_1m": config.promotion.max_load_1m,
            "require_ac_power": config.promotion.require_ac_power,
            "correctness_contract": {
                "source": str(contract),
                "artifact": "correctness-contract.json",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
        },
        content,
    )


def _validate_source_for_promotion(
    config: ProbeConfig,
    source_state: dict[str, Any],
) -> None:
    if not config.promotion.require_clean_repository:
        return
    if source_state.get("available") is not True:
        raise EvidenceError("promotion requires an available Git repository")
    if source_state.get("dirty") is not False:
        status = str(source_state.get("status_porcelain_v1", "")).strip()
        suffix = f":\n{status}" if status else ""
        raise EvidenceError(f"promotion requires a clean Git repository{suffix}")


def _validate_environment_for_promotion(
    config: ProbeConfig,
    environment: dict[str, Any],
) -> None:
    load_limit = config.promotion.max_load_1m
    load_average = environment.get("load_average")
    if load_limit is not None:
        if not isinstance(load_average, list) or not load_average:
            raise EvidenceError("promotion requires an available 1-minute load average")
        observed_load = load_average[0]
        if not isinstance(observed_load, int | float) or observed_load > load_limit:
            raise EvidenceError(
                f"promotion start load is too high: observed {observed_load}, limit {load_limit}"
            )
    if config.promotion.require_ac_power:
        power = environment.get("power_source")
        stdout = power.get("stdout") if isinstance(power, dict) else None
        if not isinstance(stdout, str) or "AC Power" not in stdout:
            raise EvidenceError("promotion requires AC power at benchmark start")


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


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _write_text_atomic(path, content)


def _append_trace(path: Path, event: str, **fields: object) -> None:
    record = {"at": datetime.now(UTC).isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
