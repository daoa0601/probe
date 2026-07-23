from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
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


@dataclass(frozen=True)
class CorrectnessArtifact:
    command: list[str]
    stdout: str
    stderr: str
    transcript: str
    result: dict[str, Any]


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


def resolve_correctness_binary(
    config: ProbeConfig,
    *,
    benchmark_binary: str,
    root: Path,
) -> str:
    """Resolve the llama-cli companion used by promotion correctness checks."""

    configured = os.environ.get("PROBE_LLAMA_CLI") or config.promotion.correctness_binary
    explicit = Path(configured).expanduser()
    if explicit.parent != Path("."):
        if explicit.is_file() and os.access(explicit, os.X_OK):
            return str(explicit.resolve())
        raise BenchmarkError(f"correctness binary is not executable: {explicit}")
    sibling = Path(benchmark_binary).with_name(configured)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling.resolve())
    on_path = shutil.which(configured)
    if on_path:
        return str(Path(on_path).resolve())
    local = root / ".cache" / "llama.cpp" / "build" / "bin" / configured
    if local.is_file() and os.access(local, os.X_OK):
        return str(local.resolve())
    raise BenchmarkError(f"could not find promotion correctness binary {configured!r}")


def run_sweep(
    config: ProbeConfig,
    *,
    output_root: Path | None = None,
    root: Path | None = None,
) -> RunArtifact:
    """Run a complete llama-bench matrix and preserve raw and normalized output."""

    repository_root = (root or project_root()).resolve()
    binary = resolve_binary(config, root=repository_root)
    correctness_binary = (
        resolve_correctness_binary(
            config,
            benchmark_binary=binary,
            root=repository_root,
        )
        if config.promotion.correctness_contract is not None
        else None
    )
    try:
        if config.promotion.max_load_1m is not None or config.promotion.require_ac_power:
            _validate_environment_for_promotion(config, capture_environment())
        model_identity = capture_model_identity(config.model, repository_root)
        promotion_evidence, correctness_contract, correctness_spec = _prepare_promotion_evidence(
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
    _write_text_atomic(artifact_dir / "stdout.json", stdout)
    _write_text_atomic(artifact_dir / "stderr.log", stderr)

    correctness_artifact: CorrectnessArtifact | None = None
    if correctness_binary is not None and correctness_spec is not None:
        correctness_transcript_path = artifact_dir / "correctness-transcript.txt"
        correctness_command = _build_correctness_command(
            config,
            binary=correctness_binary,
            contract=correctness_spec,
            transcript_path=correctness_transcript_path,
        )
        _append_trace(
            trace_path,
            "correctness_started",
            command=correctness_command,
            timeout_seconds=config.runtime.timeout_seconds,
        )
        correctness_artifact = _run_correctness(
            correctness_command,
            correctness_spec,
            cwd=repository_root,
            environment=environment,
            timeout_seconds=config.runtime.timeout_seconds,
            transcript_path=correctness_transcript_path,
        )
        _write_text_atomic(
            artifact_dir / "correctness-command.txt",
            display_command(correctness_artifact.command) + "\n",
        )
        _write_text_atomic(
            artifact_dir / "correctness-stdout.json",
            correctness_artifact.stdout,
        )
        _write_text_atomic(
            artifact_dir / "correctness-stderr.log",
            correctness_artifact.stderr,
        )
        _write_text_atomic(
            artifact_dir / "correctness-transcript.txt",
            correctness_artifact.transcript,
        )
        _write_json_atomic(
            artifact_dir / "correctness-result.json",
            correctness_artifact.result,
        )
        _append_trace(
            trace_path,
            "correctness_finished",
            status=correctness_artifact.result["status"],
            return_code=correctness_artifact.result["return_code"],
            exact_match=correctness_artifact.result["exact_match"],
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

    rows: list[dict[str, Any]] = []
    parse_error: str | None = None
    if return_code == 0:
        try:
            decoded = json.loads(stdout)
            rows = _validate_benchmark_rows(decoded, config, repository_root)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)

    correctness_passed = (
        correctness_artifact is None or correctness_artifact.result["status"] == "pass"
    )
    status = (
        "succeeded" if return_code == 0 and parse_error is None and correctness_passed else "failed"
    )
    _append_trace(
        trace_path,
        "output_parsed",
        status=status,
        result_count=len(rows),
        parse_error=parse_error,
        correctness_status=(
            correctness_artifact.result["status"] if correctness_artifact is not None else None
        ),
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
        files["correctness_command"] = "correctness-command.txt"
        files["correctness_stdout"] = "correctness-stdout.json"
        files["correctness_stderr"] = "correctness-stderr.log"
        files["correctness_transcript"] = "correctness-transcript.txt"
        files["correctness_result"] = "correctness-result.json"
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
        "correctness": (correctness_artifact.result if correctness_artifact is not None else None),
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
        if correctness_artifact is not None and not correctness_passed:
            reason = f"correctness check {correctness_artifact.result['status']}"
        else:
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


def _validate_benchmark_rows(
    decoded: object,
    config: ProbeConfig,
    root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(decoded, list) or any(not isinstance(row, dict) for row in decoded):
        raise ValueError("llama-bench JSON must be an array of objects")
    rows: list[dict[str, Any]] = decoded
    if len(rows) != config.estimated_test_cases:
        raise ValueError(
            "llama-bench result count does not match the declared matrix: "
            f"observed {len(rows)}, expected {config.estimated_test_cases}"
        )

    workloads = [
        *((prompt, 0) for prompt in config.workload.prompt_tokens),
        *((0, generation) for generation in config.workload.generation_tokens),
        *config.workload.combined,
    ]
    matrix = config.matrix
    expected_workloads = Counter(
        workload for workload in workloads for _ in range(matrix.combination_count)
    )
    observed_workloads: Counter[tuple[int, int]] = Counter()
    observed_cases: Counter[tuple[object, ...]] = Counter()
    expected_cases = Counter(
        (
            prompt,
            generation,
            batch,
            ubatch,
            threads,
            gpu_layers,
            cpu_moe_layers,
            1 if flash_attention == "on" else 0,
            cache_k,
            cache_v,
            mmap,
            no_kv_offload,
        )
        for prompt, generation in workloads
        for (
            batch,
            ubatch,
            threads,
            gpu_layers,
            cpu_moe_layers,
            flash_attention,
            cache_k,
            cache_v,
            mmap,
            no_kv_offload,
        ) in product(
            matrix.batch_sizes,
            matrix.ubatch_sizes,
            matrix.threads,
            matrix.gpu_layers,
            matrix.cpu_moe_layers,
            matrix.flash_attention,
            matrix.cache_type_k,
            matrix.cache_type_v,
            matrix.mmap,
            matrix.no_kv_offload,
        )
    )
    strict_flash_attention = "auto" not in matrix.flash_attention
    expected_model_path = _expected_model_path(config, root)

    for index, row in enumerate(rows):
        prompt = _benchmark_int(row, "n_prompt", index, minimum=0)
        generation = _benchmark_int(row, "n_gen", index, minimum=0)
        if prompt == 0 and generation == 0:
            raise ValueError(f"llama-bench row {index} has a zero-token workload")
        observed_workloads[(prompt, generation)] += 1

        batch = _benchmark_int(row, "n_batch", index, minimum=1)
        ubatch = _benchmark_int(row, "n_ubatch", index, minimum=1)
        threads = _benchmark_int(row, "n_threads", index, minimum=1)
        gpu_layers = _benchmark_int(row, "n_gpu_layers", index, minimum=0)
        cpu_moe_layers = _benchmark_int(row, "n_cpu_moe", index, minimum=0)
        flash_attention = _benchmark_int(row, "flash_attn", index, minimum=0)
        cache_k = _benchmark_string(row, "type_k", index)
        cache_v = _benchmark_string(row, "type_v", index)
        mmap = _benchmark_bool(row, "use_mmap", index)
        no_kv_offload = _benchmark_bool(row, "no_kv_offload", index)
        case = (
            prompt,
            generation,
            batch,
            ubatch,
            threads,
            gpu_layers,
            cpu_moe_layers,
            flash_attention,
            cache_k,
            cache_v,
            mmap,
            no_kv_offload,
        )
        observed_cases[case] += 1

        _validate_benchmark_samples(row, index, prompt + generation, config.runtime.repetitions)
        if expected_model_path is not None:
            observed_model = _benchmark_string(row, "model_filename", index)
            observed_model_path = Path(observed_model).expanduser()
            if not observed_model_path.is_absolute():
                observed_model_path = root / observed_model_path
            if observed_model_path.resolve() != expected_model_path:
                raise ValueError(
                    f"llama-bench row {index} model path does not match the declared model"
                )

    if observed_workloads != expected_workloads:
        raise ValueError("llama-bench workloads do not match the declared workload matrix")
    if strict_flash_attention and observed_cases != expected_cases:
        raise ValueError("llama-bench settings do not match the declared parameter matrix")
    if not strict_flash_attention:
        observed_projected = Counter(case[:7] + case[8:] for case in observed_cases.elements())
        expected_projected = Counter(case[:7] + case[8:] for case in expected_cases.elements())
        if observed_projected != expected_projected:
            raise ValueError("llama-bench settings do not match the declared parameter matrix")
        allowed_flash = {
            *(1 for value in matrix.flash_attention if value in {"on", "auto"}),
            *(0 for value in matrix.flash_attention if value in {"off", "auto"}),
        }
        if any(case[7] not in allowed_flash for case in observed_cases):
            raise ValueError("llama-bench flash-attention result is outside the declared matrix")
    return rows


def _validate_benchmark_samples(
    row: dict[str, Any],
    index: int,
    tokens: int,
    repetitions: int,
) -> None:
    samples_ns = row.get("samples_ns")
    samples_ts = row.get("samples_ts")
    if not isinstance(samples_ns, list) or len(samples_ns) != repetitions:
        raise ValueError(f"llama-bench row {index} samples_ns count does not match repetitions")
    if not isinstance(samples_ts, list) or len(samples_ts) != repetitions:
        raise ValueError(f"llama-bench row {index} samples_ts count does not match repetitions")
    for sample_index, (duration, throughput) in enumerate(zip(samples_ns, samples_ts, strict=True)):
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError(f"llama-bench row {index} sample {sample_index} has invalid duration")
        if (
            isinstance(throughput, bool)
            or not isinstance(throughput, int | float)
            or not math.isfinite(float(throughput))
            or throughput <= 0
        ):
            raise ValueError(
                f"llama-bench row {index} sample {sample_index} has invalid throughput"
            )
        derived = tokens * 1_000_000_000 / duration
        if not math.isclose(float(throughput), derived, rel_tol=5e-5, abs_tol=5e-4):
            raise ValueError(
                f"llama-bench row {index} sample {sample_index} token count is inconsistent"
            )

    average = row.get("avg_ts")
    if (
        isinstance(average, bool)
        or not isinstance(average, int | float)
        or not math.isfinite(float(average))
        or average <= 0
    ):
        raise ValueError(f"llama-bench row {index} has invalid avg_ts")
    recomputed_average = math.fsum(float(value) for value in samples_ts) / repetitions
    if not math.isclose(float(average), recomputed_average, rel_tol=5e-6, abs_tol=1e-3):
        raise ValueError(f"llama-bench row {index} avg_ts does not reproduce from samples")
    standard_deviation = row.get("stddev_ts")
    if (
        isinstance(standard_deviation, bool)
        or not isinstance(standard_deviation, int | float)
        or not math.isfinite(float(standard_deviation))
        or standard_deviation < 0
    ):
        raise ValueError(f"llama-bench row {index} has invalid stddev_ts")
    recomputed_deviation = (
        statistics.stdev(float(value) for value in samples_ts) if repetitions > 1 else 0.0
    )
    if not math.isclose(
        float(standard_deviation),
        recomputed_deviation,
        rel_tol=5e-4,
        abs_tol=1e-3,
    ):
        raise ValueError(f"llama-bench row {index} stddev_ts does not reproduce from samples")


def _expected_model_path(config: ProbeConfig, root: Path) -> Path | None:
    if config.model.path is None:
        return None
    candidate = Path(config.model.path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _benchmark_int(
    row: dict[str, Any],
    key: str,
    index: int,
    *,
    minimum: int,
) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"llama-bench row {index} has invalid {key}")
    return value


def _benchmark_string(row: dict[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"llama-bench row {index} has invalid {key}")
    return value


def _benchmark_bool(row: dict[str, Any], key: str, index: int) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"llama-bench row {index} has invalid {key}")
    return value


def _with_memory_capture(command: list[str], enabled: bool) -> list[str]:
    time_binary = Path("/usr/bin/time")
    if enabled and platform.system() == "Darwin" and time_binary.is_file():
        return [str(time_binary), "-l", *command]
    return command


def _prepare_promotion_evidence(
    config: ProbeConfig,
    root: Path,
) -> tuple[dict[str, Any] | None, bytes | None, dict[str, Any] | None]:
    contract = config.promotion.correctness_contract
    if contract is None:
        return None, None, None

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
    _validate_correctness_contract(decoded, resolved)

    return (
        {
            "require_clean_repository": config.promotion.require_clean_repository,
            "correctness_binary": config.promotion.correctness_binary,
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
        decoded,
    )


def _validate_correctness_contract(contract: dict[str, Any], source: Path) -> None:
    allowed = {"id", "prompt", "expected", "generation"}
    unknown = sorted(set(contract) - allowed)
    if unknown:
        raise EvidenceError(f"unknown correctness contract keys in {source}: {unknown}")
    if not isinstance(contract.get("id"), str) or not contract["id"].strip():
        raise EvidenceError(f"correctness contract id must be a non-empty string: {source}")
    if not isinstance(contract.get("prompt"), str) or not contract["prompt"].strip():
        raise EvidenceError(f"correctness contract prompt must be a non-empty string: {source}")
    if "expected" not in contract:
        raise EvidenceError(f"correctness contract must define expected output: {source}")
    generation = contract.get("generation")
    if not isinstance(generation, dict):
        raise EvidenceError(f"correctness contract generation must be an object: {source}")
    generation_keys = {"context_size", "max_tokens", "temperature", "seed", "reasoning"}
    if set(generation) != generation_keys:
        raise EvidenceError(
            f"correctness contract generation keys must be {sorted(generation_keys)}: {source}"
        )
    for key in ("context_size", "max_tokens"):
        value = generation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvidenceError(f"correctness generation {key} must be a positive integer")
    temperature = generation["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(float(temperature))
        or temperature < 0
    ):
        raise EvidenceError("correctness generation temperature must be a finite number >= 0")
    seed = generation["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvidenceError("correctness generation seed must be an integer")
    if not isinstance(generation["reasoning"], bool):
        raise EvidenceError("correctness generation reasoning must be a boolean")


def _build_correctness_command(
    config: ProbeConfig,
    *,
    binary: str,
    contract: dict[str, Any],
    transcript_path: Path,
) -> list[str]:
    if config.model.path is None:
        raise EvidenceError("promotion correctness requires a local model path")
    generation = contract["generation"]
    if not isinstance(generation, dict):  # guarded by contract validation
        raise EvidenceError("correctness generation is missing")
    matrix = config.matrix
    command = [
        binary,
        "--model",
        config.model.path,
        "--ctx-size",
        str(generation["context_size"]),
        "--n-predict",
        str(generation["max_tokens"]),
        "--temp",
        f"{float(generation['temperature']):g}",
        "--seed",
        str(generation["seed"]),
        "--reasoning",
        "on" if generation["reasoning"] else "off",
        "--threads",
        str(matrix.threads[0]),
        "--n-gpu-layers",
        str(matrix.gpu_layers[0]),
        "--n-cpu-moe",
        str(matrix.cpu_moe_layers[0]),
        "--batch-size",
        str(matrix.batch_sizes[0]),
        "--ubatch-size",
        str(matrix.ubatch_sizes[0]),
        "--flash-attn",
        matrix.flash_attention[0],
        "--cache-type-k",
        matrix.cache_type_k[0],
        "--cache-type-v",
        matrix.cache_type_v[0],
        "--conversation",
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--simple-io",
        "--color",
        "off",
        "--log-disable",
        "--output",
        str(transcript_path),
        "--prompt",
        str(contract["prompt"]),
    ]
    if config.runtime.offline:
        command.append("--offline")
    if not matrix.mmap[0]:
        command.append("--no-mmap")
    if matrix.no_kv_offload[0]:
        command.append("--no-kv-offload")
    return command


def _run_correctness(
    command: list[str],
    contract: dict[str, Any],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    transcript_path: Path,
) -> CorrectnessArtifact:
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        stderr += f"\nprobe: correctness check timed out after {timeout_seconds}s\n"

    observed: Any = None
    parse_error: str | None = None
    try:
        transcript = transcript_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        transcript = ""
        parse_error = f"could not read correctness transcript: {exc}"
    try:
        if parse_error is None:
            observed = _parse_correctness_transcript(transcript, str(contract["prompt"]))
    except ValueError as exc:
        parse_error = str(exc)
    expected = contract["expected"]
    exact_match = parse_error is None and observed == expected
    status = "pass" if return_code == 0 and exact_match else "fail"
    return CorrectnessArtifact(
        command=command,
        stdout=stdout,
        stderr=stderr,
        transcript=transcript,
        result={
            "schema_version": 2,
            "contract_id": contract["id"],
            "response_source": "correctness-transcript.txt",
            "status": status,
            "return_code": return_code,
            "timed_out": timed_out,
            "wall_seconds": time.perf_counter() - started,
            "parse_error": parse_error,
            "expected": expected,
            "observed": observed,
            "exact_match": exact_match,
        },
    )


def _parse_correctness_transcript(transcript: str, prompt: str) -> Any:
    """Parse one assistant JSON value from llama-cli's dedicated single-turn transcript."""

    prefix = f"User:\n{prompt}\n\nAssistant:\n"
    if not transcript.startswith(prefix):
        raise ValueError(
            "correctness transcript does not start with the exact user/assistant frame"
        )
    response = transcript[len(prefix) :]
    if not response.endswith("\n\n"):
        raise ValueError(
            "correctness transcript does not end with the expected assistant delimiter"
        )
    response = response[:-2]
    if not response.strip():
        raise ValueError("correctness transcript contains an empty assistant response")
    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant response is not exactly one JSON value: {exc}") from exc


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
