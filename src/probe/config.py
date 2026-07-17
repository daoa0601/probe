from __future__ import annotations

import math
import os
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is ambiguous or invalid."""


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    output_dir: Path
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Model:
    label: str
    path: str | None
    hf_repo: str | None
    hf_file: str | None
    quantization: str
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class Runtime:
    binary: str
    repetitions: int
    delay_seconds: int
    warmup: bool
    offline: bool
    capture_peak_memory: bool
    timeout_seconds: int
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class Workload:
    prompt_tokens: tuple[int, ...]
    generation_tokens: tuple[int, ...]
    combined: tuple[tuple[int, int], ...]

    @property
    def test_count(self) -> int:
        return len(self.prompt_tokens) + len(self.generation_tokens) + len(self.combined)


@dataclass(frozen=True)
class Matrix:
    batch_sizes: tuple[int, ...]
    ubatch_sizes: tuple[int, ...]
    threads: tuple[int, ...]
    gpu_layers: tuple[int, ...]
    cpu_moe_layers: tuple[int, ...]
    flash_attention: tuple[str, ...]
    cache_type_k: tuple[str, ...]
    cache_type_v: tuple[str, ...]
    mmap: tuple[bool, ...]
    no_kv_offload: tuple[bool, ...]

    @property
    def combination_count(self) -> int:
        return math.prod(
            (
                len(self.batch_sizes),
                len(self.ubatch_sizes),
                len(self.threads),
                len(self.gpu_layers),
                len(self.cpu_moe_layers),
                len(self.flash_attention),
                len(self.cache_type_k),
                len(self.cache_type_v),
                len(self.mmap),
                len(self.no_kv_offload),
            )
        )


@dataclass(frozen=True)
class Promotion:
    require_clean_repository: bool
    correctness_binary: str
    correctness_contract: Path | None
    replication_tolerance_percent: float | None
    max_load_1m: float | None
    require_ac_power: bool


@dataclass(frozen=True)
class ProbeConfig:
    source: Path
    experiment: Experiment
    model: Model
    runtime: Runtime
    workload: Workload
    matrix: Matrix
    promotion: Promotion

    @property
    def estimated_test_cases(self) -> int:
        return self.workload.test_count * self.matrix.combination_count


_TOP_LEVEL_KEYS = {"experiment", "model", "runtime", "workload", "matrix", "promotion"}
_EXPERIMENT_KEYS = {"name", "description", "output_dir", "tags"}
_MODEL_KEYS = {
    "label",
    "path",
    "hf_repo",
    "hf_file",
    "quantization",
    "sha256",
    "size_bytes",
}
_RUNTIME_KEYS = {
    "binary",
    "repetitions",
    "delay_seconds",
    "warmup",
    "offline",
    "capture_peak_memory",
    "timeout_seconds",
    "extra_args",
}
_WORKLOAD_KEYS = {"prompt_tokens", "generation_tokens", "combined"}
_MATRIX_KEYS = {
    "batch_sizes",
    "ubatch_sizes",
    "threads",
    "gpu_layers",
    "cpu_moe_layers",
    "flash_attention",
    "cache_type_k",
    "cache_type_v",
    "mmap",
    "no_kv_offload",
}
_PROMOTION_KEYS = {
    "require_clean_repository",
    "correctness_binary",
    "correctness_contract",
    "replication_tolerance_percent",
    "max_load_1m",
    "require_ac_power",
}
_FLASH_ATTENTION_VALUES = {"on", "off", "auto"}
_KV_CACHE_TYPES = {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}


def load_config(path: Path) -> ProbeConfig:
    """Load and strictly validate a Probe TOML configuration."""

    source = path.expanduser().resolve()
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {source}: {exc}") from exc

    _reject_unknown(raw, _TOP_LEVEL_KEYS, "top level")
    experiment_raw = _table(raw, "experiment")
    model_raw = _table(raw, "model")
    runtime_raw = _table(raw, "runtime")
    workload_raw = _table(raw, "workload")
    matrix_raw = _table(raw, "matrix")
    promotion_raw = _optional_table(raw, "promotion")

    _reject_unknown(experiment_raw, _EXPERIMENT_KEYS, "[experiment]")
    _reject_unknown(model_raw, _MODEL_KEYS, "[model]")
    _reject_unknown(runtime_raw, _RUNTIME_KEYS, "[runtime]")
    _reject_unknown(workload_raw, _WORKLOAD_KEYS, "[workload]")
    _reject_unknown(matrix_raw, _MATRIX_KEYS, "[matrix]")
    _reject_unknown(promotion_raw, _PROMOTION_KEYS, "[promotion]")

    experiment = Experiment(
        name=_required_string(experiment_raw, "name", "[experiment]"),
        description=_string(experiment_raw, "description", ""),
        output_dir=Path(_string(experiment_raw, "output_dir", "results")),
        tags=_string_tuple(experiment_raw, "tags", ()),
    )

    local_path = _optional_string(model_raw, "path")
    hf_repo = _optional_string(model_raw, "hf_repo")
    if (local_path is None) == (hf_repo is None):
        raise ConfigError("[model] must define exactly one of 'path' or 'hf_repo'")
    hf_file = _optional_string(model_raw, "hf_file")
    if hf_file is not None and hf_repo is None:
        raise ConfigError("[model].hf_file requires [model].hf_repo")
    expanded_path = os.path.expandvars(os.path.expanduser(local_path)) if local_path else None
    default_label = hf_repo or Path(expanded_path or "model").name
    model = Model(
        label=_string(model_raw, "label", default_label),
        path=expanded_path,
        hf_repo=hf_repo,
        hf_file=hf_file,
        quantization=_string(model_raw, "quantization", "unknown"),
        sha256=_optional_sha256(model_raw, "sha256"),
        size_bytes=_optional_integer(model_raw, "size_bytes", minimum=1),
    )

    runtime = Runtime(
        binary=os.path.expandvars(_string(runtime_raw, "binary", "llama-bench")),
        repetitions=_integer(runtime_raw, "repetitions", 5, minimum=1),
        delay_seconds=_integer(runtime_raw, "delay_seconds", 0, minimum=0),
        warmup=_boolean(runtime_raw, "warmup", True),
        offline=_boolean(runtime_raw, "offline", False),
        capture_peak_memory=_boolean(runtime_raw, "capture_peak_memory", True),
        timeout_seconds=_integer(runtime_raw, "timeout_seconds", 7200, minimum=1),
        extra_args=_string_tuple(runtime_raw, "extra_args", ()),
    )
    forbidden_extra_args = {"-o", "--output", "-oe", "--output-err"}
    overlap = forbidden_extra_args.intersection(runtime.extra_args)
    if overlap:
        joined = ", ".join(sorted(overlap))
        raise ConfigError(f"[runtime].extra_args cannot override structured output flags: {joined}")

    workload = Workload(
        prompt_tokens=_integer_tuple(
            workload_raw, "prompt_tokens", (512,), minimum=1, allow_empty=True
        ),
        generation_tokens=_integer_tuple(
            workload_raw, "generation_tokens", (128,), minimum=1, allow_empty=True
        ),
        combined=_combined_tuple(workload_raw.get("combined", [])),
    )
    if workload.test_count == 0:
        raise ConfigError(
            "[workload] must define at least one prompt, generation, or combined test"
        )

    matrix = Matrix(
        batch_sizes=_integer_tuple(matrix_raw, "batch_sizes", (2048,), minimum=1),
        ubatch_sizes=_integer_tuple(matrix_raw, "ubatch_sizes", (512,), minimum=1),
        threads=_integer_tuple(matrix_raw, "threads", (10,), minimum=1),
        gpu_layers=_integer_tuple(matrix_raw, "gpu_layers", (99,), minimum=0),
        cpu_moe_layers=_integer_tuple(matrix_raw, "cpu_moe_layers", (0,), minimum=0),
        flash_attention=_choice_tuple(
            matrix_raw,
            "flash_attention",
            ("on",),
            _FLASH_ATTENTION_VALUES,
        ),
        cache_type_k=_choice_tuple(
            matrix_raw,
            "cache_type_k",
            ("f16",),
            _KV_CACHE_TYPES,
        ),
        cache_type_v=_choice_tuple(
            matrix_raw,
            "cache_type_v",
            ("f16",),
            _KV_CACHE_TYPES,
        ),
        mmap=_boolean_tuple(matrix_raw, "mmap", (True,)),
        no_kv_offload=_boolean_tuple(matrix_raw, "no_kv_offload", (False,)),
    )
    if max(matrix.ubatch_sizes) > min(matrix.batch_sizes):
        raise ConfigError(
            "every [matrix].ubatch_sizes value must be <= every batch_sizes value; "
            "split incompatible values into separate configs"
        )

    contract_value = _optional_string(promotion_raw, "correctness_contract")
    correctness_contract = (
        Path(os.path.expandvars(os.path.expanduser(contract_value)))
        if contract_value is not None
        else None
    )
    promotion = Promotion(
        require_clean_repository=_boolean(
            promotion_raw,
            "require_clean_repository",
            False,
        ),
        correctness_binary=os.path.expandvars(
            _string(promotion_raw, "correctness_binary", "llama-cli")
        ),
        correctness_contract=correctness_contract,
        replication_tolerance_percent=_optional_number(
            promotion_raw,
            "replication_tolerance_percent",
            minimum_exclusive=0,
            maximum=100,
        ),
        max_load_1m=_optional_number(
            promotion_raw,
            "max_load_1m",
            minimum_exclusive=0,
            maximum=10_000,
        ),
        require_ac_power=_boolean(promotion_raw, "require_ac_power", False),
    )
    if promotion.require_clean_repository:
        missing: list[str] = []
        if model.path is None:
            missing.append("a local [model].path")
        if model.sha256 is None:
            missing.append("[model].sha256")
        if model.size_bytes is None:
            missing.append("[model].size_bytes")
        if promotion.correctness_contract is None:
            missing.append("[promotion].correctness_contract")
        if promotion.replication_tolerance_percent is None:
            missing.append("[promotion].replication_tolerance_percent")
        if missing:
            raise ConfigError("promotion-eligible configs require " + ", ".join(missing))
        if matrix.combination_count != 1:
            raise ConfigError(
                "promotion-eligible configs require exactly one matrix combination so the "
                "correctness runtime is unambiguous"
            )

    return ProbeConfig(
        source=source,
        experiment=experiment,
        model=model,
        runtime=runtime,
        workload=workload,
        matrix=matrix,
        promotion=promotion,
    )


def build_command(config: ProbeConfig, *, binary: str | None = None) -> list[str]:
    """Translate a validated config to one deterministic llama-bench invocation."""

    command = [binary or config.runtime.binary, "--output", "json"]
    command.extend(["--repetitions", str(config.runtime.repetitions)])
    command.extend(["--delay", str(config.runtime.delay_seconds)])
    if not config.runtime.warmup:
        command.append("--no-warmup")

    if config.model.path is not None:
        command.extend(["--model", config.model.path])
    else:
        if config.model.hf_repo is None:  # guarded by load_config; keeps the invariant explicit
            raise ConfigError("model source is missing")
        command.extend(["--hf-repo", config.model.hf_repo])
        if config.model.hf_file is not None:
            command.extend(["--hf-file", config.model.hf_file])
    if config.runtime.offline:
        command.append("--offline")

    command.extend(["--n-prompt", _join_ints(config.workload.prompt_tokens, empty="0")])
    command.extend(["--n-gen", _join_ints(config.workload.generation_tokens, empty="0")])
    if config.workload.combined:
        # llama-bench treats commas inside one -pg value as the prompt/generation separator,
        # so multiple combined tests are emitted as repeated flags.
        for prompt, generation in config.workload.combined:
            command.extend(["-pg", f"{prompt},{generation}"])

    matrix = config.matrix
    command.extend(["--batch-size", _join_ints(matrix.batch_sizes)])
    command.extend(["--ubatch-size", _join_ints(matrix.ubatch_sizes)])
    command.extend(["--threads", _join_ints(matrix.threads)])
    command.extend(["--n-gpu-layers", _join_ints(matrix.gpu_layers)])
    command.extend(["--n-cpu-moe", _join_ints(matrix.cpu_moe_layers)])
    command.extend(["--flash-attn", ",".join(matrix.flash_attention)])
    command.extend(["--cache-type-k", ",".join(matrix.cache_type_k)])
    command.extend(["--cache-type-v", ",".join(matrix.cache_type_v)])
    command.extend(["--mmap", _join_bools(matrix.mmap)])
    command.extend(["--no-kv-offload", _join_bools(matrix.no_kv_offload)])
    command.extend(config.runtime.extra_args)
    return command


def display_command(command: list[str]) -> str:
    return shlex.join(command)


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{key}] section")
    return value


def _optional_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"invalid [{key}] section")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {location}: {', '.join(unknown)}")


def _required_string(raw: dict[str, Any], key: str, location: str) -> str:
    value = _optional_string(raw, key)
    if value is None:
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _string(raw: dict[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value.strip()


def _integer(raw: dict[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_integer(raw: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_number(
    raw: dict[str, Any],
    key: str,
    *,
    minimum_exclusive: float,
    maximum: float,
) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum_exclusive < number <= maximum:
        raise ConfigError(f"{key} must be > {minimum_exclusive} and <= {maximum}")
    return number


def _optional_sha256(raw: dict[str, Any], key: str) -> str | None:
    value = _optional_string(raw, key)
    if value is None:
        return None
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ConfigError(f"{key} must be a 64-character hexadecimal SHA-256")
    return normalized


def _boolean(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _string_tuple(raw: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = raw.get(key, list(default))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{key} must be an array of strings")
    return tuple(value)


def _integer_tuple(
    raw: dict[str, Any],
    key: str,
    default: tuple[int, ...],
    *,
    minimum: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    value = raw.get(key, list(default))
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be an array of integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < minimum for item in value):
        raise ConfigError(f"{key} values must be integers >= {minimum}")
    if not value and not allow_empty:
        raise ConfigError(f"{key} must not be empty")
    return tuple(value)


def _boolean_tuple(raw: dict[str, Any], key: str, default: tuple[bool, ...]) -> tuple[bool, ...]:
    value = raw.get(key, list(default))
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        raise ConfigError(f"{key} must be an array of booleans")
    if not value:
        raise ConfigError(f"{key} must not be empty")
    return tuple(value)


def _choice_tuple(
    raw: dict[str, Any],
    key: str,
    default: tuple[str, ...],
    choices: set[str],
) -> tuple[str, ...]:
    values = _string_tuple(raw, key, default)
    if not values:
        raise ConfigError(f"{key} must not be empty")
    invalid = sorted(set(values) - choices)
    if invalid:
        raise ConfigError(f"invalid {key} value(s): {', '.join(invalid)}")
    return values


def _combined_tuple(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ConfigError("combined must be an array of [prompt_tokens, generation_tokens] pairs")
    combined: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in pair)
        ):
            raise ConfigError("each combined value must be two positive integers")
        combined.append((pair[0], pair[1]))
    return tuple(combined)


def _join_ints(values: tuple[int, ...], *, empty: str = "") -> str:
    return ",".join(str(value) for value in values) if values else empty


def _join_bools(values: tuple[bool, ...]) -> str:
    return ",".join("1" if value else "0" for value in values)
