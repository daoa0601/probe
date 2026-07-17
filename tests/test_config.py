from __future__ import annotations

from pathlib import Path

import pytest

from probe.config import ConfigError, build_command, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "configs").rglob("*.toml")),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_checked_in_configs_are_valid(path: Path) -> None:
    config = load_config(path)

    assert config.estimated_test_cases > 0
    assert build_command(config)[1:3] == ["--output", "json"]


def test_baseline_compiles_to_expected_llama_bench_axes() -> None:
    config = load_config(ROOT / "configs" / "gemma4-26b-a4b-baseline.toml")

    command = build_command(config)

    assert config.estimated_test_cases == 3
    assert command[:3] == ["llama-bench", "--output", "json"]
    assert _argument(command, "--hf-repo") == "ggml-org/gemma-4-26B-A4B-it-GGUF"
    assert _argument(command, "--hf-file") == "gemma-4-26B-A4B-it-Q4_0.gguf"
    assert _argument(command, "--n-prompt") == "512,4096"
    assert _argument(command, "--n-gen") == "128"
    assert _argument(command, "--threads") == "10"
    assert _argument(command, "--flash-attn") == "on"


def test_combined_workloads_become_repeated_pg_flags(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[experiment]
name = "combined"

[model]
path = "models/test.gguf"

[runtime]
warmup = false
offline = true

[workload]
prompt_tokens = []
generation_tokens = []
combined = [[512, 64], [4096, 128]]

[matrix]
""",
    )

    config = load_config(config_path)
    command = build_command(config)

    assert _argument(command, "--n-prompt") == "0"
    assert _argument(command, "--n-gen") == "0"
    assert "--no-warmup" in command
    assert "--offline" in command
    assert [command[index + 1] for index, value in enumerate(command) if value == "-pg"] == [
        "512,64",
        "4096,128",
    ]


def test_unknown_keys_fail_fast(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[experiment]
name = "typo"
repetitons = 3

[model]
path = "model.gguf"

[runtime]

[workload]

[matrix]
""",
    )

    with pytest.raises(ConfigError, match="unknown key.*repetitons"):
        load_config(config_path)


def test_incompatible_batch_cartesian_product_is_rejected(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[experiment]
name = "invalid-batches"

[model]
path = "model.gguf"

[runtime]

[workload]

[matrix]
batch_sizes = [256, 512]
ubatch_sizes = [512]
""",
    )

    with pytest.raises(ConfigError, match="ubatch_sizes"):
        load_config(config_path)


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(content, encoding="utf-8")
    return path


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]
