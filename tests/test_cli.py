from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from probe.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_dry_run_prints_plan_without_requiring_runtime() -> None:
    config = ROOT / "configs" / "gemma4-26b-a4b-baseline.toml"

    result = runner.invoke(app, ["sweep", "--dry-run", str(config)])

    assert result.exit_code == 0, result.output
    assert "3 tests" in result.output
    assert "llama-bench --output json" in result.output
    assert "ggml-org/gemma-4-26B-A4B-it-GGUF" in result.output


def test_invalid_config_gets_concise_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[experiment]\nname = 'broken'\n", encoding="utf-8")

    result = runner.invoke(app, ["sweep", "--dry-run", str(path)])

    assert result.exit_code == 2
    assert "missing or invalid [model] section" in result.output


def test_doctor_supports_human_and_json_output() -> None:
    human = runner.invoke(app, ["doctor"])
    machine = runner.invoke(app, ["doctor", "--json"])

    assert human.exit_code == 0, human.output
    assert "Inference runtimes" in human.output
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.stdout)
    assert "host" in payload
    assert "runtimes" in payload
    assert all("serial" not in key.lower() for key in payload["host"])


def test_compare_can_write_a_report(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "experiment": {"name": "cli"},
                "model": {"label": "Gemma", "quantization": "Q4"},
                "execution": {"peak_rss_bytes": 1024},
                "benchmark": {
                    "n_prompt": 0,
                    "n_gen": 128,
                    "avg_ts": 42.0,
                    "stddev_ts": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.md"

    result = runner.invoke(app, ["compare", str(results), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "## tg128" in output.read_text(encoding="utf-8")


def test_sweep_executes_with_an_explicit_fixture_runtime(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    binary = tmp_path / "llama-bench"
    binary.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '"
        '[{"n_prompt":0,"n_gen":1,"n_batch":2048,"n_ubatch":512,'
        '"n_threads":10,"n_gpu_layers":99,"n_cpu_moe":0,"flash_attn":1,'
        '"type_k":"f16","type_v":"f16","use_mmap":true,"no_kv_offload":false,'
        f'"model_filename":"{model}","avg_ts":1.0,"stddev_ts":0.0,'
        '"samples_ns":[1000000000,1000000000,1000000000,1000000000,1000000000],'
        '"samples_ts":[1.0,1.0,1.0,1.0,1.0]}]\'\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[experiment]
name = "cli-run"

[model]
path = "{model}"

[runtime]
binary = "{binary}"
capture_peak_memory = false

[workload]
prompt_tokens = []
generation_tokens = [1]

[matrix]
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["sweep", str(config), "--output-dir", str(tmp_path / "artifacts")],
    )

    assert result.exit_code == 0, result.output
    assert "Completed 1 result rows" in result.output
