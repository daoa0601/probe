from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe.config import load_config
from probe.runner import BenchmarkError, parse_peak_rss, run_sweep


def test_run_sweep_preserves_and_normalizes_output(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path, exit_code=0)
    config = _local_config(tmp_path, binary)

    artifact = run_sweep(config, root=tmp_path)

    assert artifact.records == 1
    assert artifact.directory.parent == tmp_path / "results"
    assert (artifact.directory / "config.toml").is_file()
    assert (artifact.directory / "command.txt").is_file()
    assert (artifact.directory / "stderr.log").read_text(encoding="utf-8") == "fake diagnostic\n"
    record = json.loads((artifact.directory / "results.jsonl").read_text(encoding="utf-8"))
    assert record["experiment"]["name"] == "runner-test"
    assert record["model"]["source"] == "model.gguf"
    assert record["benchmark"]["avg_ts"] == 42.5
    assert record["benchmark"]["samples_ts"] == [41.0, 44.0]
    manifest = json.loads((artifact.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["result_count"] == 1


def test_failed_run_still_writes_diagnostic_artifact(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path, exit_code=7)
    config = _local_config(tmp_path, binary)

    with pytest.raises(BenchmarkError, match="status 7") as captured:
        run_sweep(config, root=tmp_path)

    artifact_dir = captured.value.artifact_dir
    assert artifact_dir is not None
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["return_code"] == 7
    assert (artifact_dir / "stderr.log").read_text(encoding="utf-8") == "fake diagnostic\n"


def test_peak_rss_parser_handles_macos_time_output() -> None:
    stderr = "        15518924800  maximum resident set size\n"

    assert parse_peak_rss(stderr) == 15_518_924_800
    assert parse_peak_rss("no telemetry") is None


def _fake_binary(tmp_path: Path, *, exit_code: int) -> Path:
    binary = tmp_path / f"fake-llama-bench-{exit_code}"
    binary.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'fake diagnostic' >&2\n"
        'printf \'%s\\n\' \'[{"build_commit":"abc123","n_prompt":0,"n_gen":128,'
        '"n_batch":512,"n_ubatch":256,"n_threads":10,"type_k":"f16",'
        '"type_v":"f16","flash_attn":1,"n_gpu_layers":99,'
        '"avg_ts":42.5,"stddev_ts":1.5,"samples_ts":[41.0,44.0]}]\'\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _local_config(tmp_path: Path, binary: Path):
    (tmp_path / "model.gguf").write_bytes(b"not-real-weights")
    path = tmp_path / "experiment.toml"
    path.write_text(
        f"""
[experiment]
name = "runner-test"
output_dir = "results"

[model]
label = "Fixture"
path = "model.gguf"
quantization = "fixture"

[runtime]
binary = "{binary}"
repetitions = 2
capture_peak_memory = false

[workload]
prompt_tokens = []
generation_tokens = [128]

[matrix]
batch_sizes = [512]
ubatch_sizes = [256]
""",
        encoding="utf-8",
    )
    return load_config(path)
