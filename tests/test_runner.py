from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from probe.config import load_config
from probe.evidence import EvidenceError, verify_artifact_seal
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
    assert manifest["schema_version"] == 2
    assert (artifact.directory / "source.json").is_file()
    assert (artifact.directory / "model.json").is_file()
    assert (artifact.directory / "environment-start.json").is_file()
    assert (artifact.directory / "environment-end.json").is_file()
    seal = json.loads((artifact.directory / "seal.json").read_text(encoding="utf-8"))
    for name, evidence in seal["files"].items():
        content = (artifact.directory / name).read_bytes()
        assert evidence == {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    verification = verify_artifact_seal(artifact.directory)
    assert verification["files_verified"] == len(seal["files"])
    assert len(verification["seal_sha256"]) == 64
    trace = [
        json.loads(line)
        for line in (artifact.directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in trace] == [
        "artifact_created",
        "source_captured",
        "model_verified",
        "host_captured",
        "environment_captured",
        "command_compiled",
        "process_started",
        "process_finished",
        "environment_captured",
        "output_parsed",
        "run_finalized",
    ]
    assert trace[-1]["status"] == "succeeded"


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
    trace = (artifact_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert '"event": "run_finalized"' in trace
    assert '"status": "failed"' in trace


def test_artifact_verifier_rejects_mutation(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path, exit_code=0)
    config = _local_config(tmp_path, binary)
    artifact = run_sweep(config, root=tmp_path)
    (artifact.directory / "stdout.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="seal mismatch: stdout.json"):
        verify_artifact_seal(artifact.directory)


def test_peak_rss_parser_handles_macos_time_output() -> None:
    stderr = "        15518924800  maximum resident set size\n"

    assert parse_peak_rss(stderr) == 15_518_924_800
    assert parse_peak_rss("no telemetry") is None


def test_promotion_run_captures_clean_source_and_locked_contract(tmp_path: Path) -> None:
    config = _promotion_config(tmp_path)
    _initialize_git(tmp_path)

    artifact = run_sweep(config, root=tmp_path)

    source = json.loads((artifact.directory / "source.json").read_text(encoding="utf-8"))
    promotion = json.loads((artifact.directory / "promotion.json").read_text(encoding="utf-8"))
    assert source["available"] is True
    assert source["dirty"] is False
    assert source["head"] == _git(tmp_path, "rev-parse", "HEAD").strip()
    assert promotion["replication_tolerance_percent"] == 5.0
    assert (
        promotion["correctness_contract"]["sha256"]
        == hashlib.sha256((tmp_path / "correctness.json").read_bytes()).hexdigest()
    )
    assert (artifact.directory / "correctness-contract.json").read_bytes() == (
        tmp_path / "correctness.json"
    ).read_bytes()


def test_promotion_run_rejects_dirty_repository(tmp_path: Path) -> None:
    config = _promotion_config(tmp_path)
    _initialize_git(tmp_path)
    (tmp_path / "untracked-source.py").write_text("dirty = True\n", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="clean Git repository") as captured:
        run_sweep(config, root=tmp_path)

    assert captured.value.artifact_dir is None


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


def _promotion_config(tmp_path: Path):
    binary = _fake_binary(tmp_path, exit_code=0)
    model_content = b"not-real-weights"
    (tmp_path / "model.gguf").write_bytes(model_content)
    (tmp_path / "correctness.json").write_text(
        '{"id":"fixture","prompt":"return 1","expected":1}\n',
        encoding="utf-8",
    )
    path = tmp_path / "promotion.toml"
    path.write_text(
        f"""
[experiment]
name = "promoted-runner-test"
output_dir = "results"

[model]
path = "model.gguf"
sha256 = "{hashlib.sha256(model_content).hexdigest()}"
size_bytes = {len(model_content)}

[runtime]
binary = "{binary}"
capture_peak_memory = false

[workload]

[matrix]

[promotion]
require_clean_repository = true
correctness_contract = "correctness.json"
replication_tolerance_percent = 5.0
max_load_1m = 10000.0
""",
        encoding="utf-8",
    )
    return load_config(path)


def _initialize_git(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Probe Tests",
        "-c",
        "user.email=probe@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
