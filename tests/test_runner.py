from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from probe.config import load_config
from probe.evidence import EvidenceError, verify_artifact_seal
from probe.runner import BenchmarkError, parse_peak_rss, run_sweep
from probe.runner import _parse_correctness_transcript as parse_correctness_transcript
from probe.runner import _validate_benchmark_rows as validate_benchmark_rows
from probe.runner import _validate_correctness_contract as validate_correctness_contract
from probe.runner import _validate_environment_for_promotion as validate_promotion_environment


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


def test_successful_process_rejects_incomplete_benchmark_matrix(tmp_path: Path) -> None:
    binary = tmp_path / "fake-empty-llama-bench"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' '[]'\n", encoding="utf-8")
    binary.chmod(0o755)
    config = _local_config(tmp_path, binary)

    with pytest.raises(BenchmarkError, match="result count") as captured:
        run_sweep(config, root=tmp_path)

    artifact_dir = captured.value.artifact_dir
    assert artifact_dir is not None
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["result_count"] == 0
    assert manifest["parse_error"] == (
        "llama-bench result count does not match the declared matrix: observed 0, expected 1"
    )
    verify_artifact_seal(artifact_dir)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_prompt": 0, "n_gen": 0}, "zero-token workload"),
        ({"n_batch": 2048}, "settings do not match"),
        ({"model_filename": "/wrong/model.gguf"}, "model path does not match"),
        ({"samples_ts": [41.0]}, "samples_ts count"),
        ({"stddev_ts": 1.0}, "stddev_ts does not reproduce"),
    ],
)
def test_benchmark_semantics_reject_invalid_rows(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    binary = _fake_binary(tmp_path, exit_code=0)
    config = _local_config(tmp_path, binary)
    row = _fixture_benchmark_row(tmp_path)
    row.update(changes)

    with pytest.raises(ValueError, match=message):
        validate_benchmark_rows([row], config, tmp_path)


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


def _generation(**overrides: object) -> dict[str, object]:
    generation: dict[str, object] = {
        "context_size": 128,
        "max_tokens": 8,
        "temperature": 0,
        "seed": 42,
        "reasoning": False,
    }
    generation.update(overrides)
    return generation


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        ({"extra": True}, "unknown correctness contract keys"),
        ({"id": "", "prompt": "p", "expected": 1, "generation": {}}, "id must"),
        ({"id": "x", "prompt": "", "expected": 1, "generation": {}}, "prompt must"),
        ({"id": "x", "prompt": "p", "generation": {}}, "define expected"),
        ({"id": "x", "prompt": "p", "expected": 1}, "generation must be an object"),
        (
            {"id": "x", "prompt": "p", "expected": 1, "generation": {}},
            "generation keys must be",
        ),
        (
            {
                "id": "x",
                "prompt": "p",
                "expected": 1,
                "generation": _generation(context_size=0),
            },
            "context_size must be a positive integer",
        ),
        (
            {
                "id": "x",
                "prompt": "p",
                "expected": 1,
                "generation": _generation(temperature=-1),
            },
            "temperature must be",
        ),
        (
            {
                "id": "x",
                "prompt": "p",
                "expected": 1,
                "generation": _generation(seed=True),
            },
            "seed must be an integer",
        ),
        (
            {
                "id": "x",
                "prompt": "p",
                "expected": 1,
                "generation": _generation(reasoning="off"),
            },
            "reasoning must be a boolean",
        ),
    ],
)
def test_correctness_contract_rejects_ambiguous_inputs(
    contract: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(EvidenceError, match=message):
        validate_correctness_contract(contract, Path("contract.json"))


def test_promotion_environment_rejects_missing_or_excess_load(tmp_path: Path) -> None:
    config = _promotion_config(tmp_path)

    with pytest.raises(EvidenceError, match="available 1-minute load"):
        validate_promotion_environment(config, {})
    with pytest.raises(EvidenceError, match="start load is too high"):
        validate_promotion_environment(config, {"load_average": [10_001.0]})


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
    correctness = json.loads(
        (artifact.directory / "correctness-result.json").read_text(encoding="utf-8")
    )
    assert correctness["status"] == "pass"
    assert correctness["observed"] == 1
    assert correctness["exact_match"] is True
    assert correctness["response_source"] == "correctness-transcript.txt"
    assert (artifact.directory / "correctness-stdout.json").read_text(
        encoding="utf-8"
    ) == "fake cli console\n"
    assert (artifact.directory / "correctness-transcript.txt").read_text(
        encoding="utf-8"
    ) == "User:\nreturn 1\n\nAssistant:\n1\n\n"


def test_correctness_transcript_parser_ignores_console_and_prompt_json() -> None:
    prompt = 'Return {"type":"object"} and no prose.'
    transcript = f'User:\n{prompt}\n\nAssistant:\n{{"answer": 1}}\n\n'

    assert parse_correctness_transcript(transcript, prompt) == {"answer": 1}


@pytest.mark.parametrize(
    ("transcript", "message"),
    [
        ('Assistant:\n{"answer":1}\n\n', "exact user/assistant frame"),
        ('User:\np\n\nAssistant:\n{"answer":1}', "assistant delimiter"),
        ("User:\np\n\nAssistant:\n\n\n", "empty assistant response"),
        (
            'User:\np\n\nAssistant:\n{"answer":1}\\nprose\n\n',
            "not exactly one JSON value",
        ),
        (
            'User:\np\n\nAssistant:\n{"answer":1}\n\nAssistant:\n{"answer":1}\n\n',
            "not exactly one JSON value",
        ),
    ],
)
def test_correctness_transcript_parser_rejects_ambiguous_output(
    transcript: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_correctness_transcript(transcript, "p")


def test_promotion_run_rejects_dirty_repository(tmp_path: Path) -> None:
    config = _promotion_config(tmp_path)
    _initialize_git(tmp_path)
    (tmp_path / "untracked-source.py").write_text("dirty = True\n", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="clean Git repository") as captured:
        run_sweep(config, root=tmp_path)

    assert captured.value.artifact_dir is None


def test_promotion_run_rejects_incorrect_model_output(tmp_path: Path) -> None:
    config = _promotion_config(tmp_path, correctness_output="2")
    _initialize_git(tmp_path)

    with pytest.raises(BenchmarkError, match="correctness check fail") as captured:
        run_sweep(config, root=tmp_path)

    artifact_dir = captured.value.artifact_dir
    assert artifact_dir is not None
    result = json.loads((artifact_dir / "correctness-result.json").read_text(encoding="utf-8"))
    assert result["observed"] == 2
    assert result["exact_match"] is False
    verify_artifact_seal(artifact_dir)


def _fake_binary(tmp_path: Path, *, exit_code: int) -> Path:
    binary = tmp_path / f"fake-llama-bench-{exit_code}"
    output = json.dumps([_fixture_benchmark_row(tmp_path)], separators=(",", ":"))
    binary.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'fake diagnostic' >&2\n"
        f"printf '%s\\n' '{output}'\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def _fixture_benchmark_row(tmp_path: Path) -> dict[str, object]:
    return {
        "build_commit": "abc123",
        "n_prompt": 0,
        "n_gen": 128,
        "n_batch": 512,
        "n_ubatch": 256,
        "n_threads": 10,
        "type_k": "f16",
        "type_v": "f16",
        "flash_attn": 1,
        "n_gpu_layers": 99,
        "n_cpu_moe": 0,
        "use_mmap": True,
        "no_kv_offload": False,
        "model_filename": str(tmp_path / "model.gguf"),
        "avg_ts": 42.5,
        "stddev_ts": 2.12132,
        "samples_ns": [3_121_951_220, 2_909_090_909],
        "samples_ts": [41.0, 44.0],
    }


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


def _promotion_config(tmp_path: Path, *, correctness_output: str = "1"):
    binary = _fake_binary(tmp_path, exit_code=0)
    correctness_binary = _fake_correctness_binary(tmp_path, output=correctness_output)
    model_content = b"not-real-weights"
    (tmp_path / "model.gguf").write_bytes(model_content)
    (tmp_path / "correctness.json").write_text(
        '{"id":"fixture","prompt":"return 1","expected":1,'
        '"generation":{"context_size":128,"max_tokens":8,"temperature":0,'
        '"seed":42,"reasoning":false}}\n',
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
repetitions = 2
capture_peak_memory = false

[workload]
prompt_tokens = []
generation_tokens = [128]

[matrix]
batch_sizes = [512]
ubatch_sizes = [256]

[promotion]
require_clean_repository = true
correctness_binary = "{correctness_binary}"
correctness_contract = "correctness.json"
replication_tolerance_percent = 5.0
max_load_1m = 10000.0
""",
        encoding="utf-8",
    )
    return load_config(path)


def _fake_correctness_binary(tmp_path: Path, *, output: str) -> Path:
    binary = tmp_path / "fake-llama-cli"
    binary.write_text(
        "#!/bin/sh\n"
        "output_file=''\n"
        'while [ "$#" -gt 0 ]; do\n'
        "  if [ \"$1\" = '--output' ]; then\n"
        "    shift\n"
        '    output_file="$1"\n'
        "  fi\n"
        "  shift\n"
        "done\n"
        "printf '%s\\n' 'fake correctness diagnostic' >&2\n"
        "printf '%s\\n' 'fake cli console'\n"
        f"printf 'User:\\nreturn 1\\n\\nAssistant:\\n%s\\n\\n' '{output}' > \"$output_file\"\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


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
