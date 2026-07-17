from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe.report import ReportError, load_records, render_markdown
from probe.report import test_label as workload_label


def test_report_groups_workloads_and_ranks_within_each_group(tmp_path: Path) -> None:
    records = [
        _record("baseline", prompt=0, generation=128, throughput=30.0),
        _record("tuned", prompt=0, generation=128, throughput=45.0),
        _record("baseline", prompt=4096, generation=0, throughput=500.0),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    loaded = load_records([tmp_path])
    markdown = render_markdown(loaded)

    assert "## pp4096" in markdown
    assert "## tg128" in markdown
    assert markdown.index("| 1 | tuned") < markdown.index("| 2 | baseline")
    assert workload_label(records[0]) == "tg128"


def test_report_can_filter_to_one_workload() -> None:
    records = [
        _record("decode", prompt=0, generation=128, throughput=30.0),
        _record("prefill", prompt=512, generation=0, throughput=400.0),
    ]

    markdown = render_markdown(records, only_test="tg128")

    assert "## tg128" in markdown
    assert "pp512" not in markdown


def test_report_rejects_missing_or_unmatched_inputs(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="does not exist"):
        load_records([tmp_path / "missing"])
    with pytest.raises(ReportError, match="no records matched"):
        render_markdown([_record("x", 0, 128, 1.0)], only_test="pp512")


def _record(name: str, prompt: int, generation: int, throughput: float) -> dict[str, object]:
    return {
        "experiment": {"name": name},
        "model": {"label": "Gemma", "quantization": "Q4"},
        "execution": {"peak_rss_bytes": 16 * 1024**3},
        "benchmark": {
            "build_commit": "abc",
            "n_prompt": prompt,
            "n_gen": generation,
            "n_depth": 0,
            "n_batch": 512,
            "n_ubatch": 256,
            "n_threads": 10,
            "type_k": "f16",
            "type_v": "f16",
            "flash_attn": 1,
            "n_gpu_layers": 99,
            "avg_ts": throughput,
            "stddev_ts": 1.0,
        },
    }
