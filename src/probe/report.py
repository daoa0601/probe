from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ReportError(ValueError):
    """Raised when result evidence cannot be loaded safely."""


def load_records(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    """Load Probe JSONL records from files or recursively from artifact directories."""

    files: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            files.update(path.rglob("results.jsonl"))
        elif path.is_file():
            files.add(path)
        else:
            raise ReportError(f"result path does not exist: {path}")
    if not files:
        raise ReportError("no results.jsonl files found")

    records: list[dict[str, Any]] = []
    for path in sorted(files):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReportError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("benchmark"), dict):
                raise ReportError(f"invalid Probe record at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ReportError("result files contained no records")
    return records


def test_label(record: dict[str, Any]) -> str:
    benchmark = _benchmark(record)
    prompt = _integer(benchmark.get("n_prompt"))
    generation = _integer(benchmark.get("n_gen"))
    depth = _integer(benchmark.get("n_depth"))
    if prompt > 0 and generation > 0:
        label = f"pp{prompt}+tg{generation}"
    elif prompt > 0:
        label = f"pp{prompt}"
    elif generation > 0:
        label = f"tg{generation}"
    else:
        label = "empty"
    return f"{label}@d{depth}" if depth > 0 else label


def render_markdown(
    records: list[dict[str, Any]],
    *,
    top: int = 10,
    only_test: str | None = None,
) -> str:
    """Render comparable tests in separate ranked tables so PP and TG are never conflated."""

    if top < 1:
        raise ReportError("top must be >= 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = test_label(record)
        if only_test is None or only_test == label:
            grouped[label].append(record)
    if not grouped:
        raise ReportError(f"no records matched test {only_test!r}")

    lines = ["# Probe benchmark comparison", ""]
    for label in sorted(grouped):
        ranked = sorted(grouped[label], key=_throughput, reverse=True)[:top]
        lines.extend(
            [
                f"## {label}",
                "",
                "| Rank | Experiment | Model | Quant | Avg tok/s | SD | Batch | uBatch | "
                "Threads | KV | FA | GPU layers | Peak RSS | Build |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | "
                "---: | ---: | --- |",
            ]
        )
        for rank, record in enumerate(ranked, start=1):
            benchmark = _benchmark(record)
            model = _mapping(record.get("model"))
            experiment = _mapping(record.get("experiment"))
            execution = _mapping(record.get("execution"))
            kv = f"{benchmark.get('type_k', '?')}/{benchmark.get('type_v', '?')}"
            fields = (
                str(rank),
                _escape(experiment.get("name", "?")),
                _escape(model.get("label", benchmark.get("model_type", "?"))),
                _escape(model.get("quantization", "?")),
                _number(benchmark.get("avg_ts"), 2),
                _number(benchmark.get("stddev_ts"), 2),
                str(benchmark.get("n_batch", "?")),
                str(benchmark.get("n_ubatch", "?")),
                str(benchmark.get("n_threads", "?")),
                _escape(kv),
                _flash_attention(benchmark.get("flash_attn")),
                str(benchmark.get("n_gpu_layers", "?")),
                _bytes_gib(execution.get("peak_rss_bytes")),
                _escape(benchmark.get("build_commit", "?")),
            )
            lines.append("| " + " | ".join(fields) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _benchmark(record: dict[str, Any]) -> dict[str, Any]:
    return _mapping(record.get("benchmark"))


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _throughput(record: dict[str, Any]) -> float:
    value = _benchmark(record).get("avg_ts")
    return float(value) if isinstance(value, int | float) else float("-inf")


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(value: object, digits: int) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, int | float) else "?"


def _bytes_gib(value: object) -> str:
    if not isinstance(value, int | float):
        return "?"
    return f"{float(value) / (1024**3):.2f} GiB"


def _flash_attention(value: object) -> str:
    if value is True or value == 1:
        return "on"
    if value is False or value == 0:
        return "off"
    if value == -1:
        return "auto"
    return _escape(value)


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
