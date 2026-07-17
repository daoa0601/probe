from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from probe import __version__
from probe.config import ConfigError, build_command, display_command, load_config
from probe.hardware import capture_host, discover_runtime_tools, project_root
from probe.report import ReportError, load_records, render_markdown
from probe.runner import BenchmarkError, run_sweep

app = typer.Typer(
    name="probe",
    help="Measure and optimize local Gemma 4 inference on Apple Silicon.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()
error_console = Console(stderr=True)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the Probe version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Inspect the host and local inference runtime prerequisites."""

    root = project_root()
    host = capture_host()
    tools = discover_runtime_tools(root)
    pin_path = root / "runtime" / "llama-cpp.ref"
    pin = pin_path.read_text(encoding="utf-8").strip() if pin_path.is_file() else None
    payload = {
        "host": host.to_dict(),
        "runtimes": [tool.to_dict() for tool in tools],
        "llama_cpp_pin": pin,
        "project_root": str(root),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    host_table = Table(title="Host", show_header=False)
    host_table.add_column("Field", style="cyan")
    host_table.add_column("Value")
    host_table.add_row("Machine", _optional(host.machine_name))
    host_table.add_row("Chip", _optional(host.chip))
    host_table.add_row(
        "CPU",
        f"{host.logical_cpu_cores} cores "
        f"({_optional(host.performance_cpu_cores)} performance, "
        f"{_optional(host.efficiency_cpu_cores)} efficiency)",
    )
    host_table.add_row("GPU", f"{_optional(host.gpu_cores)} cores")
    host_table.add_row("Unified memory", _format_bytes(host.memory_bytes))
    host_table.add_row("Metal", _optional(host.metal_support))
    host_table.add_row("OS", f"{host.operating_system} {host.os_version} ({host.architecture})")
    console.print(host_table)

    runtime_table = Table(title="Inference runtimes")
    runtime_table.add_column("Tool", style="cyan")
    runtime_table.add_column("Status")
    runtime_table.add_column("Version")
    runtime_table.add_column("Path")
    for tool in tools:
        runtime_table.add_row(
            tool.name,
            "ready" if tool.available else "missing",
            tool.version or "—",
            tool.path or "—",
        )
    console.print(runtime_table)
    if not next(tool.available for tool in tools if tool.name == "llama-bench"):
        console.print("Next: [bold]./scripts/bootstrap-llama-cpp.sh[/bold]")
    if pin:
        console.print(f"Pinned llama.cpp commit: {pin}")


@app.command()
def sweep(
    config_path: Path = typer.Argument(..., help="Experiment TOML configuration."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the complete command without executing it.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override the configured artifact root.",
    ),
) -> None:
    """Run or inspect a reproducible llama-bench parameter sweep."""

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _fail(str(exc))

    command = build_command(config)
    console.print(f"Experiment: [bold]{config.experiment.name}[/bold]")
    console.print(f"Model: {config.model.label} ({config.model.quantization})")
    console.print(
        f"Matrix: {config.matrix.combination_count} parameter combinations × "
        f"{config.workload.test_count} workloads = {config.estimated_test_cases} tests; "
        f"{config.runtime.repetitions} measured repetitions each"
    )
    if config.model.hf_repo and not config.runtime.offline:
        console.print(
            "Model source is Hugging Face; the first real run may download model weights."
        )
    if dry_run:
        console.print("\n[bold]Command[/bold]")
        console.print(display_command(command), soft_wrap=True)
        return

    console.print("Running benchmark; raw stdout and stderr will be retained even if it fails.")
    try:
        artifact = run_sweep(config, output_root=output_dir)
    except BenchmarkError as exc:
        suffix = f" Artifact: {exc.artifact_dir}" if exc.artifact_dir else ""
        _fail(f"{exc}.{suffix}")
    console.print(
        f"Completed {artifact.records} result rows in {artifact.wall_seconds:.1f}s. "
        f"Artifact: [bold]{artifact.directory}[/bold]"
    )


@app.command("compare")
def compare_command(
    result_paths: list[Path] = typer.Argument(
        ...,
        help="One or more results.jsonl files or artifact directories.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write Markdown here."),
    top: int = typer.Option(10, min=1, help="Rows to retain per workload."),
    test: str | None = typer.Option(None, help="Only show a label such as tg128 or pp4096."),
) -> None:
    """Rank comparable throughput results and render a Markdown report."""

    try:
        records = load_records(result_paths)
        markdown = render_markdown(records, top=top, only_test=test)
    except ReportError as exc:
        _fail(str(exc))
    if output is None:
        typer.echo(markdown)
        return
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    console.print(f"Wrote {destination}")


def _fail(message: str) -> NoReturn:
    error_console.print(f"Error: {message}", markup=False)
    raise typer.Exit(code=2)


def _optional(value: object) -> str:
    return str(value) if value is not None else "unknown"


def _format_bytes(value: int | None) -> str:
    return f"{value / (1024**3):.1f} GiB" if value is not None else "unknown"


if __name__ == "__main__":
    app()
