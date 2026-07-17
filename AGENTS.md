# Probe project context

Probe is an evidence-first inference optimization lab for Gemma 4 on Apple Silicon. The initial
reference machine is an Apple M4 Pro with 48 GB unified memory, 10 performance CPU cores, 4
efficiency CPU cores, and a 20-core GPU.

## Quick start

```bash
uv sync --all-extras
./scripts/bootstrap-llama-cpp.sh
uv run probe doctor
uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml
make check
```

## Architecture

- `src/probe/config.py` validates experiment TOML and builds deterministic `llama-bench` commands.
- `src/probe/hardware.py` captures a privacy-safe host fingerprint.
- `src/probe/runner.py` executes a sweep and writes immutable, self-describing artifacts.
- `src/probe/report.py` compares benchmark records without discarding raw measurements.
- `configs/` contains small, single-purpose experiment definitions.

## Conventions

- Never claim an optimization from a single run. Preserve repetitions and variance.
- Change one factor family at a time; use a new config when the hypothesis changes.
- Every result must retain the host fingerprint, llama.cpp build commit, model identity, complete
  command, and raw per-repetition samples.
- Every candidate benchmark must retain its fsynced `trace.jsonl`; optimization sessions and
  independent reviews use durable `.codex-subagents/` custody for exact prompts and raw outputs.
- Promotion-eligible configs require clean committed source, exact local model bytes, a precommitted
  correctness contract, start-environment gates, a declared replication tolerance, and a final
  `seal.json` covering every other artifact file.
- Do not edit prior result artifacts. Re-run into a new timestamped directory.
- Do not commit model weights, runtime build trees, tokens, machine serial numbers, UUIDs, hostnames,
  or usernames.
- Keep backends behind explicit adapters. llama.cpp/Metal is the first baseline; Ollama and MLX are
  comparison backends, not hidden fallbacks.
- Do not call a result a baseline or use it for a speedup claim until the independent audit protocol
  in `docs/optimization-audit-protocol.md` passes. Preserve audit disagreements and failures.
- The optimizing agent cannot approve its own headline result. A separate read-only agent must read
  final source, config, complete traces, and raw artifacts, then empirically replicate at least one
  workload and re-test suspicious caching, substitution, parsing, or grader-specific behavior.

## Testing

Run `make check`. Unit tests must not download models or require Apple hardware. Hardware-specific
experiments are manual and their prerequisites must be reported by `probe doctor`.
