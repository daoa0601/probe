# Experiment configurations

Each TOML file states one hypothesis and expands to a deterministic `llama-bench` matrix. Inspect a
matrix before spending time or downloading weights:

```bash
uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml
```

Suggested order:

1. `gemma4-e4b-smoke.toml` validates the runtime with the smaller model.
2. `gemma4-26b-a4b-baseline.toml` establishes the target-model reference.
3. `sweeps/batch-and-ubatch.toml` tunes Metal prompt processing.
4. `sweeps/threads.toml` checks whether efficiency cores help or hurt.
5. `sweeps/kv-cache.toml` identifies candidates for long-context memory and quality experiments.

Matrix axes form a Cartesian product. Keep configs narrow: nine explicit combinations are easier to
interpret than one giant sweep whose factors interact.
