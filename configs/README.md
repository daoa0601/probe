# Experiment configurations

Each TOML file states one hypothesis and expands to a deterministic `llama-bench` matrix. Inspect a
matrix before spending time or downloading weights:

```bash
uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml
```

Suggested order:

1. `gemma4-e4b-smoke.toml` validates the runtime with the smaller model.
2. `gemma4-26b-a4b-baseline.toml` reproduces the original exploratory target-model candidate.
3. `gemma4-26b-a4b-baseline-v2.toml` is the promotion-eligible baseline contract. It requires
   `PROBE_GEMMA4_26B_MODEL` to point at the exact pre-hashed local GGUF, a clean Git tree, AC power,
   start load at or below 8.0, ten repetitions, a precommitted correctness contract, and independent
   agreement within 5%.
4. `sweeps/batch-and-ubatch.toml` tunes Metal prompt processing.
5. `sweeps/threads.toml` checks whether efficiency cores help or hurt.
6. `sweeps/kv-cache.toml` identifies candidates for long-context memory and quality experiments.

Matrix axes form a Cartesian product. Keep configs narrow: nine explicit combinations are easier to
interpret than one giant sweep whose factors interact.
