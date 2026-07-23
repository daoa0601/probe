# Probe

Probe is an evidence-first lab for optimizing a locally hosted **Gemma 4** model on Apple Silicon.
The reference host is this MacBook Pro: M4 Pro, 48 GB unified memory, 10 performance CPU cores, 4
efficiency CPU cores, and a 20-core GPU.

The project starts with a llama.cpp-maintained **Gemma 4 26B-A4B Q4_0** conversion and `llama.cpp`'s
Metal backend. The model activates about 4B parameters per token while keeping all 26B parameters
resident, making it a useful target for studying throughput, memory pressure, long context, CPU/GPU
placement, and speculative decoding. Google's estimate is about 14.4 GB to load Q4_0 weights with
basic overhead, before the KV cache and application memory.

The goal is not merely maximum tokens per second. Probe seeks the best configuration on this Mac
under explicit constraints:

- low time to first token and high decode throughput;
- enough context and concurrency for agentic workloads;
- bounded unified-memory pressure and energy use;
- no unacceptable loss in tool use, coding, reasoning, or long-context quality.

## Current milestone

The first milestone is a reproducible throughput and memory baseline:

- a pinned `llama.cpp` source revision and Metal build script;
- privacy-safe hardware discovery (serial numbers, UUIDs, usernames, and hostnames are excluded);
- strictly validated TOML experiment matrices;
- `llama-bench` prompt-processing and token-generation sweeps;
- timestamped artifacts containing the config, command, host, logs, raw samples, and normalized
  JSONL records;
- Markdown comparison reports that never rank prompt processing against token generation.

See [the research plan](docs/research-plan.md) for the path from microbenchmarks to server latency,
quality gates, long context, energy, concurrency, and Gemma 4 MTP speculative decoding.
The first measured result is recorded in the
[M4 Pro E4B smoke baseline](docs/baselines/2026-07-17-m4-pro-e4b-smoke.md).
The initial 26B-A4B result is a
[conditional baseline candidate](docs/baselines/2026-07-17-m4-pro-26b-a4b-baseline.md); its
independent audit found valid inference but blocked promotion on provenance and repeatability.
All promoted results follow the
[optimization and independent-audit protocol](docs/optimization-audit-protocol.md).

Probe imports streaming SHA-256 and detached artifact-seal primitives from the sibling
[`benchmark-audit-core`](../benchmark-audit-core/) package. Its model/runtime correctness checks,
clean-source and start-environment gates, replication tolerance, and independent promotion verdict
remain Probe-owned policy. Probe does not use the core's `evaluate_headline_gate()` result as its
promotion verdict.

The core is intentionally workspace-local, and this checkout resolves it through
`../benchmark-audit-core`. A relocated Probe clone or CI job must provide that sibling source,
vendor the same code, or use an explicitly pinned Git source. Regenerate and verify `uv.lock` when
the source location changes.

## Quick start

Requirements: macOS on Apple Silicon, Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Xcode command
line tools, CMake, and enough free disk space for the selected model.

```bash
brew install cmake                 # only needed when cmake is absent
uv sync --all-extras
./scripts/bootstrap-llama-cpp.sh   # builds the pinned Metal runtime under .cache/
uv run probe doctor
make check
```

Inspect the exact command and matrix before running anything expensive:

```bash
uv run probe sweep --dry-run configs/gemma4-e4b-smoke.toml
uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml
PROBE_GEMMA4_26B_MODEL=/absolute/path/to/model.gguf \
  uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline-v2.toml
PROBE_GEMMA4_12B_MODEL=/absolute/path/to/model.gguf \
  uv run probe sweep --dry-run configs/gemma4-12b-baseline-v1.toml
```

The first non-offline run downloads the configured GGUF from Hugging Face. Start with the smaller
E4B smoke model, then establish the 26B-A4B baseline:

```bash
uv run probe sweep configs/gemma4-e4b-smoke.toml
uv run probe sweep configs/gemma4-26b-a4b-baseline.toml
```

The 26B `baseline-v2` and 12B `baseline-v1` configs are promotion-eligible contracts. They use local
model paths so Probe can verify every weight byte before execution, reject dirty source or a
busy/unplugged host, lock a 5% independent-replication tolerance, copy the correctness contract into
the artifact, execute it as an exact parsed-output gate, and seal every finalized file. Run one only
after its model path is exported and the worktree is clean.

Compare every artifact beneath `results/`, or select a comparable workload:

```bash
uv run probe verify-artifact results/<run-id>
uv run probe compare results
uv run probe compare results --test tg128 --output reports/tg128.md
```

Results and weights are intentionally ignored by Git. Archive meaningful artifacts in external
storage before deleting them.

## Experiment progression

Run narrow sweeps in order; promote a setting only when repeated measurements beat the baseline.

1. Validate the toolchain with E4B and establish the 26B-A4B reference.
2. Tune logical batch, physical batch, thread count, flash attention, and memory mapping.
3. Explore KV-cache precision and context scaling while measuring quality and memory pressure.
4. Compare official QAT Q4_0 with Q4_K_M, Q5, Q8, and higher-precision variants.
5. Benchmark MoE CPU placement, concurrent requests, and continuous batching.
6. Add endpoint TTFT/p50/p95, energy-per-token, and agent-workload quality gates.
7. Evaluate the dedicated Gemma 4 assistant model for MTP speculative decoding.
8. Compare llama.cpp against Ollama and MLX only with identical models and workloads.

## Repository layout

```text
configs/             baseline and controlled parameter sweeps
docs/                research protocol and artifact contracts
models/              ignored local weight cache
runtime/             pinned runtime revisions
scripts/             reproducible local runtime bootstrap
src/probe/           CLI, config compiler, hardware capture, policy adapters, runner, and reporting
tests/               hardware-independent unit and CLI tests
```

## Upstream references

- [Google's Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Official Gemma 4 26B-A4B instruction model](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [Official Gemma 4 26B-A4B QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
- [llama.cpp Gemma 4 26B-A4B GGUF conversion](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF)
- [llama.cpp Gemma 4 12B GGUF conversion](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp)

Gemma 4 and its runtime support are evolving quickly. Probe pins every benchmarkable runtime and
records its build commit so historical comparisons remain interpretable.
