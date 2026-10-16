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
The [Metal profiling and native-code gate](docs/metal-profiling.md) defines the mandatory
compiler/device-binary, isolated-kernel, full-graph, throughput, and correctness loop.
The first measured result is recorded in the
[M4 Pro E4B smoke baseline](docs/baselines/2026-07-17-m4-pro-e4b-smoke.md).
The initial 26B-A4B result is a
[conditional baseline candidate](docs/baselines/2026-07-17-m4-pro-26b-a4b-baseline.md); its
independent audit found valid inference but blocked promotion on provenance and repeatability.
All promoted results follow the
[optimization and independent-audit protocol](docs/optimization-audit-protocol.md).

The first retained kernel optimization is
[`runtime/patches/0001-metal-m4-pro-q4-0-nsg4.patch`](runtime/patches/0001-metal-m4-pro-q4-0-nsg4.patch).
It specializes the Q4_0 Metal matrix-vector dispatch for M4 Pro from two to four SIMD groups while
leaving other Apple GPUs unchanged. Direct, interleaved 10-repetition checks under a busy host
measured 26B decode at 65.259 versus 63.854 tokens/s (+2.20%) and 12B decode at 28.475 versus 28.154
tokens/s (+1.14%). The candidate kept the deterministic tool-call output exact; 26B prompt
processing remained within 0.3% of the reference. The bootstrap script applies the patch to the
pinned runtime and safely recognizes an already-patched checkout.

The next retained candidate is
[`runtime/patches/0002-metal-fuse-gemma4-moe-weighted-sum.patch`](runtime/patches/0002-metal-fuse-gemma4-moe-weighted-sum.patch).
For every 26B-A4B layer it replaces two `[2816,8]` scaling passes plus the eight-expert add chain
with one vectorized Metal kernel. Stage-boundary GPU timestamps measured 0.00350 ms versus 0.01410
ms for the exact three-dispatch reference shape (4.03×), while a 32-token Metal System Trace reduced
the dominant command-buffer median from 14.148 ms to 13.104 ms. Two balanced busy-host pairs
improved decode throughput by 3.26% and 3.05%; a third load-skewed pair is not used as evidence.
The deterministic tool-call transcript stayed byte-identical. Gemma 4 12B is a 48-layer dense
graph and intentionally does not select this MoE-only fusion.

The current dense-decode hill climb is packaged as
[`runtime/patches/0003-metal-gemma4-retained-decode-fusions.patch`](runtime/patches/0003-metal-gemma4-retained-decode-fusions.patch).
It adds four M4 Pro-specific families: fused Q4_0 gate/up/GEGLU, fused Q4_0
down-projection/RMS-norm/residual/scale, fused RMS/scale/RoPE with direct K/V cache writes, and a
single-workgroup direct-output FlashAttention path for Gemma 4 cache lengths through 256 tokens.
The FlashAttention cutoff is empirical: stabilized one-token GPU timestamps still improved by
11.7% for the local head and 1.6% for the global head at 256 tokens, then regressed by 6.9% and
23.8% at 288 tokens when a second serial cache pass became necessary. Every fusion has an
environment-variable kill switch and exact shape/type/use-count guards. The individual
end-to-end results are deliberately not added together; see
[the Metal hill-climb record](docs/metal-hill-climb.md) for retained and pruned measurements. Four
balanced all-enabled/all-disabled pairs improved 12B decode by 1.56%, 3.78%, 1.92%, and 4.12%;
pooled throughput was 29.133 versus 28.329 tokens/s (+2.84%).

[`runtime/patches/0004-metal-q4-nsg-control.patch`](runtime/patches/0004-metal-q4-nsg-control.patch)
adds the bounded `GGML_METAL_Q4_NSG` control used to restore upstream NSG2 during those A/B runs.
It does not change the default: M4 Pro remains on the retained NSG4 path, other Apple GPUs remain
on the upstream selector, and invalid override values are ignored.

The current 12B throughput frontier combines those retained Metal kernels with a separately
quantized Q4_0 output head and Gemma 4's dedicated MTP assistant. Quantizing only `output.weight`
from Q6_K to Q4_0 improved a balanced ten-sample decode comparison by 3.36%, while preserving the
Q8_0 input embedding; its two-chunk quality result was KLD 0.003577, RMS probability delta 1.098%,
and the same top token at 92.857% of positions. With two draft tokens and `p_min=0.995`, a
three-task ABBA server comparison then measured 35.662 versus 31.823 tokens/s for MTP versus the
same target alone: +12.065% mean and +9.926% median, with 95.98% draft acceptance and byte-identical
final choices. These gains are separate experiments and are not added together. The complete
kernel portfolio, native G16S evidence, rejected candidates, and measurement caveats are in the
[Metal hill-climb record](docs/metal-hill-climb.md).

Probe imports streaming SHA-256 and detached artifact-seal primitives from the sibling
[`benchmark-audit-core`](../benchmark-audit-core/) package. Its model/runtime correctness checks,
clean-source and start-environment gates, replication tolerance, and independent promotion verdict
remain Probe-owned policy. Probe does not use the core's `evaluate_headline_gate()` result as its
promotion verdict.

The core is resolved from the sibling checkout `../benchmark-audit-core`; CI checks it out next to
Probe. See [the catalog README](../../README.md#cross-repo-boundaries) for the layout and
cross-repo boundaries.

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
- [Unsloth Gemma 4 12B QAT GGUF and smart assistant](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
- [Fable Gemma 4 WebGPU kernel workspace](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp)

Gemma 4 and its runtime support are evolving quickly. Probe pins every benchmarkable runtime and
records its build commit so historical comparisons remain interpretable.
