# Research plan

## Research question

What is the highest-utility way to serve Gemma 4 locally on the reference M4 Pro under the latency,
memory, energy, context, concurrency, and quality demands of an agentic orchestrator?

"Highest utility" is a constrained objective, not one throughput number. A configuration is useful
only if it fits the target context and concurrency, remains responsive under memory pressure, and
passes fixed quality gates.

## Reference target

The initial target is `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, file
`gemma-4-26B_q4_0-it.gguf`, running through a pinned `llama.cpp` Metal build. Google describes this
as a 26B-parameter mixture-of-experts model with about 4B active parameters per token. The full
weights remain resident. Google's published Q4_0 loading estimate is roughly 14.4 GB with basic
overhead and excludes KV-cache growth.

Start with text-only workloads. Multimodal projection and audio/video support introduce different
memory and runtime paths and should get their own experiments.

## Metrics

Every optimization report should eventually cover four layers:

| Layer | Primary metrics | Why it matters |
| --- | --- | --- |
| Kernel | prompt tok/s, decode tok/s, variance, peak RSS | Separates raw runtime behavior from serving overhead. |
| Serving | TTFT, inter-token latency, p50/p95/p99 latency, requests/s | Captures the experience seen by an orchestrator. |
| System | memory pressure, swap, energy/token, temperature, sustained throughput | Rejects fast but unstable laptop configurations. |
| Utility | tool-call validity, task success, long-context recall, coding/reasoning quality | Prevents quantization or cache wins from silently harming the model. |

The implemented milestone records kernel throughput, per-repetition samples, wall time, and macOS
peak resident memory. The remaining metrics are explicit future work rather than inferred from
microbenchmarks.

## Experimental protocol

1. Connect power, disable Low Power Mode, close unrelated heavy workloads, and record OS/runtime
   changes. Do not alter thermal controls between compared runs.
2. Run the E4B smoke config after each runtime upgrade.
3. Use five or more measured repetitions for optimization claims. Retain warmups.
4. Compare like with like: the same model bits, prompt/generation lengths, sampler, context,
   concurrency, and host state.
5. Change one factor family per config. Do not combine quantization, batching, and cache changes in a
   single attribution claim.
6. Re-run the baseline before and after a long sweep to expose thermal or background-load drift.
7. Preserve failed artifacts. A crash, timeout, or allocation failure defines part of the feasible
   region.
8. Promote a configuration only when its confidence interval and quality gates support the claim.

Recommended microbenchmarks:

- `pp512`: short interactive prompt processing;
- `pp4096`: typical agent state or tool transcript ingestion;
- `tg128`: short response decoding;
- later: combined `pp4096+tg256`, 32K/64K/128K context, and concurrent server workloads.

## Experiment phases

### Phase 0 — reproducibility

- Pin llama.cpp and capture its reported build commit.
- Validate hardware discovery and artifact completeness.
- Run E4B smoke, then the 26B-A4B baseline twice.
- Confirm that repeated baselines remain within a predeclared drift tolerance (start with 5%).

### Phase 1 — Metal execution parameters

Tune batch size, ubatch size, performance-core vs all-core thread counts, flash attention, mmap, and
KV offload. Optimize prompt and generation paths separately. A parameter that wins `pp4096` may not
win `tg128`.

### Phase 2 — memory and long context

Measure F16, Q8_0, and Q4_0 KV caches at increasing context. Capture memory pressure and swap in
addition to process RSS. Gemma 4's KV state and context checkpoints can be large; leave operating
system headroom instead of treating all 48 GB as model memory. Validate long-context recall for each
cache precision.

### Phase 3 — weight quantization

Compare official QAT Q4_0 against equivalent GGUF variants at Q4_K_M, Q5, Q8, and BF16 where
feasible. Record exact file digests. Throughput and memory comparisons without a fixed quality suite
are exploratory only.

### Phase 4 — MoE placement

Sweep `--n-cpu-moe` to test whether placing some expert layers on CPU frees useful unified memory or
improves scheduling. Apple unified memory does not make CPU and GPU execution costs equivalent, so
measure rather than infer.

### Phase 5 — serving and concurrency

Add a managed `llama-server` adapter and streaming client. Measure cold start, TTFT, inter-token
latency, sustained p95, continuous batching, prompt caching, context checkpoints, and 1/2/4-way
concurrency using fixed agent traces.

### Phase 6 — MTP speculative decoding

Gemma 4 provides dedicated assistant/draft weights. Compare target-only decoding with the assistant
model while sweeping draft length and precision. Record acceptance rate and total memory, not just
target-model tokens per second. Runtime support has changed rapidly, which makes the pinned build
especially important.

### Phase 7 — application utility

Replay representative orchestration workloads: structured tool calls, code edits, multi-turn state,
long tool outputs, and multimodal inputs. Choose a Pareto set across latency, memory, energy, and
quality instead of declaring one universal winner.

## Initial hypotheses

These are testable starting points, not conclusions:

- Full Metal offload plus flash attention will be the strongest baseline for the Q4 target.
- Using the 10 performance cores may be steadier than using all 14 CPU cores for the host-side work.
- Larger logical batches will mainly benefit prompt processing; decode may be memory-bandwidth bound.
- Quantized KV cache will unlock longer useful contexts, but Q4 cache may require a quality guardrail.
- MTP can improve decode throughput only if assistant acceptance offsets its compute and memory cost.

## Sources and date

The target assumptions were checked on 2026-07-17 against:

- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 26B-A4B model card](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [`llama.cpp` Gemma 4 MTP support discussion](https://github.com/ggml-org/llama.cpp/discussions/22735)

Re-check upstream behavior when changing `runtime/llama-cpp.ref`.
