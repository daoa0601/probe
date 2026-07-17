# M4 Pro Gemma 4 E4B smoke baseline — 2026-07-17

Status: passed.

This is the first end-to-end validation of Probe on the reference Mac. It is a smoke baseline with
two repetitions, not yet an optimization claim.

## Provenance

| Component | Value |
| --- | --- |
| Host | Apple M4 Pro, 48 GiB unified memory, 20-core GPU |
| Runtime | llama.cpp `86d86ed4` with Metal and Accelerate |
| Model | `ggml-org/gemma-4-E4B-it-GGUF` |
| Model repository commit | `06f24bb269339b2a19a5167199b81e89ef813c10` |
| File | `gemma-4-E4B-it-Q4_0.gguf` (4,590,807,392 bytes) |
| SHA-256 | `a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2` |
| Config | `configs/gemma4-e4b-smoke.toml` |
| Successful artifact | `results/20260717T154408Z-gemma4-e4b-smoke-e66d584e` |

Parameters held fixed: 10 threads, 99 GPU layers, flash attention on, F16 K/V cache, batch 512,
ubatch 256, and memory mapping on.

## llama-bench results

| Workload | Mean | Standard deviation | Raw samples |
| --- | ---: | ---: | --- |
| `pp512` | 558.95 tok/s | 1.56 | 560.054, 557.848 |
| `tg64` | 36.10 tok/s | 0.17 | 35.9819, 36.2188 |

Peak resident set size was 4,669,308,928 bytes (4.35 GiB). The 179.6-second artifact wall time
includes the first model download and must not be interpreted as inference latency.

## Generation check

An independent single-turn `llama-cli` check ran with Metal offload, flash attention, temperature
zero, reasoning disabled, and a 2K context:

```text
User: What is 17 multiplied by 19? Reply with only the integer.
Gemma 4 E4B: 323
```

llama-cli reported 42.4 generated tok/s for this short response. This verifies coherent sampled
output but is not comparable to the fixed-length `tg64` microbenchmark.

## Compatibility finding

The first attempt used Google's newly published `google/gemma-4-E4B-it-qat-q4_0-gguf` file. Current
llama.cpp aborted while loading its vocabulary because the file contains duplicate tokenizer
entries. The failure is preserved at
`results/20260717T153749Z-gemma4-e4b-smoke-0d0f0e86`, and another user independently reported the
[same assertion](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/discussions/8). The
llama.cpp-maintained conversion loaded and ran successfully without changing benchmark parameters.
