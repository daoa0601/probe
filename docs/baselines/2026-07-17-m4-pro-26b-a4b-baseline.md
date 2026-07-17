# M4 Pro Gemma 4 26B-A4B baseline candidate — 2026-07-17

Status: **candidate; independent audit pending**.

This result cannot anchor speedup claims until the audit in
`docs/optimization-audit-protocol.md` reaches PASS. All repetitions, including the late-run
slowdown, are retained.

## Provenance

| Component | Value |
| --- | --- |
| Host | Apple M4 Pro, 48 GiB unified memory, 10 performance + 4 efficiency CPU cores, 20-core GPU |
| OS | macOS 27.0, arm64 |
| Probe source used by runner | Git commit `2dca889` |
| Runtime | llama.cpp `86d86ed4` with Metal and Accelerate |
| Model | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| Model repository commit | `3d3dca2094ff8112005fd10fc7a8e30cf4f45b56` |
| File | `gemma-4-26B-A4B-it-Q4_0.gguf` (14,618,145,824 bytes) |
| Verified SHA-256 | `d208665ab1cd3a69f7a9a4bc59430e8448c8093d9b06334f566ac59d6d504a03` |
| Config | `configs/gemma4-26b-a4b-baseline.toml` |
| Primary artifact | `results/20260717T155533Z-gemma4-26b-a4b-baseline-c1087991` |
| Execution trace | `results/20260717T155533Z-gemma4-26b-a4b-baseline-c1087991/trace.jsonl` |

The benchmark used one Metal device, 10 threads, full GPU-layer offload, flash attention, F16 K/V
cache, batch 2048, ubatch 512, mmap, warmup, and five measured repetitions. Prompt processing and
generation are reported separately.

## Primary measurements

| Workload | Mean | Standard deviation | Raw samples (tok/s) |
| --- | ---: | ---: | --- |
| `pp512` | 617.88 tok/s | 5.84 | 612.416, 625.846, 621.585, 616.994, 612.541 |
| `pp4096` | 544.52 tok/s | 40.94 | 577.246, 565.974, 549.830, 555.860, 473.684 |
| `tg128` | 43.11 tok/s | 1.38 | 43.5769, 43.6335, 43.7670, 43.9250, 40.6627 |

The 629.7-second wall time includes the first 14.6 GB download and is not request or model-load
latency. `/usr/bin/time -l` reported peak process RSS of 5,589,811,200 bytes (5.21 GiB), but that
does not include all Metal/unified-memory residency and must not be described as total model memory.

## Correctness check

The exact deterministic contract is `workloads/correctness/tool-call-smoke.json`. With temperature
zero, seed 42, reasoning disabled, a 2K context, the same verified model blob, Metal offload, and
flash attention, the model returned:

```json
{"tool": "multiply", "arguments": {"a": 17, "b": 19}}
```

The parsed object exactly matched the locked expected value. The earlier exploratory prompt and its
alternate schema are preserved in the workflow evidence instead of being silently discarded.

## Primary validity review

- The pinned llama-bench implementation calls `llama_decode` for every declared prompt or
  generation token and clears model memory state before every measured repetition.
- The warmup occurs before the measured loop. With `n_depth=0`, the optional saved-depth cache path
  is inactive. The loaded model remains resident across compatible tests, so this is warm
  steady-state throughput rather than cold-load performance.
- All five samples are present and the llama-bench aggregates reproduce from them within output
  rounding.
- The benchmark reports build `86d86ed4`, backends `MTL,BLAS`, GPU `Apple M4 Pro`, model repository
  snapshot `3d3dca2…`, and 25,233,142,046 model parameters.
- The model file's locally computed SHA-256 matches its content-addressed remote ETag.

## Run conditions and caveats

- The Mac was connected to AC power at 69% charge. AC `powermode` was 0 (Automatic), not forced High
  Power mode.
- macOS reported no thermal or performance warning.
- System swap already had 16,415 MiB allocated after the run. The benchmark process reported zero
  swaps, but system-level memory history remains a confounder.
- The fifth `pp4096` sample was about 15% below the first four; the fifth `tg128` sample was about 7%
  lower. Thermal or background drift is suspected but not established.
- No energy, total unified-memory high-water mark, or request-level TTFT was measured.

## Audit findings

Independent audit pending. The auditor is required to read final source, configuration, workflow
ledger, raw artifact, and trace; verify model/runtime identity; run at least one offline empirical
replication; investigate the late-run degradation and memory-accounting gap; and publish a
PASS/CONDITIONAL/FAIL verdict here without deleting primary samples.
