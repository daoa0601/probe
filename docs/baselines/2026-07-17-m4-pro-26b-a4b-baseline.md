# M4 Pro Gemma 4 26B-A4B baseline candidate — 2026-07-17

Status: **conditional candidate; not promoted**.

This result cannot anchor speedup claims. The independent audit validated the model, runtime,
token work, raw samples, aggregate parsing, and deterministic output, but found that this run does
not satisfy the promotion invariant in `docs/optimization-audit-protocol.md`. All repetitions,
including the late-run slowdown, remain part of the evidence.

## Provenance

| Component | Value |
| --- | --- |
| Host | Apple M4 Pro, 48 GiB unified memory, 10 performance + 4 efficiency CPU cores, 20-core GPU |
| OS | macOS 27.0, arm64 |
| Probe execution state | Uncommitted runner bytes; reconstructed as the later `2dca889`, but no run-start Git status or diff was preserved |
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

The deterministic contract is `workloads/correctness/tool-call-smoke.json`. With temperature zero,
seed 42, reasoning disabled, a 2K context, the same verified model blob, Metal offload, and flash
attention, the model returned:

```json
{"tool": "multiply", "arguments": {"a": 17, "b": 19}}
```

The parsed object exactly matched the expected value in both the primary check and the auditor's
independent replay. However, the contract file was created after the throughput run, so it was not
precommitted for this candidate. The earlier exploratory prompt and its alternate schema are
preserved in the workflow evidence instead of being silently discarded.

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

Verdict: **CONDITIONAL — do not promote this candidate to an audited or headline baseline.**

The independent agent read the final Probe and llama-bench source end to end, reviewed every
primary artifact and custody record, hashed the complete model file, recomputed all aggregates,
replayed the correctness contract, and ran ten fresh offline repetitions for the long-prompt and
generation workloads on the same M4 Pro.

| Workload | Primary mean | Independent mean | Difference | Independent samples |
| --- | ---: | ---: | ---: | --- |
| `pp4096` | 544.52 tok/s | 609.00 tok/s | +11.84% | 591.243, 598.690, 606.874, 609.107, 613.942, 612.011, 614.592, 615.747, 612.443, 615.353 |
| `tg128` | 43.11 tok/s | 50.60 tok/s | +17.37% | 48.7915, 49.9460, 48.7575, 54.0000, 47.7879, 51.8684, 52.1753, 50.9227, 52.0771, 49.6684 |

The primary fifth-sample collapse did not recur: the replica's last five samples were 1.67% faster
than its first five for `pp4096` and 2.98% faster for `tg128`. Background load, swap/page residency,
or unmeasured DVFS remain plausible causes; neither run captured GPU clocks or temperature.

Promotion-blocking findings:

1. **High — incomplete source provenance.** The run began at 18:55:33, before commit `2dca889`
   existed at 19:10:23. Source mtime, the pre-run bytecode header, and emitted trace behavior strongly
   match that later commit, but no run-start Git status or dirty diff was preserved.
2. **High — undeclared replication disagreement.** The independent means were 11.84% and 17.37%
   faster, and no acceptance tolerance had been locked before execution.
3. **Medium — controls were not precommitted.** The correctness contract and primary environmental
   observations were recorded after the throughput run. The independent exact-output replay passed,
   so this is a custody failure rather than evidence of an incorrect response.
4. **Low — artifact immutability was policy-only.** Artifact files were internally consistent, but
   the manifest did not contain per-file digests or an external timestamp authority.

The anti-gaming review passed for cached or replayed output, skipped token work, prompt/context cache
reuse, workload substitution, model mismatch, invalid correctness output, sample cherry-picking,
aggregate parsing, misleading memory claims, and test-specific branches. The auditor observed no
repository, cache, model, power-setting, or Git writes.

Memory remains intentionally unscored. The independent clean run reported 12.62 GiB peak process
RSS, while a diagnostic run observed about 13.0 GiB resident with roughly 12.2 GiB in clean mapped
model pages and a 623 MiB physical footprint. These incompatible accounting views confirm that
neither the primary 5.21 GiB RSS nor the replica's RSS is a total Metal/unified-memory high-water
mark.

Remaining risks include uncontrolled page residency and competing processes, unknown GPU
temperature/clocks/power, no independent `pp512` replay, no cold-load measurement, a single narrow
correctness contract, and no cryptographically sealed primary artifact. A fresh run with committed
source and contract, start-of-run state capture, artifact hashes, and a predeclared replication
tolerance is required for promotion.
