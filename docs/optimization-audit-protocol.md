# Optimization and audit protocol

Probe adopts an evidence and audit loop inspired by the KernelBench testing discipline. The target
is different—whole-model inference on Apple Silicon rather than a single CUDA kernel—but the core
rule transfers directly: an optimizer must not be the sole judge of its own result.

## Unit of work

One optimization cell is one autonomous session on the physical target machine, with one model,
runtime, and device configuration. The session has no fixed iteration quota or artificial wall-clock
cutoff. It ends only when the candidate is ready for audit, is empirically rejected, or has a named
external blocker.

The current scope is one M4 Pro GPU through Metal. Multi-host, distributed, and cross-device results
must use separate protocols and cannot be mixed into this baseline.

## Candidate loop

1. **Lock the contract.** Fix model bytes, runtime commit, workload, correctness checks, metrics,
   repetitions, replication tolerance, and start-environment gates before tuning.
2. **Capture the starting state.** Record the Git commit and dirty diff, host, power/thermal state,
   free memory and disk, active competing workloads, and prior baseline.
3. **Run on real hardware.** Compile or configure, check correctness, warm up, benchmark, inspect raw
   samples, and iterate. Simulated or estimated speedups are never headline results.
4. **Keep the whole trail.** Preserve failed runs, exact commands, stdout/stderr, per-repetition
   samples, process memory, model/runtime identity, and a chronological execution trace.
5. **Run application correctness.** A synthetic token-throughput result must be paired with at least
   one deterministic sampled-generation check. Later milestones add fixed agentic quality suites.
6. **Primary audit.** The optimizing session reads the final source/config and evidence end to end,
   explicitly checking the threat list below. It cannot silently discard inconvenient samples.
7. **Independent audit.** A separate read-only agent receives the final implementation, complete
   trace, raw artifacts, and audit contract—not merely the optimizer's summary. It empirically
   replicates at least one workload offline and re-tests every suspicious condition.
8. **Publish findings separately.** Every promoted report contains an `Audit findings` section with
   the independent verdict, re-test results, discrepancies, remaining risks, and rejected claims.

## Evidence required for promotion

| Evidence | Requirement |
| --- | --- |
| Source state | Git commit plus complete dirty diff, if any |
| Host | Chip, CPU/GPU cores, memory, OS, Metal capability; no private device identifiers |
| Runtime | Exact llama.cpp commit and reported build commit |
| Model | Repository revision, filename, byte size, and SHA-256 of the executed file |
| Workload | Prompt/generation lengths, context, concurrency, sampler, cache, batch, and offload flags |
| Measurements | Warmup policy, all raw samples, mean, variance, wall time, peak RSS, failures |
| Correctness | Deterministic sampled output plus milestone-specific quality gates |
| Trace | Artifact `trace.jsonl`, workflow ledger, exact worker prompt, and raw audit output |
| Audit | Independent empirical replication and explicit promotion verdict |
| Integrity | Detached SHA-256 seal covering every finalized primary artifact file |

Artifacts are immutable. Corrections create a new artifact and link back to the invalidated one.
The primary tracked report records the detached seal digest.

## Reward-hacking and validity threat model

The primary and independent audits must mark each item pass, fail, or unknown.

### Caching and skipped work

- Output, logits, tokens, or timing results are not replayed from a prior run.
- Prompt-cache reuse, context checkpoints, mmap page-cache state, and model residency are declared.
- Cold-start/download time is kept separate from steady-state inference metrics.
- The measured token count is non-zero and matches the declared workload.
- Warmups are excluded consistently; measured repetitions are not silently treated as warmups.

### Workload and model substitution

- The executed model digest matches the declared candidate; a smaller model or different quant is
  not substituted.
- Prompt length, generation length, context, concurrency, cache precision, and backend match the
  locked contract.
- Prompt-processing and generation throughput are never ranked against each other.
- Text-only, multimodal, speculative, and non-speculative paths are reported separately.

### Measurement integrity

- Every repetition is retained; no cherry-picking or best-of-N headline metric.
- Parsed aggregates reproduce from raw samples within rounding tolerance.
- Peak RSS and unified-memory pressure are not described as interchangeable.
- Download/model-load wall time is not presented as request latency.
- Thermal drift, swap, power mode, and competing processes are measured or named as gaps.
- A speedup is compared against the same runtime/model/workload contract.

### Correctness and implementation games

- The model produces valid sampled output; throughput alone is insufficient.
- The runner does not special-case known prompts, test labels, paths, or grader behavior.
- Errors, empty outputs, truncated generations, invalid tokens, and OOMs cannot score as successes.
- Quantized cache or weights must pass the declared quality guardrail before promotion.
- Speculative decoding reports draft acceptance and total memory, not only accepted target tokens.

## Result states

- **Exploratory:** valid evidence, not yet a fixed baseline or candidate.
- **Candidate:** primary checks pass and evidence is complete; independent audit pending.
- **Audited baseline:** independent PASS, empirical replication agrees within the declared tolerance,
  and remaining risks do not invalidate intended use.
- **Conditional:** useful result with named evidence or validity gaps; not a headline baseline.
- **Rejected:** correctness, provenance, measurement, or audit failure invalidates the claim.

Only `Audited baseline` results may anchor speedup claims. Audit disagreement is preserved rather
than averaged away.

Replication agreement is evaluated separately for every audited workload as
`abs(independent_mean - primary_mean) / primary_mean * 100`. Every required workload must be at or
below the tolerance declared in the candidate config before the primary run. The primary and audit
means are never averaged to manufacture agreement, and the tolerance cannot be widened afterward.

## Promotion preflight

The promotion-eligible 26B config is `configs/gemma4-26b-a4b-baseline-v2.toml`. Before artifact
creation, Probe hashes the exact local model, checks its committed size and digest, captures Git
HEAD/status/diff, rejects a dirty or unavailable repository, captures the host environment, rejects
start load above the configured ceiling, and requires AC power when declared. This preflight is a
guardrail, not a quality verdict; correctness execution and independent audit are still mandatory.

## Durable custody

Each benchmark artifact appends `trace.jsonl` events during execution and fsyncs every event. Each
multi-agent audit lives under `.codex-subagents/<workflow-id>/` with the invariant, exact prompts,
raw outputs, accepted/rejected copies, coverage, state, ledger, and synthesis inputs. Tracked reports
under `docs/baselines/` contain only validated primary evidence and accepted audit findings.
