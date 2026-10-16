# Gemma 4 Metal hill climb on M4 Pro

This record separates code that changes executed GPU work from diagnostic infrastructure. A
candidate graduates only after the relevant kernel output matches a reference, its specialized
G16S object is captured, and repeated end-to-end generation does not regress. Percentages below
are measured changes, not additive projections.

## Retained runtime changes

| Candidate | Executed work changed | Isolated result | End-to-end evidence |
| --- | --- | ---: | ---: |
| Q4_0 matrix-vector NSG4 | two to four SIMD groups on M4 Pro | shape portfolio retained NSG4 | 26B +2.20%; 12B +1.14% |
| 26B MoE weighted sum | three dispatches and two materialized scaling tensors to one dispatch | 0.01410 to 0.00350 ms, 4.03× | two balanced pairs: +3.26%, +3.05% |
| 12B Q4_0 gate/up/GEGLU | two Q4 projections plus GEGLU to one kernel | exact output; `nr0=2,nsg=8` won the native portfolio | pooled decode +1.045% |
| 12B Q4_0 down tail | projection, RMS norm, norm scale, residual add, and output scale to one kernel | five nodes to one; `nr0=2,nsg=8`, one accumulator | included in the +2.84% combined stack |
| RMS/scale/RoPE | three graph nodes to one kernel | 29–30% lower kernel latency | full-model median +0.642% |
| direct K/V cache writes | RMS/scale/RoPE plus scatter to one dispatch for each cache | local K −69.9%, V −27.5%; global K −51.2%, V −46.2% | four-pair mean +1.45%, median +0.75% |
| short-cache FlashAttention | 32 workgroups, temporary output, and reduction to one direct-output workgroup | at 256 active tokens: local −11.7%, global −1.6% | four-pair mean +2.80%, median +2.72% |
| MTP Q4_0 down tail | assistant `[8192,1024]` projection tail from three dispatches to one | 0.022019 to 0.017815 ms, −19.09%, exact | retained local win; only about 0.02% predicted end to end |

The dense candidates are exact-shape specializations for Gemma 4 12B decode and are guarded by
device, tensor type, dimensions, contiguity, dependency, and use count. Each family can be disabled
independently:

```text
GGML_METAL_Q4_GEGLU_DISABLE
GGML_METAL_Q4_DOWN_NORM_DISABLE
GGML_METAL_Q4_MTP_DOWN_NORM_DISABLE
GGML_METAL_RMS_ROPE_DISABLE
GGML_METAL_KV_CACHE_FUSION_DISABLE
```

`GGML_METAL_FATTN_VEC_NWG=32` restores the upstream FlashAttention workgroup count; setting it to
1 forces the experimental direct-output path. The automatic M4 Pro selector uses one workgroup
only for the two measured Gemma head geometries and cache lengths no greater than 256.
`GGML_METAL_Q4_NSG=2` restores the upstream Q4_0 SIMD-group count for controlled comparisons;
the retained M4 Pro default remains four.

## Combined retained stack

The final 12B comparison disabled every dense fusion, restored 32-workgroup FlashAttention, and
restored Q4_0 NSG2. Four balanced pairs, alternating which side ran first, measured:

| Pair | Enabled TPS | Disabled TPS | Change |
| ---: | ---: | ---: | ---: |
| 1 | 29.264 | 28.814 | +1.56% |
| 2 | 29.351 | 28.282 | +3.78% |
| 3 | 28.822 | 28.277 | +1.92% |
| 4 | 29.094 | 27.942 | +4.12% |
| Pooled, 12 samples per side | 29.133 | 28.329 | +2.84% |

Every run started on AC power with no thermal or performance warning and load average at or below
8.0. The disabled side was noisier under unrelated VM/storage activity, but all four paired
comparisons were positive. A separate two-chunk logit comparison reported mean KLD 0.000000, RMS
probability delta 0.000%, and the same top token for 100% of positions.

## Current 12B throughput frontier

The largest current gains come from changing the amount of work around the retained kernels, not
from pretending every microsecond-scale fusion is an end-to-end win.

First, `gguf-untie-output` preserves the original Q8_0 `token_embd.weight` and adds a separately
quantized `output.weight`. Against the same saved Q8 logits, and in a balanced Q4/Q6/Q6/Q4
ten-sample decode run:

| Untied output head | Mean TPS | Versus Q6_K | Mean KLD | RMS Δp | Same top token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q6_K | 30.414 | — | 0.000247 | 0.212% | 98.413% |
| Q4_0 | 31.437 | +3.363% | 0.003577 | 1.098% | 92.857% |

This is an explicit speed/quality choice. Q4_0 is the throughput target used for the speculative
frontier; the Q8_0 input embedding remains byte-identical in both untied files.

Second, the dedicated four-layer Q4_0 Gemma 4 MTP assistant was swept over draft length and
acceptance threshold. The promoted setting is two draft tokens with `p_min=0.995`:

| Configuration | Mean decode TPS | Acceptance | Evidence |
| --- | ---: | ---: | --- |
| target only | 31.823 | — | six runs in the final ABBA comparison |
| MTP, `n=1`, `p_min=0.995` | 33.997 | 96.64% | three-run bracket; pruned |
| MTP, `n=2`, `p_min=0.9975` | 35.445 | 95.90% | three-run bracket; pruned |
| MTP, `n=2`, `p_min=0.995` | 35.662 | 95.98% | six-run ABBA winner |

The final three-task comparison measured +12.065% in the mean and +9.926% in the median. Code,
prose, and reasoning tasks individually improved by 13.41%, 11.61%, and 11.14%; all final choices
were byte-identical to target-only decoding. The brackets are search evidence rather than paired
headline comparisons.

The assistant graph trace establishes that its real down-projection matrix is `[8192,1024]`.
The retained r4/nsg8 fusion reduced the exact three-dispatch microbenchmark from 0.022019 to
0.017815 ms. Earlier synthetic `[256,1024]` measurements used the wrong matrix width and are
explicitly superseded; they are not part of the retained claim.

Each ABBA phase used a fresh server and the same deterministic task order. The server log still
advertised prompt caching despite the attempted no-cache flag, so this result is specifically
decode throughput with matched phases—not a claim about cache-free request latency. Later top-k
experiments used `--cache-ram 0` to remove that ambiguity.

## FlashAttention crossover

The single-workgroup kernel is not a universal replacement. With one decoded token and every cache
entry active, longer-cache measurements were:

| Cache tokens | Local head, 256×8 | Global head, 512×1 |
| ---: | ---: | ---: |
| 128 | −10.43% | −4.38% |
| 192 | −9.92% | −0.53% |
| 224 | −11.28% | −2.69% |
| 256 | −11.71% | −1.61% |
| 288 | +6.87% | +23.80% |
| 512 | +8.71% | +34.47% |
| 1024 | +16.78% | +30.66% |
| 2048 | +43.56% | +60.95% |

Negative is faster. At 288 tokens the one-workgroup path crosses its 256-token coverage and needs
a second serial cache iteration, producing the observed cliff. Two-token batches can still win at
512 and 1024, but steady-state autoregressive decode is one token, so the runtime guard follows the
one-token result.

## Pruned candidates

- Fusing final logit softcap was more than 14× faster in isolation but reduced full-model
  throughput by 0.27%; it is not in the runtime patch.
- A fused QKV projection was flat or negative end to end.
- General Q4_0 two- and four-accumulator kernels lost across the measured shapes.
- The PPC-style exact full-tile specialization for the retained Q4 down-tail kernel removed its
  generic bounds path and shrank the specialized G16 `__text` section from `0xa26` to `0x8d2`
  bytes (−13.1%). A 41-sample direct kernel A/B nevertheless had identical medians
  (`0.144359376 ms`) and a slightly worse exact-specialization mean, so smaller native code was
  not treated as a performance proxy. Packed Q4 dot variants (`scalar`, `float4`, and `ushort4`)
  likewise produced distinct G16 programs without a stable latency winner. An SOA8 weight layout
  lost in a 19-sample pooled full-model replicate (28.336 versus 28.381 TPS).
- Fusing the Q4 attention-output projection with the post-attention RMS/scale/residual tail was
  bit-exact, but all 12 measured local/global geometries were slower. Local attention regressed
  16–140%; the best global geometry still regressed 5.16%.
- Extending that idea to Fable's full decode sequence—Q4 attention output, post-attention
  RMS/scale/residual, and pre-FFN RMS/scale—did select as one six-node dispatch throughout the
  12B graph, but did not survive isolated replication. The first per-workgroup-partial version
  lost all 12 width/geometry cases; retaining each thread's hidden values in a local array also
  lost and grew the G16 program. A raw-output scan then showed an apparent 4.56% win at
  `[8192,3840]`, but the identical r2/nsg8 native object
  (`40e2e3c5ec24515c...`) regressed 5.77% over 21 alternating samples. Its r2/nsg4–7 and r3/nsg8
  neighbors regressed 6.31–29.42%. Output was exact or within 7.2e-7 depending on reduction order,
  so correctness was not the blocker; last-arriver serialization was. The runtime implementation
  was removed.
- Changing only the 262,144-row Q4 output projection's SIMD-group count did not produce a robust
  winner. NSG7's initial 1.30% kernel lead reversed to a 1.94% regression in a longer replicate;
  NSG1's longer result was only 0.26%, below run variance.
- Q4 output row-reuse variants spanning one, two, eight, and sixteen rows per SIMD group also
  failed. The only initial lead, r2/nsg4 at 0.36%, became a 0.02% regression in a 21-sample
  replicate; r1 regressed 31%, r8 by 0.6–1.5%, and r16 by about 7%.
- The down-tail three-accumulator variant was slower. Two accumulators had mixed full-model
  evidence, so the one-accumulator kernel remains the default.
- Doubling the vector FlashAttention cache tile from 32 to 64 elements was rejected at the
  correctness gate. The current kernel maps one cache value to each SIMD lane; changing only the
  compile-time tile made the unmodified lane mapping omit half the values and produced
  `5.0e-3`–`1.5e-2` maximum output errors. Its timing results are invalid and the unsafe
  entrypoints were removed. Covering 64 values correctly requires two SIMD groups, which is
  already represented by the measured `NSG=2` portfolio rather than a free tile-size win.
- A fresh middle-cache FlashAttention portfolio found real isolated NWG16 wins at llama.cpp's
  512-, 768-, and 1024-token KV extents. The two Gemma head geometries were exact or within
  `2.8e-9`; the largest kernel win was 21.0% at 1024 tokens. A paired 16-lane reducer then used
  both SIMD halves, cut its launch from 512 to 256 threads, and improved the local-attention
  combined kernel by another 0.1–0.9%. It still failed the default-runtime gate: two balanced
  long-context pairs pooled 121.649 TPS enabled versus 123.627 TPS with only the middle selector
  disabled (−1.60% mean, −1.85% median). The runtime selector and paired reducer were therefore
  removed; their distinct G16 objects and microbenchmarks remain negative-control evidence.
- Increasing the local-head single-workgroup FlashAttention specialization to eight SIMD groups
  removed the reduction dispatch through 1024 cached tokens and was exact within `4.7e-9`.
  Hot-buffer microbenchmarks improved by 10–16%, but separate rotating cache buffers reduced that
  result to roughly flat, while the same idea for the global head regressed 42–84%. The local-only
  runtime trial also failed: an isolated candidate/control/control/candidate pass pooled
  122.153 versus 123.857 TPS (−1.38% mean, −1.79% median). Its specialized G16 text section was
  `0x2656` bytes versus `0x259a` for the retained NSG1/NWG32 path. The selector and NSG8 cap were
  removed.
- Direct scheduler-option sweeps produced identical G16 instruction streams for all 16 tested
  variants and therefore offered no hill-climb axis on this release toolchain.
- Fable-inspired Q4×Q8 matrix-vector kernels moved activation quantization ahead of the output
  projection and tested floating and packed integer reductions. The installed Metal toolchain
  exposes no usable int8 SIMD dot primitive (`tools/metal-integer-dot-probe.metal` reproduces the
  compile failure), and the explicit i32 path did not compensate: the
  best output-head r4/nsg4 result changed from a 0.31% lead in 11 samples to a 0.20% regression in
  a 41-sample replicate. The `[3840,8192]` gate/up and `[8192,3840]` down shapes regressed 2.56%
  and 2.34%. The experimental runtime entrypoints were removed; distinct G16S objects remain as
  negative evidence.
- The MTP graph's 262,144-logit `TOP_K(10)` was attacked with two custom kernels. A deterministic
  one-workgroup full-vocabulary scan produced one dispatch and a `0xd5a`-byte G16S text section,
  but an enabled/control/control/enabled comparison regressed 0.521%. A stronger 64-way
  4096-logit block selector plus a 640-candidate merge used two dispatches and distinct `0xeaa`
  and `0xdbe` native programs, but regressed 0.635%. All three tasks were negative in both tests;
  final choices and draft counts matched exactly. Both runtime implementations were removed.

The source tree keeps some pruned microbenchmarks because they are useful negative controls. Their
presence is not a runtime promotion.

## Optional output-matrix quantization

Gemma 4 stores one `token_embd.weight` tensor and uses it for both input embeddings and output
logits when `output.weight` is absent. Requantizing that tied 12B tensor produces a real
speed/quality frontier:

| Tied tensor | Size | Mean KL divergence | RMS probability delta | Same top token |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 reference | 1020 MiB | 0 | 0 | 100% |
| Q6_K | 787.5 MiB | 0.002590 | 1.531% | 99.206% |
| Q5_0 | 660 MiB | 0.017335 | 2.975% | 95.238% |
| Q4_0 | 540 MiB | 0.028393 | 3.848% | 90.476% |

The quality sample is two fixed 128-token chunks of the public-domain *A Tale of Two Cities*,
compared against saved Q8 logits with llama.cpp's KL-divergence mode. A balanced forward/reverse
throughput pass used ten 128-token decode samples per variant:

| Tied tensor | Pooled TPS | TPS versus Q8 |
| --- | ---: | ---: |
| Q8_0 | 29.313 | — |
| Q6_K | 30.071 | +2.58% |
| Q5_0 | 30.201 | +3.03% |
| Q4_0 | 31.036 | +5.88% |

The paired halves agreed closely: Q4 measured 31.000 then 31.072 TPS, while Q5 measured 30.214 then
30.188 TPS. Q4 is not a default because its quality movement is material even though its speed gain
is largest.

Gemma 4's llama.cpp loader already treats `output.weight` as optional and prefers it when present.
Probe's `gguf-untie-output` transformer now preserves the original Q8_0 token embedding byte for
byte and adds a separately quantized output tensor. This avoids paying the input-embedding quality
cost of a tied quantization. Q6_K and Q4_0 independent heads are measured in the current-frontier
section above; their throughput and KL-divergence checks remain separate from the tied results.

## Native-code evidence

Each retained kernel has a specialized `applegpu_g16s` slice and extracted native object under the
ignored `build/profile/native/` tree. `metal-source` proves that the archive contains a device
object; `metal-objdump` accepts that object but the installed Apple GPU backend has no instruction
printer. The native translator can emit and replay G16S code, but its advertised assembly-output
interface fails with `AIRNTEmitAssembly`, and the release compiler emits neither register-use nor
spill diagnostics. Probe records that absence rather than inventing occupancy counters.

An explicitly enabled Shader Timeline template adds shader names and GPU program-counter ranges,
but instruction sampling is still unavailable. The trace contains zero samples and the GPU service
reports `Selected counter profile is not supported on target device`. The profile/native mapping
therefore comes from graph traces, command/per-kernel timestamps, Shader Timeline PC intervals, and
extracted G16S objects; it does not claim unsupported per-instruction stall or occupancy counters.

The practical lowest-level loop is consequently MSL/AIR specialization, G16 binary extraction,
section/hash/size comparison, controlled differential byte patching where an encoding is isolated,
archive-hit replay, and empirical GPU timing. See [Direct G16S execution](direct-isa.md).
