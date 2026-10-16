# Direct G16S execution

Probe has executed a controlled native-code modification on the M4 Pro GPU. This is stronger than
inspecting AIR or hashing a compiler output: changing one byte in the native `__TEXT,__text`
instruction stream changed the result produced by a fresh GPU process.

## Acceptance contract

The direct-ISA path is accepted only if:

1. the runtime loads the original Metal library;
2. pipeline creation uses `MTLPipelineOptionFailOnBinaryArchiveMiss`;
3. only the `applegpu_g16s` slice differs;
4. a machine-code change alters all output elements as predicted; and
5. negative controls reject the unmodified and incorrectly expected results.

Recompiling MSL/AIR, silently missing the archive, changing reflection, or merely changing a binary
hash does not satisfy the contract.

## Reproducer

`tools/kernels/metal-isa-canary.metal` computes `destination = source + 1`. Compile it and create a
binary archive using `metal-pipeline-archive`. Extracting the native object with `metal-source`
shows that compiling addends 1 and 2 changes the instruction byte at object offset 775 from `0x02`
to `0x04`. In the serialized G16 slice, the corresponding byte is offset 2631.

Patch the original `+1` native slice:

```bash
build/tools/metal-native-patch \
  build/profile/direct-isa/add1/canary.g16.metallib \
  build/profile/direct-isa/add1/canary.add3-synthesized.g16.metallib \
  2631 0x02 0x06

xcrun metal-lipo build/profile/direct-isa/add1/canary.binaryarchive \
  -replace applegpu_g16s \
  build/profile/direct-isa/add1/canary.add3-synthesized.g16.metallib \
  -output build/profile/direct-isa/add1/canary.add3-synthesized.binaryarchive

build/tools/metal-archive-replay \
  build/profile/direct-isa/add1/canary.metallib \
  build/profile/direct-isa/add1/canary.add3-synthesized.binaryarchive \
  probe_isa_canary 3
```

The synthesized `+3` program was never compiled from MSL. All 256 results matched. The replay tool
loads the original `+1` metallib and requires a binary-archive hit. Judging the original archive as
`+2`, or the patched archive as `+1`, produced 256 mismatches in each negative control.

## Established boundary

The public `MTLBinaryArchive` loader accepts a modified native device program when the archive
lookup metadata remains intact. A full assembler is therefore not required for controlled native
experiments: differential compilation can identify encodings and `metal-lipo` can repackage a
patched G16 slice.

The remaining gap is semantic coverage. Apple still ships no G16S instruction printer, and a
one-byte canary does not provide register allocation, scheduling, branch encoding, descriptor
construction, or a general assembler. Probe will use differential native patches only when:

- the exact instruction field is isolated by controlled source variants;
- the patch changes no metadata or unrelated bytes;
- a CPU oracle checks every output;
- a fresh process requires the archive hit; and
- the end-to-end benchmark uses the same correctness and repetition gates as MSL kernels.

## Translator assembly-output boundary

The installed native translator advertises `-S`, but the Apple GPU plugin does not implement that
interface. Probe exercised two independently selected paths:

- AIR 2.8, macOS 26, with `libapplegpu-nt.dylib` selected explicitly; and
- AIR 2.7, macOS 15, through the legacy wrapper/plugin selection path.

Both accept valid LLVM AIR and reach the G16S translator, then fail with:

```text
[AGX] Plugin interface not implemented: AIRNTEmitAssembly
```

This rules out a hidden command-line switch as the missing textual assembler route. `metal-as` is
an LLVM bitcode assembler rather than an Apple-GPU machine assembler, while `metal-objdump` accepts
the `agx3---macho` object but reports that it has no instruction printer. The practical direct path
on this toolchain is therefore controlled differential compilation plus native byte patching and
archive replay, not textual G16S assembly.

`metal-nt -mllvm -print-regusage -save-temps` also completes without emitting register-use or spill
statistics in the installed release toolchain. The binary metadata uses Apple's BFBS schemas, but
the extracted `__GPU_STATS_MD` section is empty for the measured pipelines. Kernel reports must
therefore say that register pressure and spills are unavailable; binary size or latency is not a
valid substitute for either counter.

## Shader timeline and counter boundary

Probe also enabled the normally hidden Shader Timeline instrument in a copied Metal System Trace
template. On this M4 Pro it resolves shader names, command intervals, and GPU program-counter
ranges, so it is useful for mapping a runtime dispatch to its native G16S object. It produced zero
instruction samples, however, and the trace recorded the device-service error:

```text
GPU Service reported error: Selected counter profile is not supported on target device
```

This is a hardware/driver capability result, not a permission or template-selection failure. A
user-space program cannot make an unsupported counter block appear: doing so would require Apple
to expose a supported G16S counter profile in the GPU service/driver, or unsupported privileged
driver and firmware work. Probe will not weaken system integrity to attempt the latter.

The practical profiler therefore combines three independent views:

- llama.cpp graph/fusion traces for exact node-to-dispatch attribution;
- Metal command-buffer and per-kernel GPU timestamps for measured latency; and
- Shader Timeline program-counter ranges plus extracted G16S objects for shader/native-code
  identity.

That stack reaches the native program and measures its effect, but it cannot truthfully report
per-instruction occupancy, stalls, register pressure, cache misses, or dispatch-boundary counters
on this machine. Direct byte patches remain available for isolated encoding experiments; they do
not manufacture missing performance counters.
