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
