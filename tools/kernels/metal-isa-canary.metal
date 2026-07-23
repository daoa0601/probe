#include <metal_stdlib>

using namespace metal;

#ifndef PROBE_ISA_ADDEND
#define PROBE_ISA_ADDEND 1
#endif

kernel void probe_isa_canary(
        device const uint * source [[buffer(0)]],
        device uint * destination [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    destination[index] = source[index] + PROBE_ISA_ADDEND;
}
