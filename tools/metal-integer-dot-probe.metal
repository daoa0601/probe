// Negative control: this kernel is expected NOT to compile. The installed Metal toolchain has no
// integer dot product for char4 (only half/float overloads), which is why the Q4xQ8 experiments in
// docs/metal-hill-climb.md fell back to explicit i32 reductions. Reproduce with:
//   xcrun -sdk macosx metal -c tools/metal-integer-dot-probe.metal -o /dev/null
// and expect "no matching function for call to 'dot'".
#include <metal_stdlib>

using namespace metal;

kernel void integer_dot_probe(
        device const char4 * lhs [[buffer(0)]],
        device const char4 * rhs [[buffer(1)]],
        device       int   * out [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    out[index] = dot(lhs[index], rhs[index]);
}
