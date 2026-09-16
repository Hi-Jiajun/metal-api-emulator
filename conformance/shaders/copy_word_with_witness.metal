#include <metal_stdlib>
using namespace metal;

// Reviewed compute fixture for the depth readback increment (`research/docs/23`
// §3.3, v43). The declaring pass of the depth-store render case has to declare
// every view the render pass resolves, and the depth attachment is no
// exception: `witness` is the depth view, bound read-only exactly like the
// colour attachment's own read binding. The two stores write the same bytes in
// every fixture that uses this module — both sources carry the same guard word
// — so the reviewed result stays the copied word and the third binding is what
// makes the depth view part of the pass's own declaration.
kernel void copy_word_with_witness(device const uint *input [[buffer(0)]],
                                   device uint *output [[buffer(1)]],
                                   device const uint *witness [[buffer(2)]]) {
    uint value = input[0];
    uint seen = witness[0];
    output[0] = value;
    output[0] = seen;
}
