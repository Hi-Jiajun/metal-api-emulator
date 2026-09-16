#include <metal_stdlib>
using namespace metal;

// Reviewed compute fixture for the combined stencil resolve increment
// (`research/docs/23` §3.3, v60). The declaring pass of the combined render
// case has to declare both landings, so the kernel reads the depth view and
// the stencil view beside the colour attachment's own read binding. The three
// stores write the same bytes in every fixture that uses this module — both
// sources carry the same guard word — so the reviewed result stays the copied
// word and the two witnesses are what make both views part of the pass's own
// declaration.
kernel void copy_word_with_witnesses(device const uint *input [[buffer(0)]],
                                     device uint *output [[buffer(1)]],
                                     device const uint *depth_witness [[buffer(2)]],
                                     device const uint *stencil_witness [[buffer(3)]]) {
    uint value = input[0];
    uint seen_depth = depth_witness[0];
    uint seen_stencil = stencil_witness[0];
    output[0] = value;
    output[0] = seen_depth;
    output[0] = seen_stencil;
}
