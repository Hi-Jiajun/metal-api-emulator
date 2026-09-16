#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to kernel_copy_16.ll. One 16-word
// linear copy: invocation `x` copies word `x`, so a [16, 1, 1] grid moves
// exactly 64 bytes between the read binding 4 and the write binding 9.
kernel void copy_16(device const uint *input [[buffer(4)]],
                    device uint *output [[buffer(9)]],
                    uint3 gid [[thread_position_in_grid]]) {
    output[gid.x] = input[gid.x];
}
