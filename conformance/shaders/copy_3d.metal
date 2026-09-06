#include <metal_stdlib>
using namespace metal;

// Owned two-buffer stage for a producer/consumer command sequence.
kernel void copy_3d(device const uint *input [[buffer(4)]],
                    device uint *output [[buffer(9)]],
                    uint3 gid [[thread_position_in_grid]]) {
    uint index = (gid.z * 3 + gid.y) * 5 + gid.x;
    output[index] = input[index];
}
