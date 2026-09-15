#include <metal_stdlib>
using namespace metal;

// Owned fixture with different slot numbers, ordering, and input access roles.
kernel void remap_3d(device const uint *bias [[buffer(1)]],
                     device const uint *input [[buffer(3)]],
                     device uint *output [[buffer(7)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint index = (gid.z * 3 + gid.y) * 5 + gid.x;
    output[index] = (input[index] * 7u + bias[0]) ^ 0x3c3ca5a5u;
}
