#include <metal_stdlib>
using namespace metal;

// Owned bounded fixture: each invocation updates one word and a separate output.
kernel void mix_3d(device uint *values [[buffer(0)]],
                         device const uint *bias [[buffer(2)]],
                         device uint *output [[buffer(5)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint index = (gid.z * 3 + gid.y) * 5 + gid.x;
    uint value = (values[index] ^ bias[0]) * 3u + 1u;
    values[index] = value;
    output[index] = value + 0x10203040u;
}
