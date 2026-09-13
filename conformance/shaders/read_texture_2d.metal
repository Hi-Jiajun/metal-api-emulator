#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to kernel_read_texture_2d.ll. The AIR
// fixture is the reviewed input; this file exists for the capture tool's
// source-pair validation and for the native oracle's future texture case.
kernel void read_texture_2d(texture2d<uint, access::read> texture [[texture(0)]],
                            device uint *output [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
    const uint2 coord = uint2(gid.x, gid.y);
    const uint cell = (gid.y << 2) + gid.x;
    output[cell] = texture.read(coord).x;
}
