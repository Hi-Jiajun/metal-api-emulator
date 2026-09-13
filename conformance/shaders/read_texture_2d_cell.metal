#include <metal_stdlib>
using namespace metal;

// Owned synthetic source counterpart to kernel_read_texture_2d_cell.ll, for the
// v12 texture-cell cases. The AIR fixture is the reviewed input; this file keeps
// the same per-invocation semantics for the capture tool's source-pair
// validation and for the native oracle's texture cases.
kernel void read_texture_2d_cell(texture2d<uint, access::read> texture [[texture(0)]],
                                 device uint *output [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    const uint2 coord = uint2(gid.x, gid.y);
    // Keep the arithmetic identical to the AIR fixture's `4*y + x` and its
    // `+ 100` bias so both rails land the same cell for every invocation.
    const uint cell = (gid.y * 4) + gid.x;
    output[cell] = texture.read(coord).x + 100;
}
