// Owned synthetic fixture (C2): the Metal source shape the rail's AIR fixture
// (`crates/metal-api-vulkan/tests/fixtures/write_texture_2d_bias_2000.ll`)
// mirrors. Not derived from a third-party metallib.
//
// The kernel body is the sibling of `write_texture_2d_bias_1000.metal` with
// only the bias moved, so the landed bytes have to follow the module that ran.
#include <metal_stdlib>
using namespace metal;

kernel void write_texture_2d(texture2d<float, access::write> tex [[texture(0)]],
                             uint3 gid [[thread_position_in_grid]])
{
    const uint cell = (gid.y * 4u) + gid.x;
    tex.write(float4(float(cell) + 2000.0f, 0.0f, 0.0f, 0.0f), uint2(gid.x, gid.y));
}
