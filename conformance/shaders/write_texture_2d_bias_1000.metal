// Owned synthetic fixture (C2): the Metal source shape the rail's AIR fixture
// (`crates/metal-api-vulkan/tests/fixtures/write_texture_2d_bias_1000.ll`)
// mirrors. Not derived from a third-party metallib.
//
// One invocation writes `float(4 * y + x) + 1000` into the texel at its own
// grid coordinate of a 4x4 `R32Float` storage image, so the landed bytes are
// the little-endian floats `[1000.0 .. 1015.0]`. Its sibling
// `write_texture_2d_bias_2000.metal` moves only that constant.
#include <metal_stdlib>
using namespace metal;

kernel void write_texture_2d(texture2d<float, access::write> tex [[texture(0)]],
                             uint3 gid [[thread_position_in_grid]])
{
    const uint cell = (gid.y * 4u) + gid.x;
    tex.write(float4(float(cell) + 1000.0f, 0.0f, 0.0f, 0.0f), uint2(gid.x, gid.y));
}
