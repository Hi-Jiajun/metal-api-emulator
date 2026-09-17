// Owned synthetic fixture (C2): the Metal source shape the rail's AIR fixture
// (`crates/metal-api-vulkan/tests/fixtures/read_write_texture_2d_increment.ll`)
// mirrors. Not derived from a third-party metallib.
//
// One read-modify-write per texel: the landed bytes depend on the initial
// contents the upload path staged (a 4x4 view of `[0.0 .. 15.0]` lands
// `[1.0 .. 16.0]`) and on the binding being executed as a read-write storage
// image rather than a write-only one.
#include <metal_stdlib>
using namespace metal;

kernel void read_write_texture_2d(texture2d<float, access::read_write> tex [[texture(0)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    const uint2 coord = uint2(gid.x, gid.y);
    tex.write(float4(tex.read(coord).x + 1.0f, 0.0f, 0.0f, 0.0f), coord);
}
