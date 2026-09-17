// Owned synthetic fixture (C1b): the Metal source shape the rail's AIR fixture
// (`crates/metal-api-vulkan/tests/fixtures/sample_texture_2d_nearest_repeat.ll`)
// mirrors. Not derived from a third-party metallib.
//
// Byte-for-byte the same body as `sample_texture_2d_nearest_clamp.metal` except
// for the constexpr sampler state: repeat addressing wraps the out-of-range
// coordinate 1.375 back to the centre of texel 1, so the two readings against a
// row of [0, 4, 8, 12] become [4, 4].
#include <metal_stdlib>
using namespace metal;

kernel void sample_texture_2d(texture2d<float, access::sample> tex [[texture(0)]],
                              device float2 &out [[buffer(0)]])
{
    constexpr sampler state(filter::nearest, address::repeat);
    out[0] = tex.sample(state, float2(1.375, 0.125)).x;
    out[1] = tex.sample(state, float2(0.3125, 0.125)).x;
}
