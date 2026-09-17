// Owned synthetic fixture (C1b): the Metal source shape the rail's AIR fixture
// (`crates/metal-api-vulkan/tests/fixtures/sample_texture_2d_nearest_clamp.ll`)
// mirrors. Not derived from a third-party metallib.
//
// One invocation samples a 4x4 `R32Float` texture at (1.375, 0.125) and
// (0.3125, 0.125) and stores both component-zero values into one `float2`
// output buffer. Against a row of [0, 4, 8, 12] the two readings are [12, 4]
// under nearest filtering with clamp-to-edge addressing; the sibling fixtures
// move only the constexpr sampler state.
#include <metal_stdlib>
using namespace metal;

kernel void sample_texture_2d(texture2d<float, access::sample> tex [[texture(0)]],
                              device float2 &out [[buffer(0)]])
{
    constexpr sampler state(filter::nearest, address::clamp_to_edge);
    out[0] = tex.sample(state, float2(1.375, 0.125)).x;
    out[1] = tex.sample(state, float2(0.3125, 0.125)).x;
}
