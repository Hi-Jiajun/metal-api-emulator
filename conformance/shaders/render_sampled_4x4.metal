#include <metal_stdlib>
using namespace metal;

// Reviewed render-sampler fixture (`research/docs/23` §3.3, v70): one MSL
// module whose two entries are compiled into one `MTLRenderPipelineState`, the
// native rail's sibling of the Vulkan pair
// `render_spv/sampled_quad.vert.spvasm` + `render_spv/solid_unorm8_sampled.frag.spvasm`.
// The module is the whole review surface for the native rail, so the provider
// rail (`crates/metal-api-native/src/render.rs`, `REVIEWED_SAMPLED_SOURCE`) and
// the Swift oracle (`conformance/NativeOracle.swift`) pin these exact bytes; an
// edit here has to update both pins and the fixture's expected texel bytes.

// Vertex stage: the position comes from `vertex_id` alone, so the pipeline
// binds no vertex buffer (`VertexLayout::None`). The geometry is the milestone
// triangle — (-1,-1), (3,-1), (-1,3) — whose oversize shape covers every pixel
// centre of the 4x4 viewport, and the varying is the same geometry's own
// normalised coordinate `((x + 1) / 2, (1 - y) / 2)`. Across that covering
// raster the interpolated varying lands on `(column + 0.5) / width` and
// `(row + 0.5) / height` per fragment: the texel centres of an
// attachment-sized texture, which is what makes the sample an identity copy
// rather than a filtered or boundary-rule-dependent read.
struct RenderSampledVertexOut {
    float4 position [[position]];
    float2 uv [[user(sampled_uv)]];
};

vertex RenderSampledVertexOut render_sampled_quad_vertex(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    const float2 position = positions[vertex_id];
    RenderSampledVertexOut out;
    out.position = float4(position, 0.0, 1.0);
    out.uv = float2((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    return out;
}

// Fragment stage: the sample of the pass's own texture at binding 0, read
// through a `constexpr` nearest/clamp sampler — the provider-synthesised
// sampler of the Vulkan rail, spelled in the shader instead of on the encoder,
// because a fragment standing on a texel centre is exactly the read nearest
// filtering makes an identity copy. Every falsification the fixture states is
// visible here: a pass that binds no texture samples outside the review, a
// linear filter mixes neighbours, and a flipped or transposed uv reads another
// row or column.
fragment float4 render_sampled_texel(RenderSampledVertexOut in [[stage_in]],
                                     texture2d<float> tex [[texture(0)]]) {
    constexpr sampler nearest_sampler(filter::nearest, address::clamp_to_edge);
    return tex.sample(nearest_sampler, in.uv);
}
