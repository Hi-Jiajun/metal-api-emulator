#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the vertex-input milestone (`research/docs/23`
// §3.3): one MSL module whose two stage entries are compiled into one
// `MTLRenderPipelineState`, and whose vertex stage reads a caller-held vertex
// stream declared by an `MTLVertexDescriptor` instead of generating positions
// from `vertex_id`. The module is the whole review surface for the native rail,
// so the provider rail (`crates/metal-api-native/src/render.rs`) and the Swift
// oracle (`conformance/NativeOracle.swift`) pin these exact bytes; an edit here
// has to update both pins, the `quad_indexed.vert.spvasm` Vulkan counterpart and
// the fixture's expected texel bytes.

// One vertex of the reviewed stream: a two-component position at attribute 0,
// which the descriptor binds to buffer 0 at offset 0 with stride 8. The values
// are already in NDC, so the stage only widens them to a clip-space position.
struct QuadVertex {
    float2 position [[attribute(0)]];
};

struct QuadVertexOut {
    float4 position [[position]];
};

vertex QuadVertexOut render_quad_vertex(QuadVertex in [[stage_in]]) {
    QuadVertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    return out;
}

// Fragment stage: (64/255, 128/255, 192/255, 1), which an 8-bit UNORM
// attachment stores as `40 80 c0 ff`. Same byte/255 constants and the same
// reasoning as `render_offscreen_2x2.metal`: they sit far from a half-integer
// tie, which different drivers resolve differently (`research/docs/23` §3.5).
fragment float4 render_solid_rgba8() {
    return float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
}
