#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the first offscreen milestone (`research/docs/23`
// §1.2, §3.2): one MSL module whose two stage entries are compiled into one
// `MTLRenderPipelineState`. The module is the whole review surface for the
// native rail, so both the provider rail
// (`crates/metal-api-native/src/render.rs`, `REVIEWED_SOURCE`) and the Swift
// oracle (`conformance/NativeOracle.swift`, `reviewedRenderModule()`) pin these
// exact bytes; an edit here has to update both pins and the fixture's expected
// texel bytes.

// Vertex stage: the position comes from `vertex_id` alone, so the pipeline
// binds no vertex buffer and needs no `MTLVertexDescriptor`
// (`VertexLayout::None`). The three vertices are (-1,-1), (3,-1), (-1,3): the
// oversize triangle covers every pixel centre of a 2x2 viewport, which is what
// makes "the draw really ran" falsifiable (`research/docs/23` §1.3). The naive
// 2x pair (x = 2*vertex_id - 1) leaves one pixel centre uncovered.
struct RenderVertexOut {
    float4 position [[position]];
};

vertex RenderVertexOut render_fullscreen_triangle(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    RenderVertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    return out;
}

// Fragment stage: (64/255, 128/255, 192/255, 1), which an 8-bit UNORM
// attachment stores as `40 80 c0 ff`. The constants are byte/255 rather than
// round decimals on purpose: a half-integer tie such as one half times 255 is
// resolved differently by different drivers — the render probe read it back as
// 0x80 on Lavapipe but 0x7f on the NVIDIA driver and dzn (`research/docs/23`
// §3.5). Byte/255 values sit far from a tie on every driver.
fragment float4 render_solid_rgba8() {
    return float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
}
