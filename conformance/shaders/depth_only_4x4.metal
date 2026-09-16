#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the zero-colour-attachment depth pass
// (`research/docs/23` §3.3, v46). The vertex stage reads the same two-attribute
// stream the depth pair reads — a `float32x3` position at attribute 0 and a
// `float32x4` tint at attribute 1 — and forwards the position; the fragment
// stage is **void**, which is what makes a pass with no colour attachment well
// formed. The Metal Shading Language Specification states it directly: "If the
// fragment function does not generate output, it returns void", and the
// pipeline still rasterizes, so the depth test and the depth write happen for
// every covered fragment while nothing writes a colour target that does not
// exist. The Vulkan counterpart is `depth_only.frag.spvasm`, whose entry point
// declares no Output at all.

/// One vertex of the reviewed stream: the depth pair's own attributes.
struct DepthOnlyVertex {
    float3 position [[attribute(0)]];
    float4 tint [[attribute(1)]];
};

struct DepthOnlyOut {
    float4 position [[position]];
    float4 tint;
};

// The vertex stage forwards both attributes, exactly as the depth pair's does:
// the tint travels to a varying no fragment stage reads, so the *only* thing
// this module can put on the surface is depth.
vertex DepthOnlyOut render_depth_only_vertex(DepthOnlyVertex in [[stage_in]]) {
    DepthOnlyOut out;
    out.position = float4(in.position, 1.0);
    out.tint = in.tint;
    return out;
}

// The fragment stage generates no output: no colour attachment exists, and the
// rasterizer's own depth handling is the whole observation.
fragment void render_depth_only_fragment() {
}
