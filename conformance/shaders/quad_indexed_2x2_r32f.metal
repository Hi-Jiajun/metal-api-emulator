#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the single-channel float attachment
// (`research/docs/23` §3.3, v22): the indexed vertex stage of
// `quad_indexed_2x2.metal` with a *one-component* fragment stage, because a
// `r32float` attachment takes a one-component store. The Vulkan rail carries
// the same shape as `solid_r32f.frag.spv`, and both rails refuse the
// four-component module for this format rather than letting a mismatched store
// decide the bytes. Like every reviewed module, this file's exact bytes are
// pinned by the provider rail (`crates/metal-api-native/src/render.rs`), the
// Swift oracle (`conformance/NativeOracle.swift`) and the fixture's own source
// digest.

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

// Fragment stage: the red component of the reviewed colour, `64/255`, which a
// 32-bit float attachment stores as `81 80 80 3e` in little-endian byte order.
// The constant is written as a byte value over 255 for the same reason the
// other modules are: it sits far from a half-integer tie, which different
// drivers resolve differently (`research/docs/23` §3.5).
fragment float render_solid_r32f() {
    return 64.0 / 255.0;
}
