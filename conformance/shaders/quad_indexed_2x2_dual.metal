#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the native MRT milestone (wave3 R1): one MSL
// module whose vertex stage is the same reviewed `render_quad_vertex` entry
// `quad_indexed_2x2.metal` carries, and whose fragment stage writes two
// colour outputs through a `[[color(n)]]` struct instead of one return value.
// The module is the whole review surface for the native rail, so the provider
// rail (`crates/metal-api-native/src/render.rs`, `REVIEWED_DUAL_SOURCE`) and
// the Swift oracle (`conformance/NativeOracle.swift`, `reviewedDualModule()`)
// pin these exact bytes; an edit here has to update both pins and the
// fixture's expected texel bytes.

// One vertex of the reviewed stream, identical to `quad_indexed_2x2.metal`:
// a two-component position at attribute 0, which the descriptor binds to
// buffer 0 at offset 0 with stride 8. The values are already in NDC, so the
// stage only widens them to a clip-space position.
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

// The two colour outputs the pass's locations receive: the vertex-input
// milestone's texel at location 0, and the MRT milestone's second texel at
// location 1. Both are byte/255 constants for the same reason the
// single-output modules use them: they sit far from a half-integer tie,
// which different drivers resolve differently (`research/docs/23` §3.5).
// An 8-bit UNORM attachment stores them as `40 80 c0 ff` and `ff 80 40 c0`
// respectively.
struct FragmentOut {
    float4 a [[color(0)]];
    float4 b [[color(1)]];
};

fragment FragmentOut render_solid_rgba8_dual() {
    FragmentOut out;
    out.a = float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    out.b = float4(255.0 / 255.0, 128.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0);
    return out;
}
