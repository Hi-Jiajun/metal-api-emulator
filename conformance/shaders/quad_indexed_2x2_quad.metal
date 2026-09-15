#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the four-attachment ceiling (`research/docs/23`
// §3.3, v24): the indexed vertex stage of `quad_indexed_2x2.metal` with a
// fragment stage that writes all four colour locations core admission allows
// (`MAX_COLOR_ATTACHMENTS`). The four texels are pairwise distinct on purpose —
// `40 80 c0 ff`, `ff 80 40 c0`, `c0 40 ff 80`, `80 c0 40 ff` — so a capture
// that wrote one target twice, or swapped two locations, reads the wrong bytes
// and cannot pass the comparison. Every component is a multiple of 64/255, far
// from a half-integer tie. Like every reviewed module, this file's exact bytes
// are pinned by both rails and by the fixture's own source digest.

// One vertex of the reviewed stream: a two-component position at attribute 0,
// which the descriptor binds to buffer 0 at offset 0 with stride 8.
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

// The four colour locations of the reviewed MRT fixture.
struct QuadFragmentOut {
    float4 colour_0 [[color(0)]];
    float4 colour_1 [[color(1)]];
    float4 colour_2 [[color(2)]];
    float4 colour_3 [[color(3)]];
};

fragment QuadFragmentOut render_solid_rgba8_quad() {
    QuadFragmentOut out;
    out.colour_0 = float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    out.colour_1 = float4(1.0, 128.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0);
    out.colour_2 = float4(192.0 / 255.0, 64.0 / 255.0, 1.0, 128.0 / 255.0);
    out.colour_3 = float4(128.0 / 255.0, 192.0 / 255.0, 64.0 / 255.0, 1.0);
    return out;
}
