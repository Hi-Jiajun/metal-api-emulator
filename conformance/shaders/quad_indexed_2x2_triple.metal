#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the three-attachment shape (`research/docs/23`
// §3.3, v25): the indexed vertex stage of `quad_indexed_2x2.metal` with a
// fragment stage that writes three colour locations. Three is not the ceiling
// (`MAX_COLOR_ATTACHMENTS` is four) but it is a shape a guest workload asks for,
// and the four-location module cannot stand in for it: a fragment that writes a
// location with no attachment beside it is undefined. The three texels are
// pairwise distinct — `40 80 c0 ff`, `ff 80 40 c0`, `c0 40 ff 80` — so a
// capture that landed one target twice reads the wrong bytes and cannot pass.

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

// The three colour locations of the reviewed shape.
struct TripleFragmentOut {
    float4 colour_0 [[color(0)]];
    float4 colour_1 [[color(1)]];
    float4 colour_2 [[color(2)]];
};

fragment TripleFragmentOut render_solid_rgba8_triple() {
    TripleFragmentOut out;
    out.colour_0 = float4(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    out.colour_1 = float4(1.0, 128.0 / 255.0, 64.0 / 255.0, 192.0 / 255.0);
    out.colour_2 = float4(192.0 / 255.0, 64.0 / 255.0, 1.0, 128.0 / 255.0);
    return out;
}
