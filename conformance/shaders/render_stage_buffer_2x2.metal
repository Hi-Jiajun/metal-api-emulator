#include <metal_stdlib>
using namespace metal;

// Reviewed stage-buffer fixture (`research/docs/23` §83, R9g): one MSL module
// whose two stage entries read their bytes from `[[buffer(N)]]` arguments
// instead of constants or `[[stage_in]]` streams. It is the native rail's
// sibling of the Vulkan pair (`crates/metal-api-vulkan/src/render_spv/
// stage_buffer_positions.vert.spv` + `stage_buffer_tint.frag.spv`): the vertex
// stage reads the three positions, the fragment stage reads one `float4` tint,
// and each stage reads its own binding index space — a vertex `buffer(0)` and a
// fragment `buffer(0)` are two different slots, exactly as
// `setVertexBuffer(_:offset:index:)` and `setFragmentBuffer(_:offset:index:)`
// state. The module is the whole review surface for the native rail, so the
// provider rail (`crates/metal-api-native/src/render.rs`,
// `REVIEWED_STAGE_BUFFER_SOURCE`) and the Swift oracle
// (`conformance/NativeOracle.swift`) pin these exact bytes; an edit here has to
// update both pins and the fixture's expected texel bytes.

// Vertex stage: the position comes from the stage's own `[[buffer(0)]]`
// argument — three `float2` in Metal's own clip space — and `vertex_id` only
// selects the record. The pipeline therefore binds no `MTLVertexDescriptor`
// (`VertexLayout::None`), while the buffer argument is the geometry's only
// source: a rail that ignores the slot cannot draw the shape the case's bytes
// state. The triangle (-1,1), (0.25,1), (-1,-0.25) covers the pixel centre of
// the 2x2 attachment's top-left texel (`x = -0.5`, `y = +0.5`, the Metal y-up
// convention the reviewed fixtures use) with a wide margin, and no other texel
// centre: the readback's covered texel is what makes "the binding arrived"
// falsifiable texel by texel.
struct RenderStageBufferVertexOut {
    float4 position [[position]];
};

vertex RenderStageBufferVertexOut render_stage_buffer_vertex(
    uint vertex_id [[vertex_id]],
    device const float2* positions [[buffer(0)]]) {
    RenderStageBufferVertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    return out;
}

// Fragment stage: the stored texel is the stage's own `[[buffer(0)]]` argument
// — one `float4`, translated by the attachment's format quantisation and
// nothing else. Byte/255 values are what the case states, because a
// half-integer tie resolves differently across drivers (`research/docs/23`
// §3.5).
fragment float4 render_stage_buffer_tint(
    device const float4* tint [[buffer(0)]]) {
    return tint[0];
}
