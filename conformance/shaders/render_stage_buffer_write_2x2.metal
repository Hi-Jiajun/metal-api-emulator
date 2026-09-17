#include <metal_stdlib>
using namespace metal;

// Reviewed writable stage-buffer fixture (`research/docs/23` §92, R9k): one MSL
// module whose vertex stage reads its positions out of its own `[[buffer(0)]]`
// argument with a vertex-index stride, and whose fragment stage reads one
// `float4` from `[[buffer(0)]]`, writes it to `[[buffer(1)]]` and adds one to
// the `float4` its `[[buffer(2)]]` argument already holds. It is the native
// rail's sibling of the Vulkan write half and affine half R9f executes through
// translated modules: the write half's `source`/`sink` pair and the affine
// half's `positions[vertex_id]` read are the same two shapes, stated here in
// MSL and paired against a reviewed slot table rather than a SPIR-V reflection.
//
// The module is the whole review surface for this shape, so the provider rail
// (`crates/metal-api-native/src/render.rs`, `REVIEWED_STAGE_BUFFER_WRITE_SOURCE`)
// and the Swift oracle (`conformance/NativeOracle.swift`) pin these exact
// bytes; an edit here has to update both pins and the fixture's expected
// bytes.

struct RenderStageBufferWriteVertexOut {
    float4 position [[position]];
};

// Vertex stage: the position is the pair of `float` components at
// `vertex_id * 8` and `vertex_id * 8 + 4` in the stage's own `[[buffer(0)]]`
// argument, so the declared reach is the reflected affine one — two four-byte
// accesses strided by eight bytes over the vertex index, exactly the access set
// `crates/metal-api-vulkan/tests/fixtures/render_stage_buffer_positions.vert.ll`
// translates to. Two scalars rather than one `float2` because that is the
// measurement the declaration states: the rail proves the bytes the stage
// reaches, and a vector load would be a different (also true) access set. The
// pipeline binds no `MTLVertexDescriptor` (`VertexLayout::None`), while the
// buffer argument is the geometry's only source: a rail that ignores the slot
// cannot draw the shape the case's bytes state.
vertex RenderStageBufferWriteVertexOut render_stage_buffer_write_vertex(
    uint vertex_id [[vertex_id]],
    device const float* positions [[buffer(0)]]) {
    const float2 position =
        float2(positions[vertex_id * 2], positions[vertex_id * 2 + 1]);
    RenderStageBufferWriteVertexOut out;
    out.position = float4(position, 0.0, 1.0);
    return out;
}

// Fragment stage: three `[[buffer(N)]]` arguments in one stage, one per access
// arm the contract admits (`research/docs/23` §3.3, v86). `source` is read and
// returned, so the attachment texel and the two writebacks move together;
// `sink` is written with that same texel and never read, which is what a
// `Write` declaration pairs with; `accumulator` is read *and* written — one is
// added to whatever the bytes the pass bound already hold — which is what a
// `ReadWrite` declaration pairs with, and what makes "the previous bytes were
// uploaded" falsifiable: a rail that bound zeros publishes `1.0` rather than
// `initial + 1.0`.
fragment float4 render_stage_buffer_write_tint(
    device const float4* source [[buffer(0)]],
    device float4* sink [[buffer(1)]],
    device float4* accumulator [[buffer(2)]]) {
    const float4 texel = source[0];
    sink[0] = texel;
    accumulator[0] = accumulator[0] + float4(1.0);
    return texel;
}
