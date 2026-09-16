#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the instancing milestone (`research/docs/23`
// §3.3, v31): one MSL module whose two stage entries are compiled into one
// `MTLRenderPipelineState`, and whose vertex stage reads *two* caller-held
// streams declared by an `MTLVertexDescriptor` — a per-vertex `float32x2`
// position and a per-instance `float32x4` tint — plus the `instance_id` builtin.
// The module is the whole review surface for the native rail, so the provider
// rail (`crates/metal-api-native/src/render.rs`) and the Swift oracle
// (`conformance/NativeOracle.swift`) pin these exact bytes; an edit here has to
// update both pins, the `instanced_quad.vert.spvasm` / `instanced_tint.frag.spvasm`
// Vulkan counterparts and the fixture's expected texel bytes.

// One vertex of the reviewed stream pair: a two-component position at attribute
// 0 (buffer 0, offset 0, stride 8, per vertex) and a four-component tint at
// attribute 1 (buffer 1, offset 0, stride 16, per instance).
struct InstancedQuadVertex {
    float2 position [[attribute(0)]];
    float4 tint [[attribute(1)]];
};

struct InstancedQuadOut {
    float4 position [[position]];
    float4 tint;
};

// The vertex stage runs once per vertex of every instance. The position stream
// advances per vertex, the tint per instance, and `instance_id` selects which
// half of the viewport this instance's copy covers: instance 0 keeps the left
// half (`x' = x * 0.5 - 0.5`), instance 1 the right half
// (`x' = x * 0.5 + 0.5`). The y axis is untouched, so each instance covers the
// full height of its half and the two tints land in disjoint texels — which is
// what makes "the tint stream really stepped per instance" observable instead
// of a value that overwrote itself.
vertex InstancedQuadOut render_instanced_quad_vertex(InstancedQuadVertex in [[stage_in]],
                                                     uint instance_id [[instance_id]]) {
    InstancedQuadOut out;
    float shift = (instance_id == 0) ? -0.5 : 0.5;
    out.position = float4(in.position.x * 0.5 + shift, in.position.y, 0.0, 1.0);
    out.tint = in.tint;
    return out;
}

// Fragment stage: store the tint the vertex stage forwarded. The fixture's
// instance tints are (1, 0, 0, 1) and (0, 1, 0, 1) — exactly representable
// bytes (0xff / 0x00), so the two halves read back as `ff 00 00 ff` and
// `00 ff 00 ff` and neither sits on a half-integer UNORM tie
// (`research/docs/23` §3.5).
fragment float4 render_instanced_tint(InstancedQuadOut in [[stage_in]]) {
    return in.tint;
}
