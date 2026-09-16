#include <metal_stdlib>
using namespace metal;

// Reviewed render fixture for the depth milestone (`research/docs/23` §3.3,
// v36): one MSL module whose vertex stage reads a caller-held `float32x3`
// position (so the *caller* chooses each triangle's depth) and a `float32x4`
// tint, and whose fragment stage stores the forwarded tint. The module is the
// whole review surface for the native rail, so the provider rail
// (`crates/metal-api-native/src/render.rs`) and the Swift oracle
// (`conformance/NativeOracle.swift`) pin these exact bytes; an edit here has to
// update both pins, the `depth_pair.vert.spvasm` / `depth_pair_tint.frag.spvasm`
// Vulkan counterparts and the fixture's expected texel bytes.

// One vertex of the reviewed stream: a three-component position at attribute 0
// (buffer 0, offset 0, stride 32) and a four-component tint at attribute 1
// (buffer 0, offset 16). The position's z is what the depth test reads.
struct DepthPairVertex {
    float3 position [[attribute(0)]];
    float4 tint [[attribute(1)]];
};

struct DepthPairOut {
    float4 position [[position]];
    float4 tint;
};

// The vertex stage runs once per vertex. It does not reorder anything: the
// fixture's two triangles are the two oversize triangles the reviewed quad
// module already draws, one at z = 0.5 and one at z = 0.9, so the *only*
// difference the depth state can make is which of the two survives where they
// overlap — which is everywhere.
vertex DepthPairOut render_depth_pair_vertex(DepthPairVertex in [[stage_in]]) {
    DepthPairOut out;
    out.position = float4(in.position, 1.0);
    out.tint = in.tint;
    return out;
}

// Fragment stage: store the tint the vertex stage forwarded. The fixture's two
// tints are (1, 0, 0, 1) and (0, 1, 0, 1) — exactly representable bytes
// (`0xff` / `0x00`), so the surviving triangle's colour is exact and no texel
// sits on a half-integer UNORM tie (`research/docs/23` §3.5).
fragment float4 render_depth_pair_tint(DepthPairOut in [[stage_in]]) {
    return in.tint;
}
