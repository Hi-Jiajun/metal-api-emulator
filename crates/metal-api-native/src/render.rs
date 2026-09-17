//! Offscreen render rail for the native provider (`research/docs/23` §6 Steps 6
//! and 7, §6 Step 3.3 for the vertex-input half).
//!
//! The rail answers the one question Step 6 owns on the Apple side: can the
//! native provider build a colour attachment, a render pass descriptor, a
//! two-entry render pipeline state and a full-screen-triangle draw out of one
//! reviewed MSL module, and read the attachment's texels back byte for byte.
//! Step 7 is the trace path over it: [`plan_trace`] decides ordering, the
//! reviewed allowlist, the attachment landing and the load op from values
//! alone, [`TraceRenderPlan::writeback`] and [`merge_writebacks`] put the texels
//! into the same writeback channel a compute pass uses, and `native.rs` calls
//! both around `encode_offscreen_render`.
//!
//! The vertex-input increment adds the caller-held half of the same contract
//! (`docs/23` §3.3): a pass may bind vertex streams and an index buffer whose
//! views carry their own bytes, which this rail reads out of the pass itself,
//! proves the footprint of on the host ([`plan_vertex_input`]) and translates
//! into an `MTLVertexDescriptor` plus `setVertexBuffer` /
//! `drawIndexedPrimitives`. Those bytes are uploaded here rather than through the
//! compute rail's pool, which only carries views a compute binding declares
//! (`docs/23` §3.6). Two reviewed modules exist for the two shapes — `vertex_id`
//! positions ([`REVIEWED_SOURCE`]) and `[[stage_in]]` positions
//! ([`REVIEWED_VERTEX_SOURCE`]) — and the pipeline's [`VertexLayout`] is what
//! selects which one may compile, so a trace cannot reach source the rail did not
//! review.
//!
//! **The encoder body has still never run on an Apple GPU.** What is
//! different from the pre-flip state is the evidence: CI run `34774478149`
//! (`native-oracle-build`, commit `fb4f8da`) ran the oracle's `--render-selftest`
//! — the single-device check in `conformance/RENDER-CAPTURE.md` §5, over the
//! same reviewed module and the same `runRenderCase` a suite would use — on an
//! Apple Paravirtual device, read the 2x2 attachment back as `4080c0ff` four
//! times and printed `render_selftest: PASS`. That is the flip condition the
//! provider's render bits name (`native.rs`, `ProviderCapabilities`). The two
//! rules this rail shares with its sibling on the Vulkan side
//! (`crates/metal-api-vulkan/src/render.rs`, which does execute on Lavapipe and
//! the RTX 5060) are the fixed 2x2 extent and the byte/255 fragment constants
//! that keep 8-bit UNORM rounding away from a half-integer tie.
//!
//! Shape reuse: this rail consumes the core values — [`RenderPassDescriptor`]
//! and [`RenderPipelineContract`] — and re-runs their validators instead of
//! restating the first increment's field rules. What it adds is what only a
//! device can answer: the pixel-format mapping, the clear-value decoding per
//! format, and the fact that one reviewed MSL module is the only source it will
//! compile.
#![allow(dead_code)] // the trace path does not reach this rail yet (Step 4/7)

#[cfg(target_os = "macos")]
use crate::icb;
use crate::refusal;
use metal_api_core::provider::{
    AttachmentFormat, BufferSource, BufferView, BufferWriteback, ClearColor, ComputeTrace,
    ContractError, DepthResolveFilter, DepthStoreOp, DepthTest, FieldValue, IndexBufferBinding,
    IndexFormat, IndirectCommandDescriptor, LoadOp, PipelineId, PresentDescriptor, PresentMode,
    ProviderError, ProviderErrorClass, ProviderPhase, RenderPassBlend, RenderPassCull,
    RenderPassDescriptor, RenderPipelineContract, SampleCount, StencilResolveFilter, StencilTest,
    StoreOp, TextureFormat, TracePass, VertexFormat, VertexLayout, VertexStep, ViewId,
    FULL_SCREEN_TRIANGLE_VERTICES,
};
use std::collections::BTreeMap;

// The depth compare function is only named by the encoder body, which exists
// on macOS alone; the plan's own `DepthTest` travels unchanged everywhere.
#[cfg(target_os = "macos")]
use foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, DepthStencilDescriptor, Device,
    IndirectCommandBufferDescriptor, MTLBlendFactor, MTLBlendOperation, MTLClearColor,
    MTLCommandBufferStatus, MTLCompareFunction, MTLCullMode, MTLIndexType, MTLIndirectCommandType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions,
    MTLSize, MTLStencilOperation, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLVertexFormat, MTLVertexStepFunction, MTLViewport, MTLWinding, NSInteger, NSRange,
    NSUInteger, RenderPassDescriptor as MetalRenderPassDescriptor, RenderPipelineDescriptor,
    RenderPipelineState, StencilDescriptor, Texture, TextureDescriptor, VertexDescriptor,
};
#[cfg(target_os = "macos")]
use metal_api_core::provider::{
    BlendFactor, BlendOperation, CompareFunction, CullMode as ContractCullMode, StencilCompare,
    StencilOp, Winding as ContractWinding,
};
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl};

/// The reviewed render fixture: one MSL module carrying both stage entries.
///
/// Exact byte equality is the same discipline the compute allowlist uses
/// (`lib.rs::bounded_contract`): a matching entry name cannot establish the
/// footprint of caller-supplied source, so the rail compiles these bytes and
/// nothing else.
pub(crate) const REVIEWED_SOURCE: &str =
    include_str!("../../../conformance/shaders/render_offscreen_2x2.metal");

/// The reviewed indexed fixture: the same two stages, but the vertex stage
/// reads its position from `[[stage_in]]`, i.e. from a caller-held vertex
/// stream, and the pass draws through an index buffer.
///
/// A second constant rather than a second entry pair in one module, because the
/// two shapes need different pipeline state: the `vertex_id` module carries no
/// `MTLVertexDescriptor` at all, while the indexed one is meaningless without
/// the descriptor the pass's stream declares. Both files are pinned by exact
/// bytes for the same reason [`REVIEWED_SOURCE`] is: a matching entry name
/// cannot establish the footprint of caller-supplied source.
pub(crate) const REVIEWED_VERTEX_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2.metal");

/// Vertex entry of the reviewed module: positions from `vertex_id`, no vertex
/// buffer (`VertexLayout::None`).
pub(crate) const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// Fragment entry of the reviewed module: the fixed colour texel.
pub(crate) const FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// Vertex entry of the reviewed indexed module: the position arrives through
/// `[[stage_in]]`, i.e. through the vertex descriptor the pipeline carries.
pub(crate) const QUAD_VERTEX_ENTRY: &str = "render_quad_vertex";

/// The reviewed dual module: the indexed vertex entry once more, but a
/// fragment stage that writes two colour outputs, one per MRT location.
pub(crate) const REVIEWED_DUAL_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_dual.metal");

/// Fragment entry of the reviewed dual module: two `[[color(n)]]` outputs,
/// location 0 the vertex-input texel and location 1 the second MRT texel.
pub(crate) const DUAL_FRAGMENT_ENTRY: &str = "render_solid_rgba8_dual";

/// The reviewed single-channel float module: the indexed vertex stage plus a
/// one-component fragment stage (`conformance/shaders/quad_indexed_2x2_r32f.metal`).
/// An `r32float` attachment takes a one-component store, so the four-component
/// module the other single-output shapes use cannot stand in for it
/// (`research/docs/23` §3.3, v22).
pub(crate) const REVIEWED_R32F_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_r32f.metal");

/// Fragment entry of the reviewed single-channel float module.
pub(crate) const R32F_FRAGMENT_ENTRY: &str = "render_solid_r32f";

/// The reviewed four-location module: the indexed vertex stage plus a fragment
/// stage that writes all `MAX_COLOR_ATTACHMENTS` colour locations
/// (`conformance/shaders/quad_indexed_2x2_quad.metal`). The four texels are
/// pairwise distinct, so a capture that landed one target twice cannot pass
/// (`research/docs/23` §3.3, v24).
pub(crate) const REVIEWED_QUAD_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_quad.metal");

/// Fragment entry of the reviewed four-location module.
pub(crate) const QUAD_FRAGMENT_ENTRY: &str = "render_solid_rgba8_quad";

/// The reviewed three-location module: the indexed vertex stage plus a fragment
/// stage that writes three colour locations
/// (`conformance/shaders/quad_indexed_2x2_triple.metal`). Three is not the
/// ceiling, but it is its own shape (`research/docs/23` §3.3, v25).
pub(crate) const REVIEWED_TRIPLE_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_triple.metal");

/// Fragment entry of the reviewed three-location module.
pub(crate) const TRIPLE_FRAGMENT_ENTRY: &str = "render_solid_rgba8_triple";

/// The reviewed instanced module (`research/docs/23` §3.3, v31): a vertex stage
/// that reads the caller-held quad positions, a per-instance tint and the
/// `instance_id` builtin, plus a fragment stage that stores the forwarded tint.
pub(crate) const REVIEWED_INSTANCED_SOURCE: &str =
    include_str!("../../../conformance/shaders/instanced_quad_2x2.metal");

/// Vertex entry of the reviewed instanced module.
pub(crate) const INSTANCED_VERTEX_ENTRY: &str = "render_instanced_quad_vertex";

/// Fragment entry of the reviewed instanced module.
pub(crate) const INSTANCED_FRAGMENT_ENTRY: &str = "render_instanced_tint";

/// The reviewed depth module (`research/docs/23` §3.3, v36): a vertex stage
/// that reads a caller-held `float32x3` position (so the caller chooses each
/// triangle's depth) and a `float32x4` tint, plus a fragment stage that stores
/// the forwarded tint.
pub(crate) const REVIEWED_DEPTH_SOURCE: &str =
    include_str!("../../../conformance/shaders/depth_pair_4x4.metal");

/// Vertex entry of the reviewed depth module.
pub(crate) const DEPTH_VERTEX_ENTRY: &str = "render_depth_pair_vertex";

/// Fragment entry of the reviewed depth module.
pub(crate) const DEPTH_FRAGMENT_ENTRY: &str = "render_depth_pair_tint";

/// The reviewed zero-colour-attachment depth module (`research/docs/23` §3.3,
/// v46): the depth pair's vertex stage beside a **void** fragment stage, so a
/// pass with no colour attachment still tests and writes depth.
pub(crate) const REVIEWED_DEPTH_ONLY_SOURCE: &str =
    include_str!("../../../conformance/shaders/depth_only_4x4.metal");

/// Vertex entry of the reviewed zero-colour-attachment depth module.
pub(crate) const DEPTH_ONLY_VERTEX_ENTRY: &str = "render_depth_only_vertex";

/// Fragment entry of the reviewed zero-colour-attachment depth module.
pub(crate) const DEPTH_ONLY_FRAGMENT_ENTRY: &str = "render_depth_only_fragment";

/// One reviewed render module and the (vertex-input shape, colour-format
/// shape) pair it was written for.
///
/// The [`VertexLayout`] and the pipeline's [`RenderPipelineContract`]
/// `color_formats` together select the entry pair: a pipeline whose layout
/// binds streams and whose format list names one attachment can only be the
/// module whose vertex stage reads `[[stage_in]]` and whose fragment stage
/// returns one colour, a `VertexLayout::None` single-output pipeline only the
/// module that derives positions from `vertex_id`, and an indexed pipeline
/// whose two formats are both `Rgba8Unorm` only the dual module whose fragment
/// stage writes both locations. Both the entry pair and the source bytes of
/// that one module are then the allowlist, so neither a renamed entry nor an
/// edited file can execute.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReviewedModule {
    /// The exact module bytes this rail compiles.
    pub(crate) source: &'static str,
    /// The module's path, for refusals and the documentation's sake.
    pub(crate) path: &'static str,
    /// The vertex entry the module carries.
    pub(crate) vertex_entry: &'static str,
    /// The fragment entry the module carries.
    pub(crate) fragment_entry: &'static str,
    /// Whether the module's vertex stage reads a caller-held stream.
    pub(crate) binds_buffers: bool,
}

/// The reviewed modules, one per (vertex-input shape, colour-format shape)
/// pair this rail executes.
pub(crate) const REVIEWED_MODULES: [ReviewedModule; 9] = [
    ReviewedModule {
        source: REVIEWED_SOURCE,
        path: "conformance/shaders/render_offscreen_2x2.metal",
        vertex_entry: VERTEX_ENTRY,
        fragment_entry: FRAGMENT_ENTRY,
        binds_buffers: false,
    },
    ReviewedModule {
        source: REVIEWED_VERTEX_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    ReviewedModule {
        source: REVIEWED_DUAL_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_dual.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: DUAL_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    ReviewedModule {
        source: REVIEWED_R32F_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_r32f.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: R32F_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    ReviewedModule {
        source: REVIEWED_QUAD_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_quad.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: QUAD_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    ReviewedModule {
        source: REVIEWED_TRIPLE_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_triple.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: TRIPLE_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    // The reviewed instanced fixture (`research/docs/23` §3.3, v31): the one
    // module whose vertex stage reads `instance_id` and a per-instance stream,
    // and whose fragment stage stores the tint that stage forwarded.
    ReviewedModule {
        source: REVIEWED_INSTANCED_SOURCE,
        path: "conformance/shaders/instanced_quad_2x2.metal",
        vertex_entry: INSTANCED_VERTEX_ENTRY,
        fragment_entry: INSTANCED_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    // The reviewed depth fixture (`research/docs/23` §3.3, v36): the module
    // whose vertex stage reads a position whose z the caller chose, so the
    // depth state is what decides which triangle survives.
    ReviewedModule {
        source: REVIEWED_DEPTH_SOURCE,
        path: "conformance/shaders/depth_pair_4x4.metal",
        vertex_entry: DEPTH_VERTEX_ENTRY,
        fragment_entry: DEPTH_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
    // The reviewed zero-colour-attachment depth fixture (`research/docs/23`
    // §3.3, v46): the same vertex stage with a fragment stage that generates no
    // output, which is the shape a pass with no colour attachment compiles.
    ReviewedModule {
        source: REVIEWED_DEPTH_ONLY_SOURCE,
        path: "conformance/shaders/depth_only_4x4.metal",
        vertex_entry: DEPTH_ONLY_VERTEX_ENTRY,
        fragment_entry: DEPTH_ONLY_FRAGMENT_ENTRY,
        binds_buffers: true,
    },
];

/// The reviewed module a pipeline's vertex-input shape and colour-format list
/// select, or `None` for a shape no module was reviewed for.
///
/// The single-output modules are format-agnostic: the pipeline's attachment
/// format is pipeline state, not module source, so one module serves every
/// admitted single format (`SUPPORTED_COLOR_FORMATS` below). The dual module
/// is reviewed for exactly two `Rgba8Unorm` locations, because its fragment
/// stage writes two byte strings that only those two locations decode the
/// reviewed way; any other two-format list, any list above this rail's cap
/// and any `vertex_id` pipeline with more than one location has no reviewed
/// module and is refused by [`review_contract`] and [`plan`].
pub(crate) fn reviewed_module(
    layout: &VertexLayout,
    color_formats: &[AttachmentFormat],
) -> Option<&'static ReviewedModule> {
    // The 8-bit UNORM modules are layout-agnostic — the same store lands in
    // whichever channel order each attachment declares — so any mix of the two
    // 8-bit formats is served by the module of its attachment count
    // (`research/docs/23` §3.3, v26). The single-channel float module is the one
    // format-specific stage; every shape that fits no reviewed module is
    // refused rather than matched approximately.
    let unorm8 = |format: &AttachmentFormat| {
        matches!(
            format,
            AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm
        )
    };
    match (layout, color_formats) {
        (VertexLayout::None, [single]) if SUPPORTED_COLOR_FORMATS.contains(single) => {
            Some(&REVIEWED_MODULES[0])
        }
        // The depth module is selected by the layout's own shape: one stream
        // with two attributes — a `float32x3` position and a `float32x4` tint —
        // is the shape its vertex stage reads, so it is matched before the
        // instanced and plain single-output arms
        // (`research/docs/23` §3.3, v36).
        //
        // An *empty* format list is that shape with no colour target at all:
        // the zero-colour-attachment depth pass, whose fragment stage generates
        // no output (`research/docs/23` §3.3, v46).
        (VertexLayout::Buffers(buffers), [])
            if buffers.len() == 1 && buffers[0].attributes.len() == 2 =>
        {
            Some(&REVIEWED_MODULES[8])
        }
        (VertexLayout::Buffers(buffers), [single])
            if unorm8(single) && buffers.len() == 1 && buffers[0].attributes.len() == 2 =>
        {
            Some(&REVIEWED_MODULES[7])
        }
        // The instanced module is selected by the layout's own step function,
        // not by the format list alone: a per-instance binding is the shape its
        // vertex stage was written for, so it is matched before the plain
        // single-output arm (`research/docs/23` §3.3, v31).
        (VertexLayout::Buffers(buffers), [single])
            if unorm8(single)
                && buffers
                    .iter()
                    .any(|buffer| buffer.step == VertexStep::PerInstance) =>
        {
            Some(&REVIEWED_MODULES[6])
        }
        (VertexLayout::Buffers(_), [AttachmentFormat::R32Float]) => Some(&REVIEWED_MODULES[3]),
        (VertexLayout::Buffers(_), [single]) if unorm8(single) => Some(&REVIEWED_MODULES[1]),
        (VertexLayout::Buffers(_), [first, second]) if unorm8(first) && unorm8(second) => {
            Some(&REVIEWED_MODULES[2])
        }
        (VertexLayout::Buffers(_), [first, second, third])
            if unorm8(first) && unorm8(second) && unorm8(third) =>
        {
            Some(&REVIEWED_MODULES[5])
        }
        (VertexLayout::Buffers(_), formats)
            if formats.len() == usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap_or(usize::MAX)
                && formats.iter().all(unorm8) =>
        {
            Some(&REVIEWED_MODULES[4])
        }
        _ => None,
    }
}

/// Colour attachments this rail executes today: two. The core contract admits
/// the full MRT shape (up to `metal_api_core::provider::MAX_COLOR_ATTACHMENTS`,
/// now 4) while this rail's capability bit stays at 2: the reviewed dual
/// module (`REVIEWED_DUAL_SOURCE`) writes exactly two locations, and no module
/// writes three or four, so a pass above two is refused rather than rendered
/// partially. The value is restated here because a capability value has to be
/// spelled by the provider that declares it (`research/docs/23` §4.2).
pub(crate) const MAX_COLOR_ATTACHMENTS: u32 = 4;

/// Vertex streams one render pass may bind. The same value the core contract
/// states (`metal_api_core::provider::MAX_VERTEX_BUFFERS`), restated for the
/// same reason [`MAX_COLOR_ATTACHMENTS`] is: a capability value belongs to the
/// provider that declares it (`research/docs/23` §3.3, §4.2).
pub(crate) const MAX_VERTEX_BUFFERS: u32 = metal_api_core::provider::MAX_VERTEX_BUFFERS as u32;

/// Largest attachment the first milestone renders into: 2x2, so full coverage
/// stays distinguishable from "one texel was written" (`research/docs/23` §1.3).
pub(crate) const MAX_ATTACHMENT_DIMENSION: [u64; 2] = [4, 4];

/// Colour formats this rail can build an `MTLTexture` and a pipeline state from
/// — the core contract's admitted set, without `R32Uint` ([`pixel_format`]
/// refuses that one).
pub(crate) const SUPPORTED_COLOR_FORMATS: [AttachmentFormat; 3] = AttachmentFormat::ADMITTED;

/// The render bits the provider declares, in one value so the macOS capability
/// snapshot (`native.rs`) and the host-side unit tests cannot drift.
///
/// Every field is the rail's own limit, so capability admission and this rail
/// agree by construction; the unit tests assert that agreement against core's
/// [`metal_api_core::provider::ProviderCapabilities::admit`], which is the only
/// place the two could otherwise diverge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RenderCapabilityBits {
    pub(crate) supports_render_passes: bool,
    pub(crate) max_color_attachments: u32,
    pub(crate) max_attachment_dimension: [u64; 2],
    pub(crate) supported_color_formats: Vec<AttachmentFormat>,
    /// Render-sampler bits, declared next to the render bits for the same
    /// reason: the snapshot and the rail cannot disagree about what this
    /// provider samples (`research/docs/23` §3.3, v70). The three fields come
    /// from [`render_texture_capability_bits`], so their flip condition is one
    /// observation rather than a second set of inline literals that could
    /// drift from the comment.
    pub(crate) supports_render_texture_sampling: bool,
    pub(crate) max_render_textures: u32,
    pub(crate) supported_render_texture_formats: Vec<TextureFormat>,
    /// Present bits, declared next to the render bits for the same reason: the
    /// snapshot and the rail cannot disagree about what this provider runs.
    /// The four fields come from [`present_capability_bits`], so their flip
    /// condition is one observation (`research/docs/24` §6 Step 7) rather than
    /// a second set of inline literals that could drift from the comment.
    pub(crate) supports_presentation: bool,
    pub(crate) max_present_targets: u32,
    pub(crate) supported_present_modes: Vec<PresentMode>,
    pub(crate) max_present_image_count: u32,
}

/// The present bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side unit tests cannot drift.
///
/// Split out from the render bits because the flip condition is a separate
/// single-device observation: the Swift oracle's `--present-selftest` on an
/// Apple GPU, exactly as the render bits rest on `--render-selftest`
/// (`research/docs/24` §6 Step 7).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PresentCapabilityBits {
    pub(crate) supports_presentation: bool,
    pub(crate) max_present_targets: u32,
    pub(crate) supported_present_modes: Vec<PresentMode>,
    pub(crate) max_present_image_count: u32,
}

/// The first presentation increment's target cap, spelled once so the snapshot
/// and its tests cannot drift from core's value (`research/docs/24` §3.1).
pub(crate) const MAX_PRESENT_TARGETS: u32 = metal_api_core::provider::MAX_PRESENT_TARGETS as u32;

/// The first presentation increment's image cap, same rule as
/// [`MAX_PRESENT_TARGETS`].
pub(crate) const MAX_PRESENT_IMAGE_COUNT: u32 = metal_api_core::provider::MAX_PRESENT_IMAGE_COUNT;

/// The largest instance count the instancing increment executes
/// (`research/docs/23` §3.3, v31). Same rule as the Vulkan rail's ceiling: the
/// reviewed fixture draws two instances and the declared window is four.
pub(crate) const MAX_RENDER_INSTANCES: u32 = 4;

/// The render bits this provider declares as of the Step 7 flip.
///
/// Flip evidence (`research/docs/23` §4.2, §6 Steps 6-7;
/// `conformance/RENDER-CAPTURE.md` §5): CI run `34774478149` — job
/// `native-oracle-build` at commit `fb4f8da` — ran `native-oracle
/// --render-selftest` on an Apple Paravirtual device, whose report and log read
/// `4080c0ff` four times and ended with `render_selftest: PASS`. A green run
/// whose log said `SKIP` would not be that evidence, because it reports a runner
/// without an eligible device rather than an executed reviewed path.
pub(crate) fn capability_bits() -> RenderCapabilityBits {
    let present = present_capability_bits();
    RenderCapabilityBits {
        supports_render_passes: true,
        max_color_attachments: MAX_COLOR_ATTACHMENTS,
        max_attachment_dimension: MAX_ATTACHMENT_DIMENSION,
        supported_color_formats: SUPPORTED_COLOR_FORMATS.to_vec(),
        // The render-sampler bits are ship-fail-closed (`research/docs/23`
        // §3.3, v70): the contract and the wire carry them, and the encoder
        // turns them on in the increment that executes the shape. Until then
        // a texture-bearing pass is refused during admission with
        // `render_texture_input_unsupported` instead of being executed with a
        // cleared sampling result the trace did not ask for.
        supports_render_texture_sampling: false,
        max_render_textures: 0,
        supported_render_texture_formats: Vec::new(),
        supports_presentation: present.supports_presentation,
        max_present_targets: present.max_present_targets,
        supported_present_modes: present.supported_present_modes,
        max_present_image_count: present.max_present_image_count,
    }
}

/// The present bits this provider declares as of the present-track flip.
///
/// Flip evidence (`research/docs/24` §6 Step 7): CI run `34781060564`, job
/// `native-oracle-build`, step "Run native present self-test when a Metal
/// device is eligible". The probe reported an eligible Apple Paravirtual
/// device (`supports_apple4: true`), the oracle then ran the reviewed 2x2
/// present equivalent (`load: load`, `fefefefe`-sentinel preset) and printed
/// `present_selftest: PASS (4080c0ff4080c0ff4080c0ff4080c0ff)` — four
/// expected texels, never the sentinel. A green job whose log said `SKIP` is
/// not that evidence: it reports a runner without an eligible device, not an
/// executed reviewed present path. Before the flip these bits were all at
/// their defaults, so core admission refused a present-bearing trace with
/// `present_targets_unsupported` instead of running the offscreen render and
/// silently dropping the present (`research/docs/24` §4.2); the test below
/// keeps that refusal path pinned on a constructed pre-flip snapshot.
pub(crate) fn present_capability_bits() -> PresentCapabilityBits {
    PresentCapabilityBits {
        supports_presentation: true,
        max_present_targets: MAX_PRESENT_TARGETS,
        supported_present_modes: PresentMode::ADMITTED.to_vec(),
        max_present_image_count: MAX_PRESENT_IMAGE_COUNT,
    }
}

/// The vertex-input bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side unit tests cannot drift.
///
/// Split out from the render bits for the same reason the present bits are: the
/// three values are the *rail's own* limits (as many streams as the contract
/// caps, and exactly the formats the rail translates into
/// `MTLVertexFormat` / `MTLIndexType`), so capability admission and this rail
/// agree by construction instead of by a second list that could drift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VertexInputCapabilityBits {
    pub(crate) max_vertex_buffers: u32,
    pub(crate) supported_vertex_formats: Vec<VertexFormat>,
    pub(crate) supported_index_formats: Vec<IndexFormat>,
}

/// The vertex-input bits this provider declares as of the vertex-input flip.
///
/// Flip condition (`research/docs/23` §3.3, §6 Step 3.3;
/// `conformance/RENDER-CAPTURE.md` §8): the Swift oracle's `--vertex-selftest`
/// on an Apple GPU, i.e. the same shape as the render bits' `--render-selftest`
/// evidence. That self-test builds the indexed reviewed module, binds the
/// fixture's `float32x2` stream and `uint16` index buffer through an
/// `MTLVertexDescriptor`, draws with `drawIndexedPrimitives` and reads the 2x2
/// attachment back as `4080c0ff` four times, printing `vertex_selftest: PASS
/// (4080c0ff)`. A green job whose log said `SKIP` is not that evidence: it
/// reports a runner without an eligible device rather than an executed reviewed
/// path.
///
/// Before the flip these three bits were all at their defaults, so core
/// admission refused a vertex-bearing trace with `vertex_buffer_limit` /
/// `index_format_unsupported` instead of executing it with positions the trace
/// did not ask for; the test below keeps that refusal path pinned on a
/// constructed pre-flip snapshot. The host-side half this rail owns is
/// [`plan_vertex_input`]: the stream bytes, the stride and index footprints and
/// the index values are all proved before a device object exists, and the
/// remaining Apple-only question is whether Metal executes exactly that plan.
pub(crate) fn vertex_input_capability_bits() -> VertexInputCapabilityBits {
    VertexInputCapabilityBits {
        max_vertex_buffers: MAX_VERTEX_BUFFERS,
        supported_vertex_formats: VertexFormat::ADMITTED.to_vec(),
        supported_index_formats: IndexFormat::ADMITTED.to_vec(),
    }
}

/// The instancing bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v31).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstancingCapabilityBits {
    pub(crate) supports_render_instancing: bool,
    pub(crate) max_render_instances: u32,
}

/// The instancing bits this provider declares as of the v31 flip.
///
/// Flip condition (`research/docs/23` §3.3): the reviewed `instanced_pair_4x4`
/// case on the Apple rail, i.e. the macOS CI job's `native-metal` capture of
/// the suite that names every rail. That run builds the reviewed MSL module,
/// declares a `per_instance` stream through `MTLVertexDescriptor` (`
/// stepFunction = .perInstance`, `stepRate = 1`), draws `instanceCount: 2` and
/// reads back the red left half and the green right half the fixture pins. The
/// ceiling follows the Vulkan rail's: the reviewed two instances rounded up to
/// four, so a wider draw stays refused by core admission rather than silently
/// narrowed.
///
/// Before this flip both bits were at their defaults, so core admission refused
/// a multi-instance pass (or a per-instance layout) with
/// `render_instancing_unsupported` / `vertex_step_unsupported` instead of
/// executing it once; the host-side half this rail owns is
/// [`plan_vertex_input`], which proves the per-instance footprint before a
/// device object exists.
pub(crate) fn instancing_capability_bits() -> InstancingCapabilityBits {
    InstancingCapabilityBits {
        supports_render_instancing: true,
        max_render_instances: MAX_RENDER_INSTANCES,
    }
}

/// The multisample bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v51).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MultisampleCapabilityBits {
    pub(crate) supports_render_multisample: bool,
    pub(crate) max_render_sample_count: u32,
    /// The reviewed 2/4/8 counts the device admits, as a bitmask over the
    /// contract codes: bit `i` = [`SampleCount`] code `i`. The capture runner
    /// reads the device-gated sample-count cases against this mask, because
    /// the ceiling alone cannot say which of the counts a device lacks
    /// (`research/docs/23` §3.3, v61).
    pub(crate) render_sample_counts: u32,
}

/// The multisample bits this provider declares, from the device's own answer
/// (`research/docs/23` §3.3, v51/v61).
///
/// The v61 increment widens the reviewed raster family to 2x/4x/8x, so the
/// ceiling is no longer a fixed reviewed count but the largest of the three
/// rasters the device's `supportsTextureSampleCount:` probe admits. A device
/// that admits none of them keeps both bits at their defaults, so core
/// admission refuses a multisampled pass with
/// `render_multisample_unsupported` instead of executing it as a single-sample
/// draw; the host-side half this rail owns is [`plan`], which holds the state
/// to the counts the encoder builds and the texture creation in
/// [`multisample_attachment_textures`].
#[cfg(target_os = "macos")]
pub(crate) fn device_multisample_capability_bits(device: &Device) -> MultisampleCapabilityBits {
    let counts = [
        (8u32, 1 << SampleCount::Eight.code()),
        (4, 1 << SampleCount::Four.code()),
        (2, 1 << SampleCount::Two.code()),
    ];
    let mask = counts
        .iter()
        .filter(|(count, _)| device.supports_texture_sample_count(u64::from(*count)))
        .fold(0, |mask, (_, bit)| mask | bit);
    let ceiling = counts
        .into_iter()
        .find(|(count, _)| device.supports_texture_sample_count(u64::from(*count)))
        .map(|(count, _)| count);
    MultisampleCapabilityBits {
        supports_render_multisample: ceiling.is_some(),
        max_render_sample_count: ceiling.unwrap_or(0),
        render_sample_counts: mask,
    }
}

/// The depth-resolve bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v57c).
///
/// The mode mask is the contract's per-filter bit layout: bit `i` =
/// [`DepthResolveFilter`] code `i` (`metal-api-core`:
/// `ProviderCapabilities::depth_resolve_modes`). The v57e
/// `--depth-resolve-selftest` run measured an Apple Paravirtual device
/// executing all three filters — the mixed column's `min=0000003f` and
/// `max=6666663f` against `sample0=0000003f` (`f4d70e4`, CI run
/// `35112569688`) — so the mask declares Sample0|Min|Max (`0b111`) instead of
/// the fail-closed Sample0-only value the pre-evidence increments used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DepthResolveCapabilityBits {
    pub(crate) supports_render_depth_resolve: bool,
    pub(crate) depth_resolve_modes: u32,
}

/// The Sample0 mode bit, spelled once so the probe and the snapshot cannot
/// disagree about which filter the rail admits.
pub(crate) const DEPTH_RESOLVE_SAMPLE0_BIT: u32 = 1u32 << DepthResolveFilter::Sample0.code();

/// The Min mode bit, declared on the same v57e self-test evidence as Sample0
/// and Max: the mixed column's reduction really lands the nearest depth.
pub(crate) const DEPTH_RESOLVE_MIN_BIT: u32 = 1u32 << DepthResolveFilter::Min.code();

/// The Max mode bit, declared on the same v57e self-test evidence as Sample0
/// and Min: the mixed column's reduction really lands the furthest depth.
pub(crate) const DEPTH_RESOLVE_MAX_BIT: u32 = 1u32 << DepthResolveFilter::Max.code();

/// The depth-resolve bits derived from the device itself, queried exactly once
/// when the provider is created (`native.rs`).
///
/// `metal` 0.33 models depth32-float resolve as a macOS platform fact:
/// `depth32_float_capabilities` reports `Resolve` for every macOS feature set
/// (its `device.rs`, and the SDK has no finer per-device selector), while the
/// crate's `supports_msaa_depth_resolve` is iOS-gated and says false on macOS.
/// The feature-set API is deprecated in the SDK, so the probe asks the modern
/// `supportsFamily:` question for the Apple family the provider's admission
/// already requires — a real device query rather than a compile-time constant,
/// combined with the platform fact the crate models. The v57e
/// `--depth-resolve-selftest` CI output landed the evidence the mask was
/// waiting for: the Apple Paravirtual device executed all three filters and
/// printed `depth_resolve_selftest: PASS` (`f4d70e4`, CI run `35112569688`),
/// so an eligible device now declares Sample0|Min|Max (`0b111`) and an
/// ineligible one still declares `0` (`research/docs/23` §3.3, v57c/v57e).
#[cfg(target_os = "macos")]
pub(crate) fn device_depth_resolve_capability_bits(device: &Device) -> DepthResolveCapabilityBits {
    let supports = device.supports_family(metal::MTLGPUFamily::Apple4);
    DepthResolveCapabilityBits {
        supports_render_depth_resolve: supports,
        depth_resolve_modes: if supports {
            DEPTH_RESOLVE_SAMPLE0_BIT | DEPTH_RESOLVE_MIN_BIT | DEPTH_RESOLVE_MAX_BIT
        } else {
            0
        },
    }
}

/// The stencil-resolve bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v60).
///
/// The mode mask is the contract's per-filter bit layout: bit `i` =
/// [`StencilResolveFilter`] code `i`. The v59 `--stencil-resolve-selftest`
/// run measured an Apple Paravirtual device executing both filters — the
/// mixed column's `depth_resolved_sample(min)=01` and
/// `depth_resolved_sample(max)=00` against `sample0=01` (`2b877b8`, CI run
/// `35120171655`) — so the mask declares Sample0|DepthResolvedSample (`0b11`)
/// and an ineligible device still declares `0` (`research/docs/23` §3.3, v60).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StencilResolveCapabilityBits {
    pub(crate) supports_render_stencil_resolve: bool,
    pub(crate) stencil_resolve_modes: u32,
}

/// The Sample0 mode bit, spelled once so the probe and the snapshot cannot
/// disagree about which filter the rail admits.
pub(crate) const STENCIL_RESOLVE_SAMPLE0_BIT: u32 = 1u32 << StencilResolveFilter::Sample0.code();

/// The DepthResolvedSample mode bit, declared on the same v59 self-test
/// evidence as Sample0: the mixed column's reduction really follows the depth
/// resolve's selected sample.
pub(crate) const STENCIL_RESOLVE_DRS_BIT: u32 =
    1u32 << StencilResolveFilter::DepthResolvedSample.code();

/// The stencil-resolve bits derived from the device itself, queried exactly
/// once when the provider is created (`native.rs`).
///
/// The probe asks the same Apple-family question the depth resolve asks; the
/// v59 `--stencil-resolve-selftest` CI output landed the evidence the mask was
/// waiting for: the Apple Paravirtual device executed both filters and printed
/// `stencil_resolve_selftest: PASS` (`2b877b8`, CI run `35120171655`), so an
/// eligible device now declares Sample0|DepthResolvedSample (`0b11`) and an
/// ineligible one still declares `0` (`research/docs/23` §3.3, v60).
#[cfg(target_os = "macos")]
pub(crate) fn device_stencil_resolve_capability_bits(
    device: &Device,
) -> StencilResolveCapabilityBits {
    let supports = device.supports_family(metal::MTLGPUFamily::Apple4);
    StencilResolveCapabilityBits {
        supports_render_stencil_resolve: supports,
        stencil_resolve_modes: if supports {
            STENCIL_RESOLVE_SAMPLE0_BIT | STENCIL_RESOLVE_DRS_BIT
        } else {
            0
        },
    }
}

/// The texel the reviewed fragment writes, as the UNORM8 bytes an admitted
/// attachment stores. Both this rail and the Swift oracle's render self-test
/// have to recognise it, and it is what makes "the draw ran and covered the
/// whole attachment" falsifiable: a 2x2 attachment whose four texels are all
/// `40 80 c0 ff` cannot be the sentinel a `LoadOp::Clear` leaves behind.
pub(crate) const EXPECTED_TEXEL_BYTES: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The pixel format this rail maps an admitted attachment format onto.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderPixelFormat {
    Rgba8Unorm,
    Bgra8Unorm,
    R32Float,
}

impl RenderPixelFormat {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Rgba8Unorm => "rgba8_unorm",
            Self::Bgra8Unorm => "bgra8_unorm",
            Self::R32Float => "r32_float",
        }
    }
}

/// The clear value or previous contents an encoder has to start from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RenderLoadAction {
    /// Clear every texel, with the components decoded from the contract bytes.
    Clear([f64; 4]),
    /// Keep the attachment's previous contents.
    Load,
    /// Leave the attachment's previous contents undefined: the encoder sets
    /// `MTLLoadAction::DontCare` and presets nothing, so the pass neither
    /// reads nor overwrites the pre-pass bytes before drawing
    /// (`research/docs/23` §3.1, v20).
    DontCare,
}

/// The store action this rail sets. `Store` lands the attachment's texels on
/// the observable surface through the readback; `DontCare` is the contract's
/// `StoreOp::DontCare` (`research/docs/23` §3.6, v19): the attachment still
/// renders, but its bytes disappear from the observable surface — the encoder
/// sets `MTLStoreAction::DontCare` and reads nothing back, so a discarded
/// attachment cannot pass as "landed correctly". Core admission still refuses
/// a pass whose every attachment discards, so one `Store` action always keeps
/// the pass observable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderStoreAction {
    Store,
    /// Discard the attachment's writes; no readback and no writeback leave this
    /// attachment's location.
    DontCare,
}

/// Map an admitted attachment format onto the pixel format this rail builds.
pub(crate) fn pixel_format(format: AttachmentFormat) -> Result<RenderPixelFormat, ProviderError> {
    match format {
        AttachmentFormat::Rgba8Unorm => Ok(RenderPixelFormat::Rgba8Unorm),
        AttachmentFormat::Bgra8Unorm => Ok(RenderPixelFormat::Bgra8Unorm),
        AttachmentFormat::R32Float => Ok(RenderPixelFormat::R32Float),
        // Expressible in the contract for symmetry with the sampled-texture
        // rail, refused by the first render increment: its texels are integers
        // while `LoadOp::Clear` and the fragment output carry colour bytes
        // (`metal_api_core::provider::AttachmentFormat::R32Uint`). The refusal
        // reuses core admission's slug and class rather than inventing a second
        // spelling for one fact.
        AttachmentFormat::R32Uint => Err(capability_refusal("attachment_format_unsupported")
            .with_field("format", FieldValue::Unsigned(u64::from(format.code())))),
    }
}

/// Decode the contract's clear bytes into `MTLClearColor` components.
///
/// The contract carries bytes in memory order, so the mapping is per format: the
/// two 8-bit UNORM formats differ in channel order, and the single-channel float
/// format's four bytes are the red component itself. Writing the value into the
/// attachment is the driver's job; this only fixes which component each byte
/// means (`research/docs/23` §3.5).
pub(crate) fn clear_components(clear: ClearColor, format: RenderPixelFormat) -> [f64; 4] {
    let bytes = clear.bytes;
    match format {
        // Memory order is R, G, B, A.
        RenderPixelFormat::Rgba8Unorm => [
            unorm8(bytes[0]),
            unorm8(bytes[1]),
            unorm8(bytes[2]),
            unorm8(bytes[3]),
        ],
        // Memory order is B, G, R, A, so red and blue swap on the way into the
        // component order `MTLClearColor` carries: one colour has two byte
        // strings, one per format.
        RenderPixelFormat::Bgra8Unorm => [
            unorm8(bytes[2]),
            unorm8(bytes[1]),
            unorm8(bytes[0]),
            unorm8(bytes[3]),
        ],
        // One channel: the four bytes are the little-endian IEEE-754 bits of the
        // red component. Green and blue stay zero and alpha is unused, which is
        // what a single-channel attachment stores.
        RenderPixelFormat::R32Float => [f64::from(f32::from_le_bytes(bytes)), 0.0, 0.0, 1.0],
    }
}

fn unorm8(byte: u8) -> f64 {
    f64::from(byte) / 255.0
}

/// The load action an encoder has to set for this pass.
pub(crate) fn load_action(
    load: LoadOp,
    format: RenderPixelFormat,
) -> Result<RenderLoadAction, ProviderError> {
    match load {
        LoadOp::Clear(clear) => Ok(RenderLoadAction::Clear(clear_components(clear, format))),
        LoadOp::Load => Ok(RenderLoadAction::Load),
        LoadOp::DontCare => Ok(RenderLoadAction::DontCare),
    }
}

/// The store action an encoder has to set for this pass.
///
/// Both contract variants are admitted now (`research/docs/23` §3.6, v19):
/// core admission refuses the all-discarded pass before a plan is built, so
/// any pass that reaches this function still stores at least one attachment.
/// The retained `Result` spells that this is the mapping the encoder depends
/// on, not a second policy.
pub(crate) fn store_action(store: StoreOp) -> Result<RenderStoreAction, ProviderError> {
    Ok(match store {
        StoreOp::Store => RenderStoreAction::Store,
        StoreOp::DontCare => RenderStoreAction::DontCare,
    })
}

/// The vertex format this rail hands its `MTLVertexAttributeDescriptor`.
///
/// A value of its own rather than `MTLVertexFormat` directly, so the descriptor
/// translation is host-testable: `MTLVertexFormat` only exists on macOS, while
/// the binding index, the stride and the attribute offsets are decided here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderVertexFormat {
    Float2,
    Float3,
    Float4,
    Uint,
}

impl RenderVertexFormat {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Float2 => "float32x2",
            Self::Float3 => "float32x3",
            Self::Float4 => "float32x4",
            Self::Uint => "uint32",
        }
    }
}

/// Map an admitted vertex format onto the format this rail builds.
///
/// Total: `VertexFormat`'s four values are exactly the four
/// `MTLVertexFormat`s the reviewed rails translate, so a fifth wire code
/// arriving without a mapping fails to compile here rather than silently
/// becoming a different descriptor.
pub(crate) const fn vertex_format(format: VertexFormat) -> RenderVertexFormat {
    match format {
        VertexFormat::Float32x2 => RenderVertexFormat::Float2,
        VertexFormat::Float32x3 => RenderVertexFormat::Float3,
        VertexFormat::Float32x4 => RenderVertexFormat::Float4,
        VertexFormat::Uint32 => RenderVertexFormat::Uint,
    }
}

/// The index width this rail hands `drawIndexedPrimitives(indexType:)`.
///
/// The sibling of [`RenderVertexFormat`]: a host-visible value so the
/// `IndexFormat` → `MTLIndexType` mapping is testable without a device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderIndexType {
    Uint16,
    Uint32,
}

impl RenderIndexType {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
        }
    }

    /// Bytes one index of this width occupies, restated from the contract value
    /// the mapping came from so the footprint proof and the width cannot drift.
    pub(crate) const fn bytes(self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Map an admitted index format onto the width this rail draws with. Total for
/// the same reason [`vertex_format`] is.
pub(crate) const fn index_type(format: IndexFormat) -> RenderIndexType {
    match format {
        IndexFormat::Uint16 => RenderIndexType::Uint16,
        IndexFormat::Uint32 => RenderIndexType::Uint32,
    }
}

/// The step function this rail hands its `MTLVertexBufferLayoutDescriptor`
/// (`research/docs/23` §3.3, v31).
///
/// The sibling of [`RenderVertexFormat`]: a host-visible value so the
/// `VertexStep` → `MTLVertexStepFunction` mapping (and the footprint proof that
/// reads it) is testable without a device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderVertexStep {
    PerVertex,
    PerInstance,
}

impl RenderVertexStep {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::PerVertex => "per_vertex",
            Self::PerInstance => "per_instance",
        }
    }
}

/// Map an admitted step onto the function this rail builds. Total for the same
/// reason [`vertex_format`] is: the contract's closed pair is exactly the two
/// `MTLVertexStepFunction`s the rails execute.
pub(crate) const fn vertex_step(step: VertexStep) -> RenderVertexStep {
    match step {
        VertexStep::PerVertex => RenderVertexStep::PerVertex,
        VertexStep::PerInstance => RenderVertexStep::PerInstance,
    }
}

/// One vertex attribute as the descriptor builder needs it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedVertexAttribute {
    /// Shader-visible attribute location (`[[attribute(n)]]`).
    pub(crate) location: u32,
    /// Byte offset inside one vertex.
    pub(crate) offset: u64,
    /// The translated format.
    pub(crate) format: RenderVertexFormat,
}

/// One vertex stream the pass binds, resolved before any Metal object exists.
///
/// The bytes are the view's own (`BufferSource::OwnedBytes`), i.e. exactly the
/// `view.length` bytes the trace declared at the view, and `offset` is where
/// that view starts inside its allocation. The macOS encoder builds one
/// MTLBuffer with the bytes at that offset and binds it at the same offset, the
/// convention the compute pool's merged images follow (`native.rs`): a stream
/// reads the byte range the trace named instead of being silently re-based at
/// zero.
#[derive(Debug)]
pub(crate) struct PlannedVertexStream<'a> {
    /// Binding index: entry `i` of the pipeline layout is binding `i`, which is
    /// also its `MTLVertexBufferLayoutDescriptor` index and the
    /// `setVertexBuffer(_:offset:index:)` index.
    pub(crate) buffer_index: u32,
    /// Bytes between consecutive vertices.
    pub(crate) stride: u64,
    /// How often the stream advances (`research/docs/23` §3.3, v31): once per
    /// vertex or once per instance. The footprint proof below reads it to
    /// decide which count the stream has to cover.
    pub(crate) step: RenderVertexStep,
    /// Attributes this stream is read through.
    pub(crate) attributes: Vec<PlannedVertexAttribute>,
    /// The view's own bytes.
    pub(crate) bytes: &'a [u8],
    /// The view's offset inside its allocation.
    pub(crate) offset: u64,
}

/// The index buffer the pass draws through, resolved the same way.
#[derive(Debug)]
pub(crate) struct PlannedIndexStream<'a> {
    /// The view the indices come from.
    pub(crate) view_id: ViewId,
    /// The view's own bytes.
    pub(crate) bytes: &'a [u8],
    /// The view's offset inside its allocation.
    pub(crate) offset: u64,
    /// The translated index width.
    pub(crate) format: RenderIndexType,
    /// Indices the draw consumes, i.e. `RenderPassDescriptor::vertices` in the
    /// indexed shape.
    pub(crate) index_count: u32,
    /// The highest index the draw reads plus one: the number of vertices the
    /// bound streams have to cover for this draw.
    pub(crate) vertex_span: u64,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34), i.e. Metal's `baseVertex`. The stream has to cover
    /// `base_vertex + vertex_span` records, and the draw call carries the
    /// offset whenever it is non-zero.
    pub(crate) base_vertex: u64,
}

/// Resolve a pass's vertex streams and index buffer from the pass itself.
///
/// A render input declares its own bytes (`research/docs/23` §3.6): entry `i` of
/// [`RenderPassDescriptor::vertex_buffers`] is a read-only [`BufferView`] whose
/// `source` carries the vertices, and an index binding carries its view beside
/// the width. Neither has to be declared by a compute pass, so this rail uploads
/// what the pass hands it instead of resolving a name against a pool.
///
/// Two rules are checked here because a driver answers both with undefined
/// behaviour instead of an error:
///
/// * the stream's bytes are ones this rail holds (`BufferSource::OwnedBytes`).
///   The compute rail resolves leases; this one has no lease path, so a leased
///   stream is refused instead of uploaded from bytes the rail does not have;
/// * the declared range covers every vertex and index the draw reads (the
///   footprint proof `research/docs/23` §3.3 asks for: Metal would read past the
///   buffer, or index a stream out of range, without refusing). Which count the
///   proof is against depends on the draw: a non-indexed draw reads its vertex
///   count in order, while an indexed draw reads the vertices its index values
///   select, so the refusal names the index rather than the stream in that case.
///
/// What is *not* checked here is the binding label: the entry's position is the
/// binding index both rails use, and core admission already holds each view's
/// own `metal_binding` to it ([`validate_vertex_buffer_binding`]), which `plan`
/// re-runs before this function.
fn plan_vertex_input<'a>(
    pass: &'a RenderPassDescriptor,
    pipeline: &RenderPipelineContract,
) -> Result<(Vec<PlannedVertexStream<'a>>, Option<PlannedIndexStream<'a>>), ProviderError> {
    let mut streams = Vec::with_capacity(pass.vertex_buffers.len());
    // One layout entry per bound stream, in binding order; core admission
    // refuses a pass and a layout that disagree about the count
    // (`ContractError::VertexLayoutBindingMismatch`), which `plan` re-runs
    // before this function, so the zip covers every bound stream.
    for (buffer_index, (view, layout)) in pass
        .vertex_buffers
        .iter()
        .zip(pipeline.vertex_layout.buffers())
        .enumerate()
    {
        let bytes = stream_bytes(view, VERTEX_SLUG)?;
        streams.push(PlannedVertexStream {
            buffer_index: u32::try_from(buffer_index)
                .map_err(|_| capability_refusal("vertex_buffer_limit"))?,
            stride: layout.stride,
            step: vertex_step(layout.step),
            attributes: layout
                .attributes
                .iter()
                .map(|attribute| PlannedVertexAttribute {
                    location: attribute.location,
                    offset: attribute.offset,
                    format: vertex_format(attribute.format),
                })
                .collect(),
            bytes,
            offset: view.offset,
        });
    }
    // A per-instance stream's record count is the draw's instance count, and a
    // per-vertex stream's is the vertex span the draw reads; the two proofs
    // below therefore ask each stream for the right count
    // (`research/docs/23` §3.3, v31). The instance count comes from the pass,
    // which `plan` has already validated as at least one.
    let instance_count = u64::from(pass.instance_count);
    for (buffer_index, stream) in streams.iter().enumerate() {
        if stream.step != RenderVertexStep::PerInstance {
            continue;
        }
        // Saturating for the same reason the per-vertex proof is: the product
        // only has to decide whether the stream covers the count.
        let required = instance_count.saturating_mul(stream.stride);
        if u64::try_from(stream.bytes.len()).unwrap_or(u64::MAX) < required {
            return Err(vertex_footprint_refusal(
                buffer_index,
                stream.stride,
                required,
                stream.bytes.len(),
            )
            .with_field("step", FieldValue::Text(stream.step.name().to_owned()))
            .with_detail("a per-instance stream has to cover one record per instance"));
        }
    }
    let indices = match &pass.indices {
        None => None,
        Some(binding) => Some(plan_index_stream(
            binding,
            pass.vertices,
            u64::from(pass.base_vertex),
        )?),
    };
    match &indices {
        // An indexed draw reads the vertices its index values select, so each
        // stream has to cover that span. The refusal names the index that
        // reached past the stream, because that is what the trace has to change.
        Some(indices) => {
            // The offset takes part in the proof: the vertex a draw reads is
            // `base_vertex + index`, so a span that fits on its own can still
            // reach past the stream once the offset is added
            // (`research/docs/23` §3.3, v34).
            let required_span = indices.vertex_span.saturating_add(indices.base_vertex);
            for (buffer_index, stream) in streams.iter().enumerate() {
                // A per-instance stream is proved against the instance count
                // above, so the vertex span does not apply to it
                // (`research/docs/23` §3.3, v31).
                if stream.step == RenderVertexStep::PerInstance {
                    continue;
                }
                // `stride == 0` cannot reach here (`plan` re-runs the layout
                // validator), so `checked_div` is only the safe spelling of the
                // quotient: a zero stride would be refused upstairs rather than
                // read as an unbounded stream.
                let covered = u64::try_from(stream.bytes.len())
                    .unwrap_or(u64::MAX)
                    .checked_div(stream.stride)
                    .unwrap_or(0);
                if required_span > covered {
                    return Err(index_value_refusal(highest_index(required_span), covered)
                        .with_field(
                            "buffer_index",
                            FieldValue::Unsigned(u64::try_from(buffer_index).unwrap_or(u64::MAX)),
                        )
                        .with_field("base_vertex", FieldValue::Unsigned(indices.base_vertex))
                        .with_field("view", FieldValue::Unsigned(indices.view_id.get())));
                }
            }
            // The `vertex_id` shape binds no stream to bound its index values:
            // the reviewed module generates exactly
            // `FULL_SCREEN_TRIANGLE_VERTICES` positions, so an index at or above
            // that count would read a position the module does not carry.
            if pass.vertex_buffers.is_empty()
                && required_span > u64::from(FULL_SCREEN_TRIANGLE_VERTICES)
            {
                return Err(index_value_refusal(
                    highest_index(required_span),
                    u64::from(FULL_SCREEN_TRIANGLE_VERTICES),
                )
                .with_field("base_vertex", FieldValue::Unsigned(indices.base_vertex))
                .with_field("view", FieldValue::Unsigned(indices.view_id.get())));
            }
        }
        // A non-indexed draw reads vertices `0..vertices` in order, so every
        // stream has to cover the pass's own count.
        None => {
            for (buffer_index, stream) in streams.iter().enumerate() {
                // The per-vertex arm of the same split: a per-instance stream
                // covers one record per instance instead of one per vertex.
                if stream.step == RenderVertexStep::PerInstance {
                    continue;
                }
                // Saturating, because the proof only has to decide whether the
                // stream covers the count: an unrepresentable product is by
                // definition larger than any buffer this provider admits.
                let required = u64::from(pass.vertices).saturating_mul(stream.stride);
                if u64::try_from(stream.bytes.len()).unwrap_or(u64::MAX) < required {
                    return Err(vertex_footprint_refusal(
                        buffer_index,
                        stream.stride,
                        required,
                        stream.bytes.len(),
                    ));
                }
            }
        }
    }
    Ok((streams, indices))
}

/// The highest index value a span of `vertex_span` vertices reads.
fn highest_index(vertex_span: u64) -> u32 {
    u32::try_from(vertex_span.saturating_sub(1)).unwrap_or(u32::MAX)
}

/// The index buffer of one pass, with its footprint and its index values proved.
///
/// The bytes are the binding's own view, exactly as a vertex stream's are
/// ([`plan_vertex_input`]): an index buffer declares its source instead of
/// naming a view a compute pass happens to carry.
fn plan_index_stream<'a>(
    binding: &'a IndexBufferBinding,
    index_count: u32,
    base_vertex: u64,
) -> Result<PlannedIndexStream<'a>, ProviderError> {
    let view = &binding.view;
    let bytes = stream_bytes(view, INDEX_SLUG)?;
    let format = index_type(binding.format);
    let width = usize::try_from(format.bytes()).unwrap_or(usize::MAX);
    let needed = usize::try_from(index_count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| index_footprint_refusal(width, usize::MAX, bytes.len()))?;
    if bytes.len() < needed {
        return Err(index_footprint_refusal(width, needed, bytes.len()));
    }
    let highest = match format {
        RenderIndexType::Uint16 => bytes[..needed]
            .chunks_exact(2)
            .map(|chunk| u32::from(u16::from_le_bytes([chunk[0], chunk[1]])))
            .max(),
        RenderIndexType::Uint32 => bytes[..needed]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .max(),
    };
    // `u64`, so an index of `u32::MAX` is a span rather than an overflow: the
    // footprint proof below refuses it like any other index that reaches past
    // the streams.
    let vertex_span = match highest {
        None => 0,
        Some(value) => u64::from(value) + 1,
    };
    Ok(PlannedIndexStream {
        base_vertex,
        view_id: view.view_id,
        bytes,
        offset: view.offset,
        format,
        index_count,
        vertex_span,
    })
}

/// The bytes of a stream view, or the refusal that names the storage this rail
/// cannot read.
fn stream_bytes<'a>(view: &'a BufferView, slug: &'static str) -> Result<&'a [u8], ProviderError> {
    match &view.source {
        BufferSource::OwnedBytes(bytes) => Ok(bytes),
        other => Err(capability_refusal(slug)
            .with_field("view", FieldValue::Unsigned(view.view_id.get()))
            .with_field(
                "storage_mode",
                FieldValue::Text(storage_mode_name(other).to_owned()),
            )
            .with_detail(
                "the render rail binds the bytes a view declares as `OwnedBytes`; a leased \
                 stream has no path through this rail",
            )),
    }
}

/// The storage modes a stream view can carry, as the refusal spells them.
fn storage_mode_name(source: &BufferSource) -> &'static str {
    match source {
        BufferSource::OwnedBytes(_) => "owned_bytes",
        BufferSource::StagedLease(_) => "staged_lease",
        BufferSource::BorrowedNoCopy(_) => "borrowed_no_copy",
    }
}

fn vertex_footprint_refusal(
    buffer_index: usize,
    stride: u64,
    covered: u64,
    available: usize,
) -> ProviderError {
    capability_refusal("render_vertex_footprint_unsupported")
        .with_field(
            "buffer_index",
            FieldValue::Unsigned(u64::try_from(buffer_index).unwrap_or(u64::MAX)),
        )
        .with_field("stride", FieldValue::Unsigned(stride))
        .with_field("required_bytes", FieldValue::Unsigned(covered))
        .with_field(
            "available_bytes",
            FieldValue::Unsigned(u64::try_from(available).unwrap_or(u64::MAX)),
        )
        .with_detail(
            "the declared vertex view does not cover every vertex the draw reads; a bound \
             stream would be read past its end",
        )
}

fn index_footprint_refusal(width: usize, required: usize, available: usize) -> ProviderError {
    capability_refusal("render_index_footprint_unsupported")
        .with_field(
            "index_bytes",
            FieldValue::Unsigned(u64::try_from(width).unwrap_or(u64::MAX)),
        )
        .with_field(
            "required_bytes",
            FieldValue::Unsigned(u64::try_from(required).unwrap_or(u64::MAX)),
        )
        .with_field(
            "available_bytes",
            FieldValue::Unsigned(u64::try_from(available).unwrap_or(u64::MAX)),
        )
        .with_detail("the declared index view does not cover the indices the draw consumes")
}

fn index_value_refusal(highest: u32, vertices_covered: u64) -> ProviderError {
    capability_refusal("render_index_value_out_of_range")
        .with_field("highest_index", FieldValue::Unsigned(u64::from(highest)))
        .with_field("vertices_covered", FieldValue::Unsigned(vertices_covered))
        .with_detail(
            "an index selects a vertex the bound streams do not cover; the driver would \
             read outside the stream instead of refusing",
        )
}

/// Slug of a vertex stream this rail cannot read.
const VERTEX_SLUG: &str = "render_vertex_buffer_unsupported";

/// Slug of an index stream this rail cannot read.
const INDEX_SLUG: &str = "render_index_buffer_unsupported";

/// One offscreen render pass to execute.
///
/// The pass and the pipeline are the core values themselves, so this rail cannot
/// drift from the contract's field set: a field the contract adds is a field
/// admission already validates.
pub(crate) struct OffscreenRenderRequest<'a> {
    /// The trace's render pass entry.
    pub(crate) pass: &'a RenderPassDescriptor,
    /// The registered render pipeline the pass names.
    pub(crate) pipeline: &'a RenderPipelineContract,
    /// The MSL module to compile. The pipeline's [`VertexLayout`] and
    /// `color_formats` select which reviewed module is the only one accepted
    /// here — `vertex_id` positions ([`REVIEWED_SOURCE`]), single-output
    /// `[[stage_in]]` positions ([`REVIEWED_VERTEX_SOURCE`]) or the dual
    /// `[[stage_in]]` module ([`REVIEWED_DUAL_SOURCE`]) — so a caller cannot
    /// pair one shape's descriptor with another shape's module.
    pub(crate) source: &'a str,
    /// One entry per colour attachment, in location order: the tightly packed
    /// texels that attachment already holds, for [`LoadOp::Load`]. Required
    /// exactly then, refused for a clear.
    pub(crate) initial: Vec<Option<&'a [u8]>>,
}

/// Everything the encoder needs, decided before the first Metal object exists.
#[derive(Debug)]
pub(crate) struct RenderPlan<'a> {
    /// The reviewed module this plan compiles. The entry pair below is read out
    /// of the contract, which [`review_contract`] has already compared with the
    /// module, so the two cannot disagree about what runs.
    pub(crate) source: &'a str,
    /// The module's path, for diagnostics.
    pub(crate) module_path: &'static str,
    pub(crate) vertex_entry: &'a str,
    pub(crate) fragment_entry: &'a str,
    /// One entry per colour attachment, in location order: the pixel format,
    /// the load/store actions and the previous bytes the encoder writes into
    /// each attachment before the pass opens.
    pub(crate) attachments: Vec<PlannedAttachment<'a>>,
    /// Attachment extent in texels, as `[width, height]`.
    pub(crate) extent: [u32; 2],
    /// `[origin_x, origin_y, width, height]`, copied from the validated pass.
    pub(crate) viewport: [u32; 4],
    /// The pass's scissor rectangle, or `None` for the whole viewport
    /// (`research/docs/23` §3.3, v29).
    pub(crate) scissor: Option<[u32; 4]>,
    /// The pass's culling state, or `None` for "keep every triangle"
    /// (`research/docs/23` §3.3, v39).
    pub(crate) cull: Option<RenderPassCull>,
    /// The pass's blend state, or `None` for "write the fragment output"
    /// (`research/docs/23` §3.3, v40).
    pub(crate) blend: Option<RenderPassBlend>,
    /// The pass-wide multisample raster (`research/docs/23` §3.3, v51), or
    /// `None` for the single-sample raster every pre-v51 pass ran. When
    /// present, the encoder creates a four-sample texture per colour location,
    /// renders into it and resolves it into the attachment's own shared
    /// texture with `storeAction = .multisampleResolve`; the readback observes
    /// the resolve target exactly as the trace's attachment view.
    pub(crate) multisample: Option<SampleCount>,
    /// The rail-owned depth attachment this pass opens, or `None` for a pass
    /// with no depth surface (`research/docs/23` §3.3, v36).
    pub(crate) depth: Option<PlannedDepth>,
    /// The depth resolve a multisampled pass states, carried as the filter the
    /// encoder sets on the depth attachment (`research/docs/23` §3.3, v57c).
    /// `None` is every pass that does not resolve — the pre-v57 shapes and the
    /// single-sample stored depth.
    pub(crate) depth_resolve: Option<DepthResolveFilter>,
    /// The stencil resolve a multisampled pass states, carried as the filter
    /// the encoder sets on the stencil attachment (`research/docs/23` §3.3,
    /// v60). `None` is every pass that does not resolve — the pre-v60 shapes
    /// and the single-sample stored stencil.
    pub(crate) stencil_resolve: Option<StencilResolveFilter>,
    /// The rail-owned stencil attachment this pass opens, or `None` for a pass
    /// with no stencil surface (`research/docs/23` §3.3, v47).
    pub(crate) stencil: Option<PlannedStencil>,
    pub(crate) vertices: u32,
    /// Instances the draw runs (`research/docs/23` §3.3, v31): Metal's
    /// `drawPrimitives(vertexCount:instanceCount:)` second count. `1` for every
    /// pre-v31 pass.
    pub(crate) instance_count: u32,
    /// One entry per bound vertex stream, in binding order, with the bytes and
    /// footprints [`plan_vertex_input`] proved.
    pub(crate) vertex_streams: Vec<PlannedVertexStream<'a>>,
    /// The index buffer of an indexed draw, resolved from the pass's own view.
    pub(crate) indices: Option<PlannedIndexStream<'a>>,
    /// Readback length in bytes of one attachment: the tightly packed texel
    /// extent. Every admitted colour format stores four bytes per texel and
    /// every attachment of one pass shares an extent (checked in [`plan`]), so
    /// one length serves every attachment.
    pub(crate) texel_bytes: usize,
    /// Bytes per attachment row of the same shared shape (`research/docs/23`
    /// §3.5).
    pub(crate) row_pitch: usize,
}

/// The depth attachment a plan opens (`research/docs/23` §3.3, v36/v43).
///
/// Rail-owned like the Vulkan rail's: no trace identity lives here, so the plan
/// carries the shape the encoder creates and opens, and the trace's own landing
/// is resolved beside it ([`TraceRenderPlan::depth_landing`]). The store action
/// is what decides whether the surface outlives the pass: a storing one is read
/// back through the same `getBytes` shape the colour attachments use, and every
/// pre-v43 shape — no statement at all, or the explicit discard — keeps the
/// surface rail-owned and disappears with the pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedDepth {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `Some(bits)` for a clear (the `f32`'s own bits), `None` for `Load`.
    pub(crate) clear_bits: Option<u32>,
    /// The pass's depth state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub(crate) test: Option<DepthTest>,
    /// The store action the trace stated, or `None` for the pre-v43 shape
    /// (`research/docs/23` §3.3, v43). A storing surface is the only one this
    /// rail reads back, which is also why its texture is created with shared
    /// storage ([`depth_texture`]).
    pub(crate) store: Option<DepthStoreOp>,
}

impl PlannedDepth {
    /// Whether the pass keeps this surface — and therefore reads it back
    /// (`research/docs/23` §3.3, v43).
    pub(crate) fn storing(&self) -> bool {
        self.store == Some(DepthStoreOp::Store)
    }
}

/// The stencil attachment a plan opens (`research/docs/23` §3.3, v47/v49).
///
/// Rail-owned like the depth surface, so the plan carries the shape the encoder
/// creates and opens while the trace's own landing is resolved beside it
/// ([`TraceRenderPlan::stencil_landing`]). The store action is what decides
/// whether the surface outlives the pass: a storing one is read back through
/// the same `getBytes` shape the colour attachments use, one byte per texel,
/// and every pre-v49 shape — no statement at all, or the explicit discard —
/// keeps the surface rail-owned and disappears with the pass. What the trace
/// always states is the extent, the clear value or previous-contents load op,
/// and the stencil state the pass's draw tests and writes with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedStencil {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `Some(value)` for a clear, `None` for `Load`.
    pub(crate) clear_value: Option<u8>,
    /// The pass's stencil state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub(crate) test: Option<StencilTest>,
    /// The store action the trace stated, or `None` for the pre-v49 shape
    /// (`research/docs/23` §3.3, v49). A storing surface is the only one this
    /// rail reads back, which is also why its texture is created with shared
    /// storage ([`stencil_texture`]).
    pub(crate) store: Option<StoreOp>,
}

impl PlannedStencil {
    /// Whether the pass keeps this surface — and therefore reads it back
    /// (`research/docs/23` §3.3, v49).
    pub(crate) fn storing(&self) -> bool {
        self.store == Some(StoreOp::Store)
    }
}

/// One colour attachment of a planned pass, resolved before any Metal object
/// exists.
#[derive(Debug)]
pub(crate) struct PlannedAttachment<'a> {
    /// The pixel format this rail builds the attachment's texture and pipeline
    /// state with.
    pub(crate) format: RenderPixelFormat,
    /// The clear value or previous contents the attachment starts from.
    pub(crate) load: RenderLoadAction,
    /// The store action: `Store` makes the readback a landed observation;
    /// `DontCare` discards the attachment and yields no readback.
    pub(crate) store: RenderStoreAction,
    /// The tightly packed texels a [`LoadOp::Load`] uploads before the pass
    /// opens; `None` for a clear.
    pub(crate) initial: Option<&'a [u8]>,
}

/// Validate a render request against the contract and the rail's own allowlist.
///
/// Runs entirely without a device, so every refusal here is testable on a host
/// that cannot load Metal. Nothing outside the request is read: the pass carries
/// its own streams' bytes and its own attachments, so this call answers the
/// same way for a trace pass and for the device-level helper's trace-less
/// request.
pub(crate) fn plan<'a>(
    request: &OffscreenRenderRequest<'a>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<RenderPlan<'a>, ProviderError> {
    let attachments = &request.pass.color_attachments;
    // The rail's own extent-equality gate comes before the contract's viewport
    // rule: every colour location renders into the pass's one raster, so two
    // attachments of different extents cannot both land. The contract would
    // report the same shape only as a viewport disagreement, which names no
    // attachment; this refusal names the two extents instead.
    if let Some(first) = attachments.first() {
        for (index, other) in attachments.iter().enumerate().skip(1) {
            if other.width != first.width || other.height != first.height {
                return Err(args_refusal("render_attachment_extent_mismatch")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("width", FieldValue::Unsigned(other.width))
                    .with_field("height", FieldValue::Unsigned(other.height))
                    .with_field("first_width", FieldValue::Unsigned(first.width))
                    .with_field("first_height", FieldValue::Unsigned(first.height))
                    .with_detail("every colour attachment of one render pass shares one extent"));
            }
        }
    }
    // The pass's own shape rules and the pipeline/attachment format agreement
    // belong to the contract (`research/docs/23` §3.1, §3.2), not to this
    // rail.
    request.pass.validate().map_err(contract_refusal)?;
    request
        .pipeline
        .validate_against(request.pass)
        .map_err(contract_refusal)?;
    // The MRT contract admits `MAX_COLOR_ATTACHMENTS` attachments and a
    // matching format list, and this rail's reviewed modules cover one, two and
    // the full ceiling. A wider pass (reachable only through a
    // directly-constructed request, since core admission refuses it) is refused
    // instead of silently rendering only the first few locations (wave3 R1).
    if attachments.len() > usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap_or(usize::MAX) {
        return Err(mrt_attachment_count_refusal(attachments.len()));
    }
    // A present action hands exactly one attachment on to its target, and the
    // present encoder renders into that one texture: a present pass with a
    // second location would be dropped, so it is refused under the single-
    // attachment slug instead.
    let present = request.pass.present.is_some();
    if present && attachments.len() != 1 {
        return Err(capability_refusal("render_attachment_count_unsupported")
            .with_field(
                "attachments",
                FieldValue::Unsigned(attachments.len() as u64),
            )
            .with_field("maximum", FieldValue::Unsigned(1))
            .with_detail("a present pass hands exactly one colour attachment on to its target"));
    }
    // A present pass renders into the provider-owned target alone: it opens no
    // depth surface, so a trace that names one — stored or not — asks for a
    // state this shape cannot execute. Refusing here keeps the depth attachment
    // from being silently dropped instead of opened, which is what the
    // offscreen path would do with it (`research/docs/23` §3.3, v43; the same
    // slug, class and detail the Vulkan rail's present entry states).
    if present {
        if let Some(depth) = &request.pass.depth {
            return Err(capability_refusal("render_present_depth_unsupported")
                .with_field(
                    "store",
                    FieldValue::Text(
                        match depth.store {
                            Some(DepthStoreOp::Store) => "store",
                            Some(DepthStoreOp::DontCare) => "dontcare",
                            None => "unstated",
                        }
                        .to_owned(),
                    ),
                )
                .with_detail(
                    "the present rail renders into one provider-owned colour target and opens no \
                     depth surface",
                ));
        }
        // The stencil surface is the depth rule's sibling
        // (`research/docs/23` §3.3, v49): the present shape hands its one
        // colour attachment's texels on and opens no stencil surface, so a
        // trace that names one — stored or not — is refused instead of
        // executed with the attachment dropped, or, once a storing surface
        // names a landing, with those bytes silently left behind. This is the
        // same slug, class and detail the Vulkan rail's present entry states.
        if let Some(stencil) = &request.pass.stencil {
            return Err(capability_refusal("render_present_stencil_unsupported")
                .with_field(
                    "store",
                    FieldValue::Text(
                        match stencil.store {
                            Some(StoreOp::Store) => "store",
                            Some(StoreOp::DontCare) => "dontcare",
                            None => "unstated",
                        }
                        .to_owned(),
                    ),
                )
                .with_detail(
                    "the present rail renders into one provider-owned colour target and opens no \
                     stencil surface",
                ));
        }
    }
    // The pass's raster is its colour attachment's — or, for the
    // zero-colour-attachment depth pass (`research/docs/23` §3.3, v46), its
    // depth attachment's: the depth surface is the whole raster, and the
    // pipeline's empty colour-format list is what states that no colour
    // target exists beside it. A stencil surface is never the raster on its
    // own: it is a second surface beside the pass's colour or depth one, and
    // the contract refuses a colour-less pass with no depth surface
    // (`EmptyAttachmentList`) before this rail reaches the fallback
    // (`research/docs/23` §3.3, v47).
    let raster = match attachments.first() {
        Some(attachment) => [attachment.width, attachment.height],
        None => {
            let depth = request
                .pass
                .depth
                .as_ref()
                .ok_or_else(|| contract_refusal(ContractError::EmptyAttachmentList))?;
            let identity = depth.identity.ok_or_else(|| {
                args_refusal("render_depth_state_unsupported").with_detail(
                    "a pass with no colour attachment renders into its stored depth surface",
                )
            })?;
            if depth.store != Some(DepthStoreOp::Store) || identity.view_id.is_zero() {
                return Err(args_refusal("render_depth_state_unsupported").with_detail(
                    "a pass with no colour attachment renders into its stored depth surface",
                ));
            }
            [depth.width, depth.height]
        }
    };
    // The pipeline's (vertex-input shape, colour-format list) pair selects the
    // one reviewed module this call may compile; the (module, entry pair) pair
    // is then the whole allowlist, re-checked by `review_contract` below.
    let module = reviewed_module(
        &request.pipeline.vertex_layout,
        &request.pipeline.color_formats,
    );
    match module {
        Some(module) if request.source == module.source => {}
        Some(module) => {
            return Err(
                allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                    "a {} layout with {:?} compiles the bytes of `{}` and nothing else",
                    layout_name(&request.pipeline.vertex_layout),
                    request.pipeline.color_formats,
                    module.path
                )),
            );
        }
        None => {
            return Err(
                allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                    "no reviewed module carries the {:?} shape with a {} layout",
                    request.pipeline.color_formats,
                    layout_name(&request.pipeline.vertex_layout)
                )),
            );
        }
    }
    review_contract(request.pipeline)?;
    for attachment in attachments {
        if !SUPPORTED_COLOR_FORMATS.contains(&attachment.format) {
            return Err(
                capability_refusal("attachment_format_unsupported").with_field(
                    "format",
                    FieldValue::Unsigned(u64::from(attachment.format.code())),
                ),
            );
        }
    }
    if raster[0] > MAX_ATTACHMENT_DIMENSION[0] || raster[1] > MAX_ATTACHMENT_DIMENSION[1] {
        // The slug and fields capability admission uses for this fact
        // (`metal_api_core::provider::ProviderCapabilities::admit_render_passes`).
        return Err(capability_refusal("attachment_dimension_limit")
            .with_field("width", FieldValue::Unsigned(raster[0]))
            .with_field("height", FieldValue::Unsigned(raster[1]))
            .with_field(
                "maximum_width",
                FieldValue::Unsigned(MAX_ATTACHMENT_DIMENSION[0]),
            )
            .with_field(
                "maximum_height",
                FieldValue::Unsigned(MAX_ATTACHMENT_DIMENSION[1]),
            ));
    }
    // Bounded by the check above, so these conversions cannot lose a bit.
    let extent = [raster[0] as u32, raster[1] as u32];
    // Every admitted colour format and `depth32float` alike store four bytes per
    // texel, so one texel-byte count serves the colour attachments and the
    // depth-only pass's readback (`crates/metal-api-core`:
    // `AttachmentFormat::bytes_per_texel` and `DEPTH_BYTES_PER_TEXEL`). The
    // stencil surface is the one surface these two counts do not describe: one
    // byte per texel it is (`STENCIL_BYTES_PER_TEXEL`), so its own readback
    // computes the flat byte extent and the row width from the pass extent
    // beside the store it serves ([`read_stencil_texels`],
    // `research/docs/23` §3.3, v49).
    let texel_bytes = usize::try_from(
        u64::from(extent[0])
            .saturating_mul(u64::from(extent[1]))
            .saturating_mul(4),
    )
    .map_err(|_| capability_refusal("attachment_dimension_limit"))?;
    let row_pitch = usize::try_from(u64::from(extent[0]).saturating_mul(4))
        .map_err(|_| capability_refusal("attachment_dimension_limit"))?;
    // The vertex-input half: the streams with their bytes and their footprints.
    // Planned after the attachment because a stream is the draw's own input,
    // exactly as the attachment is its output.
    let (vertex_streams, indices) = plan_vertex_input(request.pass, request.pipeline)?;
    if request.initial.len() != attachments.len() {
        return Err(
            args_refusal("render_attachment_initial_mismatch").with_detail(format!(
                "{} attachments need one previous-bytes entry each, got {}",
                attachments.len(),
                request.initial.len()
            )),
        );
    }
    let mut planned_attachments = Vec::with_capacity(attachments.len());
    for (attachment, previous) in attachments.iter().zip(request.initial.iter().copied()) {
        let format = pixel_format(attachment.format)?;
        let load = load_action(attachment.load, format)?;
        let store = store_action(attachment.store)?;
        let initial = match (load, previous, present) {
            (RenderLoadAction::Clear(_), None, _) => None,
            (RenderLoadAction::DontCare, None, _) => None,
            (RenderLoadAction::Load, Some(bytes), _) if bytes.len() == texel_bytes => Some(bytes),
            (RenderLoadAction::Load, Some(bytes), _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(format!(
                        "LoadOp::Load needs {texel_bytes} tightly packed bytes, got {}",
                        bytes.len()
                    )),
                );
            }
            (RenderLoadAction::Load, None, true) => None,
            (RenderLoadAction::Load, None, false) => {
                return Err(args_refusal("render_attachment_initial_mismatch")
                    .with_detail("LoadOp::Load needs the attachment's previous texels"));
            }
            (RenderLoadAction::Clear(_), Some(_), _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(
                        "LoadOp::Clear writes every texel, so initial bytes are refused",
                    ),
                );
            }
            (RenderLoadAction::DontCare, Some(_), _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(
                        "LoadOp::DontCare reads and presets no pre-pass bytes, so initial bytes \
                         are refused",
                    ),
                );
            }
        };
        planned_attachments.push(PlannedAttachment {
            format,
            load,
            store,
            initial,
        });
    }
    // The depth resolve (`research/docs/23` §3.3, v57c) only means something
    // beside a multisample raster that keeps its depth surface: the resolve is
    // the reduction of the stored four-sample texels, so a resolve without
    // both is refused instead of silently ignored. The core contract refuses
    // the same shape with `DepthResolveWithoutStoredDepth`; this is the
    // value-level second line of defence for a directly-constructed request,
    // in the same place the Vulkan rail re-asserts it.
    if request.pass.depth_resolve.is_some() {
        let stored = request.pass.multisample.is_some()
            && request
                .pass
                .depth
                .as_ref()
                .is_some_and(|depth| depth.store == Some(DepthStoreOp::Store));
        if !stored {
            let store = request
                .pass
                .depth
                .as_ref()
                .and_then(|depth| depth.store)
                .map(DepthStoreOp::code);
            return Err(contract_refusal(
                ContractError::DepthResolveWithoutStoredDepth { store },
            ));
        }
    }
    // The stencil resolve (`research/docs/23` §3.3, v60) is the depth
    // resolve's sibling one byte wide: it only means something beside a
    // multisample raster that keeps its stencil surface, and the
    // `DepthResolvedSample` filter names the sample the depth resolve
    // selected. The core contract refuses the same shapes with
    // `StencilResolveWithoutStoredStencil` and
    // `StencilResolveWithoutDepthResolve`; this is the value-level second
    // line of defence for a directly-constructed request.
    if let Some(resolve) = request.pass.stencil_resolve {
        let stored = request.pass.multisample.is_some()
            && request
                .pass
                .stencil
                .as_ref()
                .is_some_and(|stencil| stencil.store == Some(StoreOp::Store));
        if !stored {
            let store = request
                .pass
                .stencil
                .as_ref()
                .and_then(|stencil| stencil.store);
            return Err(contract_refusal(
                ContractError::StencilResolveWithoutStoredStencil { store },
            ));
        }
        if resolve.filter == StencilResolveFilter::DepthResolvedSample
            && request.pass.depth_resolve.is_none()
        {
            return Err(contract_refusal(
                ContractError::StencilResolveWithoutDepthResolve,
            ));
        }
        let mask = 1u32 << u32::from(resolve.filter.code());
        if stencil_resolve_modes & mask == 0 {
            return Err(
                capability_refusal("render_stencil_resolve_filter_unsupported")
                    .with_field(
                        "filter",
                        FieldValue::Unsigned(u64::from(resolve.filter.code())),
                    )
                    .with_field(
                        "modes",
                        FieldValue::Unsigned(u64::from(stencil_resolve_modes)),
                    ),
            );
        }
    }
    let module = module.expect("the source check above refused a shape without a module");
    Ok(RenderPlan {
        source: module.source,
        module_path: module.path,
        vertex_entry: request.pipeline.vertex_entry.as_str(),
        fragment_entry: request.pipeline.fragment_entry.as_str(),
        attachments: planned_attachments,
        extent,
        viewport: request.pass.viewport,
        scissor: request.pass.scissor,
        cull: request.pass.cull,
        blend: request.pass.blend.clone(),
        // The multisample raster (`research/docs/23` §3.3, v51/v61). The
        // contract already refused a single-sample state, a non-clear load and
        // a depth or stencil surface beside it; the rail re-asserts the counts
        // its encoder knows how to build, so a directly-constructed request
        // cannot reach `newTextureWithDescriptor` with a raster this
        // increment does not execute.
        multisample: match request.pass.multisample {
            Some(multisample)
                if matches!(
                    multisample.sample_count,
                    SampleCount::Two | SampleCount::Four | SampleCount::Eight
                ) =>
            {
                // A stored multisampled depth surface is admitted from v57c on,
                // through the resolve the pass then has to state: its texels
                // are only observable as the resolve's reduction, so a stored
                // surface without one stays refused, and a filter the device
                // does not report is refused by the same per-filter question
                // the capability snapshot answered (`research/docs/23` §3.3,
                // v57c).
                if request
                    .pass
                    .depth
                    .as_ref()
                    .is_some_and(|depth| depth.store == Some(DepthStoreOp::Store))
                {
                    match request.pass.depth_resolve {
                        Some(resolve) => {
                            let mask = 1u32 << u32::from(resolve.filter.code());
                            if depth_resolve_modes & mask == 0 {
                                return Err(capability_refusal(
                                    "render_depth_resolve_filter_unsupported",
                                )
                                .with_field(
                                    "filter",
                                    FieldValue::Unsigned(u64::from(resolve.filter.code())),
                                )
                                .with_field(
                                    "modes",
                                    FieldValue::Unsigned(u64::from(depth_resolve_modes)),
                                ));
                            }
                        }
                        None => {
                            return Err(capability_refusal(
                                "render_multisample_depth_store_unsupported",
                            )
                            .with_detail(
                                "a multisampled depth surface cannot be kept without a depth \
                                 resolve",
                            ));
                        }
                    }
                }
                // The stencil sibling (`research/docs/23` §3.3, v55/v60): a
                // kept stencil surface needs the stencil resolve, and a
                // combined depth-stencil surface is one texture both faces
                // share, so its two store decisions have to agree: both
                // rail-owned with no resolve — the v66 write-then-test pair —
                // or both kept through their resolves, the v60 shape
                // (`research/docs/23` §3.3, v60/v66).
                if let (Some(depth), Some(stencil)) = (&request.pass.depth, &request.pass.stencil) {
                    let rail_owned = !depth.is_stored() && !stencil.is_stored();
                    let resolved = depth.is_stored()
                        && stencil.is_stored()
                        && request.pass.depth_resolve.is_some()
                        && request.pass.stencil_resolve.is_some();
                    if !rail_owned && !resolved {
                        return Err(capability_refusal(
                            "render_stencil_combined_surface_unsupported",
                        )
                        .with_detail(
                            "the multisample raster opens one depth-stencil surface: \
                                     the two faces keep both or neither — both rail-owned with \
                                     no resolve, or both stored through their resolves",
                        ));
                    }
                }
                if request
                    .pass
                    .stencil
                    .as_ref()
                    .is_some_and(|stencil| stencil.store == Some(StoreOp::Store))
                {
                    // The stored surface is admitted from v60 on, through the
                    // resolve the pass then has to state: its texels are only
                    // observable as the resolve's reduction, so a stored
                    // surface without one stays refused, and a filter the
                    // device does not report is refused by the same per-filter
                    // question the capability snapshot answered
                    // (`research/docs/23` §3.3, v60).
                    match request.pass.stencil_resolve {
                        Some(resolve) => {
                            let mask = 1u32 << u32::from(resolve.filter.code());
                            if stencil_resolve_modes & mask == 0 {
                                return Err(capability_refusal(
                                    "render_stencil_resolve_filter_unsupported",
                                )
                                .with_field(
                                    "filter",
                                    FieldValue::Unsigned(u64::from(resolve.filter.code())),
                                )
                                .with_field(
                                    "modes",
                                    FieldValue::Unsigned(u64::from(stencil_resolve_modes)),
                                ));
                            }
                        }
                        None => {
                            return Err(capability_refusal(
                                "render_multisample_stencil_store_unsupported",
                            )
                            .with_detail(
                                "a multisampled stencil surface cannot be kept without a \
                                 stencil resolve",
                            ))
                        }
                    }
                }
                Some(multisample.sample_count)
            }
            Some(_) => {
                return Err(capability_refusal("render_multisample_state_unsupported")
                    .with_detail(
                        "the multisample increment executes the two-, four- and eight-sample \
                         rasters only",
                    ));
            }
            None => None,
        },
        depth: request.pass.depth.as_ref().map(|depth| PlannedDepth {
            width: u32::try_from(depth.width).unwrap_or(u32::MAX),
            height: u32::try_from(depth.height).unwrap_or(u32::MAX),
            clear_bits: depth.load.clear_depth().map(f32::to_bits),
            test: request.pass.depth_test,
            // The store action is the pass's own statement
            // (`research/docs/23` §3.3, v43): the pre-v43 shapes leave it
            // absent, and core admission has already held the storing shape to
            // naming a landing identity.
            store: depth.store,
        }),
        depth_resolve: request.pass.depth_resolve.map(|resolve| resolve.filter),
        stencil_resolve: request.pass.stencil_resolve.map(|resolve| resolve.filter),
        stencil: request.pass.stencil.as_ref().map(|stencil| PlannedStencil {
            width: u32::try_from(stencil.width).unwrap_or(u32::MAX),
            height: u32::try_from(stencil.height).unwrap_or(u32::MAX),
            // The same decoding the depth load op uses: the clear arm
            // carries the value every stencil texel starts from, and `Load`
            // keeps the attachment's previous contents
            // (`research/docs/23` §3.3, v47).
            clear_value: stencil.load.clear_value(),
            // The state is the pass's own, and core admission has already
            // refused a test with no attachment to test
            // (`StencilTestWithoutAttachment`).
            test: request.pass.stencil_test,
            // The store action is the pass's own statement
            // (`research/docs/23` §3.3, v49): the pre-v49 shapes leave it
            // absent, and core admission has already held the storing shape to
            // naming a landing identity.
            store: stencil.store,
        }),
        vertices: request.pass.vertices,
        instance_count: request.pass.instance_count,
        vertex_streams,
        indices,
        texel_bytes,
        row_pitch,
    })
}

/// The vertex-input shape of a pipeline, as the refusals spell it.
pub(crate) fn layout_name(layout: &VertexLayout) -> &'static str {
    match layout {
        VertexLayout::None => "vertex_id",
        VertexLayout::Buffers(_) => "vertex-buffer",
    }
}

/// The rail's review gate for a render pipeline contract.
///
/// Each reviewed module carries exactly one vertex entry and one fragment entry,
/// and the contract's [`VertexLayout`] and `color_formats` say which module it
/// may be: a single-output contract names the module its layout selects, and a
/// dual-format contract with a stream layout the dual module. A shape no
/// module was reviewed for — a dual-format `vertex_id` contract, a wider
/// format list, or a dual format pair that is not two `Rgba8Unorm` locations —
/// is refused with the same slug, class and phase the compute allowlist gives
/// an unreviewed kernel (`lib.rs::bounded_contract`,
/// `native_shader_not_allowlisted`): a matching file name, an edited module or
/// a recompiled one must not be enough to run different source
/// (`research/docs/23` §6 Step 7). Registration
/// (`NativeMetalProvider::register_render_pipeline`) and [`plan`] both run it,
/// so the refusal is reachable before a submission as well as inside one.
pub(crate) fn review_contract(contract: &RenderPipelineContract) -> Result<(), ProviderError> {
    let Some(module) = reviewed_module(&contract.vertex_layout, &contract.color_formats) else {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "no reviewed module carries the {:?} shape with a {} layout",
                contract.color_formats,
                layout_name(&contract.vertex_layout),
            )),
        );
    };
    if contract.vertex_entry != module.vertex_entry
        || contract.fragment_entry != module.fragment_entry
    {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "a {} layout with {:?} compiles `{}`, which carries {:?} and {:?}",
                layout_name(&contract.vertex_layout),
                contract.color_formats,
                module.path,
                module.vertex_entry,
                module.fragment_entry,
            )),
        );
    }
    Ok(())
}

/// The previous bytes an offscreen `LoadOp::Load` pass uploads before it opens.
///
/// The declaring case's view is the only channel that carries an attachment's
/// previous contents: [`metal_api_core::provider::RenderAttachment`] restates
/// the view's identity and shape and names no bytes, so a `Load` resolves them
/// from what the declaration itself owns ([`BufferSource::OwnedBytes`])
/// (`research/docs/23` §3.3). A lease-backed declaration carries bytes this rail
/// does not hold, and the refusal reuses the slug, class and phase this rail and
/// the Vulkan rail give an unexecutable load op
/// (`crates/metal-api-vulkan/src/compute_provider.rs`, the `loading` branch).
/// `Clear` and `DontCare` resolve no bytes: a clear writes every texel and a
/// `DontCare` attachment declares its pre-pass contents undefined, so neither
/// shape presets the attachment (`research/docs/23` §3.1, v20).
pub(crate) fn previous_bytes(
    load: LoadOp,
    view: &BufferView,
) -> Result<Option<&[u8]>, ProviderError> {
    match (load, &view.source) {
        (LoadOp::Load, BufferSource::OwnedBytes(bytes)) => Ok(Some(bytes.as_slice())),
        (LoadOp::Load, other) => Err(capability_refusal("attachment_load_op_unsupported")
            .with_field("load_op", FieldValue::Text("load".to_owned()))
            .with_field(
                "storage_mode",
                FieldValue::Text(storage_mode_name(other).to_owned()),
            )
            .with_detail(
                "the first `LoadOp::Load` increment uploads the bytes the declaring view \
                 owns; a leased declaration has no path through this rail",
            )),
        (LoadOp::Clear(_) | LoadOp::DontCare, _) => Ok(None),
    }
}

/// Refuse a trace whose compute passes would be reordered against a render
/// pass's stores.
///
/// The trace path executes every compute pass before every render pass, in trace
/// order within each group (see [`plan_trace`] and
/// `NativeMetalProvider::execute_render_passes`). Compute passes that come
/// before a render pass therefore run in the order the trace asked for, and so
/// do render passes among themselves. The one shape that would silently change
/// meaning is a compute pass that *follows* a render pass and binds a view that
/// render pass stores: it would observe pre-render bytes where the trace's
/// serial order defines post-render ones. Core admission already refuses the
/// write/write half of the pair (`AttachmentComputeConflict`,
/// `attachment_resource_conflict`); this is the read half, refused rather than
/// executed in an order the bytes would not reflect.
///
/// Core admission owns both halves now (`ContractError::
/// RenderPassOrderUnsupported`, slug `render_pass_order_unsupported`, review
/// item I4, 2026-09-14), and every submitted trace reached it through
/// `ProviderCapabilities::admit`. This walk stays as the rail's own defense for
/// a value-level plan: it compares view identities, so it is at least as strict
/// as the contract's byte ranges and never admits a trace the contract refused.
pub(crate) fn refuse_reordered_render_reads(trace: &ComputeTrace) -> Result<(), ProviderError> {
    let mut render_written = BTreeMap::<ViewId, usize>::new();
    for (index, entry) in trace.passes.iter().enumerate() {
        match entry {
            TracePass::Render(pass) => {
                for attachment in &pass.color_attachments {
                    render_written.entry(attachment.view_id).or_insert(index);
                }
            }
            TracePass::Compute(pass) => {
                let bound = pass
                    .buffers
                    .iter()
                    .map(|view| view.view_id)
                    .chain(pass.textures.iter().map(|texture| texture.view_id));
                for view in bound {
                    if let Some(render_pass) = render_written.get(&view) {
                        return Err(refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "render_pass_order_unsupported",
                        )
                        .with_field("pass", FieldValue::Unsigned(index as u64))
                        .with_field("render_pass", FieldValue::Unsigned(*render_pass as u64))
                        .with_field("view", FieldValue::Unsigned(view.get()))
                        .with_detail(
                            "a compute pass that follows a render pass storing this view \
                             would observe pre-render bytes, because this increment runs \
                             every compute pass before every render pass",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn capability_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
}

fn compile_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Compile, ProviderErrorClass::Compile, slug)
}

/// The review gate: an unreviewed source or entry is a capability refusal, the
/// same class the compute allowlist gives an unreviewed kernel
/// (`lib.rs::bounded_contract`, `native_shader_not_allowlisted`).
fn allowlist_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Compile, ProviderErrorClass::Capability, slug)
}

/// A structural refusal about a fact only this rail models, hence a slug of its
/// own: the contract's pass carries no "previous contents" field, so the
/// `LoadOp::Load` agreement is the rail's to state.
fn args_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Args, slug)
}

/// The refusal for a pass whose attachment count this rail cannot execute.
///
/// The capability bit reports `MAX_COLOR_ATTACHMENTS`, so a wider pass is
/// refused here with the rail's own limit instead of silently rendering only
/// the first few locations. With the reviewed modules covering one, two and the
/// ceiling, this gate is the fail-closed half for a directly-constructed
/// request that skipped core admission (wave3 R1, v24).
fn mrt_attachment_count_refusal(attachments: usize) -> ProviderError {
    capability_refusal("render_mrt_attachment_count_unsupported")
        .with_field("attachments", FieldValue::Unsigned(attachments as u64))
        .with_field(
            "maximum",
            FieldValue::Unsigned(MAX_COLOR_ATTACHMENTS as u64),
        )
        .with_detail(
            "the reviewed modules write one, two or MAX_COLOR_ATTACHMENTS colour locations; \
             a wider pass would silently drop its later locations",
        )
}

/// Map a core contract error onto the refusal admission already uses.
///
/// `metal-api-core`'s exhaustive mapping (`contract_error_refusal`) is private to
/// that crate, so this table repeats only the variants a render pass and its
/// pipeline can raise here, with the same class and slug. It is a mirror, not a
/// second policy — and in the real flow it is defence in depth, because a trace
/// reaches a provider through `ProviderCapabilities::admit` first.
fn contract_refusal(error: ContractError) -> ProviderError {
    use ContractError as E;
    let (class, slug) = match &error {
        E::UnsupportedAttachmentFormat(_) => (
            ProviderErrorClass::Capability,
            "attachment_format_unsupported",
        ),
        E::UnsupportedAttachmentLoadOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_load_op_unsupported",
        ),
        E::UnsupportedAttachmentStoreOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_store_op_unsupported",
        ),
        E::AttachmentLimitExceeded { .. } => (
            ProviderErrorClass::Capability,
            "attachment_count_unsupported",
        ),
        E::ViewportOriginUnsupported { .. } => (
            ProviderErrorClass::Capability,
            "viewport_origin_unsupported",
        ),
        E::DrawVertexCountMismatch { .. } => {
            (ProviderErrorClass::Capability, "draw_shape_unsupported")
        }
        // Everything else a pass or a pipeline can raise here is structural: the
        // class and slug core admission gives it.
        _ => (ProviderErrorClass::Args, "trace_contract_invalid"),
    };
    refusal(ProviderPhase::Resolve, class, slug).with_detail(error.to_string())
}

/// The refusal a render pass gets when it names a pipeline this context never
/// registered as a render pipeline.
///
/// One counter and one id namespace serve both rails, so the wording is the
/// mirror of [`crate::native`]'s `unknown_pipeline`: a compute pass naming a
/// render registration and a render pass naming a compute registration are both
/// resource refusals, with the rail that refused them as the slug.
fn unknown_render_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_render_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

/// One render pass of a trace, planned before any device object exists.
///
/// The pass and the contract are borrowed from values that outlive execution —
/// the trace itself and the provider's render registrations — while the landing
/// view points into the serial view pool the encoder is built from. The planned
/// [`RenderPlan`] is the same value the encoder body consumes, so planning once
/// and encoding later cannot disagree.
#[derive(Debug)]
pub(crate) struct TraceRenderPlan<'a> {
    pub(crate) pass: &'a RenderPassDescriptor,
    pub(crate) contract: &'a RenderPipelineContract,
    /// One landing view per colour attachment, in location order: the pool
    /// view whose identity covers the attachment, which is the writeback
    /// channel each attachment's texels leave through.
    pub(crate) landings: Vec<&'a BufferView>,
    /// The landing view of a stored depth attachment, or `None` for every
    /// shape whose depth surface does not outlive its pass
    /// (`research/docs/23` §3.3, v43): the pool view whose identity is the
    /// depth identity, which is the writeback channel the stored texels leave
    /// through. A pass with no depth attachment and one that discards it both
    /// land here as `None`, exactly as they state nothing about a landing.
    pub(crate) depth_landing: Option<&'a BufferView>,
    /// The landing view of a stored stencil attachment, or `None` for every
    /// shape whose stencil surface does not outlive its pass
    /// (`research/docs/23` §3.3, v49): the pool view whose identity is the
    /// stencil identity, which is the writeback channel the stored one-byte
    /// texels leave through. A pass with no stencil attachment and one that
    /// discards it both land here as `None`, exactly as they state nothing
    /// about a landing.
    pub(crate) stencil_landing: Option<&'a BufferView>,
    pub(crate) plan: RenderPlan<'a>,
    /// The present action hanging off this pass, if any, with its sentinel
    /// already expanded to the target's whole texel extent so the macOS
    /// present path only has to upload it (`research/docs/24` §3.1, §6 Step 7).
    pub(crate) present: Option<PresentPlan<'a>>,
}

/// One present action, planned before the first Metal object exists.
///
/// The descriptor is borrowed from the pass it hands on, and the sentinel is
/// the expanded byte string the present path presets into the target before
/// the render runs. `None` means the target declares [`InitialState::Undefined`]
/// and holds whatever the device gave it (`research/docs/24` §3.1).
#[derive(Debug)]
pub(crate) struct PresentPlan<'a> {
    /// The present descriptor the pass carries.
    pub(crate) descriptor: &'a PresentDescriptor,
    /// The sentinel texel replicated across the whole target, or `None`.
    pub(crate) sentinel: Option<Vec<u8>>,
}

impl TraceRenderPlan<'_> {
    /// The writebacks this pass's readbacks become: one [`BufferWriteback`] per
    /// *stored* landing view, in location order, followed by the stored depth
    /// attachment's own when the pass has one (`research/docs/23` §3.3, v43)
    /// and then the stored stencil attachment's own (`research/docs/23` §3.3,
    /// v49).
    ///
    /// The view identity, allocation and offset are each landing view's own, so
    /// resource admission, lease bookkeeping and readback consumers need no
    /// second path: every attachment lands exactly where a compute pass writing
    /// the same view would (`research/docs/23` §6 Step 7). A discarded
    /// attachment has no readback and no writeback: its landing view stays in
    /// the plan for the load-side resolution, but its bytes never leave the
    /// pass, so it cannot present a blank readback as "landed correctly"
    /// (`research/docs/23` §3.6, v19). The caller's `readback` therefore carries
    /// one entry per stored attachment, matching the filtered landings, plus
    /// the depth texels exactly when the pass stores that surface, and the
    /// stencil texels exactly when it stores that one.
    ///
    /// The depth and stencil writebacks are the same shape as the colour ones —
    /// the landing view's own identity and offset, and the surface's own
    /// `depth32float` or one-byte `stencil8` texels — so neither needs a second
    /// channel. The list they are appended to need not be in identity order
    /// itself: every caller folds it through [`merge_writebacks`], which is
    /// where the canonical order the core contract states is established.
    pub(crate) fn writebacks(&self, readback: RenderReadback) -> Vec<BufferWriteback> {
        let mut writebacks: Vec<BufferWriteback> = self
            .landings
            .iter()
            .zip(&self.plan.attachments)
            .filter(|(_, attachment)| attachment.store == RenderStoreAction::Store)
            .zip(readback.attachments)
            .map(|((landing, _), bytes)| BufferWriteback {
                view_id: landing.view_id,
                allocation_id: landing.allocation_id,
                offset: landing.offset,
                bytes,
            })
            .collect();
        // A discarded or absent depth surface has no readback, so it adds no
        // writeback: the bytes never left the pass (`research/docs/23` §3.3,
        // v43).
        if let (Some(landing), Some(texels)) = (self.depth_landing, readback.depth) {
            writebacks.push(BufferWriteback {
                view_id: landing.view_id,
                allocation_id: landing.allocation_id,
                offset: landing.offset,
                bytes: texels,
            });
        }
        // A discarded or absent stencil surface is the same story one byte
        // wide (`research/docs/23` §3.3, v49): no readback, no writeback.
        if let (Some(landing), Some(texels)) = (self.stencil_landing, readback.stencil) {
            writebacks.push(BufferWriteback {
                view_id: landing.view_id,
                allocation_id: landing.allocation_id,
                offset: landing.offset,
                bytes: texels,
            });
        }
        writebacks
    }

    /// The single writeback a present pass's one attachment becomes.
    pub(crate) fn writeback(&self, texels: Vec<u8>) -> BufferWriteback {
        let landing = self.landings[0];
        BufferWriteback {
            view_id: landing.view_id,
            allocation_id: landing.allocation_id,
            offset: landing.offset,
            bytes: texels,
        }
    }
}

/// Plan every render pass of a trace, without a device.
///
/// Four decisions have to be made before the first Metal object exists, and all
/// of them are answerable from values: the order the rails run in
/// ([`refuse_reordered_render_reads`], whose rule core admission also states as
/// part of the contract), the reviewed allowlist, each attachment's landing
/// view, and the previous bytes a loading pass uploads ([`previous_bytes`]).
/// `pool` is [`ComputeTrace::serial_resources`], the same pool the encoder
/// binds, and `contracts` holds the render contracts the provider registered
/// for the pipeline ids this trace names — a caller-supplied table entry is
/// checked against those registrations in `native.rs`, where the registry
/// lives. The pool's only job here is each attachment's landing view: a render
/// input carries its own bytes, so the streams a draw reads are resolved from
/// the pass itself ([`plan_vertex_input`]).
pub(crate) fn plan_trace<'a>(
    trace: &'a ComputeTrace,
    pool: &'a [BufferView],
    contracts: &'a BTreeMap<PipelineId, RenderPipelineContract>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<Vec<TraceRenderPlan<'a>>, ProviderError> {
    if !trace.has_render_passes() {
        return Ok(Vec::new());
    }
    refuse_reordered_render_reads(trace)?;
    let mut planned = Vec::with_capacity(trace.render_passes().count());
    for pass in trace.render_passes() {
        let contract = contracts
            .get(&pass.pipeline)
            .ok_or_else(|| unknown_render_pipeline(pass.pipeline))?;
        // An indirect draw replays its pass through `MTLIndirectRenderCommand`
        // state, and the first indirect increment builds a command that carries
        // the pipeline state and the draw counts — not the vertex streams a
        // caller-held layout reads. A pass that binds one is refused here
        // rather than replayed from buffers nothing bound (`research/docs/25`
        // §6 Step 7b; the same slug the ICB rail uses for a shape it cannot
        // replay).
        if matches!(
            trace.indirect.as_ref().map(|indirect| indirect.command),
            Some(IndirectCommandDescriptor::Draw { .. })
        ) && (!pass.vertex_buffers.is_empty() || pass.indices.is_some())
        {
            return Err(capability_refusal("icb_command_unsupported").with_detail(
                "an indirect draw replays the vertex_id shape; a pass that binds caller-held \
                 vertex or index streams is not part of the first indirect increment",
            ));
        }
        // An attachment that no buffer view covers has no landing rail: the
        // texels would have nowhere to go, so the pass is refused instead of
        // being executed and dropped. Each declared view is resolved before its
        // load op because a loading pass reads its previous bytes from the same
        // declaration (`research/docs/23` §3.3).
        let mut landings = Vec::with_capacity(pass.color_attachments.len());
        let mut previous = Vec::with_capacity(pass.color_attachments.len());
        for attachment in &pass.color_attachments {
            let landing = pool
                .iter()
                .find(|view| {
                    view.view_id == attachment.view_id
                        && view.allocation_id == attachment.allocation_id
                })
                .ok_or_else(|| {
                    capability_refusal("render_attachment_landing_unsupported")
                        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                        .with_field(
                            "allocation",
                            FieldValue::Unsigned(attachment.allocation_id.get()),
                        )
                        .with_detail(
                            "attachment bytes land through the buffer writeback channel, \
                         and this trace declares no buffer view covering the attachment",
                        )
                })?;
            // An offscreen `Load` uploads the bytes the declaring view owns
            // before the pass opens. A present pass's `Load` keeps the target's
            // own initial state, which the present path supplies, so it
            // resolves no bytes (`research/docs/24` §3.1).
            let bytes = if pass.present.is_none() {
                previous_bytes(attachment.load, landing)?
            } else {
                None
            };
            landings.push(landing);
            previous.push(bytes);
        }
        // The stored depth attachment's landing view, resolved before the pass
        // runs for the same reason the colour ones are: the texels it receives
        // have to be named by the trace, and a storing surface without a
        // declaration is refused instead of executed and dropped
        // (`research/docs/23` §3.3, v43). Every other shape — no depth
        // attachment, or one the pass discards with itself — states no landing
        // and resolves none.
        let depth_landing = match pass.depth.as_ref() {
            Some(depth) => match (depth.store, depth.identity) {
                (Some(DepthStoreOp::Store), Some(identity)) => Some(
                    pool.iter()
                        .find(|view| {
                            view.view_id == identity.view_id
                                && view.allocation_id == identity.allocation_id
                        })
                        .ok_or_else(|| {
                            capability_refusal("render_depth_landing_unsupported")
                                .with_field("view", FieldValue::Unsigned(identity.view_id.get()))
                                .with_field(
                                    "allocation",
                                    FieldValue::Unsigned(identity.allocation_id.get()),
                                )
                                .with_detail(
                                    "a stored depth attachment's texels land through the buffer \
                                     writeback channel, and this trace declares no buffer view \
                                     covering the attachment",
                                )
                        })?,
                ),
                _ => None,
            },
            None => None,
        };
        // The stored stencil attachment's landing view is the depth rule one
        // byte wide (`research/docs/23` §3.3, v49): resolved before the pass
        // runs because the texels it receives have to be named by the trace,
        // and refused by name when no declaration covers them, rather than
        // executed and dropped. Every other shape — no stencil attachment, or
        // one the pass discards with itself — states no landing and resolves
        // none.
        let stencil_landing = match pass.stencil.as_ref() {
            Some(stencil) => match (stencil.store, stencil.identity) {
                (Some(StoreOp::Store), Some(identity)) => Some(
                    pool.iter()
                        .find(|view| {
                            view.view_id == identity.view_id
                                && view.allocation_id == identity.allocation_id
                        })
                        .ok_or_else(|| {
                            capability_refusal("render_stencil_landing_unsupported")
                                .with_field("view", FieldValue::Unsigned(identity.view_id.get()))
                                .with_field(
                                    "allocation",
                                    FieldValue::Unsigned(identity.allocation_id.get()),
                                )
                                .with_detail(
                                    "a stored stencil attachment's texels land through the buffer \
                                     writeback channel, and this trace declares no buffer view \
                                     covering the attachment",
                                )
                        })?,
                ),
                _ => None,
            },
            None => None,
        };
        let plan_of_pass = plan(
            &OffscreenRenderRequest {
                pass,
                pipeline: contract,
                source: reviewed_module(&contract.vertex_layout, &contract.color_formats)
                    .map_or("", |module| module.source),
                initial: previous,
            },
            depth_resolve_modes,
            stencil_resolve_modes,
        )?;
        let present = pass.present.as_ref().map(|descriptor| {
            let sentinel = descriptor.target.initial.sentinel().map(|texel| {
                let texels = usize::try_from(
                    u64::from(plan_of_pass.extent[0])
                        .saturating_mul(u64::from(plan_of_pass.extent[1])),
                )
                .unwrap_or(0);
                texel.repeat(texels)
            });
            PresentPlan {
                descriptor,
                sentinel,
            }
        });
        planned.push(TraceRenderPlan {
            pass,
            contract,
            landings,
            depth_landing,
            stencil_landing,
            plan: plan_of_pass,
            present,
        });
    }
    Ok(planned)
}

/// Fold compute and render writebacks into the one canonical list a submission
/// returns.
///
/// Compute and render writebacks share one channel and one rule: one complete
/// writeback per written view, keyed by identity. A view both rails could have
/// written is refused by core admission (`AttachmentComputeConflict`), and the
/// render rail runs last, so a repeated key keeps the bytes the render pass
/// ended with. The attachment's own view is in the serial pool as a written view
/// (`ComputeTrace::serial_resources`), which is exactly why the merge — and not
/// the compute readback — is what lands its texels.
pub(crate) fn merge_writebacks(
    compute: Vec<BufferWriteback>,
    render: Vec<BufferWriteback>,
) -> Vec<BufferWriteback> {
    let mut merged = BTreeMap::new();
    for writeback in compute.into_iter().chain(render) {
        merged.insert((writeback.allocation_id, writeback.view_id), writeback);
    }
    merged.into_values().collect()
}

/// The texels one offscreen render pass hands back (`research/docs/23` §3.3,
/// v43/v49).
///
/// `attachments` carries one entry per *stored* colour attachment, in location
/// order: the shape the readback channel had before the depth attachment could
/// be observed. `depth` carries the stored depth surface's own tightly packed
/// `depth32float` texels — four bytes per texel over the pass's extent, the
/// same region [`read_texels`] reads a colour attachment from — or `None` when
/// the pass discards its depth attachment, which is what every pre-v43 trace
/// states. `stencil` carries the stored stencil surface's own tightly packed
/// `stencil8` texels — one byte per texel over the same region, read by
/// [`read_stencil_texels`] — or `None` when the pass discards its stencil
/// attachment, which is what every pre-v49 trace states.
#[derive(Debug)]
pub(crate) struct RenderReadback {
    pub(crate) attachments: Vec<Vec<u8>>,
    pub(crate) depth: Option<Vec<u8>>,
    pub(crate) stencil: Option<Vec<u8>>,
}

/// Execute one offscreen render pass and return its tightly packed texel bytes,
/// one readback per colour attachment in location order, the stored depth
/// surface's own when the pass has one and the stored stencil surface's own
/// when it has that one.
///
/// Not verified on an Apple GPU: the check that would verify this encoder body
/// is the Rust provider's own render path in a committed suite, and the macOS
/// oracle's `--render-selftest` run `34774478149` is the observation behind the
/// capability flip (`conformance/RENDER-CAPTURE.md` §5, §6).
///
/// This entry point has no trace, but a render input carries its own bytes, so a
/// pass that binds vertex or index streams still executes from the bytes it
/// declared (`research/docs/23` §3.6). What such a call does not have is an
/// attachment landing view, which is why the trace path is
/// [`plan_trace`] + [`encode_offscreen_render`]: the texels of this entry point
/// are returned to the caller instead of landing in a pooled view.
#[cfg(target_os = "macos")]
pub(crate) fn execute_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    request: &OffscreenRenderRequest<'_>,
) -> Result<RenderReadback, ProviderError> {
    // The depth-resolve modes the rail re-asserts come from the same device
    // probe the capability snapshot published, so a directly-constructed
    // request is refused by the same mask admission used.
    let planned = plan(
        request,
        device_depth_resolve_capability_bits(device).depth_resolve_modes,
        device_stencil_resolve_capability_bits(device).stencil_resolve_modes,
    )?;
    encode_offscreen_render(device, queue, &planned)
}

/// Encode, commit and read back one already planned pass.
///
/// Split from [`execute_offscreen_render`] so the trace path can plan once
/// ([`plan_trace`], before the compute command buffer is committed) and then
/// encode that same decision, instead of planning a second, possibly different,
/// pass. The attachments are fresh per pass: they are created here and dropped
/// with the readbacks, which is the offscreen shape (`research/docs/23` §6
/// Step 7).
#[cfg(target_os = "macos")]
pub(crate) fn encode_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
) -> Result<RenderReadback, ProviderError> {
    objc::rc::autoreleasepool(|| {
        let attachments = attachment_textures(device, planned)?;
        encode_into_and_readback(device, queue, planned, &attachments, None)
    })
}

/// Encode, commit and read back one already planned present pass into the
/// caller-held target texture.
///
/// The target is the present action's own texture, held by (allocation, view)
/// across submissions (`research/docs/24` §6 Step 7); this function renders the
/// pass's attachment into it and reads the target back after `wait`, but it
/// neither creates nor destroys the texture. The acquire/present counts live in
/// `native.rs`, which calls this between its two counter increments.
#[cfg(target_os = "macos")]
pub(crate) fn encode_present_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    target: &Texture,
) -> Result<Vec<u8>, ProviderError> {
    objc::rc::autoreleasepool(|| {
        let readbacks =
            encode_into_and_readback(device, queue, planned, std::slice::from_ref(target), None)?;
        readbacks
            .attachments
            .into_iter()
            .next()
            .ok_or_else(|| resource_refusal("metal_render_attachment_descriptor_unavailable"))
    })
}

/// Encode, commit and read back one already planned offscreen pass whose
/// full-screen triangle is replayed from one `MTLIndirectCommandBuffer`
/// (`research/docs/25` §6 Step 7b).
///
/// The pass shape rules are the ones [`encode_offscreen_render`] already
/// enforces — this entry point only swaps the direct draw for an indirect
/// replay, so a pass that is not admitted as an offscreen render cannot reach
/// it. The draw command's vertex and instance counts are the ones the
/// [`crate::icb::plan_replay`] narrowed to a non-indexed draw.
#[cfg(target_os = "macos")]
pub(crate) fn encode_indirect_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    replay: &icb::IcbPlan,
) -> Result<RenderReadback, ProviderError> {
    // `plan_trace` refuses an indirect draw whose pass binds streams, because
    // the replay shape this rail builds carries the pipeline state and the draw
    // counts rather than the streams a caller-held layout reads. This is the
    // same rule one level down, for a caller that reaches the encoder without a
    // trace plan.
    if !planned.vertex_streams.is_empty() || planned.indices.is_some() {
        return Err(capability_refusal("icb_command_unsupported").with_detail(
            "an indirect draw replays the vertex_id shape; a pass that binds caller-held \
             vertex or index streams is not part of the first indirect increment",
        ));
    }
    objc::rc::autoreleasepool(|| {
        let attachments = attachment_textures(device, planned)?;
        encode_into_and_readback(device, queue, planned, &attachments, Some(*replay))
    })
}

/// The shared encoder body of the offscreen and present rails: build the
/// reviewed pipeline, render the pass into `targets`, wait for a terminal
/// command-buffer status, and read every attachment's texels back — the stored
/// depth and stencil surfaces' own included when the plan has them
/// (`research/docs/23` §3.3, v43/v49).
#[cfg(target_os = "macos")]
fn encode_into_and_readback(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    targets: &[Texture],
    indirect: Option<icb::IcbPlan>,
) -> Result<RenderReadback, ProviderError> {
    let pipeline = render_pipeline_state(device, planned)?;
    // A multisampled pass renders into its own four-sample textures
    // (`research/docs/23` §3.3, v51): one per colour location, created beside
    // the resolve targets `targets` already holds. The resolve targets are what
    // the readback below observes; the pass descriptor names both, and the
    // textures stay in this local for the same reason the depth and stencil
    // textures do — the encoder references them until it ends.
    let multisample_targets = planned
        .multisample
        .map(|_| multisample_attachment_textures(device, planned))
        .transpose()?;
    // The pass descriptor is autoreleased; it only has to outlive the
    // encoder creation below.
    let pass = MetalRenderPassDescriptor::new();
    // One descriptor entry per colour location: the pass's `colorAttachments`
    // array is indexed by location, so entry `i` carries attachment `i`'s
    // texture and its own load/store actions.
    for (index, (attachment, target)) in planned.attachments.iter().zip(targets).enumerate() {
        let color = pass
            .color_attachments()
            .object_at(index as u64)
            .ok_or_else(|| resource_refusal("metal_render_attachment_descriptor_unavailable"))?;
        // A multisampled location names *two* textures: the four-sample surface
        // the fragments land in and the single-sample resolve target — the
        // attachment's own texture, which is what the trace observes
        // (`research/docs/23` §3.3, v51). A single-sample location names one,
        // exactly as every pre-v51 pass did.
        let multisampled = multisample_targets
            .as_ref()
            .map(|textures| &textures[index]);
        color.set_texture(Some(multisampled.unwrap_or(target)));
        if let Some(_multisampled) = multisampled {
            color.set_resolve_texture(Some(target));
        }
        match attachment.load {
            RenderLoadAction::Clear(components) => {
                color.set_load_action(MTLLoadAction::Clear);
                color.set_clear_color(MTLClearColor::new(
                    components[0],
                    components[1],
                    components[2],
                    components[3],
                ));
            }
            RenderLoadAction::Load => color.set_load_action(MTLLoadAction::Load),
            // `MTLLoadAction::DontCare` mirrors the contract: the pre-pass
            // contents are undefined, the pass neither reads nor presets them,
            // and the draw alone defines what the attachment stores
            // (`research/docs/23` §3.1, v20).
            RenderLoadAction::DontCare => color.set_load_action(MTLLoadAction::DontCare),
        }
        // A discarded attachment still renders, but its bytes are not kept:
        // the store action is what makes the attachment disappear from the
        // observable surface, and the readback below skips it
        // (`research/docs/23` §3.6, v19).
        match (multisampled, attachment.store) {
            // The resolve is the only landing a multisampled location has:
            // `.multisampleResolve` writes the resolved texels into the resolve
            // texture and does not keep the four-sample surface, which is
            // exactly the "resolve into the attachment view" shape the trace
            // states (`research/docs/23` §3.3, v51).
            (Some(_), RenderStoreAction::Store) => {
                color.set_store_action(MTLStoreAction::MultisampleResolve)
            }
            (Some(_), RenderStoreAction::DontCare) => {
                color.set_store_action(MTLStoreAction::DontCare)
            }
            (None, RenderStoreAction::Store) => color.set_store_action(MTLStoreAction::Store),
            (None, RenderStoreAction::DontCare) => color.set_store_action(MTLStoreAction::DontCare),
        }
    }
    // The depth attachment is the rail's own texture, created when the plan
    // declares one (`research/docs/23` §3.3, v36): the pass descriptor opens it
    // with the plan's load operation and stores or discards it after the draw
    // exactly as the trace stated (v43). A storing surface is read back below.
    // The texture and the depth-stencil state stay in these locals until the
    // readback below: the pass descriptor and the encoder reference them, and
    // the rail's objects are autoreleased at the end of the pool.
    // A resolving pass names two depth textures: the four-sample surface the
    // fragments land in (private, never read back) and the single-sample
    // landing the resolve writes — the v43 shared-storage readback texture,
    // which is what the readback below observes (`research/docs/23` §3.3,
    // v57c). Every non-resolving shape keeps the one texture `depth_texture`
    // builds, exactly as before.
    // The combined depth-stencil shape (`research/docs/23` §3.3, v60) names
    // one four-sample `depth32Float_stencil8` texture from both attachment
    // descriptors, so the two targets below share the one surface instead of
    // the two single-face textures the other shapes build.
    let combined = planned.depth.is_some() && planned.stencil.is_some();
    let raster_samples = planned.multisample.map(|count| u64::from(count.samples()));
    let combined_target = if combined {
        planned
            .depth
            .as_ref()
            .map(|depth| {
                let samples = raster_samples.ok_or_else(|| {
                    capability_refusal("render_multisample_state_unsupported").with_detail(
                        "a combined depth-stencil surface states a multisampled raster",
                    )
                })?;
                combined_depth_stencil_surface(device, depth, samples)
            })
            .transpose()?
    } else {
        None
    };
    let depth_resolving = planned.depth_resolve.is_some();
    let depth_target = planned
        .depth
        .as_ref()
        .map(|depth| {
            if combined_target.is_some() {
                Ok(combined_target
                    .clone()
                    .expect("the combined target exists beside the combined plan"))
            } else if depth_resolving {
                let samples = raster_samples.ok_or_else(|| {
                    capability_refusal("render_multisample_state_unsupported")
                        .with_detail("a depth resolve states a multisampled raster")
                })?;
                multisample_depth_surface(device, depth, samples)
            } else {
                depth_texture(device, depth, planned.multisample)
            }
        })
        .transpose()?;
    let depth_resolve_target = if depth_resolving {
        planned
            .depth
            .as_ref()
            .map(|depth| depth_texture(device, depth, None))
            .transpose()?
    } else {
        None
    };
    if let (Some(depth), Some(texture)) = (&planned.depth, &depth_target) {
        let attachment = pass
            .depth_attachment()
            .ok_or_else(|| resource_refusal("metal_render_depth_descriptor_unavailable"))?;
        attachment.set_texture(Some(texture));
        match depth.clear_bits {
            Some(bits) => {
                attachment.set_load_action(MTLLoadAction::Clear);
                attachment.set_clear_depth(f64::from(f32::from_bits(bits)));
            }
            None => attachment.set_load_action(MTLLoadAction::Load),
        }
        // The store action is the pass's own statement
        // (`research/docs/23` §3.3, v43/v57c): a resolving stored surface
        // lands in its resolve target through `.multisampleResolve`, a stored
        // single-sample surface keeps its own texels, and every other shape
        // discards the rail-owned surface with the pass.
        attachment.set_store_action(match (depth.storing(), planned.depth_resolve) {
            (true, Some(_)) => MTLStoreAction::MultisampleResolve,
            (true, None) => MTLStoreAction::Store,
            (false, _) => MTLStoreAction::DontCare,
        });
        // The two depth-resolve fields are a `metal` 0.33 binding gap: the
        // depth attachment descriptor has no setter for them, so the rail
        // sends the two selectors itself. The filter ordinal is the contract's
        // code (`MTLMultisampleDepthResolveFilterSample0/Min/Max` = 0/1/2),
        // and the resolve target is the single-sample landing above.
        if let (Some(filter), Some(landing)) =
            (planned.depth_resolve, depth_resolve_target.as_ref())
        {
            unsafe {
                let _: () = msg_send![attachment, setResolveTexture: Some(landing.as_ref())];
                let _: () = msg_send![attachment, setDepthResolveFilter: u64::from(filter.code())];
            }
        }
    }
    // The stencil attachment is rail-owned in the same way (`research/docs/23`
    // §3.3, v47): created when the plan declares one, opened with the plan's
    // load operation and the pass's own clear value, and stored or discarded
    // after the draw exactly as the trace stated (v49). A storing surface has
    // to survive the pass for its texels to leave it, which is what the
    // readback below observes; every pre-v49 shape discards the rail-owned
    // surface with the pass. The texture stays in this local for the same
    // reason the depth texture does: the pass descriptor references it until
    // the encoder is done.
    let stencil_target = planned
        .stencil
        .as_ref()
        .map(|stencil| {
            if let Some(combined) = &combined_target {
                Ok(combined.clone())
            } else {
                stencil_texture(device, stencil, planned.multisample)
            }
        })
        .transpose()?;
    // The single-sample landing the stencil resolve writes into — the v49
    // shared-storage readback texture, which is what the readback below
    // observes (`research/docs/23` §3.3, v60). Every non-resolving shape
    // carries none.
    let stencil_resolve_target = if planned.stencil_resolve.is_some() {
        planned
            .stencil
            .as_ref()
            .map(|stencil| stencil_texture(device, stencil, None))
            .transpose()?
    } else {
        None
    };
    if let (Some(stencil), Some(texture)) = (&planned.stencil, &stencil_target) {
        let attachment = pass
            .stencil_attachment()
            .ok_or_else(|| resource_refusal("metal_render_stencil_descriptor_unavailable"))?;
        attachment.set_texture(Some(texture));
        match stencil.clear_value {
            Some(value) => {
                attachment.set_load_action(MTLLoadAction::Clear);
                attachment.set_clear_stencil(u32::from(value));
            }
            None => attachment.set_load_action(MTLLoadAction::Load),
        }
        attachment.set_store_action(match (stencil.storing(), planned.stencil_resolve) {
            (true, Some(_)) => MTLStoreAction::MultisampleResolve,
            (true, None) => MTLStoreAction::Store,
            (false, _) => MTLStoreAction::DontCare,
        });
        // The two stencil-resolve fields are a `metal` 0.33 binding gap: the
        // stencil attachment descriptor has no setter for them, so the rail
        // sends the two selectors itself. The resolve target is the base
        // class's `resolveTexture` — not a `stencilResolveTexture` selector,
        // which the v57c depth work showed does not exist — and the filter
        // ordinal is the contract's code
        // (`MTLMultisampleStencilResolveFilterSample0/DepthResolvedSample` =
        // 0/1).
        if let (Some(filter), Some(landing)) =
            (planned.stencil_resolve, stencil_resolve_target.as_ref())
        {
            unsafe {
                let _: () = msg_send![attachment, setResolveTexture: Some(landing.as_ref())];
                let _: () =
                    msg_send![attachment, setStencilResolveFilter: u64::from(filter.code())];
            }
        }
    }
    // Metal carries the depth and the stencil state in one descriptor
    // (`research/docs/23` §3.3, v36/v47): built when the pass opens either
    // surface, because a stencil-only pass has no depth test to state and a
    // depth-only pass no stencil state. The depth half is the pass's own test,
    // or `Always` with no write for an attachment nothing tests; the stencil
    // half arms the front and back descriptors with the same state, exactly as
    // the contract states it for both faces.
    let depth_stencil_state = (planned.depth.is_some() || planned.stencil.is_some()).then(|| {
        let descriptor = DepthStencilDescriptor::new();
        let (compare, write) = match planned.depth.as_ref().and_then(|depth| depth.test) {
            Some(test) => (
                match test.compare {
                    CompareFunction::Less => MTLCompareFunction::Less,
                    CompareFunction::Always => MTLCompareFunction::Always,
                },
                test.write,
            ),
            // An attachment with no test still clears; every fragment passes.
            // A pass with no depth attachment at all keeps these same defaults,
            // spelled out because the two halves travel in one object.
            None => (MTLCompareFunction::Always, false),
        };
        descriptor.set_depth_compare_function(compare);
        descriptor.set_depth_write_enabled(write);
        if let Some(test) = planned.stencil.as_ref().and_then(|stencil| stencil.test) {
            let stencil_descriptor = StencilDescriptor::new();
            stencil_descriptor.set_stencil_compare_function(metal_stencil_compare(test.compare));
            stencil_descriptor.set_stencil_failure_operation(metal_stencil_operation(test.fail_op));
            stencil_descriptor
                .set_depth_failure_operation(metal_stencil_operation(test.depth_fail_op));
            stencil_descriptor
                .set_depth_stencil_pass_operation(metal_stencil_operation(test.pass_op));
            stencil_descriptor.set_read_mask(u32::from(test.read_mask));
            stencil_descriptor.set_write_mask(u32::from(test.write_mask));
            descriptor.set_front_face_stencil(Some(&stencil_descriptor));
            descriptor.set_back_face_stencil(Some(&stencil_descriptor));
        }
        device.new_depth_stencil_state(&descriptor)
    });
    // The command buffer and the encoder are autoreleased and the rail is
    // synchronous, so neither has to be retained: nothing here outlives this
    // pool.
    let command = queue.new_command_buffer();
    let encoder = command.new_render_command_encoder(pass);
    encoder.set_render_pipeline_state(&pipeline);
    // Metal's depth state is encoder state (`research/docs/23` §3.3, v36): the
    // compare function and the write enable are the pass's own, exactly as the
    // contract states them.
    if let Some(state) = &depth_stencil_state {
        encoder.set_depth_stencil_state(state);
    }
    // The stencil reference value is encoder state too, not pipeline state
    // (`research/docs/23` §3.3, v47): Metal takes it per draw call, so the
    // pass's own value is set once before the draw exactly as the contract
    // states it.
    if let Some(test) = planned.stencil.as_ref().and_then(|stencil| stencil.test) {
        encoder.set_stencil_reference_value(u32::from(test.reference));
    }
    // Culling is encoder state too (`research/docs/23` §3.3, v39): the mode and
    // the winding are the pass's own, and a pass without the state keeps
    // Metal's defaults (cull none, counter-clockwise front) exactly.
    if let Some(cull) = &planned.cull {
        encoder.set_cull_mode(match cull.mode {
            ContractCullMode::None => MTLCullMode::None,
            ContractCullMode::Front => MTLCullMode::Front,
            ContractCullMode::Back => MTLCullMode::Back,
        });
        encoder.set_front_facing_winding(match cull.winding {
            ContractWinding::Clockwise => MTLWinding::Clockwise,
            ContractWinding::CounterClockwise => MTLWinding::CounterClockwise,
        });
    }
    // The vertex streams the plan resolved, bound at the same indices the
    // pipeline's `MTLVertexDescriptor` names. The MTLBuffers are kept for the
    // whole call: they have to outlive the encoder that reads them, and the
    // plan's bytes do not.
    let mut stream_buffers = Vec::with_capacity(planned.vertex_streams.len() + 1);
    for stream in &planned.vertex_streams {
        let offset = NSUInteger::try_from(stream.offset).unwrap_or(NSUInteger::MAX);
        let buffer = stream_buffer(device, stream.offset, stream.bytes)?;
        encoder.set_vertex_buffer(
            NSUInteger::from(stream.buffer_index),
            Some(buffer.as_ref()),
            offset,
        );
        stream_buffers.push(buffer);
    }
    // The viewport is explicit because the contract carries it, even though
    // the first increment only accepts the attachment-covering default. The
    // scissor below it is the pass's own rectangle when it declares one
    // (`research/docs/23` §3.3, v29).
    encoder.set_viewport(MTLViewport {
        originX: f64::from(planned.viewport[0]),
        originY: f64::from(planned.viewport[1]),
        width: f64::from(planned.viewport[2]),
        height: f64::from(planned.viewport[3]),
        znear: 0.0,
        zfar: 1.0,
    });
    let [scissor_x, scissor_y, scissor_width, scissor_height] =
        planned
            .scissor
            .unwrap_or([0, 0, planned.extent[0], planned.extent[1]]);
    encoder.set_scissor_rect(metal::MTLScissorRect {
        x: metal::NSUInteger::from(scissor_x),
        y: metal::NSUInteger::from(scissor_y),
        width: metal::NSUInteger::from(scissor_width),
        height: metal::NSUInteger::from(scissor_height),
    });
    match indirect {
        None => match &planned.indices {
            // An indexed draw names its index buffer in the draw call, which is
            // where Metal takes it: the contract's index binding becomes one
            // `drawIndexedPrimitives(indexCount:indexType:indexBuffer:
            // indexBufferOffset:)`, with the count the pass carries in the
            // indexed shape.
            Some(indices) => {
                let offset = NSUInteger::try_from(indices.offset).unwrap_or(NSUInteger::MAX);
                let buffer = stream_buffer(device, indices.offset, indices.bytes)?;
                if indices.base_vertex == 0 {
                    encoder.draw_indexed_primitives_instanced(
                        MTLPrimitiveType::Triangle,
                        u64::from(indices.index_count),
                        metal_index_type(indices.format),
                        buffer.as_ref(),
                        offset,
                        u64::from(planned.instance_count),
                    );
                } else {
                    // The offset belongs to the draw call
                    // (`research/docs/23` §3.3, v34): the base-vertex entry
                    // states it beside the instance count, and a zero-offset
                    // draw keeps the pre-v34 entry point exactly.
                    let base_vertex =
                        NSInteger::try_from(indices.base_vertex).unwrap_or(NSInteger::MAX);
                    encoder.draw_indexed_primitives_instanced_base_instance(
                        MTLPrimitiveType::Triangle,
                        u64::from(indices.index_count),
                        metal_index_type(indices.format),
                        buffer.as_ref(),
                        offset,
                        u64::from(planned.instance_count),
                        base_vertex,
                        0,
                    );
                }
                stream_buffers.push(buffer);
            }
            None => {
                // The instanced draw call (`research/docs/23` §3.3, v31): the
                // same entry point with the pass's second count, which is `1`
                // for every pre-v31 pass and therefore byte-identical to the
                // single-instance call it replaces.
                encoder.draw_primitives_instanced(
                    MTLPrimitiveType::Triangle,
                    0,
                    u64::from(planned.vertices),
                    u64::from(planned.instance_count),
                )
            }
        },
        Some(replay) => {
            let icb::IcbCommand::Draw {
                vertex_count,
                instance_count,
            } = replay.command
            else {
                return Err(capability_refusal("icb_command_unsupported")
                    .with_field("kind", FieldValue::Text("dispatch".to_owned()))
                    .with_detail("the first indirect increment replays non-indexed draws only"));
            };
            let descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.set_command_types(MTLIndirectCommandType::Draw);
            let buffer = device.new_indirect_command_buffer_with_descriptor(
                &descriptor,
                u64::from(replay.max_commands),
                MTLResourceOptions::StorageModeShared,
            );
            if buffer.as_ptr().is_null() {
                return Err(resource_refusal(
                    "metal_indirect_command_buffer_allocation_failed",
                ));
            }
            let command = buffer.indirect_render_command_at_index(u64::from(replay.range.start));
            command.set_render_pipeline_state(&pipeline);
            command.draw_primitives(
                MTLPrimitiveType::Triangle,
                0,
                u64::from(vertex_count),
                u64::from(instance_count),
                0,
            );
            encoder.execute_commands_in_buffer(
                &buffer,
                NSRange::new(u64::from(replay.range.start), u64::from(replay.range.count)),
            );
        }
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        // `MTLCommandBufferStatus` is a `#[repr(u32)]` enum, so the numeric
        // status is the diagnostic a field-less error can carry.
        let status = command.status() as u32;
        return Err(resource_refusal("metal_render_command_failed")
            .with_detail(format!("command buffer ended with status {status}")));
    }
    let attachments = planned
        .attachments
        .iter()
        .zip(targets)
        .filter(|(attachment, _)| attachment.store == RenderStoreAction::Store)
        .map(|(_, target)| read_texels(target, planned))
        .collect::<Result<Vec<Vec<u8>>, ProviderError>>()?;
    // The stored depth surface's texels leave through the same `getBytes` shape
    // the colour attachments use (`research/docs/23` §3.3, v43): the texture is
    // shared storage exactly when the pass stores it (`depth_texture`), and the
    // region is the pass's own extent, which the contract holds to the colour
    // attachments'. A discarded or absent surface keeps its bytes on the
    // device and reads nothing back.
    // A resolving pass observes the resolve target — the single-sample landing
    // `depth_resolve_target` holds — while every non-resolving stored surface
    // is read back from its own texture (`research/docs/23` §3.3, v43/v57c).
    let depth = match (&planned.depth, &depth_target, &depth_resolve_target) {
        (Some(depth), _, Some(landing)) if depth.storing() => Some(read_texels(landing, planned)?),
        (Some(depth), Some(texture), None) if depth.storing() => {
            Some(read_texels(texture, planned)?)
        }
        _ => None,
    };
    // The stored stencil surface's texels leave through the same `getBytes`
    // shape one byte wide (`research/docs/23` §3.3, v49): the texture is shared
    // storage exactly when the pass stores it (`stencil_texture`), and the
    // region is the pass's own extent, which the contract holds to the stencil
    // surface's just as it holds the depth surface's. A discarded or absent
    // surface keeps its bytes on the device and reads nothing back.
    let stencil = match (&planned.stencil, &stencil_target, &stencil_resolve_target) {
        // A resolving pass observes the resolve target — the single-sample
        // landing `stencil_resolve_target` holds — while every non-resolving
        // stored surface is read back from its own texture
        // (`research/docs/23` §3.3, v49/v60).
        (Some(stencil), _, Some(landing)) if stencil.storing() => {
            Some(read_stencil_texels(landing, planned)?)
        }
        (Some(stencil), Some(texture), None) if stencil.storing() => {
            Some(read_stencil_texels(texture, planned)?)
        }
        _ => None,
    };
    Ok(RenderReadback {
        attachments,
        depth,
        stencil,
    })
}

/// The colour attachments this rail renders into, one texture per location.
///
/// `usage = RenderTarget` states what the texture is for, and the shared storage
/// mode is what makes the texels CPU-visible for the readback on the
/// unified-memory device the provider admits — the same reason the sampled
/// texture rail uses shared storage (`research/docs/16` §4.8).
#[cfg(target_os = "macos")]
fn attachment_textures(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<Vec<Texture>, ProviderError> {
    let mut textures = Vec::with_capacity(planned.attachments.len());
    for attachment in &planned.attachments {
        let texture = present_target_texture(device, attachment.format, planned.extent)?;
        if let Some(bytes) = attachment.initial {
            upload_texels(&texture, planned, bytes);
        }
        textures.push(texture);
    }
    Ok(textures)
}

/// The multisampled surfaces one multisampled pass renders into
/// (`research/docs/23` §3.3, v51/v61), one per colour location.
///
/// The textures are render targets with shared storage for the same reason
/// every attachment texture is: the rail is synchronous and the resolved
/// texels, not these, are what leaves through the readback. The sample count is
/// the one the plan carried — the pass's own 2x/4x/8x raster — so the raster
/// state and the textures cannot disagree.
#[cfg(target_os = "macos")]
fn multisample_attachment_textures(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<Vec<Texture>, ProviderError> {
    let samples = match planned.multisample {
        Some(count @ (SampleCount::Two | SampleCount::Four | SampleCount::Eight)) => {
            u64::from(count.samples())
        }
        _ => {
            return Err(
                capability_refusal("render_multisample_state_unsupported").with_detail(
                    "the multisample increment creates two-, four- and eight-sample surfaces \
                     only",
                ),
            );
        }
    };
    let mut textures = Vec::with_capacity(planned.attachments.len());
    for attachment in &planned.attachments {
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_pixel_format(metal_pixel_format(attachment.format));
        descriptor.set_width(u64::from(planned.extent[0]));
        descriptor.set_height(u64::from(planned.extent[1]));
        descriptor.set_mipmap_level_count(1);
        descriptor.set_sample_count(samples);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let pointer: *mut metal::MTLTexture =
            unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
        if pointer.is_null() {
            return Err(resource_refusal(
                "metal_render_multisample_target_allocation_failed",
            ));
        }
        textures.push(unsafe { Texture::from_ptr(pointer) });
    }
    Ok(textures)
}

/// One render target texture, built for an offscreen attachment or a present
/// target. A present target is created once and then reused across submissions,
/// which is why this creation is split from the initial upload
/// (`research/docs/24` §6 Step 7).
#[cfg(target_os = "macos")]
pub(crate) fn present_target_texture(
    device: &Device,
    format: RenderPixelFormat,
    extent: [u32; 2],
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(metal_pixel_format(format));
    descriptor.set_width(u64::from(extent[0]));
    descriptor.set_height(u64::from(extent[1]));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_target_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// One contract blend factor as the `MTLBlendFactor` it names.
#[cfg(target_os = "macos")]
const fn metal_blend_factor(factor: BlendFactor) -> MTLBlendFactor {
    match factor {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SourceAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
    }
}

/// One contract blend operation as the `MTLBlendOperation` it names.
#[cfg(target_os = "macos")]
const fn metal_blend_operation(operation: BlendOperation) -> MTLBlendOperation {
    match operation {
        BlendOperation::Add => MTLBlendOperation::Add,
    }
}

/// One contract stencil comparison as the `MTLCompareFunction` it names
/// (`research/docs/23` §3.3, v47).
///
/// The two admitted values are the ones both APIs spell identically, so the
/// mapping is total over the contract's own list — the same shape the Vulkan
/// rail's `vk_stencil_compare` has.
#[cfg(target_os = "macos")]
const fn metal_stencil_compare(compare: StencilCompare) -> MTLCompareFunction {
    match compare {
        StencilCompare::Equal => MTLCompareFunction::Equal,
        StencilCompare::Always => MTLCompareFunction::Always,
    }
}

/// One contract stencil operation as the `MTLStencilOperation` it names
/// (`research/docs/23` §3.3, v47).
#[cfg(target_os = "macos")]
const fn metal_stencil_operation(operation: StencilOp) -> MTLStencilOperation {
    match operation {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
    }
}

/// The rail-owned depth texture of a pass that declares one
/// (`research/docs/23` §3.3, v36/v43).
///
/// Shared storage when the pass stores the surface, because `getBytes` reads no
/// `Private` texture and the readback is what makes the stored texels
/// observable — the same reason the colour attachments are shared
/// (`research/docs/16` §4.8). The pre-v43 shapes keep `Private`: nothing reads
/// those texels back, and the rail-owned surface disappears with the pass.
#[cfg(target_os = "macos")]
fn depth_texture(
    device: &Device,
    depth: &PlannedDepth,
    multisample: Option<SampleCount>,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    // A multisampled pass creates its depth surface with the raster's own
    // sample count (`research/docs/23` §3.3, v53/v61): Metal refuses an
    // encoder whose depth texture's sample count disagrees with
    // `rasterSampleCount`, so the two come from one decision. The surface
    // stays `Private` because a multisampled depth surface is rail-owned in
    // this increment — keeping it would need the depth resolve filter the
    // increment after this one reviews.
    if let Some(multisample) = multisample {
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_sample_count(u64::from(multisample.samples()));
    } else {
        descriptor.set_texture_type(MTLTextureType::D2);
    }
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(if depth.storing() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Private
    });
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_depth_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The multisampled depth surface a resolving pass renders into
/// (`research/docs/23` §3.3, v57c/v61).
///
/// The resolve's landing is the single-sample shared texture
/// [`depth_texture`] builds for a stored surface, so this surface is never
/// read back and stays private — the same reason the four-sample colour
/// surfaces are private in the Swift oracle. Metal refuses an encoder whose
/// depth texture's sample count disagrees with `rasterSampleCount`, so the two
/// come from one decision exactly as the non-resolving multisampled surface
/// does. The count is the raster's own, carried in from the plan.
#[cfg(target_os = "macos")]
fn multisample_depth_surface(
    device: &Device,
    depth: &PlannedDepth,
    samples: u64,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2Multisample);
    descriptor.set_sample_count(samples);
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal(
            "metal_render_multisample_depth_allocation_failed",
        ));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The rail-owned stencil texture of a pass that declares one
/// (`research/docs/23` §3.3, v47/v49).
///
/// Shared storage when the pass stores the surface, for the same reason a
/// storing depth surface is shared: `getBytes` reads no `Private` texture, and
/// the readback is what makes the stored texels observable
/// (`research/docs/16` §4.8). Every pre-v49 shape keeps `Private`: the stencil
/// fixture observes a discarded surface's effect through the colour attachment
/// the mask decides, and the rail-owned surface disappears with the pass.
#[cfg(target_os = "macos")]
fn stencil_texture(
    device: &Device,
    stencil: &PlannedStencil,
    multisample: Option<SampleCount>,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    // A multisampled pass creates its stencil surface with the raster's own
    // sample count (`research/docs/23` §3.3, v55/v61), exactly as the depth
    // surface does: Metal refuses an encoder whose attachment disagrees with
    // `rasterSampleCount`. The surface stays `Private` because a multisampled
    // stencil surface is rail-owned in this increment — keeping it would need
    // the stencil resolve the increment after this one reviews.
    if let Some(multisample) = multisample {
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_sample_count(u64::from(multisample.samples()));
    } else {
        descriptor.set_texture_type(MTLTextureType::D2);
    }
    descriptor.set_pixel_format(MTLPixelFormat::Stencil8);
    descriptor.set_width(u64::from(stencil.width));
    descriptor.set_height(u64::from(stencil.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(if stencil.storing() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Private
    });
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_stencil_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The combined depth-stencil surface of a pass that opens both faces
/// (`research/docs/23` §3.3, v60/v61).
///
/// Metal binds one texture to both attachment descriptors, so the combined
/// shape creates one multisampled `depth32Float_stencil8` surface the depth and
/// stencil halves share; the resolve landings below are separate single-sample
/// textures. The surface is private — its texels leave through the resolves,
/// never through `getBytes`.
#[cfg(target_os = "macos")]
fn combined_depth_stencil_surface(
    device: &Device,
    depth: &PlannedDepth,
    samples: u64,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2Multisample);
    descriptor.set_sample_count(samples);
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float_Stencil8);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal(
            "metal_render_combined_depth_stencil_allocation_failed",
        ));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// One MTLBuffer holding a stream view's bytes, for a vertex or index binding.
///
/// The image is the view's bytes placed at the view's own offset inside its
/// allocation, and the binding uses that same offset — the convention the
/// compute pool's merged images follow (`native.rs`: an allocation image is
/// bound at `view.offset`). For the reviewed fixture the offset is zero, so the
/// image is exactly the declared bytes; for a view that starts above the
/// allocation's first byte the stream still reads the byte range the trace
/// named instead of being silently re-based at zero.
#[cfg(target_os = "macos")]
fn stream_buffer(device: &Device, offset: u64, bytes: &[u8]) -> Result<Buffer, ProviderError> {
    let start = usize::try_from(offset)
        .map_err(|_| resource_refusal("metal_render_stream_offset_overflow"))?;
    let end = start
        .checked_add(bytes.len())
        .ok_or_else(|| resource_refusal("metal_render_stream_offset_overflow"))?;
    let mut image = vec![0_u8; end];
    image[start..end].copy_from_slice(bytes);
    let pointer: *mut metal::MTLBuffer = unsafe {
        msg_send![device.as_ref(),
            newBufferWithBytes:image.as_ptr().cast::<std::ffi::c_void>()
            length:image.len()
            options:MTLResourceOptions::StorageModeShared]
    };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_stream_buffer_failed"));
    }
    Ok(unsafe { Buffer::from_ptr(pointer) })
}

/// Upload tightly packed texels into a texture's whole extent.
///
/// `replace_region` takes the source stride and owns the texture-side layout,
/// so this upload cannot repeat the Vulkan rail's defect: there the host had to
/// guess the destination row pitch, while Metal keeps that distance inside the
/// driver (`research/docs/16` §4.8). The present path uses this for the
/// sentinel preset that makes "the present never happened" falsifiable
/// (`research/docs/24` §3.1).
#[cfg(target_os = "macos")]
pub(crate) fn upload_texels(texture: &Texture, planned: &RenderPlan<'_>, bytes: &[u8]) {
    texture.replace_region(
        region(planned),
        0,
        bytes.as_ptr().cast(),
        NSUInteger::try_from(planned.row_pitch).unwrap_or(NSUInteger::MAX),
    );
}

/// The two-stage pipeline state of the reviewed module.
#[cfg(target_os = "macos")]
fn render_pipeline_state(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<RenderPipelineState, ProviderError> {
    let options = CompileOptions::new();
    // The plan's own module, not the milestone's: a vertex-input pipeline is
    // built from `quad_indexed_2x2.metal`, and compiling the `vertex_id` module
    // for it would fail on the entry name rather than run the reviewed quad
    // (`research/docs/23` §3.3).
    let library = device
        .new_library_with_source(planned.source, &options)
        .map_err(|error| {
            compile_refusal("metal_render_library_compile_failed").with_detail(error)
        })?;
    let vertex = library
        .get_function(planned.vertex_entry, None)
        .map_err(|error| {
            compile_refusal("metal_render_vertex_function_missing").with_detail(error)
        })?;
    let fragment = library
        .get_function(planned.fragment_entry, None)
        .map_err(|error| {
            compile_refusal("metal_render_fragment_function_missing").with_detail(error)
        })?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(vertex.as_ref()));
    descriptor.set_fragment_function(Some(fragment.as_ref()));
    // The pipeline's raster sample count follows the pass's own multisample
    // state (`research/docs/23` §3.3, v51/v61): Metal refuses a pipeline whose
    // `rasterSampleCount` disagrees with the attachments the encoder binds, so
    // the two come from one decision. A single-sample pass keeps the default
    // exactly as every pre-v51 pass did.
    if let Some(multisample) = planned.multisample {
        descriptor.set_raster_sample_count(u64::from(multisample.samples()));
    }
    // The vertex descriptor is what makes `[[attribute(n)]]` mean a byte range
    // of a bound stream: the MSL module names the attribute locations, the
    // descriptor says which binding, stride, offset and format each one reads.
    // A `VertexLayout::None` pipeline carries none — its vertex stage takes
    // `vertex_id` and reads no stream — which is why the descriptor is built
    // only from a plan that has streams.
    if !planned.vertex_streams.is_empty() {
        let vertex_descriptor = VertexDescriptor::new();
        for stream in &planned.vertex_streams {
            let layout = vertex_descriptor
                .layouts()
                .object_at(NSUInteger::from(stream.buffer_index))
                .ok_or_else(|| {
                    resource_refusal("metal_render_vertex_layout_descriptor_unavailable")
                })?;
            layout.set_stride(NSUInteger::try_from(stream.stride).unwrap_or(NSUInteger::MAX));
            // The binding's own step function (`research/docs/23` §3.3, v31).
            // A per-instance stream advances once per instance, which is
            // Metal's default `stepRate` of one; a per-vertex stream keeps the
            // pre-v31 behavior exactly.
            layout.set_step_function(metal_vertex_step(stream.step));
            if stream.step == RenderVertexStep::PerInstance {
                layout.set_step_rate(1);
            }
            for attribute in &stream.attributes {
                let target = vertex_descriptor
                    .attributes()
                    .object_at(NSUInteger::from(attribute.location))
                    .ok_or_else(|| {
                        resource_refusal("metal_render_vertex_attribute_descriptor_unavailable")
                    })?;
                target.set_format(metal_vertex_format(attribute.format));
                target
                    .set_offset(NSUInteger::try_from(attribute.offset).unwrap_or(NSUInteger::MAX));
                target.set_buffer_index(NSUInteger::from(stream.buffer_index));
            }
        }
        descriptor.set_vertex_descriptor(Some(vertex_descriptor));
    }
    // A pass that opens a depth attachment compiles its pipeline against that
    // attachment's format; a pre-v36 pass declares none
    // (`research/docs/23` §3.3, v36).
    // The combined depth-stencil shape compiles both slots against the one
    // `depth32Float_stencil8` format the two attachment descriptors share
    // (`research/docs/23` §3.3, v60); the single-face shapes keep their own
    // formats.
    let combined = planned.depth.is_some() && planned.stencil.is_some();
    if planned.depth.is_some() {
        descriptor.set_depth_attachment_pixel_format(if combined {
            MTLPixelFormat::Depth32Float_Stencil8
        } else {
            MTLPixelFormat::Depth32Float
        });
    }
    // The stencil slot is the same rule one surface over
    // (`research/docs/23` §3.3, v47): a pass that opens a stencil attachment
    // compiles its pipeline against that attachment's format, and a pass
    // without one leaves the slot at its default exactly as before.
    if planned.stencil.is_some() {
        descriptor.set_stencil_attachment_pixel_format(if combined {
            MTLPixelFormat::Depth32Float_Stencil8
        } else {
            MTLPixelFormat::Stencil8
        });
    }
    // One pipeline attachment per colour location: entry `i` states the pixel
    // format the reviewed fragment's output `i` is compiled against, which the
    // plan already forced to agree with the pass's attachment list
    // (`research/docs/23` §3.3). A pass that states blend state states it here,
    // per location, exactly as Metal's colour attachment descriptor indexes it
    // (`research/docs/23` §3.3, v40).
    for (index, attachment) in planned.attachments.iter().enumerate() {
        let color = descriptor
            .color_attachments()
            .object_at(index as u64)
            .ok_or_else(|| resource_refusal("metal_render_pipeline_attachment_unavailable"))?;
        color.set_pixel_format(metal_pixel_format(attachment.format));
        if let Some(blend) = planned
            .blend
            .as_ref()
            .and_then(|blend| blend.attachments.get(index))
        {
            color.set_blending_enabled(true);
            color.set_rgb_blend_operation(metal_blend_operation(blend.operation));
            color.set_alpha_blend_operation(metal_blend_operation(blend.operation));
            color.set_source_rgb_blend_factor(metal_blend_factor(blend.source_rgb));
            color.set_destination_rgb_blend_factor(metal_blend_factor(blend.destination_rgb));
            color.set_source_alpha_blend_factor(metal_blend_factor(blend.source_alpha));
            color.set_destination_alpha_blend_factor(metal_blend_factor(blend.destination_alpha));
        }
    }
    device
        .new_render_pipeline_state(descriptor.as_ref())
        .map_err(|error| compile_refusal("metal_render_pipeline_compile_failed").with_detail(error))
}

/// Read the attachment's texels back into host memory, tightly packed.
#[cfg(target_os = "macos")]
fn read_texels(texture: &Texture, planned: &RenderPlan<'_>) -> Result<Vec<u8>, ProviderError> {
    let mut texels = vec![0_u8; planned.texel_bytes];
    texture.get_bytes(
        texels.as_mut_ptr().cast(),
        NSUInteger::try_from(planned.row_pitch).unwrap_or(NSUInteger::MAX),
        region(planned),
        0,
    );
    Ok(texels)
}

/// Read the stored stencil surface back into host memory, tightly packed.
///
/// `Stencil8` carries one byte per texel — `metal-api-core`'s
/// `STENCIL_BYTES_PER_TEXEL` — so this surface's flat byte extent is the texel
/// extent itself rather than the four-byte-per-texel arithmetic the colour and
/// depth readbacks share, and one texture row is `width` bytes wide
/// (`research/docs/23` §3.3, v49). The region is the pass's own extent, which
/// the contract holds to the stencil surface's exactly as it holds it to every
/// colour attachment's.
#[cfg(target_os = "macos")]
fn read_stencil_texels(
    texture: &Texture,
    planned: &RenderPlan<'_>,
) -> Result<Vec<u8>, ProviderError> {
    let texel_bytes = u64::from(planned.extent[0]).saturating_mul(u64::from(planned.extent[1]));
    let Ok(bytes) = usize::try_from(texel_bytes) else {
        return Err(capability_refusal("attachment_dimension_limit"));
    };
    let mut texels = vec![0_u8; bytes];
    texture.get_bytes(
        texels.as_mut_ptr().cast(),
        NSUInteger::try_from(u64::from(planned.extent[0])).unwrap_or(NSUInteger::MAX),
        region(planned),
        0,
    );
    Ok(texels)
}

#[cfg(target_os = "macos")]
fn region(planned: &RenderPlan<'_>) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: u64::from(planned.extent[0]),
            height: u64::from(planned.extent[1]),
            depth: 1,
        },
    }
}

#[cfg(target_os = "macos")]
const fn metal_pixel_format(format: RenderPixelFormat) -> MTLPixelFormat {
    match format {
        RenderPixelFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        RenderPixelFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        RenderPixelFormat::R32Float => MTLPixelFormat::R32Float,
    }
}

/// The `MTLVertexFormat` one planned attribute format names.
#[cfg(target_os = "macos")]
const fn metal_vertex_format(format: RenderVertexFormat) -> MTLVertexFormat {
    match format {
        RenderVertexFormat::Float2 => MTLVertexFormat::Float2,
        RenderVertexFormat::Float3 => MTLVertexFormat::Float3,
        RenderVertexFormat::Float4 => MTLVertexFormat::Float4,
        // The contract's `uint32` is Metal's scalar `UInt`, not `UInt2`: the
        // attribute is one 32-bit unsigned integer, and `UInt` is the format
        // whose width matches `VertexFormat::Uint32::bytes()`.
        RenderVertexFormat::Uint => MTLVertexFormat::UInt,
    }
}

/// The `MTLVertexStepFunction` one planned step names
/// (`research/docs/23` §3.3, v31).
#[cfg(target_os = "macos")]
const fn metal_vertex_step(step: RenderVertexStep) -> MTLVertexStepFunction {
    match step {
        RenderVertexStep::PerVertex => MTLVertexStepFunction::PerVertex,
        RenderVertexStep::PerInstance => MTLVertexStepFunction::PerInstance,
    }
}

/// The `MTLIndexType` one planned index width names.
#[cfg(target_os = "macos")]
const fn metal_index_type(format: RenderIndexType) -> MTLIndexType {
    match format {
        RenderIndexType::Uint16 => MTLIndexType::UInt16,
        RenderIndexType::Uint32 => MTLIndexType::UInt32,
    }
}

#[cfg(target_os = "macos")]
fn resource_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Resource, slug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AcquirePolicy, AliasMode, AllocationId, AllocationRecord, BufferAccess,
        BufferBindingContract, BufferSource, CompareFunction, CompiledComputePipeline,
        CompletionPolicy, ComputePass, DepthFormat, DepthLoadOp, DepthStoreOp, DeviceEpoch,
        Dispatch, DispatchKind, DispatchType, FootprintProof, FunctionIdentity, FunctionSource,
        IndirectCommandBufferDescriptor, IndirectCommandKind, IndirectCommandPayload,
        IndirectCommandRange, InitialState, LeaseId, MultisampleDepthResolve, MultisampleState,
        MultisampleStencilResolve, OperationId, PipelineContract, PresentTarget,
        ProviderCapabilities, RenderAttachment, RenderDepthAttachment, RenderDepthIdentity,
        RenderStencilAttachment, RenderStencilIdentity, ResourceTableSnapshot, SemanticDigest,
        StencilCompare, StencilFormat, StencilLoadOp, StencilOp, StencilResolveFilter, StencilTest,
        StorageMode, VertexAttribute, VertexBufferLayout, VertexLayout, ViewId,
        PROVIDER_SCHEMA_VERSION,
    };

    /// The texels the reviewed fragment writes, as `MTLClearColor` components.
    const EXPECTED_TEXEL_COMPONENTS: [f64; 4] = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0];

    /// The milestone's pass: one 2x2 `Rgba8Unorm` attachment, stored, and drawn
    /// as the full-screen triangle.
    fn milestone_pass(load: LoadOp) -> RenderPassDescriptor {
        RenderPassDescriptor {
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            base_vertex: 0,
            pipeline: PipelineId::new(3),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(7),
                allocation_id: AllocationId::new(9),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load,
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            scissor: None,
            vertices: 3,
            vertex_buffers: Vec::new(),
            indices: None,
            instance_count: 1,
            textures: Vec::new(),
            present: None,
        }
    }

    fn milestone_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: VERTEX_ENTRY.to_owned(),
            fragment_entry: FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
        }
    }

    /// The sentinel `LoadOp::Clear` leaves behind. Every channel has to differ
    /// from the fragment's texel, or "the pass never ran" would read back as a
    /// pass.
    fn sentinel() -> ClearColor {
        ClearColor::new([0xfe, 0xfe, 0xfe, 0xfe])
    }

    fn milestone_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
        initial: Option<&'a [u8]>,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_SOURCE,
            initial: vec![initial],
        }
    }

    /// [`plan`] for the device-level helper's shape.
    ///
    /// The helper has no trace, but a render input carries its own bytes, so
    /// this is the same call the trace path makes; only the attachment's landing
    /// view belongs to [`plan_trace`].
    fn plan_pass<'a>(
        request: &OffscreenRenderRequest<'a>,
    ) -> Result<RenderPlan<'a>, ProviderError> {
        plan(request, 0, 0)
    }

    /// The refusal of a (source, entry pair) triple the rail does not review.
    fn allowlist_refusal(source: &str, vertex: &str, fragment: &str) -> ProviderError {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = RenderPipelineContract {
            vertex_entry: vertex.to_owned(),
            fragment_entry: fragment.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
        };
        let request = OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source,
            initial: vec![None],
        };
        plan_pass(&request).map(|_| ()).unwrap_err()
    }

    #[test]
    fn plan_accepts_the_milestone_shape_and_fixes_the_readback_extent() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap();
        assert_eq!(planned.vertex_entry, VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, FRAGMENT_ENTRY);
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(planned.viewport, [0, 0, 2, 2]);
        assert_eq!(planned.vertices, 3);
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert_eq!(attachment.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(attachment.store, RenderStoreAction::Store);
        // 2x2 texels of a 4-byte format: 16 bytes, two rows of 8.
        assert_eq!(planned.texel_bytes, 16);
        assert_eq!(planned.row_pitch, 8);
        assert_eq!(attachment.initial, None);
        let RenderLoadAction::Clear(components) = attachment.load else {
            panic!("the milestone clears its attachment");
        };
        assert_eq!(components, [254.0 / 255.0; 4]);
    }

    #[test]
    fn plan_refuses_an_unreviewed_source_or_entry() {
        let edited = REVIEWED_SOURCE.replace("192.0", "191.0");
        assert_ne!(edited, REVIEWED_SOURCE, "the fixture still writes 192.0");
        for (label, error) in [
            (
                "edited source",
                allowlist_refusal(&edited, VERTEX_ENTRY, FRAGMENT_ENTRY),
            ),
            (
                "wrong vertex entry",
                allowlist_refusal(REVIEWED_SOURCE, "render_triangle", FRAGMENT_ENTRY),
            ),
            (
                "wrong fragment entry",
                allowlist_refusal(REVIEWED_SOURCE, VERTEX_ENTRY, "render_solid"),
            ),
        ] {
            assert_eq!(error.slug, "native_render_source_not_reviewed", "{label}");
            assert_eq!(error.class, ProviderErrorClass::Capability, "{label}");
            assert_eq!(error.phase, ProviderPhase::Compile, "{label}");
        }
    }

    #[test]
    fn reviewed_fixture_matches_the_expected_texel_bytes() {
        for literal in ["64.0 / 255.0", "128.0 / 255.0", "192.0 / 255.0"] {
            assert!(
                REVIEWED_SOURCE.contains(literal),
                "the reviewed fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_SOURCE.contains("0.5"),
            "the reviewed fixture must not carry a half-integer tie constant"
        );
        for (component, byte) in EXPECTED_TEXEL_COMPONENTS.iter().zip(EXPECTED_TEXEL_BYTES) {
            // A byte/255 constant is far from a tie, so every driver lands the
            // same byte (`research/docs/23` §3.5).
            assert_eq!((component * 255.0).round(), f64::from(byte));
        }
    }

    #[test]
    fn attachment_formats_map_to_the_admitted_pixel_formats() {
        for (format, expected) in [
            (AttachmentFormat::Rgba8Unorm, RenderPixelFormat::Rgba8Unorm),
            (AttachmentFormat::Bgra8Unorm, RenderPixelFormat::Bgra8Unorm),
            (AttachmentFormat::R32Float, RenderPixelFormat::R32Float),
        ] {
            assert_eq!(pixel_format(format).unwrap(), expected);
            assert!(SUPPORTED_COLOR_FORMATS.contains(&format));
        }
        assert_eq!(RenderPixelFormat::Rgba8Unorm.name(), "rgba8_unorm");
        assert_eq!(RenderPixelFormat::Bgra8Unorm.name(), "bgra8_unorm");
        assert_eq!(RenderPixelFormat::R32Float.name(), "r32_float");
    }

    #[test]
    fn r32uint_is_refused_as_a_colour_attachment() {
        let error = pixel_format(AttachmentFormat::R32Uint).unwrap_err();
        assert_eq!(error.slug, "attachment_format_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert!(!SUPPORTED_COLOR_FORMATS.contains(&AttachmentFormat::R32Uint));
    }

    #[test]
    fn clear_components_decode_by_format_byte_order() {
        // One colour, two byte strings: memory order is R,G,B,A for the first
        // format and B,G,R,A for the second.
        assert_eq!(
            clear_components(
                ClearColor::new([0x40, 0x80, 0xc0, 0xff]),
                RenderPixelFormat::Rgba8Unorm
            ),
            EXPECTED_TEXEL_COMPONENTS
        );
        assert_eq!(
            clear_components(
                ClearColor::new([0xc0, 0x80, 0x40, 0xff]),
                RenderPixelFormat::Bgra8Unorm
            ),
            EXPECTED_TEXEL_COMPONENTS
        );
        // The single-channel format's four bytes are the red component itself.
        assert_eq!(
            clear_components(
                ClearColor::new(0.5_f32.to_le_bytes()),
                RenderPixelFormat::R32Float
            ),
            [0.5, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn clear_load_and_dont_care_map_to_distinct_load_actions() {
        assert_eq!(
            load_action(
                LoadOp::Clear(ClearColor::new([0, 0, 0, 0xff])),
                RenderPixelFormat::Rgba8Unorm
            )
            .unwrap(),
            RenderLoadAction::Clear([0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            load_action(LoadOp::Load, RenderPixelFormat::Rgba8Unorm).unwrap(),
            RenderLoadAction::Load
        );
        assert_eq!(
            load_action(LoadOp::DontCare, RenderPixelFormat::Rgba8Unorm).unwrap(),
            RenderLoadAction::DontCare
        );
    }

    #[test]
    fn store_dont_care_is_admitted() {
        assert_eq!(
            store_action(StoreOp::Store).unwrap(),
            RenderStoreAction::Store
        );
        assert_eq!(
            store_action(StoreOp::DontCare).unwrap(),
            RenderStoreAction::DontCare
        );
    }

    #[test]
    fn plan_refuses_an_attachment_beyond_the_declared_extent() {
        // The rail executes extents up to `MAX_ATTACHMENT_DIMENSION` (four
        // texels per axis from v27); one axis beyond that is refused rather
        // than silently clamped.
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [0, 0, 5, 5];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = 5;
        attachment.height = 5;
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "attachment_dimension_limit");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn plan_accepts_a_four_by_four_attachment() {
        // The extent ceiling the capability reports is executable: a 4x4
        // attachment plans with sixty-four texel bytes.
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [0, 0, 4, 4];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = 4;
        attachment.height = 4;
        let pipeline = milestone_pipeline();
        let plan = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect("a four-by-four attachment is within the declared extent");
        assert_eq!(plan.extent, [4, 4]);
        assert_eq!(plan.texel_bytes, 64);
    }

    #[test]
    fn plan_refuses_a_pipeline_whose_format_disagrees_with_the_attachment() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let mut pipeline = milestone_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::Bgra8Unorm];
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
    }

    /// The reviewed dual shape plans two attachments, one per colour location,
    /// with the second location's texel distinct from the first's.
    #[test]
    fn plan_accepts_the_dual_shape_and_plans_two_attachments() {
        let pass = dual_pass();
        let pipeline = dual_pipeline();
        let planned = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap();
        assert_eq!(
            planned.module_path,
            "conformance/shaders/quad_indexed_2x2_dual.metal"
        );
        assert_eq!(planned.vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, DUAL_FRAGMENT_ENTRY);
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(planned.vertices, 6);
        let [first, second] = planned.attachments.as_slice() else {
            panic!("the dual shape renders two attachments");
        };
        assert_eq!(first.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(second.format, RenderPixelFormat::Rgba8Unorm);
        assert!(matches!(first.load, RenderLoadAction::Clear(_)));
        assert!(matches!(second.load, RenderLoadAction::Clear(_)));
        assert_eq!(first.store, RenderStoreAction::Store);
        assert_eq!(second.store, RenderStoreAction::Store);
        assert_eq!(first.initial, None);
        assert_eq!(second.initial, None);
        assert_eq!(planned.texel_bytes, 16);
        assert_eq!(planned.row_pitch, 8);
    }

    /// The v19 store increment: the same dual shape with location 1 discarded
    /// plans both attachments, but the discarded one carries `DontCare` rather
    /// than a landing observation.
    #[test]
    fn plan_admits_a_discarded_attachment_beside_a_stored_one() {
        let mut pass = dual_pass();
        pass.color_attachments[1].store = StoreOp::DontCare;
        let pipeline = dual_pipeline();
        let planned = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap();
        let [first, second] = planned.attachments.as_slice() else {
            panic!("the dual shape renders two attachments");
        };
        assert_eq!(first.store, RenderStoreAction::Store);
        assert_eq!(second.store, RenderStoreAction::DontCare);
        assert!(matches!(first.load, RenderLoadAction::Clear(_)));
        assert!(matches!(second.load, RenderLoadAction::Clear(_)));
    }

    /// The zero-colour-attachment depth pass (`research/docs/23` §3.3, v46): a
    /// pass with no colour attachment at all plans its raster from the stored
    /// depth attachment, compiles the reviewed module whose fragment stage
    /// generates no output, and carries no colour attachment to read back.
    ///
    /// Runs without a device, so the shape rules are the whole check here; the
    /// macOS CI executes the compiled pipeline.
    #[test]
    fn plan_admits_a_zero_colour_attachment_depth_pass() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments.clear();
        // The reviewed depth-only module reads the depth pair's stream, so the
        // pass binds one: the same two-attribute layout its vertex stage was
        // written for.
        pass.vertex_buffers = vec![BufferView {
            view_id: ViewId::new(31),
            metal_binding: 0,
            allocation_id: AllocationId::new(33),
            offset: 0,
            length: 192,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 192]),
        }];
        pass.indices = Some(IndexBufferBinding {
            view: quad_index_view(),
            format: IndexFormat::Uint16,
        });
        pass.vertices = 6;
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: DEPTH_STORE_ALLOCATION,
                view_id: DEPTH_STORE_VIEW,
            }),
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        let pipeline = RenderPipelineContract {
            vertex_entry: DEPTH_ONLY_VERTEX_ENTRY.to_owned(),
            fragment_entry: DEPTH_ONLY_FRAGMENT_ENTRY.to_owned(),
            color_formats: Vec::new(),
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 32,
                step: VertexStep::PerVertex,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x3,
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 16,
                        format: VertexFormat::Float32x4,
                    },
                ],
            }]),
        };
        let request = OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source: REVIEWED_DEPTH_ONLY_SOURCE,
            initial: Vec::new(),
        };
        let planned = plan(&request, 0, 0).expect("the zero-colour depth pass plans");
        assert!(
            planned.attachments.is_empty(),
            "a pass with no colour attachment plans no colour readback"
        );
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(
            planned.depth.as_ref().map(|depth| depth.store),
            Some(Some(DepthStoreOp::Store))
        );

        // The module pairing is the whole allowlist: the depth pair's source is
        // refused for this shape, exactly as any other unreviewed module is.
        let mismatched = OffscreenRenderRequest {
            source: REVIEWED_DEPTH_SOURCE,
            ..request
        };
        let error =
            plan(&mismatched, 0, 0).expect_err("the pair module is not this shape's module");
        assert_eq!(error.slug, "native_render_source_not_reviewed");
    }

    /// The v47 stencil shape (`research/docs/23` §3.3, v47): one colour
    /// attachment beside a rail-owned `stencil8` surface the pass clears to
    /// zero, and the reviewed state the depth pair's stream plus one colour
    /// target compiles under.
    ///
    /// Runs without a device, so the shape rules and the three planned fields
    /// are the whole check here; the macOS CI compiles the pipeline state and
    /// the depth-stencil state this plan builds.
    #[test]
    fn plan_admits_a_stencil_pass_and_plans_the_reviewed_state() {
        let pass = stencil_pass();
        let pipeline = stencil_pipeline();
        let planned =
            plan(&stencil_request(&pass, &pipeline), 0, 0).expect("the stencil pass plans");
        // The colour attachment is the raster, the stencil surface the second
        // one beside it: the pair module is what this shape compiles.
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(
            planned.module_path,
            "conformance/shaders/depth_pair_4x4.metal"
        );
        assert_eq!(planned.vertex_entry, DEPTH_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, DEPTH_FRAGMENT_ENTRY);
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the stencil shape renders one colour attachment");
        };
        assert_eq!(attachment.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(attachment.store, RenderStoreAction::Store);
        // The rail-owned surface's three planned fields, each the contract's
        // own value: the extent, the clear the load op carries, and the state
        // the draw tests and writes with. This fixture states no store action,
        // so the surface never leaves the pass and carries no landing
        // (`research/docs/23` §3.3, v49).
        let stencil = planned
            .stencil
            .expect("the pass opens the rail-owned stencil surface");
        assert_eq!(stencil.width, 2);
        assert_eq!(stencil.height, 2);
        assert_eq!(stencil.clear_value, Some(0));
        assert_eq!(stencil.test, Some(STENCIL_TEST));
        assert_eq!(stencil.store, None);
    }

    /// The contract's own rule, one level below the plan
    /// (`research/docs/23` §3.3, v47): stencil state with no stencil
    /// attachment has nothing to test, and core admission refuses it before
    /// this rail plans anything — the same shape the depth test states.
    #[test]
    fn plan_refuses_stencil_state_without_a_stencil_attachment() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.stencil_test = Some(STENCIL_TEST);
        assert_eq!(
            plan(&milestone_request(&pass, &milestone_pipeline(), None), 0, 0)
                .unwrap_err()
                .slug,
            "trace_contract_invalid"
        );
        // The contract states the refusal by name, so the rail's slug is the
        // one core admission uses rather than a second spelling of the same
        // fact.
        assert_eq!(
            pass.validate(),
            Err(ContractError::StencilTestWithoutAttachment)
        );
    }

    /// One pass whose stored multisampled depth surface states the resolve the
    /// device admits (`research/docs/23` §3.3, v57c).
    fn depth_resolving_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: DEPTH_STORE_ALLOCATION,
                view_id: DEPTH_STORE_VIEW,
            }),
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Sample0,
        });
        pass
    }

    /// The v60 combined shape: a stored multisampled depth surface and a
    /// stored stencil surface, both resolved (`research/docs/23` §3.3, v60).
    fn combined_stencil_resolving_pass() -> RenderPassDescriptor {
        let mut pass = depth_resolving_pass();
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 2,
            height: 2,
            load: StencilLoadOp::clear(0),
            store: Some(StoreOp::Store),
            identity: Some(RenderStencilIdentity {
                allocation_id: AllocationId::new(941),
                view_id: ViewId::new(951),
            }),
        });
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Always,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::IncrementWrap,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        pass.stencil_resolve = Some(MultisampleStencilResolve {
            filter: StencilResolveFilter::Sample0,
        });
        pass
    }

    /// The v60 combined shape plans through the device mask, carrying both
    /// filters the encoder sets (`research/docs/23` §3.3, v60).
    #[test]
    fn plan_admits_the_combined_stencil_resolve_in_the_device_mask() {
        let pass = combined_stencil_resolving_pass();
        let pipeline = milestone_pipeline();
        let planned = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            STENCIL_RESOLVE_SAMPLE0_BIT,
        )
        .expect("the combined resolve the device admits plans");
        assert_eq!(planned.depth_resolve, Some(DepthResolveFilter::Sample0));
        assert_eq!(planned.stencil_resolve, Some(StencilResolveFilter::Sample0));
    }

    /// A stencil filter outside the device mask is refused by the per-filter
    /// question the capability snapshot answered (`research/docs/23` §3.3,
    /// v60).
    #[test]
    fn plan_refuses_a_stencil_resolve_filter_the_device_does_not_report() {
        let pass = combined_stencil_resolving_pass();
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a stencil resolve outside the mask is refused");
        assert_eq!(refused.slug, "render_stencil_resolve_filter_unsupported");
    }

    /// The v66 rail-owned combined pair plans: one combined surface carries
    /// both faces, neither is kept, and the plan carries no resolve — the
    /// write-then-test shape whose third triangle fails against the value the
    /// second one's depth failure wrote (`research/docs/23` §3.3, v66).
    #[test]
    fn plan_admits_the_rail_owned_combined_pair() {
        let mut pass = combined_stencil_resolving_pass();
        pass.depth_resolve = None;
        pass.stencil_resolve = None;
        {
            let depth = pass
                .depth
                .as_mut()
                .expect("the fixture opens a depth attachment");
            depth.load = DepthLoadOp::clear(1.0);
            depth.store = None;
            depth.identity = None;
        }
        {
            let stencil = pass
                .stencil
                .as_mut()
                .expect("the fixture opens a stencil attachment");
            stencil.store = None;
            stencil.identity = None;
        }
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Equal,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::IncrementWrap,
            pass_op: StencilOp::Keep,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0)
            .expect("the rail-owned combined pair plans");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.depth_resolve, None);
        assert_eq!(planned.stencil_resolve, None);
    }

    /// A pair that keeps one face while dropping the other is refused by name
    /// rather than read as either reviewed shape (`research/docs/23` §3.3,
    /// v66).
    #[test]
    fn plan_refuses_a_lopsided_combined_pair() {
        let mut pass = combined_stencil_resolving_pass();
        // The stencil half drops its store and its resolve while the depth
        // half keeps both: the one surface's two faces disagree.
        pass.stencil_resolve = None;
        {
            let stencil = pass
                .stencil
                .as_mut()
                .expect("the fixture opens a stencil attachment");
            stencil.store = None;
            stencil.identity = None;
        }
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a lopsided combined pair is refused");
        // The contract owns the agreement rule, so the plan reports its own
        // spelling before the rail's re-assert can run.
        assert_eq!(refused.slug, "trace_contract_invalid");
    }

    /// The v57c shape: a stored multisampled depth surface whose resolve the
    /// device mask admits plans, and the plan carries the filter the encoder
    /// sets on the depth attachment (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_admits_a_stored_multisampled_depth_resolve_in_the_device_mask() {
        let pass = depth_resolving_pass();
        let pipeline = milestone_pipeline();
        let planned = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect("a stored multisampled depth resolve the device admits plans");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.depth_resolve, Some(DepthResolveFilter::Sample0));
        let depth = planned
            .depth
            .expect("the resolving pass opens its depth surface");
        assert_eq!(depth.store, Some(DepthStoreOp::Store));
    }

    /// The device mask is the per-filter question the capability snapshot
    /// answered: a resolve whose filter the device does not report is refused
    /// by name before any Metal object exists, exactly as the Vulkan rail
    /// refuses it (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_refuses_a_depth_resolve_filter_the_device_does_not_report() {
        let mut pass = depth_resolving_pass();
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Min,
        });
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a Min resolve outside the Sample0-only mask is refused");
        assert_eq!(refused.slug, "render_depth_resolve_filter_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("filter"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            refused.fields.get("modes"),
            Some(&FieldValue::Unsigned(u64::from(DEPTH_RESOLVE_SAMPLE0_BIT)))
        );
    }

    /// A stored multisampled depth surface without its resolve keeps the
    /// pre-v57c refusal: the surface's texels are only observable as the
    /// resolve's reduction, so keeping them without one stays impossible
    /// (`research/docs/23` §3.3, v57c). The core contract refuses the shape
    /// first, so the plan reports the contract's own spelling.
    #[test]
    fn plan_refuses_a_stored_multisampled_depth_without_a_resolve() {
        let mut pass = depth_resolving_pass();
        pass.depth_resolve = None;
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a stored multisampled depth without a resolve is refused");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(
            pass.validate(),
            Err(ContractError::MultisampleDepthStoreUnsupported)
        );
    }

    /// The v20 load increment (`research/docs/23` §3.1): a `LoadOp::DontCare`
    /// attachment plans as `MTLLoadAction::DontCare` with no preset bytes, so
    /// the encoder neither reads nor overwrites the pre-pass contents.
    #[test]
    fn plan_admits_a_dont_care_load_and_presets_nothing() {
        let pass = milestone_pass(LoadOp::DontCare);
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap();
        assert_eq!(planned.attachments.len(), 1);
        assert_eq!(planned.attachments[0].load, RenderLoadAction::DontCare);
        assert_eq!(
            planned.attachments[0].initial, None,
            "a DontCare attachment carries no pre-pass bytes"
        );
    }

    /// A pass whose only attachment discards is refused by the contract before
    /// this rail plans anything: core admission's `AllRenderAttachmentsDiscarded`
    /// reaches `plan` as `trace_contract_invalid`, so a discarded pass can never
    /// present a blank readback as "landed correctly" (`research/docs/23` §3.6).
    #[test]
    fn plan_refuses_a_pass_whose_only_attachment_discards() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments[0].store = StoreOp::DontCare;
        let pipeline = milestone_pipeline();
        let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
    }

    /// The MRT contract admits up to `MAX_COLOR_ATTACHMENTS`, and this rail's
    /// reviewed modules cover one, two and the ceiling: a wider pass is refused
    /// instead of being rendered partially (wave3 R1).
    #[test]
    fn plan_refuses_a_pass_beyond_the_attachment_ceiling() {
        let maximum = usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap();
        let mut pass = dual_pass();
        for _ in 2..(maximum + 1) {
            pass.color_attachments.push(pass.color_attachments[0]);
        }
        let mut pipeline = dual_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::Rgba8Unorm; maximum + 1];
        let error = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap_err();
        // Core admission owns the ceiling now that the rail's own maximum is
        // the contract's: the wider pass is refused as a trace-contract shape
        // before the rail's gate can see it (`attachment_count_unsupported`).
        assert_eq!(error.slug, "attachment_count_unsupported");
        // The rail's own gate stays as the fail-closed half for a directly
        // constructed request that skipped admission; its shape is asserted
        // directly so the two gates cannot drift apart.
        let refusal = mrt_attachment_count_refusal(maximum + 1);
        assert_eq!(refusal.slug, "render_mrt_attachment_count_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(
            refusal.fields.get("attachments"),
            Some(&FieldValue::Unsigned((maximum + 1) as u64))
        );
        assert_eq!(
            refusal.fields.get("maximum"),
            Some(&FieldValue::Unsigned(maximum as u64))
        );
    }

    /// Every colour location renders into the pass's one raster, so two
    /// attachments of different extents are refused by name before the
    /// contract's viewport rule reports the same shape as a viewport
    /// disagreement.
    #[test]
    fn plan_refuses_attachments_that_disagree_about_the_extent() {
        let mut pass = dual_pass();
        pass.color_attachments[1].height = 1;
        let error = plan(&dual_request(&pass, &dual_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_extent_mismatch");
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(error.fields.get("width"), Some(&FieldValue::Unsigned(2)));
        assert_eq!(error.fields.get("height"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            error.fields.get("first_height"),
            Some(&FieldValue::Unsigned(2))
        );
    }

    /// A present action hands its one attachment on to the target texture, so a
    /// present pass with a second location is refused under the
    /// single-attachment slug instead of having location 1 dropped.
    #[test]
    fn plan_refuses_a_present_pass_with_two_attachments() {
        let mut pass = dual_pass();
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(7),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Undefined,
            },
            source: ViewId::new(7),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        let error = plan(&dual_request(&pass, &dual_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_count_unsupported");
        assert_eq!(
            error.fields.get("attachments"),
            Some(&FieldValue::Unsigned(2))
        );
        assert_eq!(error.fields.get("maximum"), Some(&FieldValue::Unsigned(1)));
    }

    /// A present pass renders into the provider-owned target alone: it opens no
    /// depth surface, so a pass that names one — stored or not — is refused
    /// instead of having the attachment silently dropped, the same refusal the
    /// Vulkan rail's present entry states (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_refuses_a_present_pass_with_a_depth_attachment() {
        for (store, expected) in [
            (None, "unstated"),
            (Some(DepthStoreOp::DontCare), "dontcare"),
            (Some(DepthStoreOp::Store), "store"),
        ] {
            let mut pass = milestone_present_pass();
            pass.depth = Some(RenderDepthAttachment {
                format: DepthFormat::Depth32Float,
                width: 2,
                height: 2,
                load: DepthLoadOp::clear(1.0),
                store,
                identity: match store {
                    Some(DepthStoreOp::Store) => Some(RenderDepthIdentity {
                        allocation_id: DEPTH_STORE_ALLOCATION,
                        view_id: DEPTH_STORE_VIEW,
                    }),
                    _ => None,
                },
            });
            pass.depth_test = Some(DepthTest {
                compare: CompareFunction::Less,
                write: true,
            });
            let pipeline = milestone_pipeline();
            let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
            assert_eq!(error.slug, "render_present_depth_unsupported");
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(
                error.fields.get("store"),
                Some(&FieldValue::Text(expected.to_owned()))
            );
            assert_eq!(
                error.detail.as_deref(),
                Some(
                    "the present rail renders into one provider-owned colour target and opens \
                     no depth surface"
                )
            );
        }
    }

    /// A present pass renders into the provider-owned target alone: it opens no
    /// stencil surface either, so a pass that names one — stored or not — is
    /// refused instead of having the attachment dropped, the same refusal the
    /// Vulkan rail's present entry states (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_refuses_a_present_pass_with_a_stencil_attachment() {
        for (store, expected) in [
            (None, "unstated"),
            (Some(StoreOp::DontCare), "dontcare"),
            (Some(StoreOp::Store), "store"),
        ] {
            let mut pass = milestone_present_pass();
            pass.stencil = Some(RenderStencilAttachment {
                format: StencilFormat::Stencil8,
                width: 2,
                height: 2,
                load: StencilLoadOp::clear(0),
                store,
                identity: match store {
                    Some(StoreOp::Store) => Some(RenderStencilIdentity {
                        allocation_id: STENCIL_STORE_ALLOCATION,
                        view_id: STENCIL_STORE_VIEW,
                    }),
                    _ => None,
                },
            });
            pass.stencil_test = Some(STENCIL_TEST);
            let pipeline = milestone_pipeline();
            let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
            assert_eq!(error.slug, "render_present_stencil_unsupported");
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(
                error.fields.get("store"),
                Some(&FieldValue::Text(expected.to_owned()))
            );
            assert_eq!(
                error.detail.as_deref(),
                Some(
                    "the present rail renders into one provider-owned colour target and opens \
                     no stencil surface"
                )
            );
        }
    }

    /// A present action beside a four-sample raster is admitted from v62 on:
    /// the encoder creates the n-sample surface and resolves it into the
    /// provider-owned present target with `storeAction = .multisampleResolve`,
    /// so the plan carries the raster and the single attachment unchanged
    /// (`research/docs/24` §3.5, v62).
    #[test]
    fn plan_admits_a_multisampled_present_pass_over_its_resolve_landing() {
        let mut pass = milestone_present_pass();
        // The reviewed multisample load shape opens the attachment from a
        // clear; the sentinel the present target carries is overwritten by the
        // resolve, exactly as the v62 fixture states.
        pass.color_attachments[0].load = LoadOp::Clear(sentinel());
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0)
            .expect("the four-sample present pass plans over its resolve landing");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.attachments.len(), 1);
        assert_eq!(planned.attachments[0].store, RenderStoreAction::Store);
    }

    #[test]
    fn plan_refuses_a_draw_shape_other_than_the_full_screen_triangle() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.vertices = 6;
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "draw_shape_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn load_requires_the_previous_texels_and_clear_refuses_them() {
        let pass = milestone_pass(LoadOp::Load);
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
        assert_eq!(error.class, ProviderErrorClass::Args);

        let previous = [0x11_u8; 16];
        let planned = plan_pass(&milestone_request(&pass, &pipeline, Some(&previous))).unwrap();
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert_eq!(attachment.load, RenderLoadAction::Load);
        assert_eq!(attachment.initial, Some(previous.as_slice()));

        let short = [0x11_u8; 15];
        let error = plan_pass(&milestone_request(&pass, &pipeline, Some(&short))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");

        let cleared = milestone_pass(LoadOp::Clear(sentinel()));
        let error =
            plan_pass(&milestone_request(&cleared, &pipeline, Some(&previous))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
    }

    /// The falsifiability rule `research/docs/23` §1.3 states: a pass that never
    /// ran has to be distinguishable from one that did, in every channel.
    #[test]
    fn a_cleared_attachment_cannot_imitate_the_fragment_output() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap();
        let RenderLoadAction::Clear(components) = planned.attachments[0].load else {
            panic!("the milestone clears its attachment");
        };
        for channel in 0..4 {
            assert_ne!(
                components[channel], EXPECTED_TEXEL_COMPONENTS[channel],
                "clear channel {channel} must differ from the fragment output"
            );
        }
    }

    /// The one compute pass of the trace-path fixture: a read-only declaration
    /// of the attachment's own view.
    ///
    /// Core admission resolves every render attachment against the views the
    /// trace declares, so a render-bearing trace always carries one of these,
    /// and it is what makes the attachment's landing rail exist (see
    /// `ComputeTrace::serial_resources`).
    fn declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: ViewId::new(7),
                metal_binding: 0,
                allocation_id: AllocationId::new(9),
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0xfe; 16]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The compute registration `declaration_pass` names. Its contract reflects
    /// a read-only 16-byte binding, i.e. exactly the extent the 2x2 attachment
    /// restates, and the footprint proof is static so admission can compare it
    /// with the view.
    fn declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-wiring-fixture", vec![7]).unwrap(),
                entry_name: "declares_the_attachment_view".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![BufferBindingContract {
                    metal_binding: 0,
                    access: BufferAccess::Read,
                    footprint: FootprintProof::Static { max_bytes: 16 },
                }],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry a render registration hands back, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it: the id the
    /// render pass names, the reviewed vertex entry, the most permissive
    /// exact-thread contract (nothing reads a render entry as a compute
    /// contract) and the reviewed render contract in the `render` half core
    /// admission compares the attachment against.
    fn render_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(3),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-wiring-fixture", vec![3]).unwrap(),
                entry_name: VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(RenderPipelineContract {
                vertex_entry: VERTEX_ENTRY.to_owned(),
                fragment_entry: FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            }),
        }
    }

    /// The milestone's trace and the resource namespace it needs: the
    /// declaration pass, then the render pass under test.
    fn milestone_trace(load: LoadOp) -> (ComputeTrace, ResourceTableSnapshot) {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(1),
            pipelines: vec![declaration_pipeline(), render_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                TracePass::Render(milestone_pass(load)),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(3),
                size: 16,
            })
            .unwrap();
        (trace, resources)
    }

    /// The milestone's present pass: the same 2x2 attachment, but with a
    /// `Load` that keeps the present target's sentinel and a present action
    /// handing that target on (`research/docs/24` §3.1, §3.5 shape one).
    fn milestone_present_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Load);
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(7),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Sentinel(vec![0xfe, 0xfe, 0xfe, 0xfe]),
            },
            source: ViewId::new(7),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        pass
    }

    /// The milestone's present-bearing trace and resource namespace.
    fn milestone_present_trace() -> (ComputeTrace, ResourceTableSnapshot) {
        let (mut trace, resources) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        *pass = milestone_present_pass();
        (trace, resources)
    }

    /// The stored depth attachment's own resource (`research/docs/23` §3.3,
    /// v43): allocation 940 and view 950, covering the milestone pass's 2x2
    /// `depth32float` extent — four texels of four bytes, the same length the
    /// colour attachment's landing view has.
    const DEPTH_STORE_ALLOCATION: AllocationId = AllocationId::new(940);
    const DEPTH_STORE_VIEW: ViewId = ViewId::new(950);
    /// The stored depth surface's byte length: 2x2 texels of a four-byte
    /// `depth32float` texel each.
    const DEPTH_STORE_BYTES: u64 = 16;

    /// The depth the v43 fixture's draw leaves behind: `0.5` as
    /// `depth32float`'s little-endian bytes. The clear below writes `1.0`
    /// (`00 00 80 3f`), so a pass that never stored its depth surface cannot
    /// read back as one that did.
    const DEPTH_STORE_TEXEL: [u8; 4] = [0x00, 0x00, 0x00, 0x3f];

    /// The compute declaration of the stored depth surface's landing view: one
    /// read-only view over the whole attachment, the same shape the colour
    /// attachment's own declaration pass has.
    ///
    /// A stored depth identity has to be declared by the trace: core admission
    /// resolves it against the same declarations a colour attachment resolves
    /// against and refuses an undeclared one (`AttachmentViewUnknown`), and the
    /// declaring binding is what puts the view into the serial pool — its
    /// access is what `serial_resources` upgrades to a write — which is where
    /// [`plan_trace`] resolves the landing from (`research/docs/23` §3.3, v43).
    fn depth_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: DEPTH_STORE_VIEW,
                metal_binding: 0,
                allocation_id: DEPTH_STORE_ALLOCATION,
                offset: 0,
                length: DEPTH_STORE_BYTES,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0x80; DEPTH_STORE_BYTES as usize]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The v43 trace: the milestone pass plus a depth attachment of the same
    /// 2x2 extent, and the compute declaration of its landing view.
    ///
    /// `store` is the whole v43 axis. `Some(Store)` is the shape whose texels
    /// survive the pass and therefore need a landing; `None` is the pre-v43
    /// shape and `Some(DontCare)` the explicit discard, both of which keep the
    /// surface rail-owned and land nothing.
    fn depth_store_trace(store: Option<DepthStoreOp>) -> ComputeTrace {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store,
            identity: match store {
                Some(DepthStoreOp::Store) => Some(RenderDepthIdentity {
                    allocation_id: DEPTH_STORE_ALLOCATION,
                    view_id: DEPTH_STORE_VIEW,
                }),
                _ => None,
            },
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        // The declaration comes before the render pass, exactly as the colour
        // attachment's does: the pool it feeds is the compute rail's binding
        // set and every pass of it runs before the draw.
        trace
            .passes
            .insert(1, TracePass::Compute(depth_declaration_pass()));
        trace
    }

    /// The capability snapshot the macOS provider builds, with the render bits
    /// taken from the value under test and the compute bits from `native.rs`.
    fn capabilities(bits: &RenderCapabilityBits) -> ProviderCapabilities {
        capabilities_with(bits, &vertex_input_capability_bits())
    }

    /// The same snapshot with the vertex-input bits spelled out, so a test can
    /// construct the pre-flip declaration (`native.rs` takes both from the rail).
    fn capabilities_with(
        bits: &RenderCapabilityBits,
        vertex: &VertexInputCapabilityBits,
    ) -> ProviderCapabilities {
        ProviderCapabilities {
            max_passes: 8,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1024, 1024, 1024],
            max_invocations: 1024,
            max_group_count: [1024, 1024, 1024],
            max_storage_buffer_descriptors: 31,
            max_buffer_range: 1024,
            max_push_constant_bytes: 0,
            alias_mode: AliasMode::DistinctViews,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
            supports_render_passes: bits.supports_render_passes,
            max_color_attachments: bits.max_color_attachments,
            max_attachment_dimension: bits.max_attachment_dimension,
            supported_color_formats: bits.supported_color_formats.clone(),
            max_vertex_buffers: vertex.max_vertex_buffers,
            supported_vertex_formats: vertex.supported_vertex_formats.clone(),
            supported_index_formats: vertex.supported_index_formats.clone(),
            supports_render_instancing: false,
            max_render_instances: 0,
            // The test snapshot spells the pre-flip shape out for the same
            // reason the instancing pair above does: a test constructs the
            // "cannot multisample" declaration and asserts core admission
            // refuses the multisampled pass (`research/docs/23` §3.3, v51).
            supports_render_multisample: false,
            max_render_sample_count: 0,
            supports_render_depth_resolve: false,
            depth_resolve_modes: 0,
            supports_render_stencil_resolve: false,
            stencil_resolve_modes: 0,
            // The test snapshot spells the pre-flip shape out for the same
            // reason the multisample pair above does: a test constructs the
            // "cannot sample render-side textures" declaration and asserts
            // core admission refuses the texture-bearing pass
            // (`research/docs/23` §3.3, v70).
            supports_render_texture_sampling: false,
            max_render_textures: 0,
            supported_render_texture_formats: Vec::new(),
            supports_presentation: bits.supports_presentation,
            max_present_targets: bits.max_present_targets,
            supported_present_modes: bits.supported_present_modes.clone(),
            max_present_image_count: bits.max_present_image_count,
            supports_heaps: false,
            max_heap_bytes: 0,
            supported_heap_storage_modes: Vec::new(),
            supports_heap_aliasing: false,
            supports_indirect_command_buffers: false,
            max_indirect_commands: 0,
            supported_indirect_commands: Vec::new(),
        }
    }

    /// The registrations `plan_trace` resolves the trace's pipeline ids against.
    fn milestone_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(PipelineId::new(3), milestone_pipeline())])
    }

    /// The vertex-input fixture's identities: one 32-byte vertex stream
    /// (`view` 21 inside allocation 12) and one 12-byte index buffer (`view` 22
    /// inside allocation 13). Both are separate whole-allocation views, the
    /// shape the reviewed fixture declares, so the rail's footprint proof is
    /// stated against the bytes the trace itself carries.
    const QUAD_VERTEX_VIEW: ViewId = ViewId::new(21);
    const QUAD_VERTEX_ALLOCATION: AllocationId = AllocationId::new(12);
    const QUAD_INDEX_VIEW: ViewId = ViewId::new(22);
    const QUAD_INDEX_ALLOCATION: AllocationId = AllocationId::new(13);
    /// The reviewed quad pipeline's id: one counter with the render rail's other
    /// registration, so 3 stays the milestone's triangle.
    const QUAD_PIPELINE: PipelineId = PipelineId::new(4);
    /// The reviewed dual pipeline's id, one counter further.
    const DUAL_PIPELINE: PipelineId = PipelineId::new(5);

    /// The four NDC corners of the reviewed quad, as the 32 little-endian
    /// `float32x2` bytes the stream holds.
    fn quad_vertex_bytes() -> Vec<u8> {
        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let mut bytes = Vec::with_capacity(32);
        for corner in corners {
            for component in corner {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
        bytes
    }

    /// The six `uint16` indices that select the quad's two triangles: (0,1,2)
    /// and (2,1,3), i.e. 12 little-endian bytes covering all four vertices.
    fn quad_index_bytes() -> Vec<u8> {
        let indices: [u16; 6] = [0, 1, 2, 2, 1, 3];
        indices
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<u8>>()
    }

    /// The reviewed vertex stream as the render pass declares it: a read-only
    /// view at binding 0 whose `source` carries the four NDC corners
    /// (`RenderPassDescriptor::vertex_buffers`).
    fn quad_vertex_view() -> BufferView {
        let bytes = quad_vertex_bytes();
        BufferView {
            view_id: QUAD_VERTEX_VIEW,
            metal_binding: 0,
            allocation_id: QUAD_VERTEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the fixture's lengths fit u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        }
    }

    /// The reviewed index buffer: the same shape one level over, with the
    /// `metal_binding` an index binding always carries
    /// (`IndexBufferBinding::validate_shape` refuses any other value).
    fn quad_index_view() -> BufferView {
        let bytes = quad_index_bytes();
        BufferView {
            view_id: QUAD_INDEX_VIEW,
            metal_binding: 0,
            allocation_id: QUAD_INDEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the fixture's lengths fit u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        }
    }

    /// The reviewed indexed pipeline contract: one `float32x2` stream,
    /// `[[stage_in]]` positions.
    fn quad_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: QUAD_VERTEX_ENTRY.to_owned(),
            fragment_entry: FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
                step: VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            }]),
        }
    }

    /// The reviewed indexed pass: the same 2x2 attachment, six indices over the
    /// bound stream.
    fn quad_pass() -> RenderPassDescriptor {
        RenderPassDescriptor {
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            base_vertex: 0,
            pipeline: QUAD_PIPELINE,
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(7),
                allocation_id: AllocationId::new(9),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load: LoadOp::Clear(sentinel()),
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            scissor: None,
            vertices: 6,
            vertex_buffers: vec![quad_vertex_view()],
            indices: Some(IndexBufferBinding {
                view: quad_index_view(),
                format: IndexFormat::Uint16,
            }),
            instance_count: 1,
            textures: Vec::new(),
            present: None,
        }
    }

    /// The compute pass that declares the attachment view and both streams.
    ///
    /// The attachment lands through view 7, which only a compute declaration
    /// puts into the serial pool, so this pass is what makes the landing view
    /// real. The two streams are declared here as well — with the pass's own
    /// stream bytes, which core admission requires to agree with the render
    /// pass's views for the same identity — even though the render rail no
    /// longer needs a declaration to read them.
    fn quad_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(7),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(9),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfe; 16]),
                },
                BufferView {
                    metal_binding: 1,
                    ..quad_vertex_view()
                },
                BufferView {
                    metal_binding: 2,
                    ..quad_index_view()
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The declaration pass's compute registration: three read-only bindings
    /// whose static footprints are exactly the views' own lengths.
    fn quad_declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-vertex-fixture", vec![11]).unwrap(),
                entry_name: "declares_the_render_views".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![
                    BufferBindingContract {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 1,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 32 },
                    },
                    BufferBindingContract {
                        metal_binding: 2,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 12 },
                    },
                ],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry of the quad registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn quad_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: QUAD_PIPELINE,
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-vertex-fixture", vec![4]).unwrap(),
                entry_name: QUAD_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(quad_pipeline()),
        }
    }

    /// The trace-table entry's compute half: the most permissive exact-thread
    /// contract, which no render pass reads as a compute contract.
    fn render_table_contract() -> PipelineContract {
        PipelineContract {
            dispatch_kind: DispatchKind::ThreadsExact,
            required_local_size: None,
            fixed_grid: None,
            push_constant_offset: 0,
            push_constant_bytes: 0,
            buffer_bindings: Vec::new(),
            shader_capabilities: Vec::new(),
            translator_revision: None,
        }
    }

    /// The vertex-input fixture's trace and resource namespace: the declaration
    /// pass, then the indexed render pass.
    fn quad_trace() -> (ComputeTrace, ResourceTableSnapshot) {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(2),
            pipelines: vec![quad_declaration_pipeline(), quad_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(quad_declaration_pass()),
                TracePass::Render(quad_pass()),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        for (allocation, size) in [
            (AllocationId::new(9), 16),
            (QUAD_VERTEX_ALLOCATION, 32),
            (QUAD_INDEX_ALLOCATION, 12),
        ] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(3),
                    size,
                })
                .unwrap();
        }
        (trace, resources)
    }

    /// The registrations `plan_trace` resolves the vertex-input trace against.
    fn quad_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(QUAD_PIPELINE, quad_pipeline())])
    }

    /// The request the trace path builds for the reviewed indexed pass.
    fn quad_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_VERTEX_SOURCE,
            initial: vec![None],
        }
    }

    /// The stencil fixture's reviewed state: the v47 fixture's own values —
    /// `equal 0` with reference 0, both failure operations `keep`, and
    /// `increment_wrap` on success, over the full read and write masks
    /// (`research/docs/23` §3.3, v47). Spelled once so the pass and the
    /// assertion cannot drift.
    const STENCIL_TEST: StencilTest = StencilTest {
        compare: StencilCompare::Equal,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::Keep,
        pass_op: StencilOp::IncrementWrap,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    /// The v47 fixture's pass shape: the milestone's one 2x2 `rgba8_unorm`
    /// attachment, the reviewed pair stream with six indices over it, and the
    /// rail-owned `stencil8` surface cleared to zero that the state above
    /// tests (`research/docs/23` §3.3, v47).
    ///
    /// The stream is the depth pair's two-attribute layout — a `float32x3`
    /// position and a `float32x4` tint — which is the shape the pair module's
    /// vertex stage reads; the vertex bytes themselves are the fixture's
    /// business, not the plan's, so a zeroed 192-byte view (six records of 32)
    /// stands in for them exactly as the zero-colour depth fixture does.
    fn stencil_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.vertex_buffers = vec![BufferView {
            view_id: ViewId::new(41),
            metal_binding: 0,
            allocation_id: AllocationId::new(43),
            offset: 0,
            length: 192,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 192]),
        }];
        pass.indices = Some(IndexBufferBinding {
            view: quad_index_view(),
            format: IndexFormat::Uint16,
        });
        pass.vertices = 6;
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 2,
            height: 2,
            load: StencilLoadOp::clear(0),
            // The v47 shape states no store and names no landing: the surface
            // is the pass's own and disappears with it (`research/docs/23`
            // §3.3, v49).
            store: None,
            identity: None,
        });
        pass.stencil_test = Some(STENCIL_TEST);
        pass
    }

    /// The reviewed pair pipeline contract the stencil fixture declares: the
    /// depth pair's two stages compiled against one `rgba8_unorm` attachment,
    /// with the two-attribute stream layout its vertex stage was written for
    /// (`research/docs/23` §3.3, v47).
    fn stencil_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: DEPTH_VERTEX_ENTRY.to_owned(),
            fragment_entry: DEPTH_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 32,
                step: VertexStep::PerVertex,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x3,
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 16,
                        format: VertexFormat::Float32x4,
                    },
                ],
            }]),
        }
    }

    /// The request the trace path builds for the reviewed stencil pass: the
    /// pair module's bytes and one previous-bytes slot per colour attachment.
    fn stencil_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_DEPTH_SOURCE,
            initial: vec![None],
        }
    }

    /// The stored stencil attachment's own resource (`research/docs/23` §3.3,
    /// v49): allocation 960 and view 970, covering the stencil fixture's 2x2
    /// `stencil8` extent — four one-byte texels.
    const STENCIL_STORE_ALLOCATION: AllocationId = AllocationId::new(960);
    const STENCIL_STORE_VIEW: ViewId = ViewId::new(970);
    /// The stored stencil surface's byte length: 2x2 texels of a single-byte
    /// `stencil8` texel each.
    const STENCIL_STORE_BYTES: u64 = 4;
    /// The stencil value the v49 fixture's draw leaves behind: `1`, which is
    /// the reviewed state's `increment_wrap` on success. The pass clears the
    /// surface to `0`, so a pass that never stored its stencil surface cannot
    /// read back as one that did.
    const STENCIL_STORE_TEXEL: u8 = 0x01;

    /// The compute declaration of the stored stencil surface's landing view:
    /// one read-only view over the whole attachment, the depth declaration's
    /// own shape one byte wide.
    ///
    /// A stored stencil identity has to be declared by the trace: core
    /// admission resolves it against the same declarations a colour attachment
    /// resolves against and refuses an undeclared one (`AttachmentViewUnknown`),
    /// and the declaring binding is what puts the view into the serial pool —
    /// its access is what `serial_resources` upgrades to a write — which is
    /// where [`plan_trace`] resolves the landing from (`research/docs/23` §3.3,
    /// v49).
    fn stencil_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: STENCIL_STORE_VIEW,
                metal_binding: 0,
                allocation_id: STENCIL_STORE_ALLOCATION,
                offset: 0,
                length: STENCIL_STORE_BYTES,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0x80; STENCIL_STORE_BYTES as usize]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The v49 trace: the stencil fixture's reviewed pass with the store action
    /// under test, and the compute declaration of its landing view.
    ///
    /// `store` is the whole v49 axis. `Some(Store)` is the shape whose texels
    /// survive the pass and therefore need a landing; `None` is the pre-v49
    /// shape and `Some(DontCare)` the explicit discard, both of which keep the
    /// surface rail-owned and land nothing.
    fn stencil_store_trace(store: Option<StoreOp>) -> ComputeTrace {
        let mut pass = stencil_pass();
        let stencil = pass
            .stencil
            .as_mut()
            .expect("the stencil fixture declares a stencil attachment");
        stencil.store = store;
        stencil.identity = match store {
            Some(StoreOp::Store) => Some(RenderStencilIdentity {
                allocation_id: STENCIL_STORE_ALLOCATION,
                view_id: STENCIL_STORE_VIEW,
            }),
            _ => None,
        };
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(4),
            pipelines: vec![declaration_pipeline(), stencil_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                // The stencil declaration comes before the render pass, exactly
                // as the colour attachment's does: the pool it feeds is the
                // compute rail's binding set and every pass of it runs before
                // the draw.
                TracePass::Compute(stencil_declaration_pass()),
                TracePass::Render(pass),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    /// The trace-table entry of the stencil registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn stencil_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(3),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-stencil-fixture", vec![3]).unwrap(),
                entry_name: DEPTH_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(stencil_pipeline()),
        }
    }

    /// The registrations `plan_trace` resolves the stencil trace against.
    fn stencil_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(PipelineId::new(3), stencil_pipeline())])
    }

    /// The reviewed dual pipeline contract: the indexed vertex stage plus the
    /// two-location fragment stage, compiled against two `rgba8_unorm`
    /// attachments.
    fn dual_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: QUAD_VERTEX_ENTRY.to_owned(),
            fragment_entry: DUAL_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
                step: VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            }]),
        }
    }

    /// The reviewed dual pass: the indexed quad's vertex stream and index
    /// buffer, drawn into two 2x2 `rgba8_unorm` attachments.
    fn dual_pass() -> RenderPassDescriptor {
        let mut pass = quad_pass();
        pass.pipeline = DUAL_PIPELINE;
        pass.color_attachments.push(RenderAttachment {
            view_id: ViewId::new(8),
            allocation_id: AllocationId::new(10),
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Clear(sentinel()),
            store: StoreOp::Store,
        });
        pass
    }

    /// The request the trace path builds for the reviewed dual pass.
    fn dual_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_DUAL_SOURCE,
            initial: vec![None, None],
        }
    }

    /// The compute pass that declares both attachment views and both streams.
    ///
    /// Each attachment lands through its own view (7 in allocation 9, 8 in
    /// allocation 10), which only a compute declaration puts into the serial
    /// pool, so this pass is what makes both landing rails real. The two
    /// streams are declared here as well, the same shape the quad fixture
    /// uses.
    fn dual_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(7),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(9),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfe; 16]),
                },
                BufferView {
                    view_id: ViewId::new(8),
                    metal_binding: 1,
                    allocation_id: AllocationId::new(10),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfd; 16]),
                },
                BufferView {
                    metal_binding: 2,
                    ..quad_vertex_view()
                },
                BufferView {
                    metal_binding: 3,
                    ..quad_index_view()
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The dual declaration pass's compute registration: four read-only
    /// bindings whose static footprints are exactly the views' own lengths.
    fn dual_declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-mrt-fixture", vec![11]).unwrap(),
                entry_name: "declares_the_render_views".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![
                    BufferBindingContract {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 1,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 2,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 32 },
                    },
                    BufferBindingContract {
                        metal_binding: 3,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 12 },
                    },
                ],
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry of the dual registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn dual_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: DUAL_PIPELINE,
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-mrt-fixture", vec![5]).unwrap(),
                entry_name: QUAD_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(dual_pipeline()),
        }
    }

    /// The dual fixture's trace and resource namespace: the declaration pass,
    /// then the dual render pass with the requested load op.
    fn dual_trace(load: LoadOp) -> (ComputeTrace, ResourceTableSnapshot) {
        let mut pass = dual_pass();
        for attachment in &mut pass.color_attachments {
            attachment.load = load;
        }
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(3),
            pipelines: vec![dual_declaration_pipeline(), dual_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(dual_declaration_pass()),
                TracePass::Render(pass),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        for (allocation, size) in [
            (AllocationId::new(9), 16),
            (AllocationId::new(10), 16),
            (QUAD_VERTEX_ALLOCATION, 32),
            (QUAD_INDEX_ALLOCATION, 12),
        ] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(3),
                    size,
                })
                .unwrap();
        }
        (trace, resources)
    }

    /// The registrations `plan_trace` resolves the dual trace against.
    fn dual_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(DUAL_PIPELINE, dual_pipeline())])
    }

    /// Shrink a declared stream to `length` bytes, keeping the view's shape
    /// self-consistent so a footprint refusal is the only rule that can fire.
    fn shorten(view: &mut BufferView, length: u64) {
        let BufferSource::OwnedBytes(bytes) = &view.source else {
            panic!("the fixture's streams are owned bytes");
        };
        let end = usize::try_from(length).expect("the fixture's lengths fit a host usize");
        view.length = length;
        view.source = BufferSource::OwnedBytes(bytes[..end].to_vec());
    }

    /// Replace a declared stream's bytes and length together, for the index
    /// values a footprint test wants to vary.
    fn replace_bytes(view: &mut BufferView, bytes: Vec<u8>) {
        view.length = u64::try_from(bytes.len()).expect("the fixture's lengths fit u64");
        view.source = BufferSource::OwnedBytes(bytes);
    }

    /// The `vertex_id` milestone pass with an index buffer bound to it: three
    /// indices select through the module's three generated positions, which is
    /// the shape core admission states for a pass with indices and no stream
    /// (`RenderPassDescriptor::validate`).
    ///
    /// The index bytes travel in the binding's own view, so the serial pool this
    /// trace returns is the compute declaration's — view 7, the attachment's
    /// landing view — and holds no stream at all.
    fn vertex_id_indexed_trace(indices: [u16; 3]) -> (ComputeTrace, Vec<BufferView>) {
        let index_bytes = indices
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<u8>>();
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the milestone fixture ends with its render pass");
        };
        pass.indices = Some(IndexBufferBinding {
            view: BufferView {
                view_id: QUAD_INDEX_VIEW,
                metal_binding: 0,
                allocation_id: QUAD_INDEX_ALLOCATION,
                offset: 0,
                length: u64::try_from(index_bytes.len()).expect("the fixture's lengths fit u64"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(index_bytes),
            },
            format: IndexFormat::Uint16,
        });
        let pool = trace.serial_resources().expect("admitted serial pool");
        (trace, pool)
    }

    /// The trace's table entry and the registry have to carry the same render
    /// contract.
    ///
    /// The entry's `render` half is what core admission compares a pass against
    /// (review item I3, 2026-09-14) and the registry is what the rail executes,
    /// so a fixture where the two disagree would let a value-level test pass on
    /// an agreement the provider would refuse at submit.
    #[test]
    fn the_render_table_entry_carries_the_registered_contract() {
        let entry = render_table_entry();
        let registered = milestone_contracts();
        let contract = registered
            .get(&entry.pipeline_id)
            .expect("the registry holds the entry the render pass names");
        assert_eq!(entry.render.as_ref(), Some(contract));
    }

    /// The step that flips the capability bit needs the declared bits and core
    /// admission to agree: what the snapshot says it renders and what core
    /// admits have to be the same set, or one of the two is lying.
    #[test]
    fn declared_render_capabilities_admit_what_the_rail_plans() {
        let bits = capability_bits();
        assert!(bits.supports_render_passes);
        assert_eq!(bits.max_color_attachments, MAX_COLOR_ATTACHMENTS);
        assert_eq!(bits.max_attachment_dimension, MAX_ATTACHMENT_DIMENSION);
        assert_eq!(
            bits.supported_color_formats,
            SUPPORTED_COLOR_FORMATS.to_vec()
        );
        // The rail declares the full MRT shape now that its reviewed modules
        // cover one, two and the `MAX_COLOR_ATTACHMENTS` ceiling (v24). The bit
        // still has to stay at or below core's cap (wave3 R1).
        let core_max = u32::try_from(metal_api_core::provider::MAX_COLOR_ATTACHMENTS).unwrap();
        assert_eq!(core_max, 4);
        assert_eq!(bits.max_color_attachments, core_max);
        assert!(bits.max_color_attachments <= core_max);
        assert_eq!(
            bits.supported_color_formats,
            AttachmentFormat::ADMITTED.to_vec()
        );

        let (trace, resources) = milestone_trace(LoadOp::Clear(sentinel()));
        capabilities(&bits)
            .admit(&trace, &resources)
            .expect("the declared bits admit the milestone trace");

        // The pre-flip snapshot refuses the same trace in render admission,
        // before any compute reservation — that refusal is what the flip
        // removes.
        let mut before_the_flip = bits.clone();
        before_the_flip.supports_render_passes = false;
        let refused = capabilities(&before_the_flip)
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "render_passes_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
    }

    /// The flipped snapshot admits the present-bearing trace in the second
    /// admission gate, and its present bits are the rail's own limits rather
    /// than a second spelling that could drift from the contract's. The flip
    /// evidence is the Apple-GPU `--present-selftest` run recorded on
    /// [`present_capability_bits`].
    #[test]
    fn declared_present_capabilities_admit_what_the_rail_plans() {
        let bits = capability_bits();
        assert!(bits.supports_presentation);
        assert_eq!(bits.max_present_targets, MAX_PRESENT_TARGETS);
        assert_eq!(bits.supported_present_modes, PresentMode::ADMITTED.to_vec());
        assert_eq!(bits.max_present_image_count, MAX_PRESENT_IMAGE_COUNT);
        // The first increment admits exactly one target, one image and FIFO
        // (`research/docs/24` §3.1), which is what those constants say.
        assert_eq!(
            bits.max_present_targets,
            metal_api_core::provider::MAX_PRESENT_TARGETS as u32
        );
        assert_eq!(
            bits.max_present_image_count,
            metal_api_core::provider::MAX_PRESENT_IMAGE_COUNT
        );

        let (trace, resources) = milestone_present_trace();
        capabilities(&bits)
            .admit(&trace, &resources)
            .expect("the declared bits admit the present milestone trace");
    }

    /// The pre-flip snapshot — the same bits with the present gate closed —
    /// still refuses the present-bearing trace in the second admission gate
    /// (`present_targets_unsupported`) before any resource action. Keeping the
    /// refusal pinned on a constructed snapshot is what makes the flipped
    /// production bits above falsifiable: it is the same trace and the same
    /// gate, only the declaration differs.
    #[test]
    fn the_pre_flip_snapshot_refuses_a_present_bearing_trace() {
        let mut bits = capability_bits();
        bits.supports_presentation = false;
        bits.max_present_targets = 0;
        bits.supported_present_modes = Vec::new();
        bits.max_present_image_count = 0;

        let (trace, resources) = milestone_present_trace();
        let refused = capabilities(&bits).admit(&trace, &resources).unwrap_err();
        assert_eq!(refused.slug, "present_targets_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        // The refusal names the number of present actions the snapshot would
        // have to drop.
        assert_eq!(
            refused.fields.get("targets"),
            Some(&FieldValue::Unsigned(1))
        );
    }

    /// The host-side half of the present plan: the descriptor is borrowed from
    /// the pass and the sentinel is expanded to the target's whole extent, so
    /// the macOS present path only has to upload it.
    #[test]
    fn plan_trace_plans_the_present_action_and_expands_the_sentinel() {
        let (trace, _) = milestone_present_trace();
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed present pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the present trace carries one render pass");
        };
        let present = planned
            .present
            .as_ref()
            .expect("the pass carries a present");
        assert_eq!(present.descriptor.source, ViewId::new(7));
        assert_eq!(
            present.descriptor.target.allocation_id,
            AllocationId::new(9)
        );
        assert_eq!(present.descriptor.target.view_id, ViewId::new(7));
        assert_eq!(present.descriptor.mode, PresentMode::Fifo);
        assert_eq!(present.descriptor.acquire, AcquirePolicy::Blocking);
        // The one-texel sentinel is replicated across the 2x2 target.
        assert_eq!(
            present.sentinel.as_deref(),
            Some([0xfe, 0xfe, 0xfe, 0xfe].repeat(4).as_slice())
        );
        // The present pass keeps the target's sentinel through a `Load`, which
        // is the load op the present path can honour (`research/docs/24` §3.1).
        assert!(matches!(
            planned.plan.attachments[0].load,
            RenderLoadAction::Load
        ));
    }

    /// The host-side half of the trace path: the plan a device-free host can
    /// check, which is the same decision the macOS encoder body then executes.
    #[test]
    fn plan_trace_plans_the_milestone_pass_and_its_landing_view() {
        let (trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.pass.color_attachments[0].view_id, ViewId::new(7));
        assert_eq!(planned.contract, &milestone_pipeline());
        assert_eq!(planned.plan.extent, [2, 2]);
        assert_eq!(planned.plan.texel_bytes, 16);
        assert_eq!(planned.plan.row_pitch, 8);
        assert_eq!(
            planned.plan.attachments[0].format,
            RenderPixelFormat::Rgba8Unorm
        );
        assert_eq!(planned.plan.vertices, 3);
        assert_eq!(planned.plan.attachments[0].initial, None);
        // The landing view is the declaration's own identity and range, so the
        // writeback is the one the trace asked for and no second channel is
        // invented.
        assert_eq!(planned.landings[0].view_id, ViewId::new(7));
        assert_eq!(planned.landings[0].allocation_id, AllocationId::new(9));
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
        });
        let [writeback] = writebacks.as_slice() else {
            panic!("the milestone attachment becomes one writeback");
        };
        assert_eq!(writeback.view_id, ViewId::new(7));
        assert_eq!(writeback.allocation_id, AllocationId::new(9));
        assert_eq!(writeback.offset, 0);
        assert_eq!(writeback.bytes, [0x40, 0x80, 0xc0, 0xff].repeat(4));

        // A compute-only trace keeps the pre-render path: nothing to plan, no
        // new refusal and no attachment readback.
        let mut compute_only = trace.clone();
        compute_only
            .passes
            .retain(|pass| pass.as_compute().is_some());
        assert!(plan_trace(&compute_only, &pool, &contracts, 0, 0)
            .expect("a compute-only trace plans nothing")
            .is_empty());
    }

    /// A loading pass uploads the bytes its declaring view owns: the plan keeps
    /// `Load` as the load action and carries those bytes, which is what the
    /// encoder presets into the attachment instead of clearing it
    /// (`research/docs/23` §3.3).
    #[test]
    fn plan_trace_plans_a_loading_pass_and_its_previous_bytes() {
        let (trace, _) = milestone_trace(LoadOp::Load);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the declaring view's own bytes are what a load uploads");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::Load);
        // The declaration's 16-byte range is the tightly packed 2x2 rgba8
        // texels, so the plan carries exactly those bytes.
        assert_eq!(
            planned.plan.attachments[0].initial,
            Some([0xfe; 16].as_slice())
        );
    }

    /// A `DontCare` attachment still needs its landing declaration — the stored
    /// texels land through the writeback channel — but resolves no previous
    /// bytes, so the plan carries `DontCare` and presets nothing: the output is
    /// the draw alone (`research/docs/23` §3.1, v20).
    #[test]
    fn plan_trace_plans_a_dont_care_pass_without_declared_bytes() {
        let (trace, _) = milestone_trace(LoadOp::DontCare);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the declaring view lands the writeback; only the bytes are absent");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::DontCare);
        assert_eq!(
            planned.plan.attachments[0].initial, None,
            "a DontCare attachment presets nothing"
        );
    }

    /// A lease-backed declaration carries bytes this rail does not hold, so a
    /// loading pass is refused by name rather than executed as a clear.
    #[test]
    fn plan_trace_refuses_a_leased_attachment_load() {
        let (trace, _) = milestone_trace(LoadOp::Load);
        let mut pool = trace.serial_resources().expect("admitted serial pool");
        let view = pool
            .iter_mut()
            .find(|view| {
                view.view_id == ViewId::new(7) && view.allocation_id == AllocationId::new(9)
            })
            .expect("the milestone trace declares the attachment view");
        view.source = BufferSource::StagedLease(LeaseId::new(5));
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "attachment_load_op_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("load_op"),
            Some(&FieldValue::Text("load".to_owned()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
    }

    /// The landing rail is the writeback channel: an attachment no declared view
    /// covers has nowhere to land, so it is refused instead of executed and
    /// dropped.
    #[test]
    fn plan_trace_refuses_an_attachment_without_a_landing_view() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.color_attachments[0].view_id = ViewId::new(8);
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(8)),
            "the refusal names the attachment that has no landing rail"
        );

        // A loading pass needs the declaration twice over — as its landing and
        // as the source of the bytes it uploads — so an undeclared attachment
        // is refused under the same slug rather than planned without bytes.
        let (mut loading, _) = milestone_trace(LoadOp::Load);
        let Some(TracePass::Render(pass)) = loading.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.color_attachments[0].view_id = ViewId::new(8);
        let error = plan_trace(&loading, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
    }

    /// The v43 increment over the trace path: a depth attachment the pass
    /// stores resolves its declared view as a second landing, and the stored
    /// texels become the writeback that follows the colour ones in the same
    /// channel (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_trace_plans_a_stored_depth_pass_and_its_landing_view() {
        let trace = depth_store_trace(Some(DepthStoreOp::Store));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the stored depth surface lands in its declaring view");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        let Some(depth) = &planned.plan.depth else {
            panic!("the pass opens a depth attachment");
        };
        assert_eq!(depth.store, Some(DepthStoreOp::Store));
        // The landing is the declaration's own identity and range, so the
        // stored texels leave through the view the trace named and no second
        // channel is invented (`research/docs/23` §3.3, v43).
        let landing = planned
            .depth_landing
            .expect("a stored depth attachment names its landing view");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        assert_eq!(landing.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(landing.offset, 0);
        assert_eq!(landing.length, DEPTH_STORE_BYTES);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: Some(DEPTH_STORE_TEXEL.repeat(4)),
            stencil: None,
        });
        let [colour, depth] = writebacks.as_slice() else {
            panic!("the stored depth attachment becomes a second writeback");
        };
        assert_eq!(colour.view_id, ViewId::new(7));
        assert_eq!(colour.allocation_id, AllocationId::new(9));
        assert_eq!(colour.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(depth.view_id, DEPTH_STORE_VIEW);
        assert_eq!(depth.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(depth.offset, 0);
        assert_eq!(depth.bytes, DEPTH_STORE_TEXEL.repeat(4));
    }

    /// The v57c trace shape: the stored depth pair over a four-sample raster
    /// whose resolve the device mask admits. The stored surface's landing is
    /// the same v43 view — the resolve target is a rail-owned texture, not a
    /// second trace identity — and a mask without the filter bit refuses the
    /// pass at plan time, before any Metal object exists
    /// (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_trace_plans_a_resolving_depth_pass_when_the_device_mask_admits_it() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Sample0,
        });
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, DEPTH_RESOLVE_SAMPLE0_BIT, 0)
            .expect("the resolving pass plans in the admitted mask");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        assert_eq!(planned.plan.multisample, Some(SampleCount::Four));
        assert_eq!(
            planned.plan.depth_resolve,
            Some(DepthResolveFilter::Sample0)
        );
        let landing = planned
            .depth_landing
            .expect("the stored depth surface names its v43 landing");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        assert_eq!(landing.allocation_id, DEPTH_STORE_ALLOCATION);

        // The same trace through a mask without the Sample0 bit is refused by
        // the per-filter question the capability snapshot answered.
        let refused = plan_trace(&trace, &pool, &contracts, 0, 0).unwrap_err();
        assert_eq!(refused.slug, "render_depth_resolve_filter_unsupported");
    }

    /// The two shapes that state no store keep the surface rail-owned: no
    /// landing is resolved, and a readback that carries no depth texels adds no
    /// second writeback — the pre-v43 byte shape exactly
    /// (`research/docs/23` §3.3, v43).
    ///
    /// The depth-only shape is the same pass with every colour attachment
    /// discarding (`research/docs/23` §3.3, v45): the plan still carries the
    /// colour attachment — it renders, its bytes disappear — and the stored
    /// depth surface is the whole observation, so the readback is a depth one
    /// and nothing else.
    #[test]
    fn plan_trace_plans_a_depth_only_pass() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        let pass = trace
            .passes
            .iter_mut()
            .find_map(|pass| match pass {
                TracePass::Render(pass) => Some(pass),
                TracePass::Compute(_) => None,
            })
            .expect("the fixture carries a render pass");
        pass.color_attachments[0].store = StoreOp::DontCare;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("a pass whose only landing is its depth surface plans");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        assert_eq!(
            planned.plan.attachments[0].store,
            RenderStoreAction::DontCare
        );
        let landing = planned
            .depth_landing
            .expect("the stored depth surface is the pass's landing");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        // The readback carries no colour texels — the discarded attachment
        // lands nothing — and the depth texels become the pass's one
        // writeback, in the depth view the trace declared.
        let writebacks = planned.writebacks(RenderReadback {
            attachments: Vec::new(),
            depth: Some(DEPTH_STORE_TEXEL.repeat(4)),
            stencil: None,
        });
        let [depth] = writebacks.as_slice() else {
            panic!("a depth-only pass lands exactly one writeback");
        };
        assert_eq!(depth.view_id, DEPTH_STORE_VIEW);
        assert_eq!(depth.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(depth.bytes, DEPTH_STORE_TEXEL.repeat(4));
    }

    /// The discard shapes that state no depth store keep the pre-v45 refusal.
    #[test]
    fn plan_trace_keeps_a_discarded_depth_attachment_out_of_the_writebacks() {
        for store in [None, Some(DepthStoreOp::DontCare)] {
            let trace = depth_store_trace(store);
            let pool = trace.serial_resources().expect("admitted serial pool");
            let contracts = milestone_contracts();
            let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
                .expect("a discarded depth surface needs no landing");
            let [planned] = planned.as_slice() else {
                panic!("the depth trace carries one render pass");
            };
            let Some(depth) = &planned.plan.depth else {
                panic!("the pass opens a depth attachment");
            };
            assert_eq!(depth.store, store);
            assert!(
                planned.depth_landing.is_none(),
                "{store:?} keeps the depth surface rail-owned"
            );
            let writebacks = planned.writebacks(RenderReadback {
                attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
                depth: None,
                stencil: None,
            });
            let [writeback] = writebacks.as_slice() else {
                panic!("{store:?} lands the colour attachment alone");
            };
            assert_eq!(writeback.view_id, ViewId::new(7));
            assert_eq!(writeback.allocation_id, AllocationId::new(9));
            assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        }
    }

    /// A storing depth attachment no declared view covers has nowhere to land,
    /// so the pass is refused by name instead of executed and dropped — the
    /// shape the colour attachments' landing refusal already has
    /// (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_trace_refuses_a_stored_depth_attachment_without_a_landing_view() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        // The depth declaration is what the landing resolution reads: without
        // it the trace declares no view covering the stored surface, and the
        // colour attachment's own declaration stays in place. Core admission
        // states the same requirement one level up, which is why the trace
        // itself no longer resolves to a pool.
        trace.passes.retain(|pass| {
            !matches!(
                pass,
                TracePass::Compute(compute)
                    if compute
                        .buffers
                        .iter()
                        .any(|view| view.view_id == DEPTH_STORE_VIEW)
            )
        });
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::AttachmentViewUnknown { .. })
        ));
        // The rail's own walk keeps the refusal for a plan that never went
        // through admission: the pool below is the shape the colour attachment
        // alone declares, so the stored depth surface is the only landing left
        // without a view.
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_depth_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(DEPTH_STORE_VIEW.get())),
            "the refusal names the depth view that has no landing rail"
        );
        assert_eq!(
            error.fields.get("allocation"),
            Some(&FieldValue::Unsigned(DEPTH_STORE_ALLOCATION.get()))
        );
    }

    /// The v49 increment over the trace path: a stencil attachment the pass
    /// stores resolves its declared view as a second landing beside the depth
    /// one, and the stored one-byte texels become the writeback that follows
    /// the colour ones in the same channel (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_plans_a_stored_stencil_pass_and_its_landing_view() {
        let trace = stencil_store_trace(Some(StoreOp::Store));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = stencil_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the stored stencil surface lands in its declaring view");
        let [planned] = planned.as_slice() else {
            panic!("the stencil trace carries one render pass");
        };
        let Some(stencil) = &planned.plan.stencil else {
            panic!("the pass opens a stencil attachment");
        };
        assert_eq!(stencil.store, Some(StoreOp::Store));
        // The landing is the declaration's own identity and range, so the
        // stored texels leave through the view the trace named and no second
        // channel is invented (`research/docs/23` §3.3, v49).
        let landing = planned
            .stencil_landing
            .expect("a stored stencil attachment names its landing view");
        assert_eq!(landing.view_id, STENCIL_STORE_VIEW);
        assert_eq!(landing.allocation_id, STENCIL_STORE_ALLOCATION);
        assert_eq!(landing.offset, 0);
        assert_eq!(landing.length, STENCIL_STORE_BYTES);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: Some(vec![STENCIL_STORE_TEXEL; 4]),
        });
        let [colour, stencil] = writebacks.as_slice() else {
            panic!("the stored stencil attachment becomes a second writeback");
        };
        assert_eq!(colour.view_id, ViewId::new(7));
        assert_eq!(colour.allocation_id, AllocationId::new(9));
        assert_eq!(colour.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(stencil.view_id, STENCIL_STORE_VIEW);
        assert_eq!(stencil.allocation_id, STENCIL_STORE_ALLOCATION);
        assert_eq!(stencil.offset, 0);
        assert_eq!(stencil.bytes, vec![STENCIL_STORE_TEXEL; 4]);
    }

    /// The two shapes that state no stencil store keep the surface rail-owned:
    /// no landing is resolved, and a readback that carries no stencil texels
    /// adds no second writeback — the pre-v49 byte shape exactly
    /// (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_keeps_an_unstored_stencil_attachment_out_of_the_writebacks() {
        for store in [None, Some(StoreOp::DontCare)] {
            let trace = stencil_store_trace(store);
            let pool = trace.serial_resources().expect("admitted serial pool");
            let contracts = stencil_contracts();
            let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
                .expect("a discarded stencil surface needs no landing");
            let [planned] = planned.as_slice() else {
                panic!("the stencil trace carries one render pass");
            };
            let Some(stencil) = &planned.plan.stencil else {
                panic!("the pass opens a stencil attachment");
            };
            assert_eq!(stencil.store, store);
            assert!(
                planned.stencil_landing.is_none(),
                "{store:?} keeps the stencil surface rail-owned"
            );
            let writebacks = planned.writebacks(RenderReadback {
                attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
                depth: None,
                stencil: None,
            });
            let [writeback] = writebacks.as_slice() else {
                panic!("{store:?} lands the colour attachment alone");
            };
            assert_eq!(writeback.view_id, ViewId::new(7));
            assert_eq!(writeback.allocation_id, AllocationId::new(9));
            assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        }
    }

    /// A storing stencil attachment no declared view covers has nowhere to
    /// land, so the pass is refused by name instead of executed and dropped —
    /// the shape the colour and depth attachments' landing refusals already
    /// have (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_refuses_a_stored_stencil_attachment_without_a_landing_view() {
        let mut trace = stencil_store_trace(Some(StoreOp::Store));
        // The stencil declaration is what the landing resolution reads: without
        // it the trace declares no view covering the stored surface, and the
        // colour attachment's own declaration stays in place. Core admission
        // states the same requirement one level up, which is why the trace
        // itself no longer resolves to a pool.
        trace.passes.retain(|pass| {
            !matches!(
                pass,
                TracePass::Compute(compute)
                    if compute
                        .buffers
                        .iter()
                        .any(|view| view.view_id == STENCIL_STORE_VIEW)
            )
        });
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::AttachmentViewUnknown { .. })
        ));
        // The rail's own walk keeps the refusal for a plan that never went
        // through admission: the pool below is the shape the colour attachment
        // alone declares, so the stored stencil surface is the only landing
        // left without a view.
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &stencil_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_stencil_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(STENCIL_STORE_VIEW.get())),
            "the refusal names the stencil view that has no landing rail"
        );
        assert_eq!(
            error.fields.get("allocation"),
            Some(&FieldValue::Unsigned(STENCIL_STORE_ALLOCATION.get()))
        );
    }

    /// The ordering rule the trace path shares with the Vulkan rail: every
    /// compute pass runs before every render pass, so a compute pass that
    /// follows a render store of a view it binds would read pre-render bytes.
    ///
    /// Core admission states that order as part of the contract (review item
    /// I4, 2026-09-14), so the first half observes the refusal from
    /// `serial_resources` — the value-level entry point that runs admission —
    /// and the second half keeps the rail's own walk covered as defense in
    /// depth for a plan that never went through admission.
    #[test]
    fn plan_trace_refuses_a_compute_pass_that_reads_after_a_render_store() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        trace.passes.push(TracePass::Compute(declaration_pass()));
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::RenderPassOrderUnsupported { .. })
        ));
        let error = refuse_reordered_render_reads(&trace).unwrap_err();
        assert_eq!(error.slug, "render_pass_order_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);

        // The same rule in its legal direction: a declaration that comes before
        // the store is planned, not refused.
        let (legal, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let pool = legal.serial_resources().expect("admitted serial pool");
        assert_eq!(
            plan_trace(&legal, &pool, &milestone_contracts(), 0, 0)
                .unwrap()
                .len(),
            1
        );
    }

    /// A render registration is a review gate: an entry pair the reviewed module
    /// does not carry cannot be registered, the same way an unreviewed MSL
    /// fixture cannot be compiled.
    #[test]
    fn a_render_contract_has_to_name_the_reviewed_entries() {
        assert_eq!(review_contract(&milestone_pipeline()), Ok(()));
        let mut edited = milestone_pipeline();
        edited.fragment_entry = "render_solid_rgba8_v2".to_owned();
        let error = review_contract(&edited).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Compile);
    }

    /// The merge that puts the two rails' bytes in one channel: one writeback
    /// per written view, in the canonical order the submission protocol
    /// requires, with the render bytes winning because the render rail runs
    /// last.
    #[test]
    fn render_writebacks_replace_the_compute_bytes_of_the_same_view() {
        let later_view = BufferWriteback {
            view_id: ViewId::new(2),
            allocation_id: AllocationId::new(1),
            offset: 16,
            bytes: vec![0x11; 4],
        };
        let attachment = BufferWriteback {
            view_id: ViewId::new(1),
            allocation_id: AllocationId::new(1),
            offset: 0,
            bytes: vec![0xfe; 16],
        };
        let rendered = BufferWriteback {
            view_id: ViewId::new(1),
            allocation_id: AllocationId::new(1),
            offset: 0,
            bytes: [0x40, 0x80, 0xc0, 0xff].repeat(4),
        };
        let merged = merge_writebacks(vec![later_view.clone(), attachment], vec![rendered.clone()]);
        assert_eq!(merged, vec![rendered, later_view]);
    }

    /// The vertex-input bits the provider declares have to be the rail's own
    /// limits, and core admission has to admit exactly the trace the rail plans
    /// — the same agreement the render and present bits are held to. The
    /// pre-flip snapshot is the falsifiable half: with the three bits at their
    /// defaults the same trace is refused during admission instead of being
    /// executed with positions the trace did not ask for.
    #[test]
    fn declared_vertex_input_bits_admit_what_the_rail_plans() {
        let bits = vertex_input_capability_bits();
        assert_eq!(bits.max_vertex_buffers, MAX_VERTEX_BUFFERS);
        // The rail's limits are core's own values, not a second spelling that
        // could drift from the contract's.
        assert_eq!(
            bits.max_vertex_buffers,
            u32::try_from(metal_api_core::provider::MAX_VERTEX_BUFFERS).unwrap()
        );
        assert_eq!(
            bits.supported_vertex_formats,
            VertexFormat::ADMITTED.to_vec()
        );
        assert_eq!(bits.supported_index_formats, IndexFormat::ADMITTED.to_vec());
        // The four translated formats and both index widths are the whole
        // admitted set, which is what makes the mapping total.
        assert_eq!(
            bits.supported_vertex_formats.len(),
            VertexFormat::ADMITTED.len()
        );
        assert_eq!(
            bits.supported_index_formats.len(),
            IndexFormat::ADMITTED.len()
        );

        let (trace, resources) = quad_trace();
        capabilities(&capability_bits())
            .admit(&trace, &resources)
            .expect("the declared bits admit the vertex-input trace");

        // Closed stream count: the pass's binding is refused before its formats
        // are read, which is the order capability admission documents.
        let mut closed = vertex_input_capability_bits();
        closed.max_vertex_buffers = 0;
        let refused = capabilities_with(&capability_bits(), &closed)
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "vertex_buffer_limit");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("maximum"),
            Some(&FieldValue::Unsigned(0))
        );

        // Closed index widths: the same trace, refused one gate later.
        let mut no_indices = vertex_input_capability_bits();
        no_indices.supported_index_formats = Vec::new();
        let refused = capabilities_with(&capability_bits(), &no_indices)
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "index_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);

        // Closed vertex formats: the layout's attribute format is the fact this
        // gate reads, after the pass and the layout already agreed.
        let mut no_formats = vertex_input_capability_bits();
        no_formats.supported_vertex_formats = Vec::new();
        let refused = capabilities_with(&capability_bits(), &no_formats)
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "vertex_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
    }

    /// The descriptor translation: what the rail hands Metal is the layout's
    /// binding order, stride and attributes, plus the digits' formats and the
    /// index width. Every value here is what the macOS encoder writes into its
    /// `MTLVertexDescriptor` and its draw call.
    #[test]
    fn plan_translates_the_indexed_layout_into_a_descriptor_plan() {
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let planned = plan(&quad_request(&pass, &pipeline), 0, 0).expect("the reviewed pass plans");

        assert_eq!(planned.source, REVIEWED_VERTEX_SOURCE);
        assert_eq!(
            planned.module_path,
            "conformance/shaders/quad_indexed_2x2.metal"
        );
        assert_eq!(planned.vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, FRAGMENT_ENTRY);
        assert_eq!(planned.vertices, 6);

        let [stream] = planned.vertex_streams.as_slice() else {
            panic!("the reviewed layout binds one stream");
        };
        assert_eq!(stream.buffer_index, 0);
        assert_eq!(stream.stride, 8);
        assert_eq!(
            stream.attributes,
            vec![PlannedVertexAttribute {
                location: 0,
                offset: 0,
                format: RenderVertexFormat::Float2,
            }]
        );
        assert_eq!(stream.offset, 0);
        assert_eq!(stream.bytes, quad_vertex_bytes());

        let indices = planned.indices.as_ref().expect("the pass is indexed");
        assert_eq!(indices.format, RenderIndexType::Uint16);
        assert_eq!(indices.index_count, 6);
        assert_eq!(indices.vertex_span, 4);
        assert_eq!(indices.offset, 0);
        assert_eq!(indices.bytes, quad_index_bytes());

        // The format mappings the encoder reads these values through.
        for (format, expected) in [
            (VertexFormat::Float32x2, RenderVertexFormat::Float2),
            (VertexFormat::Float32x3, RenderVertexFormat::Float3),
            (VertexFormat::Float32x4, RenderVertexFormat::Float4),
            (VertexFormat::Uint32, RenderVertexFormat::Uint),
        ] {
            assert_eq!(vertex_format(format), expected);
        }
        assert_eq!(RenderVertexFormat::Float2.name(), "float32x2");
        assert_eq!(RenderVertexFormat::Float3.name(), "float32x3");
        assert_eq!(RenderVertexFormat::Float4.name(), "float32x4");
        assert_eq!(RenderVertexFormat::Uint.name(), "uint32");
        for (format, expected) in [
            (IndexFormat::Uint16, RenderIndexType::Uint16),
            (IndexFormat::Uint32, RenderIndexType::Uint32),
        ] {
            assert_eq!(index_type(format), expected);
            // The footprint proof and the draw's width read the same value.
            assert_eq!(expected.bytes(), format.bytes());
        }
        assert_eq!(RenderIndexType::Uint16.name(), "uint16");
        assert_eq!(RenderIndexType::Uint32.name(), "uint32");
    }

    /// A stream's bytes travel with the pass, so the rail needs no compute
    /// declaration to read them (`research/docs/23` §3.6) — what it does need is
    /// bytes it holds, which a lease-backed view does not carry here.
    #[test]
    fn plan_refuses_a_stream_whose_bytes_this_rail_does_not_hold() {
        // The reviewed pass with its stream bytes declared by no pass at all: the
        // device-level helper's shape, which the vertex-input increment admits.
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let planned = plan_pass(&quad_request(&pass, &pipeline))
            .expect("a render input carries its own bytes");
        assert_eq!(planned.vertex_streams[0].bytes, quad_vertex_bytes());
        assert_eq!(
            planned.indices.as_ref().map(|indices| indices.bytes),
            Some(quad_index_bytes().as_slice())
        );

        // A lease-backed view carries bytes this rail does not hold: the render
        // path has no lease resolver, so the stream is refused by name and the
        // storage mode it arrived with is part of the refusal.
        let mut leased_pass = quad_pass();
        leased_pass.vertex_buffers[0].source = BufferSource::StagedLease(LeaseId::new(5));
        let error = plan(&quad_request(&leased_pass, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_vertex_buffer_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(QUAD_VERTEX_VIEW.get()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );

        // The index half is refused under its own slug, so a capture can tell
        // which of the two inputs the rail could not read.
        let mut borrowed_pass = quad_pass();
        borrowed_pass.indices.as_mut().unwrap().view.source =
            BufferSource::BorrowedNoCopy(LeaseId::new(6));
        let error = plan(&quad_request(&borrowed_pass, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_buffer_unsupported");
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
    }

    /// The footprint proof `research/docs/23` §3.3 asks for: every vertex and
    /// every index the draw reads has to be inside the bytes the trace
    /// declared, and every index value has to select a vertex the streams
    /// cover. Metal reads past a short buffer without refusing, so the rail
    /// refuses first.
    ///
    /// For an indexed draw the stream-coverage rule and the index-value rule
    /// are the same condition (`stride * (highest + 1) <= bytes`), so the
    /// refusal names the index that reached past the stream; the stream's own
    /// footprint is the refusal a non-indexed draw gets, where the pass's count
    /// is the only bound.
    #[test]
    fn plan_refuses_a_stream_that_does_not_cover_the_draw() {
        // 24 bytes cover three of the four vertices the index values select.
        let mut short_stream = quad_pass();
        shorten(&mut short_stream.vertex_buffers[0], 24);
        let error = plan(&quad_request(&short_stream, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(3))
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(3))
        );
        assert_eq!(
            error.fields.get("buffer_index"),
            Some(&FieldValue::Unsigned(0))
        );

        // A non-indexed draw reads its vertex count in order, so the same
        // stream has to cover `vertices * stride` instead of the index span.
        let mut non_indexed = quad_pass();
        non_indexed.indices = None;
        non_indexed.vertices = 4;
        shorten(&mut non_indexed.vertex_buffers[0], 24);
        let error = plan(&quad_request(&non_indexed, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_vertex_footprint_unsupported");
        assert_eq!(
            error.fields.get("required_bytes"),
            Some(&FieldValue::Unsigned(32))
        );

        // 10 bytes cannot hold the six `uint16` indices.
        let mut short_indices = quad_pass();
        shorten(&mut short_indices.indices.as_mut().unwrap().view, 10);
        let error = plan(&quad_request(&short_indices, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_footprint_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("required_bytes"),
            Some(&FieldValue::Unsigned(12))
        );

        // An index value at or above the four vertices the stream covers.
        let mut out_of_range = quad_pass();
        replace_bytes(
            &mut out_of_range.indices.as_mut().unwrap().view,
            [0_u16, 1, 2, 2, 1, 9]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        );
        let error = plan(&quad_request(&out_of_range, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(9)),
            "the refusal names the largest index value the draw reads"
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(4)),
            "the refusal names the vertices the stream covers"
        );
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(QUAD_INDEX_VIEW.get()))
        );
    }

    /// The `vertex_id` shape indexed through a buffer instead of drawn in
    /// order: the same reviewed module, three indices over three generated
    /// positions. Its index values are bounded by the module's own positions,
    /// because no stream carries a vertex count.
    #[test]
    fn an_indexed_vertex_id_draw_is_bounded_by_the_modules_positions() {
        let (trace, pool) = vertex_id_indexed_trace([0, 1, 2]);
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("three indices over three generated positions plan");
        let [planned] = planned.as_slice() else {
            panic!("the computed milestone trace carries one render pass");
        };
        assert!(planned.plan.vertex_streams.is_empty());
        let indices = planned.plan.indices.as_ref().expect("the pass is indexed");
        assert_eq!(indices.index_count, 3);
        assert_eq!(indices.vertex_span, 3);

        // The fourth position the module does not carry is refused by value,
        // before a driver would read it.
        let (trace, pool) = vertex_id_indexed_trace([0, 1, 9]);
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(9))
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(u64::from(
                FULL_SCREEN_TRIANGLE_VERTICES
            )))
        );
    }

    /// The trace path over the vertex-input fixture: the same plan the macOS
    /// encoder consumes. The streams carry their own bytes, so only the
    /// attachment's landing view comes from the serial pool, and the writeback
    /// lands where the trace declared it.
    #[test]
    fn plan_trace_plans_the_indexed_pass_and_its_landing_view() {
        let (trace, _) = quad_trace();
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = quad_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed indexed pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the vertex-input trace carries one render pass");
        };
        assert_eq!(planned.contract, &quad_pipeline());
        assert_eq!(planned.plan.vertices, 6);
        assert_eq!(planned.plan.vertex_streams.len(), 1);
        assert_eq!(planned.plan.vertex_streams[0].bytes, quad_vertex_bytes());
        assert_eq!(
            planned
                .plan
                .indices
                .as_ref()
                .map(|indices| indices.index_count),
            Some(6)
        );
        assert_eq!(planned.landings[0].view_id, ViewId::new(7));
        assert_eq!(planned.landings[0].allocation_id, AllocationId::new(9));
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
        });
        let [writeback] = writebacks.as_slice() else {
            panic!("the indexed attachment becomes one writeback");
        };
        assert_eq!(writeback.view_id, ViewId::new(7));
        assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
    }

    /// The trace path over the dual fixture: one landing view per colour
    /// location, resolved from the serial pool in location order, and one
    /// writeback per landing view when the encoder's per-attachment readbacks
    /// come back.
    #[test]
    fn plan_trace_plans_the_dual_pass_with_a_landing_per_location() {
        let (trace, _) = dual_trace(LoadOp::Clear(sentinel()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed dual pass plans");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.contract, &dual_pipeline());
        assert_eq!(planned.plan.attachments.len(), 2);
        assert_eq!(planned.landings[0].view_id, ViewId::new(7));
        assert_eq!(planned.landings[0].allocation_id, AllocationId::new(9));
        assert_eq!(planned.landings[1].view_id, ViewId::new(8));
        assert_eq!(planned.landings[1].allocation_id, AllocationId::new(10));
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![
                EXPECTED_TEXEL_BYTES.repeat(4),
                [0xff, 0x80, 0x40, 0xc0].repeat(4),
            ],
            depth: None,
            stencil: None,
        });
        assert_eq!(writebacks.len(), 2);
        assert_eq!(writebacks[0].view_id, ViewId::new(7));
        assert_eq!(writebacks[0].bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(writebacks[1].view_id, ViewId::new(8));
        assert_eq!(writebacks[1].bytes, [0xff, 0x80, 0x40, 0xc0].repeat(4));
    }

    /// The v19 store increment over the trace path: location 1 discards, so
    /// its landing view stays resolved for the load side but produces no
    /// writeback — only the stored attachment's texels leave the pass
    /// (`research/docs/23` §3.6).
    #[test]
    fn plan_trace_drops_a_discarded_attachment_from_the_writebacks() {
        let (mut trace, _) = dual_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the dual fixture ends with its render pass");
        };
        pass.color_attachments[1].store = StoreOp::DontCare;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed dual pass plans");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].store, RenderStoreAction::Store);
        assert_eq!(
            planned.plan.attachments[1].store,
            RenderStoreAction::DontCare
        );
        // Both landings are still resolved — the discarded attachment keeps its
        // declaring view for the load-side resolution — but only the stored
        // attachment's readback becomes a writeback.
        assert_eq!(planned.landings.len(), 2);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
        });
        assert_eq!(writebacks.len(), 1);
        assert_eq!(writebacks[0].view_id, ViewId::new(7));
        assert_eq!(writebacks[0].allocation_id, AllocationId::new(9));
        assert_eq!(writebacks[0].offset, 0);
        assert_eq!(writebacks[0].bytes, EXPECTED_TEXEL_BYTES.repeat(4));
    }

    /// A loading dual pass uploads each attachment's own declaring bytes: the
    /// plan resolves the two previous-bytes entries independently, location 0
    /// from view 7 and location 1 from view 8.
    #[test]
    fn plan_trace_resolves_the_previous_bytes_per_location() {
        let (trace, _) = dual_trace(LoadOp::Load);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("each declaring view's own bytes are what a load uploads");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::Load);
        assert_eq!(
            planned.plan.attachments[0].initial,
            Some([0xfe; 16].as_slice())
        );
        assert_eq!(planned.plan.attachments[1].load, RenderLoadAction::Load);
        assert_eq!(
            planned.plan.attachments[1].initial,
            Some([0xfd; 16].as_slice())
        );
    }

    /// The lease refusal reuses `previous_bytes` per location: a lease-backed
    /// second declaration is refused with the same slug, class and phase a
    /// single attachment gets, naming the storage mode rather than executing
    /// location 1 as a clear.
    #[test]
    fn plan_trace_refuses_a_leased_second_attachment_load() {
        let (trace, _) = dual_trace(LoadOp::Load);
        let mut pool = trace.serial_resources().expect("admitted serial pool");
        let view = pool
            .iter_mut()
            .find(|view| {
                view.view_id == ViewId::new(8) && view.allocation_id == AllocationId::new(10)
            })
            .expect("the dual trace declares the second attachment view");
        view.source = BufferSource::StagedLease(LeaseId::new(5));
        let contracts = dual_contracts();
        let error = plan_trace(&trace, &pool, &contracts, 0, 0).unwrap_err();
        assert_eq!(error.slug, "attachment_load_op_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
    }

    /// A render input carries its own bytes, so the trace does not have to
    /// declare the streams through a compute binding (`research/docs/23` §3.6).
    /// What the pool — the compute rail's binding set — still carries is the
    /// attachment's landing view, which is why this trace keeps exactly one
    /// declaration and still plans the draw.
    #[test]
    fn a_trace_plans_streams_no_compute_binding_declares() {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(2),
            pipelines: vec![declaration_pipeline(), quad_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                TracePass::Render(quad_pass()),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        trace
            .validate()
            .expect("the declaration pass declares the attachment only");
        let pool = trace.serial_resources().expect("admitted serial pool");
        assert_eq!(
            pool.iter().map(|view| view.view_id).collect::<Vec<_>>(),
            vec![ViewId::new(7)],
            "neither stream is part of the compute rail's binding set"
        );

        let contracts = quad_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0).expect("the draw plans");
        let [planned] = planned.as_slice() else {
            panic!("the vertex-input trace carries one render pass");
        };
        assert_eq!(planned.plan.vertex_streams[0].bytes, quad_vertex_bytes());
        assert_eq!(
            planned.plan.indices.as_ref().map(|indices| indices.bytes),
            Some(quad_index_bytes().as_slice())
        );
    }

    /// An indirect draw replays its pass through `MTLIndirectRenderCommand`
    /// state, which carries the pipeline state and the draw counts rather than
    /// the streams a caller-held layout reads: the shape is refused before any
    /// Metal object exists, with the slug the ICB rail uses for a replay it
    /// cannot build.
    #[test]
    fn plan_trace_refuses_an_indirect_draw_of_a_stream_pass() {
        let (mut trace, _) = quad_trace();
        trace.indirect = Some(Box::new(IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 1,
                kinds: vec![IndirectCommandKind::Draw],
            },
            command: IndirectCommandDescriptor::Draw {
                vertex_count: 6,
                instance_count: 1,
            },
            range: IndirectCommandRange { start: 0, count: 1 },
        }));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &quad_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
    }

    /// The reviewed allowlist is a (module, layout, colour-format list) triple:
    /// a layout that binds streams may only compile the `[[stage_in]]` module
    /// for one location, a `vertex_id` pipeline only the module whose vertex
    /// stage reads no stream, and an indexed pipeline with two `rgba8_unorm`
    /// locations only the dual module. A caller cannot pair one shape's
    /// descriptor with another shape's source.
    #[test]
    fn a_vertex_layout_selects_the_reviewed_module() {
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let single = [AttachmentFormat::Rgba8Unorm];
        let dual = [AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm];

        assert_eq!(
            reviewed_module(&VertexLayout::None, &single).map(|module| module.source),
            Some(REVIEWED_SOURCE)
        );
        assert_eq!(
            reviewed_module(&pipeline.vertex_layout, &single).map(|module| module.source),
            Some(REVIEWED_VERTEX_SOURCE)
        );
        assert_eq!(
            reviewed_module(&pipeline.vertex_layout, &dual).map(|module| module.source),
            Some(REVIEWED_DUAL_SOURCE)
        );
        assert!(
            !reviewed_module(&VertexLayout::None, &single)
                .expect("the vertex_id shape is reviewed")
                .binds_buffers
        );
        assert!(
            reviewed_module(&pipeline.vertex_layout, &single)
                .expect("the single-output stream shape is reviewed")
                .binds_buffers
        );
        // A dual-format `vertex_id` contract has no reviewed module, and the
        // two-attachment stream shape now accepts any mix of the two 8-bit
        // layouts (v26) while refusing a list the modules cannot serve.
        assert!(reviewed_module(&VertexLayout::None, &dual).is_none());
        assert_eq!(
            reviewed_module(
                &pipeline.vertex_layout,
                &[AttachmentFormat::Rgba8Unorm, AttachmentFormat::Bgra8Unorm]
            )
            .map(|module| module.fragment_entry),
            Some(DUAL_FRAGMENT_ENTRY)
        );
        assert!(reviewed_module(
            &pipeline.vertex_layout,
            &[AttachmentFormat::Rgba8Unorm, AttachmentFormat::R32Float]
        )
        .is_none());
        assert_eq!(layout_name(&VertexLayout::None), "vertex_id");
        assert_eq!(layout_name(&pipeline.vertex_layout), "vertex-buffer");

        // The reviewed contracts are accepted, and so is the milestone's.
        assert_eq!(review_contract(&pipeline), Ok(()));
        assert_eq!(review_contract(&quad_pipeline()), Ok(()));
        assert_eq!(review_contract(&dual_pipeline()), Ok(()));

        // A dual-format `vertex_id` contract is refused by name: no reviewed
        // module derives positions from `vertex_id` and writes two locations.
        let mut wider = milestone_pipeline();
        wider.color_formats = dual.to_vec();
        let error = review_contract(&wider).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");

        // A stream layout with the `vertex_id` module's bytes, and the reverse.
        let crossed = OffscreenRenderRequest {
            source: REVIEWED_SOURCE,
            ..quad_request(&pass, &pipeline)
        };
        let error = plan(&crossed, 0, 0).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Compile);
        assert!(
            error
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("vertex-buffer layout")),
            "the refusal names the layout the source was paired with: {:?}",
            error.detail
        );

        let milestone = milestone_pipeline();
        let milestone_pass = milestone_pass(LoadOp::Clear(sentinel()));
        let crossed = OffscreenRenderRequest {
            source: REVIEWED_VERTEX_SOURCE,
            ..milestone_request(&milestone_pass, &milestone, None)
        };
        let error = plan_pass(&crossed).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");

        // The entry pair is checked against the selected module too: the
        // indexed entries with a `vertex_id` layout are not a reviewed pair.
        let mut wrong_entry = milestone_pipeline();
        wrong_entry.vertex_entry = QUAD_VERTEX_ENTRY.to_owned();
        let error = review_contract(&wrong_entry).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(
            error
                .detail
                .as_deref()
                .map(|detail| detail.contains(VERTEX_ENTRY)),
            Some(true)
        );
    }

    /// The indexed fixture is held to the same two falsifiability rules the
    /// `vertex_id` fixture is: the fragment writes a byte/255 texel with no
    /// half-integer tie, and the module carries exactly the reviewed entries.
    #[test]
    fn reviewed_indexed_fixture_matches_the_expected_texel_bytes() {
        for literal in ["64.0 / 255.0", "128.0 / 255.0", "192.0 / 255.0"] {
            assert!(
                REVIEWED_VERTEX_SOURCE.contains(literal),
                "the indexed fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_VERTEX_SOURCE.contains("0.5"),
            "the indexed fixture must not carry a half-integer tie constant"
        );
        for entry in [QUAD_VERTEX_ENTRY, FRAGMENT_ENTRY] {
            assert!(
                REVIEWED_VERTEX_SOURCE.contains(entry),
                "the indexed fixture no longer carries {entry}"
            );
        }
        // The vertex stage reads a stage-in attribute: a module without
        // `[[stage_in]]` would make the pass's descriptor meaningless.
        assert!(REVIEWED_VERTEX_SOURCE.contains("[[stage_in]]"));
        assert!(REVIEWED_VERTEX_SOURCE.contains("[[attribute(0)]]"));
        // The fixture's own bytes, spelled the way a suite's `initial_hex`
        // spells them, so a drift between this rail's fixture, the Swift
        // self-test and the suite is visible on a host without a GPU.
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hex(&quad_vertex_bytes()),
            "000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f"
        );
        assert_eq!(hex(&quad_index_bytes()), "000001000200020001000300");
        // The two single-output modules share one fragment entry, which is what
        // keeps a capture from telling the draws apart by their colour.
        assert_eq!(
            REVIEWED_MODULES[0].fragment_entry,
            REVIEWED_MODULES[1].fragment_entry
        );
        assert_eq!(REVIEWED_MODULES[0].vertex_entry, VERTEX_ENTRY);
        assert_eq!(REVIEWED_MODULES[1].vertex_entry, QUAD_VERTEX_ENTRY);
    }

    /// The dual fixture is held to the same falsifiability rules: its two
    /// fragment outputs are byte/255 constants with no half-integer tie, and
    /// the module carries exactly the reviewed entry pair.
    #[test]
    fn reviewed_dual_fixture_matches_the_expected_texel_bytes() {
        for literal in [
            "64.0 / 255.0",
            "128.0 / 255.0",
            "192.0 / 255.0",
            "255.0 / 255.0",
        ] {
            assert!(
                REVIEWED_DUAL_SOURCE.contains(literal),
                "the dual fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_DUAL_SOURCE.contains("0.5"),
            "the dual fixture must not carry a half-integer tie constant"
        );
        for entry in [QUAD_VERTEX_ENTRY, DUAL_FRAGMENT_ENTRY] {
            assert!(
                REVIEWED_DUAL_SOURCE.contains(entry),
                "the dual fixture no longer carries {entry}"
            );
        }
        assert!(REVIEWED_DUAL_SOURCE.contains("[[color(0)]]"));
        assert!(REVIEWED_DUAL_SOURCE.contains("[[color(1)]]"));
        assert_eq!(REVIEWED_MODULES[2].vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(REVIEWED_MODULES[2].fragment_entry, DUAL_FRAGMENT_ENTRY);
        assert_eq!(
            REVIEWED_MODULES[2].path,
            "conformance/shaders/quad_indexed_2x2_dual.metal"
        );
    }
}
