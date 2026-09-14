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
    ContractError, FieldValue, IndexBufferBinding, IndexFormat, IndirectCommandDescriptor, LoadOp,
    PipelineId, PresentDescriptor, PresentMode, ProviderError, ProviderErrorClass, ProviderPhase,
    RenderPassDescriptor, RenderPipelineContract, StoreOp, TracePass, VertexFormat, VertexLayout,
    ViewId, FULL_SCREEN_TRIANGLE_VERTICES,
};
use std::collections::BTreeMap;

#[cfg(target_os = "macos")]
use foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, Device, IndirectCommandBufferDescriptor, MTLClearColor,
    MTLCommandBufferStatus, MTLIndexType, MTLIndirectCommandType, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLVertexFormat, MTLVertexStepFunction,
    MTLViewport, NSRange, NSUInteger, RenderPassDescriptor as MetalRenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState, Texture, TextureDescriptor, VertexDescriptor,
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

/// One reviewed render module and the vertex-input shape it was written for.
///
/// The [`VertexLayout`] of a render pipeline selects the entry: a pipeline
/// whose layout binds streams can only be the module whose vertex stage reads
/// `[[stage_in]]`, and a `VertexLayout::None` pipeline can only be the module
/// that derives positions from `vertex_id`. Both the entry pair and the source
/// bytes of that one module are then the allowlist, so neither a renamed entry
/// nor an edited file can execute.
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

/// The two reviewed modules, one per vertex-input shape.
pub(crate) const REVIEWED_MODULES: [ReviewedModule; 2] = [
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
];

/// The reviewed module a pipeline's vertex-input shape selects.
///
/// Total by construction: [`VertexLayout`] has exactly two variants and
/// [`REVIEWED_MODULES`] carries exactly one module per variant, so there is no
/// layout this rail would compile nothing for.
pub(crate) fn reviewed_module(layout: &VertexLayout) -> &'static ReviewedModule {
    match layout {
        VertexLayout::None => &REVIEWED_MODULES[0],
        VertexLayout::Buffers(_) => &REVIEWED_MODULES[1],
    }
}

/// Colour attachments the first render increment admits. The same value the
/// core contract states (`metal_api_core::provider::MAX_COLOR_ATTACHMENTS`); it
/// is restated here because a capability value has to be spelled by the provider
/// that declares it (`research/docs/23` §4.2).
pub(crate) const MAX_COLOR_ATTACHMENTS: u32 = 1;

/// Vertex streams one render pass may bind. The same value the core contract
/// states (`metal_api_core::provider::MAX_VERTEX_BUFFERS`), restated for the
/// same reason [`MAX_COLOR_ATTACHMENTS`] is: a capability value belongs to the
/// provider that declares it (`research/docs/23` §3.3, §4.2).
pub(crate) const MAX_VERTEX_BUFFERS: u32 = metal_api_core::provider::MAX_VERTEX_BUFFERS as u32;

/// Largest attachment the first milestone renders into: 2x2, so full coverage
/// stays distinguishable from "one texel was written" (`research/docs/23` §1.3).
pub(crate) const MAX_ATTACHMENT_DIMENSION: [u64; 2] = [2, 2];

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
}

/// The store action this rail sets. One value, because the contract's
/// `RenderAttachment::validate_shape` refuses `StoreOp::DontCare`: a discarded
/// attachment must not be able to pass as "landed correctly"
/// (`research/docs/23` §3.6).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderStoreAction {
    Store,
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
        LoadOp::DontCare => Err(capability_refusal("attachment_load_op_unsupported")),
    }
}

/// The store action an encoder has to set for this pass.
pub(crate) fn store_action(store: StoreOp) -> Result<RenderStoreAction, ProviderError> {
    match store {
        StoreOp::Store => Ok(RenderStoreAction::Store),
        StoreOp::DontCare => Err(capability_refusal("attachment_store_op_unsupported")),
    }
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
    let indices = match &pass.indices {
        None => None,
        Some(binding) => Some(plan_index_stream(binding, pass.vertices)?),
    };
    match &indices {
        // An indexed draw reads the vertices its index values select, so each
        // stream has to cover that span. The refusal names the index that
        // reached past the stream, because that is what the trace has to change.
        Some(indices) => {
            for (buffer_index, stream) in streams.iter().enumerate() {
                // `stride == 0` cannot reach here (`plan` re-runs the layout
                // validator), so `checked_div` is only the safe spelling of the
                // quotient: a zero stride would be refused upstairs rather than
                // read as an unbounded stream.
                let covered = u64::try_from(stream.bytes.len())
                    .unwrap_or(u64::MAX)
                    .checked_div(stream.stride)
                    .unwrap_or(0);
                if indices.vertex_span > covered {
                    return Err(
                        index_value_refusal(highest_index(indices.vertex_span), covered)
                            .with_field(
                                "buffer_index",
                                FieldValue::Unsigned(
                                    u64::try_from(buffer_index).unwrap_or(u64::MAX),
                                ),
                            )
                            .with_field("view", FieldValue::Unsigned(indices.view_id.get())),
                    );
                }
            }
            // The `vertex_id` shape binds no stream to bound its index values:
            // the reviewed module generates exactly
            // `FULL_SCREEN_TRIANGLE_VERTICES` positions, so an index at or above
            // that count would read a position the module does not carry.
            if pass.vertex_buffers.is_empty()
                && indices.vertex_span > u64::from(FULL_SCREEN_TRIANGLE_VERTICES)
            {
                return Err(index_value_refusal(
                    highest_index(indices.vertex_span),
                    u64::from(FULL_SCREEN_TRIANGLE_VERTICES),
                )
                .with_field("view", FieldValue::Unsigned(indices.view_id.get())));
            }
        }
        // A non-indexed draw reads vertices `0..vertices` in order, so every
        // stream has to cover the pass's own count.
        None => {
            for (buffer_index, stream) in streams.iter().enumerate() {
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
    /// The MSL module to compile. The pipeline's [`VertexLayout`] selects which
    /// reviewed module is the only one accepted here — `vertex_id` positions
    /// ([`REVIEWED_SOURCE`]) or `[[stage_in]]` positions
    /// ([`REVIEWED_VERTEX_SOURCE`]) — so a caller cannot pair one shape's
    /// descriptor with the other shape's module.
    pub(crate) source: &'a str,
    /// Tightly packed texels the attachment already holds, for [`LoadOp::Load`].
    /// Required exactly then, refused for a clear.
    pub(crate) initial: Option<&'a [u8]>,
    /// Whether the pass hands its attachment on through a present action.
    /// A present pass's `Load` keeps the present target's initial state — the
    /// sentinel preset by the present path, or undefined — so no `initial`
    /// bytes are required here (`research/docs/24` §3.1).
    pub(crate) present: bool,
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
    pub(crate) format: RenderPixelFormat,
    /// Attachment extent in texels, as `[width, height]`.
    pub(crate) extent: [u32; 2],
    /// `[origin_x, origin_y, width, height]`, copied from the validated pass.
    pub(crate) viewport: [u32; 4],
    pub(crate) load: RenderLoadAction,
    pub(crate) store: RenderStoreAction,
    pub(crate) vertices: u32,
    /// One entry per bound vertex stream, in binding order, with the bytes and
    /// footprints [`plan_vertex_input`] proved.
    pub(crate) vertex_streams: Vec<PlannedVertexStream<'a>>,
    /// The index buffer of an indexed draw, resolved from the pass's own view.
    pub(crate) indices: Option<PlannedIndexStream<'a>>,
    /// Readback length in bytes: the tightly packed texel extent.
    pub(crate) texel_bytes: usize,
    /// Bytes per attachment row, which is the tight pitch the contract's bytes
    /// are written in (`research/docs/23` §3.5).
    pub(crate) row_pitch: usize,
    pub(crate) initial: Option<&'a [u8]>,
}

/// Validate a render request against the contract and the rail's own allowlist.
///
/// Runs entirely without a device, so every refusal here is testable on a host
/// that cannot load Metal. Nothing outside the request is read: the pass carries
/// its own streams' bytes and its own attachment, so this call answers the same
/// way for a trace pass and for the device-level helper's trace-less request.
pub(crate) fn plan<'a>(
    request: &OffscreenRenderRequest<'a>,
) -> Result<RenderPlan<'a>, ProviderError> {
    // The pipeline's vertex-input shape selects the one reviewed module this
    // call may compile; the (module, entry pair) pair is then the whole
    // allowlist, re-checked by `review_contract` below.
    let module = reviewed_module(&request.pipeline.vertex_layout);
    if request.source != module.source {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "a {} layout compiles the bytes of `{}` and nothing else",
                layout_name(&request.pipeline.vertex_layout),
                module.path
            )),
        );
    }
    review_contract(request.pipeline)?;
    // Core admission first: the pass's own shape rules and the
    // pipeline/attachment format agreement belong to the contract
    // (`research/docs/23` §3.1, §3.2), not to this rail.
    request.pass.validate().map_err(contract_refusal)?;
    request
        .pipeline
        .validate_against(request.pass)
        .map_err(contract_refusal)?;
    let Some(attachment) = request.pass.color_attachments.first() else {
        return Err(contract_refusal(ContractError::EmptyAttachmentList));
    };
    if !SUPPORTED_COLOR_FORMATS.contains(&attachment.format) {
        return Err(
            capability_refusal("attachment_format_unsupported").with_field(
                "format",
                FieldValue::Unsigned(u64::from(attachment.format.code())),
            ),
        );
    }
    let format = pixel_format(attachment.format)?;
    if attachment.width > MAX_ATTACHMENT_DIMENSION[0]
        || attachment.height > MAX_ATTACHMENT_DIMENSION[1]
    {
        // The slug and fields capability admission uses for this fact
        // (`metal_api_core::provider::ProviderCapabilities::admit_render_passes`).
        return Err(capability_refusal("attachment_dimension_limit")
            .with_field("width", FieldValue::Unsigned(attachment.width))
            .with_field("height", FieldValue::Unsigned(attachment.height))
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
    let extent = [attachment.width as u32, attachment.height as u32];
    let texel_bytes = usize::try_from(attachment.expected_bytes().map_err(contract_refusal)?)
        .map_err(|_| capability_refusal("attachment_dimension_limit"))?;
    let row_pitch =
        usize::try_from(u64::from(extent[0]).saturating_mul(attachment.format.bytes_per_texel()))
            .map_err(|_| capability_refusal("attachment_dimension_limit"))?;
    let load = load_action(attachment.load, format)?;
    let store = store_action(attachment.store)?;
    // The vertex-input half: the streams with their bytes and their footprints.
    // Planned after the attachment because a stream is the draw's own input,
    // exactly as the attachment is its output.
    let (vertex_streams, indices) = plan_vertex_input(request.pass, request.pipeline)?;
    let initial = match (load, request.initial, request.present) {
        (RenderLoadAction::Clear(_), None, _) => None,
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
            return Err(args_refusal("render_attachment_initial_mismatch")
                .with_detail("LoadOp::Clear writes every texel, so initial bytes are refused"));
        }
    };
    Ok(RenderPlan {
        source: module.source,
        module_path: module.path,
        vertex_entry: request.pipeline.vertex_entry.as_str(),
        fragment_entry: request.pipeline.fragment_entry.as_str(),
        format,
        extent,
        viewport: request.pass.viewport,
        load,
        store,
        vertices: request.pass.vertices,
        vertex_streams,
        indices,
        texel_bytes,
        row_pitch,
        initial,
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
/// and the contract's [`VertexLayout`] says which module it may be: a contract
/// whose layout binds streams has to name the `[[stage_in]]` module's entries,
/// and a `VertexLayout::None` contract the `vertex_id` module's. Anything else
/// is refused with the same slug, class and phase the compute allowlist gives
/// an unreviewed kernel (`lib.rs::bounded_contract`,
/// `native_shader_not_allowlisted`): a matching file name, an edited module or a
/// recompiled one must not be enough to run different source
/// (`research/docs/23` §6 Step 7). Registration
/// (`NativeMetalProvider::register_render_pipeline`) and [`plan`] both run it,
/// so the refusal is reachable before a submission as well as inside one.
pub(crate) fn review_contract(contract: &RenderPipelineContract) -> Result<(), ProviderError> {
    let module = reviewed_module(&contract.vertex_layout);
    if contract.vertex_entry != module.vertex_entry
        || contract.fragment_entry != module.fragment_entry
    {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "a {} layout compiles `{}`, which carries {:?} and {:?}",
                layout_name(&contract.vertex_layout),
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
    pub(crate) landing: &'a BufferView,
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
    /// The writeback this pass's texels become.
    ///
    /// The view identity, allocation and offset are the landing view's own, so
    /// resource admission, lease bookkeeping and readback consumers need no
    /// second path: the attachment lands exactly where a compute pass writing
    /// the same view would (`research/docs/23` §6 Step 7).
    pub(crate) fn writeback(&self, texels: Vec<u8>) -> BufferWriteback {
        BufferWriteback {
            view_id: self.landing.view_id,
            allocation_id: self.landing.allocation_id,
            offset: self.landing.offset,
            bytes: texels,
        }
    }
}

/// Plan every render pass of a trace, without a device.
///
/// Four decisions have to be made before the first Metal object exists, and all
/// of them are answerable from values: the order the rails run in
/// ([`refuse_reordered_render_reads`], whose rule core admission also states as
/// part of the contract), the reviewed allowlist, the attachment's landing view,
/// and the previous bytes a loading pass uploads ([`previous_bytes`]). `pool` is
/// [`ComputeTrace::serial_resources`], the same pool the encoder binds, and
/// `contracts` holds the render contracts the provider registered for the
/// pipeline ids this trace names — a caller-supplied table entry is checked
/// against those registrations in `native.rs`, where the registry lives. The
/// pool's only job here is the attachment's landing view: a render input carries
/// its own bytes, so the streams a draw reads are resolved from the pass itself
/// ([`plan_vertex_input`]).
pub(crate) fn plan_trace<'a>(
    trace: &'a ComputeTrace,
    pool: &'a [BufferView],
    contracts: &'a BTreeMap<PipelineId, RenderPipelineContract>,
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
        let Some(attachment) = pass.color_attachments.first() else {
            return Err(contract_refusal(ContractError::EmptyAttachmentList));
        };
        // An attachment that no buffer view covers has no landing rail: the
        // texels would have nowhere to go, so the pass is refused instead of
        // being executed and dropped. The declared view is resolved before the
        // load op because a loading pass reads its previous bytes from the same
        // declaration (`research/docs/23` §3.3).
        let landing = pool
            .iter()
            .find(|view| {
                view.view_id == attachment.view_id && view.allocation_id == attachment.allocation_id
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
        // An offscreen `Load` uploads the bytes the declaring view owns before
        // the pass opens. A present pass's `Load` keeps the target's own initial
        // state, which the present path supplies, so it resolves no bytes
        // (`research/docs/24` §3.1).
        let previous = if pass.present.is_none() {
            previous_bytes(attachment.load, landing)?
        } else {
            None
        };
        let plan_of_pass = plan(&OffscreenRenderRequest {
            pass,
            pipeline: contract,
            source: reviewed_module(&contract.vertex_layout).source,
            initial: previous,
            present: pass.present.is_some(),
        })?;
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
            landing,
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

/// Execute one offscreen render pass and return its tightly packed texel bytes.
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
) -> Result<Vec<u8>, ProviderError> {
    let planned = plan(request)?;
    encode_offscreen_render(device, queue, &planned)
}

/// Encode, commit and read back one already planned pass.
///
/// Split from [`execute_offscreen_render`] so the trace path can plan once
/// ([`plan_trace`], before the compute command buffer is committed) and then
/// encode that same decision, instead of planning a second, possibly different,
/// pass. The attachment is fresh per pass: it is created here and dropped with
/// the readback, which is the offscreen shape (`research/docs/23` §6 Step 7).
#[cfg(target_os = "macos")]
pub(crate) fn encode_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
) -> Result<Vec<u8>, ProviderError> {
    objc::rc::autoreleasepool(|| {
        let attachment = attachment_texture(device, planned)?;
        encode_into_and_readback(device, queue, planned, &attachment, None)
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
    objc::rc::autoreleasepool(|| encode_into_and_readback(device, queue, planned, target, None))
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
) -> Result<Vec<u8>, ProviderError> {
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
        let attachment = attachment_texture(device, planned)?;
        encode_into_and_readback(device, queue, planned, &attachment, Some(*replay))
    })
}

/// The shared encoder body of the offscreen and present rails: build the
/// reviewed pipeline, render the pass into `target`, wait for a terminal
/// command-buffer status, and read the texels back.
#[cfg(target_os = "macos")]
fn encode_into_and_readback(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    target: &Texture,
    indirect: Option<icb::IcbPlan>,
) -> Result<Vec<u8>, ProviderError> {
    let pipeline = render_pipeline_state(device, planned)?;
    // The pass descriptor is autoreleased; it only has to outlive the
    // encoder creation below.
    let pass = MetalRenderPassDescriptor::new();
    let color = pass
        .color_attachments()
        .object_at(0)
        .ok_or_else(|| resource_refusal("metal_render_attachment_descriptor_unavailable"))?;
    color.set_texture(Some(target));
    match planned.load {
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
    }
    color.set_store_action(MTLStoreAction::Store);
    // The command buffer and the encoder are autoreleased and the rail is
    // synchronous, so neither has to be retained: nothing here outlives this
    // pool.
    let command = queue.new_command_buffer();
    let encoder = command.new_render_command_encoder(pass);
    encoder.set_render_pipeline_state(&pipeline);
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
    // the first increment only accepts the attachment-covering default.
    encoder.set_viewport(MTLViewport {
        originX: f64::from(planned.viewport[0]),
        originY: f64::from(planned.viewport[1]),
        width: f64::from(planned.viewport[2]),
        height: f64::from(planned.viewport[3]),
        znear: 0.0,
        zfar: 1.0,
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
                encoder.draw_indexed_primitives(
                    MTLPrimitiveType::Triangle,
                    u64::from(indices.index_count),
                    metal_index_type(indices.format),
                    buffer.as_ref(),
                    offset,
                );
                stream_buffers.push(buffer);
            }
            None => {
                encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, u64::from(planned.vertices))
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
    read_texels(target, planned)
}

/// The colour attachment or present target this rail renders into.
///
/// `usage = RenderTarget` states what the texture is for, and the shared storage
/// mode is what makes the texels CPU-visible for the readback on the
/// unified-memory device the provider admits — the same reason the sampled
/// texture rail uses shared storage (`research/docs/16` §4.8).
#[cfg(target_os = "macos")]
fn attachment_texture(device: &Device, planned: &RenderPlan<'_>) -> Result<Texture, ProviderError> {
    let texture = present_target_texture(device, planned.format, planned.extent)?;
    if let Some(bytes) = planned.initial {
        upload_texels(&texture, planned, bytes);
    }
    Ok(texture)
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
            // One stream advance per vertex: per-instance step rates are not
            // part of this increment (`VertexBufferLayout` carries no step
            // rate), so the descriptor states the only rate it can mean.
            layout.set_step_function(MTLVertexStepFunction::PerVertex);
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
    // Attachment 0 is the pass's only colour attachment, and its pixel format is
    // the one the pipeline is compiled against (`render_targets` locations are
    // deferred, `research/docs/23` §3.3).
    let color = descriptor
        .color_attachments()
        .object_at(0)
        .ok_or_else(|| resource_refusal("metal_render_pipeline_attachment_unavailable"))?;
    color.set_pixel_format(metal_pixel_format(planned.format));
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
        BufferBindingContract, BufferSource, CompiledComputePipeline, CompletionPolicy,
        ComputePass, DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof,
        FunctionIdentity, FunctionSource, IndirectCommandBufferDescriptor, IndirectCommandKind,
        IndirectCommandPayload, IndirectCommandRange, InitialState, LeaseId, OperationId,
        PipelineContract, PresentTarget, ProviderCapabilities, RenderAttachment,
        ResourceTableSnapshot, SemanticDigest, StorageMode, VertexAttribute, VertexBufferLayout,
        VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    /// The texels the reviewed fragment writes, as `MTLClearColor` components.
    const EXPECTED_TEXEL_COMPONENTS: [f64; 4] = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0];

    /// The milestone's pass: one 2x2 `Rgba8Unorm` attachment, stored, and drawn
    /// as the full-screen triangle.
    fn milestone_pass(load: LoadOp) -> RenderPassDescriptor {
        RenderPassDescriptor {
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
            vertices: 3,
            vertex_buffers: Vec::new(),
            indices: None,
            present: None,
        }
    }

    fn milestone_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            vertex_entry: VERTEX_ENTRY.to_owned(),
            fragment_entry: FRAGMENT_ENTRY.to_owned(),
            color_format: AttachmentFormat::Rgba8Unorm,
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
            initial,
            present: false,
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
        plan(request)
    }

    /// The refusal of a (source, entry pair) triple the rail does not review.
    fn allowlist_refusal(source: &str, vertex: &str, fragment: &str) -> ProviderError {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = RenderPipelineContract {
            vertex_entry: vertex.to_owned(),
            fragment_entry: fragment.to_owned(),
            color_format: AttachmentFormat::Rgba8Unorm,
            vertex_layout: VertexLayout::None,
        };
        let request = OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source,
            initial: None,
            present: false,
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
        assert_eq!(planned.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(planned.viewport, [0, 0, 2, 2]);
        assert_eq!(planned.vertices, 3);
        assert_eq!(planned.store, RenderStoreAction::Store);
        // 2x2 texels of a 4-byte format: 16 bytes, two rows of 8.
        assert_eq!(planned.texel_bytes, 16);
        assert_eq!(planned.row_pitch, 8);
        assert_eq!(planned.initial, None);
        let RenderLoadAction::Clear(components) = planned.load else {
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
    fn clear_and_load_actions_are_distinct_and_dont_care_is_refused() {
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
        let error = load_action(LoadOp::DontCare, RenderPixelFormat::Rgba8Unorm).unwrap_err();
        assert_eq!(error.slug, "attachment_load_op_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn store_dont_care_is_refused() {
        assert_eq!(
            store_action(StoreOp::Store).unwrap(),
            RenderStoreAction::Store
        );
        let error = store_action(StoreOp::DontCare).unwrap_err();
        assert_eq!(error.slug, "attachment_store_op_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn plan_refuses_an_attachment_beyond_the_fixed_extent() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [0, 0, 4, 4];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = 4;
        attachment.height = 4;
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "attachment_dimension_limit");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn plan_refuses_a_pipeline_whose_format_disagrees_with_the_attachment() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let mut pipeline = milestone_pipeline();
        pipeline.color_format = AttachmentFormat::Bgra8Unorm;
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
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
        assert_eq!(planned.load, RenderLoadAction::Load);
        assert_eq!(planned.initial, Some(previous.as_slice()));

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
        let RenderLoadAction::Clear(components) = planned.load else {
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
                color_format: AttachmentFormat::Rgba8Unorm,
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
            color_format: AttachmentFormat::Rgba8Unorm,
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
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
            vertices: 6,
            vertex_buffers: vec![quad_vertex_view()],
            indices: Some(IndexBufferBinding {
                view: quad_index_view(),
                format: IndexFormat::Uint16,
            }),
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
            initial: None,
            present: false,
        }
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
        // The rail's limits are core's own values, not a second spelling that
        // could drift from the contract's.
        assert_eq!(
            bits.max_color_attachments,
            u32::try_from(metal_api_core::provider::MAX_COLOR_ATTACHMENTS).unwrap()
        );
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
            plan_trace(&trace, &pool, &contracts).expect("the reviewed present pass plans");
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
        assert!(matches!(planned.plan.load, RenderLoadAction::Load));
    }

    /// The host-side half of the trace path: the plan a device-free host can
    /// check, which is the same decision the macOS encoder body then executes.
    #[test]
    fn plan_trace_plans_the_milestone_pass_and_its_landing_view() {
        let (trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts).expect("the reviewed pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.pass.color_attachments[0].view_id, ViewId::new(7));
        assert_eq!(planned.contract, &milestone_pipeline());
        assert_eq!(planned.plan.extent, [2, 2]);
        assert_eq!(planned.plan.texel_bytes, 16);
        assert_eq!(planned.plan.row_pitch, 8);
        assert_eq!(planned.plan.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(planned.plan.vertices, 3);
        assert_eq!(planned.plan.initial, None);
        // The landing view is the declaration's own identity and range, so the
        // writeback is the one the trace asked for and no second channel is
        // invented.
        assert_eq!(planned.landing.view_id, ViewId::new(7));
        assert_eq!(planned.landing.allocation_id, AllocationId::new(9));
        let writeback = planned.writeback(EXPECTED_TEXEL_BYTES.repeat(4));
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
        assert!(plan_trace(&compute_only, &pool, &contracts)
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
        let planned = plan_trace(&trace, &pool, &contracts)
            .expect("the declaring view's own bytes are what a load uploads");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.plan.load, RenderLoadAction::Load);
        // The declaration's 16-byte range is the tightly packed 2x2 rgba8
        // texels, so the plan carries exactly those bytes.
        assert_eq!(planned.plan.initial, Some([0xfe; 16].as_slice()));
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
        let error = plan_trace(&trace, &pool, &milestone_contracts()).unwrap_err();
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
        let error = plan_trace(&trace, &pool, &milestone_contracts()).unwrap_err();
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
        let error = plan_trace(&loading, &pool, &milestone_contracts()).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
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
            plan_trace(&legal, &pool, &milestone_contracts())
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
        let planned = plan(&quad_request(&pass, &pipeline)).expect("the reviewed pass plans");

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
        let error = plan(&quad_request(&leased_pass, &quad_pipeline())).unwrap_err();
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
        let error = plan(&quad_request(&borrowed_pass, &quad_pipeline())).unwrap_err();
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
        let error = plan(&quad_request(&short_stream, &quad_pipeline())).unwrap_err();
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
        let error = plan(&quad_request(&non_indexed, &quad_pipeline())).unwrap_err();
        assert_eq!(error.slug, "render_vertex_footprint_unsupported");
        assert_eq!(
            error.fields.get("required_bytes"),
            Some(&FieldValue::Unsigned(32))
        );

        // 10 bytes cannot hold the six `uint16` indices.
        let mut short_indices = quad_pass();
        shorten(&mut short_indices.indices.as_mut().unwrap().view, 10);
        let error = plan(&quad_request(&short_indices, &quad_pipeline())).unwrap_err();
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
        let error = plan(&quad_request(&out_of_range, &quad_pipeline())).unwrap_err();
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
        let planned = plan_trace(&trace, &pool, &contracts)
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
        let error = plan_trace(&trace, &pool, &milestone_contracts()).unwrap_err();
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
            plan_trace(&trace, &pool, &contracts).expect("the reviewed indexed pass plans");
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
        assert_eq!(planned.landing.view_id, ViewId::new(7));
        assert_eq!(planned.landing.allocation_id, AllocationId::new(9));
        let writeback = planned.writeback(EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(writeback.view_id, ViewId::new(7));
        assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
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
        let planned = plan_trace(&trace, &pool, &contracts).expect("the draw plans");
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
        let error = plan_trace(&trace, &pool, &quad_contracts()).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
    }

    /// The reviewed allowlist is a pair of (module, layout): a layout that binds
    /// streams may only compile the `[[stage_in]]` module, and a `vertex_id`
    /// pipeline only the module whose vertex stage reads no stream. A caller
    /// cannot pair one shape's descriptor with the other shape's source.
    #[test]
    fn a_vertex_layout_selects_the_reviewed_module() {
        let pass = quad_pass();
        let pipeline = quad_pipeline();

        assert_eq!(reviewed_module(&VertexLayout::None).source, REVIEWED_SOURCE);
        assert_eq!(
            reviewed_module(&pipeline.vertex_layout).source,
            REVIEWED_VERTEX_SOURCE
        );
        assert!(!reviewed_module(&VertexLayout::None).binds_buffers);
        assert!(reviewed_module(&pipeline.vertex_layout).binds_buffers);
        assert_eq!(layout_name(&VertexLayout::None), "vertex_id");
        assert_eq!(layout_name(&pipeline.vertex_layout), "vertex-buffer");

        // The reviewed contract is accepted, and so is the milestone's.
        assert_eq!(review_contract(&pipeline), Ok(()));
        assert_eq!(review_contract(&quad_pipeline()), Ok(()));

        // A stream layout with the `vertex_id` module's bytes, and the reverse.
        let crossed = OffscreenRenderRequest {
            source: REVIEWED_SOURCE,
            ..quad_request(&pass, &pipeline)
        };
        let error = plan(&crossed).unwrap_err();
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
        // The two modules share one fragment entry, which is what keeps a
        // capture from telling the draws apart by their colour.
        assert_eq!(
            REVIEWED_MODULES[0].fragment_entry,
            REVIEWED_MODULES[1].fragment_entry
        );
        assert_eq!(REVIEWED_MODULES[0].vertex_entry, VERTEX_ENTRY);
        assert_eq!(REVIEWED_MODULES[1].vertex_entry, QUAD_VERTEX_ENTRY);
    }
}
