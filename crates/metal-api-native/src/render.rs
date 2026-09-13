//! Offscreen render rail for the native provider (`research/docs/23` §6 Steps 6
//! and 7).
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

use crate::refusal;
use metal_api_core::provider::{
    AttachmentFormat, BufferView, BufferWriteback, ClearColor, ComputeTrace, ContractError,
    FieldValue, LoadOp, PipelineId, ProviderError, ProviderErrorClass, ProviderPhase,
    RenderPassDescriptor, RenderPipelineContract, StoreOp, TracePass, ViewId,
};
use std::collections::BTreeMap;

#[cfg(target_os = "macos")]
use foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    CommandQueue, CompileOptions, Device, MTLClearColor, MTLCommandBufferStatus, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLViewport, NSUInteger,
    RenderPassDescriptor as MetalRenderPassDescriptor, RenderPipelineDescriptor,
    RenderPipelineState, Texture, TextureDescriptor,
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

/// Vertex entry of the reviewed module: positions from `vertex_id`, no vertex
/// buffer (`VertexLayout::None`).
pub(crate) const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// Fragment entry of the reviewed module: the fixed colour texel.
pub(crate) const FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// Colour attachments the first render increment admits. The same value the
/// core contract states (`metal_api_core::provider::MAX_COLOR_ATTACHMENTS`); it
/// is restated here because a capability value has to be spelled by the provider
/// that declares it (`research/docs/23` §4.2).
pub(crate) const MAX_COLOR_ATTACHMENTS: u32 = 1;

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
}

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
    RenderCapabilityBits {
        supports_render_passes: true,
        max_color_attachments: MAX_COLOR_ATTACHMENTS,
        max_attachment_dimension: MAX_ATTACHMENT_DIMENSION,
        supported_color_formats: SUPPORTED_COLOR_FORMATS.to_vec(),
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
    /// The MSL module to compile. Only [`REVIEWED_SOURCE`] is accepted.
    pub(crate) source: &'a str,
    /// Tightly packed texels the attachment already holds, for [`LoadOp::Load`].
    /// Required exactly then, refused for a clear.
    pub(crate) initial: Option<&'a [u8]>,
}

/// Everything the encoder needs, decided before the first Metal object exists.
#[derive(Debug)]
pub(crate) struct RenderPlan<'a> {
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
/// that cannot load Metal.
pub(crate) fn plan<'a>(
    request: &OffscreenRenderRequest<'a>,
) -> Result<RenderPlan<'a>, ProviderError> {
    // The reviewed (source, entry pair) triple is the rail's whole allowlist.
    if request.source != REVIEWED_SOURCE {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(
                "the rail compiles the bytes of \
             `conformance/shaders/render_offscreen_2x2.metal` and nothing else",
            ),
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
    let initial = match (load, request.initial) {
        (RenderLoadAction::Clear(_), None) => None,
        (RenderLoadAction::Load, Some(bytes)) if bytes.len() == texel_bytes => Some(bytes),
        (RenderLoadAction::Load, Some(bytes)) => {
            return Err(
                args_refusal("render_attachment_initial_mismatch").with_detail(format!(
                    "LoadOp::Load needs {texel_bytes} tightly packed bytes, got {}",
                    bytes.len()
                )),
            );
        }
        (RenderLoadAction::Load, None) => {
            return Err(args_refusal("render_attachment_initial_mismatch")
                .with_detail("LoadOp::Load needs the attachment's previous texels"));
        }
        (RenderLoadAction::Clear(_), Some(_)) => {
            return Err(args_refusal("render_attachment_initial_mismatch")
                .with_detail("LoadOp::Clear writes every texel, so initial bytes are refused"));
        }
    };
    Ok(RenderPlan {
        vertex_entry: request.pipeline.vertex_entry.as_str(),
        fragment_entry: request.pipeline.fragment_entry.as_str(),
        format,
        extent,
        viewport: request.pass.viewport,
        load,
        store,
        vertices: request.pass.vertices,
        texel_bytes,
        row_pitch,
        initial,
    })
}

/// The rail's review gate for a render pipeline contract.
///
/// The reviewed module carries exactly one vertex entry and one fragment entry,
/// so a contract naming anything else is refused with the same slug, class and
/// phase the compute allowlist gives an unreviewed kernel
/// (`lib.rs::bounded_contract`, `native_shader_not_allowlisted`): a matching
/// file name, an edited module or a recompiled one must not be enough to run
/// different source (`research/docs/23` §6 Step 7). Registration
/// (`NativeMetalProvider::register_render_pipeline`) and [`plan`] both run it,
/// so the refusal is reachable before a submission as well as inside one.
pub(crate) fn review_contract(contract: &RenderPipelineContract) -> Result<(), ProviderError> {
    if contract.vertex_entry != VERTEX_ENTRY || contract.fragment_entry != FRAGMENT_ENTRY {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "the reviewed module carries {VERTEX_ENTRY:?} and {FRAGMENT_ENTRY:?}"
            )),
        );
    }
    Ok(())
}

/// The load op the trace path can honour.
///
/// The rail itself executes `LoadOp::Load` when it is handed the attachment's
/// previous texels, and the tests below exercise that. `ComputeTrace` has no
/// channel that carries those bytes: [`metal_api_core::provider::RenderAttachment`]
/// restates the attachment's shape and names no contents, so a trace asking for
/// `Load` would have to be executed as a clear. The refusal reuses the slug,
/// class and phase this rail and the Vulkan rail give an unexecutable load op.
pub(crate) fn admit_trace_load(load: LoadOp) -> Result<(), ProviderError> {
    match load {
        LoadOp::Load => Err(capability_refusal("attachment_load_op_unsupported")
            .with_field("load_op", FieldValue::Text("load".to_owned()))
            .with_detail(
                "the trace carries no attachment-initial-bytes channel, so an executed \
                 `Load` would silently become a clear",
            )),
        _ => Ok(()),
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
/// four are answerable from values: the order the rails run in
/// ([`refuse_reordered_render_reads`]), the reviewed allowlist, the attachment's
/// landing view, and the load op the trace can carry
/// ([`admit_trace_load`]). `pool` is [`ComputeTrace::serial_resources`], the same
/// pool the encoder binds, and `contracts` holds the render contracts the
/// provider registered for the pipeline ids this trace names — a caller-supplied
/// table entry is checked against those registrations in `native.rs`, where the
/// registry lives.
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
        let Some(attachment) = pass.color_attachments.first() else {
            return Err(contract_refusal(ContractError::EmptyAttachmentList));
        };
        admit_trace_load(attachment.load)?;
        // An attachment that no buffer view covers has no landing rail: the
        // texels would have nowhere to go, so the pass is refused instead of
        // being executed and dropped.
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
        let plan_of_pass = plan(&OffscreenRenderRequest {
            pass,
            pipeline: contract,
            source: REVIEWED_SOURCE,
            initial: None,
        })?;
        planned.push(TraceRenderPlan {
            pass,
            contract,
            landing,
            plan: plan_of_pass,
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
/// pass.
#[cfg(target_os = "macos")]
pub(crate) fn encode_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
) -> Result<Vec<u8>, ProviderError> {
    objc::rc::autoreleasepool(|| {
        let attachment = attachment_texture(device, planned)?;
        let pipeline = render_pipeline_state(device, planned)?;
        // The pass descriptor is autoreleased; it only has to outlive the
        // encoder creation below.
        let pass = MetalRenderPassDescriptor::new();
        let color = pass
            .color_attachments()
            .object_at(0)
            .ok_or_else(|| resource_refusal("metal_render_attachment_descriptor_unavailable"))?;
        color.set_texture(Some(attachment.as_ref()));
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
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, u64::from(planned.vertices));
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
        read_texels(&attachment, planned)
    })
}

/// The colour attachment this rail renders into.
///
/// `usage = RenderTarget` states what the texture is for, and the shared storage
/// mode is what makes the texels CPU-visible for the readback on the
/// unified-memory device the provider admits — the same reason the sampled
/// texture rail uses shared storage (`research/docs/16` §4.8).
#[cfg(target_os = "macos")]
fn attachment_texture(device: &Device, planned: &RenderPlan<'_>) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(metal_pixel_format(planned.format));
    descriptor.set_width(u64::from(planned.extent[0]));
    descriptor.set_height(u64::from(planned.extent[1]));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_target_allocation_failed"));
    }
    let texture = unsafe { Texture::from_ptr(pointer) };
    if let Some(bytes) = planned.initial {
        // `replace_region` takes the source stride and owns the texture-side
        // layout, so this upload cannot repeat the Vulkan rail's defect: there
        // the host had to guess the destination row pitch, while Metal keeps that
        // distance inside the driver (`research/docs/16` §4.8).
        texture.replace_region(
            region(planned),
            0,
            bytes.as_ptr().cast(),
            NSUInteger::try_from(planned.row_pitch).unwrap_or(NSUInteger::MAX),
        );
    }
    Ok(texture)
}

/// The two-stage pipeline state of the reviewed module.
#[cfg(target_os = "macos")]
fn render_pipeline_state(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<RenderPipelineState, ProviderError> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(REVIEWED_SOURCE, &options)
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

#[cfg(target_os = "macos")]
fn resource_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Resource, slug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AliasMode, AllocationId, AllocationRecord, BufferAccess, BufferBindingContract,
        BufferSource, CompiledComputePipeline, CompletionPolicy, ComputePass, DeviceEpoch,
        Dispatch, DispatchKind, DispatchType, FootprintProof, FunctionIdentity, FunctionSource,
        OperationId, PipelineContract, ProviderCapabilities, RenderAttachment,
        ResourceTableSnapshot, SemanticDigest, StorageMode, VertexLayout, ViewId,
        PROVIDER_SCHEMA_VERSION,
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
        }
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
        };
        plan(&request).map(|_| ()).unwrap_err()
    }

    #[test]
    fn plan_accepts_the_milestone_shape_and_fixes_the_readback_extent() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None)).unwrap();
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
        let error = plan(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "attachment_dimension_limit");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn plan_refuses_a_pipeline_whose_format_disagrees_with_the_attachment() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let mut pipeline = milestone_pipeline();
        pipeline.color_format = AttachmentFormat::Bgra8Unorm;
        let error = plan(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
    }

    #[test]
    fn plan_refuses_a_draw_shape_other_than_the_full_screen_triangle() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.vertices = 6;
        let pipeline = milestone_pipeline();
        let error = plan(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "draw_shape_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn load_requires_the_previous_texels_and_clear_refuses_them() {
        let pass = milestone_pass(LoadOp::Load);
        let pipeline = milestone_pipeline();
        let error = plan(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
        assert_eq!(error.class, ProviderErrorClass::Args);

        let previous = [0x11_u8; 16];
        let planned = plan(&milestone_request(&pass, &pipeline, Some(&previous))).unwrap();
        assert_eq!(planned.load, RenderLoadAction::Load);
        assert_eq!(planned.initial, Some(previous.as_slice()));

        let short = [0x11_u8; 15];
        let error = plan(&milestone_request(&pass, &pipeline, Some(&short))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");

        let cleared = milestone_pass(LoadOp::Clear(sentinel()));
        let error = plan(&milestone_request(&cleared, &pipeline, Some(&previous))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
    }

    /// The falsifiability rule `research/docs/23` §1.3 states: a pass that never
    /// ran has to be distinguishable from one that did, in every channel.
    #[test]
    fn a_cleared_attachment_cannot_imitate_the_fragment_output() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None)).unwrap();
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
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: Vec::new(),
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
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

    /// The capability snapshot the macOS provider builds, with the render bits
    /// taken from the value under test and the compute bits from `native.rs`.
    fn capabilities(bits: &RenderCapabilityBits) -> ProviderCapabilities {
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
        }
    }

    /// The registrations `plan_trace` resolves the trace's pipeline ids against.
    fn milestone_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(PipelineId::new(3), milestone_pipeline())])
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

    /// `LoadOp::Load` is the one load op the trace cannot carry, and it is
    /// refused before the pass is planned rather than executed as a clear.
    #[test]
    fn plan_trace_refuses_a_load_op_the_trace_cannot_carry() {
        let (trace, _) = milestone_trace(LoadOp::Load);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &milestone_contracts()).unwrap_err();
        assert_eq!(error.slug, "attachment_load_op_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
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
    }

    /// The ordering rule the trace path shares with the Vulkan rail: every
    /// compute pass runs before every render pass, so a compute pass that
    /// follows a render store of a view it binds would read pre-render bytes.
    #[test]
    fn plan_trace_refuses_a_compute_pass_that_reads_after_a_render_store() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        trace.passes.push(TracePass::Compute(declaration_pass()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &milestone_contracts()).unwrap_err();
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
}
