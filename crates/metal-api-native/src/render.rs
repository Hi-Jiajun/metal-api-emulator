//! Offscreen render rail for the native provider (`research/docs/23` §6 Step 6).
//!
//! The rail answers the one question Step 6 owns on the Apple side: can the
//! native provider build a colour attachment, a render pass descriptor, a
//! two-entry render pipeline state and a full-screen-triangle draw out of one
//! reviewed MSL module, and read the attachment's texels back byte for byte.
//!
//! **It has never run on an Apple GPU.** The provider's render bits stay at
//! their defaults, so admission refuses a render-bearing trace with
//! `render_passes_unsupported` before this code is reached (`native.rs`,
//! `ProviderCapabilities`); that refusal is the honest state until the
//! single-device check in `conformance/RENDER-CAPTURE.md` passes. The two rules
//! this rail shares with its sibling on the Vulkan side
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
    AttachmentFormat, ClearColor, ContractError, FieldValue, LoadOp, ProviderError,
    ProviderErrorClass, ProviderPhase, RenderPassDescriptor, RenderPipelineContract, StoreOp,
};

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
    if request.source != REVIEWED_SOURCE
        || request.pipeline.vertex_entry != VERTEX_ENTRY
        || request.pipeline.fragment_entry != FRAGMENT_ENTRY
    {
        return Err(allowlist_refusal("native_render_source_not_reviewed"));
    }
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

/// Execute one offscreen render pass and return its tightly packed texel bytes.
///
/// Not verified on an Apple GPU. The check that would verify it is the one
/// `conformance/RENDER-CAPTURE.md` records, and it is the condition for flipping
/// `ProviderCapabilities::supports_render_passes`.
#[cfg(target_os = "macos")]
pub(crate) fn execute_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    request: &OffscreenRenderRequest<'_>,
) -> Result<Vec<u8>, ProviderError> {
    let planned = plan(request)?;
    objc::rc::autoreleasepool(|| {
        let attachment = attachment_texture(device, &planned)?;
        let pipeline = render_pipeline_state(device, &planned)?;
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
        read_texels(&attachment, &planned)
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
        AllocationId, PipelineId, RenderAttachment, VertexLayout, ViewId,
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
}
