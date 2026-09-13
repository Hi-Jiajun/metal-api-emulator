//! Offscreen render execution rail (`research/docs/23` §6 Step 3b).
//!
//! One colour attachment, one full-screen triangle, one `vkCmdDraw`, then
//! `vkCmdCopyImageToBuffer` back into host-visible memory. The rail answers the
//! one question this step owns — can the provider build a render pass, a
//! framebuffer and a graphics pipeline out of two SPIR-V modules and read the
//! attachment back byte-for-byte — and it fixes the two rules the driver probe
//! left behind (`/var/tmp/render-probe`, `research/docs/23` §3.5, §9):
//!
//! * the attachment is `VK_IMAGE_TILING_OPTIMAL` plus one
//!   `vkCmdCopyImageToBuffer` (`copy_out = 1`), because the RTX 5060 native
//!   driver and the dzn/D3D12 backend both refuse a linear colour attachment;
//! * admission asks for `VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT` on the exact
//!   format **and** tiling before `vkCreateImage`, because ignoring the bit
//!   lets `vkCreateImage` and `vkCreateGraphicsPipelines` both succeed and then
//!   kills the device with a late `VK_ERROR_OUT_OF_HOST_MEMORY` on dzn.
//!
//! Step 3c owns the rest: tagging the trace's pass list with the render arm
//! (the core value types already exist), the `MCC1` payload for that arm and
//! flipping `ProviderCapabilities::supports_render_passes`. Step 4 hands the
//! rail to the trace path: [`execute_render_pass`] translates one admitted
//! `RenderPassDescriptor` into a rail request, and the compute provider's
//! submit path calls it for every render entry and publishes the readback
//! through the existing buffer-writeback channel.

use ash::vk;
use metal_api_core::provider::{
    AttachmentFormat, ClearColor, FieldValue, LoadOp, ProviderError, ProviderErrorClass,
    ProviderPhase, RenderPassDescriptor, RenderPipelineContract, Retryability, StoreOp,
};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use crate::VulkanContext;

/// `VK_FORMAT_*` texel width shared by every format the render contract admits.
///
/// `AttachmentFormat::bytes_per_texel` fixes this at the contract layer; the
/// rail restates it so the readback extent is computed without a second
/// format-to-width mapping.
const BYTES_PER_TEXEL: u64 = 4;

/// The vertices of the milestone's single draw: the full-screen triangle
/// (`research/docs/23` §1.2).
const FULL_SCREEN_TRIANGLE_VERTICES: u32 = 3;

// The reviewed stage modules of the milestone live in `render_spv/`: a
// full-screen triangle vertex stage plus one solid fragment stage per admitted
// colour-attachment format. The vertex stage stays a host registration's value,
// while the fragment stage is *not*: it is a function of the attachment format
// (`solid_fragment_spirv`), because a fragment stage built for one format does
// not describe another one. Pairing the fixed `vec4` store with every format is
// exactly the "admitted, then read back the wrong bytes" path the 2026-09-14
// review filed as I2.

/// Entry point every reviewed solid fragment module declares.
///
/// One name for the whole set keeps the registration check and the execution
/// check in terms of the same binding: `VkPipelineShaderStageCreateInfo::pName`
/// names this entry, and a module that does not declare it would build a
/// pipeline the trace did not describe.
pub(crate) const SOLID_FRAGMENT_ENTRY: &str = "fragment_main";

/// The reviewed solid fragment module for an 8-bit UNORM attachment.
///
/// One module serves both `VK_FORMAT_R8G8B8A8_UNORM` and
/// `VK_FORMAT_B8G8R8A8_UNORM`, because which channel lands in which byte is the
/// *image format's* decision rather than the shader's: the stage stores
/// `(64/255, 128/255, 192/255, 1)` either way, so an R,G,B,A layout reads back
/// `40 80 c0 ff` per texel and a B,G,R,A layout reads back `c0 80 40 ff`.
/// Swizzling the store as well would undo the format's own reordering twice and
/// land the other colour in those same bytes.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_unorm8.frag.spv");

/// The reviewed solid fragment module for a single-channel float attachment
/// (`VK_FORMAT_R32_SFLOAT`).
///
/// A one-component attachment takes a one-component store: the same colour's
/// red component, `64/255`, written as one `float`. The `vec4` store of the
/// 8-bit module would not match this attachment's component shape, which is the
/// half of I2 that is a genuine mismatch rather than a byte-order expectation.
const SOLID_R32F_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_r32f.frag.spv");

/// The solid fragment module the offscreen rail builds for `format`.
///
/// The match is exhaustive over [`AttachmentFormat`] and has no default arm: a
/// contract format that gains no arm here is a compile error, which is what
/// makes "admitted but executed with another format's fragment stage"
/// unrepresentable instead of merely tested. `R32Uint` is refused with the slug
/// the contract and the format rail already use for it, so an integer
/// attachment cannot reach a colour store.
pub(crate) fn solid_fragment_spirv(
    format: AttachmentFormat,
) -> Result<&'static [u8], ProviderError> {
    Ok(match format {
        AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm => SOLID_UNORM8_FRAG_SPV,
        AttachmentFormat::R32Float => SOLID_R32F_FRAG_SPV,
        AttachmentFormat::R32Uint => {
            return Err(attachment_format_refusal()
                .with_field(
                    "format_code",
                    FieldValue::Unsigned(u64::from(format.code())),
                )
                .with_detail("this rail has no colour fragment stage for the format"));
        }
    })
}

/// One offscreen render pass to execute.
///
/// The shape mirrors `metal_api_core::provider::RenderPassDescriptor` for the
/// fields this rail consumes: the core type carries wiring identities
/// (pipeline/view/allocation ids and a resolved byte source) that Step 3c maps,
/// while Step 3b fixes the Vulkan-side execution against an already-chosen
/// format, extent and clear value. The shader *pair* is not a field: the
/// fragment half is chosen from the format by [`solid_fragment_spirv`], so a
/// request cannot name a fragment stage the format was not compiled for.
pub(crate) struct OffscreenRenderRequest<'a> {
    /// Colour attachment format, in render-contract terms.
    pub format: AttachmentFormat,
    /// Attachment extent in texels. The milestone fixes 2×2 (`docs/23` §1.3) so
    /// full coverage is distinguishable from a single stored texel.
    pub extent: [u32; 2],
    /// The `LoadOp::Clear` value. Carried as bytes for the same reason the
    /// contract carries bytes: a float clear is not parity-stable
    /// (`research/docs/23` §3.5). The bytes are in the attachment format's
    /// *memory* order; [`clear_value_for`] maps them onto Vulkan's component
    /// order, which is not the same thing (`Bgra8Unorm` needs a swap, and
    /// `R32Float` is one component, not four).
    pub clear: ClearColor,
    /// The vertex stage the graphics pipeline is built from.
    pub vertex: OffscreenVertexStage<'a>,
}

/// The vertex stage entry one offscreen render pipeline pairs with the rail's
/// own fragment stage.
///
/// The entry name travels with the module because it is what the pipeline
/// actually binds: `VkPipelineShaderStageCreateInfo::pName` names the SPIR-V
/// entry point, so an entry the module does not declare would build a pipeline
/// the trace did not describe. Host-side registrations own this module; this
/// rail only reads it. The fragment half is deliberately absent — it belongs to
/// the rail, because it is a function of the attachment format.
pub(crate) struct OffscreenVertexStage<'a> {
    /// Entry point of the vertex-stage module.
    pub entry: &'a str,
    /// Vertex-stage SPIR-V module.
    pub spirv: &'a [u8],
}

/// One host-registered render pipeline: the two compiled stage modules and the
/// contract they were built against.
///
/// The compute rail keeps one translated artifact per `PipelineId` in the
/// provider registry; this is the render sibling of that value, stored in the
/// same registry namespace so a trace's pipeline table stays the single source
/// of which pipeline a pass names. The fragment module is not free-form: it is
/// the reviewed module of the contract's colour format, which is what
/// [`fragment_stage_is_reviewed`] binds.
pub(crate) struct RenderStages {
    pub contract: RenderPipelineContract,
    pub vertex_spirv: Vec<u8>,
    pub fragment_spirv: Vec<u8>,
}

impl RenderStages {
    /// Structural validation of one registration, before any trace can name it.
    ///
    /// The entry names and the module bytes are checked here because both are
    /// per-registration facts: an empty entry name, a name carrying an interior
    /// NUL and a module that is not a whole number of SPIR-V words are refused
    /// once, at registration, instead of on every submission that names the
    /// pipeline. The same place binds the fragment stage to the contract's
    /// colour format, for the same reason: the registration is where the pairing
    /// is settled, so a trace never sees a pairing this rail cannot execute.
    pub(crate) fn validate(&self) -> Result<(), ProviderError> {
        self.contract
            .validate()
            .map_err(|error| render_pipeline_contract_refusal(&error.to_string()))?;
        for (stage, entry) in [
            ("vertex", self.contract.vertex_entry.as_str()),
            ("fragment", self.contract.fragment_entry.as_str()),
        ] {
            if CString::new(entry).is_err() {
                return Err(stage_entry_refusal(
                    stage,
                    "the entry name carries an interior NUL",
                ));
            }
        }
        for (stage, module) in [
            ("vertex", self.vertex_spirv.as_slice()),
            ("fragment", self.fragment_spirv.as_slice()),
        ] {
            if spirv_words(module).is_none() {
                return Err(
                    spirv_refusal("the module is empty or not a multiple of four bytes")
                        .with_field("stage", FieldValue::Text(stage.to_owned())),
                );
            }
        }
        // The fragment stage has to be the reviewed module for the format this
        // contract declares. The rail cannot read a module's semantics, so
        // binding the registration to the reviewed set is what refuses "this
        // format, that format's fragment stage" *before* a submission can read
        // back bytes the format claim does not cover (review item I2,
        // 2026-09-14): an `R32Float` pipeline handed the 8-bit module's `vec4`
        // store is a component-shape mismatch, not a byte-order preference.
        if !fragment_stage_is_reviewed(self) {
            return Err(fragment_stage_mismatch_refusal(
                self.contract.color_format,
                &self.contract.fragment_entry,
            ));
        }
        Ok(())
    }
}

/// Whether a registration's fragment stage is exactly the module this rail
/// builds for the contract's colour format, under the entry that module
/// declares.
///
/// Both ends of the rail ask this question — registration refuses a pairing
/// once, and execution re-asks it of the value it was handed, so a
/// directly-constructed [`RenderStages`] cannot skip the registration gate.
fn fragment_stage_is_reviewed(stages: &RenderStages) -> bool {
    stages.contract.fragment_entry == SOLID_FRAGMENT_ENTRY
        && solid_fragment_spirv(stages.contract.color_format)
            .is_ok_and(|module| module == stages.fragment_spirv.as_slice())
}

/// The refusal for a fragment stage that is not the reviewed module of the
/// pipeline's colour format.
///
/// A capability fact, like the other stage-module refusals: the rail has one
/// reviewed fragment stage per admitted format and no second translation path,
/// so it refuses the pairing instead of executing a module whose semantics it
/// cannot check.
fn fragment_stage_mismatch_refusal(format: AttachmentFormat, entry: &str) -> ProviderError {
    capability_refusal("render_fragment_stage_mismatch")
        .with_field(
            "format_code",
            FieldValue::Unsigned(u64::from(format.code())),
        )
        .with_field("fragment_entry", FieldValue::Text(entry.to_owned()))
        .with_field(
            "reviewed_entry",
            FieldValue::Text(SOLID_FRAGMENT_ENTRY.to_owned()),
        )
        .with_detail(
            "the fragment stage is not the module this rail builds for the colour format, so \
             running it would land bytes the format claim does not cover",
        )
}

/// Execute one admitted render pass and return the attachment's tightly packed
/// texel bytes.
///
/// This is the trace-side entry point of the rail: the pass's shape rules were
/// already checked by core admission, so what is left here is the agreement
/// between the pass and the registered pipeline it names
/// ([`RenderPipelineContract::validate_against`]) and the two shapes the first
/// increment cannot execute — a `Load` that would have to carry the
/// attachment's previous bytes into the image, and a pass whose attachment
/// list is not the single target the registered pipeline was built for. Both
/// are refused as capability facts before any Vulkan object exists, never
/// downgraded to a clear. The registered fragment stage is re-checked against
/// the pipeline's declared format in the same place, for the same reason: the
/// pass is about to be executed with it.
pub(crate) fn execute_render_pass(
    context: &VulkanContext,
    stages: &RenderStages,
    pass: &RenderPassDescriptor,
) -> Result<Vec<u8>, ProviderError> {
    let request = prepare_render_request(stages, pass)?;
    execute_offscreen_render(context, &request)
}

/// Validate one render pass against the pipeline it names and build the
/// rail-side request both execution shapes consume.
///
/// Shared by [`execute_render_pass`] and [`execute_present_render`]: the two
/// shapes disagree only about *which* image the pass renders into and whether a
/// present tail action follows, not about the pass's own contract. Keeping the
/// agreement check in one place means a present pass cannot reach execution
/// through a weaker gate than the offscreen one.
fn prepare_render_request<'a>(
    stages: &'a RenderStages,
    pass: &RenderPassDescriptor,
) -> Result<OffscreenRenderRequest<'a>, ProviderError> {
    stages
        .contract
        .validate_against(pass)
        .map_err(|error| contract_refusal(&error.to_string()))?;
    if !fragment_stage_is_reviewed(stages) {
        return Err(fragment_stage_mismatch_refusal(
            stages.contract.color_format,
            &stages.contract.fragment_entry,
        ));
    }
    let [attachment] = pass.color_attachments.as_slice() else {
        return Err(capability_refusal("color_attachment_limit")
            .with_field(
                "requested",
                FieldValue::Unsigned(pass.color_attachments.len() as u64),
            )
            .with_field(
                "maximum",
                FieldValue::Unsigned(metal_api_core::provider::MAX_COLOR_ATTACHMENTS as u64),
            )
            .with_detail("this rail executes exactly one colour attachment"));
    };
    let clear = match attachment.load {
        LoadOp::Clear(clear) => clear,
        LoadOp::Load => {
            return Err(capability_refusal("attachment_load_op_unsupported")
                .with_field("load_op", FieldValue::Text("load".to_owned()))
                .with_detail(
                    "the first render increment has no rail that uploads an attachment's \
                     previous bytes into the image, so `Load` would silently become a clear",
                ))
        }
        LoadOp::DontCare => {
            return Err(capability_refusal("attachment_load_op_unsupported")
                .with_field("load_op", FieldValue::Text("dont_care".to_owned()))
                .with_detail("core admission refuses `LoadOp::DontCare` for this increment"));
        }
    };
    match attachment.store {
        StoreOp::Store => {}
        StoreOp::DontCare => {
            return Err(capability_refusal("attachment_store_op_unsupported")
                .with_field("store_op", FieldValue::Text("dont_care".to_owned()))
                .with_detail("core admission refuses `StoreOp::DontCare` for this increment"));
        }
    }
    let width = narrow_dimension(attachment.width)?;
    let height = narrow_dimension(attachment.height)?;
    let request = OffscreenRenderRequest {
        format: attachment.format,
        extent: [width, height],
        clear,
        vertex: OffscreenVertexStage {
            entry: &stages.contract.vertex_entry,
            spirv: &stages.vertex_spirv,
        },
    };
    Ok(request)
}

/// Narrow one attachment dimension to the `u32` the Vulkan image extent uses.
///
/// The capability bits already bound every admitted attachment, so this is the
/// second line of defence the capability snapshot is not: it keeps the widening
/// explicit instead of letting a cast truncate an extent the caller asked for.
fn narrow_dimension(dimension: u64) -> Result<u32, ProviderError> {
    u32::try_from(dimension).map_err(|_| {
        capability_refusal("attachment_dimension_limit")
            .with_field("requested", FieldValue::Unsigned(dimension))
            .with_field("maximum", FieldValue::Unsigned(u64::from(u32::MAX)))
    })
}

/// The `VkFormat` a render-contract attachment format names.
///
/// `AttachmentFormat::R32Uint` is expressible in the contract for symmetry with
/// the sampled-texture rail but refused by the first render increment (its
/// texels are integers while `Clear` and the fragment output carry colour
/// bytes), so the rail refuses it with the same slug the core admission uses.
pub(crate) fn attachment_vk_format(format: AttachmentFormat) -> Result<vk::Format, ProviderError> {
    if !format.is_admitted_for_color_attachment() {
        return Err(attachment_format_refusal()
            .with_field(
                "format_code",
                FieldValue::Unsigned(u64::from(format.code())),
            )
            .with_detail("format is expressible but outside the first render increment"));
    }
    Ok(match format {
        AttachmentFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        AttachmentFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        AttachmentFormat::R32Float => vk::Format::R32_SFLOAT,
        // Refused above; the arm keeps the match exhaustive so adding a
        // contract format forces a decision here.
        AttachmentFormat::R32Uint => vk::Format::R32_UINT,
    })
}

/// Map a contract [`ClearColor`] onto the `VkClearColorValue` components for an
/// admitted attachment format.
///
/// The contract's bytes are in the format's **memory** order (that is what the
/// readback compares against), while `VkClearColorValue` takes **component**
/// values: `Bgra8Unorm` therefore needs its red and blue components swapped,
/// and `R32Float` is a single component whose bits are the contract's four
/// bytes reinterpreted, not four components. Getting this wrong is silent: a
/// clear value is only observable where the fragment stage does not store, so
/// a wrong conversion survives every full-coverage fixture. The `R32Uint` arm
/// exists to keep the match exhaustive; the format is refused long before a
/// clear value is built.
pub(crate) fn clear_value_for(format: AttachmentFormat, clear: ClearColor) -> vk::ClearColorValue {
    let bytes = clear.bytes;
    let unorm = |byte: u8| f32::from(byte) / 255.0;
    match format {
        AttachmentFormat::Rgba8Unorm => vk::ClearColorValue {
            float32: [
                unorm(bytes[0]),
                unorm(bytes[1]),
                unorm(bytes[2]),
                unorm(bytes[3]),
            ],
        },
        AttachmentFormat::Bgra8Unorm => vk::ClearColorValue {
            float32: [
                unorm(bytes[2]),
                unorm(bytes[1]),
                unorm(bytes[0]),
                unorm(bytes[3]),
            ],
        },
        AttachmentFormat::R32Float => vk::ClearColorValue {
            float32: [f32::from_le_bytes(bytes), 0.0, 0.0, 0.0],
        },
        AttachmentFormat::R32Uint => vk::ClearColorValue {
            uint32: [u32::from_le_bytes(bytes), 0, 0, 0],
        },
    }
}

/// The `VkFormatFeatureFlags` the selected device reports for one format and
/// tiling.
pub(crate) fn format_features(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
) -> vk::FormatFeatureFlags {
    let properties = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, format)
    };
    match tiling {
        vk::ImageTiling::LINEAR => properties.linear_tiling_features,
        vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features,
        _ => vk::FormatFeatureFlags::empty(),
    }
}

/// Whether the selected device can use `format` as a colour attachment with
/// `tiling`.
///
/// The predicate is exactly `VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT`, not "some
/// attachment-related bit". NVIDIA's linear features on this box carry
/// `COLOR_ATTACHMENT_BLEND` (0x100) without `COLOR_ATTACHMENT` (0x80) —
/// `0x0001dd03` against the optimal `0x0001dd83` — so a broader test admits a
/// combination the driver never promised (`research/docs/23` §3.5).
pub(crate) fn format_supports_color_attachment(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
) -> bool {
    format_features(context, format, tiling).contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
}

/// Admission before any `vkCreateImage`: refuse a format/tiling pair whose
/// `COLOR_ATTACHMENT` bit is clear.
///
/// The refusal is the structured [`ProviderError`] the core admission also
/// produces for this case — `attachment_format_unsupported`, class
/// `Capability`, phase `Resolve` — with the raw format and tiling attached, so
/// the provider layer cannot spell one refusal two ways.
pub(crate) fn admit_color_attachment(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
) -> Result<(), ProviderError> {
    if format_supports_color_attachment(context, format, tiling) {
        return Ok(());
    }
    Err(attachment_format_refusal()
        .with_field("vk_format", FieldValue::Unsigned(format.as_raw() as u64))
        .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
        .with_detail("vkGetPhysicalDeviceFormatProperties reports no COLOR_ATTACHMENT bit"))
}

/// Execute one offscreen render pass and return the attachment's tightly packed
/// texel bytes (`width * height * 4`).
///
/// Contract format admission, the fragment stage the format selects
/// ([`solid_fragment_spirv`]), the device's `COLOR_ATTACHMENT` bit and the
/// `TRANSFER_SRC` bit the readback needs all run before the first
/// `vkCreateImage`, so an unsupported request is refused instead of being
/// handed to the driver.
pub(crate) fn execute_offscreen_render(
    context: &VulkanContext,
    request: &OffscreenRenderRequest<'_>,
) -> Result<Vec<u8>, ProviderError> {
    let format = attachment_vk_format(request.format)?;
    // The fragment stage is the format's, not the caller's: `request` carries no
    // fragment module, so this is the only place one is named and there is no
    // pairing left to get wrong.
    let fragment_spirv = solid_fragment_spirv(request.format)?;
    let tiling = vk::ImageTiling::OPTIMAL;
    admit_color_attachment(context, format, tiling)?;
    if !format_features(context, format, tiling).contains(vk::FormatFeatureFlags::TRANSFER_SRC) {
        return Err(attachment_format_refusal()
            .with_field("vk_format", FieldValue::Unsigned(format.as_raw() as u64))
            .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
            .with_field(
                "missing_feature",
                FieldValue::Text("transfer_src".to_owned()),
            )
            .with_detail(
                "the milestone reads the attachment back through vkCmdCopyImageToBuffer",
            ));
    }

    let [width, height] = request.extent;
    if width == 0 || height == 0 {
        return Err(contract_refusal("render attachment has a zero dimension"));
    }
    let byte_length = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(BYTES_PER_TEXEL))
        .ok_or_else(|| contract_refusal("render attachment bytes overflow u64"))?;

    crate::terminal_refusal(&context.lock_lifecycle())?;
    let queue_index = select_graphics_queue(context)?;
    let vertex_words = spirv_words(request.vertex.spirv)
        .ok_or_else(|| spirv_refusal("vertex SPIR-V is empty or not a multiple of four bytes"))?;
    let fragment_words = spirv_words(fragment_spirv)
        .ok_or_else(|| spirv_refusal("fragment SPIR-V is empty or not a multiple of four bytes"))?;
    let vertex_entry = stage_entry_cstring("vertex", request.vertex.entry)?;
    let fragment_entry = stage_entry_cstring("fragment", SOLID_FRAGMENT_ENTRY)?;

    let mut objects = OffscreenObjects::new(context);
    objects.create_attachment(format, width, height)?;
    objects.create_render_pass(format)?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
    )?;
    let readback_mapping = objects.create_readback(byte_length)?;
    objects.create_command_pool(queue_index)?;
    objects.record(request.format, request.clear, width, height)?;
    objects.submit_and_wait(queue_index)?;

    let texels = unsafe {
        std::slice::from_raw_parts(readback_mapping as *const u8, byte_length as usize).to_vec()
    };
    context.record_buffer_readback();
    context.record_buffer_readback_bytes(texels.len());
    Ok(texels)
}

/// One provider-owned presentable target image (`research/docs/24` §3.6).
///
/// Unlike the one-shot attachment [`OffscreenObjects`] creates per pass, this
/// image is owned by the provider and survives every submission until the
/// allocation lease it backs is released. That is what makes `docs/24` §3.3's
/// second rule ("the target stays readable after `wait`") hold: the target's
/// bytes have to remain observable after a submission, so the image cannot live
/// in the pass's own drop scope (`docs/24` §5.2). It is created once per
/// `(allocation, view)` identity and reused by later submissions that present
/// the same target.
///
/// The image carries `TRANSFER_DST` in addition to the
/// `COLOR_ATTACHMENT | TRANSFER_SRC` pair the offscreen attachment uses: the
/// sentinel pre-fill (`docs/24` §3.1) uploads through
/// `vkCmdClearColorImage`, which is a transfer-destination operation.
pub(crate) struct PresentTargetImage {
    context: Arc<VulkanContext>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// The layout the image is currently in. `UNDEFINED` until a sentinel is
    /// preset or a pass has presented once; a preset sentinel ends in
    /// `COLOR_ATTACHMENT_OPTIMAL`, and each present ends in
    /// `TRANSFER_SRC_OPTIMAL` so the next render pass can start there.
    layout: Mutex<vk::ImageLayout>,
}

impl PresentTargetImage {
    pub(crate) fn create(
        context: Arc<VulkanContext>,
        format: AttachmentFormat,
        width: u64,
        height: u64,
    ) -> Result<Self, ProviderError> {
        let vk_format = attachment_vk_format(format)?;
        let width = narrow_dimension(width)?;
        let height = narrow_dimension(height)?;
        if width == 0 || height == 0 {
            return Err(contract_refusal("present target has a zero dimension"));
        }
        let tiling = vk::ImageTiling::OPTIMAL;
        admit_color_attachment(&context, vk_format, tiling)?;
        let features = format_features(&context, vk_format, tiling);
        for (bit, name) in [
            (vk::FormatFeatureFlags::TRANSFER_SRC, "transfer_src"),
            (vk::FormatFeatureFlags::TRANSFER_DST, "transfer_dst"),
        ] {
            if !features.contains(bit) {
                return Err(attachment_format_refusal()
                    .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field("missing_feature", FieldValue::Text(name.to_owned()))
                    .with_detail("a present target is read back and sentinel-preset"));
            }
        }
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, memory, _) = crate::allocate_image_backing(
            &context,
            &info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "present target",
        )
        .map_err(|error| execution_refusal("create present target image", &error.detail))?;
        let view = crate::create_color_image_view(&context, image, vk_format, "present target")
            .map_err(|error| execution_refusal("create present target view", &error.detail))?;
        Ok(Self {
            context,
            image,
            memory,
            view,
            layout: Mutex::new(vk::ImageLayout::UNDEFINED),
        })
    }

    /// Pre-fill the target with one sentinel texel tiled over the whole image,
    /// then leave it in `COLOR_ATTACHMENT_OPTIMAL` for the pass that will clear
    /// and draw over it (`docs/24` §3.1). `vkCmdClearColorImage` is the byte
    /// upload, with the sentinel's four bytes mapped onto the format's
    /// component order by the same [`clear_value_for`] the render clear uses.
    pub(crate) fn preset_sentinel(
        &mut self,
        format: AttachmentFormat,
        sentinel: &[u8],
    ) -> Result<(), ProviderError> {
        let context = &self.context;
        let clear = ClearColor::new(
            sentinel
                .try_into()
                .map_err(|_| contract_refusal("a present sentinel is exactly four bytes"))?,
        );
        crate::terminal_refusal(&context.lock_lifecycle())?;
        let queue_index = select_graphics_queue(context)?;
        let family = context
            .queue_families
            .get(queue_index)
            .copied()
            .ok_or_else(|| {
                execution_refusal("preset present sentinel", "queue index is unknown")
            })?;
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        let pool =
            unsafe { context.device.create_command_pool(&pool_info, None) }.map_err(|error| {
                execution_refusal("create sentinel command pool", &error.to_string())
            })?;
        let command = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| {
            execution_refusal("allocate sentinel command buffer", &error.to_string())
        })?[0];

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            context
                .device
                .begin_command_buffer(command, &begin)
                .map_err(|error| {
                    execution_refusal("begin sentinel command buffer", &error.to_string())
                })?;
            let undefined_to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(self.image)
                .subresource_range(subresource);
            context.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[undefined_to_dst],
            );
            context.device.cmd_clear_color_image(
                command,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value_for(format, clear),
                &[subresource],
            );
            let dst_to_color = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .image(self.image)
                .subresource_range(subresource);
            context.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[dst_to_color],
            );
            context
                .device
                .end_command_buffer(command)
                .map_err(|error| {
                    execution_refusal("end sentinel command buffer", &error.to_string())
                })?;
        }

        let result = (|| {
            let _execution = context.lock_queue(queue_index).map_err(|_| {
                submission_refusal("submit present sentinel", "queue lock is poisoned")
            })?;
            context.notify_enqueue(queue_index);
            let fence = unsafe {
                context
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }
            .map_err(|error| execution_refusal("create sentinel fence", &error.to_string()))?;
            let commands = [command];
            let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
            let submitted = context
                .submit_commands(queue_index, &submits, fence)
                .map_err(|result| {
                    driver_refusal(
                        context,
                        ProviderPhase::Submit,
                        "submit present sentinel",
                        result,
                    )
                });
            if let Err(error) = submitted {
                unsafe { context.device.destroy_fence(fence, None) };
                return Err(error);
            }
            context.record_queue_submission(queue_index);
            let waited = context
                .wait_for_fence(fence, crate::FENCE_TIMEOUT_NS)
                .map_err(|result| {
                    driver_refusal(
                        context,
                        ProviderPhase::Wait,
                        "wait for present sentinel",
                        result,
                    )
                });
            unsafe { context.device.destroy_fence(fence, None) };
            waited?;
            context.record_queue_retirement(queue_index);
            Ok(())
        })();
        unsafe { context.device.destroy_command_pool(pool, None) };
        result?;
        *self
            .layout
            .get_mut()
            .expect("unlocked present target layout") = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        Ok(())
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Begin one present round trip on this target, holding its layout lock
    /// until the returned guard is dropped.
    ///
    /// The guard is the target's serialization point: a present action must
    /// read the layout it is about to submit against and publish the new
    /// layout before another present on the same target can start. Without
    /// it, two concurrent presents of one target could interleave so that the
    /// second submits `initialLayout = COLOR_ATTACHMENT_OPTIMAL` after the
    /// first has already moved the image to `TRANSFER_SRC_OPTIMAL`, which is a
    /// layout mismatch the driver is entitled to reject (`research/docs/24`
    /// §3.3 rule 1). The caller publishes the terminal layout by writing
    /// through the guard; dropping it releases the next round trip.
    pub(crate) fn begin_present(&self) -> std::sync::MutexGuard<'_, vk::ImageLayout> {
        self.layout_lock()
    }

    fn layout_lock(&self) -> std::sync::MutexGuard<'_, vk::ImageLayout> {
        self.layout
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for PresentTargetImage {
    fn drop(&mut self) {
        unsafe {
            if self.view != vk::ImageView::null() {
                self.context.device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                self.context.device.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.memory, None);
            }
        }
    }
}

/// The present path chains the colour store out of the render pass and into
/// the copy-out through one access class, `COLOR_ATTACHMENT_WRITE` at
/// `COLOR_ATTACHMENT_OUTPUT`. The render pass's `0 → EXTERNAL` dependency and
/// the explicit terminal transition below must both name it: a dependency
/// whose second scope was `COLOR_ATTACHMENT_READ` would leave the barrier's
/// `srcAccessMask` outside the availability chain, so the copy-out would not
/// be synchronized with the colour store on a validation-layer-strict driver
/// (`research/docs/24` §3.3 rule 1).
const PRESENT_WRITE_STAGE: vk::PipelineStageFlags = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;

/// The access class [`PRESENT_WRITE_STAGE`] hands from the pass to the present
/// transition.
const PRESENT_WRITE_ACCESS: vk::AccessFlags = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

/// The render pass's `0 → EXTERNAL` dependency for a present pass: the stored
/// colour write is made available to the explicit present transition that
/// follows the pass.
fn present_subpass_dependency() -> vk::SubpassDependency {
    vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(PRESENT_WRITE_STAGE)
        .dst_stage_mask(PRESENT_WRITE_STAGE)
        .src_access_mask(PRESENT_WRITE_ACCESS)
        .dst_access_mask(PRESENT_WRITE_ACCESS)
}

/// The present action's terminal transition (`docs/24` §3.6): the rendered
/// target moves from the colour-attachment state to the host-readable state
/// the copy-out consumes.
fn present_transition_barrier(image: vk::Image) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(PRESENT_WRITE_ACCESS)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

/// Execute one render pass whose attachment is a provider-owned present target,
/// and return the target's tightly packed texel bytes.
///
/// The pass renders into `target` (rather than a fresh offscreen image), then
/// records the present action's terminal transition and copy-out in the same
/// command buffer. The one acquire and one present are counted around the pass
/// (`docs/24` §5.3). The target image itself is not destroyed here: it is the
/// provider's, so it stays readable after `wait` (`docs/24` §3.3 rule 2).
pub(crate) fn execute_present_render(
    context: &VulkanContext,
    stages: &RenderStages,
    pass: &RenderPassDescriptor,
    target: &PresentTargetImage,
) -> Result<Vec<u8>, ProviderError> {
    let request = prepare_render_request(stages, pass)?;
    let [width, height] = request.extent;
    if width == 0 || height == 0 {
        return Err(contract_refusal("render attachment has a zero dimension"));
    }
    let byte_length = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(BYTES_PER_TEXEL))
        .ok_or_else(|| contract_refusal("render attachment bytes overflow u64"))?;

    crate::terminal_refusal(&context.lock_lifecycle())?;
    let queue_index = select_graphics_queue(context)?;
    let fragment_spirv = solid_fragment_spirv(request.format)?;
    let vk_format = attachment_vk_format(request.format)?;
    let vertex_words = spirv_words(request.vertex.spirv)
        .ok_or_else(|| spirv_refusal("vertex SPIR-V is empty or not a multiple of four bytes"))?;
    let fragment_words = spirv_words(fragment_spirv)
        .ok_or_else(|| spirv_refusal("fragment SPIR-V is empty or not a multiple of four bytes"))?;
    let vertex_entry = stage_entry_cstring("vertex", request.vertex.entry)?;
    let fragment_entry = stage_entry_cstring("fragment", SOLID_FRAGMENT_ENTRY)?;

    // One acquire per present action, before the pass runs (`docs/24` §3.6).
    //
    // The guard serializes the whole round trip on this target's layout: the
    // submission below declares `*layout` as its `initialLayout`, and the
    // terminal layout is published through the same guard before it drops, so
    // a concurrent present of this target cannot read a layout that another
    // submission has already changed.
    let mut layout = target.begin_present();
    context.record_present_acquire();
    let mut objects = OffscreenObjects::new(context);
    objects.attach_present_target(target, *layout);
    objects.create_render_pass(vk_format)?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
    )?;
    let readback_mapping = objects.create_readback(byte_length)?;
    objects.create_command_pool(queue_index)?;
    objects.record(request.format, request.clear, width, height)?;
    objects.submit_and_wait(queue_index)?;

    let texels = unsafe {
        std::slice::from_raw_parts(readback_mapping as *const u8, byte_length as usize).to_vec()
    };
    context.record_buffer_readback();
    context.record_buffer_readback_bytes(texels.len());
    // One present per present action, after the terminal transition and
    // readback have landed (`docs/24` §3.6).
    context.record_present();
    *layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    Ok(texels)
}

/// The queue this rail submits graphics work to.
///
/// The selected device only *may* have created a graphics-capable family: the
/// primary family is chosen for compute reasons and the design keeps the family
/// plan unchanged for the first render increment, so a device without a
/// graphics family is a capability refusal rather than a silent fallback
/// (`research/docs/23` §7.3).
fn select_graphics_queue(context: &VulkanContext) -> Result<usize, ProviderError> {
    let families = unsafe {
        context
            .instance
            .get_physical_device_queue_family_properties(context.physical)
    };
    context
        .queue_families
        .iter()
        .position(|family| {
            families
                .get(*family as usize)
                .is_some_and(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        })
        .ok_or_else(|| {
            capability_refusal("render_graphics_queue_unavailable")
                .with_detail("the selected device created no queue in a graphics-capable family")
        })
}

/// Owns every object one offscreen render pass creates.
///
/// A failure at any step destroys exactly what already exists, which is the
/// ownership shape `ExecutionResources` gives the compute rail scoped to this
/// one-shot rail (`research/docs/23` §6 Step 3b).
struct OffscreenObjects<'a> {
    context: &'a VulkanContext,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Whether this scope created `image`/`memory`/`view` and must destroy them
    /// on Drop. A present pass borrows the provider-owned [`PresentTargetImage`]
    /// instead, so its per-pass scope must not destroy the target when it
    /// finishes (`docs/24` §5.2: the target survives the submission).
    owns_attachment: bool,
    /// Whether this pass hands its attachment on as a present target. When set,
    /// the render pass ends in `COLOR_ATTACHMENT_OPTIMAL` and `record` inserts
    /// the explicit present layout transition before the copy-out
    /// (`docs/24` §3.3 rule 1).
    present: bool,
    /// The layout `image` is in when the render pass begins. Offscreen
    /// attachments start `UNDEFINED`; a present target may have been preset
    /// with a sentinel (→ `COLOR_ATTACHMENT_OPTIMAL`) or already presented once
    /// (→ `TRANSFER_SRC_OPTIMAL`).
    initial_layout: vk::ImageLayout,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'a> OffscreenObjects<'a> {
    fn new(context: &'a VulkanContext) -> Self {
        Self {
            context,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            owns_attachment: true,
            present: false,
            initial_layout: vk::ImageLayout::UNDEFINED,
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vertex_module: vk::ShaderModule::null(),
            fragment_module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            readback_buffer: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            command_pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        }
    }

    /// Render this pass into a provider-owned present target instead of a
    /// freshly created offscreen attachment. The target's image and view are
    /// borrowed for the pass's lifetime; its memory stays owned by the provider
    /// (`docs/24` §5.2).
    ///
    /// `initial_layout` is passed in rather than read from the target: the
    /// caller holds the target's present round-trip guard, and the guard's
    /// value *is* the layout this submission must declare.
    fn attach_present_target(
        &mut self,
        target: &PresentTargetImage,
        initial_layout: vk::ImageLayout,
    ) {
        self.image = target.image();
        self.view = target.view();
        self.owns_attachment = false;
        self.present = true;
        self.initial_layout = initial_layout;
    }

    /// The 2D single-sample optimal-tiling colour attachment.
    ///
    /// `TRANSFER_SRC` is part of the usage because the readback copies the
    /// attachment out; `DEVICE_LOCAL` is the memory class the probe used for
    /// every optimal-tiling candidate.
    fn create_attachment(
        &mut self,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<(), ProviderError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, memory, _) = crate::allocate_image_backing(
            self.context,
            &info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "attachment",
        )
        .map_err(|error| execution_refusal("create attachment image", &error.detail))?;
        self.image = image;
        self.memory = memory;
        self.view = crate::create_color_image_view(self.context, image, format, "attachment")
            .map_err(|error| execution_refusal("create attachment view", &error.detail))?;
        Ok(())
    }

    /// The single-colour-attachment render pass with the probe's dependency
    /// pair: `EXTERNAL → 0` makes the clear/write visible to colour output, and
    /// `0 → EXTERNAL` makes the stored texels visible to the copy that reads
    /// them. An offscreen pass ends directly in
    /// `finalLayout = TRANSFER_SRC_OPTIMAL` so the copy runs without a further
    /// transition (`research/docs/23` §7.1); a present pass ends in
    /// `COLOR_ATTACHMENT_OPTIMAL` instead, and `record` inserts the explicit
    /// present layout transition before the copy (`docs/24` §3.3).
    fn create_render_pass(&mut self, format: vk::Format) -> Result<(), ProviderError> {
        let final_layout = if self.present {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        let attachments = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(self.initial_layout)
            .final_layout(final_layout)];
        let color_refs = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)];
        let mut dependencies = vec![vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        dependencies.push(if self.present {
            // The present path does its own `COLOR_ATTACHMENT_OPTIMAL →
            // TRANSFER_SRC_OPTIMAL` transition in `record`, so the render pass
            // only has to make the store available to the barrier that follows
            // (`docs/24` §3.3 rule 1). The dependency's second scope and the
            // barrier's first scope are the *same* colour-write access class:
            // a dependency that hands the store on as `COLOR_ATTACHMENT_READ`
            // would leave the barrier's `srcAccessMask =
            // COLOR_ATTACHMENT_WRITE` outside the availability chain, so the
            // copy-out would not be synchronized with the colour store.
            present_subpass_dependency()
        } else {
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ)
        });
        let info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        self.render_pass = unsafe { self.context.device.create_render_pass(&info, None) }
            .map_err(|error| execution_refusal("create render pass", &error.to_string()))?;
        Ok(())
    }

    fn create_framebuffer(&mut self, width: u32, height: u32) -> Result<(), ProviderError> {
        let views = [self.view];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        self.framebuffer = unsafe { self.context.device.create_framebuffer(&info, None) }
            .map_err(|error| execution_refusal("create framebuffer", &error.to_string()))?;
        Ok(())
    }

    /// The graphics pipeline of the milestone: two stages, no vertex input, no
    /// dynamic state beyond the explicit viewport/scissor, no blend/cull/depth.
    /// Every absent state is expressed by not enabling it (`research/docs/23`
    /// §3.2).
    fn create_pipeline(
        &mut self,
        vertex_words: &[u32],
        fragment_words: &[u32],
        vertex_entry: &CStr,
        fragment_entry: &CStr,
    ) -> Result<(), ProviderError> {
        self.vertex_module = unsafe {
            self.context.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(vertex_words),
                None,
            )
        }
        .map_err(|error| execution_refusal("create vertex shader module", &error.to_string()))?;
        self.fragment_module = unsafe {
            self.context.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(fragment_words),
                None,
            )
        }
        .map_err(|error| execution_refusal("create fragment shader module", &error.to_string()))?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex_module)
                .name(vertex_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_module)
                .name(fragment_entry),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        self.pipeline_layout = unsafe {
            self.context
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        }
        .map_err(|error| execution_refusal("create pipeline layout", &error.to_string()))?;

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);
        let pipelines = unsafe {
            self.context
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(pipelines, error)| {
            for pipeline in pipelines {
                unsafe { self.context.device.destroy_pipeline(pipeline, None) };
            }
            execution_refusal("create graphics pipeline", &error.to_string())
        })?;
        self.pipeline = pipelines.into_iter().next().ok_or_else(|| {
            execution_refusal("create graphics pipeline", "driver returned no pipeline")
        })?;
        Ok(())
    }

    /// The host-visible destination of the attachment copy.
    ///
    /// Returns the persistent mapping of the readback memory.
    fn create_readback(&mut self, byte_length: u64) -> Result<usize, ProviderError> {
        let info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.context.device.create_buffer(&info, None) }
            .map_err(|error| execution_refusal("create readback buffer", &error.to_string()))?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    "find readback memory type",
                    &error.to_string(),
                ));
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    "allocate readback memory",
                    &error.to_string(),
                ));
            }
        };
        if let Err(error) = unsafe { self.context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.context.device.destroy_buffer(buffer, None);
                self.context.device.free_memory(memory, None);
            }
            return Err(execution_refusal(
                "bind readback memory",
                &error.to_string(),
            ));
        }
        let mapping = match unsafe {
            self.context.device.map_memory(
                memory,
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapping) => mapping as usize,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(execution_refusal("map readback memory", &error.to_string()));
            }
        };
        self.readback_buffer = buffer;
        self.readback_memory = memory;
        Ok(mapping)
    }

    fn create_command_pool(&mut self, queue_index: usize) -> Result<(), ProviderError> {
        let family = self
            .context
            .queue_families
            .get(queue_index)
            .copied()
            .ok_or_else(|| execution_refusal("create command pool", "queue index is unknown"))?;
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        self.command_pool = unsafe { self.context.device.create_command_pool(&pool_info, None) }
            .map_err(|error| execution_refusal("create command pool", &error.to_string()))?;
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let commands = unsafe { self.context.device.allocate_command_buffers(&allocation) }
            .map_err(|error| execution_refusal("allocate command buffer", &error.to_string()))?;
        self.command = commands.into_iter().next().ok_or_else(|| {
            execution_refusal(
                "allocate command buffer",
                "driver returned no command buffer",
            )
        })?;
        Ok(())
    }

    /// Record clear → draw → copy-out on the one command buffer.
    ///
    /// The clear value is a function of the attachment format: the contract's
    /// bytes are in the format's memory order, while `VkClearColorValue`
    /// components follow the format's *component* order.
    fn record(
        &mut self,
        format: AttachmentFormat,
        clear: ClearColor,
        width: u32,
        height: u32,
    ) -> Result<(), ProviderError> {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command, &begin)
        }
        .map_err(|error| execution_refusal("begin command buffer", &error.to_string()))?;

        let clear_value = vk::ClearValue {
            color: clear_value_for(format, clear),
        };
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(render_area)
            .clear_values(std::slice::from_ref(&clear_value));
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.context.device.cmd_begin_render_pass(
                self.command,
                &pass_begin,
                vk::SubpassContents::INLINE,
            );
            self.context.device.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.context
                .device
                .cmd_set_viewport(self.command, 0, std::slice::from_ref(&viewport));
            self.context
                .device
                .cmd_set_scissor(self.command, 0, std::slice::from_ref(&scissor));
            self.context
                .device
                .cmd_draw(self.command, FULL_SCREEN_TRIANGLE_VERTICES, 1, 0, 0);
            self.context.device.cmd_end_render_pass(self.command);
        }

        if self.present {
            // The present action's "terminal transition": make the completed
            // colour store visible to the copy-out, in the same command buffer
            // as the render so the present cannot run before its writer
            // (`docs/24` §3.3 rule 1). The equivalent terminal state is
            // `TRANSFER_SRC_OPTIMAL`, i.e. "readable by the host after `wait`"
            // (`docs/24` §3.6), not a real `VkQueuePresentKHR`.
            let barrier = present_transition_barrier(self.image);
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    PRESENT_WRITE_STAGE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }

        let copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.context.device.cmd_copy_image_to_buffer(
                self.command,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback_buffer,
                std::slice::from_ref(&copy),
            );
        }
        unsafe { self.context.device.end_command_buffer(self.command) }
            .map_err(|error| execution_refusal("end command buffer", &error.to_string()))?;
        Ok(())
    }

    fn submit_and_wait(&mut self, queue_index: usize) -> Result<(), ProviderError> {
        let _execution = self
            .context
            .lock_queue(queue_index)
            .map_err(|_| submission_refusal("submit render pass", "queue lock is poisoned"))?;
        self.context.notify_enqueue(queue_index);
        self.fence = unsafe {
            self.context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|error| execution_refusal("create render fence", &error.to_string()))?;
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        if let Err(result) = self
            .context
            .submit_commands(queue_index, &submits, self.fence)
        {
            return Err(driver_refusal(
                self.context,
                ProviderPhase::Submit,
                "submit render pass",
                result,
            ));
        }
        self.context.record_queue_submission(queue_index);
        if let Err(result) = self
            .context
            .wait_for_fence(self.fence, crate::FENCE_TIMEOUT_NS)
        {
            return Err(driver_refusal(
                self.context,
                ProviderPhase::Wait,
                "wait for render fence",
                result,
            ));
        }
        self.context.record_queue_retirement(queue_index);
        Ok(())
    }
}

impl<'a> Drop for OffscreenObjects<'a> {
    fn drop(&mut self) {
        unsafe {
            if self.fence != vk::Fence::null() {
                self.context.device.destroy_fence(self.fence, None);
            }
            if self.command_pool != vk::CommandPool::null() {
                self.context
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                self.context.device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.context
                    .device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.fragment_module != vk::ShaderModule::null() {
                self.context
                    .device
                    .destroy_shader_module(self.fragment_module, None);
            }
            if self.vertex_module != vk::ShaderModule::null() {
                self.context
                    .device
                    .destroy_shader_module(self.vertex_module, None);
            }
            if self.framebuffer != vk::Framebuffer::null() {
                self.context
                    .device
                    .destroy_framebuffer(self.framebuffer, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.context
                    .device
                    .destroy_render_pass(self.render_pass, None);
            }
            if self.owns_attachment {
                if self.view != vk::ImageView::null() {
                    self.context.device.destroy_image_view(self.view, None);
                }
                if self.image != vk::Image::null() {
                    self.context.device.destroy_image(self.image, None);
                }
                if self.memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(self.memory, None);
                }
            }
            if self.readback_memory != vk::DeviceMemory::null() {
                self.context.device.unmap_memory(self.readback_memory);
            }
            if self.readback_buffer != vk::Buffer::null() {
                self.context
                    .device
                    .destroy_buffer(self.readback_buffer, None);
            }
            if self.readback_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.readback_memory, None);
            }
        }
    }
}

fn spirv_words(bytes: &[u8]) -> Option<Vec<u32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect(),
    )
}

/// The `CStr` a graphics-pipeline stage binds for one entry name.
fn stage_entry_cstring(stage: &'static str, entry: &str) -> Result<CString, ProviderError> {
    if entry.is_empty() {
        return Err(stage_entry_refusal(stage, "the entry name is empty"));
    }
    CString::new(entry)
        .map_err(|_| stage_entry_refusal(stage, "the entry name carries an interior NUL"))
}

fn tiling_name(tiling: vk::ImageTiling) -> &'static str {
    match tiling {
        vk::ImageTiling::LINEAR => "linear",
        vk::ImageTiling::OPTIMAL => "optimal",
        _ => "unknown",
    }
}

/// The one structured refusal the core admission and this rail share for a
/// colour-attachment format the device cannot take.
fn attachment_format_refusal() -> ProviderError {
    capability_refusal("attachment_format_unsupported")
}

fn capability_refusal(slug: &'static str) -> ProviderError {
    let mut error =
        ProviderError::new(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
            .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error
}

fn contract_refusal(detail: &str) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Resolve,
        ProviderErrorClass::Args,
        "trace_contract_invalid",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(detail.to_owned())
}

fn spirv_refusal(detail: &str) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Resolve,
        ProviderErrorClass::Capability,
        "spirv_module_invalid",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(detail.to_owned())
}

/// A registered render pipeline that cannot be built at all.
///
/// Registration is where a render pipeline's own shape is settled, so this is
/// deliberately not the trace-level `trace_contract_invalid` the execution path
/// uses when a pass disagrees with the pipeline it names: no trace exists yet.
fn render_pipeline_contract_refusal(detail: &str) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Compile,
        ProviderErrorClass::Args,
        "render_pipeline_contract_invalid",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(detail.to_owned())
}

/// A stage entry name the graphics pipeline cannot bind.
fn stage_entry_refusal(stage: &'static str, detail: &str) -> ProviderError {
    capability_refusal("render_stage_entry_invalid")
        .with_field("stage", FieldValue::Text(stage.to_owned()))
        .with_detail(detail.to_owned())
}

fn execution_refusal(step: &str, detail: &str) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Encode,
        ProviderErrorClass::Execute,
        "render_execution_failed",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(format!("{step}: {detail}"))
}

fn submission_refusal(step: &str, detail: &str) -> ProviderError {
    let mut error = ProviderError::new(
        ProviderPhase::Wait,
        ProviderErrorClass::Execute,
        "render_submission_failed",
    )
    .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error.with_detail(format!("{step}: {detail}"))
}

/// Structured error for one driver answer at a render queue boundary.
///
/// A render submission meets the same driver as a compute submission, so a
/// `VK_ERROR_DEVICE_LOST` here is routed through the core lifecycle exactly
/// like the compute rail's loss: `VulkanContext::observe_device_loss` marks the
/// instance terminal and queries `VK_EXT_device_fault`, and the error
/// carries the raw result plus the fault record. Every other answer keeps the
/// render rail's own refusal, whose detail is the driver text. The slug stays
/// the rail's so a caller can still tell which boundary reported the loss.
fn driver_refusal(
    context: &VulkanContext,
    phase: ProviderPhase,
    step: &str,
    result: vk::Result,
) -> ProviderError {
    if result != vk::Result::ERROR_DEVICE_LOST {
        return submission_refusal(step, &result.to_string());
    }
    crate::device_loss_refusal(context, phase, "render_submission_failed", step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AllocationId, PipelineId, RenderAttachment, VertexLayout, ViewId,
    };

    /// Vertex stage: positions from `gl_VertexIndex`, no vertex buffers, no
    /// varyings. `spirv-as` output of `render_spv/fullscreen_triangle.vert.spvasm`,
    /// which selects `(-1,-1) (3,-1) (-1,3)` — the oversize triangle covers every
    /// pixel centre of a 2×2 viewport, so "the draw really ran" stays falsifiable
    /// (`research/docs/23` §1.3).
    const FULL_SCREEN_TRIANGLE_VERT_SPV: &[u8] =
        include_bytes!("render_spv/fullscreen_triangle.vert.spv");

    /// Fragment stage: writes `(64/255, 128/255, 192/255, 1)`, which an 8-bit UNORM
    /// attachment stores as `40 80 c0 ff` (R,G,B,A) or `c0 80 40 ff` (B,G,R,A).
    ///
    /// The constants are byte/255 rather than the round decimals `0.25/0.5/0.75`
    /// on purpose: `0.5 * 255 = 127.5` is a half-integer tie, and the probe read
    /// `0x80` back on Lavapipe but `0x7f` on both the NVIDIA driver and dzn
    /// (`research/docs/23` §3.5). Byte/255 values sit at least 3.7e-6 away from a
    /// tie on every driver, so they are the parity-stable discipline the fixture
    /// has to follow. The same discipline is what the float stage's `64/255`
    /// follows, so the three formats write the same nominal colour.
    const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_unorm8.frag.spv");

    /// Fragment stage of the `R32_SFLOAT` attachment: the same colour's red
    /// component, `64/255`, as one `float`.
    const SOLID_R32F_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_r32f.frag.spv");

    /// The readback a 2×2 `R8G8B8A8_UNORM` attachment must hold when the
    /// fragment shader stores `64/255, 128/255, 192/255, 1`.
    const EXPECTED_RGBA8_TEXELS: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

    /// The same colour in a `B8G8R8A8_UNORM` attachment: the *bytes* carry the
    /// B,G,R,A order the image format asks for, so the first byte is the stored
    /// blue (`0xc0` = 192) and the third is the stored red (`0x40` = 64).
    const EXPECTED_BGRA8_TEXELS: [u8; 4] = [0xc0, 0x80, 0x40, 0xff];

    /// One texel of an `R32_SFLOAT` attachment: the little-endian bytes of the
    /// `float` `64/255` (`0x3e808081`) the float stage stores. Every texel is the
    /// same four bytes, because a float attachment quantises nothing.
    const EXPECTED_R32F_TEXEL: [u8; 4] = [0x81, 0x80, 0x80, 0x3e];

    /// The `LoadOp::Clear` sentinel (`research/docs/23` §1.3): a texel that
    /// still holds it proves the draw did not cover that pixel.
    const CLEAR_SENTINEL: u8 = 0xfe;

    /// Vertex stage that collapses the triangle onto `(-1,-1)`: in a 2×2
    /// viewport it covers only the pixel at `(0,0)`, so the other three texels
    /// keep the `LoadOp::Clear` bytes. `spirv-as` output of
    /// `render_spv/single_pixel.vert.spvasm`.
    const SINGLE_PIXEL_VERT_SPV: &[u8] = include_bytes!("render_spv/single_pixel.vert.spv");

    /// The clear value a format test uses: four distinct bytes so a swapped or
    /// reinterpreted component order cannot coincide with the expected bytes.
    const DISTINCT_CLEAR: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

    fn single_pixel_vertex() -> OffscreenVertexStage<'static> {
        OffscreenVertexStage {
            entry: "single_pixel",
            spirv: SINGLE_PIXEL_VERT_SPV,
        }
    }

    /// The reviewed vertex stage of the milestone, under the entry name its
    /// `.spvasm` source declares.
    fn milestone_vertex() -> OffscreenVertexStage<'static> {
        OffscreenVertexStage {
            entry: "vertex_main",
            spirv: FULL_SCREEN_TRIANGLE_VERT_SPV,
        }
    }

    /// One registration for `format` whose fragment stage is the reviewed module
    /// for that format.
    fn reviewed_stages(format: AttachmentFormat) -> RenderStages {
        RenderStages {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: SOLID_FRAGMENT_ENTRY.to_owned(),
                color_format: format,
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: solid_fragment_spirv(format)
                .expect("every admitted format has a reviewed stage")
                .to_vec(),
        }
    }

    /// One 2×2 render pass naming an attachment of `format`, holding the clear
    /// sentinel the coverage assertions look for.
    fn milestone_pass(format: AttachmentFormat) -> RenderPassDescriptor {
        RenderPassDescriptor {
            pipeline: PipelineId::new(11),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(21),
                allocation_id: AllocationId::new(31),
                format,
                width: 2,
                height: 2,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            vertices: 3,
            present: None,
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn device_context() -> Option<VulkanContext> {
        match VulkanContext::new() {
            Ok(context) => Some(context),
            Err(error) => {
                eprintln!("SKIP: no Vulkan device: {error}");
                None
            }
        }
    }

    /// The render pass's `0 → EXTERNAL` dependency and the explicit present
    /// transition have to agree about the access class that carries the colour
    /// store out of the pass. They disagreed once (`dst_access =
    /// COLOR_ATTACHMENT_READ` against the barrier's `src_access =
    /// COLOR_ATTACHMENT_WRITE`), which Lavapipe tolerated but a validation
    /// layer would not; this pins the agreement at the constructors both call
    /// sites use.
    #[test]
    fn the_present_subpass_dependency_hands_the_colour_write_to_the_present_barrier() {
        let dependency = present_subpass_dependency();
        let barrier = present_transition_barrier(vk::Image::null());
        assert_eq!(dependency.dst_subpass, vk::SUBPASS_EXTERNAL);
        // The bits are compared through `as_raw`: ash's bitflags and layout
        // newtypes do not implement `Debug` without the crate's `debug`
        // feature, which `assert_eq!` would need. `VkImageMemoryBarrier`
        // carries no stage mask — the stage is a parameter of
        // `vkCmdPipelineBarrier` — so the command's source stage is pinned by
        // both call sites naming [`PRESENT_WRITE_STAGE`].
        assert_eq!(
            dependency.dst_stage_mask.as_raw(),
            PRESENT_WRITE_STAGE.as_raw()
        );
        assert_eq!(
            dependency.dst_access_mask.as_raw(),
            PRESENT_WRITE_ACCESS.as_raw()
        );
        assert_eq!(
            barrier.src_access_mask.as_raw(),
            PRESENT_WRITE_ACCESS.as_raw()
        );
        assert_eq!(
            barrier.dst_access_mask.as_raw(),
            vk::AccessFlags::TRANSFER_READ.as_raw()
        );
        assert_eq!(
            barrier.old_layout.as_raw(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw()
        );
        assert_eq!(
            barrier.new_layout.as_raw(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw()
        );
    }

    /// One present round trip holds the target's layout lock until it has
    /// published the terminal layout, so a second present of the same target
    /// cannot submit against a layout the previous submission already left.
    /// The guard's blocking behaviour is the invariant the fix relies on, so
    /// the test observes it directly instead of hoping a race shows up.
    #[test]
    fn a_present_round_trip_excludes_a_second_present_on_the_same_target() {
        let Some(context) = device_context() else {
            return;
        };
        let context = std::sync::Arc::new(context);
        let target = std::sync::Arc::new(
            PresentTargetImage::create(
                std::sync::Arc::clone(&context),
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
            )
            .expect("the present target is created"),
        );

        let first = target.begin_present();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let waiting = std::sync::Arc::clone(&target);
        let handle = std::thread::spawn(move || {
            let _second = waiting.begin_present();
            let _ = started_tx.send(());
        });
        assert!(
            started_rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a second present round trip must wait for the first to publish its layout"
        );
        drop(first);
        assert!(
            started_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .is_ok(),
            "the second round trip starts once the first published its layout"
        );
        handle.join().expect("the waiting thread ends");
    }

    /// Execute the milestone's 2×2 offscreen pass against `format` and return the
    /// readback, printing the raw bytes so the run's log carries the evidence the
    /// assertions below are about.
    fn offscreen_readback(context: &VulkanContext, format: AttachmentFormat) -> Vec<u8> {
        let texels = execute_offscreen_render(
            context,
            &OffscreenRenderRequest {
                format,
                extent: [2, 2],
                clear: ClearColor::new([CLEAR_SENTINEL; 4]),
                vertex: milestone_vertex(),
            },
        )
        .unwrap_or_else(|error| panic!("the 2x2 {format:?} render pass executes: {error:?}"));
        eprintln!(
            "{format:?} readback: {} (first texel: {})",
            hex(&texels),
            hex(&texels[..4])
        );
        texels
    }

    /// The contract's bytes are memory order; `VkClearColorValue` wants
    /// components. The three admitted formats disagree about how those two
    /// relate, and the disagreement is silent in every full-coverage fixture —
    /// this test states the mapping without a device (the partial-coverage
    /// tests below then prove it against a real attachment).
    #[test]
    fn clear_components_follow_the_format_not_the_byte_order() {
        let clear = ClearColor::new(DISTINCT_CLEAR);

        // SAFETY: `clear_value_for` initialises exactly one union field, and
        // this test reads the same field it sets.
        let rgba = unsafe { clear_value_for(AttachmentFormat::Rgba8Unorm, clear).float32 };
        assert_eq!(
            rgba,
            [
                0x11u8 as f32 / 255.0,
                0x22u8 as f32 / 255.0,
                0x33u8 as f32 / 255.0,
                0x44u8 as f32 / 255.0,
            ]
        );

        // B,G,R,A memory order: the stored blue is component 0 and the stored
        // red is component 2, so the components are the bytes with 0 and 2
        // exchanged.
        let bgra = unsafe { clear_value_for(AttachmentFormat::Bgra8Unorm, clear).float32 };
        assert_eq!(
            bgra,
            [
                0x33u8 as f32 / 255.0,
                0x22u8 as f32 / 255.0,
                0x11u8 as f32 / 255.0,
                0x44u8 as f32 / 255.0,
            ]
        );
        assert_eq!(bgra[0], rgba[2], "the blue component is the same value");
        assert_eq!(bgra[2], rgba[0], "the red component is the same value");

        // One float, not four components: the four bytes reinterpreted.
        let r32f = unsafe { clear_value_for(AttachmentFormat::R32Float, clear).float32 };
        let expected = f32::from_le_bytes(DISTINCT_CLEAR);
        assert_eq!(r32f[0].to_bits(), expected.to_bits());
        assert_eq!(r32f[1..], [0.0, 0.0, 0.0]);
    }

    /// The rail's structural guarantee, stated without a device: `format` selects
    /// the fragment stage, the map is total over the admitted formats, and the one
    /// format outside the increment has no stage at all.
    #[test]
    fn every_admitted_format_selects_a_reviewed_fragment_stage() {
        // `solid_fragment_spirv` matches every `AttachmentFormat` variant with no
        // default arm, so "the map covers the contract" is a compile-time fact;
        // what is checked here is the content of each arm.
        for format in AttachmentFormat::ADMITTED {
            let module = solid_fragment_spirv(format).expect("an admitted format has a stage");
            assert!(
                spirv_words(module).is_some(),
                "{format:?} must name a whole number of SPIR-V words: {} bytes",
                module.len()
            );
        }
        assert_eq!(
            solid_fragment_spirv(AttachmentFormat::Rgba8Unorm).expect("admitted"),
            SOLID_UNORM8_FRAG_SPV
        );
        // The B,G,R,A layout shares the 8-bit module on purpose: the channel order
        // lives in the image format, so the bytes differ while the stage does not.
        assert_eq!(
            solid_fragment_spirv(AttachmentFormat::Bgra8Unorm).expect("admitted"),
            SOLID_UNORM8_FRAG_SPV
        );
        assert_eq!(
            solid_fragment_spirv(AttachmentFormat::R32Float).expect("admitted"),
            SOLID_R32F_FRAG_SPV
        );
        // ... and the float stage is a different module: a one-component
        // attachment cannot take the 8-bit module's `vec4` store.
        assert_ne!(SOLID_UNORM8_FRAG_SPV, SOLID_R32F_FRAG_SPV);
        // `R32Uint` is the contract format outside the increment, refused with the
        // slug the contract and the format rail already use for it.
        let refused =
            solid_fragment_spirv(AttachmentFormat::R32Uint).expect_err("R32Uint has no stage");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("format_code"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// The registration gate: a colour format cannot be paired with another
    /// format's fragment stage, and the pairing that used to be the I2 path (the
    /// 8-bit module's `vec4` store on an `R32Float` attachment) is refused
    /// instead of executed.
    #[test]
    fn a_registration_refuses_a_format_with_another_formats_fragment_stage() {
        for format in AttachmentFormat::ADMITTED {
            reviewed_stages(format)
                .validate()
                .unwrap_or_else(|error| panic!("{format:?} is a reviewed pairing: {error:?}"));
        }

        // The pre-fix pairing: a `vec4` store on the one-component float
        // attachment.
        let mut stages = reviewed_stages(AttachmentFormat::R32Float);
        stages.fragment_spirv = SOLID_UNORM8_FRAG_SPV.to_vec();
        let refused = stages
            .validate()
            .expect_err("a vec4 store cannot describe an R32Float attachment");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("format_code"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::R32Float.code()
            )))
        );

        // The same module under the other UNORM layout stays accepted: one module,
        // two layouts. The gate refuses formats' shapes, not the fixture's hash.
        let mut stages = reviewed_stages(AttachmentFormat::Bgra8Unorm);
        stages.fragment_spirv = SOLID_UNORM8_FRAG_SPV.to_vec();
        stages
            .validate()
            .expect("the 8-bit module is reviewed for both UNORM layouts");

        // An entry name that is not the module's own entry cannot be bound, so it
        // is refused with the same slug.
        let mut stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        stages.contract.fragment_entry = "frag_other".to_owned();
        let refused = stages
            .validate()
            .expect_err("the reviewed module declares fragment_main");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");
    }

    #[test]
    fn the_contract_format_rail_refuses_r32uint_before_any_device_call() {
        let refused = attachment_vk_format(AttachmentFormat::R32Uint)
            .err()
            .expect("R32Uint is outside the first render increment");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("format_code"),
            Some(&FieldValue::Unsigned(0))
        );
        // `vk::Format` has no `Debug` in ash, so compare raw codes.
        assert_eq!(
            attachment_vk_format(AttachmentFormat::Rgba8Unorm).map(vk::Format::as_raw),
            Ok(vk::Format::R8G8B8A8_UNORM.as_raw())
        );
        assert_eq!(
            attachment_vk_format(AttachmentFormat::Bgra8Unorm).map(vk::Format::as_raw),
            Ok(vk::Format::B8G8R8A8_UNORM.as_raw())
        );
        assert_eq!(
            attachment_vk_format(AttachmentFormat::R32Float).map(vk::Format::as_raw),
            Ok(vk::Format::R32_SFLOAT.as_raw())
        );
    }

    #[test]
    fn a_format_without_color_attachment_is_refused_structurally() {
        let Some(context) = device_context() else {
            return;
        };
        // Depth/stencil formats are the portable example of a format the
        // COLOR_ATTACHMENT bit does not cover. Pick whichever candidate this
        // device refuses under both tilings instead of assuming one by name.
        let candidates = [
            vk::Format::D16_UNORM,
            vk::Format::D32_SFLOAT,
            vk::Format::X8_D24_UNORM_PACK32,
            vk::Format::S8_UINT,
        ];
        let Some(format) = candidates.into_iter().find(|format| {
            !format_supports_color_attachment(&context, *format, vk::ImageTiling::OPTIMAL)
                && !format_supports_color_attachment(&context, *format, vk::ImageTiling::LINEAR)
        }) else {
            eprintln!("SKIP: this device reports COLOR_ATTACHMENT for every depth candidate");
            return;
        };
        let refused = admit_color_attachment(&context, format, vk::ImageTiling::OPTIMAL)
            .expect_err("a format without COLOR_ATTACHMENT is refused");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("vk_format"),
            Some(&FieldValue::Unsigned(format.as_raw() as u64))
        );
        assert_eq!(
            refused.fields.get("tiling"),
            Some(&FieldValue::Text("optimal".to_owned()))
        );
        // The supported side of the same query is the milestone's format.
        assert!(format_supports_color_attachment(
            &context,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL
        ));
    }

    #[test]
    fn offscreen_render_refuses_r32uint_without_touching_the_device() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            format: AttachmentFormat::R32Uint,
            extent: [2, 2],
            clear: ClearColor::new([CLEAR_SENTINEL; 4]),
            vertex: milestone_vertex(),
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("R32Uint is refused before any Vulkan object exists");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        // No copy in either direction: the refusal precedes vkCreateImage.
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// Partial coverage: one texel keeps the fragment output, the other three
    /// keep the `LoadOp::Clear` bytes. A full-coverage triangle overwrites every
    /// texel, which is why the clear conversion could be wrong without any
    /// assertion noticing; this test makes it observable for every admitted
    /// format (the single-point vertex covers only pixel (0,0) of the 2×2
    /// viewport).
    #[test]
    fn a_partial_attachment_shows_the_clear_bytes_in_the_format_memory_order() {
        let Some(context) = device_context() else {
            return;
        };
        let clear = ClearColor::new(DISTINCT_CLEAR);
        for (format, clear_texel, stored_texel) in [
            (
                AttachmentFormat::Rgba8Unorm,
                DISTINCT_CLEAR,
                EXPECTED_RGBA8_TEXELS,
            ),
            (
                AttachmentFormat::Bgra8Unorm,
                // The contract's bytes are memory order, so an uncovered texel
                // reads back exactly those bytes in every format; only the
                // *component* mapping inside the clear differs (`Bgra8Unorm`
                // swaps red and blue, `R32Float` is one component). The
                // unit test above states that mapping.
                DISTINCT_CLEAR,
                EXPECTED_BGRA8_TEXELS,
            ),
            (
                AttachmentFormat::R32Float,
                DISTINCT_CLEAR,
                EXPECTED_R32F_TEXEL,
            ),
        ] {
            let texels = execute_offscreen_render(
                &context,
                &OffscreenRenderRequest {
                    format,
                    extent: [2, 2],
                    clear,
                    vertex: single_pixel_vertex(),
                },
            )
            .unwrap_or_else(|error| panic!("the partial {format:?} pass executes: {error:?}"));
            eprintln!(
                "{format:?} partial readback: {} (clear {clear_texel:02x?}, stored {stored_texel:02x?})",
                hex(&texels)
            );
            assert_eq!(texels.len(), 16);
            assert_eq!(
                texels[..4],
                stored_texel,
                "{format:?}: the covered texel holds the fragment output"
            );
            for (index, texel) in texels[4..].chunks(4).enumerate() {
                assert_eq!(
                    texel,
                    clear_texel,
                    "{format:?}: uncovered texel {} holds the clear bytes in memory order",
                    index + 1
                );
            }
        }
    }

    /// The 8-bit R,G,B,A layout of the milestone's colour.
    #[test]
    fn offscreen_rgba8_attachment_reads_back_the_stored_colour_bytes() {
        let Some(context) = device_context() else {
            return;
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let texels = offscreen_readback(&context, AttachmentFormat::Rgba8Unorm);
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(texels.len(), 16);
        assert_eq!(texels, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert!(
            !texels.contains(&CLEAR_SENTINEL),
            "a surviving clear sentinel means the triangle did not cover every texel: {}",
            hex(&texels)
        );
        // `LoadOp::Clear` needs no staging upload, and the attachment leaves
        // through exactly one image→buffer copy (`research/docs/23` §5.3).
        assert_eq!(uploads_after, uploads_before);
        assert_eq!(readbacks_after, readbacks_before + 1);
    }

    /// The same colour through the same module, read back in the B,G,R,A byte
    /// order the image format asks for.
    #[test]
    fn offscreen_bgra8_attachment_reads_back_the_same_colour_in_bgra_order() {
        let Some(context) = device_context() else {
            return;
        };
        let texels = offscreen_readback(&context, AttachmentFormat::Bgra8Unorm);

        assert_eq!(texels.len(), 16);
        assert_eq!(texels, EXPECTED_BGRA8_TEXELS.repeat(4));
        // The channel order is asserted, not just the byte count: the stage's
        // first component is the stored red `0x40`, and a B,G,R,A texel puts it
        // third, behind the stored blue `0xc0` and green `0x80`.
        assert_eq!(texels[0], 0xc0, "byte 0 is the stored blue (192/255)");
        assert_eq!(texels[1], 0x80, "byte 1 is the stored green (128/255)");
        assert_eq!(texels[2], 0x40, "byte 2 is the stored red (64/255)");
        assert_eq!(texels[3], 0xff, "byte 3 is the stored alpha");
        // Same colour, different layout: the R,G,B,A expectation is this
        // readback with the red and blue bytes exchanged.
        assert_eq!(
            EXPECTED_RGBA8_TEXELS,
            [texels[2], texels[1], texels[0], texels[3]]
        );
        assert_ne!(
            texels,
            EXPECTED_RGBA8_TEXELS.repeat(4),
            "a B,G,R,A attachment cannot read back the R,G,B,A bytes"
        );
        assert!(
            !texels.contains(&CLEAR_SENTINEL),
            "a surviving clear sentinel means the triangle did not cover every texel: {}",
            hex(&texels)
        );
    }

    /// The single-channel float attachment: one `float` per texel and no UNORM
    /// quantisation, so the four bytes are the stored float's own bytes.
    #[test]
    fn offscreen_r32float_attachment_reads_back_one_float_per_texel() {
        let Some(context) = device_context() else {
            return;
        };
        let texels = offscreen_readback(&context, AttachmentFormat::R32Float);

        assert_eq!(texels.len(), 16);
        assert_eq!(texels, EXPECTED_R32F_TEXEL.repeat(4));
        let stored = f32::from_le_bytes(texels[..4].try_into().expect("four bytes per texel"));
        // The readback is the float the stage stored, bit for bit, and that float
        // is the same `64/255` the 8-bit module's red channel carries.
        assert_eq!(stored.to_bits(), 0x3e80_8081);
        assert_eq!(stored, 64.0_f32 / 255.0);
        assert_eq!(
            (64.0_f32 / 255.0).to_le_bytes(),
            EXPECTED_R32F_TEXEL,
            "the frozen expectation is the f32 form of the 8-bit path's red channel"
        );
        // A float attachment's clear is `byte/255` as a float, exactly as
        // `OffscreenObjects::record` builds it, so the sentinel is those four
        // bytes rather than the 8-bit sentinel itself.
        let clear_bytes = (f32::from(CLEAR_SENTINEL) / 255.0).to_le_bytes();
        assert!(
            !texels
                .chunks_exact(4)
                .any(|texel| texel == clear_bytes.as_slice()),
            "a surviving clear sentinel means the triangle did not cover every texel: {}",
            hex(&texels)
        );
    }

    #[test]
    fn offscreen_render_refuses_a_zero_extent() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            format: AttachmentFormat::Rgba8Unorm,
            extent: [2, 0],
            clear: ClearColor::new([CLEAR_SENTINEL; 4]),
            vertex: milestone_vertex(),
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("a zero-dimension attachment is a contract refusal");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);
    }

    /// A registration whose contract is well formed but whose modules are not
    /// is refused once, at registration, instead of on every submission.
    #[test]
    fn render_stage_registration_refuses_an_empty_module_and_a_nul_entry() {
        let stages = |vertex_entry: &str, vertex_spirv: Vec<u8>| RenderStages {
            contract: RenderPipelineContract {
                vertex_entry: vertex_entry.to_owned(),
                fragment_entry: SOLID_FRAGMENT_ENTRY.to_owned(),
                color_format: AttachmentFormat::Rgba8Unorm,
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv,
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
        };
        assert!(
            stages("vertex_main", FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec())
                .validate()
                .is_ok()
        );

        let empty = stages("vertex_main", Vec::new())
            .validate()
            .expect_err("an empty module cannot be a SPIR-V entry point");
        eprintln!("refused: {empty:?}");
        assert_eq!(empty.slug, "spirv_module_invalid");
        assert_eq!(empty.class, ProviderErrorClass::Capability);
        assert_eq!(
            empty.fields.get("stage"),
            Some(&FieldValue::Text("vertex".to_owned()))
        );

        // A module that is not a whole number of SPIR-V words is refused by the
        // same predicate, so an odd tail cannot reach `vkCreateShaderModule`.
        let truncated = stages("vertex_main", vec![0; 6])
            .validate()
            .expect_err("a truncated module is refused");
        assert_eq!(truncated.slug, "spirv_module_invalid");

        let nul_entry = stages("ver\0tex", FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec())
            .validate()
            .expect_err("an entry name with an interior NUL cannot be bound");
        eprintln!("refused: {nul_entry:?}");
        assert_eq!(nul_entry.slug, "render_stage_entry_invalid");
        assert_eq!(
            nul_entry.fields.get("stage"),
            Some(&FieldValue::Text("vertex".to_owned()))
        );
    }

    #[test]
    fn execute_render_pass_refuses_a_mismatched_fragment_stage_before_the_device() {
        let Some(context) = device_context() else {
            return;
        };
        // A `RenderStages` built by hand, i.e. one that never passed the
        // registration gate: the execution side re-asks the same question, so the
        // pairing is still refused rather than executed.
        let mut stages = reviewed_stages(AttachmentFormat::R32Float);
        stages.fragment_spirv = SOLID_UNORM8_FRAG_SPV.to_vec();
        let pass = milestone_pass(AttachmentFormat::R32Float);
        pass.validate().expect("the fixture pass is a legal shape");
        let refused = execute_render_pass(&context, &stages, &pass)
            .expect_err("the mismatched pairing is refused before any Vulkan object exists");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        // No upload and no readback: the refusal precedes vkCreateImage.
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// `LoadOp::Load` has no rail that carries an attachment's previous bytes
    /// into the image, so the first increment refuses it instead of storing a
    /// clear under a name the trace did not ask for.
    #[test]
    fn execute_render_pass_refuses_load_before_touching_the_device() {
        let Some(context) = device_context() else {
            return;
        };
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.validate().expect("the fixture pass is a legal shape");
        pass.color_attachments[0].load = LoadOp::Load;
        let refused = execute_render_pass(&context, &stages, &pass)
            .expect_err("`Load` needs an upload rail this increment does not have");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_load_op_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// The render rail enqueues through the same driver boundary as the
    /// compute rail, so a loss it observes is a terminal device event too.
    ///
    /// Before this rail reported losses, a `VK_ERROR_DEVICE_LOST` here became
    /// an ordinary execution refusal and the instance kept admitting work.
    #[test]
    fn execute_render_pass_reports_a_driver_loss_as_a_terminal_device_event() {
        let Some(context) = device_context() else {
            return;
        };
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.validate().expect("the fixture pass is a legal shape");
        context.arm_driver_loss_injection(crate::DeviceLossPoint::Submit);
        let error = execute_render_pass(&context, &stages, &pass)
            .expect_err("the substituted driver answer refuses the render submission");
        eprintln!("render device loss: {error:?}");
        assert_eq!(error.class, ProviderErrorClass::DeviceLost);
        assert_eq!(error.slug, "render_submission_failed");
        assert_eq!(error.phase, ProviderPhase::Submit);
        assert_eq!(error.retryability, Retryability::RetryAfterRecreate);
        assert_eq!(
            error.completion,
            metal_api_core::provider::CompletionDisposition::DeviceLost { token: None }
        );
        assert_eq!(
            error.fields.get("vk_result"),
            Some(&FieldValue::Text("VK_ERROR_DEVICE_LOST".to_owned()))
        );
        assert_eq!(
            error.fields.get("vk_result_raw"),
            Some(&FieldValue::Signed(i64::from(
                vk::Result::ERROR_DEVICE_LOST.as_raw()
            )))
        );
        let fault = context
            .last_device_fault()
            .expect("the rail recorded a fault snapshot");
        assert_eq!(
            error.fields.get("device_fault_extension"),
            Some(&FieldValue::Bool(fault.extension_present))
        );

        // The loss is the core lifecycle's, so the instance stops admitting
        // work through the documented refusal and repeats it unchanged.
        assert_eq!(
            context.health(),
            metal_api_core::provider::ProviderHealth::DeviceLost
        );
        let first = context.admit().expect_err("a lost device admits no work");
        let second = context.admit().expect_err("a lost device admits no work");
        assert_eq!(first, second);
        assert_eq!(first.slug, "device_lost");
        assert_eq!(
            context.queue_submission_counts().iter().sum::<usize>(),
            0,
            "the substituted loss never reached the driver"
        );
    }
}
