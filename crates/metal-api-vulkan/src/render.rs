//! Offscreen render execution rail (`research/docs/23` §6 Step 3b).
//!
//! One or two colour attachments, one full-screen triangle, one `vkCmdDraw`,
//! then one `vkCmdCopyImageToBuffer` per *stored* attachment back into
//! host-visible memory. The rail answers the question this step owns — can the
//! provider build a render pass, a framebuffer and a graphics pipeline out of
//! two SPIR-V modules and read every stored attachment back byte-for-byte — and
//! it fixes the two rules the driver probe left behind (`/var/tmp/render-probe`,
//! `research/docs/23` §3.5, §9):
//!
//! * the attachment is `VK_IMAGE_TILING_OPTIMAL` plus one
//!   `vkCmdCopyImageToBuffer` per stored attachment (`copy_out` = declaring
//!   write allocations ∪ stored attachment allocations; a `StoreOp::DontCare`
//!   attachment gets `VK_ATTACHMENT_STORE_OP_DONT_CARE` and no readback at
//!   all), because the RTX 5060 native driver and the dzn/D3D12 backend both
//!   refuse a linear colour attachment;
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
//! through the existing buffer-writeback channel, one writeback per
//! attachment.

use ash::vk;
use metal_api_core::provider::{
    AttachmentFormat, BufferSource, BufferView, ClearColor, FieldValue, IndexFormat,
    IndirectCommandDescriptor, LoadOp, ProviderError, ProviderErrorClass, ProviderPhase,
    RenderPassDescriptor, RenderPipelineContract, Retryability, StoreOp, VertexBufferLayout,
    VertexFormat,
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
// colour-attachment format, and one reviewed dual-output module for the
// `[Rgba8Unorm, Rgba8Unorm]` MRT shape. The vertex stage stays a host
// registration's value, while the fragment stage is *not*: it is a function of
// the attachment format list (`solid_fragment_spirv`), because a fragment stage
// built for one format does not describe another one. Pairing the fixed `vec4`
// store with every format is exactly the "admitted, then read back the wrong
// bytes" path the 2026-09-14 review filed as I2.

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

/// The reviewed solid fragment module for the two-output
/// `[Rgba8Unorm, Rgba8Unorm]` shape.
///
/// The rail's only reviewed MRT fixture (`research/docs/23` v18 Step 3): the
/// module writes `(64/255, 128/255, 192/255, 1)` to `Location 0` and
/// `(1, 128/255, 64/255, 192/255)` to `Location 1`, so two 2×2
/// `R8G8B8A8_UNORM` attachments read back `40 80 c0 ff` and `ff 80 40 c0`
/// respectively. The constants reuse the single-output module's byte/255
/// discipline, so neither output sits on a half-integer UNORM tie.
const SOLID_UNORM8_DUAL_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_unorm8_dual.frag.spv");

/// The reviewed four-output fragment module for the four-attachment ceiling
/// (`research/docs/23` §3.3, v24).
///
/// One module serves the `MAX_COLOR_ATTACHMENTS`-many `[Rgba8Unorm; 4]` list:
/// its four locations write pairwise-distinct byte strings, so a capture that
/// landed one target twice cannot pass the comparison.
const SOLID_UNORM8_QUAD_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_unorm8_quad.frag.spv");

/// The solid fragment module the offscreen rail builds for a format list.
///
/// The match is exhaustive over [`AttachmentFormat`] and has no default arm: a
/// contract format that gains no arm here is a compile error, which is what
/// makes "admitted but executed with another format's fragment stage"
/// unrepresentable instead of merely tested. A one-format list keeps the
/// pre-MRT per-format module (byte zero drift); the two-output list is the
/// reviewed dual module and every other dual combination is refused with
/// `render_mrt_format_combination_unsupported` before any Vulkan object exists.
/// `R32Uint` is refused with the slug the contract and the format rail already
/// use for it, so an integer attachment cannot reach a colour store. An empty
/// or over-two list is refused as an attachment-count capability fact, so the
/// map is total over every list shape the frozen core contract can carry.
pub(crate) fn solid_fragment_spirv(
    formats: &[AttachmentFormat],
) -> Result<&'static [u8], ProviderError> {
    Ok(match formats {
        [format] => match format {
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
        },
        [AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm] => SOLID_UNORM8_DUAL_FRAG_SPV,
        [
            AttachmentFormat::Rgba8Unorm,
            AttachmentFormat::Rgba8Unorm,
            AttachmentFormat::Rgba8Unorm,
            AttachmentFormat::Rgba8Unorm,
        ] => SOLID_UNORM8_QUAD_FRAG_SPV,
        [first, second] if formats.len() == 2 => {
            return Err(mrt_format_combination_refusal(*first, *second));
        }
        [first, ..] => return Err(mrt_format_combination_refusal(*first, formats[1])),
        _ => return Err(mrt_attachment_count_refusal(formats.len())),
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
    /// Colour attachments, in location order: entry `i` is the target the
    /// fragment stage's output `i` lands in. One or two entries; the rail
    /// refuses every other count before any Vulkan object exists.
    pub attachments: Vec<OffscreenColorAttachment<'a>>,
    /// Attachment extent in texels, shared by every entry of
    /// [`Self::attachments`] (`prepare_render_request` refuses a pass whose
    /// attachments disagree). The milestone fixes 2×2 (`docs/23` §1.3) so full
    /// coverage is distinguishable from a single stored texel.
    pub extent: [u32; 2],
    /// The vertex stage the graphics pipeline is built from.
    pub vertex: OffscreenVertexStage<'a>,
    /// The caller-held vertex streams the pass binds, in binding order
    /// (`research/docs/23` §3.3). Empty for the `vertex_id` milestone.
    pub vertex_streams: Vec<VertexStream<'a>>,
    /// How the draw issues when [`Self::indirect`] is `None`.
    pub draw: DrawShape,
    /// The caller-held index buffer, when the draw is indexed.
    pub index_stream: Option<IndexStream<'a>>,
    /// When set, the full-screen triangle is replayed from one CPU-encoded
    /// indirect command instead of being issued with `vkCmdDraw`
    /// (`research/docs/25` §6 Step 4). `Draw` carries the command's vertex and
    /// instance counts and fixes `firstVertex`/`firstInstance` to zero;
    /// `DrawIndexed` carries its index and instance counts and replays through
    /// the rail's own `[0, 1, 2]` index buffer.
    pub indirect: Option<IndirectReplay>,
}

/// One colour attachment of an offscreen render request.
///
/// Each entry carries its own format, load operation and previous bytes,
/// exactly like the pass's per-location attachment list: the format list
/// selects the fragment module ([`solid_fragment_spirv`]), the `Clear` colour
/// is mapped onto the format's component order by [`clear_value_for`], and
/// `previous` carries the bytes a `LoadOp::Load` entry uploads before the pass
/// opens.
pub(crate) struct OffscreenColorAttachment<'a> {
    /// Colour attachment format, in render-contract terms.
    pub format: AttachmentFormat,
    /// The attachment's store operation (`docs/23` §3.6, v19). `Store` reads
    /// the attachment back into the writeback channel; `DontCare` marks the
    /// attachment's store with `VK_ATTACHMENT_STORE_OP_DONT_CARE` and gives it
    /// no readback, so the discarded attachment disappears from the observable
    /// surface instead of passing as "landed correctly".
    pub store: StoreOp,
    /// How the pass establishes this attachment's contents. `Clear(color)` is
    /// carried as bytes for the same reason the contract carries bytes: a
    /// float clear is not parity-stable (`research/docs/23` §3.5), and the
    /// bytes are in the format's *memory* order while [`clear_value_for`]
    /// maps them onto Vulkan's component order. `Load` uploads `previous`
    /// before the pass opens, and `DontCare` discards the pre-pass contents
    /// without reading or uploading them (`docs/23` §3.1, v20).
    pub load: LoadOp,
    /// The attachment's previous bytes for a `LoadOp::Load` pass
    /// (`research/docs/23` §3.3). `Some` means the rail uploads them into the
    /// image and opens the render pass with `LOAD_OP_LOAD`; `None` is the
    /// `Clear`/`DontCare` shape.
    pub previous: Option<&'a [u8]>,
}

/// One caller-held vertex stream: the layout the pipeline is built from plus
/// the pool view whose bytes the draw reads.
///
/// The view is the trace's own declaration, so the bytes, their range and the
/// footprint proof come from one place. `offset` inside the view is what the
/// bind call uses; the first increment uploads the view's bytes into their own
/// device buffer, so the offset is zero by construction.
pub(crate) struct VertexStream<'a> {
    pub layout: &'a VertexBufferLayout,
    pub view: &'a BufferView,
}

/// One caller-held index buffer: its width and the pool view holding it.
pub(crate) struct IndexStream<'a> {
    pub format: IndexFormat,
    pub view: &'a BufferView,
}

/// How an offscreen pass issues its direct draw.
///
/// The three shapes are the reviewed ones: the `vertex_id` milestone triangle,
/// a non-indexed draw over caller-held streams, and an indexed draw over them.
/// An indirect replay replaces all three (`Self::indirect`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrawShape {
    /// The milestone's full-screen triangle, positions from `vertex_id`.
    Milestone,
    /// A non-indexed draw over the bound vertex streams.
    Vertices { vertex_count: u32 },
    /// An indexed draw through the bound index buffer.
    Indexed { index_count: u32 },
}

/// The indirect command one offscreen pass replays (`research/docs/25` §6
/// Step 4). The first increment supports exactly the reviewed draw shapes: a
/// non-indexed draw over the milestone's three vertices, and an indexed draw
/// whose `[0, 1, 2]` index buffer selects the same three `gl_VertexIndex`
/// values the vertex stage already maps to the full-screen triangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndirectReplay {
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
    },
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
        // The fragment stage has to be the reviewed module for the format list
        // this contract declares. The rail cannot read a module's semantics, so
        // binding the registration to the reviewed set is what refuses "this
        // format list, that format list's fragment stage" *before* a submission
        // can read back bytes the format claim does not cover (review item I2,
        // 2026-09-14): an `R32Float` pipeline handed the 8-bit module's `vec4`
        // store is a component-shape mismatch, not a byte-order preference, and
        // a dual-attachment pipeline handed the single-output module would
        // never store `Location 1`.
        if !fragment_stage_is_reviewed(self) {
            return Err(fragment_stage_mismatch_refusal(
                &self.contract.color_formats,
                &self.contract.fragment_entry,
            ));
        }
        Ok(())
    }
}

/// Whether a registration's fragment stage is exactly the module this rail
/// builds for the contract's colour format list, under the entry that module
/// declares.
///
/// Both ends of the rail ask this question — registration refuses a pairing
/// once, and execution re-asks it of the value it was handed, so a
/// directly-constructed [`RenderStages`] cannot skip the registration gate.
fn fragment_stage_is_reviewed(stages: &RenderStages) -> bool {
    stages.contract.fragment_entry == SOLID_FRAGMENT_ENTRY
        && solid_fragment_spirv(&stages.contract.color_formats)
            .is_ok_and(|module| module == stages.fragment_spirv.as_slice())
}

/// The refusal for a fragment stage that is not the reviewed module of the
/// pipeline's colour format list.
///
/// A capability fact, like the other stage-module refusals: the rail has one
/// reviewed fragment stage per admitted format list and no second translation
/// path, so it refuses the pairing instead of executing a module whose
/// semantics it cannot check.
fn fragment_stage_mismatch_refusal(formats: &[AttachmentFormat], entry: &str) -> ProviderError {
    let mut refusal = capability_refusal("render_fragment_stage_mismatch")
        .with_field("format_count", FieldValue::Unsigned(formats.len() as u64))
        .with_field("fragment_entry", FieldValue::Text(entry.to_owned()))
        .with_field(
            "reviewed_entry",
            FieldValue::Text(SOLID_FRAGMENT_ENTRY.to_owned()),
        )
        .with_detail(
            "the fragment stage is not the module this rail builds for the colour format list, \
             so running it would land bytes the format claim does not cover",
        );
    if let [format] = formats {
        refusal = refusal.with_field(
            "format_code",
            FieldValue::Unsigned(u64::from(format.code())),
        );
    } else if let [first, second] = formats {
        refusal = refusal
            .with_field(
                "format_code_0",
                FieldValue::Unsigned(u64::from(first.code())),
            )
            .with_field(
                "format_code_1",
                FieldValue::Unsigned(u64::from(second.code())),
            );
    }
    refusal
}

/// Execute one admitted render pass and return, in location order, each stored
/// attachment's tightly packed texel bytes and `None` for each discarded one.
///
/// This is the trace-side entry point of the rail: the pass's shape rules were
/// already checked by core admission, so what is left here is the agreement
/// between the pass and the registered pipeline it names
/// ([`RenderPipelineContract::validate_against`]) and the two shapes the first
/// increment cannot execute — a `Load` that would have to carry the
/// attachment's previous bytes into the image, and a pass whose attachment
/// list or format combination is outside the reviewed set. All of them are
/// refused as capability facts before any Vulkan object exists, never
/// downgraded to a clear. The registered fragment stage is re-checked against
/// the pipeline's declared format list in the same place, for the same reason:
/// the pass is about to be executed with it.
pub(crate) fn execute_render_pass(
    context: &VulkanContext,
    stages: &RenderStages,
    pass: &RenderPassDescriptor,
    previous: &[Option<&[u8]>],
) -> Result<Vec<Option<Vec<u8>>>, ProviderError> {
    let request = prepare_render_request(stages, pass, previous)?;
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
    pass: &'a RenderPassDescriptor,
    previous: &'a [Option<&'a [u8]>],
) -> Result<OffscreenRenderRequest<'a>, ProviderError> {
    stages
        .contract
        .validate_against(pass)
        .map_err(|error| contract_refusal(&error.to_string()))?;
    // The MRT increment executes one or two attachments; a three- or
    // four-attachment pass is admitted by the frozen core contract but refused
    // here, fail-closed, instead of silently rendering the first two locations.
    // This replaces the pre-MRT "exactly one" gate rather than layering a
    // second check on top of it.
    if pass.color_attachments.len() > metal_api_core::provider::MAX_COLOR_ATTACHMENTS {
        return Err(mrt_attachment_count_refusal(pass.color_attachments.len()));
    }
    if previous.len() != pass.color_attachments.len() {
        return Err(contract_refusal(
            "the previous-byte list must carry one entry per colour attachment",
        ));
    }
    if !fragment_stage_is_reviewed(stages) {
        return Err(fragment_stage_mismatch_refusal(
            &stages.contract.color_formats,
            &stages.contract.fragment_entry,
        ));
    }
    let mut attachments = Vec::with_capacity(pass.color_attachments.len());
    let mut extent: Option<[u32; 2]> = None;
    for (index, (attachment, previous)) in pass.color_attachments.iter().zip(previous).enumerate() {
        match attachment.load {
            LoadOp::Clear(_) => {}
            LoadOp::Load => {
                // The rail uploads the attachment's previous bytes before
                // opening the render pass (`research/docs/23` §3.3). The caller
                // resolves them from the trace's own declaration, so a `Load`
                // that carries no bytes is refused rather than silently
                // executed as a clear.
                if previous.is_none() {
                    return Err(capability_refusal("attachment_load_op_unsupported")
                        .with_field("attachment", FieldValue::Unsigned(index as u64))
                        .with_field("load_op", FieldValue::Text("load".to_owned()))
                        .with_detail(
                            "a `LoadOp::Load` pass needs the attachment's previous bytes from \
                             the trace's own view declaration; this pass resolved none",
                        ));
                }
            }
            LoadOp::DontCare => {
                // The attachment's pre-pass contents are undefined, so the
                // rail neither reads nor uploads declaring bytes. A caller
                // that resolves bytes anyway is refused rather than silently
                // ignored (`docs/23` §3.1, v20).
                if previous.is_some() {
                    return Err(capability_refusal("attachment_load_op_unsupported")
                        .with_field("attachment", FieldValue::Unsigned(index as u64))
                        .with_field("load_op", FieldValue::Text("dont_care".to_owned()))
                        .with_detail(
                            "a `LoadOp::DontCare` attachment declares no previous bytes; the \
                             trace must resolve none",
                        ));
                }
            }
        }
        let width = narrow_dimension(attachment.width)?;
        let height = narrow_dimension(attachment.height)?;
        match extent {
            None => extent = Some([width, height]),
            Some([expected_width, expected_height])
                if [width, height] != [expected_width, expected_height] =>
            {
                return Err(capability_refusal("render_attachment_extent_mismatch")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("width", FieldValue::Unsigned(u64::from(width)))
                    .with_field("height", FieldValue::Unsigned(u64::from(height)))
                    .with_field(
                        "expected_width",
                        FieldValue::Unsigned(u64::from(expected_width)),
                    )
                    .with_field(
                        "expected_height",
                        FieldValue::Unsigned(u64::from(expected_height)),
                    )
                    .with_detail("every colour attachment of one pass shares one extent"));
            }
            Some(_) => {}
        }
        attachments.push(OffscreenColorAttachment {
            format: attachment.format,
            store: attachment.store,
            load: attachment.load,
            previous: *previous,
        });
    }
    let extent = extent.expect("core admission refuses an empty attachment list");
    // Vertex input (`research/docs/23` §3.3): every bound stream declares its
    // own bytes, so the rail proves the footprint the draw reads and refuses
    // anything the reviewed shape does not cover.
    let streams = resolve_vertex_streams(stages, pass)?;
    let (draw, index_stream) = match &pass.indices {
        Some(indices) => {
            let view = &indices.view;
            let required = u64::from(pass.vertices)
                .checked_mul(indices.format.bytes())
                .ok_or_else(|| contract_refusal("index buffer footprint overflows u64"))?;
            if view.length < required {
                return Err(
                    capability_refusal("render_index_buffer_footprint_unsupported")
                        .with_field(
                            "index_count",
                            FieldValue::Unsigned(u64::from(pass.vertices)),
                        )
                        .with_field("required_bytes", FieldValue::Unsigned(required))
                        .with_field("declared_bytes", FieldValue::Unsigned(view.length))
                        .with_detail(
                            "the draw reads more index bytes than the view the trace declares",
                        ),
                );
            }
            let index_values = decode_indices(view, indices.format, pass.vertices)?;
            for stream in &streams {
                let vertex_capacity = stream.view.length / stream.layout.stride;
                if let Some(index) = index_values
                    .iter()
                    .find(|index| u64::from(**index) >= vertex_capacity)
                {
                    return Err(
                        capability_refusal("render_vertex_buffer_footprint_unsupported")
                            .with_field("index", FieldValue::Unsigned(u64::from(*index)))
                            .with_field("vertex_capacity", FieldValue::Unsigned(vertex_capacity))
                            .with_field("stride", FieldValue::Unsigned(stream.layout.stride))
                            .with_field("declared_bytes", FieldValue::Unsigned(stream.view.length))
                            .with_detail(
                                "the index buffer names a vertex the bound stream does not cover",
                            ),
                    );
                }
            }
            (
                DrawShape::Indexed {
                    index_count: pass.vertices,
                },
                Some(IndexStream {
                    format: indices.format,
                    view,
                }),
            )
        }
        None => {
            for stream in &streams {
                let required = u64::from(pass.vertices)
                    .checked_mul(stream.layout.stride)
                    .ok_or_else(|| contract_refusal("vertex buffer footprint overflows u64"))?;
                if stream.view.length < required {
                    return Err(
                        capability_refusal("render_vertex_buffer_footprint_unsupported")
                            .with_field(
                                "vertex_count",
                                FieldValue::Unsigned(u64::from(pass.vertices)),
                            )
                            .with_field("required_bytes", FieldValue::Unsigned(required))
                            .with_field("declared_bytes", FieldValue::Unsigned(stream.view.length))
                            .with_detail(
                                "the draw reads more vertex bytes than the view the trace declares",
                            ),
                    );
                }
            }
            if streams.is_empty() {
                (DrawShape::Milestone, None)
            } else {
                (
                    DrawShape::Vertices {
                        vertex_count: pass.vertices,
                    },
                    None,
                )
            }
        }
    };
    let request = OffscreenRenderRequest {
        attachments,
        extent,
        vertex: OffscreenVertexStage {
            entry: &stages.contract.vertex_entry,
            spirv: &stages.vertex_spirv,
        },
        vertex_streams: streams,
        draw,
        index_stream,
        indirect: None,
    };
    Ok(request)
}

/// Pair the pipeline's layout with the pass's bound streams.
///
/// Core admission already refused a pass whose binding count disagrees with the
/// layout, and this rail re-runs `validate_against` before reaching here, so the
/// zip is length-checked by construction. What is added is the rail's own
/// minimum: a stream has to hold at least one vertex, and the source has to be
/// trace-owned bytes, because the first vertex-input increment uploads the
/// trace's own bytes rather than a lease.
fn resolve_vertex_streams<'a>(
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
) -> Result<Vec<VertexStream<'a>>, ProviderError> {
    let mut streams = Vec::with_capacity(pass.vertex_buffers.len());
    for (index, (view, layout)) in pass
        .vertex_buffers
        .iter()
        .zip(stages.contract.vertex_layout.buffers())
        .enumerate()
    {
        if layout.stride == 0 {
            return Err(contract_refusal("vertex buffer declares a zero stride"));
        }
        if !matches!(view.source, BufferSource::OwnedBytes(_)) {
            return Err(capability_refusal("render_vertex_buffer_unsupported")
                .with_field("binding", FieldValue::Unsigned(index as u64))
                .with_detail("the first vertex-input increment executes trace-owned bytes only"));
        }
        if view.length < layout.stride {
            return Err(
                capability_refusal("render_vertex_buffer_footprint_unsupported")
                    .with_field("binding", FieldValue::Unsigned(index as u64))
                    .with_field("required_bytes", FieldValue::Unsigned(layout.stride))
                    .with_field("declared_bytes", FieldValue::Unsigned(view.length))
                    .with_detail("one vertex does not fit in the view the trace declares"),
            );
        }
        streams.push(VertexStream { layout, view });
    }
    Ok(streams)
}

/// The Vulkan index type for one contract index width.
fn indices_format(format: IndexFormat) -> vk::IndexType {
    match format {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

/// The `VkFormat` one contract vertex format names.
///
/// Like [`indices_format`], this is a closed translation with no default arm:
/// a contract format that gains no arm here is a compile error rather than a
/// silently misread stream.
fn vertex_vk_format(format: VertexFormat) -> Result<vk::Format, ProviderError> {
    Ok(match format {
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
    })
}

/// Read the `count` indices the draw consumes out of one pool view.
///
/// The bytes are the trace's own (`BufferSource::OwnedBytes`, whose length
/// `BufferView::validate_shape` already pinned to the view's length), so this
/// is a pure translation. The values feed the footprint proof: every index has
/// to name a vertex the bound stream covers.
fn decode_indices(
    view: &BufferView,
    format: IndexFormat,
    count: u32,
) -> Result<Vec<u32>, ProviderError> {
    let BufferSource::OwnedBytes(bytes) = &view.source else {
        return Err(capability_refusal("render_index_buffer_unsupported")
            .with_detail("the first vertex-input increment executes trace-owned bytes only"));
    };
    let width = usize::try_from(format.bytes()).expect("index widths are two or four");
    let count = usize::try_from(count).map_err(|_| contract_refusal("index count overflows"))?;
    let mut indices = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(width).take(count) {
        indices.push(match format {
            IndexFormat::Uint16 => u32::from(u16::from_ne_bytes([chunk[0], chunk[1]])),
            IndexFormat::Uint32 => u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        });
    }
    if indices.len() != count {
        return Err(
            capability_refusal("render_index_buffer_footprint_unsupported")
                .with_detail("the index view is shorter than the draw's index count"),
        );
    }
    Ok(indices)
}

/// The colour subresource range every attachment barrier names.
fn color_subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Execute one admitted render pass whose full-screen triangle is replayed from
/// one CPU-encoded `VkDrawIndirectCommand` or `VkDrawIndexedIndirectCommand`
/// (`research/docs/25` §6 Step 4).
///
/// The pass shape rules are the ones [`execute_render_pass`] already enforces —
/// this entry point only swaps the draw for an indirect replay, so a trace that
/// is not admitted as a render pass cannot reach it. Compute dispatches and the
/// un-reviewed indexed shapes are outside the first indirect increment and are
/// refused with the capability slug the contract publishes for them.
pub(crate) fn execute_indirect_render_pass(
    context: &VulkanContext,
    stages: &RenderStages,
    pass: &RenderPassDescriptor,
    command: &IndirectCommandDescriptor,
    previous: &[Option<&[u8]>],
) -> Result<Vec<Option<Vec<u8>>>, ProviderError> {
    let replay = match command {
        IndirectCommandDescriptor::Draw {
            vertex_count,
            instance_count,
        } => IndirectReplay::Draw {
            vertex_count: *vertex_count,
            instance_count: *instance_count,
        },
        IndirectCommandDescriptor::DrawIndexed {
            index_count,
            instance_count,
        } => {
            // The reviewed indexed shape is the same full-screen triangle the
            // non-indexed rail draws: three indices into a vertex stage that
            // selects its positions from `gl_VertexIndex`. Any other index
            // count would name vertices the reviewed fixture does not cover, so
            // it is refused rather than executed with an un-reviewed shape.
            if *index_count != 3 {
                return Err(capability_refusal("icb_command_unsupported")
                    .with_field("kind", FieldValue::Text("DrawIndexed".to_owned()))
                    .with_field("index_count", FieldValue::Unsigned(u64::from(*index_count)))
                    .with_detail(
                        "the first indexed indirect increment replays exactly the milestone's \
                         three-index full-screen triangle",
                    ));
            }
            if *instance_count == 0 {
                return Err(capability_refusal("icb_command_unsupported")
                    .with_field("kind", FieldValue::Text("DrawIndexed".to_owned()))
                    .with_field("instance_count", FieldValue::Unsigned(0))
                    .with_detail("an indexed indirect draw needs at least one instance"));
            }
            IndirectReplay::DrawIndexed {
                index_count: *index_count,
                instance_count: *instance_count,
            }
        }
        other => {
            return Err(capability_refusal("icb_command_unsupported")
                .with_field("kind", FieldValue::Text(format!("{:?}", other.kind())))
                .with_detail("the first indirect increment replays draws only"));
        }
    };
    let mut request = prepare_render_request(stages, pass, previous)?;
    request.indirect = Some(replay);
    execute_offscreen_render(context, &request)
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

/// Execute one offscreen render pass and return, in location order, `Some` of
/// each stored attachment's tightly packed texel bytes (`width * height * 4`)
/// and `None` for each discarded attachment.
///
/// Contract format admission, the fragment stage the format list selects
/// ([`solid_fragment_spirv`]), the device's `COLOR_ATTACHMENT` bit and the
/// `TRANSFER_SRC` bit a stored attachment's readback needs all run before the
/// first `vkCreateImage`, so an unsupported request is refused instead of being
/// handed to the driver. The attachment-count, format-combination and
/// at-least-one-store gates are re-run here for a directly-constructed request,
/// so the fail-closed shape does not depend on the caller having gone through
/// `prepare_render_request`.
pub(crate) fn execute_offscreen_render(
    context: &VulkanContext,
    request: &OffscreenRenderRequest<'_>,
) -> Result<Vec<Option<Vec<u8>>>, ProviderError> {
    // The attachment count is the rail's own gate, re-run on the request so a
    // hand-built request cannot skip `prepare_render_request`'s admission.
    if request.attachments.len() > metal_api_core::provider::MAX_COLOR_ATTACHMENTS {
        return Err(mrt_attachment_count_refusal(request.attachments.len()));
    }
    if request.attachments.is_empty() {
        return Err(contract_refusal(
            "render pass declares no colour attachment",
        ));
    }
    // `docs/23` §3.6, v19: core admission refuses an all-discarded pass as
    // `AllRenderAttachmentsDiscarded`, and the rail re-asserts the same
    // at-least-one-store rule for a directly-constructed request. Discarding
    // every attachment would turn "nothing landed" into a blank proof of
    // "landed correctly".
    if request
        .attachments
        .iter()
        .all(|attachment| attachment.store == StoreOp::DontCare)
    {
        return Err(render_all_attachments_discarded_refusal());
    }
    let formats = request
        .attachments
        .iter()
        .map(|attachment| attachment.format)
        .collect::<Vec<_>>();
    // The fragment stage is the format list's, not the caller's: `request`
    // carries no fragment module, so this is the only place one is named and
    // there is no pairing left to get wrong. The refusal covers the
    // dual-combination and count shapes before any device call.
    let fragment_spirv = solid_fragment_spirv(&formats)?;
    let tiling = vk::ImageTiling::OPTIMAL;
    let vk_formats = formats
        .iter()
        .map(|format| attachment_vk_format(*format))
        .collect::<Result<Vec<_>, _>>()?;
    for (vk_format, attachment) in vk_formats.iter().zip(&request.attachments) {
        admit_color_attachment(context, *vk_format, tiling)?;
        // Only a stored attachment is read back, so `TRANSFER_SRC` is asked of
        // stored attachments alone (`docs/23` §3.6, v19): a discarded
        // attachment is not copied out and must not be refused for a feature
        // its execution never needs.
        if attachment.store == StoreOp::Store
            && !format_features(context, *vk_format, tiling)
                .contains(vk::FormatFeatureFlags::TRANSFER_SRC)
        {
            return Err(attachment_format_refusal()
                .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
                .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                .with_field(
                    "missing_feature",
                    FieldValue::Text("transfer_src".to_owned()),
                )
                .with_detail(
                    "the milestone reads the attachment back through vkCmdCopyImageToBuffer",
                ));
        }
        // A loading attachment declares its load operation before the image
        // exists: the transfer-destination usage is only legal on the image
        // when the rail is actually going to upload into it
        // (`research/docs/23` §3.3).
        if attachment.previous.is_some()
            && !format_features(context, *vk_format, tiling)
                .contains(vk::FormatFeatureFlags::TRANSFER_DST)
        {
            return Err(attachment_format_refusal()
                .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
                .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                .with_field(
                    "missing_feature",
                    FieldValue::Text("transfer_dst".to_owned()),
                )
                .with_detail(
                    "a `LoadOp::Load` pass uploads the attachment's previous bytes with \
                     vkCmdCopyBufferToImage",
                ));
        }
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
    for (attachment, vk_format) in request.attachments.iter().zip(&vk_formats) {
        objects.create_attachment(
            *vk_format,
            width,
            height,
            attachment.load,
            attachment.store == StoreOp::Store,
        )?;
    }
    objects.create_render_pass(&vk_formats)?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
        &request.vertex_streams,
    )?;
    // One readback destination per stored attachment; a discarded attachment
    // creates none, because its bytes leave no observable surface to land in
    // (`docs/23` §3.6, v19).
    let mut readback_mappings = Vec::with_capacity(request.attachments.len());
    for attachment in &request.attachments {
        if attachment.store == StoreOp::Store {
            readback_mappings.push(objects.create_readback(byte_length)?);
        }
    }
    objects.create_vertex_inputs(&request.vertex_streams, request.index_stream.as_ref())?;
    objects.draw = request.draw;
    for (index, attachment) in request.attachments.iter().enumerate() {
        if let Some(previous) = attachment.previous {
            objects.create_previous_bytes(index, previous)?;
        }
    }
    match request.indirect {
        Some(IndirectReplay::Draw {
            vertex_count,
            instance_count,
        }) => {
            objects.create_indirect_draw(vertex_count, instance_count)?;
        }
        Some(IndirectReplay::DrawIndexed {
            index_count,
            instance_count,
        }) => {
            objects.create_indirect_draw_indexed(index_count, instance_count)?;
        }
        None => {}
    }
    objects.create_command_pool(queue_index)?;
    objects.record(&request.attachments, width, height)?;
    objects.submit_and_wait(queue_index)?;

    // One readback record per stored attachment: `copy_out` equals the stored
    // attachment count, so a caller can observe that a stored location really
    // left the device and that a discarded one produced no bytes at all.
    let mut results = Vec::with_capacity(request.attachments.len());
    let mut mappings = readback_mappings.into_iter();
    for attachment in &request.attachments {
        if attachment.store == StoreOp::Store {
            let mapping = mappings.next().expect("one readback per stored attachment");
            let texels = unsafe {
                std::slice::from_raw_parts(mapping as *const u8, byte_length as usize).to_vec()
            };
            context.record_buffer_readback();
            context.record_buffer_readback_bytes(texels.len());
            results.push(Some(texels));
        } else {
            results.push(None);
        }
    }
    Ok(results)
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
    previous: Option<&[u8]>,
) -> Result<Vec<u8>, ProviderError> {
    // The present path stays single-attachment: it renders into one
    // provider-owned target and hands that target on, so a pass whose
    // attachment list is not exactly one entry is outside this increment's
    // present shape.
    let previous = [previous];
    let request = prepare_render_request(stages, pass, &previous)?;
    let [attachment] = request.attachments.as_slice() else {
        return Err(contract_refusal(
            "the present rail executes exactly one colour attachment",
        ));
    };
    // A present attachment is the pass's only observable landing point, so a
    // `StoreOp::DontCare` present pass is the all-discarded shape the rail
    // refuses for an offscreen request (`docs/23` §3.6, v19). Core admission
    // already refused it as `AllRenderAttachmentsDiscarded`; this is the
    // value-level second line of defence.
    if attachment.store == StoreOp::DontCare {
        return Err(render_all_attachments_discarded_refusal());
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
    let fragment_spirv = solid_fragment_spirv(&[attachment.format])?;
    let vk_format = attachment_vk_format(attachment.format)?;
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
    objects.create_render_pass(&[vk_format])?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
        &request.vertex_streams,
    )?;
    let readback_mapping = objects.create_readback(byte_length)?;
    objects.create_vertex_inputs(&request.vertex_streams, request.index_stream.as_ref())?;
    objects.draw = request.draw;
    objects.create_command_pool(queue_index)?;
    objects.record(std::slice::from_ref(attachment), width, height)?;
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
    /// One entry per colour attachment, in location order.
    attachments: Vec<AttachmentObjects>,
    /// Whether this scope created every `attachments` image/memory/view and
    /// must destroy them on Drop. A present pass borrows the provider-owned
    /// [`PresentTargetImage`] instead, so its per-pass scope must not destroy
    /// the target when it finishes (`docs/24` §5.2: the target survives the
    /// submission).
    owns_attachments: bool,
    /// Whether this pass hands its attachment on as a present target. When set,
    /// the render pass ends in `COLOR_ATTACHMENT_OPTIMAL` and `record` inserts
    /// the explicit present layout transition before the copy-out
    /// (`docs/24` §3.3 rule 1).
    present: bool,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    /// One readback per attachment, in location order.
    readbacks: Vec<ReadbackObjects>,
    /// The host-visible `INDIRECT_BUFFER` an indirect draw replays from. Null
    /// for a direct draw.
    indirect_buffer: vk::Buffer,
    indirect_memory: vk::DeviceMemory,
    /// The rail's own `INDEX_BUFFER` holding `[0, 1, 2]` for an indexed
    /// indirect draw. Null for a direct or non-indexed draw.
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    /// The caller-held vertex streams the pass binds, in binding order
    /// (`research/docs/23` §3.3). Each entry is the device buffer holding one
    /// pool view's bytes; empty for the `vertex_id` milestone.
    vertex_inputs: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// The caller-held index buffer, when the draw is indexed.
    input_index_buffer: vk::Buffer,
    input_index_memory: vk::DeviceMemory,
    /// How the draw issues: the milestone triangle, a vertex-buffer draw or an
    /// indexed one. An indirect replay replaces it.
    draw: DrawShape,
    /// Index width of the caller-held index buffer.
    input_index_type: vk::IndexType,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

/// The Vulkan objects one colour attachment owns inside [`OffscreenObjects`].
///
/// `load_op` and `initial_layout` travel with the attachment because both feed
/// the render pass's per-attachment description: a loading attachment opens
/// with `LOAD_OP_LOAD` from `COLOR_ATTACHMENT_OPTIMAL` (the layout the upload
/// leaves it in), a clearing one with `LOAD_OP_CLEAR` from `UNDEFINED`, and a
/// `DontCare` one with `LOAD_OP_DONT_CARE` from `UNDEFINED` (`docs/23` §3.1,
/// v20).
struct AttachmentObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    load_op: vk::AttachmentLoadOp,
    /// The attachment's store operation: `STORE` keeps the rendered bytes for
    /// the copy-out, `DONT_CARE` discards them so no readback exists
    /// (`docs/23` §3.6, v19).
    store_op: vk::AttachmentStoreOp,
    initial_layout: vk::ImageLayout,
    /// The host-visible staging buffer holding this attachment's previous bytes
    /// for a `LoadOp::Load` pass. Null unless the attachment loads.
    previous_buffer: vk::Buffer,
    previous_memory: vk::DeviceMemory,
}

/// One readback destination: the `TRANSFER_DST` buffer, its host-visible
/// memory and the mapping the copy-out lands in.
struct ReadbackObjects {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl<'a> OffscreenObjects<'a> {
    fn new(context: &'a VulkanContext) -> Self {
        Self {
            context,
            attachments: Vec::new(),
            owns_attachments: true,
            present: false,
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vertex_module: vk::ShaderModule::null(),
            fragment_module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            readbacks: Vec::new(),
            indirect_buffer: vk::Buffer::null(),
            indirect_memory: vk::DeviceMemory::null(),
            index_buffer: vk::Buffer::null(),
            index_memory: vk::DeviceMemory::null(),
            vertex_inputs: Vec::new(),
            input_index_buffer: vk::Buffer::null(),
            input_index_memory: vk::DeviceMemory::null(),
            draw: DrawShape::Milestone,
            input_index_type: vk::IndexType::UINT16,
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
        self.attachments.push(AttachmentObjects {
            image: target.image(),
            memory: vk::DeviceMemory::null(),
            view: target.view(),
            load_op: vk::AttachmentLoadOp::CLEAR,
            // A present target is the observable landing of the pass, so its
            // store is always `STORE`; `execute_present_render` refuses a
            // `StoreOp::DontCare` present attachment before this runs.
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout,
            previous_buffer: vk::Buffer::null(),
            previous_memory: vk::DeviceMemory::null(),
        });
        self.owns_attachments = false;
        self.present = true;
    }

    /// The 2D single-sample optimal-tiling colour attachment.
    ///
    /// `TRANSFER_SRC` is part of the usage exactly when the attachment is
    /// stored, because the readback copies the stored attachment out and a
    /// discarded attachment is never read back (`docs/23` §3.6, v19);
    /// `TRANSFER_DST` is part of the usage exactly when the attachment loads,
    /// because only a loading attachment receives bytes through
    /// `vkCmdCopyBufferToImage` — a `DontCare` attachment neither uploads nor
    /// reads its pre-pass contents (`docs/23` §3.1, v20);
    /// `DEVICE_LOCAL` is the memory class the probe used for every
    /// optimal-tiling candidate.
    fn create_attachment(
        &mut self,
        format: vk::Format,
        width: u32,
        height: u32,
        load: LoadOp,
        storing: bool,
    ) -> Result<(), ProviderError> {
        let loading = matches!(load, LoadOp::Load);
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
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | if storing {
                        vk::ImageUsageFlags::TRANSFER_SRC
                    } else {
                        vk::ImageUsageFlags::empty()
                    }
                    // A loading attachment receives its previous bytes through
                    // `vkCmdCopyBufferToImage`, so the image needs the transfer
                    // destination usage exactly when one is uploaded
                    // (`research/docs/23` §3.3).
                    | if loading {
                        vk::ImageUsageFlags::TRANSFER_DST
                    } else {
                        vk::ImageUsageFlags::empty()
                    },
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, memory, _) = crate::allocate_image_backing(
            self.context,
            &info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "attachment",
        )
        .map_err(|error| execution_refusal("create attachment image", &error.detail))?;
        let view = crate::create_color_image_view(self.context, image, format, "attachment")
            .map_err(|error| execution_refusal("create attachment view", &error.detail))?;
        self.attachments.push(AttachmentObjects {
            image,
            memory,
            view,
            // A loading attachment keeps the upload's layout as the pass's
            // initial one; a clearing or `DontCare` attachment opens from
            // `UNDEFINED`, because nothing defines its bytes before the pass
            // (`research/docs/23` §3.3, `docs/23` §3.1 v20).
            load_op: match load {
                LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
            },
            store_op: if storing {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            },
            initial_layout: if loading {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::UNDEFINED
            },
            previous_buffer: vk::Buffer::null(),
            previous_memory: vk::DeviceMemory::null(),
        });
        Ok(())
    }

    /// The render pass over every attachment this scope holds, with the
    /// probe's dependency pair: `EXTERNAL → 0` makes the clear/write visible to
    /// colour output, and `0 → EXTERNAL` makes the stored texels visible to the
    /// copy that reads them. A stored offscreen attachment ends in
    /// `finalLayout = TRANSFER_SRC_OPTIMAL` so the copy runs without a further
    /// transition (`research/docs/23` §7.1); a discarded attachment ends in
    /// `COLOR_ATTACHMENT_OPTIMAL`, since nothing reads it after the pass
    /// (`docs/23` §3.6, v19). A present pass ends in `COLOR_ATTACHMENT_OPTIMAL`
    /// instead, and `record` inserts the explicit present layout transition
    /// before the copy (`docs/24` §3.3).
    ///
    /// Each entry of `formats` is the `VkFormat` of the attachment at the same
    /// location; the per-attachment load operation and initial layout come
    /// from the scope's own attachment records.
    fn create_render_pass(&mut self, formats: &[vk::Format]) -> Result<(), ProviderError> {
        let attachments = self
            .attachments
            .iter()
            .zip(formats)
            .map(|(attachment, format)| {
                let final_layout = if self.present {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                } else if attachment.store_op == vk::AttachmentStoreOp::STORE {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                };
                vk::AttachmentDescription::default()
                    .format(*format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(attachment.load_op)
                    .store_op(attachment.store_op)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(attachment.initial_layout)
                    .final_layout(final_layout)
            })
            .collect::<Vec<_>>();
        let color_refs = (0..self.attachments.len())
            .map(|index| {
                vk::AttachmentReference::default()
                    .attachment(index as u32)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            })
            .collect::<Vec<_>>();
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
        let views = self
            .attachments
            .iter()
            .map(|attachment| attachment.view)
            .collect::<Vec<_>>();
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
        vertex_streams: &[VertexStream<'_>],
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
        // The vertex input state is derived from the pipeline's own layout, so
        // the state the pipeline is built with and the buffers `record` binds
        // come from one description (`research/docs/23` §3.3). A `vertex_id`
        // pipeline declares no binding and keeps the empty state exactly.
        let mut binding_descriptions = Vec::with_capacity(vertex_streams.len());
        let mut attribute_descriptions = Vec::with_capacity(
            vertex_streams
                .iter()
                .map(|stream| stream.layout.attributes.len())
                .sum(),
        );
        for (binding, stream) in vertex_streams.iter().enumerate() {
            binding_descriptions.push(
                vk::VertexInputBindingDescription::default()
                    .binding(binding as u32)
                    .stride(
                        u32::try_from(stream.layout.stride)
                            .map_err(|_| contract_refusal("vertex stride exceeds u32"))?,
                    )
                    .input_rate(vk::VertexInputRate::VERTEX),
            );
            for attribute in &stream.layout.attributes {
                attribute_descriptions.push(
                    vk::VertexInputAttributeDescription::default()
                        .location(attribute.location)
                        .binding(binding as u32)
                        .format(vertex_vk_format(attribute.format)?)
                        .offset(u32::try_from(attribute.offset).map_err(|_| {
                            contract_refusal("vertex attribute offset exceeds u32")
                        })?),
                );
            }
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);
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
        // One no-blend, all-writes state per colour attachment: the blend
        // state is indexed by location exactly like the subpass's attachment
        // references, so a dual-attachment pipeline declares two states and
        // lets both fragment outputs land.
        let blend_attachments = self
            .attachments
            .iter()
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect::<Vec<_>>();
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

    /// Upload every caller-held stream the pass binds into its own host-visible
    /// device buffer (`research/docs/23` §3.3).
    ///
    /// One buffer per pool view, holding that view's bytes: the provider's
    /// compute path binds a lone owned view at its own offset, and this rail
    /// does the same, so the stream starts at byte zero of the view exactly as
    /// the footprint proof assumed. The pool upload for the same view may have
    /// happened on the compute path, but these buffers are the rail's own and
    /// are destroyed with the pass.
    fn create_vertex_inputs(
        &mut self,
        streams: &[VertexStream<'_>],
        index: Option<&IndexStream<'_>>,
    ) -> Result<(), ProviderError> {
        for stream in streams {
            let BufferSource::OwnedBytes(bytes) = &stream.view.source else {
                return Err(
                    capability_refusal("render_vertex_buffer_unsupported").with_detail(
                        "the first vertex-input increment executes trace-owned bytes only",
                    ),
                );
            };
            let (buffer, memory) = self.create_host_visible_buffer(
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                vk::BufferUsageFlags::VERTEX_BUFFER,
                bytes,
                "vertex input",
            )?;
            self.vertex_inputs.push((buffer, memory));
        }
        if let Some(index) = index {
            let BufferSource::OwnedBytes(bytes) = &index.view.source else {
                return Err(
                    capability_refusal("render_index_buffer_unsupported").with_detail(
                        "the first vertex-input increment executes trace-owned bytes only",
                    ),
                );
            };
            let (buffer, memory) = self.create_host_visible_buffer(
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                vk::BufferUsageFlags::INDEX_BUFFER,
                bytes,
                "index input",
            )?;
            self.input_index_buffer = buffer;
            self.input_index_memory = memory;
            self.input_index_type = indices_format(index.format);
        }
        Ok(())
    }

    /// Upload an attachment's previous bytes into a host-visible staging buffer
    /// for the `vkCmdCopyBufferToImage` a `LoadOp::Load` pass issues
    /// (`research/docs/23` §3.3).
    fn create_previous_bytes(&mut self, index: usize, bytes: &[u8]) -> Result<(), ProviderError> {
        let (buffer, memory) = self.create_host_visible_buffer(
            u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            vk::BufferUsageFlags::TRANSFER_SRC,
            bytes,
            "attachment previous bytes",
        )?;
        self.attachments[index].previous_buffer = buffer;
        self.attachments[index].previous_memory = memory;
        self.attachments[index].load_op = vk::AttachmentLoadOp::LOAD;
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
        self.readbacks.push(ReadbackObjects { buffer, memory });
        Ok(mapping)
    }

    /// Encode one `VkDrawIndirectCommand` into a host-visible
    /// `INDIRECT_BUFFER` the pass replays with `vkCmdDrawIndirect`
    /// (`research/docs/25` §6 Step 4). The command is written by the CPU, which
    /// is what makes this the ICB *equivalent* rather than device-generated
    /// commands: `VK_EXT_device_generated_commands` is not enabled and the
    /// first increment never needs it.
    fn create_indirect_draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), ProviderError> {
        let command = vk::DrawIndirectCommand {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        };
        let byte_length = std::mem::size_of::<vk::DrawIndirectCommand>() as u64;
        let info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(vk::BufferUsageFlags::INDIRECT_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.context.device.create_buffer(&info, None) }
            .map_err(|error| execution_refusal("create indirect buffer", &error.to_string()))?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    "find indirect memory type",
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
                    "allocate indirect memory",
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
                "bind indirect memory",
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
            Ok(mapping) => mapping,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(execution_refusal("map indirect memory", &error.to_string()));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &command as *const vk::DrawIndirectCommand as *const u8,
                mapping as *mut u8,
                byte_length as usize,
            );
            self.context.device.unmap_memory(memory);
        }
        self.indirect_buffer = buffer;
        self.indirect_memory = memory;
        Ok(())
    }

    /// Create one host-visible, host-coherent buffer, copy `bytes` into it and
    /// return the bound buffer/memory pair. The name labels the refusal details
    /// so a failure spells which rail buffer it was creating.
    fn create_host_visible_buffer(
        &self,
        byte_length: u64,
        usage: vk::BufferUsageFlags,
        bytes: &[u8],
        name: &'static str,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), ProviderError> {
        let info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.context.device.create_buffer(&info, None) }.map_err(|error| {
                execution_refusal(&format!("create {name} buffer"), &error.to_string())
            })?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    &format!("find {name} memory type"),
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
                    &format!("allocate {name} memory"),
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
                &format!("bind {name} memory"),
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
            Ok(mapping) => mapping,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_buffer(buffer, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(execution_refusal(
                    &format!("map {name} memory"),
                    &error.to_string(),
                ));
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapping as *mut u8, bytes.len());
            self.context.device.unmap_memory(memory);
        }
        Ok((buffer, memory))
    }

    /// Encode one `VkDrawIndexedIndirectCommand` into a host-visible
    /// `INDIRECT_BUFFER` and build the rail's own `[0, 1, 2]` `UINT32` index
    /// buffer the pass replays with `vkCmdDrawIndexedIndirect`
    /// (`research/docs/25` §6 Step 4). The index buffer is an implementation
    /// detail of the first increment: the reviewed vertex stage picks its
    /// positions from `gl_VertexIndex`, so the three index values select exactly
    /// the same full-screen triangle the non-indexed draw issues. Caller-owned
    /// vertex/index buffers belong to the render-generalisation rail and stay
    /// out of scope here.
    fn create_indirect_draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
    ) -> Result<(), ProviderError> {
        // The reviewed indexed shape is exactly three indices and the caller
        // (`execute_indirect_render_pass`) refused anything else, so the rail
        // always writes this fixed index list.
        let indices: [u32; 3] = [0, 1, 2];
        let index_bytes = indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect::<Vec<u8>>();
        let (index_buffer, index_memory) = self.create_host_visible_buffer(
            index_bytes.len() as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &index_bytes,
            "index",
        )?;
        self.index_buffer = index_buffer;
        self.index_memory = index_memory;

        let command = vk::DrawIndexedIndirectCommand {
            index_count,
            instance_count,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        };
        let byte_length = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let command_bytes = unsafe {
            std::slice::from_raw_parts(
                &command as *const vk::DrawIndexedIndirectCommand as *const u8,
                byte_length as usize,
            )
        };
        let (buffer, memory) = self.create_host_visible_buffer(
            byte_length,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            command_bytes,
            "indirect",
        )?;
        self.indirect_buffer = buffer;
        self.indirect_memory = memory;
        Ok(())
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

    /// Record clear → draw → copy-out on the one command buffer, once per
    /// attachment.
    ///
    /// The clear value is a function of the attachment format: the contract's
    /// bytes are in the format's memory order, while `VkClearColorValue`
    /// components follow the format's *component* order. Each entry of
    /// `attachments` is the pass's request record at the same location as the
    /// scope's own attachment objects.
    fn record(
        &mut self,
        attachments: &[OffscreenColorAttachment<'_>],
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

        let clear_values = attachments
            .iter()
            .map(|attachment| vk::ClearValue {
                color: clear_value_for(
                    attachment.format,
                    match attachment.load {
                        LoadOp::Clear(clear) => clear,
                        // A loading or `DontCare` attachment carries no clear
                        // colour: Vulkan ignores this entry when the load op
                        // is not `CLEAR`.
                        LoadOp::Load | LoadOp::DontCare => ClearColor::new([0; 4]),
                    },
                ),
            })
            .collect::<Vec<_>>();
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);
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
        // A loading attachment fills its image before the render pass opens:
        // the previous bytes travel through a host-visible staging buffer, land
        // in the image with `vkCmdCopyBufferToImage`, and the image is then
        // transitioned to the colour-attachment layout the render pass declares
        // as its initial layout (`research/docs/23` §3.3). Both barriers run in
        // the same command buffer, so the copy cannot be observed after the
        // draw. One round trip per attachment, in location order.
        for attachment in &self.attachments {
            if attachment.previous_buffer == vk::Buffer::null() {
                continue;
            }
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(attachment.image)
                        .subresource_range(color_subresource())
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
                );
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
                self.context.device.cmd_copy_buffer_to_image(
                    self.command,
                    attachment.previous_buffer,
                    attachment.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    std::slice::from_ref(&copy),
                );
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(attachment.image)
                        .subresource_range(color_subresource())
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)],
                );
            }
        }
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
            // Caller-held streams first (`research/docs/23` §3.3): they are the
            // shape this increment adds, and they cannot be combined with an
            // indirect replay (the pass's own bindings are the direct draw's).
            if !self.vertex_inputs.is_empty() {
                let buffers = self
                    .vertex_inputs
                    .iter()
                    .map(|(buffer, _)| *buffer)
                    .collect::<Vec<_>>();
                // The pool upload puts each view's bytes at offset zero of its
                // own buffer, so the bind offsets are zero by construction.
                let offsets = vec![0_u64; buffers.len()];
                self.context
                    .device
                    .cmd_bind_vertex_buffers(self.command, 0, &buffers, &offsets);
                match self.draw {
                    DrawShape::Indexed { index_count } => {
                        self.context.device.cmd_bind_index_buffer(
                            self.command,
                            self.input_index_buffer,
                            0,
                            self.input_index_type,
                        );
                        self.context
                            .device
                            .cmd_draw_indexed(self.command, index_count, 1, 0, 0, 0);
                    }
                    DrawShape::Vertices { vertex_count } => {
                        self.context
                            .device
                            .cmd_draw(self.command, vertex_count, 1, 0, 0);
                    }
                    // The milestone shape binds no stream, and an indirect
                    // replay is a separate arm below.
                    DrawShape::Milestone => {
                        return Err(contract_refusal(
                            "a vertex-buffer draw reached the rail without a draw shape",
                        ));
                    }
                }
            } else if self.index_buffer != vk::Buffer::null() {
                // An indexed indirect replay binds the rail's own `[0, 1, 2]`
                // index buffer and reads its counts from the `INDIRECT_BUFFER`
                // the CPU encoded above. `stride` is the struct size because
                // the first increment writes exactly one command.
                self.context.device.cmd_bind_index_buffer(
                    self.command,
                    self.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                self.context.device.cmd_draw_indexed_indirect(
                    self.command,
                    self.indirect_buffer,
                    0,
                    1,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            } else if self.indirect_buffer == vk::Buffer::null() {
                self.context
                    .device
                    .cmd_draw(self.command, FULL_SCREEN_TRIANGLE_VERTICES, 1, 0, 0);
            } else {
                // The indirect replay reads its counts from the buffer the CPU
                // encoded above; `stride` is the struct size because the first
                // increment writes exactly one command.
                self.context.device.cmd_draw_indirect(
                    self.command,
                    self.indirect_buffer,
                    0,
                    1,
                    std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
                );
            }
            self.context.device.cmd_end_render_pass(self.command);
        }

        if self.present {
            // The present action's "terminal transition": make the completed
            // colour store visible to the copy-out, in the same command buffer
            // as the render so the present cannot run before its writer
            // (`docs/24` §3.3 rule 1). The equivalent terminal state is
            // `TRANSFER_SRC_OPTIMAL`, i.e. "readable by the host after `wait`"
            // (`docs/24` §3.6), not a real `VkQueuePresentKHR`.
            let barrier = present_transition_barrier(self.attachments[0].image);
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

        // One copy per stored attachment into its own readback buffer, in
        // location order, so the host side receives the bytes of every stored
        // location and can tell them apart. A discarded attachment has no
        // readback buffer and is left out of the copy entirely (`docs/23`
        // §3.6, v19).
        for (attachment, readback) in self
            .attachments
            .iter()
            .filter(|attachment| attachment.store_op == vk::AttachmentStoreOp::STORE)
            .zip(&self.readbacks)
        {
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
                    attachment.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    std::slice::from_ref(&copy),
                );
            }
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
            if self.owns_attachments {
                for attachment in &self.attachments {
                    if attachment.view != vk::ImageView::null() {
                        self.context
                            .device
                            .destroy_image_view(attachment.view, None);
                    }
                    if attachment.image != vk::Image::null() {
                        self.context.device.destroy_image(attachment.image, None);
                    }
                    if attachment.memory != vk::DeviceMemory::null() {
                        self.context.device.free_memory(attachment.memory, None);
                    }
                }
            }
            for readback in &self.readbacks {
                if readback.memory != vk::DeviceMemory::null() {
                    self.context.device.unmap_memory(readback.memory);
                }
                if readback.buffer != vk::Buffer::null() {
                    self.context.device.destroy_buffer(readback.buffer, None);
                }
                if readback.memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(readback.memory, None);
                }
            }
            // The indirect buffer is unbound by construction (its memory is
            // freed right after), so destroy before free.
            if self.indirect_buffer != vk::Buffer::null() {
                self.context
                    .device
                    .destroy_buffer(self.indirect_buffer, None);
            }
            if self.indirect_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.indirect_memory, None);
            }
            // The index buffer is unbound by construction (its memory is freed
            // right after), so destroy before free.
            if self.index_buffer != vk::Buffer::null() {
                self.context.device.destroy_buffer(self.index_buffer, None);
            }
            if self.index_memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.index_memory, None);
            }
            // The caller-held streams are unbound by construction (their memory
            // is freed right after), so destroy before free.
            for (buffer, memory) in self.vertex_inputs.drain(..) {
                if buffer != vk::Buffer::null() {
                    self.context.device.destroy_buffer(buffer, None);
                }
                if memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(memory, None);
                }
            }
            if self.input_index_buffer != vk::Buffer::null() {
                self.context
                    .device
                    .destroy_buffer(self.input_index_buffer, None);
            }
            if self.input_index_memory != vk::DeviceMemory::null() {
                self.context
                    .device
                    .free_memory(self.input_index_memory, None);
            }
            // The staging buffer is unbound by construction (its memory is
            // freed right after), so destroy before free.
            for attachment in &self.attachments {
                if attachment.previous_buffer != vk::Buffer::null() {
                    self.context
                        .device
                        .destroy_buffer(attachment.previous_buffer, None);
                }
                if attachment.previous_memory != vk::DeviceMemory::null() {
                    self.context
                        .device
                        .free_memory(attachment.previous_memory, None);
                }
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

/// The refusal for a pass whose attachment count the MRT rail cannot execute.
///
/// The capability bit reports `MAX_COLOR_ATTACHMENTS`, so a larger pass is
/// refused here with the rail's own limit instead of silently rendering the
/// first few locations. The empty-list arm is the fail-closed half of the same
/// gate for a directly-constructed request that skipped core admission.
fn mrt_attachment_count_refusal(attachments: usize) -> ProviderError {
    let maximum = metal_api_core::provider::MAX_COLOR_ATTACHMENTS as u64;
    capability_refusal("render_mrt_attachment_count_unsupported")
        .with_field("attachments", FieldValue::Unsigned(attachments as u64))
        .with_field("maximum", FieldValue::Unsigned(maximum))
        .with_detail(
            "the MRT rail executes one to four colour attachments; a larger pass would silently \
             drop its later locations",
        )
}

/// The refusal for a dual-attachment format combination outside the reviewed
/// `[Rgba8Unorm, Rgba8Unorm]` shape.
///
/// The dual-output fragment module writes exactly the reviewed pair, so any
/// other two-format list is refused before any Vulkan object exists instead of
/// running a module whose output locations the format list does not describe.
fn mrt_format_combination_refusal(
    first: AttachmentFormat,
    second: AttachmentFormat,
) -> ProviderError {
    capability_refusal("render_mrt_format_combination_unsupported")
        .with_field(
            "format_code_0",
            FieldValue::Unsigned(u64::from(first.code())),
        )
        .with_field(
            "format_code_1",
            FieldValue::Unsigned(u64::from(second.code())),
        )
        .with_detail(
            "the reviewed dual-output module serves [Rgba8Unorm, Rgba8Unorm] only; this \
             format pair has no colour fragment stage",
        )
}

/// The rail's value-level refusal for a pass whose every attachment discards
/// (`docs/23` §3.6, v19). Core admission refuses the same shape as
/// `AllRenderAttachmentsDiscarded` → `trace_contract_invalid`; a
/// directly-constructed request skips that gate, so the rail spells its own
/// capability slug instead of executing a pass that lands nothing.
fn render_all_attachments_discarded_refusal() -> ProviderError {
    capability_refusal("render_all_attachments_discarded").with_detail(
        "every colour attachment's store operation is `DontCare`, so the pass would leave no \
             observable landing point",
    )
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

    /// Fragment stage of the reviewed `[Rgba8Unorm, Rgba8Unorm]` dual shape:
    /// `Location 0` stores `(64/255, 128/255, 192/255, 1)` and `Location 1`
    /// stores `(1, 128/255, 64/255, 192/255)`, both under the byte/255
    /// discipline of the single-output module.
    const SOLID_UNORM8_DUAL_FRAG_SPV: &[u8] =
        include_bytes!("render_spv/solid_unorm8_dual.frag.spv");

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
                color_formats: vec![format],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: solid_fragment_spirv(&[format])
                .expect("every admitted format has a reviewed stage")
                .to_vec(),
        }
    }

    /// One registration for the reviewed dual `[Rgba8Unorm, Rgba8Unorm]` shape.
    fn reviewed_dual_stages() -> RenderStages {
        RenderStages {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: SOLID_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: solid_fragment_spirv(&[
                AttachmentFormat::Rgba8Unorm,
                AttachmentFormat::Rgba8Unorm,
            ])
            .expect("the reviewed dual format list has a stage")
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
            vertex_buffers: Vec::new(),
            indices: None,
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
        let mut blobs = execute_offscreen_render(
            context,
            &OffscreenRenderRequest {
                attachments: vec![OffscreenColorAttachment {
                    format,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                    previous: None,
                }],
                extent: [2, 2],
                vertex: milestone_vertex(),
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                index_stream: None,
                indirect: None,
            },
        )
        .unwrap_or_else(|error| panic!("the 2x2 {format:?} render pass executes: {error:?}"));
        let texels = blobs.remove(0).expect("a stored attachment reads back");
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

    /// The rail's structural guarantee, stated without a device: the format list
    /// selects the fragment stage, the map is total over the admitted single
    /// formats and the reviewed dual shape, and the one format outside the
    /// increment has no stage at all.
    #[test]
    fn every_admitted_format_selects_a_reviewed_fragment_stage() {
        // `solid_fragment_spirv` matches every `AttachmentFormat` variant with no
        // default arm, so "the map covers the contract" is a compile-time fact;
        // what is checked here is the content of each arm.
        for format in AttachmentFormat::ADMITTED {
            let module = solid_fragment_spirv(&[format]).expect("an admitted format has a stage");
            assert!(
                spirv_words(module).is_some(),
                "{format:?} must name a whole number of SPIR-V words: {} bytes",
                module.len()
            );
        }
        assert_eq!(
            solid_fragment_spirv(&[AttachmentFormat::Rgba8Unorm]).expect("admitted"),
            SOLID_UNORM8_FRAG_SPV
        );
        // The B,G,R,A layout shares the 8-bit module on purpose: the channel order
        // lives in the image format, so the bytes differ while the stage does not.
        assert_eq!(
            solid_fragment_spirv(&[AttachmentFormat::Bgra8Unorm]).expect("admitted"),
            SOLID_UNORM8_FRAG_SPV
        );
        assert_eq!(
            solid_fragment_spirv(&[AttachmentFormat::R32Float]).expect("admitted"),
            SOLID_R32F_FRAG_SPV
        );
        // ... and the float stage is a different module: a one-component
        // attachment cannot take the 8-bit module's `vec4` store.
        assert_ne!(SOLID_UNORM8_FRAG_SPV, SOLID_R32F_FRAG_SPV);
        // The reviewed dual shape selects the dual-output module; every other
        // two-format list is refused with the MRT combination slug.
        assert_eq!(
            solid_fragment_spirv(&[AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm,])
                .expect("the reviewed dual list has a stage"),
            SOLID_UNORM8_DUAL_FRAG_SPV
        );
        let refused =
            solid_fragment_spirv(&[AttachmentFormat::Rgba8Unorm, AttachmentFormat::R32Float])
                .expect_err("an unreviewed dual format list has no stage");
        assert_eq!(refused.slug, "render_mrt_format_combination_unsupported");
        // `R32Uint` is the contract format outside the increment, refused with the
        // slug the contract and the format rail already use for it.
        let refused =
            solid_fragment_spirv(&[AttachmentFormat::R32Uint]).expect_err("R32Uint has no stage");
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

        // The reviewed dual shape is accepted under the dual-output module...
        reviewed_dual_stages()
            .validate()
            .expect("the dual format list is a reviewed pairing");

        // ... but registering the dual shape under the single-output module is
        // refused: that stage never stores `Location 1`, so the second
        // attachment would read back bytes the format claim does not cover.
        let mut stages = reviewed_dual_stages();
        stages.fragment_spirv = SOLID_UNORM8_FRAG_SPV.to_vec();
        let refused = stages
            .validate()
            .expect_err("a single-output module cannot describe a dual-attachment pipeline");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("format_code_0"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::Rgba8Unorm.code()
            )))
        );
        assert_eq!(
            refused.fields.get("format_code_1"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::Rgba8Unorm.code()
            )))
        );
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
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::R32Uint,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
            }],
            extent: [2, 2],
            vertex: milestone_vertex(),
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            index_stream: None,
            indirect: None,
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
            let mut blobs = execute_offscreen_render(
                &context,
                &OffscreenRenderRequest {
                    attachments: vec![OffscreenColorAttachment {
                        format,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(clear),
                        previous: None,
                    }],
                    extent: [2, 2],
                    vertex: single_pixel_vertex(),
                    vertex_streams: Vec::new(),
                    draw: DrawShape::Milestone,
                    index_stream: None,
                    indirect: None,
                },
            )
            .unwrap_or_else(|error| panic!("the partial {format:?} pass executes: {error:?}"));
            let texels = blobs.remove(0).expect("a stored attachment reads back");
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
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
            }],
            extent: [2, 0],
            vertex: milestone_vertex(),
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            index_stream: None,
            indirect: None,
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
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
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

    /// The MRT contract admits exactly `MAX_COLOR_ATTACHMENTS` attachments, and
    /// this rail now executes all of them: a five-attachment pass (which only a
    /// directly-constructed request can reach, because core admission refuses it
    /// first) is refused before any Vulkan object exists rather than silently
    /// rendering the first four locations. Host-side: `prepare_render_request`
    /// reads no device.
    #[test]
    fn prepare_render_request_refuses_a_pass_beyond_the_attachment_ceiling() {
        let maximum = metal_api_core::provider::MAX_COLOR_ATTACHMENTS;
        let mut stages = reviewed_dual_stages();
        stages.contract.color_formats =
            vec![AttachmentFormat::Rgba8Unorm; maximum + 1];
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        for _ in 0..maximum {
            pass.color_attachments.push(pass.color_attachments[0]);
        }
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes one format per location");
        let previous = vec![None; maximum + 1];

        let refused = match prepare_render_request(&stages, &pass, &previous) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a pass beyond the ceiling"),
        };
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_mrt_attachment_count_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("attachments"),
            Some(&FieldValue::Unsigned((maximum + 1) as u64))
        );
        assert_eq!(
            refused.fields.get("maximum"),
            Some(&FieldValue::Unsigned(maximum as u64))
        );
    }

    /// The reviewed dual shape end to end on one draw: both 2×2
    /// `Rgba8Unorm` attachments read back their own location's bytes, and the
    /// copy-out counter advances by two (`copy_out == 2`).
    #[test]
    fn dual_attachments_read_back_both_locations() {
        let Some(context) = device_context() else {
            return;
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed dual pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(blobs.len(), 2, "one readback per attachment");
        let location_0 = blobs[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        let location_1 = blobs[1]
            .as_ref()
            .expect("location 1 is stored and reads back");
        eprintln!("location 0: {}", hex(location_0));
        eprintln!("location 1: {}", hex(location_1));
        assert_eq!(*location_0, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert_eq!(*location_1, [0xff, 0x80, 0x40, 0xc0].repeat(4));
        assert_eq!(
            uploads_after, uploads_before,
            "no staging upload for a clear"
        );
        assert_eq!(
            readbacks_after,
            readbacks_before + 2,
            "copy_out is one per attachment"
        );
    }

    /// The v19 discard shape (`docs/23` §3.6): location 0 stores, location 1
    /// is `DontCare`, so only the stored location reads back — `Some` bytes in
    /// location order, `None` for the discarded location — and the copy-out
    /// counter advances by one rather than two.
    #[test]
    fn dual_attachments_discard_one_location_reads_back_only_the_stored_one() {
        let Some(context) = device_context() else {
            return;
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::DontCare,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed store-plus-discard pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(blobs.len(), 2, "one result entry per attachment");
        let stored = blobs[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        assert_eq!(blobs[1], None, "the discarded location reads back nothing");
        eprintln!("stored location: {}", hex(stored));
        assert_eq!(*stored, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert_eq!(
            uploads_after, uploads_before,
            "no staging upload for a clear pass"
        );
        assert_eq!(
            readbacks_after,
            readbacks_before + 1,
            "copy_out counts the stored attachment only"
        );
    }

    /// The discard removes only the store side: a `DontCare` attachment that
    /// loads is still admitted and its previous bytes are still uploaded
    /// through `vkCmdCopyBufferToImage` before the pass opens
    /// (`research/docs/23` §3.3) — the discard removes the readback, not the
    /// load upload — while the stored location keeps landing its own bytes.
    #[test]
    fn a_discarded_attachment_still_receives_its_previous_bytes() {
        let Some(context) = device_context() else {
            return;
        };
        let previous: [u8; 16] = [0x11; 16];
        let blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::DontCare,
                        load: LoadOp::Load,
                        previous: Some(&previous),
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the load-plus-discard pass executes");

        let stored = blobs[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        assert_eq!(*stored, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert_eq!(
            blobs[1], None,
            "the loaded location is discarded after the draw"
        );
    }

    /// A directly-constructed request that discards every attachment skips core
    /// admission, so the rail refuses it itself with the fail-closed slug
    /// rather than executing a pass that lands nothing (`docs/23` §3.6, v19).
    #[test]
    fn an_all_discarded_request_is_refused_before_any_device_work() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::DontCare,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
            }],
            extent: [2, 2],
            vertex: milestone_vertex(),
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            index_stream: None,
            indirect: None,
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("an all-discarded request is refused");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_all_attachments_discarded");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        // The refusal precedes every Vulkan object and every copy.
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// A dual-format list outside the reviewed `[Rgba8Unorm, Rgba8Unorm]` shape
    /// has no fragment stage and is refused before any Vulkan object exists.
    #[test]
    fn an_unreviewed_dual_format_combination_is_refused_before_the_device() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            attachments: vec![
                OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba8Unorm,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                    previous: None,
                },
                OffscreenColorAttachment {
                    format: AttachmentFormat::R32Float,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                    previous: None,
                },
            ],
            extent: [2, 2],
            vertex: milestone_vertex(),
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            index_stream: None,
            indirect: None,
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("an unreviewed format combination is refused");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_mrt_format_combination_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("format_code_0"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::Rgba8Unorm.code()
            )))
        );
        assert_eq!(
            refused.fields.get("format_code_1"),
            Some(&FieldValue::Unsigned(u64::from(
                AttachmentFormat::R32Float.code()
            )))
        );
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// Every attachment of one pass shares one extent; a pass whose attachments
    /// disagree is refused with `render_attachment_extent_mismatch` before any
    /// Vulkan object exists.
    #[test]
    fn prepare_render_request_refuses_attachments_with_mismatched_extents() {
        let stages = reviewed_dual_stages();
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.color_attachments.push(RenderAttachment {
            view_id: ViewId::new(22),
            allocation_id: AllocationId::new(32),
            format: AttachmentFormat::Rgba8Unorm,
            width: 3,
            height: 2,
            load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
            store: StoreOp::Store,
        });
        // Core admission refuses this shape (`ViewportExtentMismatch`); the
        // rail's own check is the second line of defence for a
        // directly-constructed pass, so the test skips `pass.validate()` on
        // purpose.
        stages
            .contract
            .validate_against(&pass)
            .expect("the pipeline compiles one format per location");

        let refused = match prepare_render_request(&stages, &pass, &[None, None]) {
            Err(error) => error,
            Ok(_) => panic!("attachments of one pass share one extent"),
        };
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_attachment_extent_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("attachment"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(refused.fields.get("width"), Some(&FieldValue::Unsigned(3)));
        assert_eq!(
            refused.fields.get("expected_width"),
            Some(&FieldValue::Unsigned(2))
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
        let refused = execute_render_pass(&context, &stages, &pass, &[None])
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
        let refused = execute_render_pass(&context, &stages, &pass, &[None])
            .expect_err("`Load` needs an upload rail this increment does not have");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_load_op_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// The v20 load increment (`docs/23` §3.1): a `LoadOp::DontCare`
    /// attachment declares its pre-pass contents undefined, so the rail opens
    /// the pass with `LOAD_OP_DONT_CARE` from `UNDEFINED`, uploads nothing,
    /// asks for no `TRANSFER_DST` and reads the draw's own writes back. The
    /// full-coverage milestone quad makes those writes the whole 2×2 extent.
    #[test]
    fn a_dont_care_attachment_discards_the_pre_pass_contents_and_stores_the_draw() {
        let Some(context) = device_context() else {
            return;
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                attachments: vec![OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba8Unorm,
                    store: StoreOp::Store,
                    load: LoadOp::DontCare,
                    previous: None,
                }],
                extent: [2, 2],
                vertex: milestone_vertex(),
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed single-attachment DontCare pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();
        let texels = blobs[0].as_ref().expect("the stored attachment reads back");
        eprintln!("dont_care readback: {}", hex(texels));
        assert_eq!(*texels, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert_eq!(
            uploads_after, uploads_before,
            "a DontCare attachment uploads no previous bytes"
        );
        assert_eq!(
            readbacks_after,
            readbacks_before + 1,
            "copy_out counts the stored attachment only"
        );
    }

    /// The trace path resolves no bytes for a `DontCare` attachment, and the
    /// rail holds that invariant: carrying bytes for one is refused under the
    /// retained slug rather than silently ignored. Host-side:
    /// `prepare_render_request` reads no device.
    #[test]
    fn prepare_render_request_refuses_bytes_carried_for_a_dont_care_attachment() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.color_attachments[0].load = LoadOp::DontCare;
        pass.validate()
            .expect("the DontCare pass is a legal core shape now");

        // Without bytes the shape plans; with bytes it is refused by name.
        prepare_render_request(&stages, &pass, &[None])
            .expect("a DontCare attachment with no previous bytes plans");
        let previous: [u8; 16] = [0x11; 16];
        let refused = match prepare_render_request(&stages, &pass, &[Some(&previous)]) {
            Err(error) => error,
            Ok(_) => panic!("bytes carried for a DontCare attachment are refused"),
        };
        assert_eq!(refused.slug, "attachment_load_op_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("load_op"),
            Some(&FieldValue::Text("dont_care".to_owned()))
        );
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
        let error = execute_render_pass(&context, &stages, &pass, &[None])
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
