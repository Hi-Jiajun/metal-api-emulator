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
use metal2vulkan::reflect::{ShaderReflection, ShaderStage};
use metal_api_core::provider::{
    AttachmentFormat, BlendFactor, BlendOperation, BorrowedLeaseRegistry, BorrowedView,
    BufferSource, BufferView, ClearColor, CompareFunction, CullMode, DepthLoadOp,
    DepthResolveFilter, DepthStoreOp, DepthTest, DeviceEpoch, FieldValue, IndexFormat,
    IndirectCommandDescriptor, LeaseId, LeaseRegistry, LoadOp, MultisampleDepthResolve,
    MultisampleState, MultisampleStencilResolve, ProviderError, ProviderErrorClass, ProviderPhase,
    RenderPassBlend, RenderPassCull, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, Retryability, SampleCount, StencilCompare, StencilLoadOp, StencilOp,
    StencilResolveFilter, StencilTest, StoreOp, TextureFormat, TextureSource, TextureType,
    TextureView, VertexBufferLayout, VertexFormat, VertexStep, ViewId, Winding,
    MAX_RENDER_TEXTURES,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use crate::{SpirvFeaturePolicy, VulkanContext};

/// The four-byte texel the pre-v78 rail computed every readback extent with.
///
/// `AttachmentFormat::bytes_per_texel` fixes each attachment's own width at the
/// contract layer, and a colour readback asks *that* (`research/docs/23` §78).
/// This constant is the depth surface's own `VK_FORMAT_D32_SFLOAT` width — and
/// the name the narrow colour class goes by — so the two numbers stay
/// distinguishable at the call sites that mean one and not the other.
const NARROW_BYTES_PER_TEXEL: u64 = 4;

/// The stencil surface's texel width (`research/docs/23` §3.3, v49).
///
/// `VK_FORMAT_S8_UINT` and `MTLPixelFormatStencil8` both carry one byte per
/// texel; the rail restates the contract's constant for the same reason it
/// restates the colour width above.
const STENCIL_BYTES_PER_TEXEL: u64 = 1;

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

/// The two stages a render pipeline is built from.
///
/// This is the rail's own vocabulary, not the translator's: it names the two
/// halves of a graphics pipeline and nothing else, so a caller of
/// [`RenderStage::translate`](crate::TranslatedRenderStage) cannot ask the rail
/// for a compute stage by mistake. The rail maps each variant onto the
/// translator's stage and onto the SPIR-V execution model its module has to
/// declare.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderStage {
    Vertex,
    Fragment,
}

impl RenderStage {
    /// The stage's name as every refusal and every field spells it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }

    /// The translator stage this one names.
    pub(crate) const fn translator_stage(self) -> metal2vulkan::passes::Stage {
        match self {
            Self::Vertex => metal2vulkan::passes::Stage::Vertex,
            Self::Fragment => metal2vulkan::passes::Stage::Fragment,
        }
    }

    /// The reflection stage a translation of this stage reports.
    pub(crate) const fn reflected_stage(self) -> ShaderStage {
        match self {
            Self::Vertex => ShaderStage::Vertex,
            Self::Fragment => ShaderStage::Fragment,
        }
    }

    /// The SPIR-V execution model the module has to declare.
    pub(crate) const fn execution_model(self) -> spirv::ExecutionModel {
        match self {
            Self::Vertex => spirv::ExecutionModel::Vertex,
            Self::Fragment => spirv::ExecutionModel::Fragment,
        }
    }
}

/// The reviewed vertex stages this rail compiles, in the order
/// [`reviewed_vertex_module`] lists them.
///
/// The vertex half is the caller's module, so the rail cannot derive it from
/// the contract the way it derives the fragment half from the format list. What
/// it can do is enumerate the modules the review covered: a registration whose
/// vertex module is not one of these has no reviewed semantics, and is refused
/// unless it arrives as a translated stage with its reflection.
///
/// One entry per reviewed `.spvasm` source under `render_spv/`: the milestone's
/// full-screen triangle, the caller-stream quad, the single-pixel stage the
/// coverage fixtures collapse the triangle onto, the instanced pair's vertex
/// half, and the depth pair's vertex half.
const FULL_SCREEN_TRIANGLE_VERT_SPV: &[u8] =
    include_bytes!("render_spv/fullscreen_triangle.vert.spv");
/// Entry point [`FULL_SCREEN_TRIANGLE_VERT_SPV`] declares.
const FULL_SCREEN_TRIANGLE_VERTEX_ENTRY: &str = "vertex_main";
/// The reviewed vertex stage that reads the caller-held quad positions.
const QUAD_VERTEX_SPV: &[u8] = include_bytes!("render_spv/quad_indexed.vert.spv");
/// Entry point [`QUAD_VERTEX_SPV`] declares.
const QUAD_VERTEX_ENTRY: &str = "vertex_buffer_main";
/// The reviewed vertex stage that collapses the triangle onto one pixel.
const SINGLE_PIXEL_VERT_SPV: &[u8] = include_bytes!("render_spv/single_pixel.vert.spv");
/// Entry point [`SINGLE_PIXEL_VERT_SPV`] declares.
const SINGLE_PIXEL_VERTEX_ENTRY: &str = "single_pixel";

/// The reviewed solid fragment module for a four-component colour attachment.
///
/// One module serves every admitted format with four components — the two 8-bit
/// UNORM layouts and (from `research/docs/23` §78) the eight-byte
/// `VK_FORMAT_R16G16B16A16_SFLOAT`. Which channel lands in which byte, and
/// whether each channel lands as 8-bit UNORM or as a half float, are the *image
/// format's* decisions rather than the shader's: the stage stores
/// `(64/255, 128/255, 192/255, 1)` as a `vec4` of `float`s either way, so an
/// R,G,B,A layout reads back `40 80 c0 ff` per texel, a B,G,R,A layout
/// `c0 80 40 ff`, and the float layout the same colour rounded to four halves.
/// Swizzling or pre-quantising the store would do the format's job twice and
/// land another value in those same bytes.
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

/// The reviewed three-output fragment module (`research/docs/23` §3.3, v25).
///
/// Three locations are not the ceiling, but they are their own shape: the
/// four-output module cannot stand in for it, because a fragment that writes a
/// location with no attachment beside it is undefined.
const SOLID_UNORM8_TRIPLE_FRAG_SPV: &[u8] =
    include_bytes!("render_spv/solid_unorm8_triple.frag.spv");

/// The reviewed instanced fixture's vertex stage (`research/docs/23` §3.3,
/// v31).
///
/// The module reads the caller-held quad positions at `Location 0`, the
/// per-instance tint at `Location 1`, shifts each instance's copy by half the
/// viewport with the `InstanceIndex` builtin, and forwards the tint to the
/// fragment stage. The rail pairs it with [`INSTANCED_TINT_FRAG_SPV`] and
/// nothing else, exactly as it pairs the format list with its own solid
/// fragment module.
const INSTANCED_VERTEX_SPV: &[u8] = include_bytes!("render_spv/instanced_quad.vert.spv");
/// Entry point [`INSTANCED_VERTEX_SPV`] declares.
const INSTANCED_VERTEX_ENTRY: &str = "instanced_quad_main";
/// The reviewed fragment stage of the instanced fixture: the vertex stage's
/// forwarded tint, stored to `Location 0` of the single 8-bit attachment.
const INSTANCED_TINT_FRAG_SPV: &[u8] = include_bytes!("render_spv/instanced_tint.frag.spv");
/// Entry point [`INSTANCED_TINT_FRAG_SPV`] declares.
const INSTANCED_TINT_FRAGMENT_ENTRY: &str = "instanced_tint_main";

/// The reviewed depth fixture's vertex stage (`research/docs/23` §3.3, v36).
///
/// The module reads the caller-held `float32x3` positions (so the caller
/// chooses each triangle's depth) and their `float32x4` tints, and forwards the
/// tint. The rail pairs it with [`DEPTH_TINT_FRAG_SPV`] and nothing else.
const DEPTH_VERTEX_SPV: &[u8] = include_bytes!("render_spv/depth_pair.vert.spv");
/// Entry point [`DEPTH_VERTEX_SPV`] declares.
const DEPTH_VERTEX_ENTRY: &str = "depth_pair_main";
/// The reviewed fragment stage of the depth fixture: the vertex stage's
/// forwarded tint, stored to `Location 0` of the single 8-bit attachment.
const DEPTH_TINT_FRAG_SPV: &[u8] = include_bytes!("render_spv/depth_pair_tint.frag.spv");
/// Entry point [`DEPTH_TINT_FRAG_SPV`] declares.
const DEPTH_TINT_FRAGMENT_ENTRY: &str = "depth_pair_tint_main";

/// The reviewed no-output fragment stage (`research/docs/23` §3.3, v46).
///
/// The module declares no `Output` at all, which is what makes a pass with no
/// colour attachment well formed: nothing writes colour, and the per-fragment
/// operations still test and write the depth attachment. The Metal counterpart
/// is MSL's `fragment void` (`conformance/shaders/depth_only_4x4.metal`).
const DEPTH_ONLY_FRAG_SPV: &[u8] = include_bytes!("render_spv/depth_only.frag.spv");
/// Entry point [`DEPTH_ONLY_FRAG_SPV`] declares.
const DEPTH_ONLY_FRAGMENT_ENTRY: &str = "depth_only_fragment_main";

/// The reviewed vertex stage of the render-sampler fixture
/// (`research/docs/23` §3.3, v70).
///
/// The milestone triangle's sibling: the same full-screen geometry with the
/// same Metal-NDC y flip, plus one `float32x2` varying at `Location 0` holding
/// the geometry's own normalised coordinate. Across a covering raster that
/// varying lands on `(column + 0.5) / width` and `(row + 0.5) / height` per
/// fragment — the texel centres of an attachment-sized texture — which is what
/// makes the sampling expectation independent of the driver's boundary rules.
const SAMPLED_QUAD_VERT_SPV: &[u8] = include_bytes!("render_spv/sampled_quad.vert.spv");
/// Entry point [`SAMPLED_QUAD_VERT_SPV`] declares.
const SAMPLED_QUAD_VERTEX_ENTRY: &str = "vertex_main";

/// The reviewed fragment stage of the render-sampler fixture
/// (`research/docs/23` §3.3, v70).
///
/// The solid 8-bit module's sibling: the same single `Location 0` store of a
/// `vec4`, with the sample of the pass's `DescriptorSet 0 / Binding 0` combined
/// image sampler as its value. The provider synthesises the nearest/clamp
/// sampler the compute rail already uses, so a fragment standing on a texel
/// centre reads that texel's own bytes back.
const SAMPLED_UNORM8_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_unorm8_sampled.frag.spv");

/// The sampled pass's single texture binding: the fragment stage's
/// `DescriptorSet 0 / Binding 0`.
const SAMPLED_TEXTURE_BINDING: u32 = 0;

/// The fragment stage this rail owns for the reviewed sampling pair
/// (`research/docs/23` §3.3, v70).
///
/// `None` means the request's vertex stage is not the reviewed sampling module,
/// so the fragment half stays the format list's solid module. The pair is
/// reviewed for the single 8-bit UNORM attachment the fixture draws into; any
/// other format list is refused rather than rendered with a store the review
/// never covered — the same rule [`instanced_fragment_stage`] states.
fn sampled_fragment_stage(
    vertex_entry: &str,
    vertex_spirv: &[u8],
    formats: &[AttachmentFormat],
) -> Result<Option<(&'static [u8], &'static str)>, ProviderError> {
    if vertex_entry != SAMPLED_QUAD_VERTEX_ENTRY || vertex_spirv != SAMPLED_QUAD_VERT_SPV {
        return Ok(None);
    }
    match formats {
        [AttachmentFormat::Rgba8Unorm] => Ok(Some((SAMPLED_UNORM8_FRAG_SPV, SOLID_FRAGMENT_ENTRY))),
        _ => Err(capability_refusal("render_texture_format_unsupported")
            .with_field("attachments", FieldValue::Unsigned(formats.len() as u64))
            .with_detail("the reviewed sampling module draws into one Rgba8Unorm attachment")),
    }
}

/// Whether a request's vertex stage is the reviewed sampling module, and so
/// requires the pass to bind the texture the module samples.
fn vertex_stage_is_sampled(vertex_entry: &str, vertex_spirv: &[u8]) -> bool {
    vertex_entry == SAMPLED_QUAD_VERTEX_ENTRY && vertex_spirv == SAMPLED_QUAD_VERT_SPV
}

/// The depth fixture's reviewed fragment stage, when the request's vertex stage
/// is the reviewed depth module.
///
/// `None` means the vertex stage is not that module. The pair is reviewed for
/// the single 8-bit UNORM attachment the fixture draws into; any other format
/// list is refused rather than rendered with a store the review never covered —
/// the same rule [`instanced_fragment_stage`] states. An *empty* list is the
/// zero-colour-attachment depth pass (`research/docs/23` §3.3, v46): the same
/// vertex stage beside the reviewed stage that declares no output at all.
fn depth_fragment_stage(
    vertex_entry: &str,
    vertex_spirv: &[u8],
    formats: &[AttachmentFormat],
) -> Result<Option<(&'static [u8], &'static str)>, ProviderError> {
    if vertex_entry != DEPTH_VERTEX_ENTRY || vertex_spirv != DEPTH_VERTEX_SPV {
        return Ok(None);
    }
    match formats {
        [] => Ok(Some((DEPTH_ONLY_FRAG_SPV, DEPTH_ONLY_FRAGMENT_ENTRY))),
        [AttachmentFormat::Rgba8Unorm] | [AttachmentFormat::Bgra8Unorm] => {
            Ok(Some((DEPTH_TINT_FRAG_SPV, DEPTH_TINT_FRAGMENT_ENTRY)))
        }
        _ => Err(capability_refusal("render_depth_format_unsupported")
            .with_field("attachments", FieldValue::Unsigned(formats.len() as u64))
            .with_detail("the reviewed depth module draws into one 8-bit UNORM attachment")),
    }
}

/// Whether a request's vertex stage is the reviewed depth module, and so
/// requires the pass to carry a depth attachment with a test.
fn vertex_stage_is_depth(vertex_entry: &str, vertex_spirv: &[u8]) -> bool {
    vertex_entry == DEPTH_VERTEX_ENTRY && vertex_spirv == DEPTH_VERTEX_SPV
}

/// The fragment stage this rail owns for one request's vertex stage.
///
/// `None` means the request's vertex stage is not the reviewed instanced
/// module, so the fragment half stays the format list's solid module. The
/// instanced pair is reviewed for the single 8-bit UNORM attachment the
/// fixture draws into; any other format list is refused rather than rendered
/// with a store the review never covered.
fn instanced_fragment_stage(
    vertex_entry: &str,
    vertex_spirv: &[u8],
    formats: &[AttachmentFormat],
) -> Result<Option<(&'static [u8], &'static str)>, ProviderError> {
    if vertex_entry != INSTANCED_VERTEX_ENTRY || vertex_spirv != INSTANCED_VERTEX_SPV {
        return Ok(None);
    }
    match formats {
        [AttachmentFormat::Rgba8Unorm] | [AttachmentFormat::Bgra8Unorm] => Ok(Some((
            INSTANCED_TINT_FRAG_SPV,
            INSTANCED_TINT_FRAGMENT_ENTRY,
        ))),
        _ => Err(capability_refusal("render_instanced_format_unsupported")
            .with_field("attachments", FieldValue::Unsigned(formats.len() as u64))
            .with_detail("the reviewed instanced module draws into one 8-bit UNORM attachment")),
    }
}

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
/// list is the depth-only pipeline's no-output stage (`v46`); an over-long list
/// is refused as an attachment-count capability fact, so the map is total over
/// every list shape the frozen core contract can carry.
pub(crate) fn solid_fragment_spirv(
    formats: &[AttachmentFormat],
) -> Result<&'static [u8], ProviderError> {
    // The reviewed four-component modules are layout- and storage-class-
    // agnostic: the same `vec4` store lands in whichever channel order *and*
    // storage width each attachment declares, because both are the `VkFormat`'s
    // and not the module's. The single-location module therefore serves every
    // admitted four-component format — the two 8-bit UNORM layouts and the
    // eight-byte `Rgba16Float` (`research/docs/23` §3.3 v26, §78) — while the
    // MRT modules stay reviewed for their own 8-bit format lists and the
    // single-channel float module stays the one format-specific stage. An empty
    // or over-long list is refused as a count question.
    let colour4 = |format: AttachmentFormat| {
        matches!(
            format,
            AttachmentFormat::Rgba8Unorm
                | AttachmentFormat::Bgra8Unorm
                | AttachmentFormat::Rgba16Float
        )
    };
    let unorm8 = |format: AttachmentFormat| {
        matches!(
            format,
            AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm
        )
    };
    Ok(match formats {
        // An empty list is the depth-only pipeline (`research/docs/23` §3.3,
        // v46): the reviewed fragment stage declares no output at all, so the
        // subpass has nothing to write colour into and the per-fragment
        // operations still test and write depth.
        [] => DEPTH_ONLY_FRAG_SPV,
        [AttachmentFormat::R32Float] => SOLID_R32F_FRAG_SPV,
        // The single-component arm above is the format-specific one; every
        // four-component format shares the `vec4` module, whose store the
        // attachment's own `VkFormat` converts (`research/docs/23` §78).
        [format] if colour4(*format) => SOLID_UNORM8_FRAG_SPV,
        [format] => {
            return Err(attachment_format_refusal().with_field(
                "format_code",
                FieldValue::Unsigned(u64::from(format.code())),
            ))
        }
        [first, second] if unorm8(*first) && unorm8(*second) => SOLID_UNORM8_DUAL_FRAG_SPV,
        [first, second] => {
            return Err(mrt_format_combination_refusal(*first, *second));
        }
        [first, second, third] if unorm8(*first) && unorm8(*second) && unorm8(*third) => {
            SOLID_UNORM8_TRIPLE_FRAG_SPV
        }
        [first, second, third, fourth]
            if unorm8(*first) && unorm8(*second) && unorm8(*third) && unorm8(*fourth) =>
        {
            SOLID_UNORM8_QUAD_FRAG_SPV
        }
        [first, second, ..] => {
            return Err(mrt_format_combination_refusal(*first, *second));
        }
    })
}

/// The reviewed fragment stage a colour-format list selects, paired with the
/// entry point that module declares.
///
/// Two reviewed stages serve this map: the solid modules (one entry between
/// them, [`SOLID_FRAGMENT_ENTRY`]) and the depth-only module an empty list
/// selects ([`DEPTH_ONLY_FRAGMENT_ENTRY`], `research/docs/23` §3.3, v46). The
/// pairing is stated once here so a pipeline cannot name one module's entry
/// while compiling another's bytes.
fn solid_fragment_stage(
    formats: &[AttachmentFormat],
) -> Result<(&'static [u8], &'static str), ProviderError> {
    let module = solid_fragment_spirv(formats)?;
    let entry = if formats.is_empty() {
        DEPTH_ONLY_FRAGMENT_ENTRY
    } else {
        SOLID_FRAGMENT_ENTRY
    };
    Ok((module, entry))
}

/// One contract stencil comparison as the `VkCompareOp` it names
/// (`research/docs/23` §3.3, v47).
///
/// The two admitted values are the two both APIs spell identically, so the
/// mapping is total over the contract's own list.
fn vk_stencil_compare(compare: StencilCompare) -> vk::CompareOp {
    match compare {
        StencilCompare::Equal => vk::CompareOp::EQUAL,
        StencilCompare::Always => vk::CompareOp::ALWAYS,
    }
}

/// One contract stencil operation as the `VkStencilOp` it names
/// (`research/docs/23` §3.3, v47).
fn vk_stencil_op(operation: StencilOp) -> vk::StencilOp {
    match operation {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
    }
}

/// One offscreen render pass to execute.
///
/// The shape mirrors `metal_api_core::provider::RenderPassDescriptor` for the
/// fields this rail consumes: the core type carries wiring identities
/// (pipeline/view/allocation ids and a resolved byte source) that Step 3c maps,
/// while Step 3b fixes the Vulkan-side execution against an already-chosen
/// format, extent and clear value. The shader *pair* travels as the
/// registration settled it: the fragment half is the module the registration
/// named ([`Self::translated_fragment`]), or — for a reviewed registration — the
/// module the format list selects by [`solid_fragment_spirv`], so a request
/// cannot name a fragment stage the format was not compiled for.
pub(crate) struct OffscreenRenderRequest<'a> {
    /// Colour attachments, in location order: entry `i` is the target the
    /// fragment stage's output `i` lands in. One to four entries; the rail
    /// refuses every other count before any Vulkan object exists.
    pub attachments: Vec<OffscreenColorAttachment<'a>>,
    /// The pass's scissor rectangle, or `None` for the whole render area
    /// (`research/docs/23` §3.3, v29).
    pub scissor: Option<[u32; 4]>,
    /// Instances the draw runs (`research/docs/23` §3.3, v31): the second count
    /// of `vkCmdDraw`/`vkCmdDrawIndexed`. `1` for every pre-v31 pass.
    pub instance_count: u32,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34): `vkCmdDrawIndexed`'s `vertexOffset`. `0` for every pre-v34 pass.
    pub base_vertex: u32,
    /// The culling state the pass's pipeline is built with
    /// (`research/docs/23` §3.3, v39), or `None` for "keep every triangle".
    pub cull: Option<RenderPassCull>,
    /// The blend state the pass's pipeline is built with
    /// (`research/docs/23` §3.3, v40), or `None` for "write the fragment
    /// output".
    pub blend: Option<RenderPassBlend>,
    /// The pass-wide multisample raster (`research/docs/23` §3.3, v51/v61), or
    /// `None` for the single-sample raster every pre-v51 pass ran. When
    /// present, every colour attachment is opened as a surface of the raster's
    /// own sample count and resolved into the attachment's own single-sample
    /// image before the readback copies it out, so the trace observes the
    /// resolve's bytes and not the multisampled surface's. A present pass
    /// (`research/docs/24` §3.5, v62) resolves into the provider-owned present
    /// target instead, which is the same single-sample landing the present
    /// hands on.
    pub multisample: Option<MultisampleState>,
    /// The depth resolve a stored multisampled depth surface states
    /// (`research/docs/23` §3.3, v57), or `None` for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose depth attachment
    /// is stored: the resolve is how a four-sample depth surface's texels
    /// become observable, so the request carries the filter the observation
    /// reduces with.
    pub depth_resolve: Option<MultisampleDepthResolve>,
    /// The stencil resolve a stored multisampled stencil surface states
    /// (`research/docs/23` §3.3, v60), or `None` for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose stencil
    /// attachment is stored: the resolve is how a four-sample stencil
    /// surface's texels become observable, so the request carries the filter
    /// the observation reduces with.
    pub stencil_resolve: Option<MultisampleStencilResolve>,
    /// Attachment extent in texels, shared by every entry of
    /// [`Self::attachments`] (`prepare_render_request` refuses a pass whose
    /// attachments disagree). The milestone fixes 2×2 (`docs/23` §1.3) so full
    /// coverage is distinguishable from a single stored texel.
    pub extent: [u32; 2],
    /// The depth attachment this pass opens, or `None` for a pass with no
    /// depth surface (`research/docs/23` §3.3, v36). The attachment is
    /// rail-owned: it has no trace identity and is never read back, so what it
    /// carries is the shape the rail creates and opens.
    pub depth: Option<OffscreenDepthAttachment>,
    /// The stencil attachment this pass opens, or `None` for a pass with no
    /// stencil surface (`research/docs/23` §3.3, v47). Rail-owned like the
    /// depth surface: the rail creates it, clears it and lets it go with the
    /// pass.
    pub stencil: Option<OffscreenStencilAttachment>,
    /// The vertex stage the graphics pipeline is built from.
    pub vertex: OffscreenVertexStage<'a>,
    /// The fragment stage the graphics pipeline is built from, when the
    /// registration carries a translated one. `None` is a reviewed
    /// registration: its fragment module is the format list's own, derived at
    /// execution from the attachment list, which is also where the per-format
    /// refusals live.
    pub translated_fragment: Option<OffscreenFragmentStage<'a>>,
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
    /// The sampled textures the fragment stage reads, in binding order
    /// (`research/docs/23` §3.3, v70). Empty for every pre-v70 pass, which is
    /// the shape the pipeline layout and the descriptor bind below branch on.
    pub textures: Vec<OffscreenRenderTexture<'a>>,
}

/// One sampled texture a render pass binds: the source of its texel bytes plus
/// the shape the rail executes them with (`research/docs/23` §3.3, v70).
///
/// The entry's position in [`OffscreenRenderRequest::textures`] is the binding
/// index — the contract already held the view's own label to it — and the rail
/// uploads these bytes into an image of its own, exactly as the compute rail
/// uploads a pass's texture bindings; the no-copy arm imports the owner's own
/// pages as the copy's transfer source instead (`research/docs/23` §75, R5c).
/// The first increment executes one `rgba8_unorm` 2D surface whose extent
/// matches the render area, so every fragment stands on a texel centre and the
/// nearest sample is an identity copy rather than a filtered or
/// boundary-dependent read.
pub(crate) struct OffscreenRenderTexture<'a> {
    /// Where the texture's tightly packed, row-major texel bytes come from.
    /// The three arms are the three [`TextureSource`] arms, resolved before any
    /// device object exists.
    pub source: RenderInputSource<'a>,
    /// Extent in texels, which the pass requires to equal the render area
    /// (`prepare_render_request` refuses the pass otherwise).
    pub extent: [u32; 2],
}

/// The depth attachment one offscreen pass opens (`research/docs/23` §3.3,
/// v36/v43).
pub(crate) struct OffscreenDepthAttachment {
    /// Extent in texels; core admission already held it to the colour
    /// attachments' own extent.
    pub width: u32,
    pub height: u32,
    /// `Some(depth)` for a clear load, `None` for `Load`.
    pub clear: Option<f32>,
    /// The pass's depth state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub test: Option<DepthTest>,
    /// The store action the trace stated, or `None` for the pre-v43 shape
    /// (`research/docs/23` §3.3, v43). A storing surface is the only one this
    /// rail reads back: it ends the render pass in `TRANSFER_SRC_OPTIMAL`, gets
    /// its own readback buffer, and its texels leave through the same
    /// `vkCmdCopyImageToBuffer` the colour attachments use.
    pub store: Option<DepthStoreOp>,
}

impl OffscreenDepthAttachment {
    /// Whether the pass keeps this surface — and therefore reads it back.
    fn storing(&self) -> bool {
        self.store == Some(DepthStoreOp::Store)
    }
}

/// The stencil attachment one offscreen pass opens (`research/docs/23` §3.3,
/// v47/v49).
///
/// Rail-owned like the depth increment's surface: the reviewed fixture masks
/// with it — the stored values decide which primitives survive — so what it
/// carries is the shape the rail creates and opens plus the state its draw
/// tests and writes with. From v49 on the trace may also keep the surface: a
/// storing attachment is a landing, one byte per texel, and leaves through the
/// same readback channel the colour and depth attachments use.
pub(crate) struct OffscreenStencilAttachment {
    /// Extent in texels; core admission already held it to the colour
    /// attachments' own extent.
    pub width: u32,
    pub height: u32,
    /// `Some(value)` for a clear load, `None` for `Load`.
    pub clear: Option<u8>,
    /// The pass's stencil state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub test: Option<StencilTest>,
    /// The store action the trace stated, or `None` for the rail-owned shape
    /// every pre-v49 trace means (`research/docs/23` §3.3, v49). A storing
    /// surface is the second surface this rail reads back: it ends the render
    /// pass in `TRANSFER_SRC_OPTIMAL`, gets its own readback buffer, and its
    /// one-byte texels leave through the same `vkCmdCopyImageToBuffer` the
    /// colour and depth attachments use.
    pub store: Option<StoreOp>,
}

impl OffscreenStencilAttachment {
    /// Whether the pass keeps this surface — and therefore reads it back.
    fn storing(&self) -> bool {
        self.store == Some(StoreOp::Store)
    }
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
    /// The attachment's previous contents for a `LoadOp::Load` pass
    /// (`research/docs/23` §3.3/§74). `Some` means the rail uploads them into
    /// the image and opens the render pass with `LOAD_OP_LOAD`; `None` is the
    /// `Clear`/`DontCare` shape. The source was resolved from the declaring
    /// view before any device object exists, so a `Load` cannot be executed as
    /// a clear and a lease-backed declaration cannot be read as stale bytes: an
    /// owner window is imported as the copy's own source, and the retain the
    /// pass took keeps it alive until the fence signals (R5b).
    pub previous: Option<RenderInputSource<'a>>,
    /// The one texel a multisampled `Load` seeds every sample with
    /// (`research/docs/23` §82, v82).
    ///
    /// A multisampled attachment cannot receive its previous bytes through
    /// `vkCmdCopyBufferToImage` (the command's own valid usage holds
    /// `dstImage` to a sample count of one), so the reviewed load route is a
    /// *seed pass*: a render pass the rail records before the measured one,
    /// opening the same image from `CLEAR` — every sample of the render area
    /// takes the clear value — and storing it, after which the measured pass
    /// opens the image with `LOAD_OP_LOAD`. A clear value is one colour for the
    /// whole attachment, so the declared window has to be one repeated texel;
    /// `Some` exactly for that shape and `None` for every other load.
    pub seed: Option<ClearColor>,
    /// The provider-owned image this attachment renders into instead of a
    /// per-pass attachment image, or `None` for the offscreen shape every
    /// earlier increment published (`research/docs/23` §76, R7).
    ///
    /// `Some` means the pass's store lands in the provider's own image under
    /// the attachment's `(allocation, view)` identity, and a `Resident` load
    /// keeps that image's own contents instead of uploading `previous`. The
    /// image is borrowed for the pass; the provider keeps owning it, exactly
    /// as the present rail borrows its target (`docs/24` §5.2).
    pub resident: Option<&'a ProviderTargetImage>,
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
    /// The window this stream's bytes come from, resolved before any device
    /// object exists (`research/docs/23` §71, R3c).
    pub source: RenderInputSource<'a>,
}

/// One caller-held index buffer: its width and the pool view holding it.
pub(crate) struct IndexStream<'a> {
    pub format: IndexFormat,
    /// The window this index stream's bytes come from, resolved beside the
    /// vertex streams (`research/docs/23` §71, R3c).
    pub source: RenderInputSource<'a>,
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
    /// Entry point of the vertex-stage module, as `VkPipelineShaderStageCreateInfo`
    /// binds it: the contract's own entry for a reviewed module, and the single
    /// entry point the module declares for a translated one.
    pub entry: Cow<'a, str>,
    /// Vertex-stage SPIR-V module.
    pub spirv: &'a [u8],
}

/// The fragment stage a translated registration binds.
///
/// The reviewed path names no fragment module here: the rail derives it from
/// the attachment format list (`solid_fragment_stage`) and that derivation is
/// also where the per-format refusals live. A translated registration names its
/// own module, so the rail executes exactly the module the contract's
/// reflection was checked against instead of the reviewed one of the format.
pub(crate) struct OffscreenFragmentStage<'a> {
    /// Entry point of the fragment-stage module, read from the module's own
    /// `OpEntryPoint` (the translator emits `"main"`).
    pub entry: String,
    /// Fragment-stage SPIR-V module.
    pub spirv: &'a [u8],
}

/// One host-registered render pipeline: the two compiled stage modules and the
/// contract they were built against, plus — when a module did not come from
/// this rail's reviewed set — the reflection that describes it.
///
/// The compute rail keeps one translated artifact per `PipelineId` in the
/// provider registry; this is the render sibling of that value, stored in the
/// same registry namespace so a trace's pipeline table stays the single source
/// of which pipeline a pass names.
///
/// Each stage is either one of the rail's reviewed modules or a module the
/// translator produced for this very registration, and the registration gate
/// holds both arms ([`RenderStages::validate`]): a reviewed fragment module has
/// to be the one the contract's colour format list selects
/// ([`fragment_stage_is_reviewed`]), and a translated stage has to agree with
/// the contract field by field ([`validate_translated_stage`]).
pub(crate) struct RenderStages {
    pub contract: RenderPipelineContract,
    pub vertex_spirv: Vec<u8>,
    pub fragment_spirv: Vec<u8>,
    /// The translation of the vertex module, when the module is not one of the
    /// rail's reviewed vertex stages.
    pub vertex_translation: Option<ShaderReflection>,
    /// The translation of the fragment module, when the module is not the
    /// reviewed module of the contract's colour format list.
    pub fragment_translation: Option<ShaderReflection>,
}

impl RenderStages {
    /// Structural validation of one registration, before any trace can name it.
    ///
    /// The entry names and the module bytes are checked here because both are
    /// per-registration facts: an empty entry name, a name carrying an interior
    /// NUL and a module that is not a whole number of SPIR-V words are refused
    /// once, at registration, instead of on every submission that names the
    /// pipeline. The same place settles each stage's identity, for the same
    /// reason: the registration is where the pairing is settled, so a trace
    /// never sees a pair this rail cannot execute.
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
        self.validate_stage_pair()
    }

    /// Which module each half of the pipeline executes, and whether the rail
    /// can account for it.
    ///
    /// Two arms per stage, and the execution path re-asks exactly this pair of
    /// questions of the value it was handed, so a directly-constructed
    /// [`RenderStages`] cannot skip the registration gate:
    ///
    /// * a reviewed module, identified by its bytes and the entry it declares.
    ///   The fragment half has to be the module the contract's colour format
    ///   list selects: the rail cannot read a module's semantics, so binding
    ///   the registration to the reviewed set is what refuses "this format
    ///   list, that format list's fragment stage" *before* a submission can
    ///   read back bytes the format claim does not cover (review item I2,
    ///   2026-09-14) — an `R32Float` pipeline handed the 8-bit module's `vec4`
    ///   store is a component-shape mismatch, not a byte-order preference, and
    ///   a dual-attachment pipeline handed the single-output module would never
    ///   store `Location 1`;
    /// * a translated module, identified by the reflection the translator
    ///   produced beside it. The reflection is checked against the contract
    ///   field by field, so an unreviewed module is only ever executed under a
    ///   stated interface the contract covers.
    ///
    /// A stage that is neither is refused by name
    /// (`render_stage_translation_unavailable`): the rail has no semantics for
    /// it and will not execute it on the strength of its bytes alone.
    pub(crate) fn validate_stage_pair(&self) -> Result<(), ProviderError> {
        self.validate_vertex_half()?;
        self.validate_fragment_half()?;
        if let (Some(vertex), Some(fragment)) =
            (&self.vertex_translation, &self.fragment_translation)
        {
            validate_varying_linkage(vertex, fragment, &self.contract.fragment_entry)?;
        }
        Ok(())
    }

    /// The vertex half: a reviewed module under the entry it declares, or a
    /// translated module described by its reflection.
    fn validate_vertex_half(&self) -> Result<(), ProviderError> {
        match &self.vertex_translation {
            Some(reflection) => validate_translated_stage(self, RenderStage::Vertex, reflection),
            None => {
                if reviewed_vertex_module(&self.contract.vertex_entry, &self.vertex_spirv) {
                    Ok(())
                } else {
                    Err(render_stage_translation_unavailable_refusal(
                        RenderStage::Vertex,
                        &self.contract.vertex_entry,
                    ))
                }
            }
        }
    }

    /// The fragment half: the reviewed module of the contract's colour format
    /// list, or a translated module described by its reflection.
    fn validate_fragment_half(&self) -> Result<(), ProviderError> {
        match &self.fragment_translation {
            Some(reflection) => validate_translated_stage(self, RenderStage::Fragment, reflection),
            None => {
                if fragment_stage_is_reviewed(self) {
                    Ok(())
                } else {
                    Err(fragment_stage_mismatch_refusal(
                        &self.contract.color_formats,
                        &self.contract.fragment_entry,
                    ))
                }
            }
        }
    }
}

/// Whether one registration's vertex module is one of the reviewed stages the
/// rail compiles.
///
/// The reviewed vertex modules are a closed set for the same reason the
/// fragment modules are: the rail compares the bytes it was handed against the
/// modules whose behaviour the review covers, because it cannot read a module's
/// semantics itself. A vertex stage outside the set is only executable as a
/// translated stage, under the reflection that describes it
/// ([`validate_translated_stage`]).
fn reviewed_vertex_module(entry: &str, module: &[u8]) -> bool {
    [
        (
            FULL_SCREEN_TRIANGLE_VERTEX_ENTRY,
            FULL_SCREEN_TRIANGLE_VERT_SPV,
        ),
        (QUAD_VERTEX_ENTRY, QUAD_VERTEX_SPV),
        (SINGLE_PIXEL_VERTEX_ENTRY, SINGLE_PIXEL_VERT_SPV),
        (INSTANCED_VERTEX_ENTRY, INSTANCED_VERTEX_SPV),
        (DEPTH_VERTEX_ENTRY, DEPTH_VERTEX_SPV),
        (SAMPLED_QUAD_VERTEX_ENTRY, SAMPLED_QUAD_VERT_SPV),
    ]
    .into_iter()
    .any(|(reviewed_entry, reviewed_module)| entry == reviewed_entry && module == reviewed_module)
}

/// Whether a registration's fragment stage is exactly the module this rail
/// builds for the contract's colour format list, under the entry that module
/// declares.
///
/// Both ends of the rail ask this question — registration refuses a pairing
/// once, and execution re-asks it of the value it was handed, so a
/// directly-constructed [`RenderStages`] cannot skip the registration gate.
fn fragment_stage_is_reviewed(stages: &RenderStages) -> bool {
    // The instanced fixture owns a second reviewed pair
    // (`research/docs/23` §3.3, v31): the vertex module that forwards a
    // per-instance tint selects the tint-storing fragment module, and every
    // other vertex stage keeps the format list's solid module.
    let stage = match depth_fragment_stage(
        &stages.contract.vertex_entry,
        &stages.vertex_spirv,
        &stages.contract.color_formats,
    ) {
        Ok(Some(pair)) => pair,
        Ok(None) => match instanced_fragment_stage(
            &stages.contract.vertex_entry,
            &stages.vertex_spirv,
            &stages.contract.color_formats,
        ) {
            Ok(Some(pair)) => pair,
            Ok(None) => match sampled_fragment_stage(
                &stages.contract.vertex_entry,
                &stages.vertex_spirv,
                &stages.contract.color_formats,
            ) {
                Ok(Some(pair)) => pair,
                Ok(None) => match solid_fragment_stage(&stages.contract.color_formats) {
                    Ok(stage) => stage,
                    Err(_) => return false,
                },
                Err(_) => return false,
            },
            Err(_) => return false,
        },
        Err(_) => return false,
    };
    stages.contract.fragment_entry == stage.1 && stages.fragment_spirv.as_slice() == stage.0
}

/// Validate one translated stage against the contract it is registered under.
///
/// This is the second arm of the registration gate (the R2 increment of
/// `research/docs/23`): a module the rail did not compile itself is executable
/// exactly when the translation that produced it was checked against the
/// contract, so the registration can say *what* the module does instead of only
/// *that* it is a module. Four questions, in this order:
///
/// 1. identity — the reflection has to be this stage's, under the entry the
///    contract names. The pipeline binds the module's own entry point (the
///    translator names it; see [`module_entry_point`]), while the reflection
///    carries the AIR function's name, so the entry check is what keeps "which
///    function did I translate" answerable at all;
/// 2. the module's own shape — exactly one entry point of this stage, which is
///    what a translation of one stage produces;
/// 3. the interface the rail does not execute yet
///    ([`unsupported_interface_field`]) is refused by name instead of being
///    silently ignored;
/// 4. the contract's own shape — vertex attributes against the declared vertex
///    layout, render targets against the declared colour format list.
///
/// Everything refused here is refused at registration, before any Vulkan object
/// exists, and the same function runs again when a submission names the pipeline
/// (`prepare_render_request`), so a directly-constructed [`RenderStages`] cannot
/// skip the gate.
fn validate_translated_stage(
    stages: &RenderStages,
    stage: RenderStage,
    reflection: &ShaderReflection,
) -> Result<(), ProviderError> {
    let (entry, module) = match stage {
        RenderStage::Vertex => (
            stages.contract.vertex_entry.as_str(),
            stages.vertex_spirv.as_slice(),
        ),
        RenderStage::Fragment => (
            stages.contract.fragment_entry.as_str(),
            stages.fragment_spirv.as_slice(),
        ),
    };
    if reflection.stage != stage.reflected_stage()
        || reflection.entry_point.as_deref() != Some(entry)
    {
        return Err(reflection_mismatch_refusal(stage, entry)
            .with_field(
                "reflected_stage",
                FieldValue::Text(reflected_stage_name(reflection.stage).to_owned()),
            )
            .with_field(
                "reflected_entry",
                FieldValue::Text(
                    reflection
                        .entry_point
                        .clone()
                        .unwrap_or_else(|| "<none>".to_owned()),
                ),
            )
            .with_detail(
                "the reflection describes another stage or another entry than the contract names, \
                 so the module the pipeline would bind is not the module the contract was settled \
                 against",
            ));
    }
    if module_entry_point(module, stage.execution_model()).is_none() {
        return Err(
            render_stage_translation_unavailable_refusal(stage, entry).with_detail(
                "the module does not declare exactly one entry point of this stage, and the one \
                 entry point is what the pipeline binds",
            ),
        );
    }
    if let Some(field) = unsupported_interface_field(reflection) {
        return Err(unsupported_interface_refusal(stage, entry, field));
    }
    match stage {
        RenderStage::Vertex => validate_translated_vertex(stages, entry, reflection),
        RenderStage::Fragment => validate_translated_fragment(stages, entry, reflection),
    }
}

/// The capability subset check every module this rail executes has to pass (R8).
///
/// The translation entry points ask the same gate while a module is decoded,
/// with the device's own [`SpirvFeaturePolicy`]; this is the rail's half, asked
/// again wherever a module is about to become a pipeline or a command buffer.
/// It is asked of both arms — a reviewed module and a translated one — so the
/// registration gate and the execution gate answer the same question, and a
/// caller that translated a stage under another device's policy cannot hand
/// this rail a module the device could not create.
pub(crate) fn validate_module_capabilities(
    stages: &RenderStages,
    policy: SpirvFeaturePolicy,
) -> Result<(), ProviderError> {
    for (stage, entry, module) in [
        (
            RenderStage::Vertex,
            stages.contract.vertex_entry.as_str(),
            stages.vertex_spirv.as_slice(),
        ),
        (
            RenderStage::Fragment,
            stages.contract.fragment_entry.as_str(),
            stages.fragment_spirv.as_slice(),
        ),
    ] {
        crate::validate_spirv_capabilities(module, policy).map_err(|error| {
            render_stage_capability_refusal(stage, entry).with_detail(error.message().to_owned())
        })?;
    }
    Ok(())
}

/// The vertex half's own agreement with the contract.
///
/// The raster pipeline needs a clip position, the contract's vertex layout is
/// the whole input side of the stage, and a vertex stage cannot write a colour
/// attachment. Each of the three is a field-by-field comparison, so a missing or
/// an extra reflected field is refused rather than left to a driver's vertex
/// input state.
fn validate_translated_vertex(
    stages: &RenderStages,
    entry: &str,
    reflection: &ShaderReflection,
) -> Result<(), ProviderError> {
    let mismatch = |field: &str| {
        reflection_mismatch_refusal(RenderStage::Vertex, entry)
            .with_field("field", FieldValue::Text(field.to_owned()))
    };
    match reflection.vertex_builtins {
        Some(builtins) if builtins.writes_position => {}
        Some(_) => {
            return Err(mismatch("vertex_builtins").with_detail(
                "the reflection reports no clip position, so no primitive could leave the raster",
            ))
        }
        None => {
            return Err(mismatch("vertex_builtins")
                .with_detail("the reflection reports no vertex builtin usage at all"))
        }
    }
    if !reflection.render_targets.is_empty() {
        return Err(mismatch("render_targets").with_detail(
            "a vertex stage writes no colour attachment; the contract's colour format list is \
             the fragment stage's",
        ));
    }
    validate_translated_vertex_attributes(stages, entry, reflection)
}

/// The contract's vertex layout against the reflection's attributes.
///
/// A stream the layout describes and the reflection does not read (or the
/// reverse) is a different interface, not a preference: the pipeline's vertex
/// input state is built from the layout, so a mismatch would either bind a
/// stream the shader never consumes or leave a location the shader does read
/// undefined.
fn validate_translated_vertex_attributes(
    stages: &RenderStages,
    entry: &str,
    reflection: &ShaderReflection,
) -> Result<(), ProviderError> {
    let mismatch = |field: &str| {
        reflection_mismatch_refusal(RenderStage::Vertex, entry)
            .with_field("field", FieldValue::Text(field.to_owned()))
    };
    let declared = stages
        .contract
        .vertex_layout
        .buffers()
        .iter()
        .flat_map(|buffer| buffer.attributes.iter())
        .collect::<Vec<_>>();
    let mut reflected = BTreeMap::new();
    for attribute in &reflection.vertex_attributes {
        if reflected.insert(attribute.location, attribute).is_some() {
            return Err(mismatch("vertex_attributes")
                .with_field(
                    "location",
                    FieldValue::Unsigned(u64::from(attribute.location)),
                )
                .with_detail("two reflected attributes share one location"));
        }
    }
    if declared.len() != reflected.len() {
        return Err(mismatch("vertex_attributes")
            .with_field(
                "declared_attributes",
                FieldValue::Unsigned(declared.len() as u64),
            )
            .with_field(
                "reflected_attributes",
                FieldValue::Unsigned(reflected.len() as u64),
            )
            .with_detail(
                "the contract's vertex layout and the reflection have to name the same \
                 attributes, because the layout is what the pipeline's vertex input state is \
                 built from",
            ));
    }
    for attribute in declared {
        let Some(reflected) = reflected.get(&attribute.location) else {
            return Err(mismatch("vertex_attributes")
                .with_field(
                    "location",
                    FieldValue::Unsigned(u64::from(attribute.location)),
                )
                .with_detail(
                    "the reflection does not read the attribute the contract's vertex layout \
                     declares at this location",
                ));
        };
        if !air_type_name_names_vertex_format(reflected.type_name.as_deref(), attribute.format) {
            return Err(mismatch("vertex_attributes")
                .with_field(
                    "location",
                    FieldValue::Unsigned(u64::from(attribute.location)),
                )
                .with_field(
                    "format_code",
                    FieldValue::Unsigned(u64::from(attribute.format.code())),
                )
                .with_field(
                    "type_name",
                    FieldValue::Text(
                        reflected
                            .type_name
                            .clone()
                            .unwrap_or_else(|| "<none>".to_owned()),
                    ),
                )
                .with_detail(
                    "the reflected AIR type is not the component shape the contract declares for \
                     this attribute",
                ));
        }
    }
    Ok(())
}

/// The fragment half's own agreement with the contract.
///
/// One render target per declared colour format, in location order: the count,
/// the locations and each target's component shape. A fragment stage reads no
/// vertex attribute and uses no vertex builtin, so either of those in the
/// reflection is the wrong stage's interface.
fn validate_translated_fragment(
    stages: &RenderStages,
    entry: &str,
    reflection: &ShaderReflection,
) -> Result<(), ProviderError> {
    let mismatch = |field: &str| {
        reflection_mismatch_refusal(RenderStage::Fragment, entry)
            .with_field("field", FieldValue::Text(field.to_owned()))
    };
    if !reflection.vertex_attributes.is_empty() {
        return Err(mismatch("vertex_attributes").with_detail(
            "a fragment stage reads no vertex attribute; those streams belong to the contract's \
             vertex layout and the vertex stage",
        ));
    }
    if reflection.vertex_builtins.is_some() {
        return Err(
            mismatch("vertex_builtins").with_detail("a fragment stage consumes no vertex builtin")
        );
    }
    let declared = &stages.contract.color_formats;
    if declared.len() != reflection.render_targets.len() {
        return Err(mismatch("render_targets")
            .with_field(
                "declared_targets",
                FieldValue::Unsigned(declared.len() as u64),
            )
            .with_field(
                "reflected_targets",
                FieldValue::Unsigned(reflection.render_targets.len() as u64),
            )
            .with_detail(
                "the contract's colour format list and the reflection have to name the same \
                 render targets; a stage that stores a location with no attachment beside it (or \
                 skips one it has) is a different interface",
            ));
    }
    for (location, (format, target)) in declared.iter().zip(&reflection.render_targets).enumerate()
    {
        let location = location as u32;
        if target.location != location {
            return Err(mismatch("render_targets")
                .with_field("location", FieldValue::Unsigned(u64::from(location)))
                .with_field(
                    "reflected_location",
                    FieldValue::Unsigned(u64::from(target.location)),
                )
                .with_detail(
                    "the reflection stores this location out of order, so it would not land in \
                     the attachment the contract declares at this position",
                ));
        }
        if !air_type_name_names_attachment(target.type_name.as_deref(), *format) {
            return Err(mismatch("render_targets")
                .with_field("location", FieldValue::Unsigned(u64::from(location)))
                .with_field(
                    "format_code",
                    FieldValue::Unsigned(u64::from(format.code())),
                )
                .with_field(
                    "type_name",
                    FieldValue::Text(
                        target
                            .type_name
                            .clone()
                            .unwrap_or_else(|| "<none>".to_owned()),
                    ),
                )
                .with_detail(
                    "the reflected AIR type is not the component shape the contract's attachment \
                     format stores",
                ));
        }
    }
    Ok(())
}

/// The two translated stages have to describe one interface between them.
///
/// A vertex stage's user-varying outputs are the fragment stage's `stage_in`
/// inputs: Vulkan requires every consumed input to have a producing output at
/// the same `Location`. Each reflection names its own end of that pair, so the
/// pair is checked when both halves of a registration are translated — a varying
/// one side names and the other does not is a linkage this rail would otherwise
/// discover as an undefined readback.
fn validate_varying_linkage(
    vertex: &ShaderReflection,
    fragment: &ShaderReflection,
    fragment_entry: &str,
) -> Result<(), ProviderError> {
    let produced = vertex
        .varyings
        .iter()
        .map(|varying| (varying.location, varying.type_name.as_deref()))
        .collect::<BTreeMap<_, _>>();
    let consumed = fragment
        .varyings
        .iter()
        .map(|varying| (varying.location, varying.type_name.as_deref()))
        .collect::<BTreeMap<_, _>>();
    let refusal = |detail: &str| {
        reflection_mismatch_refusal(RenderStage::Fragment, fragment_entry)
            .with_field("field", FieldValue::Text("varyings".to_owned()))
            .with_field(
                "produced_varyings",
                FieldValue::Text(varying_locations(&produced)),
            )
            .with_field(
                "consumed_varyings",
                FieldValue::Text(varying_locations(&consumed)),
            )
            .with_detail(detail.to_owned())
    };
    if produced.len() != consumed.len() || produced.keys().ne(consumed.keys()) {
        return Err(refusal(
            "the vertex stage's varying outputs and the fragment stage's stage_in inputs have to \
             name the same locations",
        ));
    }
    for (location, produced_type) in &produced {
        if let (Some(produced_type), Some(consumed_type)) =
            (produced_type, consumed.get(location).copied().flatten())
        {
            if *produced_type != consumed_type {
                return Err(refusal(
                    "a varying's reflected AIR type differs between the two stages, so the two \
                     interfaces do not match component-wise",
                )
                .with_field("location", FieldValue::Unsigned(u64::from(*location))));
            }
        }
    }
    Ok(())
}

/// The locations of a reflected varying list, for one refusal's own fields.
fn varying_locations(varyings: &BTreeMap<u32, Option<&str>>) -> String {
    varyings
        .keys()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The first reflected interface field this rail does not execute yet.
///
/// The list is exhaustive over the interface the rail *does* execute — the
/// vertex attributes, the varyings, the render targets and the vertex builtins —
/// plus everything a translation can state beyond it. Everything here is a
/// capability fact rather than a mismatch: the translation may describe its
/// stage perfectly and the rail still has no shape for it, so it refuses by name
/// instead of executing the stage with that interface silently dropped.
fn unsupported_interface_field(reflection: &ShaderReflection) -> Option<&'static str> {
    [
        (!reflection.bindings.is_empty(), "bindings"),
        (
            !reflection.argument_buffer_fields.is_empty(),
            "argument_buffer_fields",
        ),
        (!reflection.depth_members.is_empty(), "depth_members"),
        (reflection.depth_qualifier.is_some(), "depth_qualifier"),
        (!reflection.stencil_members.is_empty(), "stencil_members"),
        (reflection.tessellation.is_some(), "tessellation"),
        (
            !reflection.imageblock_layouts.is_empty(),
            "imageblock_layouts",
        ),
        (
            !reflection.implicit_imageblock_attachments.is_empty(),
            "implicit_imageblock_attachments",
        ),
        (
            reflection.fragment_imageblock.is_some(),
            "fragment_imageblock",
        ),
        (
            !reflection.runtime_sampler_specializations.is_empty(),
            "runtime_sampler_specializations",
        ),
        (
            !reflection.runtime_storage_image_specializations.is_empty(),
            "runtime_storage_image_specializations",
        ),
        (
            !reflection.function_constants.is_empty(),
            "function_constants",
        ),
        (reflection.local_size.is_some(), "local_size"),
        (
            reflection.max_work_group_size.is_some(),
            "max_work_group_size",
        ),
        (reflection.kernel_dispatch.is_some(), "kernel_dispatch"),
    ]
    .into_iter()
    .find_map(|(present, field)| present.then_some(field))
}

/// Whether one AIR type name is the component shape a contract vertex format
/// declares.
///
/// The reflected name is the AIR type the attribute was compiled as (`float3`,
/// `uint`, …), and it is the only end of the pair that can state the component
/// shape of the shader's own read. The arithmetic width is not part of the
/// question: a `half2` read of a two-component stream is the same interface as
/// its `float2` sibling, because the contract's format is what the pipeline's
/// vertex input state is built from either way.
fn air_type_name_names_vertex_format(type_name: Option<&str>, format: VertexFormat) -> bool {
    match format {
        VertexFormat::Float32x2 => matches!(type_name, Some("float2" | "half2")),
        VertexFormat::Float32x3 => matches!(type_name, Some("float3" | "half3")),
        VertexFormat::Float32x4 => matches!(type_name, Some("float4" | "half4")),
        VertexFormat::Uint32 => matches!(type_name, Some("uint" | "uint1")),
    }
}

/// Whether one AIR type name is the component shape a contract attachment
/// format stores, for the reason [`air_type_name_names_vertex_format`] states.
///
/// The component *count* is what this asks about: an attachment format stores
/// one channel per component, so a `float4` store into an 8-bit RGBA attachment
/// is the reviewed shape, while a one-component store into the same attachment
/// would leave three channels to a `StoreOp` nothing wrote. The storage *width*
/// is not part of the question — `float4`/`half4` are the same interface to a
/// 16-bit float attachment as to an 8-bit UNORM one (`research/docs/23` §78),
/// and the `AIR` type name is the same spelling either way.
fn air_type_name_names_attachment(type_name: Option<&str>, format: AttachmentFormat) -> bool {
    match format {
        AttachmentFormat::Rgba8Unorm
        | AttachmentFormat::Bgra8Unorm
        | AttachmentFormat::Rgba16Float => {
            matches!(type_name, Some("float4" | "half4"))
        }
        AttachmentFormat::R32Float => matches!(type_name, Some("float" | "float1" | "half")),
        AttachmentFormat::R32Uint => matches!(type_name, Some("uint" | "uint1")),
    }
}

/// The entry point name one translated module declares for `model`, or `None`
/// when the module does not declare exactly one entry point of that stage.
///
/// The translator emits `OpEntryPoint … "main"` for every stage and keeps the
/// AIR function's own name in the reflection, so the rail reads the name the
/// pipeline has to bind out of the module itself instead of repeating the
/// translator's literal: a module whose entry point is renamed still binds, and a
/// module that declares two entry points of one stage (or none) is refused at
/// registration, because the pipeline would have to pick one.
fn module_entry_point(module: &[u8], model: spirv::ExecutionModel) -> Option<String> {
    let words = spirv_words(module)?;
    let mut found = None;
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let word_count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = cursor
            .checked_add(word_count)
            .filter(|end| word_count != 0 && *end <= words.len())?;
        if opcode == spirv::Op::EntryPoint as u32 {
            // `OpEntryPoint`: ExecutionModel, EntryPoint <id>, then the entry
            // name as a NUL-terminated literal in the instruction's words. A
            // shorter instruction cannot carry a name, so a module that ships
            // one is refused rather than indexed past its own words.
            if word_count < 4 {
                return None;
            }
            if words[cursor + 1] == model as u32 {
                if found.is_some() {
                    return None;
                }
                let mut name = Vec::new();
                for word in &words[cursor + 3..end] {
                    name.extend_from_slice(&word.to_le_bytes());
                }
                let end_of_name = name.iter().position(|byte| *byte == 0)?;
                found = Some(String::from_utf8(name[..end_of_name].to_vec()).ok()?);
            }
        }
        cursor = end;
    }
    found
}

/// The stage name a reflection reports, spelled as this module spells stages.
fn reflected_stage_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        _ => "kernel",
    }
}

/// The refusal for a translation that does not describe the pipeline it is
/// registered under.
///
/// A capability fact: the rail will not execute a module whose stated interface
/// disagrees with the contract, because the disagreement is exactly which bytes
/// the pipeline would land.
fn reflection_mismatch_refusal(stage: RenderStage, entry: &str) -> ProviderError {
    capability_refusal("render_stage_reflection_mismatch")
        .with_field("stage", FieldValue::Text(stage.name().to_owned()))
        .with_field("entry", FieldValue::Text(entry.to_owned()))
}

/// The refusal for a translated stage that names interface this rail does not
/// execute yet.
fn unsupported_interface_refusal(
    stage: RenderStage,
    entry: &str,
    field: &'static str,
) -> ProviderError {
    capability_refusal("render_stage_unsupported_interface")
        .with_field("stage", FieldValue::Text(stage.name().to_owned()))
        .with_field("entry", FieldValue::Text(entry.to_owned()))
        .with_field("field", FieldValue::Text(field.to_owned()))
        .with_detail(
            "the translation names Metal interface this rail does not execute yet, so the stage \
             is refused instead of being executed with that interface silently dropped",
        )
}

/// The refusal for a stage module the rail cannot account for: neither one of
/// its reviewed modules nor a translated module under a reflection.
fn render_stage_translation_unavailable_refusal(stage: RenderStage, entry: &str) -> ProviderError {
    capability_refusal("render_stage_translation_unavailable")
        .with_field("stage", FieldValue::Text(stage.name().to_owned()))
        .with_field("entry", FieldValue::Text(entry.to_owned()))
        .with_detail(
            "the stage module is neither one of this rail's reviewed modules nor a translated \
             module described by its reflection, so the rail has no semantics it could execute",
        )
}

/// The refusal for a module whose SPIR-V demands a capability this device did
/// not enable (R8).
///
/// The detail carries the capability gate's own sentence, so the refusal a
/// registration or an execution reports reads exactly like the one the
/// translation reports for the same module on a device without the feature.
fn render_stage_capability_refusal(stage: RenderStage, entry: &str) -> ProviderError {
    capability_refusal("render_stage_capability_unavailable")
        .with_field("stage", FieldValue::Text(stage.name().to_owned()))
        .with_field("entry", FieldValue::Text(entry.to_owned()))
}

/// The entry point name one stage's module declares, as the pipeline binds it.
///
/// A reviewed module declares the entry the contract names; a translated module
/// declares the translator's own entry point, so the contract names the AIR
/// function and this is where the two are told apart.
fn bound_stage_entry<'a>(
    stages: &'a RenderStages,
    stage: RenderStage,
) -> Result<Cow<'a, str>, ProviderError> {
    let (translated, entry, module) = match stage {
        RenderStage::Vertex => (
            stages.vertex_translation.is_some(),
            stages.contract.vertex_entry.as_str(),
            stages.vertex_spirv.as_slice(),
        ),
        RenderStage::Fragment => (
            stages.fragment_translation.is_some(),
            stages.contract.fragment_entry.as_str(),
            stages.fragment_spirv.as_slice(),
        ),
    };
    if !translated {
        return Ok(Cow::Borrowed(entry));
    }
    module_entry_point(module, stage.execution_model())
        .map(Cow::Owned)
        .ok_or_else(|| render_stage_translation_unavailable_refusal(stage, entry))
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
            FieldValue::Text(
                if formats.is_empty() {
                    DEPTH_ONLY_FRAGMENT_ENTRY
                } else {
                    SOLID_FRAGMENT_ENTRY
                }
                .to_owned(),
            ),
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
/// attachment's tightly packed texel bytes and `None` for each discarded one,
/// plus the stored depth surface's own texels when the pass keeps it
/// (`research/docs/23` §3.3, v43).
///
/// This is the trace-side entry point of the rail: the pass's shape rules were
/// already checked by core admission, so what is left here is the agreement
/// between the pass and the registered pipeline it names
/// ([`RenderPipelineContract::validate_against`]) and the shapes this increment
/// cannot execute — a pass whose attachment list or format combination is
/// outside the reviewed set, and a `Load` whose declaring view carries no
/// readable contents. All of them are refused as capability facts before any
/// Vulkan object exists, never downgraded to a clear. The registered fragment
/// stage is re-checked against the pipeline's declared format list in the same
/// place, for the same reason: the pass is about to be executed with it.
///
/// `previous` carries one entry per colour attachment, in location order: the
/// view the trace declares for that attachment when the pass opens it with
/// `LoadOp::Load`, and `None` for every other load operation. The rail resolves
/// that declaration into the bytes it uploads (`research/docs/23` §74, R5b), so
/// the caller hands over the declaration rather than a snapshot of it.
///
/// `resident` carries the same arity again: the provider-owned image of each
/// attachment that declares [`LoadOp::Resident`] or [`StoreOp::Resident`], and
/// `None` for every attachment that does not (`research/docs/23` §76, R7). The
/// provider owns the registry that decides which identities are resident, so
/// the rail only checks that the two sides agree — a declaration without an
/// image, or an image without a declaration, is refused by name rather than
/// rendered into a per-pass image the trace did not ask for.
/// `resident` is either empty — the shape every pre-R7 caller hands over, which
/// means "this pass declares no resident target" — or one entry per colour
/// attachment, exactly as `previous` is (`research/docs/23` §76, R7).
pub(crate) fn execute_render_pass<'a>(
    context: &VulkanContext,
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
    previous: &'a [Option<&'a BufferView>],
    resident: &[Option<&'a ProviderTargetImage>],
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<OffscreenReadback, ProviderError> {
    refuse_attachment_extent(context, pass)?;
    let request = prepare_render_request_with_resident(
        stages,
        pass,
        previous,
        if resident.is_empty() {
            None
        } else {
            Some(resident)
        },
        leases,
        context.admitted_depth_resolve_modes(),
        context.admitted_stencil_resolve_modes(),
        context.spirv_feature_policy(),
    )?;
    let retains = RenderInputRetains::retain(leases, &request)?;
    execute_offscreen_render_with_retains(context, &request, retains)
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
    previous: &'a [Option<&'a BufferView>],
    leases: Option<&RenderLeaseContext<'_>>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
    policy: SpirvFeaturePolicy,
) -> Result<OffscreenRenderRequest<'a>, ProviderError> {
    // The pre-R7 shape every non-resident caller states: the pass declares no
    // provider-resident target, so the general form below is handed no
    // resident list at all and refuses a pass that declares one
    // (`research/docs/23` §76, R7).
    prepare_render_request_with_resident(
        stages,
        pass,
        previous,
        None,
        leases,
        depth_resolve_modes,
        stencil_resolve_modes,
        policy,
    )
}

/// [`prepare_render_request`] with the resident-target declarations of a pass
/// that names them (`research/docs/23` §76, R7).
///
/// `resident` is `None` for a pass that declares no resident target, and
/// otherwise carries one entry per colour attachment, exactly as `previous`
/// does.
// The parameter list is the pass's own declaration surface: attachments,
// their previous contents, their resident targets, the lease context, the two
// admitted resolve-mode masks, and the device's SPIR-V policy. The blank
// wrapper above already models the non-resident shape; splitting this into a
// struct would just move the same seven fields somewhere else (R7 + R8).
#[allow(clippy::too_many_arguments)]
fn prepare_render_request_with_resident<'a>(
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
    previous: &'a [Option<&'a BufferView>],
    resident: Option<&[Option<&'a ProviderTargetImage>]>,
    leases: Option<&RenderLeaseContext<'_>>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
    policy: SpirvFeaturePolicy,
) -> Result<OffscreenRenderRequest<'a>, ProviderError> {
    // The device's capability subset is asked first (R8), in the order the
    // translation entry point asks it: a module this device could not create is
    // refused by name before the rail asks what the module or the pass is.
    validate_module_capabilities(stages, policy)?;
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
            "the previous-contents list must carry one entry per colour attachment",
        ));
    }
    // The resident list is the previous-contents list's sibling
    // (`research/docs/23` §76, R7): one entry per colour attachment, and the
    // two declarations have to agree in both directions. A pass that declares
    // the resident target without an image would be executed as a clear over a
    // fresh per-pass image — the silent downgrade the arm exists to prevent —
    // and an image handed over for an attachment that declares no residency
    // would render the provider's bytes where the trace asked for a clear.
    if let Some(resident) = resident {
        if resident.len() != pass.color_attachments.len() {
            return Err(contract_refusal(
                "the resident-target list must carry one entry per colour attachment",
            ));
        }
    }
    let resident_of = |index: usize| -> Option<&'a ProviderTargetImage> {
        resident.and_then(|list| list.get(index).copied().flatten())
    };
    // The registration gate is re-asked of the value the rail was handed, so a
    // directly-constructed `RenderStages` cannot skip it: the same two arms that
    // settled the pairing at registration decide here, whether a stage arrived
    // as a reviewed module or as a translation.
    stages.validate_stage_pair()?;
    // The reviewed pair module is executed only by the fixtures whose state the
    // review covers: the depth fixture states a depth attachment and a test,
    // the stencil fixture a stencil attachment and a test, the cull fixture a
    // culling state, and the blend fixture a blend state. A pass that states
    // none of them is a shape no review covered
    // (`research/docs/23` §3.3, v36/v39/v40/v47).
    if vertex_stage_is_depth(&stages.contract.vertex_entry, &stages.vertex_spirv)
        && (pass.depth.is_none() || pass.depth_test.is_none())
        && pass.stencil.is_none()
        && pass.cull.is_none()
        && pass.blend.is_none()
    {
        return Err(
            capability_refusal("render_depth_state_unsupported").with_detail(
                "the reviewed pair module is only executed by the depth, culling and blending \
             fixtures; a pass that states none of those is a shape no review covered",
            ),
        );
    }
    // The depth resolve (`research/docs/23` §3.3, v57) only means something
    // beside a multisample raster that keeps its depth surface: the resolve is
    // the reduction of the stored four-sample texels, so a resolve without
    // both is refused instead of silently ignored. The core contract refuses
    // the same shape with `DepthResolveWithoutStoredDepth`; this is the
    // value-level second line of defence for a directly-constructed request.
    if pass.depth_resolve.is_some() {
        let stored = pass.multisample.is_some()
            && pass
                .depth
                .as_ref()
                .is_some_and(|depth| depth.store == Some(DepthStoreOp::Store));
        if !stored {
            let store = pass
                .depth
                .as_ref()
                .and_then(|depth| depth.store)
                .map(DepthStoreOp::code);
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::DepthResolveWithoutStoredDepth { store }
                    .to_string(),
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
    if let Some(resolve) = pass.stencil_resolve {
        let stored = pass.multisample.is_some()
            && pass
                .stencil
                .as_ref()
                .is_some_and(|stencil| stencil.store == Some(StoreOp::Store));
        if !stored {
            let store = pass.stencil.as_ref().and_then(|stencil| stencil.store);
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::StencilResolveWithoutStoredStencil {
                    store,
                }
                .to_string(),
            ));
        }
        if resolve.filter == StencilResolveFilter::DepthResolvedSample
            && pass.depth_resolve.is_none()
        {
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::StencilResolveWithoutDepthResolve
                    .to_string(),
            ));
        }
        // The Vulkan rail has no stencil mode for Metal's
        // `DepthResolvedSample`, so the only admitted filter is Sample0 and the
        // per-filter question the capability snapshot answered refuses the
        // rest here for a directly-constructed request
        // (`research/docs/23` §3.3, v60).
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
    // The multisample raster (`research/docs/23` §3.3, v51) is executed as a
    // four-sample pass whose resolve target is the attachment view itself. Core
    // admission already holds the shape (`RenderPassDescriptor::validate`);
    // the rail re-asserts the two facts its execution depends on, so a
    // directly-constructed request cannot reach `vkCreateImage` with a shape
    // the rail would silently narrow: the state names a multisampled count, and
    // every attachment opens from a clear. Uploading single-sample previous
    // bytes into a multisampled image is the load increment this one does not
    // review, and a depth or stencil surface or a present action beside the
    // raster is refused at the contract before this point.
    if let Some(multisample) = pass.multisample {
        if multisample.sample_count == SampleCount::One {
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::SingleSampleMultisampleState.to_string(),
            ));
        }
        // The attachment's load (`research/docs/23` §3.3, v51/v67): a clear or
        // a `dontcare` load is admitted from v67 on — `dontcare` opens the
        // multisampled image from undefined contents exactly as its
        // single-sample sibling does, and the resolve then lands whatever the
        // samples hold, which is the shape the constrained wildcard
        // expectation states. A `load` still needs previous bytes on a
        // multisampled image, which is the load increment this rail does not
        // execute, and the per-attachment loop below refuses it by name.
        // The present action beside the raster (`research/docs/24` §3.5, v62)
        // is executed by `execute_present_render`: the pass renders into a
        // rail-owned n-sample surface and resolves into the provider-owned
        // present target, whose single-sample texels are what the present
        // hands on. The contract's own rule holds the present source to one
        // of the pass's colour attachment views — the resolve landing — and
        // the present rail's single-attachment gate below keeps that view
        // unique, so no second rail-side refusal is needed here.
        // The depth surface beside the raster is admitted from v53 on
        // (`research/docs/23` §3.3, v53) and the stencil surface from v55, both
        // created with the pass's own sample count. Keeping either surface's
        // texels is admitted from v57/v60 through the resolve the pass then
        // has to state. A pass that opens both surfaces together is the
        // combined depth-stencil shape: one attachment both faces share, opened
        // from the reviewed clears. Two shapes of it are executed — the v60
        // resolve shape, which stores both faces through their resolves, and
        // the v66 rail-owned pair, which keeps neither face and observes the
        // colour resolve — so the two faces' store decisions have to agree
        // exactly as the contract states them.
        if let (Some(depth), Some(stencil)) = (&pass.depth, &pass.stencil) {
            if !matches!(depth.load, DepthLoadOp::Clear(_))
                || !matches!(stencil.load, StencilLoadOp::Clear(_))
            {
                return Err(
                    capability_refusal("render_combined_depth_stencil_load_unsupported")
                        .with_detail(
                            "the combined depth-stencil shape opens both faces from a clear: \
                             a loading combined surface is a later increment",
                        ),
                );
            }
            let rail_owned = !depth.is_stored() && !stencil.is_stored();
            let resolved = depth.is_stored()
                && stencil.is_stored()
                && pass.depth_resolve.is_some()
                && pass.stencil_resolve.is_some();
            if !rail_owned && !resolved {
                return Err(
                    capability_refusal("render_combined_depth_stencil_store_unsupported")
                        .with_detail(
                            "the combined depth-stencil shape keeps both faces or neither: \
                             both rail-owned with no resolve, or both stored through their \
                             two resolves, which are the only combined shapes this rail \
                             executes",
                        ),
                );
            }
        }
        if let Some(depth) = &pass.depth {
            if depth.store == Some(DepthStoreOp::Store) {
                // The stored surface is admitted from v57 on, through the
                // resolve the pass then has to state: its texels are only
                // observable as the resolve's reduction, so a stored surface
                // without one stays refused, and a filter the device does not
                // report is refused by the same per-filter question the
                // capability snapshot answered (`research/docs/23` §3.3, v57).
                match pass.depth_resolve {
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
                            "a multisampled depth surface cannot be kept without a \
                                     depth resolve",
                        ))
                    }
                }
            }
        }
        if let Some(stencil) = &pass.stencil {
            if stencil.store == Some(StoreOp::Store) {
                // The stored surface is admitted from v60 on, through the
                // resolve the pass then has to state: its texels are only
                // observable as the resolve's reduction, so a stored surface
                // without one stays refused, and a filter the device does not
                // report is refused by the same per-filter question the
                // capability snapshot answered (`research/docs/23` §3.3, v60).
                match pass.stencil_resolve {
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
                            "a multisampled stencil surface cannot be kept without a stencil \
                             resolve",
                        ))
                    }
                }
            }
        }
    }
    let mut attachments = Vec::with_capacity(pass.color_attachments.len());
    let mut extent: Option<[u32; 2]> = None;
    for (index, (attachment, declared)) in pass.color_attachments.iter().zip(previous).enumerate() {
        let resident = resident_of(index);
        // The two declarations have to agree in both directions
        // (`research/docs/23` §76, R7). A pass that declares the resident
        // target and hands over no image would be executed as a per-pass
        // attachment image — a clear where the trace asked for the provider's
        // own bytes — and an image handed over for an attachment that declares
        // no residency would render the provider's image where the trace
        // declared a per-pass one. Both are refused here, before any device
        // object exists, rather than resolved into "whichever image came
        // first".
        let declares_resident = attachment.declares_resident_target();
        match (declares_resident, resident.is_some()) {
            (true, false) => {
                return Err(capability_refusal("resident_target_undeclared")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_detail(
                        "the pass declares the provider-resident target and the provider handed \
                         over no image for it",
                    ));
            }
            (false, true) => {
                return Err(capability_refusal("resident_target_undeclared")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_detail(
                        "the provider holds a resident target for this attachment and the pass \
                         declares neither a resident load nor a resident store",
                    ));
            }
            _ => {}
        }
        // The attachment's previous contents come from the view the trace
        // declares for it, exactly as a stream's bytes come from their own
        // declaration: the source is resolved here, before any device object
        // exists, and a `Load` whose declaration cannot be read is refused
        // rather than silently executed as a clear (`research/docs/23`
        // §3.3/§74, R5b).
        let previous = match (attachment.load, declared) {
            (LoadOp::Load, Some(view)) => Some(resolve_attachment_load(
                view,
                leases,
                index,
                attachment
                    .expected_bytes()
                    .map_err(|error| contract_refusal(&error.to_string()))?,
            )?),
            _ => None,
        };
        match attachment.load {
            LoadOp::Clear(_) => {}
            LoadOp::Resident => {
                // The attachment's previous contents are the provider image's
                // own (`research/docs/23` §76, R7): there is nothing to
                // resolve, nothing to upload, and the render pass opens the
                // borrowed image from the layout the provider published. A
                // caller that also declared bytes for this attachment is
                // refused rather than having one of the two sources silently
                // win — the trace named both, so the rail cannot know which
                // one it meant.
                if declared.is_some() {
                    return Err(capability_refusal("resident_target_undeclared")
                        .with_field("attachment", FieldValue::Unsigned(index as u64))
                        .with_field("load_op", FieldValue::Text("resident".to_owned()))
                        .with_detail(
                            "a `LoadOp::Resident` attachment keeps the provider image's own \
                             contents; the trace declares no previous bytes for it",
                        ));
                }
            }
            LoadOp::Load => {
                // The rail uploads the attachment's previous bytes before
                // opening the render pass (`research/docs/23` §3.3). The caller
                // resolves them from the trace's own declaration, so a `Load`
                // that carries no declaration is refused rather than silently
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
                if declared.is_some() {
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
        // The multisampled load's seed (`research/docs/23` §82, v82): a
        // multisampled image cannot take its previous bytes from a buffer
        // copy, so the rail's load route is a seed pass whose clear value is
        // host state. The declaration therefore has to be one repeated texel
        // read from a window this submission owns; both deviations are refused
        // by name here, before any device object exists.
        let seed = match previous.as_ref() {
            Some(source)
                if pass
                    .multisample
                    .is_some_and(|state| state.sample_count != SampleCount::One) =>
            {
                Some(resolve_multisample_seed(source, index, attachment.format)?)
            }
            _ => None,
        };
        attachments.push(OffscreenColorAttachment {
            format: attachment.format,
            store: attachment.store,
            load: attachment.load,
            previous,
            resident,
            seed,
        });
    }
    // A pass with no colour attachment takes its extent from the depth
    // attachment it renders into: the depth surface is the whole raster
    // (`research/docs/23` §3.3, v46), and its extent is what the pipeline's
    // viewport and the readback use.
    let extent = match (extent, pass.depth.as_ref()) {
        (Some(extent), _) => extent,
        (None, Some(depth)) => {
            let width = narrow_dimension(depth.width)?;
            let height = narrow_dimension(depth.height)?;
            if width == 0 || height == 0 {
                return Err(contract_refusal("depth attachment has a zero dimension"));
            }
            [width, height]
        }
        (None, None) => {
            return Err(contract_refusal(
                "a render pass opens a colour attachment or a depth attachment",
            ))
        }
    };
    // Vertex input (`research/docs/23` §3.3): every bound stream declares its
    // own bytes, so the rail proves the footprint the draw reads and refuses
    // anything the reviewed shape does not cover.
    // The render sampler (`research/docs/23` §3.3, v70) is the same kind of
    // question one dimension up: the reviewed fragment stage samples exactly
    // one `rgba8_unorm` surface whose extent matches the render area, so a
    // fragment standing on a texel centre reads that texel's own bytes rather
    // than a filtered or boundary-rule-dependent neighbour.
    let textures = resolve_render_textures(pass, extent, leases)?;
    let streams = resolve_vertex_streams(stages, pass, leases)?;
    // A per-instance stream's record count is the draw's instance count, so its
    // footprint is proved against that count instead of the vertex span
    // (`research/docs/23` §3.3, v31).
    for (index, stream) in streams.iter().enumerate() {
        if stream.layout.step != VertexStep::PerInstance {
            continue;
        }
        let required = u64::from(pass.instance_count)
            .checked_mul(stream.layout.stride)
            .ok_or_else(|| contract_refusal("vertex buffer footprint overflows u64"))?;
        if stream.view.length < required {
            return Err(
                capability_refusal("render_vertex_buffer_footprint_unsupported")
                    .with_field("binding", FieldValue::Unsigned(index as u64))
                    .with_field("step", FieldValue::Text("per_instance".to_owned()))
                    .with_field(
                        "instance_count",
                        FieldValue::Unsigned(u64::from(pass.instance_count)),
                    )
                    .with_field("required_bytes", FieldValue::Unsigned(required))
                    .with_field("declared_bytes", FieldValue::Unsigned(stream.view.length))
                    .with_detail("a per-instance stream has to cover one record per instance"),
            );
        }
    }
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
            let index_source = resolve_render_input(view, leases, RenderInputRole::Index, 0)?;
            let index_values = decode_indices(&index_source, indices.format, pass.vertices)?;
            for stream in &streams {
                // A per-instance stream is proved against the instance count,
                // not the index span (`research/docs/23` §3.3, v31).
                if stream.layout.step == VertexStep::PerInstance {
                    continue;
                }
                let vertex_capacity = stream.view.length / stream.layout.stride;
                // The vertex a draw reads is `base_vertex + index`, so the
                // offset takes part in the proof (`research/docs/23` §3.3,
                // v34): an index that fits on its own can still reach past the
                // stream once the offset is added.
                if let Some(index) = index_values.iter().find(|index| {
                    u64::from(**index) + u64::from(pass.base_vertex) >= vertex_capacity
                }) {
                    return Err(
                        capability_refusal("render_vertex_buffer_footprint_unsupported")
                            .with_field("index", FieldValue::Unsigned(u64::from(*index)))
                            .with_field(
                                "base_vertex",
                                FieldValue::Unsigned(u64::from(pass.base_vertex)),
                            )
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
                    source: index_source,
                }),
            )
        }
        None => {
            for stream in &streams {
                if stream.layout.step == VertexStep::PerInstance {
                    continue;
                }
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
    let depth = pass.depth.as_ref().map(|depth| OffscreenDepthAttachment {
        width: u32::try_from(depth.width).unwrap_or(u32::MAX),
        height: u32::try_from(depth.height).unwrap_or(u32::MAX),
        clear: depth.load.clear_depth(),
        test: pass.depth_test,
        store: depth.store,
    });
    // The stencil attachment is rail-owned (`research/docs/23` §3.3, v47): the
    // rail creates the surface, clears it with the value the trace states and
    // arms the draw with the state the pass declares. A trace that keeps the
    // surface states its store action too, and the rail reads the texels back
    // through its own copy-out (v49).
    let stencil = pass
        .stencil
        .as_ref()
        .map(|stencil| OffscreenStencilAttachment {
            width: u32::try_from(stencil.width).unwrap_or(u32::MAX),
            height: u32::try_from(stencil.height).unwrap_or(u32::MAX),
            clear: stencil.load.clear_value(),
            test: pass.stencil_test,
            store: stencil.store,
        });
    let request = OffscreenRenderRequest {
        attachments,
        depth,
        stencil,
        // The sampled textures travel with the request exactly as the trace
        // stated them (`research/docs/23` §3.3, v70); the extent and format
        // refusals above already ran.
        textures,
        // The pass-wide raster decision travels with the request exactly as
        // the trace stated it (`research/docs/23` §3.3, v51); the load-op and
        // surface refusals above already ran.
        multisample: pass.multisample,
        // The resolve filter travels with the request exactly as the trace
        // stated it (`research/docs/23` §3.3, v57); the shape and per-filter
        // refusals above already ran.
        depth_resolve: pass.depth_resolve,
        // The stencil resolve filter travels with the request exactly as the
        // trace stated it (`research/docs/23` §3.3, v60); the shape and
        // per-filter refusals above already ran.
        stencil_resolve: pass.stencil_resolve,
        scissor: pass.scissor,
        instance_count: pass.instance_count,
        base_vertex: pass.base_vertex,
        // The culling and blend states belong to the pipeline the rail builds
        // for this pass (`research/docs/23` §3.3, v39/v40).
        cull: pass.cull,
        blend: pass.blend.clone(),
        extent,
        vertex: OffscreenVertexStage {
            entry: bound_stage_entry(stages, RenderStage::Vertex)?,
            spirv: &stages.vertex_spirv,
        },
        translated_fragment: match stages.fragment_translation {
            Some(_) => Some(OffscreenFragmentStage {
                entry: bound_stage_entry(stages, RenderStage::Fragment)?.into_owned(),
                spirv: &stages.fragment_spirv,
            }),
            None => None,
        },
        vertex_streams: streams,
        draw,
        index_stream,
        indirect: None,
    };
    Ok(request)
}

/// Resolve one pass's sampled textures into the rail's own request shape
/// (`research/docs/23` §3.3, v70).
///
/// The contract already holds the binding label, the read-only access and the
/// single-sample requirement (`RenderPassDescriptor::validate`); this is the
/// rail's own window, restated for a directly-constructed pass and narrowed to
/// what the reviewed sampling module covers: one `rgba8_unorm` 2D surface,
/// whose extent equals the render area. The extent rule is what keeps the
/// fixture's expectation driver-independent — a texture of another size puts
/// some fragment's `(column + 0.5) / width` sample either on a texel boundary
/// or inside a neighbour, which is a filtered read the review never covered, so
/// the pass is refused by name instead of sampled.
///
/// The texture's bytes are resolved through the same three-arm channel the
/// streams and the loading attachments use (`research/docs/23` §75, R5c): the
/// trace's own bytes, the provider's staged copy of an owner lease, or the
/// owner's own mapping. The resolution runs before the first device object
/// exists, and the window a lease resolves to has to be the texture's own
/// tightly packed extent — the shape the trace-owned arm's `validate_shape`
/// already holds.
fn resolve_render_textures<'a>(
    pass: &'a RenderPassDescriptor,
    extent: [u32; 2],
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<Vec<OffscreenRenderTexture<'a>>, ProviderError> {
    if pass.textures.len() > MAX_RENDER_TEXTURES {
        return Err(capability_refusal("render_texture_limit")
            .with_field(
                "requested",
                FieldValue::Unsigned(pass.textures.len() as u64),
            )
            .with_field("maximum", FieldValue::Unsigned(MAX_RENDER_TEXTURES as u64)));
    }
    let mut textures = Vec::with_capacity(pass.textures.len());
    for (index, view) in pass.textures.iter().enumerate() {
        if view.format != TextureFormat::Rgba8Unorm {
            return Err(capability_refusal("render_texture_format_unsupported")
                .with_field("binding", FieldValue::Unsigned(index as u64))
                .with_field("format", FieldValue::Text(format!("{:?}", view.format)))
                .with_detail("the reviewed sampling module reads one rgba8_unorm surface"));
        }
        if view.texture_type != TextureType::D2
            || view.sample_count != 1
            || view.depth != 1
            || view.array_length != 1
        {
            return Err(capability_refusal("render_texture_shape_unsupported")
                .with_field("binding", FieldValue::Unsigned(index as u64))
                .with_field(
                    "texture_type",
                    FieldValue::Text(format!("{:?}", view.texture_type)),
                )
                .with_field("sample_count", FieldValue::Unsigned(view.sample_count))
                .with_field("depth", FieldValue::Unsigned(view.depth))
                .with_field("array_length", FieldValue::Unsigned(view.array_length))
                .with_detail("the reviewed sampling module reads a single-sample 2D surface"));
        }
        let source = resolve_render_texture_source(view, leases, index)?;
        let width = narrow_dimension(view.width)?;
        let height = narrow_dimension(view.height)?;
        if width == 0 || height == 0 {
            return Err(contract_refusal("render texture has a zero dimension"));
        }
        if [width, height] != extent {
            return Err(capability_refusal("render_texture_extent_unsupported")
                .with_field("binding", FieldValue::Unsigned(index as u64))
                .with_field("width", FieldValue::Unsigned(u64::from(width)))
                .with_field("height", FieldValue::Unsigned(u64::from(height)))
                .with_field("render_width", FieldValue::Unsigned(u64::from(extent[0])))
                .with_field("render_height", FieldValue::Unsigned(u64::from(extent[1])))
                .with_detail(
                    "the reviewed sampling shape samples a texture of the render area's own \
                     extent, so every fragment stands on a texel centre",
                ));
        }
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|texels| texels.checked_mul(view.format.bytes_per_texel()))
            .ok_or_else(|| contract_refusal("render texture bytes overflow u64"))?;
        if u64::try_from(source.len()).unwrap_or(u64::MAX) != expected {
            return Err(contract_refusal(&format!(
                "render texture {index} resolves {} bytes for a {width}x{height} surface",
                source.len()
            )));
        }
        textures.push(OffscreenRenderTexture {
            source,
            extent: [width, height],
        });
    }
    Ok(textures)
}

/// The owner-issued lease material one render submission resolves its render
/// inputs from (`research/docs/23` §71, R3c).
///
/// The compute rail already owns both registries; the render rail shares them
/// instead of keeping a second copy, so one import per device epoch serves both
/// rails and both agree on when the owner's backing may be released.
/// `resources` is the admitted snapshot and is authoritative for every
/// reservation, exactly as it is on the compute side. `host_import_alignment`
/// is the device's own `VK_EXT_external_memory_host` alignment, and zero when
/// the device cannot import host memory at all.
pub(crate) struct RenderLeaseContext<'a> {
    pub(crate) staging: &'a LeaseRegistry,
    pub(crate) borrowed: &'a Arc<BorrowedLeaseRegistry>,
    pub(crate) resources: &'a ResourceTableSnapshot,
    pub(crate) device_epoch: DeviceEpoch,
    pub(crate) host_import_alignment: u64,
}

/// Which of a pass's render inputs a refusal is about.
#[derive(Clone, Copy)]
enum RenderInputRole {
    /// A vertex stream of the pass's layout.
    Vertex,
    /// The pass's index buffer.
    Index,
    /// The previous contents a `LoadOp::Load` attachment uploads
    /// (`research/docs/23` §74, R5b).
    Attachment,
    /// A sampled texture the pass's fragment stage reads
    /// (`research/docs/23` §75, R5c).
    Texture,
}

impl RenderInputRole {
    /// The capability slug this role's unreadable source is refused with. The
    /// two stream names are the ones this rail published before the lease
    /// channel existed, so a capture that could not read a stream keeps its
    /// slug; the attachment name is the source-arm sibling the sampler
    /// publishes (`render_texture_source_unsupported`), because what could not
    /// be read is the attachment's own prior contents rather than a load
    /// operation the contract refuses, and the texture role keeps that sampler
    /// name itself (`research/docs/23` §75, R5c).
    const fn slug(self) -> &'static str {
        match self {
            Self::Vertex => "render_vertex_buffer_unsupported",
            Self::Index => "render_index_buffer_unsupported",
            Self::Attachment => "render_attachment_load_source_unsupported",
            Self::Texture => "render_texture_source_unsupported",
        }
    }

    /// The field this role's own slot is reported under: a stream or an index
    /// buffer is a *binding* of the pipeline's layout, a texture is the
    /// *binding* its view's own label holds it to (`docs/23` §3.3, v70), while
    /// an attachment's previous contents are named by their location.
    const fn slot_field(self) -> &'static str {
        match self {
            Self::Vertex | Self::Index | Self::Texture => "binding",
            Self::Attachment => "attachment",
        }
    }
}

/// Where one render input's bytes come from (`research/docs/23` §71/§74,
/// R3c/R5b).
///
/// One type serves every render input the rail reads from a declaration: the
/// bytes a vertex stream or index buffer carries, and the previous contents a
/// `LoadOp::Load` attachment uploads. The three arms are the three
/// [`BufferSource`] arms, resolved before any device object exists.
#[derive(Debug)]
pub(crate) enum RenderInputSource<'a> {
    /// The trace's own bytes (`BufferSource::OwnedBytes`), unchanged from the
    /// pre-lease increments.
    TraceBytes(&'a [u8]),
    /// The provider's staged copy of an owner lease
    /// (`BufferSource::StagedLease`); the rail uploads it like trace bytes.
    StagedBytes(Vec<u8>),
    /// The owner's own mapping (`BufferSource::BorrowedNoCopy`): the rail
    /// imports this window instead of copying it, and holds a retain on the
    /// lease until the pass's fence has signalled.
    Borrowed {
        lease: LeaseId,
        window: BorrowedView,
    },
}

impl RenderInputSource<'_> {
    /// The bytes the rail reads out of this source.
    ///
    /// A borrowed window is measured at the owner's mapping — the same window
    /// the import binds, so a length the attachment's extent disagrees with is
    /// refused rather than copied from out of range.
    fn len(&self) -> usize {
        match self {
            Self::TraceBytes(bytes) => bytes.len(),
            Self::StagedBytes(bytes) => bytes.len(),
            Self::Borrowed { window, .. } => window.len,
        }
    }

    /// The bytes the rail's footprint proof reads.
    ///
    /// A borrowed window is read through the owner's mapping, because that is
    /// where the proof's bytes are: the import contract keeps the mapping valid
    /// at this address until the provider releases the import, so this reads
    /// the same bytes the device will read rather than copying them into
    /// provider-owned storage. Trace-owned and staged bytes are read from the
    /// window the rail is about to upload.
    fn proof_bytes(&self) -> &[u8] {
        match self {
            Self::TraceBytes(bytes) => bytes,
            Self::StagedBytes(bytes) => bytes,
            // SAFETY: the window was resolved by the no-copy registry for an
            // imported lease, whose contract keeps the mapping readable over
            // exactly this window until the import is released.
            Self::Borrowed { window, .. } => unsafe {
                std::slice::from_raw_parts(window.pointer as *const u8, window.len)
            },
        }
    }

    /// The no-copy lease this source reads, when it is one.
    const fn borrowed_lease(&self) -> Option<LeaseId> {
        match self {
            Self::TraceBytes(_) | Self::StagedBytes(_) => None,
            Self::Borrowed { lease, .. } => Some(*lease),
        }
    }
}

/// Resolve one render input's source into the window the rail reads
/// (`research/docs/23` §71/§74, R3c/R5b).
///
/// `OwnedBytes` resolves to the trace's own bytes exactly as before. A
/// `StagedLease` resolves through the provider's staged registry, which holds
/// the owner's staged copy; the rail uploads those bytes into a buffer of its
/// own. A `BorrowedNoCopy` resolves through the no-copy registry, which hands
/// back the owner's address and never copies. Every unresolvable arm is refused
/// by name before any device object exists: a lease that was never imported or
/// admitted, a reservation that does not cover the view, a device that cannot
/// import host memory, and a pointer that misses the import alignment. `role`
/// decides which of the three inputs the refusal names — a vertex stream, the
/// index buffer, or a loading attachment's own prior contents — and `slot` is
/// that input's own binding or attachment location.
fn resolve_render_input<'a>(
    view: &'a BufferView,
    leases: Option<&RenderLeaseContext<'_>>,
    role: RenderInputRole,
    slot: usize,
) -> Result<RenderInputSource<'a>, ProviderError> {
    match &view.source {
        BufferSource::OwnedBytes(bytes) => Ok(RenderInputSource::TraceBytes(bytes)),
        BufferSource::StagedLease(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    role,
                    slot,
                    "staged_lease",
                    "the render submission carries no lease channel, so a lease-backed render \
                     input cannot be read",
                )
            })?;
            let bytes = leases.staging.view_bytes(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            Ok(RenderInputSource::StagedBytes(bytes))
        }
        BufferSource::BorrowedNoCopy(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    role,
                    slot,
                    "borrowed_no_copy",
                    "the render submission carries no lease channel, so a lease-backed render \
                     input cannot be read",
                )
            })?;
            if leases.host_import_alignment == 0 {
                return Err(host_import_refusal(role, slot, view.view_id));
            }
            let window = leases.borrowed.view_pointer(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            let alignment = usize::try_from(leases.host_import_alignment).unwrap_or(usize::MAX);
            if !window.pointer.is_multiple_of(alignment) {
                return Err(lease_alignment_refusal(
                    *lease_id,
                    window.pointer,
                    leases.host_import_alignment,
                    role,
                    slot,
                ));
            }
            Ok(RenderInputSource::Borrowed {
                lease: *lease_id,
                window,
            })
        }
    }
}

/// Resolve one loading attachment's previous contents into the window the rail
/// uploads into the image (`research/docs/23` §74, R5b).
///
/// The declaring view is the only channel that carries an attachment's previous
/// bytes — the attachment restates the view's identity and shape and names no
/// contents (`docs/23` §3.3) — so the three arms are exactly the ones
/// [`resolve_render_input`] decides for a stream, one role wider: the
/// trace-owned bytes, the provider's staged copy of an owner lease, or the
/// owner's own mapping. The window a `Load` uploads has to be the attachment's
/// own tightly packed extent, which is the length `BufferView::validate_shape`
/// already holds a trace-owned declaration to; a lease that resolves to a
/// different length is refused by name instead of being read past its end.
fn resolve_attachment_load<'a>(
    view: &'a BufferView,
    leases: Option<&RenderLeaseContext<'_>>,
    attachment: usize,
    expected_bytes: u64,
) -> Result<RenderInputSource<'a>, ProviderError> {
    let source = resolve_render_input(view, leases, RenderInputRole::Attachment, attachment)?;
    let resolved = u64::try_from(source.len()).unwrap_or(u64::MAX);
    if resolved != expected_bytes {
        return Err(args_refusal("render_attachment_initial_mismatch")
            .with_field("attachment", FieldValue::Unsigned(attachment as u64))
            .with_field("view", FieldValue::Unsigned(view.view_id.get()))
            .with_field("expected_bytes", FieldValue::Unsigned(expected_bytes))
            .with_field("resolved_bytes", FieldValue::Unsigned(resolved))
            .with_detail(
                "the window a loading attachment reads has to be the attachment's own tightly \
                 packed byte extent",
            ));
    }
    Ok(source)
}

/// The one texel a multisampled `Load` seeds every sample with
/// (`research/docs/23` §82, v82).
///
/// A multisampled image cannot take its previous bytes from a buffer copy:
/// `vkCmdCopyBufferToImage` holds its destination to
/// `VK_SAMPLE_COUNT_1_BIT` (`VUID-vkCmdCopyBufferToImage-dstImage-07973`), a
/// blit is single-sample at both ends
/// (`VUID-vkCmdBlitImage-srcImage-00233`/`-dstImage-00234`), and a resolve
/// reduces a multisampled source into a single-sample destination
/// (`VUID-vkCmdResolveImage-srcImage-00257`/`-dstImage-00259`) — the other
/// direction. What is left is a *seed pass*: a render pass that opens the same
/// image from `CLEAR`, which writes the clear value to every sample of the
/// render area, before the measured pass opens it with `LOAD`. A clear value is
/// one colour for the whole attachment, so the declaration has to be one
/// repeated texel of its own format byte extent.
///
/// Two deviations are refused by name instead of being read as something else
/// (`research/docs/23` §82):
///
/// - a window that is not one repeated texel
///   (`render_multisample_load_nonuniform_unsupported`): a per-texel seed needs
///   a full-coverage fragment shader writing every sample, and the rail owns no
///   shader of its own;
/// - an owner's own mapping (`render_multisample_load_borrowed_unsupported`):
///   the seed's clear value is host state read before the first command exists,
///   so a borrowed window would be a snapshot here, not the device read of the
///   owner's live pages §74 promises for the load channel. The staged arm
///   states the same bytes through the provider's own copy.
fn resolve_multisample_seed(
    source: &RenderInputSource<'_>,
    attachment: usize,
    format: AttachmentFormat,
) -> Result<ClearColor, ProviderError> {
    if source.borrowed_lease().is_some() {
        return Err(
            capability_refusal("render_multisample_load_borrowed_unsupported")
                .with_field("attachment", FieldValue::Unsigned(attachment as u64))
                .with_field(
                    "storage_mode",
                    FieldValue::Text("borrowed_no_copy".to_owned()),
                )
                .with_detail(
                    "a multisampled `Load` is executed by a seed pass whose clear value is host \
                     state; a borrowed seed would be read here instead of by the device at \
                     execution",
                ),
        );
    }
    let width = usize::try_from(format.bytes_per_texel()).unwrap_or(usize::MAX);
    let bytes = source.proof_bytes();
    if width == 0 || bytes.len() < width {
        return Err(args_refusal("render_attachment_initial_mismatch")
            .with_field("attachment", FieldValue::Unsigned(attachment as u64))
            .with_field("byte_length", FieldValue::Unsigned(bytes.len() as u64))
            .with_detail("a multisampled load's seed window holds at least one texel"));
    }
    let (texel, rest) = bytes.split_at(width);
    if rest.chunks(width).any(|chunk| chunk != texel) {
        return Err(
            capability_refusal("render_multisample_load_nonuniform_unsupported")
                .with_field("attachment", FieldValue::Unsigned(attachment as u64))
                .with_detail(
                    "the seed route states one clear value for the whole raster, so the loaded \
                     window has to be one repeated texel; a per-texel seed needs a \
                     full-coverage fragment shader this rail does not own",
                ),
        );
    }
    ClearColor::from_bytes(texel).ok_or_else(|| {
        args_refusal("render_attachment_initial_mismatch")
            .with_field("attachment", FieldValue::Unsigned(attachment as u64))
            .with_field("byte_length", FieldValue::Unsigned(texel.len() as u64))
            .with_detail("a seed texel is one to eight bytes")
    })
}

/// Resolve one sampled texture's bytes into the source the rail uploads or
/// imports (`research/docs/23` §75, R5c).
///
/// The render sampler's third declaration of the same three arms: a texture
/// carries its whole byte extent (no offset and length, unlike a buffer view),
/// so the lease window is the texture's own tightly packed extent at the
/// reservation's start — the window rule core's registries state for
/// `TextureSource`. `OwnedBytes` behaves byte for byte as before, a
/// `StagedLease` uploads the provider's own copy of the owner's window, and a
/// `BorrowedNoCopy` imports the owner's pages, so `vkCmdCopyBufferToImage`
/// reads what the owner wrote rather than a snapshot of it.
fn resolve_render_texture_source<'a>(
    view: &'a TextureView,
    leases: Option<&RenderLeaseContext<'_>>,
    binding: usize,
) -> Result<RenderInputSource<'a>, ProviderError> {
    match &view.source {
        TextureSource::OwnedBytes(bytes) => Ok(RenderInputSource::TraceBytes(bytes)),
        TextureSource::StagedLease(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    RenderInputRole::Texture,
                    binding,
                    "staged_lease",
                    "the render submission carries no lease channel, so a lease-backed render \
                     texture cannot be read",
                )
            })?;
            let bytes = leases.staging.texture_bytes(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            Ok(RenderInputSource::StagedBytes(bytes))
        }
        TextureSource::BorrowedNoCopy(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    RenderInputRole::Texture,
                    binding,
                    "borrowed_no_copy",
                    "the render submission carries no lease channel, so a lease-backed render \
                     texture cannot be read",
                )
            })?;
            if leases.host_import_alignment == 0 {
                return Err(host_import_refusal(
                    RenderInputRole::Texture,
                    binding,
                    view.view_id,
                ));
            }
            let window = leases.borrowed.texture_pointer(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            let alignment = usize::try_from(leases.host_import_alignment).unwrap_or(usize::MAX);
            if !window.pointer.is_multiple_of(alignment) {
                return Err(lease_alignment_refusal(
                    *lease_id,
                    window.pointer,
                    leases.host_import_alignment,
                    RenderInputRole::Texture,
                    binding,
                ));
            }
            Ok(RenderInputSource::Borrowed {
                lease: *lease_id,
                window,
            })
        }
    }
}

/// One render input whose source this rail cannot read
/// (`docs/23` §71/§74/§75).
fn render_input_refusal(
    role: RenderInputRole,
    slot: usize,
    storage_mode: &'static str,
    detail: &'static str,
) -> ProviderError {
    capability_refusal(role.slug())
        .with_field(role.slot_field(), FieldValue::Unsigned(slot as u64))
        .with_field("storage_mode", FieldValue::Text(storage_mode.to_owned()))
        .with_detail(detail)
}

/// The borrowed arm a device without host-memory import cannot execute.
///
/// The slug is the same one core admission and the compute rail publish for an
/// unsupported storage mode, so a capture reads one name for one fact: this
/// device cannot bind owner memory without copying it.
fn host_import_refusal(role: RenderInputRole, slot: usize, view: ViewId) -> ProviderError {
    capability_refusal("storage_mode_unsupported")
        .with_field(role.slot_field(), FieldValue::Unsigned(slot as u64))
        .with_field("view", FieldValue::Unsigned(view.get()))
        .with_field(
            "storage_mode",
            FieldValue::Text("borrowed_no_copy".to_owned()),
        )
        .with_field("role", FieldValue::Text(role.slug().to_owned()))
        .with_detail(
            "the device does not import host memory, so a no-copy render input cannot be bound",
        )
}

/// An owner pointer that misses the device's import alignment (`docs/23` §71).
///
/// The name and fields are the ones the compute rail's import publishes, so the
/// two rails refuse the same fact the same way.
fn lease_alignment_refusal(
    lease_id: LeaseId,
    pointer: usize,
    alignment: u64,
    role: RenderInputRole,
    slot: usize,
) -> ProviderError {
    capability_refusal("lease_alignment_unsupported")
        .with_field("lease", FieldValue::Unsigned(lease_id.get()))
        .with_field("pointer", FieldValue::Unsigned(pointer as u64))
        .with_field("alignment", FieldValue::Unsigned(alignment))
        .with_field(role.slot_field(), FieldValue::Unsigned(slot as u64))
        .with_detail("a no-copy render input has to meet the device's host import alignment")
}

/// Pair the pipeline's layout with the pass's bound streams.
///
/// Core admission already refused a pass whose binding count disagrees with the
/// layout, and this rail re-runs `validate_against` before reaching here, so the
/// zip is length-checked by construction. What is added is the rail's own
/// minimum: a stream has to hold at least one vertex, and its bytes have to be
/// resolvable — trace-owned bytes, a staged lease's copy, or a no-copy owner
/// window (`research/docs/23` §71, R3c). The resolution runs before any device
/// object exists, so an unresolvable stream is refused with nothing partially
/// built.
fn resolve_vertex_streams<'a>(
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
    leases: Option<&RenderLeaseContext<'_>>,
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
        let source = resolve_render_input(view, leases, RenderInputRole::Vertex, index)?;
        if view.length < layout.stride {
            return Err(
                capability_refusal("render_vertex_buffer_footprint_unsupported")
                    .with_field("binding", FieldValue::Unsigned(index as u64))
                    .with_field("required_bytes", FieldValue::Unsigned(layout.stride))
                    .with_field("declared_bytes", FieldValue::Unsigned(view.length))
                    .with_detail("one vertex does not fit in the view the trace declares"),
            );
        }
        streams.push(VertexStream {
            layout,
            view,
            source,
        });
    }
    Ok(streams)
}

/// One contract blend factor as the `VkBlendFactor` it names. Closed for the
/// same reason every other translation here is: a factor that gains no arm is a
/// compile error rather than a silently different one.
fn vk_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    }
}

/// One contract blend operation as the `VkBlendOp` it names.
fn vk_blend_operation(operation: BlendOperation) -> vk::BlendOp {
    match operation {
        BlendOperation::Add => vk::BlendOp::ADD,
    }
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

/// Read the `count` indices the draw consumes out of one resolved window.
///
/// Every source arm arrives with its length already pinned to the view's own
/// (`OwnedBytes` by `BufferView::validate_shape`, a staged lease by the
/// registry's reservation check, a borrowed window by the no-copy registry), so
/// this is a pure translation. The values feed the footprint proof: every index
/// has to name a vertex the bound stream covers.
fn decode_indices(
    source: &RenderInputSource<'_>,
    format: IndexFormat,
    count: u32,
) -> Result<Vec<u32>, ProviderError> {
    let bytes = source.proof_bytes();
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
pub(crate) fn execute_indirect_render_pass<'a>(
    context: &VulkanContext,
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
    command: &IndirectCommandDescriptor,
    previous: &'a [Option<&'a BufferView>],
    resident: &[Option<&'a ProviderTargetImage>],
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<OffscreenReadback, ProviderError> {
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
    refuse_attachment_extent(context, pass)?;
    let mut request = prepare_render_request_with_resident(
        stages,
        pass,
        previous,
        if resident.is_empty() {
            None
        } else {
            Some(resident)
        },
        leases,
        context.admitted_depth_resolve_modes(),
        context.admitted_stencil_resolve_modes(),
        context.spirv_feature_policy(),
    )?;
    request.indirect = Some(replay);
    let retains = RenderInputRetains::retain(leases, &request)?;
    execute_offscreen_render_with_retains(context, &request, retains)
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

/// Refuse an attachment extent outside the rail's declared window.
///
/// R1b (`research/docs/23` §70) makes the window a declared, device-gated fact:
/// the capability snapshot publishes the reviewed ceiling clamped by this
/// device's `maxFramebuffer{Width,Height}`, and core admission refuses a wider
/// attachment as `attachment_dimension_limit`. This is the rail's own second
/// line, re-asked of the value it was handed so a directly-constructed request
/// cannot skip either half. The two halves are asked in the order the caller
/// can act on: the device's own answer first — an extent the device cannot open
/// a framebuffer for is `attachment_extent_device_limit` — and the reviewed
/// ceiling second, with the same slug and fields core admission uses. Neither
/// answer narrows the request: the extent the caller asked for is the extent the
/// refusal reports.
pub(crate) fn refuse_attachment_extent(
    context: &VulkanContext,
    pass: &RenderPassDescriptor,
) -> Result<(), ProviderError> {
    let limits = context.physical_device_limits();
    let device = [
        u64::from(limits.max_framebuffer_width),
        u64::from(limits.max_framebuffer_height),
    ];
    let extent_refusal = |slug: &'static str, width: u64, height: u64, maximum: [u64; 2]| {
        capability_refusal(slug)
            .with_field("width", FieldValue::Unsigned(width))
            .with_field("height", FieldValue::Unsigned(height))
            .with_field("maximum_width", FieldValue::Unsigned(maximum[0]))
            .with_field("maximum_height", FieldValue::Unsigned(maximum[1]))
    };
    let extents = pass
        .color_attachments
        .iter()
        .map(|attachment| (attachment.width, attachment.height))
        .chain(pass.depth.iter().map(|depth| (depth.width, depth.height)))
        .chain(
            pass.stencil
                .iter()
                .map(|stencil| (stencil.width, stencil.height)),
        );
    for (width, height) in extents {
        if width > device[0] || height > device[1] {
            return Err(extent_refusal(
                "attachment_extent_device_limit",
                width,
                height,
                device,
            ));
        }
        if width > crate::provider::REVIEWED_ATTACHMENT_CEILING[0]
            || height > crate::provider::REVIEWED_ATTACHMENT_CEILING[1]
        {
            return Err(extent_refusal(
                "attachment_dimension_limit",
                width,
                height,
                crate::provider::REVIEWED_ATTACHMENT_CEILING,
            ));
        }
    }
    Ok(())
}

/// The tightly packed bytes one stored attachment's readback occupies.
///
/// The render area's texel count times the attachment's **own** format width
/// (`research/docs/23` §78). A two-location pass whose formats differ in width —
/// every shape `Rgba16Float` can appear in — therefore lands two different byte
/// extents from one copy-out per location, and a direct
/// `attachment.format.bytes_per_texel()` at each of them is what keeps the
/// staging buffers honest.
pub(crate) fn attachment_readback_bytes(
    extent: [u32; 2],
    format: AttachmentFormat,
) -> Result<u64, ProviderError> {
    u64::from(extent[0])
        .checked_mul(u64::from(extent[1]))
        .and_then(|texels| texels.checked_mul(format.bytes_per_texel()))
        .ok_or_else(|| contract_refusal("render attachment bytes overflow u64"))
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
        // The census's `0x73` shape (`research/docs/23` §78): the first admitted
        // format whose texel is eight bytes, and the reason a colour readback
        // asks the attachment's own width instead of one rail-wide constant.
        AttachmentFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
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
    let bytes = clear.as_bytes();
    let four = narrow_four(bytes);
    let unorm = |byte: u8| f32::from(byte) / 255.0;
    match format {
        AttachmentFormat::Rgba8Unorm => vk::ClearColorValue {
            float32: [
                unorm(four[0]),
                unorm(four[1]),
                unorm(four[2]),
                unorm(four[3]),
            ],
        },
        AttachmentFormat::Bgra8Unorm => vk::ClearColorValue {
            float32: [
                unorm(four[2]),
                unorm(four[1]),
                unorm(four[0]),
                unorm(four[3]),
            ],
        },
        AttachmentFormat::R32Float => vk::ClearColorValue {
            float32: [f32::from_le_bytes(four), 0.0, 0.0, 0.0],
        },
        // Four half floats, in the format's memory order (R, G, B, A
        // little-endian halves). The widening is exact
        // (`metal_api_core::provider::half_to_f32`), so the components the
        // driver clears with are the very values the contract's bytes name, and
        // the attachment stores them back as the bytes the readback compares
        // (`research/docs/23` §78).
        AttachmentFormat::Rgba16Float => vk::ClearColorValue {
            float32: [
                half_from_memory_order(bytes, 0),
                half_from_memory_order(bytes, 1),
                half_from_memory_order(bytes, 2),
                half_from_memory_order(bytes, 3),
            ],
        },
        AttachmentFormat::R32Uint => vk::ClearColorValue {
            uint32: [u32::from_le_bytes(four), 0, 0, 0],
        },
    }
}

/// One half-precision component of an `Rgba16Float` clear, from its
/// little-endian pair of bytes.
///
/// A payload shorter than four halves cannot reach a driver call — admission and
/// the rail's own admission compare the clear's length with its attachment's
/// texel width — so a hand-built value that somehow got past both reads zero
/// rather than panicking inside a clear value.
fn half_from_memory_order(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    let pair = bytes.get(offset..offset + 2).unwrap_or(&[0, 0]);
    metal_api_core::provider::half_to_f32(u16::from_le_bytes([pair[0], pair[1]]))
}

/// The first four bytes of a clear payload, or four zeros for a value no
/// admission could have built: the same total-decode rule as
/// [`half_from_memory_order`], for the four-byte class.
fn narrow_four(bytes: &[u8]) -> [u8; 4] {
    let mut four = [0_u8; 4];
    if let Some(pair) = bytes.get(..4) {
        four.copy_from_slice(pair);
    }
    four
}

/// The Vulkan resolve mode one admitted depth filter names
/// (`research/docs/23` §3.3, v57/v70).
///
/// The three filters map one-to-one: Metal's `.sample0`, `.min` and `.max` are
/// `SAMPLE_ZERO`, `MIN` and `MAX`. The mode is what the subpass description
/// carries and what the agreement rule below compares, so both read it from
/// here rather than spelling the match twice.
fn depth_resolve_mode(filter: DepthResolveFilter) -> vk::ResolveModeFlags {
    match filter {
        DepthResolveFilter::Sample0 => vk::ResolveModeFlags::SAMPLE_ZERO,
        DepthResolveFilter::Min => vk::ResolveModeFlags::MIN,
        DepthResolveFilter::Max => vk::ResolveModeFlags::MAX,
    }
}

/// The Vulkan resolve mode one admitted stencil filter names
/// (`research/docs/23` §3.3, v60/v70).
///
/// Metal's `.sample0` — the one filter with a Vulkan counterpart — is
/// `SAMPLE_ZERO`. `.depthResolvedSample` reduces the stencil of the sample the
/// *depth* resolve picked; Vulkan's `MIN`/`MAX` reduce the stencil component
/// itself, so no mode names it and the probe never admits its bit. The
/// mapping answers `NONE` for it, and the per-filter mask check refuses that
/// filter before any mode question is asked.
fn stencil_resolve_mode(filter: StencilResolveFilter) -> vk::ResolveModeFlags {
    match filter {
        StencilResolveFilter::Sample0 => vk::ResolveModeFlags::SAMPLE_ZERO,
        StencilResolveFilter::DepthResolvedSample => vk::ResolveModeFlags::NONE,
    }
}

/// Whether a device that resolves depth and stencil with one shared mode
/// (`independentResolve` is false) can execute this pair of filters
/// (`research/docs/23` §3.3, v60/v70).
///
/// The Vulkan rule is that such a device requires `depthResolveMode` and
/// `stencilResolveMode` to be equal, so a pair whose two filters name two
/// different modes is a subpass description the device rejects. The reviewed
/// stored pair resolves both faces through `sample0` — one shared mode — which
/// is why the v60 fixtures never needed this question; the rule is what keeps a
/// mismatched pair a named refusal instead of a driver error.
fn shared_resolve_mode_admits(depth: DepthResolveFilter, stencil: StencilResolveFilter) -> bool {
    depth_resolve_mode(depth) == stencil_resolve_mode(stencil)
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

/// Whether the selected device can use `format` as a colour attachment with
/// `tiling` at the requested `samples` (`research/docs/23` §3.3, v51/v61;
/// §78).
///
/// `vkGetPhysicalDeviceFormatProperties` answers the single-sample
/// `COLOR_ATTACHMENT` bit but carries no sample count; the combination question
/// — one format, one usage, one sample count — is
/// `vkGetPhysicalDeviceImageFormatProperties`'s own `sampleCounts` field, which
/// is why this rail asks it before the first image exists. Both the
/// multisampled rasters and the wide-texel class ask it: the first because the
/// bit cannot answer their sample count, the second because a driver may answer
/// the bit while refusing the image. A format the device refuses, and a format
/// whose requested combination the driver rejects outright, both answer
/// `false`.
pub(crate) fn format_supports_color_attachment_samples(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
    samples: vk::SampleCountFlags,
) -> bool {
    let properties = unsafe {
        context
            .instance
            .get_physical_device_image_format_properties(
                context.physical,
                format,
                vk::ImageType::TYPE_2D,
                tiling,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageCreateFlags::empty(),
            )
    };
    match properties {
        Ok(properties) => properties.sample_counts.contains(samples),
        Err(_) => false,
    }
}

/// Whether the selected device can use `format` as a depth-stencil attachment
/// with `tiling` at the requested `samples` (`research/docs/23` §3.3,
/// v53/v61).
///
/// The colour sibling's rule one usage over: a multisampled pass's depth
/// surface has to carry the same sample count as the colour attachments, so the
/// device answers the requested combination through the same
/// `vkGetPhysicalDeviceImageFormatProperties` query.
pub(crate) fn format_supports_multisample_depth_attachment(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
    samples: vk::SampleCountFlags,
) -> bool {
    let properties = unsafe {
        context
            .instance
            .get_physical_device_image_format_properties(
                context.physical,
                format,
                vk::ImageType::TYPE_2D,
                tiling,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageCreateFlags::empty(),
            )
    };
    match properties {
        Ok(properties) => properties.sample_counts.contains(samples),
        Err(_) => false,
    }
}

/// The combined depth-stencil formats this rail admits, in preference order
/// (`research/docs/23` §3.3, v60/v66).
///
/// `D32_SFLOAT_S8_UINT` is the reviewed combined format: its depth aspect is
/// the same four-byte `depth32float` texel the single-face depth shape reads
/// back, so the v60 stored shape's copy-out is the depth channel's own.
/// `D24_UNORM_S8_UINT` is the fallback for the v66 rail-owned pair — neither
/// face leaves the pass, so no byte layout has to agree with the depth
/// channel — and a device that answers neither refuses the combined shape by
/// name.
pub(crate) const COMBINED_DEPTH_STENCIL_FORMATS: [vk::Format; 2] = [
    vk::Format::D32_SFLOAT_S8_UINT,
    vk::Format::D24_UNORM_S8_UINT,
];

/// The first reviewed combined format the `supports` predicate answers, or
/// `None` when it answers none (`research/docs/23` §3.3, v66).
///
/// The preference order is the constant above; keeping the choice a pure
/// function of the predicate is what lets the order be tested without a
/// device, while the probe itself stays the per-device question
/// `vkGetPhysicalDeviceImageFormatProperties` answers.
pub(crate) fn first_supported_combined_format(
    mut supports: impl FnMut(vk::Format) -> bool,
) -> Option<vk::Format> {
    COMBINED_DEPTH_STENCIL_FORMATS
        .into_iter()
        .find(|format| supports(*format))
}

/// The first combined format the device admits at this raster's sample count,
/// or `None` when it answers none of the reviewed formats
/// (`research/docs/23` §3.3, v66).
///
/// The probe is the depth surface's own question one format list over: the
/// combined attachment is created with these samples and used as a
/// `DEPTH_STENCIL_ATTACHMENT`, so `vkGetPhysicalDeviceImageFormatProperties`
/// is what answers whether the pair is executable.
pub(crate) fn combined_depth_stencil_format(
    context: &VulkanContext,
    tiling: vk::ImageTiling,
    samples: vk::SampleCountFlags,
) -> Option<vk::Format> {
    first_supported_combined_format(|format| {
        format_supports_multisample_depth_attachment(context, format, tiling, samples)
    })
}

/// The largest of the reviewed two-, four- and eight-sample rasters the
/// device's whole framebuffer admits, or `None` when it admits none of them
/// (`research/docs/23` §3.3, v51/v61).
///
/// `VkPhysicalDeviceLimits::framebufferColorSampleCounts` is the framebuffer
/// ceiling every colour attachment of a subpass shares, so it is the first
/// question the capability snapshot answers; the per-format question above is
/// what the rail asks before it creates the image.
pub(crate) fn limits_render_sample_count_ceiling(limits: &vk::PhysicalDeviceLimits) -> Option<u32> {
    let counts = limits.framebuffer_color_sample_counts;
    if counts.contains(vk::SampleCountFlags::TYPE_8) {
        Some(8)
    } else if counts.contains(vk::SampleCountFlags::TYPE_4) {
        Some(4)
    } else if counts.contains(vk::SampleCountFlags::TYPE_2) {
        Some(2)
    } else {
        None
    }
}

/// The reviewed 2/4/8 sample counts the device's whole framebuffer admits, as
/// a bitmask over the contract's codes: bit `i` = [`SampleCount`] code `i`
/// (`research/docs/23` §3.3, v61).
///
/// `VkPhysicalDeviceLimits::framebufferColorSampleCounts` is defined over
/// *all* framebuffer colour attachments, so a count the mask carries is one
/// every admitted colour format can run — which is why the device-gated
/// sample-count fixtures are owed exactly when their count's bit is present.
/// The counts are not a ladder: Lavapipe reports 4x and 8x without 2x, so the
/// ceiling alone cannot answer the 2x question.
pub(crate) fn limits_render_sample_count_mask(limits: &vk::PhysicalDeviceLimits) -> u32 {
    let counts = limits.framebuffer_color_sample_counts;
    let mut mask = 0;
    if counts.contains(vk::SampleCountFlags::TYPE_2) {
        mask |= 1 << SampleCount::Two.code();
    }
    if counts.contains(vk::SampleCountFlags::TYPE_4) {
        mask |= 1 << SampleCount::Four.code();
    }
    if counts.contains(vk::SampleCountFlags::TYPE_8) {
        mask |= 1 << SampleCount::Eight.code();
    }
    mask
}

/// The no-copy leases one render pass's inputs read, retained across its GPU
/// span (`research/docs/23` §71, R3c).
///
/// The disposition mirrors the compute rail's own rule: a retain is dropped
/// once the GPU can no longer read the owner's mapping — the pass's fence has
/// signalled, or the device was lost, which is a teardown guarantee. A
/// submission that failed without a lost device leaves every retain
/// outstanding on purpose, so the owner is blocked rather than the mapping
/// being freed under a GPU that may still be reading it; the rail is
/// synchronous, so the alternative would be a silent use-after-free whose only
/// remedy is a later device teardown.
struct RenderInputRetains {
    registry: Arc<BorrowedLeaseRegistry>,
    lease_ids: Vec<LeaseId>,
    /// Whether the holds are still outstanding. Cleared by [`Self::retire`]
    /// and by [`Self::after_submission_failure`]'s fail-closed arm.
    armed: bool,
}

impl RenderInputRetains {
    /// Retain every no-copy lease the prepared pass reads, before a single
    /// buffer is imported. `Ok(None)` for a pass whose inputs are uploaded
    /// bytes, which is every pre-R3c submission.
    fn retain(
        leases: Option<&RenderLeaseContext<'_>>,
        request: &OffscreenRenderRequest<'_>,
    ) -> Result<Option<Self>, ProviderError> {
        let mut lease_ids = Vec::new();
        for stream in &request.vertex_streams {
            if let Some(lease) = stream.source.borrowed_lease() {
                lease_ids.push(lease);
            }
        }
        if let Some(index) = &request.index_stream {
            if let Some(lease) = index.source.borrowed_lease() {
                lease_ids.push(lease);
            }
        }
        // A loading attachment whose previous contents come from an owner's
        // mapping is the third input of the same shape (`research/docs/23`
        // §74, R5b): the imported transfer source reads the owner's pages, so
        // the hold covers it exactly like a stream's.
        for attachment in &request.attachments {
            if let Some(source) = &attachment.previous {
                if let Some(lease) = source.borrowed_lease() {
                    lease_ids.push(lease);
                }
            }
        }
        // A sampled texture whose bytes come from an owner's mapping is the
        // fourth (`research/docs/23` §75, R5c), one hold per window: the
        // imported transfer source reads those pages until the pass's fence
        // signals, so a pass that binds several textures retains each lease
        // once per window it appears in.
        for texture in &request.textures {
            if let Some(lease) = texture.source.borrowed_lease() {
                lease_ids.push(lease);
            }
        }
        if lease_ids.is_empty() {
            return Ok(None);
        }
        let Some(leases) = leases else {
            // Resolution already refused a borrowed window without a lease
            // channel, so this arm is the belt-and-braces restatement: a
            // borrowed input is never executed without its registry.
            return Err(capability_refusal("storage_mode_unsupported")
                .with_field(
                    "storage_mode",
                    FieldValue::Text("borrowed_no_copy".to_owned()),
                )
                .with_detail(
                    "a no-copy render input was resolved without a lease channel to retain it in",
                ));
        };
        let registry = Arc::clone(leases.borrowed);
        registry.retain_all(&lease_ids)?;
        Ok(Some(Self {
            registry,
            lease_ids,
            armed: true,
        }))
    }

    /// The pass's fence has signalled, or the device is gone: drop every hold.
    fn retire(&mut self) {
        if self.armed {
            self.registry.retire_all(&self.lease_ids);
            self.armed = false;
        }
    }

    /// Dispose the retains after the pass's one submission returned an error.
    ///
    /// A lost device is a teardown guarantee, so the holds go, and so does a
    /// failure that never reached `vkQueueSubmit` (the compute rail's own
    /// `submitted` boundary). Every other error leaves them outstanding: the
    /// queue may still be reading the owner mapping, and only a later teardown
    /// can prove otherwise.
    fn after_submission_failure(&mut self, error: &ProviderError, submitted: bool) {
        if error.class == ProviderErrorClass::DeviceLost || !submitted {
            self.retire();
        } else {
            self.armed = false;
        }
    }
}

impl Drop for RenderInputRetains {
    fn drop(&mut self) {
        // An early return is always before the pass's submission: the import
        // may exist, but nothing has been queued to read it, so the holds go.
        // A failed submission clears `armed` first.
        self.retire();
    }
}

/// Execute one prepared offscreen request whose inputs are all uploaded bytes.
///
/// The rail's own unit fixtures go through this entry point: a request they
/// build always carries trace-owned bytes, so there is no no-copy lease to
/// retain. A submission whose inputs resolved to owner windows goes through
/// [`execute_offscreen_render_with_retains`] instead, which is the same
/// execution with the lease lifecycle around it (`research/docs/23` §71).
#[cfg(test)]
pub(crate) fn execute_offscreen_render(
    context: &VulkanContext,
    request: &OffscreenRenderRequest<'_>,
) -> Result<OffscreenReadback, ProviderError> {
    execute_offscreen_render_with_retains(context, request, None)
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
/// The texels one offscreen pass hands back (`research/docs/23` §3.3, v43/v49).
///
/// `attachments` carries one entry per colour attachment, in location order,
/// with `None` for each discarded one — the shape the readback channel had
/// before the depth attachment could be observed. `depth` carries the stored
/// depth surface's own tightly packed `depth32float` texels, or `None` when the
/// pass discards its depth attachment (which is what every pre-v43 trace
/// states). `stencil` carries the stored stencil surface's own tightly packed
/// one-byte texels, or `None` when the pass discards it (which is what every
/// pre-v49 trace states).
#[derive(Debug)]
pub(crate) struct OffscreenReadback {
    pub attachments: Vec<Option<Vec<u8>>>,
    pub depth: Option<Vec<u8>>,
    pub stencil: Option<Vec<u8>>,
}

fn execute_offscreen_render_with_retains(
    context: &VulkanContext,
    request: &OffscreenRenderRequest<'_>,
    mut retains: Option<RenderInputRetains>,
) -> Result<OffscreenReadback, ProviderError> {
    // The attachment count is the rail's own gate, re-run on the request so a
    // hand-built request cannot skip `prepare_render_request`'s admission.
    if request.attachments.len() > metal_api_core::provider::MAX_COLOR_ATTACHMENTS {
        return Err(mrt_attachment_count_refusal(request.attachments.len()));
    }
    // `docs/23` §3.6, v19: core admission refuses an all-discarded pass as
    // `AllRenderAttachmentsDiscarded`, and the rail re-asserts the same
    // at-least-one-store rule for a directly-constructed request. Discarding
    // every attachment would turn "nothing landed" into a blank proof of
    // "landed correctly" — but the depth attachment is a landing too from v43
    // on, so the depth-only shape (every colour attachment discards, the pass
    // keeps its depth surface) is the one exception (`docs/23` §3.3, v45).
    let stored_depth = request
        .depth
        .as_ref()
        .is_some_and(OffscreenDepthAttachment::storing);
    if !stored_depth
        && request
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
    // A translated registration's fragment stage is the module the translation
    // produced, so the pipeline binds exactly the module the reflection gate
    // checked (`request.translated_fragment`). A reviewed registration names no
    // fragment module: the format list does, and the refusal covers the
    // dual-combination and count shapes before any device call. The instanced
    // and depth fixtures own reviewed module pairs of their own
    // (`research/docs/23` §3.3, v31/v36): their vertex stages select the
    // fragment module that stores the varying they forward, and every other
    // vertex stage keeps the format list's solid module.
    let (fragment_spirv, fragment_entry_name) = match &request.translated_fragment {
        Some(fragment) => {
            // A translated fragment stage describes the interface the
            // translation produced, and the reviewed sampling module's
            // descriptor binding is not part of that description yet: the two
            // shapes are refused together instead of binding a descriptor the
            // reflection never mentioned (`research/docs/23` §3.3, v70).
            if !request.textures.is_empty() {
                return Err(
                    capability_refusal("render_texture_stage_unsupported").with_detail(
                        "render texture sampling is executed by the reviewed sampling pair; a \
                         translated fragment stage carries no image binding",
                    ),
                );
            }
            (fragment.spirv, fragment.entry.as_str())
        }
        None => {
            match depth_fragment_stage(&request.vertex.entry, request.vertex.spirv, &formats)? {
                Some(pair) => pair,
                None => match instanced_fragment_stage(
                    &request.vertex.entry,
                    request.vertex.spirv,
                    &formats,
                )? {
                    Some(pair) => pair,
                    None => {
                        match sampled_fragment_stage(
                            &request.vertex.entry,
                            request.vertex.spirv,
                            &formats,
                        )? {
                            Some(pair) => pair,
                            None => solid_fragment_stage(&formats)?,
                        }
                    }
                },
            }
        }
    };
    // The reviewed sampling pair and the pass's texture bindings are one
    // decision (`research/docs/23` §3.3, v70): the module samples the pass's
    // `DescriptorSet 0 / Binding 0`, so a pass that names the module without a
    // texture would sample an unbound descriptor, and a pass that binds a
    // texture with another fragment stage would ignore it. Both are refused by
    // name instead of executed as the other shape.
    if vertex_stage_is_sampled(&request.vertex.entry, request.vertex.spirv) {
        if request.textures.is_empty() {
            return Err(
                capability_refusal("render_texture_binding_required").with_detail(
                    "the reviewed sampling pair samples the pass's own texture binding; this \
                     pass binds none",
                ),
            );
        }
    } else if !request.textures.is_empty() {
        return Err(
            capability_refusal("render_texture_stage_unsupported").with_detail(
                "the pass binds a render texture but its fragment stage is not the reviewed \
                 sampling module",
            ),
        );
    }
    let tiling = vk::ImageTiling::OPTIMAL;
    let vk_formats = formats
        .iter()
        .map(|format| attachment_vk_format(*format))
        .collect::<Result<Vec<_>, _>>()?;
    // The multisample raster's device half (`research/docs/23` §3.3,
    // v51/v61): a multisampled colour attachment is a per-format question at
    // the pass's own sample count, and every colour attachment of the pass has
    // to answer it before the first `vkCreateImage`.
    // The load half is re-asserted here too, for a directly-constructed request
    // that skipped `prepare_render_request`: the only shape this increment
    // reviews opens every attachment from a clear, because uploading
    // single-sample previous bytes into a multisampled image is the load
    // increment this one does not execute.
    let samples = match request.multisample.map(|state| state.sample_count) {
        Some(SampleCount::Two) => vk::SampleCountFlags::TYPE_2,
        Some(SampleCount::Four) => vk::SampleCountFlags::TYPE_4,
        Some(SampleCount::Eight) => vk::SampleCountFlags::TYPE_8,
        Some(SampleCount::One) => {
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::SingleSampleMultisampleState.to_string(),
            ))
        }
        None => vk::SampleCountFlags::TYPE_1,
    };
    // The wide-texel class is device-gated at the pass's own sample count
    // (`research/docs/23` §78): `vkGetPhysicalDeviceImageFormatProperties` is
    // the question that names one format/usage/sample-count combination, and a
    // driver may answer the single-sample `COLOR_ATTACHMENT` bit while refusing
    // this format's own image — the bit-only admission is exactly what the
    // 2026-09-14 review filed after dzn answered it with a late
    // `VK_ERROR_OUT_OF_HOST_MEMORY`. The single-sample rasters ask it here; a
    // multisampled one asks the same question in the per-format loop below, so
    // no format is asked twice.
    if samples == vk::SampleCountFlags::TYPE_1 {
        for (vk_format, attachment) in vk_formats.iter().zip(&request.attachments) {
            if attachment.format.bytes_per_texel() <= NARROW_BYTES_PER_TEXEL {
                continue;
            }
            if !format_supports_color_attachment_samples(context, *vk_format, tiling, samples) {
                return Err(attachment_format_refusal()
                    .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field(
                        "missing_feature",
                        FieldValue::Text(format!("color_attachment_samples_{}", samples.as_raw())),
                    )
                    .with_detail(
                        "vkGetPhysicalDeviceImageFormatProperties reports no COLOR_ATTACHMENT \
                         combination at this raster's sample count for a format wider than the \
                         four-byte class",
                    ));
            }
        }
    }
    // A multisampled raster resolves into a single-sample landing the pass
    // owns, while a resident target *is* the landing (`research/docs/23` §76,
    // R7). The two shapes meet only through the present rail's v62 resolve
    // (`research/docs/24` §3.5), which R7 does not widen: a pass that declares
    // a resident target beside a multisample raster is refused here, before
    // the first image exists, instead of resolving into an image the trace
    // believes it named.
    if samples != vk::SampleCountFlags::TYPE_1
        && request
            .attachments
            .iter()
            .any(|attachment| attachment.resident.is_some())
    {
        return Err(
            capability_refusal("resident_target_multisample_unsupported")
                .with_field("samples", FieldValue::Unsigned(u64::from(samples.as_raw())))
                .with_detail(
                    "a multisampled pass resolves into a single-sample landing of its own; the \
                 resident target is the landing of a single-sample raster in this increment",
                ),
        );
    }
    if samples != vk::SampleCountFlags::TYPE_1 {
        if request
            .depth
            .as_ref()
            .is_some_and(OffscreenDepthAttachment::storing)
        {
            // The stored surface is admitted from v57 on, through the resolve
            // the request then has to state (`research/docs/23` §3.3, v57):
            // the same shape and per-filter questions `prepare_render_request`
            // asked, re-asserted here for a hand-built request that skipped
            // it.
            match request.depth_resolve {
                Some(resolve) => {
                    let mask = 1u32 << u32::from(resolve.filter.code());
                    if context.admitted_depth_resolve_modes() & mask == 0 {
                        return Err(
                            capability_refusal("render_depth_resolve_filter_unsupported")
                                .with_field(
                                    "filter",
                                    FieldValue::Unsigned(u64::from(resolve.filter.code())),
                                )
                                .with_field(
                                    "modes",
                                    FieldValue::Unsigned(u64::from(
                                        context.admitted_depth_resolve_modes(),
                                    )),
                                ),
                        );
                    }
                }
                None => {
                    return Err(
                        capability_refusal("render_multisample_depth_store_unsupported")
                            .with_detail(
                                "a multisampled depth surface cannot be kept without a depth \
                                 resolve",
                            ),
                    )
                }
            }
        }
        if request
            .stencil
            .as_ref()
            .is_some_and(OffscreenStencilAttachment::storing)
        {
            // A stencil-only resolve states `depthResolveMode = NONE` beside a
            // non-NONE stencil mode; a device whose
            // `independentResolveNone` is false requires the two to be both
            // NONE or both non-NONE, so the rail refuses the shape instead of
            // submitting a render pass the device rejects
            // (`research/docs/23` §3.3, v60).
            if request.depth_resolve.is_none() && !context.independent_resolve_none {
                return Err(
                    capability_refusal("render_stencil_resolve_independent_unsupported")
                        .with_detail(
                            "this device resolves the stencil face only together with the depth \
                     face; a stencil-only resolve needs a device whose \
                     independentResolveNone is true",
                        ),
                );
            }
            // The stored surface is admitted from v60 on, through the resolve
            // the request then has to state (`research/docs/23` §3.3, v60):
            // the same shape and per-filter questions `prepare_render_request`
            // asked, re-asserted here for a hand-built request that skipped
            // it.
            match request.stencil_resolve {
                Some(resolve) => {
                    let mask = 1u32 << u32::from(resolve.filter.code());
                    if context.admitted_stencil_resolve_modes() & mask == 0 {
                        return Err(capability_refusal(
                            "render_stencil_resolve_filter_unsupported",
                        )
                        .with_field(
                            "filter",
                            FieldValue::Unsigned(u64::from(resolve.filter.code())),
                        )
                        .with_field(
                            "modes",
                            FieldValue::Unsigned(u64::from(
                                context.admitted_stencil_resolve_modes(),
                            )),
                        ));
                    }
                    // The other half of the same device property
                    // (`research/docs/23` §3.3, v60/v70): a device whose
                    // `independentResolve` is false resolves the two faces
                    // with one shared mode, so a pass that names two different
                    // resolve filters is refused here rather than submitted as
                    // a subpass description the device rejects at creation.
                    // The reviewed stored pair resolves both faces through
                    // `sample0` — one shared mode — so no reviewed fixture
                    // reaches this branch; a hand-built mismatched pair does.
                    if let Some(depth_resolve) = request.depth_resolve {
                        if !context.independent_resolve
                            && !shared_resolve_mode_admits(depth_resolve.filter, resolve.filter)
                        {
                            return Err(capability_refusal(
                                "render_stencil_resolve_independent_unsupported",
                            )
                            .with_field(
                                "depth_filter",
                                FieldValue::Unsigned(u64::from(depth_resolve.filter.code())),
                            )
                            .with_field(
                                "stencil_filter",
                                FieldValue::Unsigned(u64::from(resolve.filter.code())),
                            )
                            .with_detail(
                                "this device resolves the depth and stencil faces with \
                                         one shared mode; a pass whose two faces resolve through \
                                         two different filters needs a device whose \
                                         independentResolve is true",
                            ));
                        }
                    }
                }
                None => {
                    return Err(
                        capability_refusal("render_multisample_stencil_store_unsupported")
                            .with_detail(
                                "a multisampled stencil surface cannot be kept without a \
                                 stencil resolve",
                            ),
                    )
                }
            }
        }
        for (vk_format, attachment) in vk_formats.iter().zip(&request.attachments) {
            if !format_supports_color_attachment_samples(context, *vk_format, tiling, samples) {
                return Err(attachment_format_refusal()
                    .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field(
                        "missing_feature",
                        FieldValue::Text(format!("color_attachment_samples_{}", samples.as_raw())),
                    )
                    .with_detail(
                        "vkGetPhysicalDeviceImageFormatProperties reports no sample-count \
                         combination this raster states for the COLOR_ATTACHMENT usage of \
                         this format",
                    ));
            }
            // A multisampled attachment's load (`research/docs/23` §3.3,
            // v51/v67/v82): `clear` and `dontcare` both execute — the second
            // opens the multisampled image from undefined contents and resolves
            // whatever the samples hold — and `load` executes from v82 on
            // through the seed pass the request's own `seed` names. The shape
            // decision itself was made before any device object existed
            // (`resolve_multisample_seed`); what is re-asserted here is the
            // agreement between a hand-built request's load and its seed, so a
            // request that states one without the other is refused by name
            // instead of reaching `vkCreateImage` with a route it has no colour
            // for.
            if samples != vk::SampleCountFlags::TYPE_1 {
                match (attachment.load, attachment.seed) {
                    (LoadOp::Load, None) => {
                        return Err(capability_refusal("render_multisample_load_unsupported")
                            .with_detail(
                                "a multisampled surface's previous contents are seeded by a \
                                 clear the request has to state",
                            ));
                    }
                    (LoadOp::Load, Some(_)) => {}
                    (_, Some(_)) => {
                        return Err(contract_refusal(
                            "only a multisampled `Load` carries a seed clear",
                        ));
                    }
                    (_, None) => {}
                }
            }
        }
        // The depth surface beside the raster is created with the same sample
        // count (`research/docs/23` §3.3, v53), so the device has to admit the
        // depth combination at the raster's own count before the first depth
        // image exists. A pass that opens both faces creates the one combined
        // surface instead, so its own question below is the one that runs
        // (`research/docs/23` §3.3, v60/v66).
        if request.depth.is_some()
            && request.stencil.is_none()
            && !format_supports_multisample_depth_attachment(
                context,
                vk::Format::D32_SFLOAT,
                tiling,
                samples,
            )
        {
            return Err(attachment_format_refusal()
                .with_field(
                    "vk_format",
                    FieldValue::Unsigned(vk::Format::D32_SFLOAT.as_raw() as u64),
                )
                .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                .with_field(
                    "missing_feature",
                    FieldValue::Text(format!("depth_attachment_samples_{}", samples.as_raw())),
                )
                .with_detail(
                    "vkGetPhysicalDeviceImageFormatProperties reports no sample-count \
                     combination this raster states for the DEPTH_STENCIL_ATTACHMENT usage \
                     of D32_SFLOAT",
                ));
        }
        // The stencil surface's own sample-count question, through the same
        // `DEPTH_STENCIL_ATTACHMENT` probe the depth surface uses
        // (`research/docs/23` §3.3, v55/v61). The combined pass asks its own
        // format's question below instead.
        if request.stencil.is_some()
            && request.depth.is_none()
            && !format_supports_multisample_depth_attachment(
                context,
                vk::Format::S8_UINT,
                tiling,
                samples,
            )
        {
            return Err(attachment_format_refusal()
                .with_field(
                    "vk_format",
                    FieldValue::Unsigned(vk::Format::S8_UINT.as_raw() as u64),
                )
                .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                .with_field(
                    "missing_feature",
                    FieldValue::Text(format!("stencil_attachment_samples_{}", samples.as_raw())),
                )
                .with_detail(
                    "vkGetPhysicalDeviceImageFormatProperties reports no sample-count \
                     combination this raster states for the DEPTH_STENCIL_ATTACHMENT usage \
                     of S8_UINT",
                ));
        }
        // The combined surface's own sample-count question
        // (`research/docs/23` §3.3, v60/v66): one attachment carries both
        // faces, so the device has to admit one of the reviewed combined
        // formats at this raster's sample count before the combined image
        // exists. The stored shape reads its depth aspect back as
        // `depth32float`, so it needs the four-byte format; the v66
        // rail-owned pair is the shape the packed fallback exists for.
        if request.depth.is_some() && request.stencil.is_some() {
            let Some(combined_format) = combined_depth_stencil_format(context, tiling, samples)
            else {
                return Err(attachment_format_refusal()
                    .with_field(
                        "vk_format",
                        FieldValue::Unsigned(vk::Format::D32_SFLOAT_S8_UINT.as_raw() as u64),
                    )
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field(
                        "missing_feature",
                        FieldValue::Text(format!(
                            "depth_stencil_attachment_samples_{}",
                            samples.as_raw()
                        )),
                    )
                    .with_detail(
                        "vkGetPhysicalDeviceImageFormatProperties reports no sample-count \
                         combination this raster states for the DEPTH_STENCIL_ATTACHMENT usage \
                         of either reviewed combined depth-stencil format",
                    ));
            };
            let storing = request.depth.as_ref().is_some_and(|depth| depth.storing())
                || request
                    .stencil
                    .as_ref()
                    .is_some_and(|stencil| stencil.storing());
            if storing && combined_format != vk::Format::D32_SFLOAT_S8_UINT {
                return Err(attachment_format_refusal()
                    .with_field(
                        "vk_format",
                        FieldValue::Unsigned(combined_format.as_raw() as u64),
                    )
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field(
                        "missing_feature",
                        FieldValue::Text("transfer_src".to_owned()),
                    )
                    .with_detail(
                        "the stored combined depth-stencil shape reads its depth aspect back \
                         as depth32float, so only D32_SFLOAT_S8_UINT carries a landing this \
                         rail's depth channel observes",
                    ));
            }
        }
    }
    for (vk_format, attachment) in vk_formats.iter().zip(&request.attachments) {
        admit_color_attachment(context, *vk_format, tiling)?;
        // A clear is exactly one texel of its own attachment's format
        // (`research/docs/23` §78), which is the same rule and the same name
        // core admission states: the rail reads the payload at the format's
        // width, so a short payload would otherwise be read past its own end
        // and a long one would carry bytes no texel holds.
        if let LoadOp::Clear(clear) = attachment.load {
            let expected = attachment.format.bytes_per_texel();
            let actual = clear.len() as u64;
            if actual != expected {
                return Err(args_refusal("attachment_clear_length_mismatch")
                    .with_field(
                        "format_code",
                        FieldValue::Unsigned(u64::from(attachment.format.code())),
                    )
                    .with_field("expected", FieldValue::Unsigned(expected))
                    .with_field("actual", FieldValue::Unsigned(actual))
                    .with_detail(
                        "a clear payload is exactly one tightly packed texel of its attachment's \
                         format",
                    ));
            }
        }
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
    // The depth surface's own extent: `depth32float` is four bytes per texel
    // (`DEPTH_BYTES_PER_TEXEL`), which the colour attachments' widths no longer
    // imply (`research/docs/23` §78).
    let depth_byte_length = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(NARROW_BYTES_PER_TEXEL))
        .ok_or_else(|| contract_refusal("render attachment bytes overflow u64"))?;
    // The stencil surface's readback extent is the same render area one byte
    // wide, which is what both its staging buffer and its copy state
    // (`research/docs/23` §3.3, v49).
    let stencil_byte_length = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|texels| texels.checked_mul(STENCIL_BYTES_PER_TEXEL))
        .ok_or_else(|| contract_refusal("render stencil attachment bytes overflow u64"))?;

    crate::terminal_refusal(&context.lock_lifecycle())?;
    let queue_index = select_graphics_queue(context)?;
    let vertex_words = spirv_words(request.vertex.spirv)
        .ok_or_else(|| spirv_refusal("vertex SPIR-V is empty or not a multiple of four bytes"))?;
    let fragment_words = spirv_words(fragment_spirv)
        .ok_or_else(|| spirv_refusal("fragment SPIR-V is empty or not a multiple of four bytes"))?;
    let vertex_entry = stage_entry_cstring("vertex", &request.vertex.entry)?;
    let fragment_entry = stage_entry_cstring("fragment", fragment_entry_name)?;

    // One layout guard per resident attachment, taken in attachment order
    // before the first device object exists (`research/docs/23` §76, R7). The
    // guard is the target's serialization point, exactly as it is for the
    // present action (`research/docs/24` §3.3 rule 1): the value it holds *is*
    // the `initialLayout` this submission has to declare, and the terminal
    // layout is published through the same guard before it drops. Taking them
    // in location order is what keeps two passes naming two resident targets in
    // opposite orders from interleaving their transitions.
    let mut resident_layouts = ResidentTargetLayouts::acquire(request);
    let mut objects = OffscreenObjects::new(context);
    for (index, (attachment, vk_format)) in request.attachments.iter().zip(&vk_formats).enumerate()
    {
        match attachment.resident {
            Some(target) => objects.attach_resident_target(
                index,
                target,
                resident_layouts.layout(index),
                attachment,
            )?,
            None => objects.create_attachment(
                *vk_format,
                width,
                height,
                attachment.load,
                attachment.seed,
                attachment.store == StoreOp::Store,
                samples,
            )?,
        }
    }
    // The combined depth-stencil shape (`research/docs/23` §3.3, v60): Vulkan
    // binds one attachment for both faces, so a pass that opens both creates
    // the one `D32_SFLOAT_S8_UINT` surface and its one resolve landing instead
    // of the two single-face surfaces the branches below build. From v66 on
    // the rail-owned pair is the same surface without a landing: neither face
    // is stored, so the surface's texels leave with the pass and the colour
    // resolve is the whole observation.
    if let (Some(depth), Some(stencil)) = (&request.depth, &request.stencil) {
        // The format is the device's answer to the same question the gates
        // above asked, re-asserted here so a directly-constructed request
        // cannot reach `vkCreateImage` with a format the pair was not reviewed
        // against (`research/docs/23` §3.3, v66).
        let combined_format =
            combined_depth_stencil_format(context, tiling, samples).ok_or_else(|| {
                attachment_format_refusal()
                    .with_field(
                        "vk_format",
                        FieldValue::Unsigned(vk::Format::D32_SFLOAT_S8_UINT.as_raw() as u64),
                    )
                    .with_field("tiling", FieldValue::Text(tiling_name(tiling).to_owned()))
                    .with_field(
                        "missing_feature",
                        FieldValue::Text(format!(
                            "depth_stencil_attachment_samples_{}",
                            samples.as_raw()
                        )),
                    )
                    .with_detail(
                        "the combined depth-stencil image needs a format the device admits at \
                         this raster's sample count",
                    )
            })?;
        objects.create_combined_depth_stencil(
            width,
            height,
            samples,
            combined_format,
            request.depth_resolve.is_some() && request.stencil_resolve.is_some(),
        )?;
        if depth.storing() {
            objects.create_depth_readback(depth_byte_length)?;
        }
        if stencil.storing() {
            objects.create_stencil_readback(stencil_byte_length)?;
        }
    } else if let Some(depth) = &request.depth {
        // The depth image is created before the render pass that names it, and
        // its `Load`/clear choice is settled by the image's own load operation
        // (`research/docs/23` §3.3, v36). A pass that keeps the surface also
        // creates the readback destination its texels land in, before the
        // render pass is built from the same decision (v43).
        objects.create_depth(
            depth.width,
            depth.height,
            depth.clear.is_none(),
            depth.storing(),
            samples,
            // A stored multisampled surface states its resolve filter; every
            // other shape carries none (`research/docs/23` §3.3, v57).
            request.depth_resolve.map(|resolve| resolve.filter),
        )?;
        if depth.storing() {
            objects.create_depth_readback(depth_byte_length)?;
        }
    } else if let Some(stencil) = &request.stencil {
        // The stencil image is created before the render pass that names it,
        // and a pass that keeps the surface also creates the readback
        // destination its texels land in, exactly as the depth path does since
        // v43 (`research/docs/23` §3.3, v49).
        objects.create_stencil(
            stencil.width,
            stencil.height,
            stencil.clear.is_none(),
            stencil.storing(),
            samples,
            // A stored multisampled stencil surface states its resolve
            // filter; every other shape carries none
            // (`research/docs/23` §3.3, v60).
            request.stencil_resolve.map(|resolve| resolve.filter),
        )?;
        if stencil.storing() {
            objects.create_stencil_readback(stencil_byte_length)?;
        }
    }
    objects.create_render_pass(
        &vk_formats,
        request.depth.as_ref(),
        request.stencil.as_ref(),
        // A stored multisampled depth surface states its resolve filter; every
        // other pass carries none (`research/docs/23` §3.3, v57).
        request.depth_resolve.map(|resolve| resolve.filter),
        // A stored multisampled stencil surface states its resolve filter;
        // every other pass carries none (`research/docs/23` §3.3, v60).
        request.stencil_resolve.map(|resolve| resolve.filter),
    )?;
    // The seed pass a multisampled `Load` is executed with
    // (`research/docs/23` §82, v82) is created beside the measured pass: both
    // name the same images, and `record` runs the seed pass first so the
    // measured pass's `LOAD` opens samples the clear already defined.
    objects.create_seed_render_pass(&vk_formats)?;
    objects.create_seed_framebuffer(width, height)?;
    objects.create_framebuffer(width, height)?;
    // The sampled textures are created before the pipeline, because the
    // sampled pipeline's layout is built from the descriptor set layout they
    // install (`research/docs/23` §3.3, v70).
    objects.create_render_textures(&request.textures)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
        &request.vertex_streams,
        request.depth.as_ref(),
        request.stencil.as_ref(),
        request.cull,
        request.blend.as_ref(),
    )?;
    // One readback destination per stored attachment; a discarded attachment
    // creates none, because its bytes leave no observable surface to land in
    // (`docs/23` §3.6, v19).
    let mut readback_mappings = Vec::with_capacity(request.attachments.len());
    for attachment in &request.attachments {
        if attachment.store == StoreOp::Store {
            // Each stored attachment lands `width * height * its own texel
            // width` bytes (`research/docs/23` §78), so a two-location pass
            // whose formats differ in width still reads both back whole.
            readback_mappings.push(objects.create_readback(attachment_readback_bytes(
                request.extent,
                attachment.format,
            )?)?);
        }
    }
    objects.create_vertex_inputs(&request.vertex_streams, request.index_stream.as_ref())?;
    objects.draw = request.draw;
    objects.instance_count = request.instance_count;
    objects.base_vertex = request.base_vertex;
    for (index, attachment) in request.attachments.iter().enumerate() {
        if let Some(previous) = &attachment.previous {
            // A multisampled attachment's `Load` is seeded by the clear of the
            // rail's own render pass instead of a buffer copy
            // (`research/docs/23` §82, v82): `vkCmdCopyBufferToImage` cannot
            // address a multisampled image, so no previous-byte buffer exists
            // for this arm and the seed is what `record` states. Every
            // single-sample load keeps the upload it always had.
            if attachment.seed.is_none() {
                objects.create_previous_bytes(index, previous)?;
            }
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
    objects.record(
        &request.attachments,
        request.depth.as_ref(),
        request.stencil.as_ref(),
        // The resolve filter the subpass was built with; `record` reads the
        // same request field for its clear-value placeholder and copy-out
        // source (`research/docs/23` §3.3, v57).
        request.depth_resolve.map(|resolve| resolve.filter),
        // The stencil resolve filter the subpass was built with; `record`
        // reads the same request field for its clear-value placeholder and
        // copy-out source (`research/docs/23` §3.3, v60).
        request.stencil_resolve.map(|resolve| resolve.filter),
        request.scissor,
        width,
        height,
    )?;
    // The retained no-copy leases outlive the submission: the fence below is
    // what proves the GPU can no longer read the owner's mapping
    // (`research/docs/23` §71, R3c).
    match objects.submit_and_wait(queue_index) {
        Ok(()) => {
            if let Some(retains) = retains.as_mut() {
                retains.retire();
            }
            // The pass completed, so every resident target is now in the
            // layout the next submission starts from — and, for the provider's
            // own bookkeeping, in a state a later `LoadOp::Resident` may read
            // (`research/docs/23` §76, R7).
            resident_layouts.publish(true);
        }
        Err(error) => {
            if let Some(retains) = retains.as_mut() {
                retains.after_submission_failure(&error, objects.submitted);
            }
            // A submission the driver refused before it ran leaves the image
            // exactly as it was; a submission that reached the queue may have
            // left any layout and any bytes behind, so the target states
            // `UNDEFINED` — the one old layout that is always legal to
            // declare — and the provider marks the identity undefined rather
            // than letting a later resident load read an image of unknown
            // state.
            resident_layouts.publish(!objects.submitted);
            return Err(error);
        }
    }

    // One readback record per stored attachment: `copy_out` equals the stored
    // attachment count, so a caller can observe that a stored location really
    // left the device and that a discarded one produced no bytes at all. A
    // stored depth surface adds its own record on top of that count, through
    // the same copy-out channel (`research/docs/23` §3.3, v43).
    let mut results = Vec::with_capacity(request.attachments.len());
    let mut mappings = readback_mappings.into_iter();
    for attachment in &request.attachments {
        if attachment.store == StoreOp::Store {
            let mapping = mappings.next().expect("one readback per stored attachment");
            let bytes = attachment_readback_bytes(request.extent, attachment.format)?;
            let texels = unsafe {
                std::slice::from_raw_parts(mapping as *const u8, bytes as usize).to_vec()
            };
            context.record_buffer_readback();
            context.record_buffer_readback_bytes(texels.len());
            results.push(Some(texels));
        } else {
            results.push(None);
        }
    }
    let depth = objects.depth_readback_bytes(depth_byte_length as usize, context)?;
    // The stored stencil surface follows the depth one through the same
    // copy-out channel; its byte extent is one per texel, not the colour
    // attachments' four (`research/docs/23` §3.3, v49).
    let stencil = objects.stencil_readback_bytes(stencil_byte_length as usize, context)?;
    Ok(OffscreenReadback {
        attachments: results,
        depth,
        stencil,
    })
}

/// One provider-owned target image: the present rail's presentable target
/// (`research/docs/24` §3.6) and the resident render target the R7 increment
/// adds (`research/docs/23` §76) are the same object.
///
/// Unlike the one-shot attachment [`OffscreenObjects`] creates per pass, this
/// image is owned by the provider and survives every submission until the
/// allocation lease it backs is released. That is what makes `docs/24` §3.3's
/// second rule ("the target stays readable after `wait`") hold — and it is the
/// same property a resident target needs to be loadable by a later
/// submission: the bytes have to remain observable (or loadable) after a
/// submission, so the image cannot live in the pass's own drop scope
/// (`docs/24` §5.2, `research/docs/23` §76). It is created once per
/// `(allocation, view)` identity and reused by later submissions that present
/// or render into the same target.
///
/// The image carries `TRANSFER_DST` in addition to the
/// `COLOR_ATTACHMENT | TRANSFER_SRC` pair the offscreen attachment uses: the
/// sentinel pre-fill (`docs/24` §3.1) uploads through
/// `vkCmdClearColorImage`, which is a transfer-destination operation, and the
/// pre-fill path is the same `Clear` a resident target's first pass states.
pub(crate) struct ProviderTargetImage {
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

impl ProviderTargetImage {
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

    /// Begin one round trip on this target, holding its layout lock until the
    /// returned guard is dropped. Both rails that render into a provider-owned
    /// image take it: the present action (`research/docs/24` §3.6) and a pass
    /// that declares the resident target (`research/docs/23` §76, R7).
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
    pub(crate) fn begin_target_pass(&self) -> std::sync::MutexGuard<'_, vk::ImageLayout> {
        self.layout_lock()
    }

    fn layout_lock(&self) -> std::sync::MutexGuard<'_, vk::ImageLayout> {
        self.layout
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for ProviderTargetImage {
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
fn present_subpass_dependency() -> vk::SubpassDependency2<'static> {
    vk::SubpassDependency2::default()
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
pub(crate) fn execute_present_render<'a>(
    context: &VulkanContext,
    stages: &'a RenderStages,
    pass: &'a RenderPassDescriptor,
    target: &ProviderTargetImage,
    previous: Option<&'a BufferView>,
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<Vec<u8>, ProviderError> {
    // The present path stays single-attachment: it renders into one
    // provider-owned target and hands that target on, so a pass whose
    // attachment list is not exactly one entry is outside this increment's
    // present shape.
    let previous = [previous];
    refuse_attachment_extent(context, pass)?;
    let request = prepare_render_request(
        stages,
        pass,
        &previous,
        leases,
        context.admitted_depth_resolve_modes(),
        context.admitted_stencil_resolve_modes(),
        context.spirv_feature_policy(),
    )?;
    let [attachment] = request.attachments.as_slice() else {
        return Err(contract_refusal(
            "the present rail executes exactly one colour attachment",
        ));
    };
    // The present rail selects its fragment half by the same rule the
    // offscreen executor states: a translated registration binds the module
    // the translation produced (whose reflection the registration gate already
    // checked against the contract), and a reviewed registration binds the
    // format list's solid module. The registration gate itself
    // (`RenderStages::validate_stage_pair`, re-asked by
    // `prepare_render_request`) is what keeps a stage the rail has no account
    // of out of both rails, so the present rail adds no second gate of its own
    // (`research/docs/23` §3.3; R4a increment).
    // A present attachment is the pass's only observable landing point, so a
    // `StoreOp::DontCare` present pass is the all-discarded shape the rail
    // refuses for an offscreen request (`docs/23` §3.6, v19). Core admission
    // already refused it as `AllRenderAttachmentsDiscarded`; this is the
    // value-level second line of defence.
    if attachment.store == StoreOp::DontCare {
        return Err(render_all_attachments_discarded_refusal());
    }
    // A present pass renders into the provider-owned target alone: it opens no
    // depth surface, so a trace that names one — stored or not — asks for a
    // state this shape cannot execute. Refusing here keeps the depth attachment
    // from being silently dropped instead of opened, which is what the
    // offscreen path would do with it (`research/docs/23` §3.3, v43).
    if let Some(depth) = &request.depth {
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
    // The stencil surface is the depth rule's sibling: the present shape opens
    // neither surface, so a trace that names one — stored or not — is refused
    // here rather than having it silently dropped by the pass that follows
    // (`research/docs/23` §3.3, v49). No reviewed fixture pairs presentation
    // with a stencil attachment; refusing keeps that combination fail-closed
    // until one arrives.
    if let Some(stencil) = &request.stencil {
        return Err(capability_refusal("render_present_stencil_unsupported")
            .with_field(
                "store",
                FieldValue::Text(
                    match stencil.store {
                        Some(StoreOp::Store) => "store",
                        Some(StoreOp::DontCare) => "dontcare",
                        // A resident stencil store is refused by core admission
                        // (`research/docs/23` §76, R7): the stencil surface has
                        // no provider-owned identity to keep its bytes under.
                        Some(StoreOp::Resident) => "resident",
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
    let [width, height] = request.extent;
    if width == 0 || height == 0 {
        return Err(contract_refusal("render attachment has a zero dimension"));
    }
    // The present target inherits its one attachment's format
    // (`research/docs/24` §3.2), so its readback is that format's own texel
    // width over the pass extent — eight bytes per texel for the census's
    // `Rgba16Float` shape (`research/docs/23` §78).
    let byte_length = attachment_readback_bytes(request.extent, attachment.format)?;

    crate::terminal_refusal(&context.lock_lifecycle())?;
    let queue_index = select_graphics_queue(context)?;
    // The fragment module is the registration's when the registration is a
    // translation, and the format list's reviewed solid module otherwise — the
    // same pair the offscreen rail binds (`research/docs/23` §3.3, R2/V70).
    let (fragment_spirv, fragment_entry_name): (&[u8], &str) = match &request.translated_fragment {
        Some(fragment) => (fragment.spirv, fragment.entry.as_str()),
        None => solid_fragment_stage(&[attachment.format])?,
    };
    // The reviewed sampling pair (`research/docs/23` §3.3, v70) is the
    // offscreen rail's own fragment selection; the present rail binds the
    // format's solid module instead, and a translated fragment stage carries
    // no image binding at all. A pass that names the sampling vertex stage or
    // binds a render texture is therefore refused by the same slugs the
    // offscreen rail uses, rather than executed with a module that silently
    // ignores the binding.
    if vertex_stage_is_sampled(&request.vertex.entry, request.vertex.spirv) {
        if request.textures.is_empty() {
            return Err(
                capability_refusal("render_texture_binding_required").with_detail(
                    "the reviewed sampling pair samples the pass's own texture binding; this \
                 present pass binds none",
                ),
            );
        }
        return Err(
            capability_refusal("render_texture_stage_unsupported").with_detail(
                "the present rail binds the format's solid module; the reviewed sampling pair is \
             executed by the offscreen rail",
            ),
        );
    }
    if !request.textures.is_empty() {
        return Err(
            capability_refusal("render_texture_stage_unsupported").with_detail(
                "the pass binds a render texture but its fragment stage is not the reviewed \
             sampling module",
            ),
        );
    }
    let vk_format = attachment_vk_format(attachment.format)?;
    // The multisample raster (`research/docs/23` §3.3, v51/v61) is executed
    // at the pass's own sample count. A present pass's n-sample surface is a
    // per-format question at that count, asked before the first image exists —
    // the same probe the offscreen rail runs.
    let samples = match request.multisample.map(|state| state.sample_count) {
        Some(SampleCount::Two) => vk::SampleCountFlags::TYPE_2,
        Some(SampleCount::Four) => vk::SampleCountFlags::TYPE_4,
        Some(SampleCount::Eight) => vk::SampleCountFlags::TYPE_8,
        Some(SampleCount::One) => {
            return Err(contract_refusal(
                &metal_api_core::provider::ContractError::SingleSampleMultisampleState.to_string(),
            ))
        }
        None => vk::SampleCountFlags::TYPE_1,
    };
    if samples != vk::SampleCountFlags::TYPE_1
        && !format_supports_color_attachment_samples(
            context,
            vk_format,
            vk::ImageTiling::OPTIMAL,
            samples,
        )
    {
        return Err(attachment_format_refusal()
            .with_field("vk_format", FieldValue::Unsigned(vk_format.as_raw() as u64))
            .with_field(
                "tiling",
                FieldValue::Text(tiling_name(vk::ImageTiling::OPTIMAL).to_owned()),
            )
            .with_field(
                "missing_feature",
                FieldValue::Text(format!("color_attachment_samples_{}", samples.as_raw())),
            )
            .with_detail(
                "vkGetPhysicalDeviceImageFormatProperties reports no sample-count \
                 combination this raster states for the COLOR_ATTACHMENT usage of \
                 this format",
            ));
    }
    let vertex_words = spirv_words(request.vertex.spirv)
        .ok_or_else(|| spirv_refusal("vertex SPIR-V is empty or not a multiple of four bytes"))?;
    let fragment_words = spirv_words(fragment_spirv)
        .ok_or_else(|| spirv_refusal("fragment SPIR-V is empty or not a multiple of four bytes"))?;
    let vertex_entry = stage_entry_cstring("vertex", &request.vertex.entry)?;
    let fragment_entry = stage_entry_cstring("fragment", fragment_entry_name)?;

    // The no-copy leases this pass reads are retained before the first import
    // and dropped once the fence below proves the GPU is done with them
    // (`research/docs/23` §71, R3c).
    let mut retains = RenderInputRetains::retain(leases, &request)?;
    // One acquire per present action, before the pass runs (`docs/24` §3.6).
    //
    // The guard serializes the whole round trip on this target's layout: the
    // submission below declares `*layout` as its `initialLayout`, and the
    // terminal layout is published through the same guard before it drops, so
    // a concurrent present of this target cannot read a layout that another
    // submission has already changed.
    let mut layout = target.begin_target_pass();
    context.record_present_acquire();
    let mut objects = OffscreenObjects::new(context);
    objects.attach_present_target(target, *layout, samples, vk_format, width, height)?;
    objects.create_render_pass(&[vk_format], None, None, None, None)?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(
        &vertex_words,
        &fragment_words,
        &vertex_entry,
        &fragment_entry,
        &request.vertex_streams,
        None,
        None,
        None,
        None,
    )?;
    let readback_mapping = objects.create_readback(byte_length)?;
    objects.create_vertex_inputs(&request.vertex_streams, request.index_stream.as_ref())?;
    objects.draw = request.draw;
    objects.instance_count = request.instance_count;
    objects.base_vertex = request.base_vertex;
    objects.create_command_pool(queue_index)?;
    objects.record(
        std::slice::from_ref(attachment),
        None,
        None,
        None,
        None,
        None,
        width,
        height,
    )?;
    match objects.submit_and_wait(queue_index) {
        Ok(()) => {
            if let Some(retains) = retains.as_mut() {
                retains.retire();
            }
        }
        Err(error) => {
            if let Some(retains) = retains.as_mut() {
                retains.after_submission_failure(&error, objects.submitted);
            }
            return Err(error);
        }
    }

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
    /// The rail-owned depth image of a pass that declares one
    /// (`research/docs/23` §3.3, v36). `None` for every pre-v36 pass, which is
    /// why the render pass, framebuffer and pipeline below all branch on it.
    depth: Option<DepthObjects>,
    /// The rail-owned stencil image of a pass that declares one
    /// (`research/docs/23` §3.3, v47). `None` for every pre-v47 pass, which is
    /// why the render pass, framebuffer and pipeline below all branch on it.
    stencil: Option<StencilObjects>,
    /// Whether this pass hands its attachment on as a present target. When set,
    /// the render pass ends in `COLOR_ATTACHMENT_OPTIMAL` and `record` inserts
    /// the explicit present layout transition before the copy-out
    /// (`docs/24` §3.3 rule 1).
    present: bool,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    /// The rail's own render pass over the seeded multisampled attachments
    /// (`research/docs/23` §82, v82): one `CLEAR`-opened subpass whose
    /// clear values are the seeds `record` states, storing the image so the
    /// measured pass can open it with `LOAD`. Null for every pass that seeds
    /// nothing, which is every pre-v82 shape.
    seed_render_pass: vk::RenderPass,
    /// The framebuffer of [`Self::seed_render_pass`]: the seeded attachments'
    /// own views, in location order.
    seed_framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    /// One readback per attachment, in location order.
    readbacks: Vec<ReadbackObjects>,
    /// The sampled textures the pass reads, in binding order
    /// (`research/docs/23` §3.3, v70). Empty for every pre-v70 pass, which is
    /// the shape the pipeline layout and the descriptor bind branch on.
    textures: Vec<SampledTextureObjects>,
    /// Whether this pass's command buffer reached `vkQueueSubmit`
    /// (`research/docs/23` §71, R3c).
    ///
    /// A failure after this point may still leave the GPU reading the owner
    /// mappings the pass imported, so it is the boundary the no-copy retain
    /// disposition keys on; a failure before it leaves nothing that could read
    /// them.
    submitted: bool,
    /// The descriptor set layout the sampled pipeline is built from, or null
    /// for a pass that samples nothing.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// The pool the descriptor set was allocated from, or null for a pass that
    /// samples nothing.
    descriptor_pool: vk::DescriptorPool,
    /// The set `record` binds before the draw, or null for a pass that samples
    /// nothing.
    descriptor_set: vk::DescriptorSet,
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
    /// Instances a direct draw runs (`research/docs/23` §3.3, v31); `1` for
    /// every pre-v31 pass.
    instance_count: u32,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34); `0` for every pre-v34 pass.
    base_vertex: u32,
    /// Index width of the caller-held index buffer.
    input_index_type: vk::IndexType,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

/// The Vulkan objects one sampled render texture owns
/// (`research/docs/23` §3.3, v70).
///
/// The compute rail's sampled texture, scoped to the pass instead of to a
/// dispatch: a host-visible `LINEAR` `R8G8B8A8_UNORM` image, its view and the
/// provider-synthesised nearest/clamp sampler the descriptor binds. The bytes
/// are written once, when the pass's objects are created, and the `record`
/// step's barrier is what makes them visible to the fragment stage.
///
/// A no-copy texture's image is device-local instead (`research/docs/23` §75,
/// R5c), and `copy_source` carries the owner's imported window the `record`
/// step's `vkCmdCopyBufferToImage` reads it from. The buffer and its memory are
/// the pass's own, like the image beside them.
struct SampledTextureObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
    /// The extent the image was created with, which is also the region the
    /// no-copy arm's `vkCmdCopyBufferToImage` covers (`research/docs/23` §75,
    /// R5c).
    extent: [u32; 2],
    /// The owner-window buffer a no-copy texture is copied out of, or `None`
    /// for the two uploaded arms.
    copy_source: Option<(vk::Buffer, vk::DeviceMemory)>,
}

/// The Vulkan objects the rail-owned depth attachment owns
/// (`research/docs/23` §3.3, v36).
struct DepthObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// The `VkFormat` the image was created with (`research/docs/23` §3.3,
    /// v66): `D32_SFLOAT` for every single-face depth surface, and whichever
    /// reviewed combined format the device answered for a pass that opens both
    /// faces. The render pass's own attachment description restates it, so the
    /// description cannot disagree with the image the framebuffer binds.
    format: vk::Format,
    /// Whether the pass opens the image from the attachment layout a previous
    /// pass left it in (`Load`) or from `UNDEFINED` (a clear).
    loading: bool,
    /// The sample count the image was created with (`research/docs/23` §3.3,
    /// v53): `TYPE_1` for every pre-v53 depth surface, `TYPE_4` when the pass
    /// states a multisample raster, which the render pass's own depth
    /// description restates so the two cannot disagree.
    samples: vk::SampleCountFlags,
    /// The single-sample image a stored multisampled depth surface resolves
    /// into (`research/docs/23` §3.3, v57), or `None` for a surface that
    /// resolves nothing — every pre-v57 depth surface and every single-sample
    /// stored one. The resolve target is what the copy-out reads and what the
    /// trace observes as the depth view.
    resolve: Option<DepthResolveObjects>,
    /// The host-visible buffer this pass's depth texels land in, present
    /// exactly when the pass stores the surface (`research/docs/23` §3.3, v43).
    /// A discarded surface is never copied out, so it needs no destination.
    readback: Option<ReadbackObjects>,
    /// The mapping of [`Self::readback`], as the pointer the host reads after
    /// the fence signals — the same shape a colour attachment's readback has.
    mapping: Option<usize>,
}

/// The single-sample resolve target of a stored multisampled depth attachment
/// (`research/docs/23` §3.3, v57).
///
/// The colour sibling's shape for the depth aspect: the four-sample surface's
/// depth texels reduce into this `D32_SFLOAT` image inside the subpass, and
/// the copy-out reads the resolve target rather than the multisampled surface.
struct DepthResolveObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// The Vulkan objects one rail-owned stencil attachment owns
/// (`research/docs/23` §3.3, v47/v49).
///
/// The depth sibling's shape: the rail creates the image, clears it with the
/// value the trace states and — when the trace keeps the surface — copies its
/// one-byte texels into a readback buffer of its own before the pass's objects
/// go with the pass.
struct StencilObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Whether the pass opens the image from the attachment layout a previous
    /// pass left it in (`Load`) or from `UNDEFINED` (a clear).
    loading: bool,
    /// The sample count the image was created with (`research/docs/23` §3.3,
    /// v55): `TYPE_1` for every pre-v55 stencil surface, `TYPE_4` when the pass
    /// states a multisample raster, which the render pass's own description
    /// restates so the two cannot disagree.
    samples: vk::SampleCountFlags,
    /// The single-sample image a stored multisampled stencil surface resolves
    /// into (`research/docs/23` §3.3, v60), or `None` for a surface that
    /// resolves nothing — every pre-v60 stencil surface and every
    /// single-sample stored one. The resolve target is what the copy-out
    /// reads and what the trace observes as the stencil view.
    resolve: Option<StencilResolveObjects>,
    /// The host-visible buffer this pass's stencil texels land in, present
    /// exactly when the pass stores the surface (`research/docs/23` §3.3,
    /// v49). A discarded surface is never copied out, so it needs no
    /// destination.
    readback: Option<ReadbackObjects>,
    /// The mapping of [`Self::readback`], as the pointer the host reads after
    /// the fence signals — the same shape the depth and colour attachments'
    /// readbacks have.
    mapping: Option<usize>,
    /// Whether the image and memory behind this surface are shared with the
    /// depth sibling of a combined depth-stencil attachment
    /// (`research/docs/23` §3.3, v60). The combined shape creates one backing
    /// image the two faces share; the depth half owns it, so this half marks
    /// the flag and skips the free.
    shares_backing: bool,
}

/// The single-sample resolve target of a stored multisampled stencil
/// attachment (`research/docs/23` §3.3, v60).
///
/// The depth sibling's shape for the stencil aspect: the four-sample
/// surface's stencil texels reduce into this `S8_UINT` image inside the
/// subpass, and the copy-out reads the resolve target rather than the
/// multisampled surface.
struct StencilResolveObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Whether the resolve image and its memory are shared with the depth
    /// sibling's resolve of a combined depth-stencil attachment
    /// (`research/docs/23` §3.3, v60). The combined shape resolves both faces
    /// into one `D32_SFLOAT_S8_UINT` landing; the depth half owns it, so this
    /// half marks the flag and skips the free.
    shares_backing: bool,
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
    /// The sample count the image was created with (`research/docs/23` §3.3,
    /// v51): `TYPE_1` for every pre-v51 attachment, `TYPE_4` for a
    /// multisampled pass's own attachment. The render pass and the pipeline
    /// read it back from here so the raster state cannot disagree with the
    /// images the framebuffer binds.
    samples: vk::SampleCountFlags,
    /// The single-sample image a multisampled attachment resolves into
    /// (`research/docs/23` §3.3, v51), or `None` for a single-sample
    /// attachment. The resolve target is what the copy-out reads and what the
    /// trace observes as the attachment view.
    resolve: Option<ResolveObjects>,
    load_op: vk::AttachmentLoadOp,
    /// The attachment's store operation: `STORE` keeps the rendered bytes for
    /// the copy-out, `DONT_CARE` discards them so no readback exists
    /// (`docs/23` §3.6, v19).
    store_op: vk::AttachmentStoreOp,
    initial_layout: vk::ImageLayout,
    /// The single colour every sample of this attachment is seeded with before
    /// a multisampled `Load` opens it (`research/docs/23` §82, v82). `Some`
    /// exactly for the seeded load shape, whose seed pass
    /// ([`OffscreenObjects::create_seed_render_pass`]) clears the image in
    /// place of the `vkCmdCopyBufferToImage` a single-sample load records.
    seed: Option<ClearColor>,
    /// Whether this pass scope created the attachment's own image, memory and
    /// view and has to destroy them on Drop. A single-sample present pass
    /// borrows the provider-owned [`ProviderTargetImage`] for this half, so it
    /// does not own them; a multisampled present pass creates the n-sample
    /// surface itself and owns it (`research/docs/24` §5.2, v62).
    owns_image: bool,
    /// Whether this pass scope created the resolve target beside the
    /// attachment and has to destroy it on Drop. A rail-owned resolve target
    /// is the pass's own; a present pass's resolve target is the
    /// provider-owned present image, which survives the submission
    /// (`research/docs/24` §5.2, v62).
    owns_resolve: bool,
    /// Whether this attachment's bytes leave through the pass's readback
    /// channel: the trace's [`StoreOp::Store`] alone. A resident store keeps
    /// them in the provider's image instead, so the attachment carries a
    /// `STORE` action (the image must keep its bytes) but no readback buffer —
    /// which is why the copy-out pairs buffers by this flag rather than by the
    /// Vulkan store action (`research/docs/23` §76, R7).
    publishes: bool,
    /// The host-visible staging buffer holding this attachment's previous bytes
    /// for a `LoadOp::Load` pass. Null unless the attachment loads.
    previous_buffer: vk::Buffer,
    previous_memory: vk::DeviceMemory,
}

/// The layout guards one pass holds on the provider-owned images it renders
/// into (`research/docs/23` §76, R7).
///
/// The guard is the target's serialization point, exactly as it is for the
/// present action (`research/docs/24` §3.3 rule 1): the value it holds is the
/// `initialLayout` the submission declares, and the terminal layout is
/// published through the same guard before the round trip ends. Acquiring every
/// guard in attachment order before the first device object exists is what
/// keeps two passes that name two resident targets in opposite orders from
/// interleaving their transitions.
struct ResidentTargetLayouts<'a> {
    /// One entry per colour attachment, in location order. `Some` exactly for
    /// the attachments whose declaration named the provider's resident target.
    guards: Vec<Option<std::sync::MutexGuard<'a, vk::ImageLayout>>>,
}

impl<'a> ResidentTargetLayouts<'a> {
    fn acquire(request: &'a OffscreenRenderRequest<'a>) -> Self {
        let guards = request
            .attachments
            .iter()
            .map(|attachment| {
                attachment
                    .resident
                    .map(ProviderTargetImage::begin_target_pass)
            })
            .collect();
        Self { guards }
    }

    /// The layout the attachment's submission has to declare as its
    /// `initialLayout`, or `UNDEFINED` for an attachment that renders into a
    /// per-pass image of its own.
    fn layout(&self, index: usize) -> vk::ImageLayout {
        self.guards
            .get(index)
            .and_then(|guard| guard.as_ref())
            .map_or(vk::ImageLayout::UNDEFINED, |guard| **guard)
    }

    /// Publish the terminal layout of every resident image: the layout the next
    /// submission starts from once the pass completed, or `UNDEFINED` when a
    /// submission that reached the queue failed and the image's state is
    /// unknown. `UNDEFINED` is the one old layout that is always legal to
    /// declare, and the provider marks the identity undefined in the same
    /// case, so a later `LoadOp::Resident` refuses instead of reading it.
    fn publish(&mut self, completed: bool) {
        let terminal = if completed {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        for guard in self.guards.iter_mut().flatten() {
            **guard = terminal;
        }
    }
}

/// The single-sample resolve target of one multisampled colour attachment
/// (`research/docs/23` §3.3, v51).
///
/// The image is owned by the pass exactly as its multisampled sibling is; the
/// resolve runs as part of the subpass, so its final layout is the copy-out's
/// `TRANSFER_SRC_OPTIMAL` when the pass keeps the bytes.
struct ResolveObjects {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// The layout the image is actually in when the render pass opens it.
    /// A rail-owned resolve target is freshly created and opens from
    /// `UNDEFINED`; a present pass's resolve target is the provider-owned
    /// [`ProviderTargetImage`], which a sentinel preset or an earlier present
    /// has already moved, so its description has to restate that layout
    /// (`research/docs/24` §3.3 rule 1, v62).
    initial_layout: vk::ImageLayout,
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
            depth: None,
            stencil: None,
            present: false,
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            seed_render_pass: vk::RenderPass::null(),
            seed_framebuffer: vk::Framebuffer::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            vertex_module: vk::ShaderModule::null(),
            fragment_module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            readbacks: Vec::new(),
            textures: Vec::new(),
            submitted: false,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            indirect_buffer: vk::Buffer::null(),
            indirect_memory: vk::DeviceMemory::null(),
            index_buffer: vk::Buffer::null(),
            index_memory: vk::DeviceMemory::null(),
            vertex_inputs: Vec::new(),
            input_index_buffer: vk::Buffer::null(),
            input_index_memory: vk::DeviceMemory::null(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            base_vertex: 0,
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
    /// A single-sample raster renders straight into the target. A multisampled
    /// raster (`research/docs/24` §3.5, v62) creates its own n-sample colour
    /// surface beside the target and resolves into the target, so the
    /// single-sample texels the present hands on are the resolve's own
    /// landing.
    ///
    /// `initial_layout` is passed in rather than read from the target: the
    /// caller holds the target's present round-trip guard, and the guard's
    /// value *is* the layout this submission must declare — the attachment's
    /// own for the single-sample shape, the resolve attachment's for the
    /// multisampled one.
    fn attach_present_target(
        &mut self,
        target: &ProviderTargetImage,
        initial_layout: vk::ImageLayout,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<(), ProviderError> {
        if samples == vk::SampleCountFlags::TYPE_1 {
            self.attachments.push(AttachmentObjects {
                image: target.image(),
                memory: vk::DeviceMemory::null(),
                view: target.view(),
                samples: vk::SampleCountFlags::TYPE_1,
                resolve: None,
                load_op: vk::AttachmentLoadOp::CLEAR,
                // A present target is the observable landing of the pass, so
                // its store is always `STORE`; `execute_present_render`
                // refuses a `StoreOp::DontCare` present attachment before
                // this runs.
                store_op: vk::AttachmentStoreOp::STORE,
                initial_layout,
                seed: None,
                // The target itself is the provider's: the pass scope destroys
                // none of its image, memory or view (`docs/24` §5.2).
                owns_image: false,
                owns_resolve: false,
                publishes: true,
                previous_buffer: vk::Buffer::null(),
                previous_memory: vk::DeviceMemory::null(),
            });
            self.present = true;
            return Ok(());
        }
        // The multisampled present shape: a rail-owned n-sample surface the
        // fragments land in, resolved into the provider-owned target. The
        // n-sample surface opens from a clear — the reviewed multisample load
        // shape — and is consumed by the resolve, so it carries neither
        // transfer usage nor a readback of its own.
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
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (image, memory, _) = crate::allocate_image_backing(
            self.context,
            &info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "present attachment",
        )
        .map_err(|error| execution_refusal("create present attachment image", &error.detail))?;
        let view =
            crate::create_color_image_view(self.context, image, format, "present attachment")
                .map_err(|error| {
                    execution_refusal("create present attachment view", &error.detail)
                })?;
        self.attachments.push(AttachmentObjects {
            image,
            memory,
            view,
            samples,
            // The resolve target is the present image itself: the pass scope
            // borrows it and never destroys it (`docs/24` §5.2). The resolve
            // writes every texel, so its load operation is `DONT_CARE`; its
            // initial layout is the target's own current layout, not
            // `UNDEFINED`, because the sentinel preset or an earlier present
            // has already moved the image.
            resolve: Some(ResolveObjects {
                image: target.image(),
                memory: vk::DeviceMemory::null(),
                view: target.view(),
                initial_layout,
            }),
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            // The n-sample surface opens from a clear, so nothing defines its
            // bytes before the pass.
            initial_layout: vk::ImageLayout::UNDEFINED,
            seed: None,
            owns_image: true,
            owns_resolve: false,
            publishes: true,
            previous_buffer: vk::Buffer::null(),
            previous_memory: vk::DeviceMemory::null(),
        });
        self.present = true;
        Ok(())
    }

    /// Render this attachment into the provider-owned resident target instead
    /// of a per-pass image (`research/docs/23` §76, R7).
    ///
    /// The image and view are borrowed for the pass's lifetime; the provider's
    /// registry keeps owning them, exactly as the present rail borrows its
    /// target (`docs/24` §5.2). The load decision is the trace's own: a
    /// `Resident` load keeps the image's contents (`LOAD` from the layout the
    /// provider published), a `Clear` clears it (`CLEAR`, opened from
    /// `UNDEFINED` because the pre-pass contents are discarded either way), and
    /// a `Load` uploads the trace's declared previous bytes into it first
    /// (`LOAD` from `COLOR_ATTACHMENT_OPTIMAL`, which is where the upload
    /// leaves the image).
    ///
    /// The store action is `STORE` for both store arms: a resident store keeps
    /// the bytes in the image, and a [`StoreOp::Store`] publishes them through
    /// the readback channel as well. What the two disagree about is
    /// [`AttachmentObjects::publishes`], which is the flag the copy-out pairs
    /// its buffers by.
    fn attach_resident_target(
        &mut self,
        index: usize,
        target: &ProviderTargetImage,
        layout: vk::ImageLayout,
        attachment: &OffscreenColorAttachment<'_>,
    ) -> Result<(), ProviderError> {
        let uploading = attachment.previous.is_some();
        let load_op = match attachment.load {
            LoadOp::Resident | LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
            LoadOp::DontCare => {
                // Core admission refuses this pair (a resident target whose
                // pre-pass contents are undefined), so reaching it means a
                // hand-built request skipped that gate. The rail refuses it by
                // name rather than letting the image hold bytes no pass
                // defined.
                return Err(capability_refusal("resident_target_undeclared")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("load_op", FieldValue::Text("dont_care".to_owned()))
                    .with_detail(
                        "a resident attachment defines its contents before the pass: \
                         `DontCare` is refused",
                    ));
            }
        };
        // Three load arms, three `initialLayout`s — and the rule is which of
        // them the render pass may *keep*:
        //
        // - a `Load` uploads the trace's own bytes into the image and its
        //   barriers leave it in the attachment layout, so that is the layout
        //   the pass declares (`research/docs/23` §3.3, R5b);
        // - a `Resident` load keeps the image's own bytes, so the pass declares
        //   the layout the provider published through the guard;
        // - a `Clear` discards whatever the image held, so it opens from
        //   `UNDEFINED` — the layout that is legal from any state.
        //
        // Declaring `UNDEFINED` beside a `LOAD` action is the one combination
        // that must not happen: it is the "contents are discarded" spelling.
        let initial_layout = if uploading {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else if matches!(attachment.load, LoadOp::Clear(_)) {
            vk::ImageLayout::UNDEFINED
        } else {
            layout
        };
        self.attachments.push(AttachmentObjects {
            image: target.image(),
            memory: vk::DeviceMemory::null(),
            view: target.view(),
            // A resident raster is single-sample in this increment: the
            // multisampled shape resolves into a landing of its own, which the
            // executor refuses before this runs.
            samples: vk::SampleCountFlags::TYPE_1,
            resolve: None,
            load_op,
            store_op: vk::AttachmentStoreOp::STORE,
            initial_layout,
            seed: None,
            // The image is the provider's: this pass scope destroys none of
            // its image, memory or view (`research/docs/23` §76, R7).
            owns_image: false,
            owns_resolve: false,
            publishes: attachment.store == StoreOp::Store,
            previous_buffer: vk::Buffer::null(),
            previous_memory: vk::DeviceMemory::null(),
        });
        Ok(())
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
    /// Create the depth image of a pass that declares one.
    ///
    /// The image is a `D32_SFLOAT` depth attachment. A pass that keeps the
    /// surface (`research/docs/23` §3.3, v43) also asks for `TRANSFER_SRC`,
    /// because its texels leave through `vkCmdCopyImageToBuffer`; a discarded
    /// surface needs no transfer usage and stays in its attachment layout after
    /// the pass, exactly as every pre-v43 depth image did. `Load` opens the
    /// image from the attachment layout a previous pass left it in — which only
    /// a trace that wrote it in the same submission can rely on — and a clear
    /// opens it from `UNDEFINED`.
    fn create_depth(
        &mut self,
        width: u32,
        height: u32,
        loading: bool,
        storing: bool,
        samples: vk::SampleCountFlags,
        resolve: Option<DepthResolveFilter>,
    ) -> Result<(), ProviderError> {
        // A resolving pass keeps its depth surface through the single-sample
        // resolve target, so the four-sample image itself is never copied out;
        // the resolve target below carries the transfer usage instead, the
        // same split the multisampled colour attachment states
        // (`research/docs/23` §3.3, v57).
        let resolving = resolve.is_some();
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | if storing && !resolving {
                        vk::ImageUsageFlags::TRANSFER_SRC
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
            "depth",
        )
        .map_err(|error| execution_refusal("create depth image", &error.detail))?;
        let view =
            crate::create_depth_image_view(self.context, image, vk::Format::D32_SFLOAT, "depth")
                .map_err(|error| execution_refusal("create depth view", &error.detail))?;
        // The resolve target of a stored multisampled depth surface
        // (`research/docs/23` §3.3, v57): one single-sample `D32_SFLOAT` image
        // created beside its four-sample sibling, so the render pass and the
        // framebuffer can name both. It carries `TRANSFER_SRC` because it is
        // the surface the copy-out reads; the filter itself is the subpass's
        // resolve state, not an image property, so the image carries no
        // filter field.
        let resolve_objects = if resolving {
            let resolve_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
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
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let (resolve_image, resolve_memory, _) = crate::allocate_image_backing(
                self.context,
                &resolve_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "depth resolve",
            )
            .map_err(|error| execution_refusal("create depth resolve image", &error.detail))?;
            let resolve_view = crate::create_depth_image_view(
                self.context,
                resolve_image,
                vk::Format::D32_SFLOAT,
                "depth resolve",
            )
            .map_err(|error| execution_refusal("create depth resolve view", &error.detail))?;
            Some(DepthResolveObjects {
                image: resolve_image,
                memory: resolve_memory,
                view: resolve_view,
            })
        } else {
            None
        };
        self.depth = Some(DepthObjects {
            image,
            memory,
            view,
            format: vk::Format::D32_SFLOAT,
            loading,
            samples,
            resolve: resolve_objects,
            readback: None,
            mapping: None,
        });
        Ok(())
    }

    /// Create the rail-owned stencil image of a pass that declares one
    /// (`research/docs/23` §3.3, v47/v49).
    ///
    /// The image is a `VK_FORMAT_S8_UINT` stencil attachment opened from
    /// `UNDEFINED` for a clear and from the attachment layout for a load. It
    /// carries `DEPTH_STENCIL_ATTACHMENT` and, when the pass keeps the
    /// surface, `TRANSFER_SRC` — the same pair the depth path states since
    /// v43: a discarded surface is never copied out and must not be refused
    /// for a feature its execution never needs.
    fn create_stencil(
        &mut self,
        width: u32,
        height: u32,
        loading: bool,
        storing: bool,
        samples: vk::SampleCountFlags,
        resolve: Option<StencilResolveFilter>,
    ) -> Result<(), ProviderError> {
        // A resolving pass keeps its stencil surface through the
        // single-sample resolve target, so the four-sample image itself is
        // never copied out; the resolve target below carries the transfer
        // usage instead, the same split the multisampled colour attachment
        // states (`research/docs/23` §3.3, v60).
        let resolving = resolve.is_some();
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::S8_UINT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | if storing && !resolving {
                        vk::ImageUsageFlags::TRANSFER_SRC
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
            "stencil",
        )
        .map_err(|error| execution_refusal("create stencil image", &error.detail))?;
        let view =
            crate::create_stencil_image_view(self.context, image, vk::Format::S8_UINT, "stencil")
                .map_err(|error| execution_refusal("create stencil view", &error.detail))?;
        // The resolve target of a stored multisampled stencil surface
        // (`research/docs/23` §3.3, v60): one single-sample `S8_UINT` image
        // created beside its four-sample sibling, so the render pass and the
        // framebuffer can name both. It carries `TRANSFER_SRC` because it is
        // the surface the copy-out reads; the filter itself is the subpass's
        // resolve state, not an image property, so the image carries no
        // filter field.
        let resolve_objects = if resolving {
            let resolve_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::S8_UINT)
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
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let (resolve_image, resolve_memory, _) = crate::allocate_image_backing(
                self.context,
                &resolve_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "stencil resolve",
            )
            .map_err(|error| execution_refusal("create stencil resolve image", &error.detail))?;
            let resolve_view = crate::create_stencil_image_view(
                self.context,
                resolve_image,
                vk::Format::S8_UINT,
                "stencil resolve",
            )
            .map_err(|error| execution_refusal("create stencil resolve view", &error.detail))?;
            Some(StencilResolveObjects {
                image: resolve_image,
                memory: resolve_memory,
                view: resolve_view,
                shares_backing: false,
            })
        } else {
            None
        };
        self.stencil = Some(StencilObjects {
            image,
            memory,
            view,
            loading,
            samples,
            resolve: resolve_objects,
            readback: None,
            mapping: None,
            shares_backing: false,
        });
        Ok(())
    }

    /// Create the combined depth-stencil surface of a pass that opens both
    /// faces (`research/docs/23` §3.3, v60/v66).
    ///
    /// Vulkan binds one attachment for both faces, so the two surfaces this
    /// function builds share one combined image: the depth half owns the
    /// backing, the stencil half marks `shares_backing` and lets the depth
    /// half free it. The `resolve` flag is the pass's own store decision: the
    /// v60 shape keeps both faces, so the one single-sample landing both
    /// resolves reduce into is built and shared the same way; the v66
    /// rail-owned pair keeps neither face, so no landing exists and both
    /// surfaces end with no resolve target. Both faces open from a clear —
    /// every reviewed combined shape does — so a loading combined surface is
    /// refused by `prepare_render_request` before this runs.
    fn create_combined_depth_stencil(
        &mut self,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        resolve: bool,
    ) -> Result<(), ProviderError> {
        let combined_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (combined_image, combined_memory, _) = crate::allocate_image_backing(
            self.context,
            &combined_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "combined depth-stencil",
        )
        .map_err(|error| execution_refusal("create combined depth-stencil image", &error.detail))?;
        // The framebuffer binds one view per attachment: the combined
        // attachment's view covers both aspects, which is what
        // `create_depth_stencil_image_view` builds. The stencil half keeps an
        // aspect-only view for its own identity; the copy-outs read the
        // images' aspects directly and never use a view.
        let depth_view = crate::create_depth_stencil_image_view(
            self.context,
            combined_image,
            format,
            "combined depth",
        )
        .map_err(|error| execution_refusal("create combined depth view", &error.detail))?;
        let stencil_view = crate::create_stencil_image_view(
            self.context,
            combined_image,
            format,
            "combined stencil",
        )
        .map_err(|error| execution_refusal("create combined stencil view", &error.detail))?;
        if !resolve {
            // The v66 rail-owned pair: one surface, no landing, both faces
            // discarded with the pass (`research/docs/23` §3.3, v66).
            self.depth = Some(DepthObjects {
                image: combined_image,
                memory: combined_memory,
                view: depth_view,
                format,
                loading: false,
                samples,
                resolve: None,
                readback: None,
                mapping: None,
            });
            self.stencil = Some(StencilObjects {
                image: combined_image,
                memory: combined_memory,
                view: stencil_view,
                loading: false,
                samples,
                resolve: None,
                readback: None,
                mapping: None,
                shares_backing: true,
            });
            return Ok(());
        }
        // The one single-sample landing both resolves reduce into: a
        // combined-format image whose depth and stencil aspects are the two
        // readback sources, exactly as the two surfaces' own aspects are the
        // two faces of the four-sample attachment.
        let resolve_info = vk::ImageCreateInfo::default()
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
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let (resolve_image, resolve_memory, _) = crate::allocate_image_backing(
            self.context,
            &resolve_info,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "combined depth-stencil resolve",
        )
        .map_err(|error| {
            execution_refusal("create combined depth-stencil resolve image", &error.detail)
        })?;
        let depth_resolve_view = crate::create_depth_stencil_image_view(
            self.context,
            resolve_image,
            format,
            "combined depth resolve",
        )
        .map_err(|error| execution_refusal("create combined depth resolve view", &error.detail))?;
        let stencil_resolve_view = crate::create_stencil_image_view(
            self.context,
            resolve_image,
            format,
            "combined stencil resolve",
        )
        .map_err(|error| {
            execution_refusal("create combined stencil resolve view", &error.detail)
        })?;
        self.depth = Some(DepthObjects {
            image: combined_image,
            memory: combined_memory,
            view: depth_view,
            format,
            loading: false,
            samples,
            resolve: Some(DepthResolveObjects {
                image: resolve_image,
                memory: resolve_memory,
                view: depth_resolve_view,
            }),
            readback: None,
            mapping: None,
        });
        self.stencil = Some(StencilObjects {
            image: combined_image,
            memory: combined_memory,
            view: stencil_view,
            loading: false,
            samples,
            resolve: Some(StencilResolveObjects {
                image: resolve_image,
                memory: resolve_memory,
                view: stencil_resolve_view,
                shares_backing: true,
            }),
            readback: None,
            mapping: None,
            shares_backing: true,
        });
        Ok(())
    }

    // One argument per fact the reviewed shape states — the format, the two
    // dimensions, the load op, its seed, the store decision and the raster's
    // sample count — the same spelling the request itself carries.
    #[allow(clippy::too_many_arguments)]
    fn create_attachment(
        &mut self,
        format: vk::Format,
        width: u32,
        height: u32,
        load: LoadOp,
        seed: Option<ClearColor>,
        storing: bool,
        samples: vk::SampleCountFlags,
    ) -> Result<(), ProviderError> {
        let loading = matches!(load, LoadOp::Load);
        // A multisampled attachment's `Load` is seeded by a clear inside a
        // rail-owned render pass (`research/docs/23` §82, v82), so the image
        // is never a transfer destination: `vkCmdCopyBufferToImage` cannot
        // address it (`VUID-vkCmdCopyBufferToImage-dstImage-07973`) and asking
        // for the usage would put a combination the transfer stage can never
        // use in front of `vkCreateImage`.
        let single_sample = samples == vk::SampleCountFlags::TYPE_1;
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
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    // A multisampled attachment's bytes are consumed by the
                    // resolve inside the subpass, so the multisampled image
                    // itself is never copied out; the resolve target below
                    // carries the transfer usage instead
                    // (`research/docs/23` §3.3, v51).
                    | if storing && samples == vk::SampleCountFlags::TYPE_1 {
                        vk::ImageUsageFlags::TRANSFER_SRC
                    } else {
                        vk::ImageUsageFlags::empty()
                    }
                    // A loading attachment receives its previous bytes through
                    // `vkCmdCopyBufferToImage`, so the image needs the transfer
                    // destination usage exactly when one is uploaded
                    // (`research/docs/23` §3.3).
                    | if loading && single_sample {
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
        // The resolve target of a multisampled attachment
        // (`research/docs/23` §3.3, v51): one single-sample image per colour
        // location, created beside its multisampled sibling so the render pass
        // and the framebuffer can name both. `TRANSFER_SRC` is asked of it
        // exactly when the pass keeps the bytes, the same rule the
        // single-sample attachment states above.
        let resolve = if samples == vk::SampleCountFlags::TYPE_1 {
            None
        } else {
            let resolve_info = vk::ImageCreateInfo::default()
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
                        },
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let (resolve_image, resolve_memory, _) = crate::allocate_image_backing(
                self.context,
                &resolve_info,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                "resolve attachment",
            )
            .map_err(|error| execution_refusal("create resolve image", &error.detail))?;
            let resolve_view =
                crate::create_color_image_view(self.context, resolve_image, format, "resolve")
                    .map_err(|error| execution_refusal("create resolve view", &error.detail))?;
            Some(ResolveObjects {
                image: resolve_image,
                memory: resolve_memory,
                view: resolve_view,
                // A rail-owned resolve target is created for this pass, so
                // nothing has defined its bytes and the render pass opens it
                // from `UNDEFINED` (`research/docs/23` §3.3, v51).
                initial_layout: vk::ImageLayout::UNDEFINED,
            })
        };
        self.attachments.push(AttachmentObjects {
            image,
            memory,
            view,
            samples,
            resolve,
            // A loading attachment keeps the upload's layout as the pass's
            // initial one; a clearing or `DontCare` attachment opens from
            // `UNDEFINED`, because nothing defines its bytes before the pass
            // (`research/docs/23` §3.3, `docs/23` §3.1 v20).
            load_op: match load {
                LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                LoadOp::Clear(_) => vk::AttachmentLoadOp::CLEAR,
                LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                // A resident load never creates a per-pass attachment image:
                // the pass renders into the provider's own image, which is
                // [`OffscreenObjects::attach_resident_target`]'s half
                // (`research/docs/23` §76, R7). Reaching this arm means the
                // request's two declaration lists disagreed, so the rail
                // refuses instead of creating an image the trace did not ask
                // for.
                LoadOp::Resident => return Err(capability_refusal("resident_target_undeclared")
                    .with_detail(
                        "a `LoadOp::Resident` attachment renders into the provider's own image; \
                         this request handed the rail no resident target for it",
                    )),
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
            // The seed the rail's own render pass clears every sample with
            // (`research/docs/23` §82, v82); `None` for every other attachment.
            seed,
            publishes: storing,
            // An offscreen attachment and its resolve target are both created
            // by this pass scope and destroyed with it
            // (`research/docs/23` §3.3, v51).
            owns_image: true,
            owns_resolve: true,
            previous_buffer: vk::Buffer::null(),
            previous_memory: vk::DeviceMemory::null(),
        });
        Ok(())
    }

    /// The seed pass a multisampled `Load` is executed with
    /// (`research/docs/23` §82, v82).
    ///
    /// One subpass over the seeded attachments, in location order: each is a
    /// `CLEAR`-opened, `STORE`d description of the same multisampled image the
    /// measured pass then opens with `LOAD`. A clear writes its value to every
    /// sample of the render area, which is what makes the samples defined
    /// before the measured pass reads them; the seed pass leaves each image in
    /// `COLOR_ATTACHMENT_OPTIMAL`, the layout the measured pass declares as its
    /// initial one. The two dependencies are the pair the load needs: the
    /// external→subpass one scopes the clear's own destination access, and the
    /// subpass→external one hands the writes on as colour-attachment reads and
    /// writes — a load is a read of the attachment, so an availability-only
    /// hand-off would leave it unsynchronized.
    ///
    /// Nothing is created when the scope seeds no attachment, so every earlier
    /// shape keeps exactly the objects it had.
    fn create_seed_render_pass(&mut self, vk_formats: &[vk::Format]) -> Result<(), ProviderError> {
        let seeded = self
            .attachments
            .iter()
            .enumerate()
            .filter(|(_, attachment)| attachment.seed.is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if seeded.is_empty() {
            return Ok(());
        }
        let descriptions = seeded
            .iter()
            .map(|index| {
                let attachment = &self.attachments[*index];
                vk::AttachmentDescription2::default()
                    .format(vk_formats[*index])
                    .samples(attachment.samples)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            })
            .collect::<Vec<_>>();
        let refs = (0..descriptions.len())
            .map(|index| {
                vk::AttachmentReference2::default()
                    .attachment(index as u32)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            })
            .collect::<Vec<_>>();
        let subpasses = [vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&refs)];
        let dependencies = [
            vk::SubpassDependency2::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency2::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
        ];
        let info = vk::RenderPassCreateInfo2::default()
            .attachments(&descriptions)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        self.seed_render_pass = unsafe { self.context.device.create_render_pass2(&info, None) }
            .map_err(|error| execution_refusal("create seed render pass", &error.to_string()))?;
        Ok(())
    }

    /// The framebuffer of [`Self::create_seed_render_pass`]: the seeded
    /// attachments' own views, in the same order the seed pass lists them.
    /// Nothing is created when the scope seeds no attachment.
    fn create_seed_framebuffer(&mut self, width: u32, height: u32) -> Result<(), ProviderError> {
        if self.seed_render_pass == vk::RenderPass::null() {
            return Ok(());
        }
        let views = self
            .attachments
            .iter()
            .filter(|attachment| attachment.seed.is_some())
            .map(|attachment| attachment.view)
            .collect::<Vec<_>>();
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(self.seed_render_pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        self.seed_framebuffer = unsafe { self.context.device.create_framebuffer(&info, None) }
            .map_err(|error| execution_refusal("create seed framebuffer", &error.to_string()))?;
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
    fn create_render_pass(
        &mut self,
        formats: &[vk::Format],
        depth: Option<&OffscreenDepthAttachment>,
        stencil: Option<&OffscreenStencilAttachment>,
        depth_resolve: Option<DepthResolveFilter>,
        stencil_resolve: Option<StencilResolveFilter>,
    ) -> Result<(), ProviderError> {
        // The multisampled shape (`research/docs/23` §3.3, v51) lists two
        // attachment descriptions per colour location — the four-sample
        // surface the fragments land in, then the single-sample resolve target
        // — so the resolve references below are the colour locations shifted
        // by the attachment count. The single-sample shape keeps the list and
        // the indices it always had.
        let multisampled = self
            .attachments
            .iter()
            .any(|attachment| attachment.samples != vk::SampleCountFlags::TYPE_1);
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
                vk::AttachmentDescription2::default()
                    .format(*format)
                    .samples(attachment.samples)
                    .load_op(attachment.load_op)
                    // A multisampled attachment's own contents are consumed by
                    // the resolve inside the subpass, so the surface itself is
                    // never stored; the resolve target below carries the pass's
                    // own store decision (`research/docs/23` §3.3, v51).
                    .store_op(if attachment.samples != vk::SampleCountFlags::TYPE_1 {
                        vk::AttachmentStoreOp::DONT_CARE
                    } else {
                        attachment.store_op
                    })
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(attachment.initial_layout)
                    .final_layout(final_layout)
            })
            .chain(
                self.attachments
                    .iter()
                    .zip(formats)
                    .filter(|(attachment, _)| attachment.samples != vk::SampleCountFlags::TYPE_1)
                    .map(|(attachment, format)| {
                        // A resolve attachment's load operation is `DONT_CARE`
                        // by construction: the resolve writes every texel, so
                        // the pre-pass contents of the single-sample image are
                        // never read. Its store action is the pass's own, and a
                        // storing resolve ends in `TRANSFER_SRC_OPTIMAL` for the
                        // copy-out exactly as a single-sample stored attachment
                        // does (`research/docs/23` §3.3, v51). A present
                        // pass's resolve target is the provider-owned present
                        // image, whose sentinel preset or earlier present has
                        // already chosen a layout, so the description restates
                        // that layout instead of opening from `UNDEFINED`
                        // (`research/docs/24` §3.3 rule 1, v62).
                        let final_layout = if attachment.store_op == vk::AttachmentStoreOp::STORE
                            && !self.present
                        {
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                        } else {
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        };
                        vk::AttachmentDescription2::default()
                            .format(*format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(attachment.store_op)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(
                                attachment
                                    .resolve
                                    .as_ref()
                                    .map_or(vk::ImageLayout::UNDEFINED, |resolve| {
                                        resolve.initial_layout
                                    }),
                            )
                            .final_layout(final_layout)
                    }),
            )
            .chain(depth.map(|_| {
                let combined = stencil.is_some();
                if combined {
                    // The combined depth-stencil attachment
                    // (`research/docs/23` §3.3, v60/v66): one surface both
                    // faces share, so its two load operations open the two
                    // aspects from the reviewed clear. In the v60 shape its
                    // four-sample contents are consumed by the two resolves
                    // inside the subpass and the combined resolve target below
                    // carries the pass's store decision; in the v66 rail-owned
                    // pair neither face is kept, so the attachment simply
                    // discards both aspects and there is no target below.
                    let samples = self
                        .depth
                        .as_ref()
                        .map_or(vk::SampleCountFlags::TYPE_1, |objects| objects.samples);
                    // The format is the one the image was created with, so the
                    // description cannot disagree with the framebuffer's view
                    // (`research/docs/23` §3.3, v66).
                    let format = self
                        .depth
                        .as_ref()
                        .map_or(vk::Format::D32_SFLOAT_S8_UINT, |objects| objects.format);
                    return vk::AttachmentDescription2::default()
                        .format(format)
                        .samples(samples)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                }
                // The depth attachment: opened from `UNDEFINED` for a clear and
                // from the attachment layout for a load
                // (`research/docs/23` §3.3, v36). A pass that states no store
                // action discards the surface and leaves it in its attachment
                // layout; a storing pass ends in `TRANSFER_SRC_OPTIMAL`, so the
                // copy-out below runs without a further barrier (v43).
                // A multisampled pass creates the surface with the raster's own
                // sample count (v53), so the description restates what the
                // image was built with.
                let loading = self.depth.as_ref().is_some_and(|objects| objects.loading);
                let storing = self
                    .depth
                    .as_ref()
                    .is_some_and(|objects| objects.readback.is_some());
                let resolving = self
                    .depth
                    .as_ref()
                    .is_some_and(|objects| objects.resolve.is_some());
                let samples = self
                    .depth
                    .as_ref()
                    .map_or(vk::SampleCountFlags::TYPE_1, |objects| objects.samples);
                vk::AttachmentDescription2::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(samples)
                    .load_op(if loading {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::CLEAR
                    })
                    // A resolving surface's own contents are consumed by the
                    // depth resolve inside the subpass, so the four-sample
                    // image itself is never stored; the resolve target below
                    // carries the pass's store decision
                    // (`research/docs/23` §3.3, v57).
                    .store_op(if storing && !resolving {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(if loading {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .final_layout(if storing && !resolving {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    })
            }))
            .chain(depth_resolve.map(|_| {
                if stencil.is_some() {
                    // The combined resolve target (`research/docs/23` §3.3,
                    // v60): one single-sample combined-format image both
                    // resolves reduce into. Its load operations are
                    // `DONT_CARE` by construction and it ends in
                    // `TRANSFER_SRC_OPTIMAL` for both copy-outs.
                    let format = self
                        .depth
                        .as_ref()
                        .map_or(vk::Format::D32_SFLOAT_S8_UINT, |objects| objects.format);
                    return vk::AttachmentDescription2::default()
                        .format(format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::STORE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                }
                // The depth resolve target (`research/docs/23` §3.3, v57): a
                // single-sample `D32_SFLOAT` image the four-sample surface's
                // depth texels reduce into. Its load operation is `DONT_CARE`
                // by construction — the resolve writes every texel — and it
                // ends in `TRANSFER_SRC_OPTIMAL` for the copy-out, exactly as
                // a stored single-sample depth attachment does.
                vk::AttachmentDescription2::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            }))
            .chain(stencil.filter(|_| depth.is_none()).map(|_| {
                // The rail-owned stencil attachment (`research/docs/23` §3.3,
                // v47): a `S8_UINT` surface whose *stencil* load and store
                // operations carry the pass's decision — the depth aspect's
                // pair is `DONT_CARE` for a format that has no depth aspect.
                // A clearing pass opens it from `UNDEFINED`, a loading one from
                // the attachment layout a previous pass left it in. A pass
                // that states no store action discards the surface and leaves
                // it in that layout; a storing pass ends in
                // `TRANSFER_SRC_OPTIMAL`, so the copy-out below runs without a
                // further barrier (v49).
                let storing = self
                    .stencil
                    .as_ref()
                    .is_some_and(|objects| objects.readback.is_some());
                // A resolving pass's four-sample contents are consumed by the
                // stencil resolve inside the subpass, so the multisampled
                // image itself is never stored; the resolve target below
                // carries the pass's store decision
                // (`research/docs/23` §3.3, v60).
                let resolving = self
                    .stencil
                    .as_ref()
                    .is_some_and(|objects| objects.resolve.is_some());
                // The stencil surface is created with the raster's own sample
                // count when the pass states one (`research/docs/23` §3.3,
                // v55), so the description restates what the image was built
                // with.
                let samples = self
                    .stencil
                    .as_ref()
                    .map_or(vk::SampleCountFlags::TYPE_1, |objects| objects.samples);
                vk::AttachmentDescription2::default()
                    .format(vk::Format::S8_UINT)
                    .samples(samples)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(
                        if self.stencil.as_ref().is_some_and(|objects| objects.loading) {
                            vk::AttachmentLoadOp::LOAD
                        } else {
                            vk::AttachmentLoadOp::CLEAR
                        },
                    )
                    .stencil_store_op(if storing && !resolving {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(
                        if self.stencil.as_ref().is_some_and(|objects| objects.loading) {
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        } else {
                            vk::ImageLayout::UNDEFINED
                        },
                    )
                    .final_layout(if storing && !resolving {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    })
            }))
            .chain(
                stencil_resolve
                    .and(stencil.filter(|_| depth.is_none()))
                    .map(|_| {
                        // The stencil resolve target (`research/docs/23` §3.3,
                        // v60): a single-sample `S8_UINT` image the
                        // four-sample surface's stencil texels reduce into.
                        // Its load operation is `DONT_CARE` by construction —
                        // the resolve writes every texel — and it ends in
                        // `TRANSFER_SRC_OPTIMAL` for the copy-out, exactly as
                        // a stored single-sample stencil attachment does.
                        vk::AttachmentDescription2::default()
                            .format(vk::Format::S8_UINT)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::STORE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    }),
            )
            .collect::<Vec<_>>();
        let color_refs = (0..self.attachments.len())
            .map(|index| {
                vk::AttachmentReference2::default()
                    .attachment(index as u32)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            })
            .collect::<Vec<_>>();
        // The resolve references of a multisampled pass, one per colour
        // location: entry `i` names the single-sample image created beside
        // attachment `i`, i.e. the attachment list's second half
        // (`research/docs/23` §3.3, v51). A single-sample pass keeps the
        // subpass it always had.
        let resolve_refs = multisampled.then(|| {
            (0..self.attachments.len())
                .map(|index| {
                    vk::AttachmentReference2::default()
                        .attachment((self.attachments.len() + index) as u32)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect::<Vec<_>>()
        });
        // The depth reference follows the colour references, so its attachment
        // index is the colour count (`research/docs/23` §3.3, v36).
        // The stencil attachment takes the same reference slot as the depth one
        // when it is the only depth-stencil surface the pass opens; the two are
        // mutually exclusive in this increment (`research/docs/23` §3.3, v47).
        let depth_ref = depth
            .and(Some(()))
            .or_else(|| stencil.map(|_| ()))
            .map(|_| {
                vk::AttachmentReference2::default()
                    // The depth-stencil surface always follows the colour
                    // attachments: a pass carries one such reference and the one
                    // surface behind it, so the index is the colour count in both
                    // the depth and the stencil-only case (`research/docs/23` §3.3,
                    // v36/v47). A multisampled pass lists the resolve targets
                    // between the two, so the surface follows *both* halves of
                    // the colour list (v51); no multisampled pass carries the
                    // surface today, and the index stays correct if one does.
                    .attachment((self.attachments.len() * if multisampled { 2 } else { 1 }) as u32)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            });
        // The depth resolve reference (`research/docs/23` §3.3, v57): the
        // single-sample landing follows the four-sample depth surface, which
        // itself follows both halves of the colour list, so its index is one
        // past the depth reference's. The stencil-only resolve takes the same
        // slot — the pass opens one depth-stencil surface, so the stencil
        // landing is one past the stencil surface's own reference
        // (`research/docs/23` §3.3, v60).
        let depth_resolve_ref = depth_resolve.map(|_| {
            vk::AttachmentReference2::default()
                .attachment((self.attachments.len() * if multisampled { 2 } else { 1 } + 1) as u32)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        // The resolve reference of a stencil-only resolving pass: the
        // single-sample `S8_UINT` landing follows the four-sample stencil
        // surface, whose own reference occupies the depth slot, so its index
        // is one past that slot's (`research/docs/23` §3.3, v60).
        let stencil_resolve_ref = stencil_resolve.map(|_| {
            vk::AttachmentReference2::default()
                .attachment((self.attachments.len() * if multisampled { 2 } else { 1 } + 1) as u32)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        // The depth-stencil resolve state the subpass carries
        // (`research/docs/23` §3.3, v57/v60): the filter is the pass's own
        // statement and the resolve reference is the single-sample landing
        // above. Depth and stencil resolve are mutually exclusive on this rail
        // (the combined surface is refused), so one slot states its filter
        // while the other stays `NONE`; the struct lives beside the subpass so
        // its pNext chain stays valid for the `create_render_pass2` call.
        let mut depth_stencil_resolve = vk::SubpassDescriptionDepthStencilResolve::default();
        if let (Some(resolve), Some(depth_resolve_ref)) = (depth_resolve, &depth_resolve_ref) {
            depth_stencil_resolve = depth_stencil_resolve
                .depth_resolve_mode(depth_resolve_mode(resolve))
                .stencil_resolve_mode(vk::ResolveModeFlags::NONE)
                .depth_stencil_resolve_attachment(depth_resolve_ref);
        }
        if let (Some(stencil_filter), Some(stencil_resolve_ref)) =
            (stencil_resolve, &stencil_resolve_ref)
        {
            depth_stencil_resolve = depth_stencil_resolve
                // The combined shape resolves both faces into the one landing:
                // the depth slot keeps its own filter, and the stencil slot
                // takes the only admitted stencil filter, Sample0, mapped onto
                // the device's SAMPLE_ZERO mode. The stencil-only shape has no
                // depth face, so its depth slot is `NONE`
                // (`research/docs/23` §3.3, v60).
                .depth_resolve_mode(if let Some(resolve) = depth_resolve {
                    depth_resolve_mode(resolve)
                } else {
                    vk::ResolveModeFlags::NONE
                })
                // The mask check above admitted this filter, and Sample0 is
                // the only stencil filter with a Vulkan mode
                // (`research/docs/23` §3.3, v60/v70): the mode comes from the
                // one mapping the agreement rule reads.
                .stencil_resolve_mode(stencil_resolve_mode(stencil_filter))
                .depth_stencil_resolve_attachment(stencil_resolve_ref);
        }
        let mut subpass = vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        if let Some(resolve_refs) = &resolve_refs {
            subpass = subpass.resolve_attachments(resolve_refs);
        }
        if let Some(depth_ref) = &depth_ref {
            subpass = subpass.depth_stencil_attachment(depth_ref);
        }
        if depth_resolve.is_some() || stencil_resolve.is_some() {
            subpass = subpass.push_next(&mut depth_stencil_resolve);
        }
        let subpasses = [subpass];
        // A seeded attachment's load reads the seed pass's clear
        // (`research/docs/23` §82, v82), so the external→subpass dependency
        // states the write class the seed pass handed on as well as the
        // load's own read. Every unseeded pass keeps the availability-only
        // scope it always had, byte for byte.
        let seeded = self
            .attachments
            .iter()
            .any(|attachment| attachment.seed.is_some());
        let mut dependencies = vec![vk::SubpassDependency2::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(if seeded {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::empty()
            })
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | if seeded {
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                    } else {
                        vk::AccessFlags::empty()
                    },
            )];
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
            vk::SubpassDependency2::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ)
        });
        if depth.is_some() || stencil.is_some() {
            // The depth clear and the test's depth writes are their own access
            // class: the same pair the colour side states, named for the
            // early/late fragment tests (`research/docs/23` §3.3, v36).
            // The stencil surface's writes travel the same pair (`v47`), so a
            // surface either rail keeps hands its writes on to the transfer
            // stage the copy-out runs in.
            let storing_surface = self
                .depth
                .as_ref()
                .is_some_and(|objects| objects.readback.is_some())
                || self
                    .stencil
                    .as_ref()
                    .is_some_and(|objects| objects.readback.is_some());
            dependencies.push(
                vk::SubpassDependency2::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    )
                    .dst_stage_mask(
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            );
            dependencies.push(
                vk::SubpassDependency2::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    )
                    // A storing surface hands its depth writes on to the
                    // transfer stage the copy-out runs in — the same rule for a
                    // storing stencil surface's stencil writes (`v49`); a
                    // discarded surface has nothing to hand on, so the
                    // dependency only has to witness the pass
                    // (`research/docs/23` §3.3, v43/v49).
                    .dst_stage_mask(vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TRANSFER)
                    .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_access_mask(if storing_surface {
                        vk::AccessFlags::TRANSFER_READ
                    } else {
                        vk::AccessFlags::empty()
                    }),
            );
        }
        let info = vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        self.render_pass = unsafe { self.context.device.create_render_pass2(&info, None) }
            .map_err(|error| execution_refusal("create render pass", &error.to_string()))?;
        Ok(())
    }

    fn create_framebuffer(&mut self, width: u32, height: u32) -> Result<(), ProviderError> {
        let mut views = self
            .attachments
            .iter()
            .map(|attachment| attachment.view)
            .collect::<Vec<_>>();
        // The resolve targets follow the colour views in the render pass's own
        // order — one single-sample view per multisampled attachment
        // (`research/docs/23` §3.3, v51). A single-sample pass adds none.
        for attachment in &self.attachments {
            if let Some(resolve) = &attachment.resolve {
                views.push(resolve.view);
            }
        }
        if let Some(depth) = &self.depth {
            // The framebuffer's attachment list is the render pass's, in the
            // same order: the colour views first, the depth view last
            // (`research/docs/23` §3.3, v36).
            views.push(depth.view);
            // The depth resolve target follows its four-sample sibling in the
            // render pass's own order (`research/docs/23` §3.3, v57): a
            // resolving pass names both, and every other depth pass adds
            // nothing here.
            if let Some(resolve) = &depth.resolve {
                views.push(resolve.view);
            }
        }
        if self.depth.is_none() {
            if let Some(stencil) = &self.stencil {
                // The stencil view follows the depth view in the render pass's own
                // order; a stencil-only pass has no depth view to precede it
                // (`research/docs/23` §3.3, v47).
                views.push(stencil.view);
                // The stencil resolve target follows its four-sample sibling in
                // the render pass's own order (`research/docs/23` §3.3, v60): a
                // resolving pass names both, and every other stencil pass adds
                // nothing here.
                if let Some(resolve) = &stencil.resolve {
                    views.push(resolve.view);
                }
            }
        }
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

    /// Write tightly packed texel rows into one host-visible `LINEAR` image.
    ///
    /// The two uploaded arms' half of [`Self::create_render_textures`],
    /// unchanged from the pre-lease increments: the source bytes are tightly
    /// packed `width * 4`-byte rows, while the driver's `row_pitch` is where
    /// each row actually starts. Writing row `r` at `r * width * 4` would land
    /// every row after the first in bytes the driver never reads, the same trap
    /// the compute rail's upload documents. A failure disposes the image and
    /// its memory, because nothing else owns them yet.
    fn upload_render_texture(
        &self,
        image: vk::Image,
        memory: vk::DeviceMemory,
        requirements: &vk::MemoryRequirements,
        width: u32,
        height: u32,
        texels: &[u8],
    ) -> Result<(), ProviderError> {
        let mapped = match unsafe {
            self.context.device.map_memory(
                memory,
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
        } {
            Ok(mapped) => mapped,
            Err(error) => {
                unsafe {
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(execution_refusal(
                    "map render texture memory",
                    &error.to_string(),
                ));
            }
        };
        let tight_row_bytes = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| {
                unsafe {
                    self.context.device.unmap_memory(memory);
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                contract_refusal("render texture row pitch overflows usize")
            })?;
        let (base_offset, row_pitch) = if height == 1 {
            (0, tight_row_bytes)
        } else {
            let layout = unsafe {
                self.context.device.get_image_subresource_layout(
                    image,
                    vk::ImageSubresource {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        array_layer: 0,
                    },
                )
            };
            (
                usize::try_from(layout.offset)
                    .map_err(|_| contract_refusal("render texture row offset overflows"))?,
                usize::try_from(layout.row_pitch)
                    .map_err(|_| contract_refusal("render texture row pitch overflows"))?,
            )
        };
        for (row, chunk) in texels.chunks(tight_row_bytes).enumerate() {
            let destination = base_offset + row * row_pitch;
            if destination + chunk.len() > usize::try_from(requirements.size).unwrap_or(0) {
                unsafe {
                    self.context.device.unmap_memory(memory);
                    self.context.device.destroy_image(image, None);
                    self.context.device.free_memory(memory, None);
                }
                return Err(contract_refusal(
                    "render texture rows reach past the image's own allocation",
                ));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    (mapped.cast::<u8>()).add(destination),
                    chunk.len(),
                );
            }
        }
        unsafe { self.context.device.unmap_memory(memory) };
        Ok(())
    }

    /// Upload one pass's sampled textures and build the descriptor the fragment
    /// stage reads them through (`research/docs/23` §3.3, v70).
    ///
    /// The rail mirrors the compute track's texture handling for the two
    /// uploaded arms: a host-visible `LINEAR` image per binding, written
    /// through the driver's own `VkSubresourceLayout.row_pitch` (a linear
    /// image's rows are only *at least* the tightly packed width apart), and a
    /// provider-synthesised nearest/clamp sampler. The image stays in
    /// `PREINITIALIZED` until [`Self::record`] transitions it, which is where
    /// the host writes become visible to the fragment stage — the same two-step
    /// shape the compute rail's own upload uses.
    ///
    /// The no-copy arm cannot upload host bytes at all: writing the owner's
    /// mapping into the rail's own image here would freeze the pages at the
    /// moment the pass was built, which is exactly the snapshot R5c's
    /// falsification refuses (`research/docs/23` §75). It imports the owner's
    /// window as a `TRANSFER_SRC` buffer instead, and [`Self::record`] issues
    /// the `vkCmdCopyBufferToImage` that reads those pages at execution time.
    fn create_render_textures(
        &mut self,
        textures: &[OffscreenRenderTexture<'_>],
    ) -> Result<(), ProviderError> {
        if textures.is_empty() {
            return Ok(());
        }
        let format = vk::Format::R8G8B8A8_UNORM;
        for texture in textures {
            let [width, height] = texture.extent;
            // The no-copy arm's image is a transfer destination, never a host
            // write: the device copy lands in it, so it lives in device-local
            // `OPTIMAL` memory and starts undefined.
            let borrowing = matches!(texture.source, RenderInputSource::Borrowed { .. });
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
                .tiling(if borrowing {
                    vk::ImageTiling::OPTIMAL
                } else {
                    vk::ImageTiling::LINEAR
                })
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | if borrowing {
                            vk::ImageUsageFlags::TRANSFER_DST
                        } else {
                            vk::ImageUsageFlags::empty()
                        },
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(if borrowing {
                    vk::ImageLayout::UNDEFINED
                } else {
                    vk::ImageLayout::PREINITIALIZED
                });
            let (image, memory, requirements) = crate::allocate_image_backing(
                self.context,
                &info,
                if borrowing {
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                } else {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                },
                "render texture",
            )
            .map_err(|error| execution_refusal("create render texture image", &error.detail))?;
            // The imported buffer's lifetime is the pass's: the copy reads it
            // until the fence signals, so it is destroyed with the image.
            let copy_source = match &texture.source {
                RenderInputSource::Borrowed { window, .. } => {
                    match self.import_host_pointer_buffer(
                        window,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        "render texture",
                    ) {
                        Ok(source) => Some(source),
                        Err(error) => {
                            unsafe {
                                self.context.device.destroy_image(image, None);
                                self.context.device.free_memory(memory, None);
                            }
                            return Err(error);
                        }
                    }
                }
                RenderInputSource::TraceBytes(bytes) => {
                    self.upload_render_texture(image, memory, &requirements, width, height, bytes)?;
                    None
                }
                RenderInputSource::StagedBytes(bytes) => {
                    self.upload_render_texture(image, memory, &requirements, width, height, bytes)?;
                    None
                }
            };
            let view =
                crate::create_color_image_view(self.context, image, format, "render texture")
                    .map_err(|error| {
                        unsafe {
                            if let Some((buffer, buffer_memory)) = copy_source {
                                self.context.device.destroy_buffer(buffer, None);
                                self.context.device.free_memory(buffer_memory, None);
                            }
                            self.context.device.destroy_image(image, None);
                            self.context.device.free_memory(memory, None);
                        }
                        execution_refusal("create render texture view", &error.detail)
                    })?;
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            let sampler = unsafe { self.context.device.create_sampler(&sampler_info, None) }
                .map_err(|error| {
                    unsafe {
                        if let Some((buffer, buffer_memory)) = copy_source {
                            self.context.device.destroy_buffer(buffer, None);
                            self.context.device.free_memory(buffer_memory, None);
                        }
                        self.context.device.destroy_image_view(view, None);
                        self.context.device.destroy_image(image, None);
                        self.context.device.free_memory(memory, None);
                    }
                    execution_refusal("create render texture sampler", &error.to_string())
                })?;
            if copy_source.is_none() {
                self.context.record_buffer_upload();
                self.context
                    .record_buffer_upload_bytes(texture.source.len());
            }
            self.textures.push(SampledTextureObjects {
                image,
                memory,
                view,
                sampler,
                extent: [width, height],
                copy_source,
            });
        }
        // The descriptor set layout is the sampled pipeline's own: one
        // fragment-stage combined image sampler per binding, in binding order
        // (`research/docs/23` §3.3, v70).
        let bindings = self
            .textures
            .iter()
            .enumerate()
            .map(|(index, _)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(u32::try_from(index).unwrap_or(SAMPLED_TEXTURE_BINDING))
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            })
            .collect::<Vec<_>>();
        self.descriptor_set_layout = unsafe {
            self.context.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| execution_refusal("create descriptor set layout", &error.to_string()))?;
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(bindings.len() as u32)];
        self.descriptor_pool = unsafe {
            self.context.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| execution_refusal("create descriptor pool", &error.to_string()))?;
        let layouts = [self.descriptor_set_layout];
        let sets = unsafe {
            self.context.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|error| execution_refusal("allocate descriptor set", &error.to_string()))?;
        self.descriptor_set = sets.into_iter().next().ok_or_else(|| {
            execution_refusal("allocate descriptor set", "driver returned no set")
        })?;
        let image_infos = self
            .textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::default()
                    .sampler(texture.sampler)
                    .image_view(texture.view)
                    // The upload lands the image in `GENERAL` before the draw
                    // (`Self::record`), the same layout the compute rail binds
                    // its sampled textures in.
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();
        let writes = image_infos
            .iter()
            .enumerate()
            .map(|(index, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(u32::try_from(index).unwrap_or(SAMPLED_TEXTURE_BINDING))
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe {
            self.context.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    /// The graphics pipeline of the milestone: two stages, no vertex input, no
    /// dynamic state beyond the explicit viewport/scissor, no blend/cull/depth.
    /// Every absent state is expressed by not enabling it (`research/docs/23`
    /// §3.2), and the sampled pipeline's one extra input is the descriptor set
    /// layout `create_render_textures` installed (`research/docs/23` §3.3,
    /// v70).
    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        &mut self,
        vertex_words: &[u32],
        fragment_words: &[u32],
        vertex_entry: &CStr,
        fragment_entry: &CStr,
        vertex_streams: &[VertexStream<'_>],
        depth: Option<&OffscreenDepthAttachment>,
        stencil: Option<&OffscreenStencilAttachment>,
        cull: Option<RenderPassCull>,
        blend: Option<&RenderPassBlend>,
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
                    // The binding's own step (`research/docs/23` §3.3, v31):
                    // Vulkan core's instance rate advances one record per
                    // instance, which is exactly the contract's fixed rate.
                    .input_rate(match stream.layout.step {
                        VertexStep::PerVertex => vk::VertexInputRate::VERTEX,
                        VertexStep::PerInstance => vk::VertexInputRate::INSTANCE,
                    }),
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
        // The pass's culling state (`research/docs/23` §3.3, v39): both fields
        // are stated in framebuffer coordinates by both APIs, and the v38
        // alignment is what makes the two rails agree about them. A pass with
        // no state culls nothing and keeps the pre-v39 pipeline exactly.
        let (cull_mode, front_face) = match cull {
            Some(state) => (
                match state.mode {
                    CullMode::None => vk::CullModeFlags::NONE,
                    CullMode::Front => vk::CullModeFlags::FRONT,
                    CullMode::Back => vk::CullModeFlags::BACK,
                },
                match state.winding {
                    Winding::Clockwise => vk::FrontFace::CLOCKWISE,
                    Winding::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                },
            ),
            None => (vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE),
        };
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(cull_mode)
            .front_face(front_face)
            .line_width(1.0);
        // The pipeline's raster state follows the attachments' own sample
        // count, which is what keeps the subpass's references and the pipeline
        // from disagreeing (`research/docs/23` §3.3, v51). Per-fragment shading
        // (the default) shades each covered fragment once and replicates the
        // output to its covered samples, which is exactly the resolve
        // semantics the fixture proves.
        let rasterization_samples = self
            .attachments
            .first()
            .map_or(vk::SampleCountFlags::TYPE_1, |attachment| {
                attachment.samples
            });
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(rasterization_samples);
        // One blend state per colour attachment, indexed by location exactly
        // like the subpass's attachment references. A pass that states none
        // keeps the pre-v40 no-blend, all-writes state
        // (`research/docs/23` §3.3, v40).
        let blend_attachments = self
            .attachments
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let state = blend.and_then(|blend| blend.attachments.get(index));
                let mapped = match state {
                    Some(attachment) => vk::PipelineColorBlendAttachmentState::default()
                        .blend_enable(true)
                        .src_color_blend_factor(vk_blend_factor(attachment.source_rgb))
                        .dst_color_blend_factor(vk_blend_factor(attachment.destination_rgb))
                        .color_blend_op(vk_blend_operation(attachment.operation))
                        .src_alpha_blend_factor(vk_blend_factor(attachment.source_alpha))
                        .dst_alpha_blend_factor(vk_blend_factor(attachment.destination_alpha))
                        .alpha_blend_op(vk_blend_operation(attachment.operation)),
                    None => vk::PipelineColorBlendAttachmentState::default().blend_enable(false),
                };
                mapped.color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect::<Vec<_>>();
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        // The sampled pipeline's layout carries the descriptor set layout its
        // fragment stage reads through, in binding order
        // (`research/docs/23` §3.3, v70). A pass that samples nothing keeps the
        // empty layout every earlier increment built.
        let descriptor_set_layouts =
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                vec![self.descriptor_set_layout]
            } else {
                Vec::new()
            };
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
        self.pipeline_layout = unsafe {
            self.context
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|error| execution_refusal("create pipeline layout", &error.to_string()))?;

        // The depth state is built only when the pass carries a depth
        // attachment: Vulkan refuses a depth-stencil state on a subpass with no
        // depth reference, and a pre-v36 pass has none
        // (`research/docs/23` §3.3, v36).
        // The depth-stencil state is built when the pass opens *either*
        // surface (`research/docs/23` §3.3, v36/v47): a stencil-only pass has
        // no depth test to state, and its stencil test still has to be armed —
        // so the two halves are filled from whichever attachment exists, with
        // the other disabled.
        let depth_state = (depth.is_some() || stencil.is_some()).then(|| {
            let stencil_test = stencil.and_then(|stencil| stencil.test);
            let op_state = |test: StencilTest| {
                vk::StencilOpState::default()
                    .fail_op(vk_stencil_op(test.fail_op))
                    .pass_op(vk_stencil_op(test.pass_op))
                    .depth_fail_op(vk_stencil_op(test.depth_fail_op))
                    .compare_op(vk_stencil_compare(test.compare))
                    .compare_mask(u32::from(test.read_mask))
                    .write_mask(u32::from(test.write_mask))
                    .reference(u32::from(test.reference))
            };
            let mut state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(depth.is_some_and(|attachment| attachment.test.is_some()))
                .depth_write_enable(
                    depth.is_some_and(|attachment| attachment.test.is_some_and(|test| test.write)),
                )
                .depth_compare_op(
                    match depth.and_then(|attachment| attachment.test.map(|t| t.compare)) {
                        Some(CompareFunction::Less) => vk::CompareOp::LESS,
                        // `Always` is also what an attachment with no test states:
                        // the pass still clears it, and every fragment passes.
                        Some(CompareFunction::Always) | None => vk::CompareOp::ALWAYS,
                    },
                )
                .depth_bounds_test_enable(false)
                .stencil_test_enable(stencil_test.is_some());
            if let Some(test) = stencil_test {
                state = state.front(op_state(test)).back(op_state(test));
            }
            state
        });
        let mut info = vk::GraphicsPipelineCreateInfo::default()
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
        if let Some(depth_state) = &depth_state {
            info = info.depth_stencil_state(depth_state);
        }
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

    /// Bind every caller-held stream the pass reads (`research/docs/23` §3.3,
    /// §71).
    ///
    /// One buffer per pool view, holding that view's bytes: the provider's
    /// compute path binds a lone owned view at its own offset, and this rail
    /// does the same, so the stream starts at byte zero of the view exactly as
    /// the footprint proof assumed. The pool upload for the same view may have
    /// happened on the compute path, but these buffers are the rail's own and
    /// are destroyed with the pass.
    ///
    /// Trace-owned and staged bytes are uploaded into a host-visible buffer of
    /// the rail's own; a borrowed window is imported at the owner's address
    /// instead, which is the no-copy arm of the lease channel (R3c). Every
    /// source was resolved before this call, so the bytes here come from the
    /// window the footprint proof read and nothing is copied that the proof
    /// did not see.
    fn create_vertex_inputs(
        &mut self,
        streams: &[VertexStream<'_>],
        index: Option<&IndexStream<'_>>,
    ) -> Result<(), ProviderError> {
        for stream in streams {
            let (buffer, memory) = self.bind_render_input(
                &stream.source,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "vertex input",
            )?;
            self.vertex_inputs.push((buffer, memory));
        }
        if let Some(index) = index {
            let (buffer, memory) = self.bind_render_input(
                &index.source,
                vk::BufferUsageFlags::INDEX_BUFFER,
                "index input",
            )?;
            self.input_index_buffer = buffer;
            self.input_index_memory = memory;
            self.input_index_type = indices_format(index.format);
        }
        Ok(())
    }

    /// Bind one resolved render input with `usage`.
    ///
    /// The two uploaded arms land in a host-visible buffer of the rail's own;
    /// the borrowed arm imports the owner's mapping instead, which is the only
    /// shape that reads the owner's pages directly (`research/docs/23`
    /// §71/§74). The three roles that arrive here are a vertex stream, the
    /// index buffer, and a loading attachment's transfer source.
    fn bind_render_input(
        &self,
        source: &RenderInputSource<'_>,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), ProviderError> {
        match source {
            RenderInputSource::TraceBytes(bytes) => self.create_host_visible_buffer(
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                usage,
                bytes,
                name,
            ),
            RenderInputSource::StagedBytes(bytes) => self.create_host_visible_buffer(
                u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                usage,
                bytes,
                name,
            ),
            RenderInputSource::Borrowed { window, .. } => {
                self.import_host_pointer_buffer(window, usage, name)
            }
        }
    }

    /// Import the owner's own mapping for one resolved no-copy window
    /// (`research/docs/23` §71, R3c).
    ///
    /// The shape mirrors the compute rail's import: one buffer of the window's
    /// own length, and one memory allocation taken from
    /// `VK_EXT_external_memory_host` at the owner's address. Nothing is copied;
    /// the buffer reads and writes the owner's pages. A driver that refuses the
    /// import is a typed execution refusal, not a fallback to a copy.
    fn import_host_pointer_buffer(
        &self,
        window: &BorrowedView,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), ProviderError> {
        let Some(host) = self.context.external_memory_host.as_ref() else {
            // Resolution asked the same question before this point; the second
            // line of defence keeps a directly-constructed request fail-closed
            // instead of importing through a device that never advertised the
            // extension.
            return Err(capability_refusal("storage_mode_unsupported")
                .with_field(
                    "storage_mode",
                    FieldValue::Text("borrowed_no_copy".to_owned()),
                )
                .with_detail(
                    "the device does not import host memory, so a no-copy render input cannot be \
                     bound",
                ));
        };
        let info = vk::BufferCreateInfo::default()
            .size(u64::try_from(window.len).unwrap_or(u64::MAX))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.context.device.create_buffer(&info, None) }.map_err(|error| {
                execution_refusal(&format!("create {name} buffer"), &error.to_string())
            })?;
        let requirements = unsafe { self.context.device.get_buffer_memory_requirements(buffer) };
        if requirements.size > u64::try_from(window.capacity).unwrap_or(u64::MAX) {
            unsafe { self.context.device.destroy_buffer(buffer, None) };
            return Err(execution_refusal(
                &format!("import {name} host memory"),
                &format!(
                    "one buffer needs {} imported bytes but the lease reserves {}",
                    requirements.size, window.capacity
                ),
            ));
        }
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        let result = unsafe {
            (host.device.fp().get_memory_host_pointer_properties_ext)(
                host.device.device(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                window.pointer as *const std::ffi::c_void,
                &mut properties,
            )
        };
        if result != vk::Result::SUCCESS {
            unsafe { self.context.device.destroy_buffer(buffer, None) };
            return Err(execution_refusal(
                &format!("query {name} host pointer"),
                &result.to_string(),
            ));
        }
        let memory_type = match self.context.memory_type(
            requirements.memory_type_bits & properties.memory_type_bits,
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
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(window.pointer as *mut std::ffi::c_void);
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut import);
        let memory = match unsafe { self.context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    &format!("import {name} host memory"),
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
                &format!("bind {name} imported memory"),
                &error.to_string(),
            ));
        }
        Ok((buffer, memory))
    }

    /// Provide the transfer source an attachment's previous contents leave for
    /// the `vkCmdCopyBufferToImage` a `LoadOp::Load` pass issues
    /// (`research/docs/23` §3.3/§74).
    ///
    /// The bytes the trace owns and the provider's staged copy are uploaded
    /// into a host-visible staging buffer, exactly as before. An owner's own
    /// mapping is imported at the owner's address instead — the same
    /// [`Self::bind_render_input`] arm a no-copy stream takes — so the device
    /// reads the pages the footprint and the upload both name rather than a
    /// snapshot of them (R5b).
    fn create_previous_bytes(
        &mut self,
        index: usize,
        source: &RenderInputSource<'_>,
    ) -> Result<(), ProviderError> {
        let (buffer, memory) = self.bind_render_input(
            source,
            vk::BufferUsageFlags::TRANSFER_SRC,
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
        let (objects, mapping) = Self::allocate_readback(self.context, byte_length)?;
        self.readbacks.push(objects);
        Ok(mapping)
    }

    /// The depth attachment's own readback destination
    /// (`research/docs/23` §3.3, v43).
    ///
    /// The stored depth surface lands through the same staging shape a colour
    /// attachment uses, but it is not part of `readbacks`: that list is zipped
    /// with the colour attachments in location order, and the depth copy is
    /// recorded separately after them. Returns the mapping the host reads once
    /// the fence signals.
    fn create_depth_readback(&mut self, byte_length: u64) -> Result<usize, ProviderError> {
        let (objects, mapping) = Self::allocate_readback(self.context, byte_length)?;
        let depth = self.depth.as_mut().ok_or_else(|| {
            contract_refusal("a depth readback needs the depth attachment it copies out of")
        })?;
        depth.readback = Some(objects);
        depth.mapping = Some(mapping);
        Ok(mapping)
    }

    /// The stored depth surface's texels, copied out of the readback mapping
    /// once the fence has signalled (`research/docs/23` §3.3, v43).
    ///
    /// `None` for a pass whose depth attachment has no readback — either no
    /// depth attachment at all or one the trace discards, which is the shape
    /// every pre-v43 frame states.
    fn depth_readback_bytes(
        &self,
        byte_length: usize,
        context: &VulkanContext,
    ) -> Result<Option<Vec<u8>>, ProviderError> {
        let Some(mapping) = self.depth.as_ref().and_then(|depth| depth.mapping) else {
            return Ok(None);
        };
        let texels =
            unsafe { std::slice::from_raw_parts(mapping as *const u8, byte_length).to_vec() };
        context.record_buffer_readback();
        context.record_buffer_readback_bytes(texels.len());
        Ok(Some(texels))
    }

    /// The stencil attachment's own readback destination
    /// (`research/docs/23` §3.3, v49).
    ///
    /// The depth sibling's shape one byte wide: the stored stencil surface
    /// lands through the same staging shape a colour attachment uses, kept
    /// beside the stencil image rather than in `readbacks`, because that list is
    /// zipped with the colour attachments in location order. Returns the
    /// mapping the host reads once the fence signals.
    fn create_stencil_readback(&mut self, byte_length: u64) -> Result<usize, ProviderError> {
        let (objects, mapping) = Self::allocate_readback(self.context, byte_length)?;
        let stencil = self.stencil.as_mut().ok_or_else(|| {
            contract_refusal("a stencil readback needs the stencil attachment it copies out of")
        })?;
        stencil.readback = Some(objects);
        stencil.mapping = Some(mapping);
        Ok(mapping)
    }

    /// The stored stencil surface's texels, copied out of the readback mapping
    /// once the fence has signalled (`research/docs/23` §3.3, v49).
    ///
    /// `None` for a pass whose stencil attachment has no readback — either no
    /// stencil attachment at all or one the trace discards, which is the shape
    /// every pre-v49 frame states.
    fn stencil_readback_bytes(
        &self,
        byte_length: usize,
        context: &VulkanContext,
    ) -> Result<Option<Vec<u8>>, ProviderError> {
        let Some(mapping) = self.stencil.as_ref().and_then(|stencil| stencil.mapping) else {
            return Ok(None);
        };
        let texels =
            unsafe { std::slice::from_raw_parts(mapping as *const u8, byte_length).to_vec() };
        context.record_buffer_readback();
        context.record_buffer_readback_bytes(texels.len());
        Ok(Some(texels))
    }

    /// Create one `TRANSFER_DST` host-visible buffer and map it, without
    /// attaching it to any list: the colour path pushes it into `readbacks`,
    /// the depth and stencil paths keep it beside their own image.
    fn allocate_readback(
        context: &VulkanContext,
        byte_length: u64,
    ) -> Result<(ReadbackObjects, usize), ProviderError> {
        let info = vk::BufferCreateInfo::default()
            .size(byte_length)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { context.device.create_buffer(&info, None) }
            .map_err(|error| execution_refusal("create readback buffer", &error.to_string()))?;
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = match context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe { context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    "find readback memory type",
                    &error.to_string(),
                ));
            }
        };
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { context.device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { context.device.destroy_buffer(buffer, None) };
                return Err(execution_refusal(
                    "allocate readback memory",
                    &error.to_string(),
                ));
            }
        };
        if let Err(error) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                context.device.destroy_buffer(buffer, None);
                context.device.free_memory(memory, None);
            }
            return Err(execution_refusal(
                "bind readback memory",
                &error.to_string(),
            ));
        }
        let mapping = match unsafe {
            context
                .device
                .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        } {
            Ok(mapping) => mapping as usize,
            Err(error) => {
                unsafe {
                    context.device.destroy_buffer(buffer, None);
                    context.device.free_memory(memory, None);
                }
                return Err(execution_refusal("map readback memory", &error.to_string()));
            }
        };
        Ok((ReadbackObjects { buffer, memory }, mapping))
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
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        attachments: &[OffscreenColorAttachment<'_>],
        depth: Option<&OffscreenDepthAttachment>,
        stencil: Option<&OffscreenStencilAttachment>,
        depth_resolve: Option<DepthResolveFilter>,
        stencil_resolve: Option<StencilResolveFilter>,
        scissor: Option<[u32; 4]>,
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

        // `pClearValues` is indexed by *attachment*, not by colour location, so
        // the list has to mirror the render pass's own attachment order: the
        // colour entries first, then one entry per resolve target, then the
        // depth-stencil surface. A resolve target ignores its entry (its load
        // op is `DONT_CARE`), but the array still has to reach the depth index
        // — a shorter list silently shifts the depth clear, and Lavapipe then
        // clears the depth surface to zero instead of the trace's own value
        // (`research/docs/23` §3.3, v51/v53).
        let mut clear_values = attachments
            .iter()
            .map(|attachment| vk::ClearValue {
                color: clear_value_for(
                    attachment.format,
                    match attachment.load {
                        LoadOp::Clear(clear) => clear,
                        // A loading or `DontCare` attachment carries no clear
                        // colour: Vulkan ignores this entry when the load op
                        // is not `CLEAR`. A resident load is `Load`'s sibling
                        // here: the image's own bytes are kept, so the clear
                        // value is never read either
                        // (`research/docs/23` §76, R7).
                        LoadOp::Load | LoadOp::Resident | LoadOp::DontCare => {
                            ClearColor::new([0; 4])
                        }
                    },
                ),
            })
            .collect::<Vec<_>>();
        for (objects, attachment) in self.attachments.iter().zip(attachments) {
            if objects.resolve.is_some() {
                // The placeholder the resolve attachment's own index needs:
                // its load op is `DONT_CARE`, so the value is never read.
                clear_values.push(vk::ClearValue {
                    color: clear_value_for(attachment.format, ClearColor::new([0; 4])),
                });
            }
        }
        if let Some(depth) = depth {
            clear_values.push(vk::ClearValue {
                // The depth entry follows the colour entries, exactly as the
                // render pass's attachment list does
                // (`research/docs/23` §3.3, v36). Vulkan ignores it when the
                // depth load op is not `CLEAR`.
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth.clear.unwrap_or(1.0),
                    // The combined attachment clears both aspects with one
                    // value: the stencil face's own clear travels here when
                    // the pass opens both faces (`research/docs/23` §3.3,
                    // v60).
                    stencil: stencil
                        .and_then(|surface| surface.clear)
                        .map_or(0, u32::from),
                },
            });
            if depth_resolve.is_some() {
                // The placeholder the depth resolve attachment's own index
                // needs: its load op is `DONT_CARE`, so the value is never
                // read — the same rule the colour resolve placeholders state —
                // but the array still has to reach the attachment count, or
                // the clear list would be one entry short of the render
                // pass's own (`research/docs/23` §3.3, v57).
                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            }
        }
        if let Some(stencil) = stencil.filter(|_| depth.is_none()) {
            clear_values.push(vk::ClearValue {
                // The stencil entry follows the colour entries when the pass
                // opens no depth surface — the two share one reference slot,
                // so a pass never carries both entries (`research/docs/23`
                // §3.3, v47). Vulkan ignores it when the stencil load op is
                // not `CLEAR`.
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: u32::from(stencil.clear.unwrap_or(0)),
                },
            });
            if stencil_resolve.is_some() {
                // The placeholder the stencil resolve attachment's own index
                // needs: its load op is `DONT_CARE`, so the value is never
                // read — the same rule the depth resolve placeholder states —
                // but the array still has to reach the attachment count
                // (`research/docs/23` §3.3, v60).
                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            }
        }
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        // The seed pass a multisampled `Load` is executed with
        // (`research/docs/23` §82, v82) runs before the measured pass: the one
        // clear per seeded attachment writes every sample of the render area,
        // and the measured pass below opens the same images with `LOAD`. No
        // draw is recorded — the load operation *is* the seed pass's work — so
        // the encoder only has to begin and end the subpass.
        if self.seed_render_pass != vk::RenderPass::null() {
            let seed_values = self
                .attachments
                .iter()
                .zip(attachments)
                .filter_map(|(objects, attachment)| {
                    objects.seed.map(|seed| vk::ClearValue {
                        color: clear_value_for(attachment.format, seed),
                    })
                })
                .collect::<Vec<_>>();
            let seed_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.seed_render_pass)
                .framebuffer(self.seed_framebuffer)
                .render_area(render_area)
                .clear_values(&seed_values);
            unsafe {
                self.context.device.cmd_begin_render_pass(
                    self.command,
                    &seed_begin,
                    vk::SubpassContents::INLINE,
                );
                self.context.device.cmd_end_render_pass(self.command);
            }
        }
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
        // The scissor is dynamic pipeline state, so the pass's own rectangle (or
        // the whole render area for a pass that declares none) is recorded here
        // (`research/docs/23` §3.3, v29).
        let [scissor_x, scissor_y, scissor_width, scissor_height] =
            scissor.unwrap_or([0, 0, width, height]);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: i32::try_from(scissor_x)
                    .map_err(|_| contract_refusal("render scissor origin reaches beyond i32"))?,
                y: i32::try_from(scissor_y)
                    .map_err(|_| contract_refusal("render scissor origin reaches beyond i32"))?,
            },
            extent: vk::Extent2D {
                width: scissor_width,
                height: scissor_height,
            },
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
        // Host-visible linear textures are uploaded in `PREINITIALIZED` and
        // the sampled descriptor binds them in `GENERAL`, so the first
        // transition needs only the new layout, not an access scope — the same
        // two-step shape the compute rail's own texture upload records
        // (`research/docs/23` §3.3, v70).
        //
        // A no-copy texture is entered by the device copy instead
        // (`research/docs/23` §75, R5c): the image starts `UNDEFINED`, the
        // first barrier hands the transfer stage a `TRANSFER_DST_OPTIMAL`
        // destination, `vkCmdCopyBufferToImage` reads the owner's own pages the
        // pass imported, and the second barrier publishes those texels to the
        // fragment stage that samples them.
        for texture in &self.textures {
            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            if let Some((buffer, _)) = texture.copy_source {
                let [width, height] = texture.extent;
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
                            .image(texture.image)
                            .subresource_range(subresource)],
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
                        buffer,
                        texture.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&copy),
                    );
                    self.context.device.cmd_pipeline_barrier(
                        self.command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::GENERAL)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(texture.image)
                            .subresource_range(subresource)
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                    );
                }
                continue;
            }
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::PREINITIALIZED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource);
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
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
            // The sampled textures are bound before the draw, in the one
            // descriptor set the pipeline layout carries
            // (`research/docs/23` §3.3, v70).
            if self.descriptor_set != vk::DescriptorSet::null() {
                self.context.device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptor_set),
                    &[],
                );
            }
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
                        self.context.device.cmd_draw_indexed(
                            self.command,
                            index_count,
                            self.instance_count,
                            0,
                            i32::try_from(self.base_vertex).unwrap_or(i32::MAX),
                            0,
                        );
                    }
                    DrawShape::Vertices { vertex_count } => {
                        self.context.device.cmd_draw(
                            self.command,
                            vertex_count,
                            self.instance_count,
                            0,
                            0,
                        );
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
                // The `vertex_id` shape instances the same way the
                // vertex-buffer arms do (`research/docs/23` §3.3, v31): the
                // reviewed triangle is replayed once per instance, so a pass
                // that asks for more than one keeps its own count here too.
                self.context.device.cmd_draw(
                    self.command,
                    FULL_SCREEN_TRIANGLE_VERTICES,
                    self.instance_count,
                    0,
                    0,
                );
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
            // (`docs/24` §3.6), not a real `VkQueuePresentKHR`. A multisampled
            // pass presents its resolve target — the provider-owned present
            // image — so the transition runs on that image rather than the
            // n-sample surface the subpass consumed (`docs/24` §3.5, v62).
            let present_image = self.attachments[0]
                .resolve
                .as_ref()
                .map_or(self.attachments[0].image, |resolve| resolve.image);
            let barrier = present_transition_barrier(present_image);
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
            // The pair is the trace's own store decision, not the Vulkan
            // action: a resident store also renders with `STORE` (the image
            // keeps its bytes) but lands no readback buffer, so filtering by
            // `store_op` would shift every later attachment's copy by one
            // (`research/docs/23` §76, R7).
            .filter(|attachment| attachment.publishes)
            .zip(&self.readbacks)
        {
            // A multisampled location's bytes are the resolve target's, not the
            // four-sample image's, which the subpass consumed
            // (`research/docs/23` §3.3, v51). The render pass already left the
            // resolve image in `TRANSFER_SRC_OPTIMAL`, so the copy needs no
            // barrier of its own — the same rule the stored single-sample
            // attachment states (`docs/23` §3.6, v19).
            let source = attachment
                .resolve
                .as_ref()
                .map_or(attachment.image, |resolve| resolve.image);
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
                    source,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    std::slice::from_ref(&copy),
                );
            }
        }
        // The stored depth attachment's own copy, after the colour ones and
        // with the depth aspect (`research/docs/23` §3.3, v43). The render
        // pass already left the image in `TRANSFER_SRC_OPTIMAL`, so the copy
        // needs no barrier of its own; a discarded surface has no readback
        // buffer and is not copied at all.
        if let Some(depth) = &self.depth {
            if let Some(readback) = &depth.readback {
                // A resolving pass's bytes are the resolve target's, not the
                // four-sample image's, which the subpass consumed
                // (`research/docs/23` §3.3, v57). The render pass already left
                // the resolve image in `TRANSFER_SRC_OPTIMAL`, so the copy
                // needs no barrier of its own — the same rule the stored
                // single-sample attachment states (v43).
                let source = depth
                    .resolve
                    .as_ref()
                    .map_or(depth.image, |resolve| resolve.image);
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
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
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        readback.buffer,
                        std::slice::from_ref(&copy),
                    );
                }
            }
        }
        // The stored stencil attachment's own copy, with the *stencil* aspect
        // and one byte per texel (`research/docs/23` §3.3, v49). The render pass
        // already left the image in `TRANSFER_SRC_OPTIMAL`, so the copy needs no
        // barrier of its own; a discarded surface has no readback buffer and is
        // not copied at all. The depth and stencil surfaces are mutually
        // exclusive in this increment, so the two copies never share a pass.
        if let Some(stencil) = &self.stencil {
            if let Some(readback) = &stencil.readback {
                // A resolving pass's bytes are the resolve target's, not the
                // four-sample image's, which the subpass consumed
                // (`research/docs/23` §3.3, v60). The render pass already left
                // the resolve image in `TRANSFER_SRC_OPTIMAL`, so the copy
                // needs no barrier of its own — the same rule the stored
                // single-sample attachment states (v49).
                let source = stencil
                    .resolve
                    .as_ref()
                    .map_or(stencil.image, |resolve| resolve.image);
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::STENCIL,
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
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        readback.buffer,
                        std::slice::from_ref(&copy),
                    );
                }
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
        self.submitted = true;
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
            if self.seed_framebuffer != vk::Framebuffer::null() {
                self.context
                    .device
                    .destroy_framebuffer(self.seed_framebuffer, None);
            }
            if self.seed_render_pass != vk::RenderPass::null() {
                self.context
                    .device
                    .destroy_render_pass(self.seed_render_pass, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.context
                    .device
                    .destroy_render_pass(self.render_pass, None);
            }
            // The sampled textures and the descriptor the fragment stage read
            // them through are the pass's own (`research/docs/23` §3.3, v70):
            // the pool owns the set, so destroying the pool releases both and
            // the layout goes with it.
            for texture in &self.textures {
                if texture.sampler != vk::Sampler::null() {
                    self.context.device.destroy_sampler(texture.sampler, None);
                }
                if texture.view != vk::ImageView::null() {
                    self.context.device.destroy_image_view(texture.view, None);
                }
                if texture.image != vk::Image::null() {
                    self.context.device.destroy_image(texture.image, None);
                }
                if texture.memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(texture.memory, None);
                }
                // The no-copy arm's imported window is the pass's own too
                // (`research/docs/23` §75, R5c): the object is destroyed once
                // the pass is terminal, exactly like the image it fed.
                if let Some((buffer, memory)) = texture.copy_source {
                    if buffer != vk::Buffer::null() {
                        self.context.device.destroy_buffer(buffer, None);
                    }
                    if memory != vk::DeviceMemory::null() {
                        self.context.device.free_memory(memory, None);
                    }
                }
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.context
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.context
                    .device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if let Some(depth) = &self.depth {
                // The depth resolve target is owned by the same pass scope
                // (`research/docs/23` §3.3, v57), so it is destroyed beside
                // the four-sample image it was created with.
                if let Some(resolve) = &depth.resolve {
                    if resolve.view != vk::ImageView::null() {
                        self.context.device.destroy_image_view(resolve.view, None);
                    }
                    if resolve.image != vk::Image::null() {
                        self.context.device.destroy_image(resolve.image, None);
                    }
                    if resolve.memory != vk::DeviceMemory::null() {
                        self.context.device.free_memory(resolve.memory, None);
                    }
                }
                if depth.view != vk::ImageView::null() {
                    self.context.device.destroy_image_view(depth.view, None);
                }
                if depth.image != vk::Image::null() {
                    self.context.device.destroy_image(depth.image, None);
                }
                if depth.memory != vk::DeviceMemory::null() {
                    self.context.device.free_memory(depth.memory, None);
                }
            }
            if let Some(stencil) = &self.stencil {
                // The combined shape shares one backing image and one resolve
                // landing with the depth half, which owns both; the shared
                // halves therefore only destroy their own aspect views and
                // leave the backings to the depth half
                // (`research/docs/23` §3.3, v60).
                if stencil.view != vk::ImageView::null() {
                    self.context.device.destroy_image_view(stencil.view, None);
                }
                if let Some(resolve) = &stencil.resolve {
                    if resolve.view != vk::ImageView::null() {
                        self.context.device.destroy_image_view(resolve.view, None);
                    }
                    if resolve.image != vk::Image::null() && !resolve.shares_backing {
                        self.context.device.destroy_image(resolve.image, None);
                    }
                    if resolve.memory != vk::DeviceMemory::null() && !resolve.shares_backing {
                        self.context.device.free_memory(resolve.memory, None);
                    }
                }
                if stencil.image != vk::Image::null() && !stencil.shares_backing {
                    self.context.device.destroy_image(stencil.image, None);
                }
                if stencil.memory != vk::DeviceMemory::null() && !stencil.shares_backing {
                    self.context.device.free_memory(stencil.memory, None);
                }
            }
            for attachment in &self.attachments {
                // The resolve target of a rail-owned multisampled attachment
                // is owned by the same pass scope (`research/docs/23` §3.3,
                // v51), so it is destroyed beside the multisampled image it
                // was created with. A present pass's resolve target is the
                // provider-owned present image, which the scope borrows and
                // must not destroy (`docs/24` §5.2, v62).
                if attachment.owns_resolve {
                    if let Some(resolve) = &attachment.resolve {
                        if resolve.view != vk::ImageView::null() {
                            self.context.device.destroy_image_view(resolve.view, None);
                        }
                        if resolve.image != vk::Image::null() {
                            self.context.device.destroy_image(resolve.image, None);
                        }
                        if resolve.memory != vk::DeviceMemory::null() {
                            self.context.device.free_memory(resolve.memory, None);
                        }
                    }
                }
                if attachment.owns_image {
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

/// A structurally wrong value a caller handed the rail, under a slug that names
/// the field instead of the generic contract refusal.
///
/// The class and phase are the ones core admission uses for the same fact
/// (`Args`, `Resolve`), which is also the pair the native rail refuses a
/// loading attachment's mismatched previous bytes with
/// (`render_attachment_initial_mismatch`), so the two rails report one name.
fn args_refusal(slug: &'static str) -> ProviderError {
    let mut error = ProviderError::new(ProviderPhase::Resolve, ProviderErrorClass::Args, slug)
        .expect("static provider refusal slug");
    error.retryability = Retryability::Never;
    error
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
        AcquirePolicy, AllocationId, AllocationRecord, BorrowedLease, BufferAccess, BufferLease,
        DepthFormat, DepthLoadOp, InitialState, LeaseReservation, PipelineId, PresentDescriptor,
        PresentMode, PresentTarget, RenderAttachment, RenderDepthAttachment, RenderDepthIdentity,
        RenderStencilAttachment, RenderStencilIdentity, StagedLease, StencilFormat, StencilLoadOp,
        TextureAccess, TextureFormat, TextureSource, TextureType, TextureView, VertexLayout,
        ViewId,
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

    /// One texel of an `R16G16B16A16_SFLOAT` attachment (`research/docs/23`
    /// §78): the same `(64/255, 128/255, 192/255, 1)` the `vec4` stage stores,
    /// rounded by the driver to four half floats. None of the three colour
    /// constants is on the half grid, so the rounding is the evidence: the
    /// halves are `0x3404`, `0x3804`, `0x3a06` and the exact `0x3c00`, in
    /// little-endian memory order.
    const EXPECTED_RGBA16F_TEXEL: [u8; 8] = [0x04, 0x34, 0x04, 0x38, 0x06, 0x3a, 0x00, 0x3c];

    /// The clear a 16-bit float attachment test uses: four halves, each the
    /// value `0.99609375` (`0x3bf8`, the half nearest `254/255`), so no half of
    /// the clear coincides with any half the draw stores and a clear read with
    /// the wrong width cannot land these bytes.
    const RGBA16F_CLEAR: [u8; 8] = [0xf8, 0x3b, 0xf8, 0x3b, 0xf8, 0x3b, 0xf8, 0x3b];

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
            entry: "single_pixel".into(),
            spirv: SINGLE_PIXEL_VERT_SPV,
        }
    }

    /// The reviewed vertex stage of the milestone, under the entry name its
    /// `.spvasm` source declares.
    fn milestone_vertex() -> OffscreenVertexStage<'static> {
        OffscreenVertexStage {
            entry: "vertex_main".into(),
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
            vertex_translation: None,
            fragment_translation: None,
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
            vertex_translation: None,
            fragment_translation: None,
        }
    }

    /// The reviewed sampling pair (`research/docs/23` §3.3, v70): the
    /// full-screen geometry with its uv varying and the fragment stage that
    /// samples the pass's own texture binding.
    fn reviewed_sampled_stages() -> RenderStages {
        RenderStages {
            contract: RenderPipelineContract {
                vertex_entry: SAMPLED_QUAD_VERTEX_ENTRY.to_owned(),
                fragment_entry: SOLID_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: SAMPLED_QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SAMPLED_UNORM8_FRAG_SPV.to_vec(),
            vertex_translation: None,
            fragment_translation: None,
        }
    }

    /// The 4×4 texture the sampling fixtures bind, with one distinct texel per
    /// position (`research/docs/23` §3.3, v70).
    fn sampled_texture_view(width: u64, height: u64) -> TextureView {
        let bytes = (0..height as u8)
            .flat_map(|y| (0..width as u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect::<Vec<_>>();
        TextureView {
            view_id: ViewId::new(83),
            metal_binding: 0,
            allocation_id: AllocationId::new(53),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width,
            height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(bytes),
        }
    }

    /// One `size`×`size` render pass for the sampling fixtures.
    fn sampled_pass(size: u64) -> RenderPassDescriptor {
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.color_attachments[0].width = size;
        pass.color_attachments[0].height = size;
        pass.viewport = [0, 0, size as u32, size as u32];
        pass.textures = vec![sampled_texture_view(size, size)];
        pass
    }

    /// The rail's own window for the sampling shape (`research/docs/23` §3.3,
    /// v70): the reviewed pair is admitted only for one `rgba8_unorm` texture
    /// of the render area's own extent, and a pass that names the pair without
    /// binding that texture is refused by name rather than sampled through an
    /// unbound descriptor. Host-side: `prepare_render_request` reads no device.
    #[test]
    fn prepare_render_request_admits_the_reviewed_sampling_shape_only() {
        let stages = reviewed_sampled_stages();
        stages
            .validate_stage_pair()
            .expect("the reviewed sampling pair is executable");
        let pass = sampled_pass(4);
        let previous = vec![None];
        let request = prepare_render_request(
            &stages,
            &pass,
            &previous,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the reviewed sampling shape is admitted");
        assert_eq!(request.textures.len(), 1);
        assert_eq!(request.textures[0].extent, [4, 4]);
        assert_eq!(request.textures[0].source.len(), 64);

        // Another extent puts some fragment's sample on a texel boundary or
        // inside a neighbour, which is a filtered read the review never
        // covered.
        let mut other_extent = sampled_pass(4);
        other_extent.textures = vec![sampled_texture_view(2, 2)];
        let refused = match prepare_render_request(
            &stages,
            &other_extent,
            &previous,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a texture of another extent"),
        };
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_texture_extent_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);

        // The reviewed stage reads one `rgba8_unorm` surface.
        let mut other_format = sampled_pass(4);
        let mut view = sampled_texture_view(4, 4);
        view.format = TextureFormat::Bgra8Unorm;
        other_format.textures = vec![view];
        let refused = match prepare_render_request(
            &stages,
            &other_format,
            &previous,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a texture of another format"),
        };
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_texture_format_unsupported");

        // A registration that pairs one reviewed half with the other pair's
        // half is refused at the gate, so it cannot reach the device.
        let mismatched = RenderStages {
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            ..reviewed_sampled_stages()
        };
        let refused = mismatched
            .validate_stage_pair()
            .expect_err("the sampling vertex stage needs its own fragment stage");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");

        // The command buffer is where the pair and the binding meet, so the
        // rail re-asks that question there too: a pass that names the pair
        // without a texture is refused instead of sampling an unbound
        // descriptor.
        let mut unbound = sampled_pass(4);
        unbound.textures = Vec::new();
        let request = prepare_render_request(
            &stages,
            &unbound,
            &previous,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the shape is admitted; the binding question is the execution's");
        assert!(request.textures.is_empty());
    }

    /// Append `OpCapability FloatControls2` + `OpExtension "SPV_KHR_float_controls2"`
    /// to one module: the pair the translator emits for a float result that
    /// withholds a fast-math permission (R8). Inserted after the five-word
    /// header, which is where the capability stream starts.
    fn with_float_controls2(module: &[u8]) -> Vec<u8> {
        let mut words = module
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let mut extension = Vec::new();
        for chunk in "SPV_KHR_float_controls2\0".as_bytes().chunks(4) {
            let mut word = [0_u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            extension.push(u32::from_le_bytes(word));
        }
        let mut injected = vec![
            (2_u32 << 16) | spirv::Op::Capability as u32,
            spirv::Capability::FloatControls2 as u32,
            ((1 + extension.len()) as u32) << 16 | spirv::Op::Extension as u32,
        ];
        injected.extend_from_slice(&extension);
        words.splice(5..5, injected);
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    /// R8: the rail re-asks the device's capability subset where a module
    /// becomes a pipeline.
    ///
    /// A module that demands `FloatControls2` is refused by the capability gate
    /// when the policy does not admit it. The second half of the test asks the
    /// same module under an admitting policy and then asks the module
    /// accounting: the two refusals have different names, so the first one
    /// demonstrably came from the capability gate and not from the module's own
    /// identity.
    #[test]
    fn registration_refuses_a_module_the_device_did_not_answer_for() {
        let mut stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        stages.vertex_spirv = with_float_controls2(&stages.vertex_spirv);
        let refused = validate_module_capabilities(&stages, SpirvFeaturePolicy::PHASE1)
            .expect_err("a module that demands FloatControls2 is refused without the feature");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_capability_unavailable");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("stage"),
            Some(&FieldValue::Text("vertex".to_owned()))
        );
        assert!(
            refused
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("capability 6029")),
            "the gate's own sentence rides along: {:?}",
            refused.detail
        );

        // The device's own policy admits the same module, so the refusal above
        // was the capability gate's and not the module accounting's.
        assert!(validate_module_capabilities(
            &stages,
            SpirvFeaturePolicy::PHASE1.with_float_controls2(true)
        )
        .is_ok());
        let refused = stages
            .validate()
            .expect_err("the modified module is no longer the reviewed one");
        assert_eq!(refused.slug, "render_stage_translation_unavailable");
    }

    /// One 2×2 render pass naming an attachment of `format`, holding the clear
    /// sentinel the coverage assertions look for.
    fn milestone_pass(format: AttachmentFormat) -> RenderPassDescriptor {
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
            scissor: None,
            vertices: 3,
            vertex_buffers: Vec::new(),
            indices: None,
            instance_count: 1,
            textures: Vec::new(),
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
            ProviderTargetImage::create(
                std::sync::Arc::clone(&context),
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
            )
            .expect("the present target is created"),
        );

        let first = target.begin_target_pass();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let waiting = std::sync::Arc::clone(&target);
        let handle = std::thread::spawn(move || {
            let _second = waiting.begin_target_pass();
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
        // The clear is one texel of the format under test (`research/docs/23`
        // §78): the four-byte sentinel for the four-byte class, four sentinel
        // halves for the eight-byte one. A four-byte payload on the wide format
        // is refused by name, which is the admission rule this helper's caller
        // relies on.
        let clear = match format {
            AttachmentFormat::Rgba16Float => {
                ClearColor::from_bytes(&RGBA16F_CLEAR).expect("one eight-byte texel")
            }
            _ => ClearColor::new([CLEAR_SENTINEL; 4]),
        };
        let mut blobs = execute_offscreen_render(
            context,
            &OffscreenRenderRequest {
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![OffscreenColorAttachment {
                    format,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(clear),
                    previous: None,
                    seed: None,
                    resident: None,
                }],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .unwrap_or_else(|error| panic!("the 2x2 {format:?} render pass executes: {error:?}"));
        let texels = blobs
            .attachments
            .remove(0)
            .expect("a stored attachment reads back");
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

        // Four halves, one per component, in the format's memory order: the
        // clear's eight bytes are four little-endian half values and the driver
        // is handed those components (`research/docs/23` §78). A rail that read
        // the payload as four bytes would put two payload bytes per component
        // and land four other values.
        let wide = ClearColor::from_bytes(&RGBA16F_CLEAR).expect("one eight-byte texel");
        let rgba16f = unsafe { clear_value_for(AttachmentFormat::Rgba16Float, wide).float32 };
        assert_eq!(rgba16f, [0.996_093_75; 4]);
        assert_ne!(
            rgba16f[0],
            254.0_f32 / 255.0,
            "the half's value is not the f32 of 254/255, so the decode is visible"
        );
        // The remaining byte pairs are free: the same clear with only the last
        // two bytes changed moves the alpha component alone.
        let mut other = RGBA16F_CLEAR;
        other[6..].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let other = ClearColor::from_bytes(&other).expect("one eight-byte texel");
        let moved = unsafe { clear_value_for(AttachmentFormat::Rgba16Float, other).float32 };
        assert_eq!(moved[..3], rgba16f[..3]);
        assert_eq!(moved[3], 1.0);
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
        // The eight-byte format is a four-component colour attachment, so the
        // same `vec4` module serves it: the store's channel count is the
        // module's fact and the storage width is the `VkFormat`'s
        // (`research/docs/23` §78).
        assert_eq!(
            solid_fragment_spirv(&[AttachmentFormat::Rgba16Float]).expect("admitted"),
            SOLID_UNORM8_FRAG_SPV
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
        assert_eq!(
            attachment_vk_format(AttachmentFormat::Rgba16Float).map(vk::Format::as_raw),
            Ok(vk::Format::R16G16B16A16_SFLOAT.as_raw())
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
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            stencil: None,
            scissor: None,
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::R32Uint,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
                seed: None,
                resident: None,
            }],
            extent: [2, 2],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            base_vertex: 0,
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
                    textures: Vec::new(),
                    blend: None,
                    multisample: None,
                    depth_resolve: None,
                    stencil_resolve: None,
                    cull: None,
                    depth: None,
                    base_vertex: 0,
                    stencil: None,
                    scissor: None,
                    attachments: vec![OffscreenColorAttachment {
                        format,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(clear),
                        previous: None,
                        seed: None,
                        resident: None,
                    }],
                    extent: [2, 2],
                    vertex: single_pixel_vertex(),
                    translated_fragment: None,
                    vertex_streams: Vec::new(),
                    draw: DrawShape::Milestone,
                    instance_count: 1,
                    index_stream: None,
                    indirect: None,
                },
            )
            .unwrap_or_else(|error| panic!("the partial {format:?} pass executes: {error:?}"));
            let texels = blobs
                .attachments
                .remove(0)
                .expect("a stored attachment reads back");
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

    /// The wide-texel class's device question (`research/docs/23` §78): the
    /// rail asks `vkGetPhysicalDeviceImageFormatProperties` for the exact
    /// format/usage/sample-count combination before the first image exists, and
    /// this records the answer the running device gives for the census's
    /// `0x73` shape. The readback tests below execute the same shape, so a
    /// device that answered `false` here would refuse it by name instead.
    #[test]
    fn the_wide_texel_format_answers_the_image_format_probe() {
        let Some(context) = device_context() else {
            return;
        };
        for format in [vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8G8B8A8_UNORM] {
            let answered = format_supports_color_attachment_samples(
                &context,
                format,
                vk::ImageTiling::OPTIMAL,
                vk::SampleCountFlags::TYPE_1,
            );
            eprintln!(
                "vk_format raw={} COLOR_ATTACHMENT at 1x: {answered}",
                format.as_raw()
            );
            assert!(
                answered,
                "the device must admit raw={} as a single-sample colour attachment for the \
                 rail's readback fixtures to mean anything",
                format.as_raw()
            );
        }
    }

    /// The census's `0x73` shape, end to end on the offscreen rail
    /// (`research/docs/23` §78): a 2×2 `R16G16B16A16_SFLOAT` attachment,
    /// cleared, covered by the reviewed `vec4` stage, and read back at eight
    /// bytes per texel.
    ///
    /// This is the case a four-byte-class rail cannot pass: the byte extent is
    /// twice what the pre-v78 formats land, and the bytes are the driver's half
    /// rounding of the stage's own colour constants rather than their `f32`
    /// bits (`0x3e808081`…) or their 8-bit quantisation (`40 80 c0 ff`).
    #[test]
    fn offscreen_rgba16float_attachment_lands_half_rounded_texels() {
        let Some(context) = device_context() else {
            return;
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let texels = offscreen_readback(&context, AttachmentFormat::Rgba16Float);
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(texels.len(), 32, "2x2 texels of eight bytes are 32 bytes");
        assert_eq!(texels, EXPECTED_RGBA16F_TEXEL.repeat(4));
        // The first half is the rounding, not the stored float: the stage's
        // `64/255` has f32 bits `0x3e808081` and the attachment holds `0x3404`.
        assert_eq!(
            texels[..2],
            EXPECTED_RGBA16F_TEXEL[..2],
            "the first component is the rounded half"
        );
        assert_eq!((64.0_f32 / 255.0).to_le_bytes(), [0x81, 0x80, 0x80, 0x3e]);
        assert_ne!(
            texels[..4],
            [0x81, 0x80, 0x80, 0x3e],
            "the attachment cannot read back the stored f32's own bytes"
        );
        assert_ne!(
            texels[..4],
            EXPECTED_RGBA8_TEXELS,
            "nor the 8-bit quantisation of the same colour"
        );
        assert!(
            !texels.chunks_exact(8).any(|texel| texel == RGBA16F_CLEAR),
            "a surviving clear means the triangle did not cover every texel: {}",
            hex(&texels)
        );
        // `LoadOp::Clear` needs no staging upload, and the attachment leaves
        // through exactly one image→buffer copy (`research/docs/23` §5.3).
        assert_eq!(uploads_after, uploads_before);
        assert_eq!(readbacks_after, readbacks_before + 1);
    }

    /// Partial coverage of the eight-byte format: one texel keeps the fragment
    /// output, the other three keep the `LoadOp::Clear` bytes.
    ///
    /// A wrong clear decode is observable only where the draw does not reach,
    /// so this is the sibling the full-coverage case cannot replace: the clear
    /// payload is four half floats in memory order, and a rail that read it as
    /// four bytes (or as two `u32`s) would land four other texels here
    /// (`research/docs/23` §78).
    #[test]
    fn a_partial_rgba16float_attachment_shows_the_clear_bytes_in_half_order() {
        let Some(context) = device_context() else {
            return;
        };
        let mut blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba16Float,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(
                        ClearColor::from_bytes(&RGBA16F_CLEAR).expect("one eight-byte texel"),
                    ),
                    previous: None,
                    seed: None,
                    resident: None,
                }],
                extent: [2, 2],
                vertex: single_pixel_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .unwrap_or_else(|error| panic!("the partial Rgba16Float pass executes: {error:?}"));
        let texels = blobs
            .attachments
            .remove(0)
            .expect("a stored attachment reads back");
        eprintln!("Rgba16Float partial readback: {}", hex(&texels));
        assert_eq!(texels.len(), 32);
        assert_eq!(
            texels[..8],
            EXPECTED_RGBA16F_TEXEL,
            "the covered texel holds the fragment stage's rounded halves"
        );
        for (index, texel) in texels[8..].chunks(8).enumerate() {
            assert_eq!(
                texel,
                RGBA16F_CLEAR,
                "uncovered texel {} holds the clear's own eight bytes",
                index + 1
            );
        }
    }

    #[test]
    fn offscreen_render_refuses_a_zero_extent() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            stencil: None,
            scissor: None,
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
                seed: None,
                resident: None,
            }],
            extent: [2, 0],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            base_vertex: 0,
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
            vertex_translation: None,
            fragment_translation: None,
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
        stages.contract.color_formats = vec![AttachmentFormat::Rgba8Unorm; maximum + 1];
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        for _ in 0..maximum {
            pass.color_attachments.push(pass.color_attachments[0]);
        }
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes one format per location");
        let previous = vec![None; maximum + 1];

        let refused = match prepare_render_request(
            &stages,
            &pass,
            &previous,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
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

    /// One pass whose stored multisampled depth surface states the resolve the
    /// device admits (`research/docs/23` §3.3, v57).
    fn depth_resolving_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
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
                allocation_id: AllocationId::new(940),
                view_id: ViewId::new(950),
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

    #[test]
    fn prepare_render_request_admits_a_stored_multisampled_depth_resolve_in_the_device_mask() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = depth_resolving_pass();
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes the reviewed single-attachment shape");
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0b1,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("a stored multisampled depth resolve the device admits is well formed");
        assert_eq!(
            request.depth_resolve.map(|resolve| resolve.filter),
            Some(DepthResolveFilter::Sample0)
        );
    }

    #[test]
    fn prepare_render_request_refuses_a_depth_resolve_filter_the_device_does_not_report() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = depth_resolving_pass();
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0b10,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a filter outside the device mask"),
        };
        assert_eq!(refused.slug, "render_depth_resolve_filter_unsupported");
        assert_eq!(refused.fields.get("filter"), Some(&FieldValue::Unsigned(0)));
        assert_eq!(
            refused.fields.get("modes"),
            Some(&FieldValue::Unsigned(0b10))
        );
    }

    #[test]
    fn prepare_render_request_refuses_a_stored_multisampled_depth_without_a_resolve() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = depth_resolving_pass();
        pass.depth_resolve = None;
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0b1,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a stored surface without a resolve"),
        };
        assert_eq!(refused.slug, "render_multisample_depth_store_unsupported");
    }

    /// The present action beside the raster is admitted from v62 on: the
    /// request carries both the raster and the present tail, and the contract's
    /// own rule holds the present source to the attachment view the resolve
    /// lands in (`research/docs/24` §3.5, v62).
    #[test]
    fn prepare_render_request_admits_a_present_action_beside_the_multisample_raster() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(31),
                view_id: ViewId::new(21),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Undefined,
            },
            source: ViewId::new(21),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes the reviewed single-attachment shape");
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("a present action beside the raster is well formed");
        assert_eq!(
            request.multisample.map(|state| state.sample_count),
            Some(SampleCount::Four)
        );
    }

    /// The device property behind the combined surface's mode agreement
    /// (`research/docs/23` §3.3, v60/v70): a device whose `independentResolve`
    /// is false resolves depth and stencil with one shared mode, so only a pair
    /// of filters that name the same Vulkan mode can be executed. The reviewed
    /// stored pair resolves both faces through `sample0`; the other shapes are
    /// the ones the rule refuses, and the `depthResolvedSample` filter has no
    /// Vulkan mode at all (its bit never enters the mask, so the per-filter
    /// check refuses it before this question is asked).
    #[test]
    fn a_shared_resolve_mode_admits_only_the_pairs_that_name_one_mode() {
        assert!(shared_resolve_mode_admits(
            DepthResolveFilter::Sample0,
            StencilResolveFilter::Sample0
        ));
        assert!(!shared_resolve_mode_admits(
            DepthResolveFilter::Min,
            StencilResolveFilter::Sample0
        ));
        assert!(!shared_resolve_mode_admits(
            DepthResolveFilter::Max,
            StencilResolveFilter::Sample0
        ));
        assert!(!shared_resolve_mode_admits(
            DepthResolveFilter::Sample0,
            StencilResolveFilter::DepthResolvedSample
        ));
        // `ResolveModeFlags` has no `Debug`, so the bit is compared as a
        // boolean: Metal's depth-following filter has no Vulkan counterpart and
        // the mapping answers `NONE` for it.
        assert!(
            stencil_resolve_mode(StencilResolveFilter::DepthResolvedSample).is_empty(),
            "Metal's depth-following filter maps to no Vulkan mode"
        );
    }

    /// A stored multisampled stencil surface whose resolve the device mask
    /// admits (`research/docs/23` §3.3, v60).
    fn stencil_resolving_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
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
            compare: StencilCompare::Equal,
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

    #[test]
    fn prepare_render_request_admits_a_stored_multisampled_stencil_resolve_in_the_device_mask() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = stencil_resolving_pass();
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes the reviewed single-attachment shape");
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0b1,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("a stored multisampled stencil resolve the device admits is well formed");
        assert_eq!(
            request.stencil_resolve.map(|resolve| resolve.filter),
            Some(StencilResolveFilter::Sample0)
        );
    }

    #[test]
    fn prepare_render_request_refuses_a_stencil_resolve_filter_the_device_does_not_report() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = stencil_resolving_pass();
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0b10,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a filter outside the device mask"),
        };
        assert_eq!(refused.slug, "render_stencil_resolve_filter_unsupported");
        assert_eq!(refused.fields.get("filter"), Some(&FieldValue::Unsigned(0)));
        assert_eq!(
            refused.fields.get("modes"),
            Some(&FieldValue::Unsigned(0b10))
        );
    }

    #[test]
    fn prepare_render_request_refuses_a_stored_multisampled_stencil_without_a_resolve() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = stencil_resolving_pass();
        pass.stencil_resolve = None;
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0b1,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a stored surface without a resolve"),
        };
        assert_eq!(refused.slug, "render_multisample_stencil_store_unsupported");
    }

    #[test]
    fn prepare_render_request_refuses_the_depth_resolved_sample_without_a_depth_resolve() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = stencil_resolving_pass();
        pass.stencil_resolve = Some(MultisampleStencilResolve {
            filter: StencilResolveFilter::DepthResolvedSample,
        });
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0b1,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse the filter without the depth resolve it names"),
        };
        assert_eq!(refused.slug, "trace_contract_invalid");
    }

    /// The v66 rail-owned combined pair: one multisampled raster that opens
    /// both faces of the one combined surface, keeps neither, and states no
    /// resolve (`research/docs/23` §3.3, v66).
    fn rail_owned_combined_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: None,
            identity: None,
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 2,
            height: 2,
            load: StencilLoadOp::clear(0),
            store: None,
            identity: None,
        });
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Equal,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::IncrementWrap,
            pass_op: StencilOp::Keep,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        pass
    }

    #[test]
    fn prepare_render_request_admits_the_rail_owned_combined_pair() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = rail_owned_combined_pass();
        stages
            .contract
            .validate_against(&pass)
            .expect("the fixture describes the reviewed single-attachment shape");
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the rail-owned combined pair is well formed");
        assert!(request.depth.is_some() && request.stencil.is_some());
        assert!(request.depth_resolve.is_none() && request.stencil_resolve.is_none());
    }

    #[test]
    fn prepare_render_request_refuses_a_lopsided_combined_pair() {
        // The two faces share one surface, so the rail re-asserts the
        // contract's agreement rule for a directly-constructed request: a pair
        // that keeps one face while dropping the other is refused by name
        // (`research/docs/23` §3.3, v66).
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = rail_owned_combined_pass();
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: AllocationId::new(940),
                view_id: ViewId::new(950),
            }),
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Sample0,
        });
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0b1,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("the rail must refuse a pair that keeps one face"),
        };
        assert_eq!(
            refused.slug,
            "render_combined_depth_stencil_store_unsupported"
        );
    }

    #[test]
    fn the_reviewed_combined_format_order_prefers_the_four_byte_one() {
        // The format list's order is the review's own decision
        // (`research/docs/23` §3.3, v66): the four-byte `D32_SFLOAT_S8_UINT`
        // first, because the stored v60 shape reads its depth aspect back as
        // `depth32float`, and the packed `D24_UNORM_S8_UINT` as the rail-owned
        // pair's fallback.
        assert_eq!(
            first_supported_combined_format(|_| true).map(|format| format.as_raw()),
            Some(vk::Format::D32_SFLOAT_S8_UINT.as_raw())
        );
        assert_eq!(
            first_supported_combined_format(|format| format == vk::Format::D24_UNORM_S8_UINT)
                .map(|format| format.as_raw()),
            Some(vk::Format::D24_UNORM_S8_UINT.as_raw())
        );
        assert!(first_supported_combined_format(|_| false).is_none());
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
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                        seed: None,
                        resident: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                        seed: None,
                        resident: None,
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed dual pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(blobs.attachments.len(), 2, "one readback per attachment");
        let location_0 = blobs.attachments[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        let location_1 = blobs.attachments[1]
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
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                        seed: None,
                        resident: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::DontCare,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                        seed: None,
                        resident: None,
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed store-plus-discard pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        assert_eq!(
            blobs.attachments.len(),
            2,
            "one result entry per attachment"
        );
        let stored = blobs.attachments[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        assert_eq!(
            blobs.attachments[1], None,
            "the discarded location reads back nothing"
        );
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
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::Store,
                        load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                        previous: None,
                        seed: None,
                        resident: None,
                    },
                    OffscreenColorAttachment {
                        format: AttachmentFormat::Rgba8Unorm,
                        store: StoreOp::DontCare,
                        load: LoadOp::Load,
                        previous: Some(RenderInputSource::TraceBytes(&previous)),
                        seed: None,
                        resident: None,
                    },
                ],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the load-plus-discard pass executes");

        let stored = blobs.attachments[0]
            .as_ref()
            .expect("location 0 is stored and reads back");
        assert_eq!(*stored, EXPECTED_RGBA8_TEXELS.repeat(4));
        assert_eq!(
            blobs.attachments[1], None,
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
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            stencil: None,
            scissor: None,
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::DontCare,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
                seed: None,
                resident: None,
            }],
            extent: [2, 2],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            base_vertex: 0,
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

        // v45: the depth surface is a landing, so the same all-discarded colour
        // list executes as soon as the pass keeps its depth attachment — and
        // the readback is the depth texels, with no colour entry at all
        // (`research/docs/23` §3.3, v45).
        let depth_only = OffscreenRenderRequest {
            depth: Some(OffscreenDepthAttachment {
                width: 2,
                height: 2,
                clear: Some(1.0),
                test: Some(DepthTest {
                    compare: CompareFunction::Less,
                    write: true,
                }),
                store: Some(DepthStoreOp::Store),
            }),
            ..request
        };
        let readback = execute_offscreen_render(&context, &depth_only)
            .expect("a pass whose only landing is its depth surface executes");
        assert_eq!(
            readback.attachments,
            vec![None],
            "a discarded colour attachment keeps its place in the result list and lands nothing"
        );
        let depth = readback
            .depth
            .expect("the stored depth surface reads back its texels");
        eprintln!("depth-only readback: {}", hex(&depth));
        assert_eq!(depth.len(), 16);
        assert_eq!(depth, 0.0_f32.to_le_bytes().repeat(4));
        assert_ne!(depth, 1.0_f32.to_le_bytes().repeat(4));

        // v46: with *no* colour attachment at all the pass is the depth-only
        // shape proper — the reviewed no-output fragment stage runs, the
        // per-fragment operations still test and write depth, and the depth
        // texels are the entire readback (`research/docs/23` §3.3, v46).
        let no_colour = OffscreenRenderRequest {
            attachments: Vec::new(),
            ..depth_only
        };
        let readback = execute_offscreen_render(&context, &no_colour)
            .expect("a pass with no colour attachment renders into its depth surface");
        assert!(
            readback.attachments.is_empty(),
            "a pass with no colour attachment owes no colour readback"
        );
        let depth = readback
            .depth
            .expect("the stored depth surface reads back its texels");
        eprintln!("zero-colour depth readback: {}", hex(&depth));
        assert_eq!(depth, 0.0_f32.to_le_bytes().repeat(4));
    }

    /// A dual-format list outside the reviewed `[Rgba8Unorm, Rgba8Unorm]` shape
    /// has no fragment stage and is refused before any Vulkan object exists.
    #[test]
    fn an_unreviewed_dual_format_combination_is_refused_before_the_device() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            stencil: None,
            scissor: None,
            attachments: vec![
                OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba8Unorm,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                    previous: None,
                    seed: None,
                    resident: None,
                },
                OffscreenColorAttachment {
                    format: AttachmentFormat::R32Float,
                    store: StoreOp::Store,
                    load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                    previous: None,
                    seed: None,
                    resident: None,
                },
            ],
            extent: [2, 2],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            base_vertex: 0,
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

    /// R1b (`research/docs/23` §70): the declared attachment window has two
    /// named halves, and the rail re-asks both of the value it was handed so a
    /// directly-constructed pass cannot skip admission. The device's own
    /// `maxFramebuffer{Width,Height}` is asked first — it is the half that
    /// cannot be widened by a contract change — and the reviewed ceiling
    /// second, with the slug and fields core admission uses.
    #[test]
    fn attachment_extent_refusals_name_the_device_half_then_the_ceiling() {
        let Some(context) = device_context() else {
            return;
        };
        let limits = context.physical_device_limits();
        let device = [
            u64::from(limits.max_framebuffer_width),
            u64::from(limits.max_framebuffer_height),
        ];
        let ceiling = crate::provider::REVIEWED_ATTACHMENT_CEILING;

        // A request beyond the device's own framebuffer limit reports the
        // device's number, even when it is also beyond the ceiling: that is the
        // half no contract change can widen.
        let mut too_wide = milestone_pass(AttachmentFormat::Rgba8Unorm);
        too_wide.color_attachments[0].width = device[0] + 1;
        too_wide.color_attachments[0].height = 1;
        let refused = refuse_attachment_extent(&context, &too_wide)
            .expect_err("the device's own framebuffer limit is a refusal");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_extent_device_limit");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("width"),
            Some(&FieldValue::Unsigned(device[0] + 1))
        );
        assert_eq!(
            refused.fields.get("maximum_width"),
            Some(&FieldValue::Unsigned(device[0]))
        );

        // A device at least as wide as the review still refuses an extent the
        // reviewed ceiling does not cover, with the pair of limits admission
        // publishes.
        if device[0] >= ceiling[0] && device[1] >= ceiling[1] {
            let mut over_ceiling = milestone_pass(AttachmentFormat::Rgba8Unorm);
            over_ceiling.color_attachments[0].width = ceiling[0] + 1;
            over_ceiling.color_attachments[0].height = ceiling[1] + 1;
            let refused = refuse_attachment_extent(&context, &over_ceiling)
                .expect_err("the reviewed ceiling is a refusal");
            eprintln!("refused: {refused:?}");
            assert_eq!(refused.slug, "attachment_dimension_limit");
            assert_eq!(refused.class, ProviderErrorClass::Capability);
            assert_eq!(
                refused.fields.get("maximum_width"),
                Some(&FieldValue::Unsigned(ceiling[0]))
            );
            assert_eq!(
                refused.fields.get("maximum_height"),
                Some(&FieldValue::Unsigned(ceiling[1]))
            );

            // The boundary extent itself is inside the window: the R1b
            // fixture's 64×64 pass gets past this gate.
            let mut boundary = milestone_pass(AttachmentFormat::Rgba8Unorm);
            boundary.color_attachments[0].width = ceiling[0];
            boundary.color_attachments[0].height = ceiling[1];
            refuse_attachment_extent(&context, &boundary)
                .expect("the reviewed boundary extent is inside the window");
        }
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

        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None, None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
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
        let refused = execute_render_pass(&context, &stages, &pass, &[None], &[], None)
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
        let refused = execute_render_pass(&context, &stages, &pass, &[None], &[], None)
            .expect_err("`Load` needs an upload rail this increment does not have");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_load_op_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    /// The v82 seed shape at the rail's own request level
    /// (`research/docs/23` §82): a four-sample attachment whose `Load` the rail
    /// executes with a `CLEAR`-opened seed pass.
    ///
    /// The device half: the seed defines every sample of the image, a scissor
    /// keeps the draw off the right column, and the resolve then lands the
    /// declared texel there byte for byte and the fragment output in the drawn
    /// column. A rail that dropped the seed would land the driver's own
    /// undefined contents in the kept column, which is exactly what the
    /// pre-v82 shape could not distinguish.
    #[test]
    fn a_multisampled_load_is_seeded_by_one_texel() {
        let Some(context) = device_context() else {
            return;
        };
        let seed_texel: [u8; 4] = [0x22, 0x44, 0x66, 0x89];
        let previous = seed_texel.repeat(4);
        let blobs = execute_offscreen_render(
            &context,
            &OffscreenRenderRequest {
                textures: Vec::new(),
                blend: None,
                multisample: Some(MultisampleState {
                    sample_count: SampleCount::Four,
                }),
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                // The left column is drawn; the right one keeps the seed.
                scissor: Some([0, 0, 1, 2]),
                attachments: vec![OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba8Unorm,
                    store: StoreOp::Store,
                    load: LoadOp::Load,
                    previous: Some(RenderInputSource::TraceBytes(&previous)),
                    seed: Some(ClearColor::new(seed_texel)),
                    resident: None,
                }],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the seeded multisampled load executes");
        let texels = blobs.attachments[0]
            .as_ref()
            .expect("the stored attachment reads back");
        eprintln!("seeded multisampled readback: {texels:02x?}");
        let expected = [
            EXPECTED_RGBA8_TEXELS,
            seed_texel,
            EXPECTED_RGBA8_TEXELS,
            seed_texel,
        ]
        .concat();
        assert_eq!(
            *texels, expected,
            "the drawn column resolves the fragment output and the kept one the seed"
        );
    }

    /// The two declarations the seed route cannot read, refused before any
    /// device object exists (`research/docs/23` §82).
    ///
    /// A four-sample attachment whose declared window is not one repeated
    /// texel has no clear value the seed pass could state, and a borrowed
    /// window's bytes would have to be read on the host — the §74 property this
    /// route cannot keep. Both are value-level facts, so they hold on a host
    /// with no device.
    #[test]
    fn a_multisampled_load_refuses_a_nonuniform_seed_and_states_the_admitted_one() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut nonuniform = milestone_pass(AttachmentFormat::Rgba8Unorm);
        nonuniform.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        nonuniform.color_attachments[0].load = LoadOp::Load;
        nonuniform
            .validate()
            .expect("the fixture pass is a legal shape");
        let bytes = [0x22, 0x44, 0x66, 0x89, 0x22, 0x44, 0x66, 0xff].repeat(2);
        let view = BufferView {
            view_id: ViewId::new(11),
            metal_binding: 0,
            allocation_id: AllocationId::new(12),
            offset: 0,
            length: u64::try_from(bytes.len()).expect("fixture length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes.clone()),
        };
        // `prepare_render_request` is where the seed is resolved, so it is also
        // where both refusals live; the no-copy arm needs a lease channel the
        // caller can state, which the borrowed case below does.
        let error = match prepare_render_request(
            &stages,
            &nonuniform,
            &[Some(&view)],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a per-texel seed has no clear value"),
        };
        eprintln!("nonuniform multisampled seed refused: {error:?}");
        assert_eq!(error.slug, "render_multisample_load_nonuniform_unsupported");
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );

        let uniform = [0x22, 0x44, 0x66, 0x89].repeat(4);
        let uniform_view = BufferView {
            length: u64::try_from(uniform.len()).expect("fixture length"),
            source: BufferSource::OwnedBytes(uniform.clone()),
            ..view
        };
        let declared = [Some(&uniform_view)];
        let admitted = prepare_render_request(
            &stages,
            &nonuniform,
            &declared,
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("one repeated texel is the seed the rail can state");
        assert_eq!(
            admitted.attachments[0].seed,
            Some(ClearColor::new([0x22, 0x44, 0x66, 0x89])),
            "the seed travels with the request for the seed pass to clear"
        );
        assert!(matches!(admitted.attachments[0].load, LoadOp::Load));
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
                textures: Vec::new(),
                blend: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
                cull: None,
                depth: None,
                base_vertex: 0,
                stencil: None,
                scissor: None,
                attachments: vec![OffscreenColorAttachment {
                    format: AttachmentFormat::Rgba8Unorm,
                    store: StoreOp::Store,
                    load: LoadOp::DontCare,
                    previous: None,
                    seed: None,
                    resident: None,
                }],
                extent: [2, 2],
                vertex: milestone_vertex(),
                translated_fragment: None,
                vertex_streams: Vec::new(),
                draw: DrawShape::Milestone,
                instance_count: 1,
                index_stream: None,
                indirect: None,
            },
        )
        .expect("the reviewed single-attachment DontCare pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();
        let texels = blobs.attachments[0]
            .as_ref()
            .expect("the stored attachment reads back");
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
        prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("a DontCare attachment with no previous bytes plans");
        let declared = attachment_previous_view(
            BufferSource::OwnedBytes(vec![0x11; 16]),
            AllocationId::new(9),
        );
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&declared)],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
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

    /// A stored depth attachment lands its own texels through the readback
    /// channel (`research/docs/23` §3.3, v43).
    ///
    /// The pass clears the surface to one and then draws the milestone's
    /// full-screen triangle with a `less` test and depth writes on, so every
    /// texel the draw covers takes the triangle's own depth (`0.0`) and the
    /// stored bytes are `00000000` rather than the clear's `0000803f`. A rail
    /// that opened the surface but never copied it out would hand back `None`,
    /// and one that skipped the depth write would hand back the clear value:
    /// both are distinguishable from the expected bytes.
    ///
    /// The discarded shape is measured in the same test: it still opens the
    /// surface — the pre-v43 shape — and hands nothing back, which is what the
    /// absence of the wide sections means.
    #[test]
    fn a_stored_depth_attachment_reads_back_its_own_texels() {
        let Some(context) = device_context() else {
            return;
        };
        let request = |store: Option<DepthStoreOp>| OffscreenRenderRequest {
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: Some(OffscreenDepthAttachment {
                width: 2,
                height: 2,
                clear: Some(1.0),
                test: Some(DepthTest {
                    compare: CompareFunction::Less,
                    write: true,
                }),
                store,
            }),
            stencil: None,
            base_vertex: 0,
            scissor: None,
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
                seed: None,
                resident: None,
            }],
            extent: [2, 2],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            index_stream: None,
            indirect: None,
        };

        let readback = execute_offscreen_render(&context, &request(Some(DepthStoreOp::Store)))
            .expect("the depth-storing pass executes");
        let depth = readback
            .depth
            .expect("a stored depth attachment reads back its texels");
        eprintln!("stored depth readback: {}", hex(&depth));
        assert_eq!(depth.len(), 16, "one four-byte texel per pixel");
        assert_eq!(depth, 0.0_f32.to_le_bytes().repeat(4));
        assert_ne!(
            depth,
            1.0_f32.to_le_bytes().repeat(4),
            "the stored texels are the draw's depth, not the clear value"
        );
        assert_eq!(
            readback.attachments[0].as_deref(),
            Some(EXPECTED_RGBA8_TEXELS.repeat(4).as_slice()),
            "the colour landing is unchanged by the depth readback"
        );

        let discarded = execute_offscreen_render(&context, &request(None))
            .expect("the discarded-depth pass executes");
        assert_eq!(
            discarded.depth, None,
            "a pass that states no store action hands back no depth bytes"
        );
        assert_eq!(
            discarded.attachments[0].as_deref(),
            Some(EXPECTED_RGBA8_TEXELS.repeat(4).as_slice())
        );
    }

    /// A stored stencil attachment lands its own one-byte texels through the
    /// readback channel (`research/docs/23` §3.3, v49).
    ///
    /// The pass clears the surface to zero and then draws the milestone's
    /// full-screen triangle over a 4×4 surface with `equal 0` against reference
    /// zero and `increment_wrap` on success, so every covered texel takes the
    /// first (and only) triangle's increment and the stored bytes are sixteen
    /// `01`s. A rail that opened the surface but never copied it out would hand
    /// back `None`, and one that skipped the stencil write would hand back the
    /// clear's sixteen `00`s: both are distinguishable from the expected bytes.
    ///
    /// The discarded shape is measured in the same test: it still opens the
    /// surface — the pre-v49 shape — and hands nothing back, which is what the
    /// absence of the wide section means.
    #[test]
    fn a_stored_stencil_attachment_reads_back_its_own_texels() {
        let Some(context) = device_context() else {
            return;
        };
        let request = |store: Option<StoreOp>| OffscreenRenderRequest {
            textures: Vec::new(),
            blend: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
            cull: None,
            depth: None,
            stencil: Some(OffscreenStencilAttachment {
                width: 4,
                height: 4,
                clear: Some(0),
                test: Some(StencilTest {
                    compare: StencilCompare::Equal,
                    reference: 0,
                    read_mask: 0xff,
                    write_mask: 0xff,
                    fail_op: StencilOp::Keep,
                    depth_fail_op: StencilOp::Keep,
                    pass_op: StencilOp::IncrementWrap,
                }),
                store,
            }),
            base_vertex: 0,
            scissor: None,
            attachments: vec![OffscreenColorAttachment {
                format: AttachmentFormat::Rgba8Unorm,
                store: StoreOp::Store,
                load: LoadOp::Clear(ClearColor::new([CLEAR_SENTINEL; 4])),
                previous: None,
                seed: None,
                resident: None,
            }],
            extent: [4, 4],
            vertex: milestone_vertex(),
            translated_fragment: None,
            vertex_streams: Vec::new(),
            draw: DrawShape::Milestone,
            instance_count: 1,
            index_stream: None,
            indirect: None,
        };
        let expected_colour = EXPECTED_RGBA8_TEXELS.repeat(16);

        let readback = execute_offscreen_render(&context, &request(Some(StoreOp::Store)))
            .expect("the stencil-storing pass executes");
        let stencil = readback
            .stencil
            .expect("a stored stencil attachment reads back its texels");
        eprintln!("stored stencil readback: {}", hex(&stencil));
        assert_eq!(
            stencil.len(),
            16,
            "one one-byte texel per pixel of the 4×4 surface"
        );
        assert_eq!(
            stencil, [0x01u8; 16],
            "`increment_wrap` leaves 01 in every texel the triangle covers"
        );
        assert_ne!(
            stencil, [0x00u8; 16],
            "the stored texels are the draw's stencil writes, not the clear value"
        );
        assert_eq!(
            readback.attachments[0].as_deref(),
            Some(expected_colour.as_slice()),
            "the colour landing is unchanged by the stencil readback"
        );

        let discarded = execute_offscreen_render(&context, &request(None))
            .expect("the discarded-stencil pass executes");
        assert_eq!(
            discarded.stencil, None,
            "a pass that states no store action hands back no stencil bytes"
        );
        assert_eq!(
            discarded.attachments[0].as_deref(),
            Some(expected_colour.as_slice())
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
        let error = execute_render_pass(&context, &stages, &pass, &[None], &[], None)
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

    /// One `OpEntryPoint` instruction as the module's own words, so the entry
    /// scan can be observed without a translation: `model` is the SPIR-V
    /// execution model, `name` the literal the instruction carries.
    fn entry_point_instruction(model: u32, name: &str) -> Vec<u32> {
        let mut literal = name.as_bytes().to_vec();
        literal.push(0);
        while !literal.len().is_multiple_of(4) {
            literal.push(0);
        }
        let mut instruction = vec![0, model, 1];
        for word in literal.chunks_exact(4) {
            instruction.push(u32::from_le_bytes(word.try_into().expect("four-byte word")));
        }
        instruction[0] = ((instruction.len() as u32) << 16) | spirv::Op::EntryPoint as u32;
        instruction
    }

    /// One `OpEntryPoint` header with no operands at all: a module that ships
    /// one cannot be indexed past its own words.
    fn truncated_entry_point_instruction() -> Vec<u32> {
        vec![(1 << 16) | spirv::Op::EntryPoint as u32]
    }

    fn module_with(instructions: &[Vec<u32>]) -> Vec<u8> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 2, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    }

    /// The rail binds the entry point a translated module declares, so the scan
    /// is what makes "which function does this module export" answerable
    /// without repeating the translator's own literal. A module that declares
    /// two entry points of one stage is refused, because the pipeline would have
    /// to pick one.
    #[test]
    fn module_entry_point_reads_the_single_declared_entry_and_refuses_two() {
        let vertex_model = spirv::ExecutionModel::Vertex as u32;
        let fragment_model = spirv::ExecutionModel::Fragment as u32;

        let one = module_with(&[entry_point_instruction(vertex_model, "main")]);
        assert_eq!(
            module_entry_point(&one, spirv::ExecutionModel::Vertex).as_deref(),
            Some("main")
        );
        // The other stage's model is not this stage's entry point.
        assert_eq!(
            module_entry_point(&one, spirv::ExecutionModel::Fragment),
            None
        );
        let fragment_one = module_with(&[entry_point_instruction(fragment_model, "main")]);
        assert_eq!(
            module_entry_point(&fragment_one, spirv::ExecutionModel::Fragment).as_deref(),
            Some("main")
        );
        assert_eq!(
            module_entry_point(&fragment_one, spirv::ExecutionModel::Vertex),
            None
        );

        let two = module_with(&[
            entry_point_instruction(vertex_model, "first"),
            entry_point_instruction(vertex_model, "second"),
        ]);
        assert_eq!(
            module_entry_point(&two, spirv::ExecutionModel::Vertex),
            None,
            "two entry points of one stage leave the pipeline nothing to bind"
        );

        // A module whose last word is an `OpEntryPoint` header with no operands
        // is refused, not indexed past: the scan reads only what the word count
        // covers.
        let malformed = module_with(&[truncated_entry_point_instruction()]);
        assert_eq!(
            module_entry_point(&malformed, spirv::ExecutionModel::Vertex),
            None
        );

        // The reviewed modules declare their own entry names, which is the same
        // question asked of a module the rail compiled itself.
        assert_eq!(
            module_entry_point(FULL_SCREEN_TRIANGLE_VERT_SPV, spirv::ExecutionModel::Vertex)
                .as_deref(),
            Some(FULL_SCREEN_TRIANGLE_VERTEX_ENTRY)
        );
        assert_eq!(
            module_entry_point(SOLID_UNORM8_FRAG_SPV, spirv::ExecutionModel::Fragment).as_deref(),
            Some(SOLID_FRAGMENT_ENTRY)
        );
    }

    /// A reviewed registration binds the entry the contract names; a translated
    /// one binds the entry the module declares. The two are told apart by which
    /// arm of the gate the stage arrived through.
    #[test]
    fn a_reviewed_stage_binds_the_entry_its_contract_names() {
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        assert_eq!(
            bound_stage_entry(&stages, RenderStage::Vertex)
                .expect("a reviewed stage binds its contract entry")
                .as_ref(),
            "vertex_main"
        );
        assert_eq!(
            bound_stage_entry(&stages, RenderStage::Fragment)
                .expect("a reviewed stage binds its contract entry")
                .as_ref(),
            SOLID_FRAGMENT_ENTRY
        );
    }

    /// A vertex module outside the reviewed set, handed to the reviewed
    /// registration, is refused by name: the rail has no reflection to check it
    /// against and will not execute it on the strength of its bytes alone.
    #[test]
    fn a_registration_refuses_a_vertex_module_outside_the_reviewed_set() {
        let mut stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        // The reviewed fragment module is a well-formed SPIR-V module and a
        // fragment stage: it is not one of the reviewed *vertex* stages, which
        // is exactly the module the rail cannot account for.
        stages.vertex_spirv = SOLID_UNORM8_FRAG_SPV.to_vec();
        let refused = stages
            .validate()
            .expect_err("a vertex module the rail did not compile is not a reviewed stage");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_translation_unavailable");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("stage"),
            Some(&FieldValue::Text("vertex".to_owned()))
        );
        assert_eq!(
            refused.fields.get("entry"),
            Some(&FieldValue::Text("vertex_main".to_owned()))
        );
    }

    /// One pool view over `allocation`, carrying `source` (`docs/23` §71).
    fn leased_stream_view(source: BufferSource, allocation: AllocationId) -> BufferView {
        BufferView {
            view_id: ViewId::new(41),
            metal_binding: 0,
            allocation_id: allocation,
            offset: 0,
            length: 32,
            access: BufferAccess::Read,
            attribute_stride: None,
            source,
        }
    }

    /// One pool view declaring a `LoadOp::Load` attachment's previous contents
    /// (`docs/23` §3.3/§74): the reviewed 2x2 `rgba8_unorm` attachment's own
    /// 16 tightly packed bytes.
    fn attachment_previous_view(source: BufferSource, allocation: AllocationId) -> BufferView {
        BufferView {
            view_id: ViewId::new(61),
            metal_binding: 0,
            allocation_id: allocation,
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source,
        }
    }

    /// One lease registration over `allocation`'s first `length` bytes.
    fn lease_registration(
        lease_id: LeaseId,
        allocation: AllocationId,
        length: u64,
        epoch: DeviceEpoch,
    ) -> LeaseReservation {
        LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: allocation,
                owner_epoch: epoch,
            },
            offset: 0,
            length,
        }
    }

    /// A lease-backed render input handed to a rail with no lease channel is
    /// refused under the name this rail published before the channel existed,
    /// with the storage mode it arrived under (`docs/23` §71, R3c).
    #[test]
    fn a_lease_backed_render_input_is_refused_without_a_lease_channel() {
        let allocation = AllocationId::new(43);
        let view = leased_stream_view(BufferSource::StagedLease(LeaseId::new(7)), allocation);
        let refused = resolve_render_input(&view, None, RenderInputRole::Vertex, 1)
            .expect_err("a lease cannot be read without the registry that imported it");
        eprintln!("no channel: {refused:?}");
        assert_eq!(refused.slug, "render_vertex_buffer_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
        assert_eq!(
            refused.fields.get("binding"),
            Some(&FieldValue::Unsigned(1))
        );

        // The index half keeps its own slug, so a capture can still tell which
        // of the two inputs the rail could not read.
        let view = leased_stream_view(BufferSource::BorrowedNoCopy(LeaseId::new(8)), allocation);
        let refused = resolve_render_input(&view, None, RenderInputRole::Index, 0)
            .expect_err("a no-copy window cannot be resolved without its registry");
        eprintln!("no channel: {refused:?}");
        assert_eq!(refused.slug, "render_index_buffer_unsupported");
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
    }

    /// A borrowed render input on a device that does not import host memory is
    /// refused under the name core admission and the compute rail publish for
    /// the same fact (`docs/23` §71, R3c).
    #[test]
    fn a_borrowed_render_input_is_refused_without_host_import() {
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = ResourceTableSnapshot::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: DeviceEpoch::new(3),
            host_import_alignment: 0,
        };
        let view = leased_stream_view(
            BufferSource::BorrowedNoCopy(LeaseId::new(9)),
            AllocationId::new(43),
        );
        let refused = resolve_render_input(&view, Some(&leases), RenderInputRole::Vertex, 0)
            .expect_err("a device without host import cannot bind the owner's window");
        eprintln!("no host import: {refused:?}");
        assert_eq!(refused.slug, "storage_mode_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(refused.fields.get("view"), Some(&FieldValue::Unsigned(41)));
    }

    /// An owner window whose address misses the import alignment is refused by
    /// name before any Vulkan import exists (`docs/23` §71, R3c).
    #[test]
    fn a_borrowed_render_input_is_refused_when_its_pointer_misses_the_alignment() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(11);
        let allocation = AllocationId::new(43);
        let reservation = lease_registration(lease_id, allocation, 32, epoch);
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 64,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers its allocation");
        let borrowed = BorrowedLeaseRegistry::new();
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000 + 1)
                    .expect("a non-null owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &Arc::new(borrowed),
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: 4096,
        };
        let view = leased_stream_view(BufferSource::BorrowedNoCopy(lease_id), allocation);
        let refused = resolve_render_input(&view, Some(&leases), RenderInputRole::Vertex, 0)
            .expect_err("a pointer one byte past the alignment cannot be imported");
        eprintln!("misaligned window: {refused:?}");
        assert_eq!(refused.slug, "lease_alignment_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("alignment"),
            Some(&FieldValue::Unsigned(4096))
        );
        assert_eq!(refused.fields.get("lease"), Some(&FieldValue::Unsigned(11)));
    }

    /// The reviewed loading pass: the milestone's 2x2 attachment opened with
    /// `LoadOp::Load`, so its previous contents have to resolve before the rail
    /// reads a device (`docs/23` §74, R5b).
    fn loading_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(AttachmentFormat::Rgba8Unorm);
        pass.color_attachments[0].load = LoadOp::Load;
        pass
    }

    /// One lease context over `resources` with the given host-import
    /// alignment, exactly as the provider builds it per submission
    /// (`docs/23` §74).
    fn attachment_lease_context<'a>(
        staging: &'a LeaseRegistry,
        borrowed: &'a Arc<BorrowedLeaseRegistry>,
        resources: &'a ResourceTableSnapshot,
        epoch: DeviceEpoch,
        host_import_alignment: u64,
    ) -> RenderLeaseContext<'a> {
        RenderLeaseContext {
            staging,
            borrowed,
            resources,
            device_epoch: epoch,
            host_import_alignment,
        }
    }

    /// A loading attachment's previous contents resolve through the same three
    /// arms a stream's bytes do (`docs/23` §74, R5b).
    ///
    /// The staged arm is the resolution the ownership states: the declaring
    /// view names an imported lease and the rail reads the provider's own copy
    /// of it, exactly as the trace-owned arm reads the view's own bytes.
    #[test]
    fn a_staged_lease_attachment_load_resolves_into_the_providers_copy() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(21);
        let allocation = AllocationId::new(45);
        let reservation = lease_registration(lease_id, allocation, 16, epoch);
        let staged = LeaseRegistry::new();
        staged
            .import(
                StagedLease::new(reservation, vec![0x2a; 16])
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 16,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers its view");
        let leases = attachment_lease_context(&staged, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = loading_pass();
        let view = attachment_previous_view(BufferSource::StagedLease(lease_id), allocation);
        let declared = [Some(&view)];
        let request = prepare_render_request(
            &stages,
            &pass,
            &declared,
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the staged copy carries the attachment's previous contents");
        let [attachment] = request.attachments.as_slice() else {
            panic!("the milestone pass carries one colour attachment");
        };
        let Some(RenderInputSource::StagedBytes(bytes)) = &attachment.previous else {
            panic!(
                "a staged lease resolves into the provider's own copy: {:?}",
                attachment.previous
            );
        };
        assert_eq!(bytes.as_slice(), [0x2a; 16]);
        assert!(
            RenderInputRetains::retain(Some(&leases), &request)
                .expect("a staged arm takes no hold")
                .is_none(),
            "a staged copy has no owner mapping to retain"
        );

        // The staged copy is the provider's, so releasing it is what makes the
        // same declaration unreadable, under the registry's own name.
        staged
            .release(lease_id)
            .expect("the staged copy is released");
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a released staged lease cannot be read"),
        };
        eprintln!("released staged lease refused: {refused:?}");
        assert_eq!(refused.slug, "lease_not_imported");
        assert_eq!(refused.class, ProviderErrorClass::Args);
    }

    /// A loading attachment whose declaring view names a lease the rail cannot
    /// resolve is refused under the source's own name, not under the load
    /// operation's (`docs/23` §74, R5b).
    #[test]
    fn an_attachment_load_is_refused_without_a_lease_channel() {
        let allocation = AllocationId::new(45);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = loading_pass();
        let view =
            attachment_previous_view(BufferSource::BorrowedNoCopy(LeaseId::new(22)), allocation);
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a lease cannot be read without the registry that imported it"),
        };
        eprintln!("no channel: {refused:?}");
        assert_eq!(refused.slug, "render_attachment_load_source_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            refused.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// A borrowed attachment load on a device that does not import host memory
    /// is refused under the storage mode's published name, with the role that
    /// could not be bound (`docs/23` §74, R5b).
    #[test]
    fn a_borrowed_attachment_load_is_refused_without_host_import() {
        let epoch = DeviceEpoch::new(3);
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = ResourceTableSnapshot::new();
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 0);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = loading_pass();
        let view = attachment_previous_view(
            BufferSource::BorrowedNoCopy(LeaseId::new(23)),
            AllocationId::new(45),
        );
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a device without host import cannot read the owner's window"),
        };
        eprintln!("no host import: {refused:?}");
        assert_eq!(refused.slug, "storage_mode_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            refused.fields.get("role"),
            Some(&FieldValue::Text(
                "render_attachment_load_source_unsupported".to_owned()
            ))
        );
    }

    /// An owner window whose address misses the device's host-import alignment
    /// is refused by name before any Vulkan import exists (`docs/23` §74).
    #[test]
    fn a_borrowed_attachment_load_is_refused_when_its_pointer_misses_the_alignment() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(24);
        let allocation = AllocationId::new(45);
        let reservation = lease_registration(lease_id, allocation, 16, epoch);
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 32,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers its view");
        let borrowed = BorrowedLeaseRegistry::new();
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000 + 1)
                    .expect("a non-null owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(borrowed);
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = loading_pass();
        let view = attachment_previous_view(BufferSource::BorrowedNoCopy(lease_id), allocation);
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a pointer one byte past the alignment cannot be imported"),
        };
        eprintln!("misaligned attachment window: {refused:?}");
        assert_eq!(refused.slug, "lease_alignment_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("lease"), Some(&FieldValue::Unsigned(24)));
        assert_eq!(
            refused.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// An owner's own mapping is not a seed the multisampled route can read
    /// (`research/docs/23` §82): the seed pass's clear value is host state, so
    /// the window would be read here instead of by the device at execution —
    /// the §74 property the borrowed arm of a single-sample load keeps. The
    /// refusal is by name, before any import or image exists.
    #[test]
    fn a_borrowed_multisampled_seed_is_refused_by_name() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(26);
        let allocation = AllocationId::new(47);
        let reservation = lease_registration(lease_id, allocation, 16, epoch);
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 16,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers its view");
        let borrowed = BorrowedLeaseRegistry::new();
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000)
                    .expect("a null-free owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(borrowed);
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let mut pass = loading_pass();
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        let view = attachment_previous_view(BufferSource::BorrowedNoCopy(lease_id), allocation);
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a borrowed seed would be read on the host"),
        };
        eprintln!("borrowed multisampled seed refused: {refused:?}");
        assert_eq!(refused.slug, "render_multisample_load_borrowed_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            refused.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// A declaring view whose window is not the attachment's own extent is
    /// refused by name, under the slug the native rail publishes for the same
    /// fact (`docs/23` §74, R5b).
    #[test]
    fn an_attachment_load_refuses_a_window_that_is_not_its_extent() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(25);
        let allocation = AllocationId::new(46);
        let reservation = lease_registration(lease_id, allocation, 32, epoch);
        let staged = LeaseRegistry::new();
        staged
            .import(
                StagedLease::new(reservation, vec![0x3b; 32])
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 32,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers its view");
        let leases = attachment_lease_context(&staged, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_stages(AttachmentFormat::Rgba8Unorm);
        let pass = loading_pass();
        let view = BufferView {
            view_id: ViewId::new(62),
            metal_binding: 0,
            allocation_id: allocation,
            offset: 0,
            length: 32,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::StagedLease(lease_id),
        };
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[Some(&view)],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a 2x2 attachment reads sixteen bytes, not thirty-two"),
        };
        eprintln!("over-wide window refused: {refused:?}");
        assert_eq!(refused.slug, "render_attachment_initial_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Args);
        assert_eq!(
            refused.fields.get("expected_bytes"),
            Some(&FieldValue::Unsigned(16))
        );
        assert_eq!(
            refused.fields.get("resolved_bytes"),
            Some(&FieldValue::Unsigned(32))
        );
    }

    /// The sampled texture a lease resolves, as the resolution fixtures build
    /// it: the reviewed 4x4 `rgba8_unorm` surface with a lease source over
    /// `allocation` (`docs/23` §75, R5c).
    fn leased_sampled_texture_view(source: TextureSource, allocation: AllocationId) -> TextureView {
        let mut view = sampled_texture_view(4, 4);
        view.allocation_id = allocation;
        view.source = source;
        view
    }

    /// A sampled texture's bytes resolve through the same three arms a stream's
    /// do (`docs/23` §75, R5c).
    ///
    /// The staged arm is the resolution the ownership states: the texture names
    /// an imported lease and the rail reads the provider's own copy of it,
    /// while the window itself is the texture's own extent at the reservation's
    /// start — a page-aligned reservation larger than the texture is padding,
    /// not a second declaration.
    #[test]
    fn a_staged_lease_render_texture_resolves_into_the_providers_copy() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(31);
        let allocation = AllocationId::new(47);
        // Four kilobytes, as the import rules make owners reserve: only the
        // first sixty-four bytes are the texture.
        let reservation = lease_registration(lease_id, allocation, 4096, epoch);
        let mut texels = vec![0x2b; 4096];
        for (index, byte) in texels.iter_mut().enumerate().take(64) {
            *byte = index as u8;
        }
        let staged = LeaseRegistry::new();
        staged
            .import(
                StagedLease::new(reservation, texels.clone())
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 4096,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers the texture");
        let leases = attachment_lease_context(&staged, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::StagedLease(lease_id),
            allocation,
        )];
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the staged copy carries the texture's texels");
        let [texture] = request.textures.as_slice() else {
            panic!("the sampled pass carries one texture");
        };
        let RenderInputSource::StagedBytes(bytes) = &texture.source else {
            panic!(
                "a staged lease resolves into the provider's own copy: {:?}",
                texture.source
            );
        };
        assert_eq!(
            bytes.as_slice(),
            &texels[..64],
            "the window is the texture's own extent at the reservation's start"
        );
        assert!(
            RenderInputRetains::retain(Some(&leases), &request)
                .expect("a staged arm takes no hold")
                .is_none(),
            "a staged copy has no owner mapping to retain"
        );

        // The staged copy is the provider's, so releasing it is what makes the
        // same declaration unreadable, under the registry's own name.
        staged
            .release(lease_id)
            .expect("the staged copy is released");
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a released staged lease cannot be read"),
        };
        eprintln!("released staged texture lease refused: {refused:?}");
        assert_eq!(refused.slug, "lease_not_imported");
        assert_eq!(refused.class, ProviderErrorClass::Args);
    }

    /// A sampled texture whose declaration names a lease the rail cannot
    /// resolve is refused under the sampler's own source name, the same one the
    /// loading attachment's source arm publishes (`docs/23` §75, R5c).
    #[test]
    fn a_render_texture_is_refused_without_a_lease_channel() {
        let allocation = AllocationId::new(47);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::BorrowedNoCopy(LeaseId::new(32)),
            allocation,
        )];
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a lease cannot be read without the registry that imported it"),
        };
        eprintln!("no channel: {refused:?}");
        assert_eq!(refused.slug, "render_texture_source_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            refused.fields.get("binding"),
            Some(&FieldValue::Unsigned(0))
        );

        // The staged arm keeps the same name and states its own storage mode:
        // one fact, one slug, whichever lease form arrived.
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::StagedLease(LeaseId::new(33)),
            allocation,
        )];
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            None,
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a staged lease cannot be read without its registry"),
        };
        eprintln!("no channel: {refused:?}");
        assert_eq!(refused.slug, "render_texture_source_unsupported");
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
    }

    /// A borrowed texture on a device that does not import host memory is
    /// refused under the storage mode's published name, with the role that
    /// could not be bound (`docs/23` §75, R5c).
    #[test]
    fn a_borrowed_render_texture_is_refused_without_host_import() {
        let epoch = DeviceEpoch::new(3);
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = ResourceTableSnapshot::new();
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 0);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::BorrowedNoCopy(LeaseId::new(34)),
            AllocationId::new(47),
        )];
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a device without host import cannot read the owner's window"),
        };
        eprintln!("no host import: {refused:?}");
        assert_eq!(refused.slug, "storage_mode_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            refused.fields.get("role"),
            Some(&FieldValue::Text(
                "render_texture_source_unsupported".to_owned()
            ))
        );
        // The texture's own view identity travels with the refusal, exactly as
        // a stream's does.
        assert_eq!(refused.fields.get("view"), Some(&FieldValue::Unsigned(83)));
    }

    /// An owner window whose address misses the device's host-import alignment
    /// is refused by name before any Vulkan import exists (`docs/23` §75).
    #[test]
    fn a_borrowed_render_texture_is_refused_when_its_pointer_misses_the_alignment() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(35);
        let allocation = AllocationId::new(47);
        let reservation = lease_registration(lease_id, allocation, 4096, epoch);
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 4096,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers the texture");
        let borrowed = BorrowedLeaseRegistry::new();
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000 + 1)
                    .expect("a non-null owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(borrowed);
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::BorrowedNoCopy(lease_id),
            allocation,
        )];
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a pointer one byte past the alignment cannot be imported"),
        };
        eprintln!("misaligned texture window: {refused:?}");
        assert_eq!(refused.slug, "lease_alignment_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("lease"), Some(&FieldValue::Unsigned(35)));
        assert_eq!(
            refused.fields.get("binding"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// The borrowed arm's window is the texture's own extent at the owner
    /// reservation's start, and the pass holds that lease exactly once
    /// (`docs/23` §75, R5c).
    ///
    /// Host side: no device exists, so the window is the registry's answer and
    /// the hold is the registry's own count. The e2e fixture is what shows the
    /// device reading those pages.
    #[test]
    fn a_borrowed_render_texture_resolves_into_the_owners_window() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(36);
        let allocation = AllocationId::new(47);
        let reservation = lease_registration(lease_id, allocation, 4096, epoch);
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 4096,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers the texture");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000)
                    .expect("an aligned non-null owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staging = LeaseRegistry::new();
        let leases = attachment_lease_context(&staging, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::BorrowedNoCopy(lease_id),
            allocation,
        )];
        let request = prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        )
        .expect("the owner's window resolves");
        let [texture] = request.textures.as_slice() else {
            panic!("the sampled pass carries one texture");
        };
        let RenderInputSource::Borrowed { lease, window } = &texture.source else {
            panic!(
                "a no-copy texture resolves into the owner's window: {:?}",
                texture.source
            );
        };
        assert_eq!(*lease, lease_id);
        assert_eq!(window.pointer, 0x2000);
        assert_eq!(
            window.len, 64,
            "the window is the texture's own extent, not the page-aligned reservation"
        );
        assert_eq!(
            borrowed.outstanding(lease_id),
            Some(0),
            "resolution alone takes no hold"
        );

        // One hold per no-copy window, taken before the first import and given
        // back at the fence: dropping the guard is this rail's retirement
        // point, exactly as the provider drops it after the pass is terminal.
        let retains = RenderInputRetains::retain(Some(&leases), &request)
            .expect("the no-copy window is retained")
            .expect("a no-copy texture takes a hold");
        assert_eq!(borrowed.outstanding(lease_id), Some(1));
        drop(retains);
        assert_eq!(borrowed.outstanding(lease_id), Some(0));
    }

    /// A reservation that does not cover the texture's own extent is refused
    /// under the registry's own range name instead of being read past its end
    /// (`docs/23` §75, R5c).
    #[test]
    fn a_render_texture_refuses_a_reservation_that_misses_its_extent() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(37);
        let allocation = AllocationId::new(47);
        let reservation = lease_registration(lease_id, allocation, 32, epoch);
        let staged = LeaseRegistry::new();
        staged
            .import(
                StagedLease::new(reservation, vec![0x3c; 32])
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: epoch,
                size: 32,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the fixture lease covers the texture");
        let leases = attachment_lease_context(&staged, &borrowed, &resources, epoch, 4096);
        let stages = reviewed_sampled_stages();
        let mut pass = sampled_pass(4);
        pass.textures = vec![leased_sampled_texture_view(
            TextureSource::StagedLease(lease_id),
            allocation,
        )];
        let refused = match prepare_render_request(
            &stages,
            &pass,
            &[None],
            Some(&leases),
            0,
            0,
            SpirvFeaturePolicy::PHASE1,
        ) {
            Err(error) => error,
            Ok(_) => panic!("a 4x4 texture reads sixty-four bytes, not thirty-two"),
        };
        eprintln!("short reservation refused: {refused:?}");
        assert_eq!(refused.slug, "lease_range_out_of_bounds");
        assert_eq!(refused.class, ProviderErrorClass::Resource);
        assert_eq!(refused.fields.get("lease"), Some(&FieldValue::Unsigned(37)));
    }
}
