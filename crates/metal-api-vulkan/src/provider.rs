//! Vulkan-to-neutral provider contract mapping.
//!
//! This module translates reflection and selected-device limits into the pure
//! values owned by `metal-api-core::provider`. It deliberately does not expose
//! Vulkan descriptors, region plans, handles, or guest memory pointers.

use ash::vk;
use metal2vulkan::reflect::{
    BufferFootprint, KernelDispatch, ResourceAccess, ResourceBinding, ResourceKind,
    ShaderReflection,
};
use metal_api_core::provider::{
    AffineAccess, AffineTerm, AliasMode, AttachmentFormat, BufferAccess, BufferBindingContract,
    DepthResolveFilter, DispatchKind, FootprintProof, IndirectCommandKind, PipelineContract,
    PresentMode, ProviderCapabilities, SemanticDigest, StencilResolveFilter, StorageMode,
    TextureBindingContract, TextureFormat, MAX_COLOR_ATTACHMENTS, MAX_COMPUTE_TEXTURES,
    MAX_PRESENT_IMAGE_COUNT, MAX_PRESENT_TARGETS, MAX_RENDER_STAGE_BUFFERS, MAX_RENDER_TEXTURES,
    MAX_RENDER_TEXTURE_DIMENSION_1D, MAX_RENDER_TEXTURE_DIMENSION_3D,
};
use metal_api_core::ExecutorError;

/// The largest attachment extent this rail's review covers, per axis.
///
/// The milestone's window was 4×4 (`research/docs/23` §1.3): every texel had to
/// be distinguishable from a single stored one, and the capability value kept a
/// larger attachment out of the rail instead of letting the driver answer a
/// size no fixture proved. R1b (`research/docs/23` §70) widened it to the first
/// family a real frame needs (16×16 and the 64×64 boundary); R5a
/// (`research/docs/23` §73) widened it once more to the desktop sizes the guest
/// profile measured — the 2048×2048 boundary, inside every conformant device's
/// own window (Vulkan's minimum `maxFramebuffer{Width,Height}` is 4096) — so
/// the declared window is this ceiling clamped by the selected device's own
/// framebuffer limits ([`attachment_dimension_window`]). Widening the ceiling
/// further is a deliberate change that owes a boundary fixture at the new
/// value, in all three review surfaces.
pub(crate) const REVIEWED_ATTACHMENT_CEILING: [u64; 2] = [2048, 2048];

/// The attachment window a device with these limits declares.
///
/// R1b (`research/docs/23` §70): per axis, the smaller of the reviewed ceiling
/// above and the device's own `maxFramebuffer{Width,Height}`. The capability
/// snapshot publishes this value and core admission refuses a wider attachment
/// by name (`attachment_dimension_limit`, carrying the maximum it crossed), so
/// the declared window is a device fact rather than a fixture-shaped constant.
/// The rail's own execution path asks the same two halves in the other order —
/// the device's answer first (`attachment_extent_device_limit`), the ceiling
/// second — so a directly-constructed request cannot jump either gate
/// (`render.rs`, `refuse_attachment_extent`).
pub(crate) fn attachment_dimension_window(limits: &vk::PhysicalDeviceLimits) -> [u64; 2] {
    [
        u64::from(limits.max_framebuffer_width).min(REVIEWED_ATTACHMENT_CEILING[0]),
        u64::from(limits.max_framebuffer_height).min(REVIEWED_ATTACHMENT_CEILING[1]),
    ]
}

/// The one-dimensional sampled window a device with these limits declares
/// (2026-09-19, census b10's `texture_shape` bucket).
///
/// The contract's review ceiling
/// ([`MAX_RENDER_TEXTURE_DIMENSION_1D`]) states how wide the widest *reviewed*
/// LUT is, and the device's own `maxImageDimension1D` states how wide a
/// one-dimensional image this device can hold — a limit of its own rather than
/// a reading of `maxImageDimension2D`. The snapshot publishes the smaller of
/// the two, so a device that cannot hold the census's `16384x1` colour-transfer
/// LUT declares a narrower window instead of a width every `vkCreateImage` of
/// the rail's own one-dimensional arm would refuse.
///
/// The rail's execution path asks the same two halves in the other order — the
/// device's answer first, the ceiling second — exactly as the attachment
/// window beside it does, so a directly-constructed request cannot jump either
/// gate (`render.rs`, the sampled view gate's one-dimensional arm).
pub(crate) fn render_texture_dimension_1d(limits: &vk::PhysicalDeviceLimits) -> u64 {
    u64::from(limits.max_image_dimension1_d).min(MAX_RENDER_TEXTURE_DIMENSION_1D)
}

/// The three-dimensional sampled window a device with these limits declares
/// (2026-09-20, the `D3` sampled texture arm).
///
/// [`render_texture_dimension_1d`]'s sibling two axes over, with the one
/// difference the device's own limit states: `maxImageDimension3D` bounds a
/// `TYPE_3D` image's width, height **and** depth, so the contract's review
/// ceiling ([`MAX_RENDER_TEXTURE_DIMENSION_3D`]) caps every one of the volume's
/// three extents rather than one row's texel count. The snapshot publishes the
/// smaller of the two, so a device whose volume window is narrower than the
/// reviewed ceiling declares its own number instead of one every
/// `vkCreateImage` of the rail's three-dimensional arm would refuse.
///
/// The rail's own view gate states the *review* half of the same rule
/// (`render.rs`, the sampled view gate's volume arm): a directly-constructed
/// request that never passed core admission is still held to
/// [`MAX_RENDER_TEXTURE_DIMENSION_3D`] per axis, and the device's own answer
/// arrives through this window for every request that did.
pub(crate) fn render_texture_dimension_3d(limits: &vk::PhysicalDeviceLimits) -> u64 {
    u64::from(limits.max_image_dimension3_d).min(MAX_RENDER_TEXTURE_DIMENSION_3D)
}

/// The stage-buffer window one device states (`research/docs/23` §3.3, §117
/// E-SB2).
///
/// The contract's ceiling ([`MAX_RENDER_STAGE_BUFFERS`]) is a *review* bound —
/// it is the set-level floor Vulkan states for a two-stage pipeline's storage
/// buffers — while the number a stage can actually carry is the device's own:
/// the spec's per-stage floor for storage buffers is four, and a device is free
/// to report exactly that. So the rail declares the smaller of the two per
/// stage, the pair's sum as the list bound, and the device's per-set window
/// beside them for the arrangement check [`crate::render`] runs before it
/// builds a descriptor set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StageBufferWindow {
    /// Stage buffer declarations one *stage* may carry.
    pub(crate) per_stage: u32,
    /// Stage buffer declarations one *pass* may carry across both stages.
    pub(crate) list: u32,
    /// Storage-buffer descriptors one descriptor *set* may hold. The reviewed
    /// arrangement gives each stage its own set, so a stage's list only has to
    /// fit this window; a translated pair whose two stages share one set has to
    /// fit both lists in it, which is what [`crate::render`] checks per set.
    pub(crate) per_set: u32,
}

/// The window `research/docs/23` §117 states from one device's limits.
///
/// `per_stage` is `min(MAX_RENDER_STAGE_BUFFERS, maxPerStageDescriptorStorageBuffers)`;
/// the list bound is two stages' worth of it, because a render pass has exactly
/// two stages. Both numbers are read from the selected device rather than
/// restated, so a narrow device declares a narrow window instead of one it
/// would have to refuse by name at pipeline-layout time.
pub(crate) fn stage_buffer_window(limits: &vk::PhysicalDeviceLimits) -> StageBufferWindow {
    let per_stage =
        (MAX_RENDER_STAGE_BUFFERS as u32).min(limits.max_per_stage_descriptor_storage_buffers);
    StageBufferWindow {
        per_stage,
        list: per_stage.saturating_mul(2),
        per_set: limits.max_descriptor_set_storage_buffers,
    }
}

/// The largest instance count the instancing increment executes
/// (`research/docs/23` §3.3, v31).
///
/// The reviewed fixture draws two instances; the ceiling is four so the rail
/// has headroom without claiming a window no case proves. A wider draw is
/// refused by core admission against this bit rather than silently narrowed.
const MAX_RENDER_INSTANCES: u32 = 4;

/// Conservative first-increment heap ceiling (`research/docs/25-heaps与ICB设计.md`
/// §4.1). The placement rail is proven on a 4096-byte heap; capping admission
/// far below the device's real single-allocation ceiling (Lavapipe reports a
/// ~23 GiB heap) keeps the snapshot fail-closed, exactly like the 2×2
/// attachment window above.
const MAX_HEAP_BYTES: u64 = 64 * 1024 * 1024;

fn failure(message: impl Into<String>) -> ExecutorError {
    ExecutorError::new(message)
}

pub(crate) fn capabilities_from_limits(limits: &vk::PhysicalDeviceLimits) -> ProviderCapabilities {
    let multisample_ceiling = crate::render::limits_render_sample_count_ceiling(limits);
    ProviderCapabilities {
        max_passes: 1,
        supports_threads_exact: true,
        supports_threadgroups: false,
        supports_serial: true,
        supports_concurrent: false,
        max_local_size: limits.max_compute_work_group_size.map(u64::from),
        max_invocations: u64::from(limits.max_compute_work_group_invocations),
        max_group_count: limits.max_compute_work_group_count.map(u64::from),
        max_storage_buffer_descriptors: limits
            .max_per_stage_descriptor_storage_buffers
            .min(limits.max_descriptor_set_storage_buffers)
            .min(limits.max_per_stage_resources),
        // Compute-side texture sampling is executed: `create_textures`
        // uploads a D2 single-sample `R32Uint` image through the driver's own
        // `VkSubresourceLayout.rowPitch` and binds it as a combined image
        // sampler. Evidence: the v11 case reads texel (0, 0) and the v12 cases
        // read every cell of a 4×4 texture in two dispatch shapes, on Lavapipe
        // and on the RTX 5060 (`research/docs/16` §4.5, §4.6). The bit names
        // exactly that shape: one binding, one format. From C1b on a
        // `R32Float` texture that pairs with the module's own AIR constexpr
        // sampler is executed too (`create_static_samplers` creates the
        // `VkSampler` from the module's decoded state and binds it at the
        // reflected descriptor binding); the registration gate refuses a float
        // texture with no static sampler, so the wider format list stays
        // paired with that window (`research/docs/26` §21.3).
        supports_compute_texture_sampling: true,
        max_compute_textures: MAX_COMPUTE_TEXTURES as u32,
        supported_compute_texture_formats: vec![TextureFormat::R32Uint, TextureFormat::R32Float],
        max_buffer_range: u64::from(limits.max_storage_buffer_range),
        max_push_constant_bytes: limits.max_push_constants_size,
        alias_mode: AliasMode::Refused,
        storage_modes: vec![StorageMode::OwnedBytes],
        host_readback: true,
        submit_only: false,
        // The offscreen render rail is admitted: `render.rs` executes one to
        // four colour attachments end to end, so the capability bits name
        // exactly what that rail covers — up to four attachments of the
        // declared window below, with a reviewed output module per attachment
        // count and per component shape: the single-output four-component
        // module serves the two 8-bit UNORM layouts and `Rgba16Float` alike
        // (`research/docs/23` §78), the dual, triple and quad modules stay
        // reviewed for their 8-bit format lists, the single-channel float module
        // is the one format-specific stage, and the depth-only stage stands
        // beside an empty colour list.
        // Formats the device itself refuses are still refused before
        // `vkCreateImage` by the rail's `COLOR_ATTACHMENT` probe; the
        // capability snapshot answers which shapes the provider can express,
        // and the device answers which of those it can run.
        supports_render_passes: true,
        // The reviewed modules cover one to four locations, so the snapshot
        // reports the contract's own ceiling; the rail still typed-refuses a
        // pass beyond it (`render_mrt_attachment_count_unsupported`) instead of
        // silently rendering a subset of the locations.
        max_color_attachments: MAX_COLOR_ATTACHMENTS as u32,
        max_attachment_dimension: attachment_dimension_window(limits),
        supported_color_formats: AttachmentFormat::ADMITTED.to_vec(),
        // Vertex input is executed (`render.rs` uploads each bound pool view,
        // builds the pipeline's vertex input state from the contract layout and
        // issues `vkCmdBindVertexBuffers` + `vkCmdDrawIndexed`). Evidence:
        // `tests/render_e2e.rs` reads the 2x2 attachment back as `40 80 c0 ff`
        // four times from the reviewed `quad_indexed` module on Lavapipe, and
        // the same suite rail on the RTX 5060. The bits name the rail's own
        // limits — the contract's binding cap and both closed format families —
        // so a wider request is still refused by core admission
        // (`research/docs/23` §3.3).
        //
        // The normalized storages arrived with E-VF1 (`research/docs/23` §103):
        // they are Vulkan's own *required* vertex input formats, so this rail
        // declares all eight of the contract's values, and
        // `tests/render_normalized_vertex_e2e.rs` measures the fetch on
        // Lavapipe — one draw whose four normalized attributes land the stored
        // integers' own quotients, and the swapped-bytes arm that changes the
        // frame. The rail that does not declare them yet is the native one
        // (`metal-api-native/src/render.rs::DECLARED_VERTEX_FORMATS`), where a
        // declaration waits for an Apple-side reading.
        max_vertex_buffers: metal_api_core::provider::MAX_VERTEX_BUFFERS as u32,
        supported_vertex_formats: metal_api_core::provider::VertexFormat::ADMITTED.to_vec(),
        supported_index_formats: metal_api_core::provider::IndexFormat::ADMITTED.to_vec(),
        // The superset vertex interface is executed (`research/docs/23` §3.3,
        // E-TX11): `render.rs` builds the pipeline's vertex input state from the
        // *contract's* own layout — one `VkVertexInputAttributeDescription` per
        // declared attribute across the streams `resolve_vertex_streams`
        // pairs with the pass's bindings — so a layout that declares more
        // locations than the module reads binds those extra streams and the
        // module simply consumes the locations it declares. The registration
        // gate is what this bit's declaration follows: it holds every reflected
        // location to a declared attribute with the same component shape, while
        // extra declared attributes are bound and ignored
        // (`tests/render_vertex_superset_e2e.rs`, whose four-attribute stream
        // lands the two-attribute frame byte for byte). The bit names exactly
        // that direction: a location the module reads with no declared
        // attribute covering it stays refused by name, because the driver would
        // leave that input undefined.
        supports_render_vertex_interface_superset: true,
        // The layout-free count above the milestone's three vertices is
        // executed (2026-09-19, census v45's `vertex_span` bucket): the rail
        // issues the pass's own count (`render.rs`'s draw record), and the
        // reviewed `vertex_id` module is a total function of the index — its
        // `isTop` select puts every index above the first two on the triangle's
        // third corner, so the extra vertices resolve to degenerate triangles
        // rather than to a position the module does not carry.
        // `tests/render_vertex_count_e2e.rs` reads both halves back: the
        // reviewed module's four- and six-vertex draws land the three-vertex
        // frame byte for byte, and a translated quad module's six-vertex draw
        // covers the texels its three-vertex draw cannot.
        supports_render_vertex_count_above_triangle: true,
        // Instancing is executed (`render.rs` builds each binding's input rate
        // from the layout's step and issues `vkCmdDraw*` with the pass's own
        // instance count). Evidence: the reviewed `instanced_pair_4x4` case on
        // Lavapipe and on the RTX 5060. The ceiling is the reviewed fixture's
        // two instances rounded up to four; a wider draw is still refused by
        // core admission (`research/docs/23` §3.3, v31).
        supports_render_instancing: true,
        max_render_instances: MAX_RENDER_INSTANCES,
        // The multisample raster is executed (`render.rs` builds a multisampled
        // subpass with one resolve attachment per colour location and copies
        // the resolve target back). Evidence: the reviewed `msaa_edge_4x4`
        // case on Lavapipe and on the RTX 5060, and the v61 2x/8x
        // full-coverage fixtures on both. Both bits are the device's own
        // framebuffer sample counts — the ceiling is the largest of the
        // reviewed 2/4/8 rasters the device admits — so a device without any
        // of them reports 0 and core admission refuses the shape before the
        // rail's per-format probe runs (`research/docs/23` §3.3, v51/v61).
        supports_render_multisample: multisample_ceiling.is_some(),
        max_render_sample_count: multisample_ceiling.unwrap_or(0),
        // The depth resolve is executed from v57 on (`render.rs` migrates the
        // render pass to RenderPass2 and resolves a stored multisampled depth
        // surface into its own single-sample landing). The two bits are the
        // device's own answer, not this function's: they come from the
        // `VkPhysicalDeviceDepthStencilResolveProperties` the context probes,
        // which a `PhysicalDeviceLimits` snapshot does not carry, so
        // [`VulkanProvider::provider_capabilities`] overlays them on this
        // struct's defaults. A device that reports no admitted filter keeps
        // both bits at the fail-closed "cannot resolve" defaults
        // (`research/docs/23` §3.3, v57).
        supports_render_depth_resolve: false,
        depth_resolve_modes: 0,
        // The stencil resolve is executed from v60 on, with the same
        // device-owned answer the depth resolve states: the two bits come from
        // the `VkPhysicalDeviceDepthStencilResolveProperties` the context
        // probes, which a `PhysicalDeviceLimits` snapshot does not carry, so
        // [`VulkanProvider::provider_capabilities`] overlays them on this
        // struct's defaults. A device that reports no admitted filter keeps
        // both bits at the fail-closed "cannot resolve" defaults
        // (`research/docs/23` §3.3, v60).
        supports_render_stencil_resolve: false,
        stencil_resolve_modes: 0,
        // The render sampler is executed (`research/docs/23` §3.3, v70):
        // `render.rs` uploads each bound texture into a host-visible linear
        // image, builds the sampled pipeline's descriptor set layout from the
        // bindings and samples them at texel centres through a
        // provider-synthesised nearest/clamp sampler. Evidence: the reviewed
        // `sampled_texel_4x4` case on Lavapipe and on the RTX 5060 (the
        // attachment reads back the uploaded texels exactly), and the rail's
        // own `render_e2e` sampling test. The bits name the reviewed window —
        // the `TextureFormat::RENDER_SAMPLED` lanes themselves
        // (`research/docs/23` §107/§113: the census's BGRA8 guest views are the
        // same texel in the other byte order, the narrow lanes fill the
        // channels their format lacks, and the eight-byte half-float lane is
        // the format the census's remaining binds name, so which byte holds
        // which channel — or how many bytes a texel is — stays the `VkFormat`'s
        // own fact, not the module's) and one texture of the render area's own
        // extent — so a wider request is refused by core admission or by the
        // rail's shape gates rather than silently narrowed. The binding count
        // is the contract's own ceiling (`research/docs/23` §3.3, v102): a
        // *translated* fragment stage samples as many textures as its
        // reflection names, wherever its own `[[texture(n)]]` arguments sit
        // (`v104`), so the rail declares the list's cap, while the reviewed
        // pair's one-texture window — index zero included — is refused at
        // execution by name (`render_texture_stage_unsupported`).
        supports_render_texture_sampling: true,
        max_render_textures: MAX_RENDER_TEXTURES as u32,
        supported_render_texture_formats: TextureFormat::RENDER_SAMPLED.to_vec(),
        // The one-dimensional sampled window is executed by the same rail
        // (2026-09-19, census b10's `texture_shape` bucket): `render.rs` uploads
        // a single-row `vk::ImageType::TYPE_1D` image in the lane's own format
        // and samples it through a `TYPE_1D`/`TYPE_1D_ARRAY` view.
        // `tests/render_texture_1d_lut_e2e.rs` reads back the LUT's own texels
        // through both rails on Lavapipe, and the census's own boot reads them
        // on the RTX 5060. The window is the device's own
        // `maxImageDimension1D` clamped by the contract's review ceiling, and a
        // device whose answer is below the reviewed LUT's 16384 texels declares
        // the narrower width rather than one it would refuse at
        // `vkCreateImage`.
        max_render_texture_dimension_1d: render_texture_dimension_1d(limits),
        // The three-dimensional sampled window is executed by the same rail
        // (2026-09-20, the `D3` sampled texture arm): `render.rs` creates a
        // `vk::ImageType::TYPE_3D` volume in the view's own format, uploads it
        // slice by slice through the driver's own `depthPitch`, and samples it
        // through a `TYPE_3D` view. `tests/render_texture_3d_volume_e2e.rs`
        // reads the volume's own texels back through both rails on Lavapipe,
        // and the census's own boot reads them on the RTX 5060. The window is
        // the device's own `maxImageDimension3D` clamped by the contract's
        // review ceiling, and a device whose answer is below the reviewed
        // volume's extents declares the narrower number rather than one it
        // would refuse at `vkCreateImage`.
        max_render_texture_dimension_3d: render_texture_dimension_3d(limits),
        // The gathered extent is executed for the arm whose source has host
        // bytes (`research/docs/23` §3.3, §111, E-TX5/E-TX10): a reviewed
        // module's sample coordinate is the fragment's own centre, so
        // `render.rs` gathers the source into the render area's integer grid
        // (`gather_render_texture`), while a *translated* fragment stage states
        // its own absolute coordinates and the rail binds the source at its own
        // extent. Both were measured before this bit existed: the three
        // reviewed shapes (`6x4→4x4`, `2x8→4x4`, `32x32→40x32`) and the
        // translated pair in `tests/render_texture_extent_e2e.rs`, whose frame
        // is the fixture's own definition (`50 10 00 ff` per fragment) on
        // Lavapipe and on the RTX 5060. The bit names exactly that arm: the
        // owner's zero-copy window has no host bytes to gather, so a source of
        // another extent in it keeps the rail's own refusal
        // (`render_texture_extent_unsupported`), and the declaration is not
        // read as "any source of any extent executes".
        supports_render_texture_gathered_extent: true,
        // The gathered extent's other arm is executed too (`research/docs/23`
        // §111, E-TX12): the owner's no-copy window of another extent stays at
        // its own mapping — no host copy exists for it — and the pair's
        // *gathered* fragment sibling reads the destination grid's texel with
        // `OpImageFetch` at an index computed on the device, while a translated
        // fragment stage keeps stating its own coordinates over the source's own
        // extent. Both arms land the same frame the host-bytes gather lands, per
        // byte, in `tests/render_texture_extent_nocopy_e2e.rs`; a registration
        // that declares the *sampling* sibling instead still gets the rail's own
        // refusal by name, which is why this bit is its own field rather than a
        // second reading of the bit above.
        supports_render_texture_gathered_extent_no_copy: true,
        // The colour attachment's landing-view arm (`research/docs/23` §115
        // 之后的增量，E-TX13) is executed: `compute_provider.rs` resolves the
        // second declaration the store arm carries, `render.rs` writes the
        // pass's frame into the owner window it names — the same retain/retire
        // landing E-TX8's borrowed store uses — and the frame still travels the
        // writeback channel, so every pre-E-TX13 reader is byte-identical.
        // Evidence: `tests/render_attachment_landing_view_e2e.rs`. The bit is
        // its own field rather than a second reading of either gathered-extent
        // bit: those describe a *sampled source's* extent, this one describes
        // where an attachment's stored frame lands.
        supports_render_attachment_landing_view: true,
        // A landing-only entry is executed (`research/docs/23` §115 之后的增量，
        // E-TX14/R4b): `compute_provider.rs` resolves the kept identity out of
        // the resident registry, `render.rs` copies the provider image back and
        // writes the owner's window, and the identity is consumed on success.
        // Evidence: `tests/render_kept_frame_landing_e2e.rs`. The bit is its own
        // field rather than the landing-view bit's second reading: this one is
        // about a frame a *previous* submission kept, that one is about where a
        // pass's own frame lands.
        supports_render_kept_frame_landing: true,
        // The pass-entry snapshot arm (`research/docs/23` §118, E-TX15): this
        // rail copies the attachment's own pass-entry content into a
        // same-format device-local image before the render pass opens and
        // binds that image as the sampled view
        // (`tests/render_pass_entry_snapshot_e2e.rs`).
        supports_render_pass_entry_snapshot: true,
        // Stage buffer bindings are executed (`research/docs/23` §3.3, v83):
        // `render.rs` uploads each bound view into a host-visible
        // `STORAGE_BUFFER` and binds the two stages' descriptor sets — set 1
        // for the vertex stage's bindings, set 2 for the fragment stage's —
        // before the draw. Evidence: the reviewed `stage_buffer_quad_2x2`
        // fixture in `tests/render_e2e.rs` (the buffers' own bytes move the
        // geometry and land in the attachment) and the rail's refusal tests
        // for a pass whose stage binds a buffer the module does not read. The
        // bit names the shape this increment reviewed — read-only, static
        // footprint, indices below the contract bound — so a wider request is
        // refused by core admission rather than silently narrowed.
        supports_render_stage_buffers: true,
        // The window is the device's own answer, clamped by the review ceiling
        // (`research/docs/23` §117, E-SB2): a stage carries at most
        // `min(MAX_RENDER_STAGE_BUFFERS, maxPerStageDescriptorStorageBuffers)`
        // declarations, a pass at most twice that, and the set each stage's
        // bindings land in at most the device's
        // `maxDescriptorSetStorageBuffers` (`stage_buffer_window`).
        max_render_stage_buffers: stage_buffer_window(limits).list,
        max_render_stage_buffers_per_stage: stage_buffer_window(limits).per_stage,
        // The folded shape is executed (`research/docs/23` §3.3, E-TX9): the
        // rail publishes one canonical arrangement for it
        // ([`metal_api_vulkan::stage_buffer_namespace_layout`], the vertex
        // stage's `[[buffer(n)]]` arguments in set 1 and the fragment stage's
        // in the translator's own set 0) and reads each module's slot back out
        // of its reflection, so a pair whose two stages read the same Metal
        // index executes with each stage's own bytes. The bit names exactly
        // that shape: a pair still folded into one slot under the translator's
        // default layout is refused by name
        // (`render_stage_buffer_layout_unsupported`), and the declaration says
        // "this rail can execute the separated arrangement", not "submit the
        // folded one". Evidence: the translated stage-buffer pair in
        // `tests/render_stage_buffer_namespace_e2e.rs`, its object-API frame,
        // and the conformance case the capture archives keep.
        supports_render_stage_buffer_namespace_split: true,
        // The texel space is executed (2026-09-19, census v43's
        // `texture_state` axis): a pass that binds a runtime `[[sampler(n)]]`
        // whose state says `normalizedCoordinates = NO` is executed through the
        // fragment module's *derived* explicit-LOD sibling
        // (`render.rs::pixel_coordinate_sampler_variant`), which is what makes
        // the unnormalized sampler a legal Vulkan use
        // (`VUID-vkCmdDraw-None-08610`/`-08611`) while every sample keeps the
        // texel the guest's own coordinates name. Evidence: the registered
        // runtime-sampler fixture's pixel-coordinate arm in
        // `tests/render_pixel_coordinate_sampler_e2e.rs`, which reads back the
        // same texels the translator's own pixel-space lowering computes. The
        // bit names exactly that window — one filter family and two address
        // modes, the ones an unnormalized `VkSampler` may state — so a wider
        // state is refused by the rail by name rather than narrowed.
        supports_render_pixel_coordinate_sampler: true,
        // Presentation is declared: `render.rs` executes the "readable
        // swapchain equivalent" end to end (`research/docs/24` §6 Step 3) — one
        // target, one `Fifo` present, single buffering. Evidence:
        // `tests/render_e2e.rs` (2×2 target lands `40 80 c0 ff`×4 and counts
        // acquire/present 1/1 on Lavapipe; see the run log). The bits name
        // exactly that window, so a wider present request is still refused by
        // core admission rather than silently narrowed.
        supports_presentation: true,
        max_present_targets: MAX_PRESENT_TARGETS as u32,
        supported_present_modes: PresentMode::ADMITTED.to_vec(),
        max_present_image_count: MAX_PRESENT_IMAGE_COUNT,
        // Heap placement is executed (`compute_provider.rs` builds one
        // `VkDeviceMemory` slab and binds each owned allocation's buffer at
        // its placement offset), and the placement observation channel is
        // exercised by `provider-smoke`'s `provider_heap_placement` case. The
        // first increment binds buffers only and refuses aliasing and texture
        // placements (`research/docs/25` §6 Step 3).
        supports_heaps: true,
        max_heap_bytes: MAX_HEAP_BYTES,
        supported_heap_storage_modes: vec![StorageMode::OwnedBytes],
        supports_heap_aliasing: false,
        // Indirect replay is executed for the reviewed draw, indexed draw and
        // dispatch shapes: the render rail encodes a `VkDrawIndirectCommand` or
        // a `VkDrawIndexedIndirectCommand` (the latter with its own `[0, 1, 2]`
        // `UINT32` index buffer) into a host-visible `INDIRECT_BUFFER` and
        // replays it with `vkCmdDrawIndirect` / `vkCmdDrawIndexedIndirect`
        // (`render.rs::execute_indirect_render_pass`), and the compute rail
        // encodes a `VkDispatchIndirectCommand` and replays it with
        // `vkCmdDispatchIndirect` (`lib.rs::ExecutionResources::record`,
        // `research/docs/25` §6 Step 4). Evidence: `tests/render_e2e.rs`
        // replays the milestone's full-screen triangle indirectly (both
        // non-indexed and indexed) and reads the same `40 80 c0 ff` texels
        // back on Lavapipe; `tests/indirect_dispatch_e2e.rs` replays one
        // compute dispatch and reads the same output bytes as a direct
        // dispatch. The first increment is one command.
        supports_indirect_command_buffers: true,
        max_indirect_commands: 1,
        supported_indirect_commands: vec![
            IndirectCommandKind::Draw,
            IndirectCommandKind::DrawIndexed,
            IndirectCommandKind::Dispatch,
        ],
    }
}

/// The contract's depth-resolve filter mask for a device's reported resolve
/// modes (`research/docs/23` §3.3, v57).
///
/// The mask maps the admitted filters onto their wire bit positions —
/// [`DepthResolveFilter::Sample0`]/[`DepthResolveFilter::Min`]/
/// [`DepthResolveFilter::Max`] are bits 0/1/2 — and drops every mode the
/// contract does not carry: `AVERAGE` is a real Vulkan resolve mode that has no
/// depth filter in the closed family, so folding it onto a neighbour would
/// admit a filter the caller did not ask for. A device that reports none of
/// the three yields `0`, the fail-closed "cannot resolve" mask.
pub(crate) fn depth_resolve_mode_mask(modes: vk::ResolveModeFlags) -> u32 {
    let mut mask = 0;
    if modes.contains(vk::ResolveModeFlags::SAMPLE_ZERO) {
        mask |= 1u32 << u32::from(DepthResolveFilter::Sample0.code());
    }
    if modes.contains(vk::ResolveModeFlags::MIN) {
        mask |= 1u32 << u32::from(DepthResolveFilter::Min.code());
    }
    if modes.contains(vk::ResolveModeFlags::MAX) {
        mask |= 1u32 << u32::from(DepthResolveFilter::Max.code());
    }
    mask
}

/// The contract's stencil-resolve filter mask for a device's reported stencil
/// resolve modes (`research/docs/23` §3.3, v60).
///
/// The mask maps the admitted filters onto their wire bit positions —
/// [`StencilResolveFilter::Sample0`] is bit 0 — and drops every mode the
/// contract does not carry. Vulkan's stencil resolve modes are
/// `SAMPLE_ZERO`/`MIN`/`MAX`, where `MIN`/`MAX` reduce the *stencil* values
/// themselves; Metal's [`StencilResolveFilter::DepthResolvedSample`] takes the
/// stencil of whichever sample the *depth* resolve selected, which no Vulkan
/// mode expresses. The two therefore have no mapping and are dropped rather
/// than folded onto a filter the caller did not ask for, so a device's mask
/// carries at most the Sample0 bit.
pub(crate) fn stencil_resolve_mode_mask(modes: vk::ResolveModeFlags) -> u32 {
    if modes.contains(vk::ResolveModeFlags::SAMPLE_ZERO) {
        1u32 << u32::from(StencilResolveFilter::Sample0.code())
    } else {
        0
    }
}

pub(crate) fn pipeline_contract(
    reflection: &ShaderReflection,
    translator_revision: Option<SemanticDigest>,
) -> Result<PipelineContract, ExecutorError> {
    let dispatch_kind = match reflection.kernel_dispatch {
        Some(KernelDispatch::ThreadsDynamic { .. } | KernelDispatch::ThreadsFixed { .. }) => {
            DispatchKind::ThreadsExact
        }
        Some(KernelDispatch::Workgroups) => DispatchKind::Threadgroups,
        None => return Err(failure("provider contract has no kernel dispatch kind")),
    };
    let reflected_local_size = reflection
        .local_size
        .ok_or_else(|| failure("provider contract has no reflected local size"))?
        .map(u64::from);
    let (required_local_size, fixed_grid) = match reflection.kernel_dispatch {
        Some(KernelDispatch::ThreadsFixed { threads_per_grid }) => (
            Some(reflected_local_size),
            Some(threads_per_grid.map(u64::from)),
        ),
        Some(KernelDispatch::Workgroups) => (Some(reflected_local_size), None),
        Some(KernelDispatch::ThreadsDynamic { .. }) => (None, None),
        None => return Err(failure("provider contract has no kernel dispatch kind")),
    };
    let (push_constant_offset, push_constant_bytes) = reflection
        .kernel_dispatch
        .and_then(KernelDispatch::push_constant_range)
        .map_or((0, 0), |range| (range.offset, range.size));

    // AIR-embedded constexpr samplers (`research/docs/26` §21.3, C1b). The
    // reflection names each one's descriptor location and its decoded state;
    // the contract carries that state on the sampled texture the sampler is
    // paired with, so a caller cannot declare a filtering the module was not
    // lowered against. This increment reviews exactly one sampler bound by
    // exactly one sampled texture, and a runtime `[[sampler(n)]]` has no
    // contract surface at all.
    let mut static_samplers = Vec::new();
    for binding in &reflection.bindings {
        if binding.kind == ResourceKind::Sampler {
            return Err(failure(format!(
                "runtime [[sampler({})]] is not part of the reviewed compute contract",
                binding.metal_index
            )));
        }
        if binding.kind != ResourceKind::StaticSampler {
            continue;
        }
        let state = binding.static_sampler.ok_or_else(|| {
            failure(format!(
                "AIR static sampler at descriptor {:?} carries no decoded state",
                binding.descriptor.map(|descriptor| descriptor.binding)
            ))
        })?;
        static_samplers.push(
            crate::static_sampler_policy(&state)
                .map_err(|error| failure(format!("static sampler state: {error}")))?,
        );
    }
    if static_samplers.len() > 1 {
        return Err(failure(format!(
            "a module with {} AIR static samplers is outside the reviewed compute contract (exactly one is reviewed)",
            static_samplers.len()
        )));
    }
    let mut buffer_bindings = Vec::with_capacity(reflection.bindings.len());
    let mut texture_bindings = Vec::new();
    let mut sampled_texture_count = 0_usize;
    for binding in &reflection.bindings {
        // Sampled textures get their own list (`research/docs/26` §21.3, step
        // 1). Before this list existed the class judge had to trust the
        // request's own texture bindings, because the contract said nothing
        // about them; now the module's reflection is the declaration the trace
        // is paired against, exactly as it already is for buffers. The
        // provider's reflection validation has admitted the binding's kind and
        // access (`research/docs/16` §4.7).
        if binding.kind == ResourceKind::Texture {
            sampled_texture_count += 1;
            texture_bindings.push(map_texture_binding(
                binding,
                static_samplers.first().copied(),
            )?);
            continue;
        }
        // A write-capable storage image joins the same texture list with the
        // `Storage` access (`research/docs/26` §21.4, C2): the pair rules
        // compare access, type, format and reach exactly as they do for a
        // sampled binding, and the execution rail binds it as a Vulkan
        // `STORAGE_IMAGE` instead of a combined image sampler.
        if binding.kind == ResourceKind::StorageImage {
            texture_bindings.push(map_storage_image_binding(binding)?);
            continue;
        }
        // The sampler itself has no separate contract entry: its state is the
        // declaration the paired texture carries.
        if binding.kind == ResourceKind::StaticSampler {
            continue;
        }
        if binding.kind != ResourceKind::Buffer {
            return Err(failure(format!(
                "provider contract only maps Metal buffers, sampled textures, storage images and AIR static samplers, found {:?} at {}",
                binding.kind, binding.metal_index
            )));
        }
        let access = map_access(binding.access, binding.metal_index)?;
        let footprint = binding
            .footprint
            .as_ref()
            .ok_or_else(|| failure(format!("buffer {} has no footprint", binding.metal_index)))?;
        buffer_bindings.push(BufferBindingContract {
            metal_binding: binding.metal_index,
            access,
            footprint: map_footprint(footprint, binding.metal_index)?,
        });
    }
    if !static_samplers.is_empty() && sampled_texture_count != 1 {
        return Err(failure(format!(
            "a module with one AIR static sampler must declare exactly one sampled texture; reflected {}",
            sampled_texture_count
        )));
    }

    buffer_bindings.sort_by_key(|binding| binding.metal_binding);
    texture_bindings.sort_by_key(|binding| binding.metal_binding);
    let contract = PipelineContract {
        dispatch_kind,
        required_local_size,
        fixed_grid,
        push_constant_offset,
        push_constant_bytes,
        buffer_bindings,
        texture_bindings,
        // Capability names are provider admission metadata. A normalized
        // cross-provider vocabulary is intentionally still an open decision.
        shader_capabilities: Vec::new(),
        translator_revision,
    };
    contract
        .validate()
        .map_err(|error| failure(format!("provider pipeline contract: {error}")))?;
    Ok(contract)
}

/// Map one reflected sampled texture onto the contract's texture face.
///
/// The mapping is deliberately closed (`research/docs/26` §21.3): this rail
/// executes D2, single-sample, non-arrayed textures whose AIR component is
/// `uint` (texel reads through the translator's synthesized read sampler, C1)
/// or `float` (samples through an AIR-embedded constexpr sampler, C1b), so
/// anything else is refused here — before a contract exists — rather than
/// registered as a shape the execution path would refuse later with a message
/// no class judge can pair against. The sampler state is the state the module
/// itself carries, named in the contract so the execution path creates exactly
/// that state instead of a provider default.
fn map_texture_binding(
    binding: &ResourceBinding,
    static_sampler: Option<metal_api_core::provider::SamplerPolicy>,
) -> Result<TextureBindingContract, ExecutorError> {
    use metal2vulkan::meta::{TextureComponent, TextureDimension};

    let shape = binding.texture_shape.as_ref().ok_or_else(|| {
        failure(format!(
            "texture {} has no reflected shape",
            binding.metal_index
        ))
    })?;
    if shape.dimension != TextureDimension::D2
        || shape.arrayed
        || shape.multisampled
        || shape.array_ref
        || shape.writable
    {
        return Err(failure(format!(
            "texture {} is not the D2 single-sample sampled shape this rail executes",
            binding.metal_index
        )));
    }
    // The component mapping is closed the same way: `uint` is the texel-read
    // component C1 executes and `float` is the component C1b samples. The
    // float sample path exists only where the module carries an AIR
    // constexpr sampler, so a float texture with none stays outside the
    // reviewed window rather than being registered with a state nobody
    // stated; the shared constructor names the shape once for both rails.
    let format = match shape.component {
        TextureComponent::Uint => metal_api_core::provider::TextureFormat::R32Uint,
        TextureComponent::Float => {
            if static_sampler.is_none() {
                return Err(failure(format!(
                    "texture {} samples float without an AIR static sampler, which the reviewed compute sampling path does not execute",
                    binding.metal_index
                )));
            }
            metal_api_core::provider::TextureFormat::R32Float
        }
        other => {
            return Err(failure(format!(
                "texture {} samples {other:?}, which the compute texture face does not execute",
                binding.metal_index
            )))
        }
    };
    Ok(TextureBindingContract::sampled(
        binding.metal_index,
        format,
        static_sampler.unwrap_or_else(metal_api_core::provider::SamplerPolicy::synthesized_read),
    ))
}

/// Map one reflected storage image onto the contract's texture face
/// (`research/docs/26` §21.4, C2).
///
/// The reviewed storage class is the sibling of the sampled one: a D2,
/// single-sample, non-arrayed texture the module declares write-capable, whose
/// AIR storage format is `R32f` — the format `texture2d<float, write>` and
/// `texture2d<float, read_write>` lower to, and the only one this increment
/// uploads and lands. Metal's `access::write` and `access::read_write` both
/// arrive here as `ResourceAccess::Storage`; the contract states the one
/// storage access the execution path has instead of inventing a second. The
/// declaration carries no sampler, because a storage descriptor has none, and
/// states the whole view as its landing (`TextureFootprintProof::WholeView`).
///
/// The format list is deliberately narrower than the sampled face's: the
/// translator spells a `texture2d<uint, write>` as `Rgba8ui`, a four-lane
/// 8-bit storage image whose landing this increment has no fixture for, so that
/// shape is refused here with the reflected format named rather than
/// registered as a size nobody proved.
fn map_storage_image_binding(
    binding: &ResourceBinding,
) -> Result<TextureBindingContract, ExecutorError> {
    use metal2vulkan::meta::{TextureDimension, TextureFormat as AirTextureFormat};

    let shape = binding.texture_shape.as_ref().ok_or_else(|| {
        failure(format!(
            "texture {} has no reflected shape",
            binding.metal_index
        ))
    })?;
    if shape.dimension != TextureDimension::D2
        || shape.arrayed
        || shape.multisampled
        || shape.array_ref
        || !shape.writable
    {
        return Err(failure(format!(
            "texture {} is not the D2 single-sample storage image this rail executes",
            binding.metal_index
        )));
    }
    if binding.access != Some(ResourceAccess::Storage) {
        return Err(failure(format!(
            "texture {} is declared writable but the module classifies it as {:?}",
            binding.metal_index, binding.access
        )));
    }
    let format = match shape.storage_format {
        Some(AirTextureFormat::R32f) => metal_api_core::provider::TextureFormat::R32Float,
        other => {
            return Err(failure(format!(
                "texture {} has storage format {other:?}; the reviewed compute storage image is R32f (texture2d<float, write> / texture2d<float, read_write>), which is the only landed shape this increment proves",
                binding.metal_index
            )))
        }
    };
    Ok(TextureBindingContract::storage(binding.metal_index, format))
}

fn map_access(access: Option<ResourceAccess>, index: u32) -> Result<BufferAccess, ExecutorError> {
    match access {
        Some(ResourceAccess::Unused) => Ok(BufferAccess::Unused),
        Some(ResourceAccess::ReadOnly) => Ok(BufferAccess::Read),
        Some(ResourceAccess::WriteOnly) => Ok(BufferAccess::Write),
        Some(ResourceAccess::ReadWrite) => Ok(BufferAccess::ReadWrite),
        Some(other) => Err(failure(format!(
            "buffer {index} has non-buffer access classification {other:?}"
        ))),
        None => Err(failure(format!(
            "buffer {index} has no access classification"
        ))),
    }
}

fn map_footprint(footprint: &BufferFootprint, index: u32) -> Result<FootprintProof, ExecutorError> {
    if footprint.has_unbounded_access {
        return Ok(FootprintProof::Unbounded);
    }
    if !footprint.strided_accesses.is_empty() {
        let mut accesses =
            Vec::with_capacity(footprint.static_ranges.len() + footprint.strided_accesses.len());
        for range in &footprint.static_ranges {
            accesses.push(AffineAccess {
                base_offset: range.offset,
                access_size: range.size,
                terms: Vec::new(),
            });
        }
        for access in &footprint.strided_accesses {
            let mut terms = Vec::with_capacity(access.terms.len());
            for term in &access.terms {
                let axis = match term.source {
                    metal2vulkan::reflect::BufferIndexSource::GlobalInvocationIdX => 0,
                    metal2vulkan::reflect::BufferIndexSource::GlobalInvocationIdY => 1,
                    metal2vulkan::reflect::BufferIndexSource::GlobalInvocationIdZ => 2,
                    other => {
                        return Err(failure(format!(
                            "buffer {index} uses unsupported affine index source {other:?}"
                        )))
                    }
                };
                terms.push(AffineTerm {
                    axis,
                    stride: term.stride,
                });
            }
            accesses.push(AffineAccess {
                base_offset: access.base_offset,
                access_size: access.access_size,
                terms,
            });
        }
        return Ok(FootprintProof::Affine { accesses });
    }
    let mut max_bytes = 0_u64;
    for range in &footprint.static_ranges {
        let end = range
            .offset
            .checked_add(range.size)
            .ok_or_else(|| failure(format!("buffer {index} footprint overflows u64")))?;
        max_bytes = max_bytes.max(end);
    }
    Ok(FootprintProof::Static { max_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::reflect::{
        BufferByteRange, BufferIndexSource, BufferStrideTerm, BufferStridedAccess,
        DescriptorLocation, ResourceBinding, ShaderStage,
    };

    #[test]
    fn footprint_mapping_preserves_bounded_and_unbounded_states() {
        let bounded = BufferFootprint {
            static_ranges: vec![BufferByteRange { offset: 4, size: 8 }],
            strided_accesses: Vec::new(),
            has_unbounded_access: false,
        };
        assert_eq!(
            map_footprint(&bounded, 0).unwrap(),
            FootprintProof::Static { max_bytes: 12 }
        );

        let affine = BufferFootprint {
            static_ranges: Vec::new(),
            strided_accesses: vec![BufferStridedAccess {
                base_offset: 0,
                access_size: 4,
                terms: vec![BufferStrideTerm {
                    source: BufferIndexSource::GlobalInvocationIdX,
                    stride: 4,
                }],
            }],
            has_unbounded_access: false,
        };
        assert!(matches!(
            map_footprint(&affine, 0).unwrap(),
            FootprintProof::Affine { .. }
        ));

        let unbounded = BufferFootprint {
            has_unbounded_access: true,
            ..BufferFootprint::default()
        };
        assert_eq!(
            map_footprint(&unbounded, 0).unwrap(),
            FootprintProof::Unbounded
        );
    }

    #[test]
    fn access_mapping_rejects_missing_or_non_buffer_classifications() {
        assert_eq!(
            map_access(Some(ResourceAccess::ReadOnly), 3).unwrap(),
            BufferAccess::Read
        );
        assert!(map_access(None, 3).is_err());
        assert!(map_access(Some(ResourceAccess::Sampled), 3).is_err());
    }

    #[test]
    fn pipeline_mapping_preserves_reflected_dispatch_and_buffer_contract() {
        let reflection = ShaderReflection {
            reflection_version: 1,
            descriptor_layout: Default::default(),
            stage: ShaderStage::Kernel,
            entry_point: Some("copy_word".to_string()),
            bindings: vec![ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: 0,
                descriptor: Some(DescriptorLocation {
                    set: 0,
                    binding: 0,
                    count: 1,
                }),
                param_index: Some(0),
                stage_input_location: None,
                address_space: Some(1),
                declared_size: Some(4),
                extent: Some(metal2vulkan::reflect::BufferExtent::Object { bytes: 4 }),
                footprint: Some(BufferFootprint {
                    static_ranges: Vec::new(),
                    strided_accesses: vec![BufferStridedAccess {
                        base_offset: 0,
                        access_size: 4,
                        terms: vec![BufferStrideTerm {
                            source: BufferIndexSource::GlobalInvocationIdX,
                            stride: 4,
                        }],
                    }],
                    has_unbounded_access: false,
                }),
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: Some(ResourceAccess::WriteOnly),
                static_sampler: None,
            }],
            argument_buffer_fields: Vec::new(),
            vertex_attributes: Vec::new(),
            varyings: Vec::new(),
            render_targets: Vec::new(),
            depth_members: Vec::new(),
            depth_qualifier: None,
            stencil_members: Vec::new(),
            local_size: Some([8, 2, 1]),
            max_work_group_size: Some(16),
            kernel_dispatch: Some(KernelDispatch::ThreadsDynamic { offset: 0 }),
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: Vec::new(),
            implicit_imageblock_attachments: Vec::new(),
            fragment_imageblock: None,
            datalayout: None,
            runtime_sampler_specializations: Vec::new(),
            runtime_storage_image_specializations: Vec::new(),
            function_constants: Vec::new(),
        };
        let contract = pipeline_contract(&reflection, None).unwrap();
        assert_eq!(contract.dispatch_kind, DispatchKind::ThreadsExact);
        assert_eq!(contract.required_local_size, None);
        assert_eq!(contract.push_constant_bytes, 48);
        assert_eq!(contract.buffer_bindings[0].access, BufferAccess::Write);
        assert!(matches!(
            contract.buffer_bindings[0].footprint,
            FootprintProof::Affine { .. }
        ));
    }

    /// The per-stage stage-buffer window is the device's own answer
    /// (`research/docs/23` §117, E-SB2): the review ceiling clamped by
    /// `maxPerStageDescriptorStorageBuffers`, the pair's sum as the list bound,
    /// and the device's per-set window beside them. A device at the spec's
    /// four-slot per-stage floor declares four rather than the review's eight —
    /// the number a pipeline layout can actually carry — and a device whose
    /// per-stage window is wider than the review's is capped by the review.
    #[test]
    fn the_stage_buffer_window_is_the_ceiling_clamped_by_the_device() {
        let core_floor = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_storage_buffers: 4,
            max_descriptor_set_storage_buffers: 8,
            ..Default::default()
        };
        assert_eq!(
            stage_buffer_window(&core_floor),
            StageBufferWindow {
                per_stage: 4,
                list: 8,
                per_set: 8,
            },
            "a device at the core per-stage floor declares four, not the review's eight"
        );

        let lavapipe = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_storage_buffers: 1_015_808,
            max_descriptor_set_storage_buffers: 1_015_808,
            ..Default::default()
        };
        assert_eq!(
            stage_buffer_window(&lavapipe),
            StageBufferWindow {
                per_stage: MAX_RENDER_STAGE_BUFFERS as u32,
                list: metal_api_core::provider::MAX_RENDER_STAGE_BUFFER_DECLARATIONS as u32,
                per_set: 1_015_808,
            },
            "the review ceiling caps a wider device at eight per stage and sixteen in the list"
        );

        let between = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_storage_buffers: 6,
            max_descriptor_set_storage_buffers: 16,
            ..Default::default()
        };
        assert_eq!(
            stage_buffer_window(&between),
            StageBufferWindow {
                per_stage: 6,
                list: 12,
                per_set: 16,
            },
            "a device between the floor and the ceiling states its own per-stage number"
        );
    }

    /// The mapped capability snapshot carries the window the helper states, so
    /// a consumer reads the same numbers the rail enforces.
    #[test]
    fn the_stage_buffer_capability_carries_the_device_window() {
        let limits = vk::PhysicalDeviceLimits {
            max_per_stage_descriptor_storage_buffers: 10,
            max_descriptor_set_storage_buffers: 12,
            ..Default::default()
        };
        let capabilities = capabilities_from_limits(&limits);
        assert!(capabilities.supports_render_stage_buffers);
        assert_eq!(
            capabilities.max_render_stage_buffers_per_stage,
            MAX_RENDER_STAGE_BUFFERS as u32
        );
        assert_eq!(
            capabilities.max_render_stage_buffers,
            metal_api_core::provider::MAX_RENDER_STAGE_BUFFER_DECLARATIONS as u32
        );
        assert!(capabilities.declares_render_stage_buffer_per_stage_ceiling());
    }

    #[test]
    fn capabilities_mapping_uses_the_tightest_descriptor_limit() {
        let limits = vk::PhysicalDeviceLimits {
            max_compute_work_group_size: [8, 4, 2],
            max_compute_work_group_invocations: 32,
            max_compute_work_group_count: [16, 8, 4],
            max_per_stage_descriptor_storage_buffers: 12,
            max_descriptor_set_storage_buffers: 10,
            max_per_stage_resources: 14,
            max_storage_buffer_range: 4096,
            max_push_constants_size: 128,
            max_framebuffer_width: 32,
            max_framebuffer_height: 8,
            ..Default::default()
        };
        let capabilities = capabilities_from_limits(&limits);
        assert_eq!(capabilities.max_local_size, [8, 4, 2]);
        assert_eq!(capabilities.max_invocations, 32);
        assert_eq!(capabilities.max_group_count, [16, 8, 4]);
        assert_eq!(capabilities.max_storage_buffer_descriptors, 10);
        assert_eq!(capabilities.max_buffer_range, 4096);
        assert_eq!(capabilities.max_push_constant_bytes, 128);
        assert_eq!(capabilities.storage_modes, vec![StorageMode::OwnedBytes]);
        assert!(!capabilities.supports_threadgroups);
        assert!(!capabilities.submit_only);
        // The render bits are the rail's own window in every format the render
        // contract admits, and the attachment window is the reviewed ceiling
        // clamped by this device's own framebuffer limits (R1b,
        // `research/docs/23` §70; R5a, §73): 32×8 rather than 2048×2048 here.
        assert!(capabilities.supports_render_passes);
        assert_eq!(
            capabilities.max_color_attachments,
            MAX_COLOR_ATTACHMENTS as u32
        );
        assert_eq!(capabilities.max_attachment_dimension, [32, 8]);
        assert_eq!(
            capabilities.supported_color_formats,
            AttachmentFormat::ADMITTED.to_vec()
        );
        assert!(capabilities.declares_render_support());
        // The present bits name the readable-swapchain-equivalent window the
        // rail executes (`research/docs/24` §6 Step 3): one target, Fifo only,
        // single buffering.
        assert!(capabilities.supports_presentation);
        assert_eq!(capabilities.max_present_targets, MAX_PRESENT_TARGETS as u32);
        assert_eq!(
            capabilities.supported_present_modes,
            PresentMode::ADMITTED.to_vec()
        );
        assert_eq!(
            capabilities.max_present_image_count,
            MAX_PRESENT_IMAGE_COUNT
        );
        assert!(capabilities.declares_presentation_support());
        // The stage-buffer face and its folded shape (`research/docs/23` §3.3,
        // v83/E-TX9): the pair's readings are unchanged by the new bit, and
        // the shape bit is declared beside them because this rail arranges the
        // two stages' buffers in different descriptor slots.
        assert!(capabilities.supports_render_stage_buffers);
        // The stage-buffer window is the device's own answer (`research/docs/23`
        // §117, E-SB2): this snapshot's device states twelve per-stage and ten
        // per-set storage buffers, so the review ceiling caps the per-stage
        // window at eight and the list bound at the pair's sixteen — a wider
        // list than the eight this snapshot used to state, which is exactly the
        // shape the increment admits.
        assert_eq!(
            capabilities.max_render_stage_buffers_per_stage,
            MAX_RENDER_STAGE_BUFFERS as u32
        );
        assert_eq!(
            capabilities.max_render_stage_buffers,
            metal_api_core::provider::MAX_RENDER_STAGE_BUFFER_DECLARATIONS as u32
        );
        assert!(capabilities.declares_render_stage_buffer_per_stage_ceiling());
        assert!(capabilities.supports_render_stage_buffer_namespace_split);
        assert!(capabilities.declares_render_stage_buffer_namespace_split());
        // The gathered extent (`research/docs/23` §3.3, E-TX10): the three
        // render-sampler readings above/below are unchanged by the new bit, and
        // the shape bit is declared because this rail already executes the
        // host-bytes arm — the translated binding and the reviewed gather in
        // `tests/render_texture_extent_e2e.rs`.
        assert!(capabilities.supports_render_texture_sampling);
        assert_eq!(capabilities.max_render_textures, MAX_RENDER_TEXTURES as u32);
        assert_eq!(
            capabilities.supported_render_texture_formats,
            TextureFormat::RENDER_SAMPLED.to_vec()
        );
        assert!(
            capabilities
                .supported_render_texture_formats
                .contains(&TextureFormat::Rgba16Float),
            "the eight-byte half-float lane is part of the window this frame states"
        );
        assert!(capabilities.supports_render_texture_gathered_extent);
        assert!(capabilities.declares_render_texture_gathered_extent_support());
        // The gathered extent's no-copy arm (`research/docs/23` §111, E-TX12):
        // the same three render-sampler readings stay where they were, and this
        // rail declares the second shape bit because it executes the owner's
        // window in place — the gathered fetch sibling and the translated
        // binding in `tests/render_texture_extent_nocopy_e2e.rs` — while the
        // sampling sibling keeps its refusal by name.
        assert!(capabilities.supports_render_texture_gathered_extent_no_copy);
        assert!(capabilities.declares_render_texture_gathered_extent_no_copy_support());
        // The superset vertex interface (`research/docs/23` §3.3, E-TX11): the
        // three vertex-input readings below are unchanged by the new bit, and
        // the shape bit is declared because this rail's vertex input state is
        // built from the contract's own layout — the four-attribute stream in
        // `tests/render_vertex_superset_e2e.rs` lands the two-attribute frame.
        assert_eq!(
            capabilities.max_vertex_buffers,
            metal_api_core::provider::MAX_VERTEX_BUFFERS as u32
        );
        assert_eq!(
            capabilities.supported_vertex_formats,
            metal_api_core::provider::VertexFormat::ADMITTED.to_vec()
        );
        assert_eq!(
            capabilities.supported_index_formats,
            metal_api_core::provider::IndexFormat::ADMITTED.to_vec()
        );
        assert!(capabilities.supports_render_vertex_interface_superset);
        assert!(capabilities.declares_render_vertex_interface_superset_support());
    }

    #[test]
    fn attachment_window_is_the_ceiling_clamped_by_the_device_limits() {
        // R1b (`research/docs/23` §70): the declared window is per axis the
        // reviewed ceiling or the device's own framebuffer limit, whichever is
        // smaller, so a device narrower than the review declares its own
        // number instead of the ceiling.
        let wide = vk::PhysicalDeviceLimits {
            max_framebuffer_width: 16384,
            max_framebuffer_height: 16384,
            ..Default::default()
        };
        assert_eq!(
            attachment_dimension_window(&wide),
            REVIEWED_ATTACHMENT_CEILING
        );

        let narrow = vk::PhysicalDeviceLimits {
            max_framebuffer_width: 32,
            max_framebuffer_height: 8,
            ..Default::default()
        };
        assert_eq!(attachment_dimension_window(&narrow), [32, 8]);

        // One axis at the device's limit and the other at the ceiling: the
        // clamp is per axis, not a single "fits or not" answer.
        let mixed = vk::PhysicalDeviceLimits {
            max_framebuffer_width: 4,
            max_framebuffer_height: 4096,
            ..Default::default()
        };
        assert_eq!(
            attachment_dimension_window(&mixed),
            [4, REVIEWED_ATTACHMENT_CEILING[1]]
        );
    }

    #[test]
    fn depth_resolve_mode_mask_maps_the_admitted_filters_and_drops_the_rest() {
        // The three admitted filters are bits 0/1/2; every Vulkan mode the
        // contract does not carry (AVERAGE and friends) is dropped rather than
        // folded onto a filter the caller did not ask for
        // (`research/docs/23` §3.3, v57).
        assert_eq!(
            depth_resolve_mode_mask(vk::ResolveModeFlags::SAMPLE_ZERO),
            0b1
        );
        assert_eq!(depth_resolve_mode_mask(vk::ResolveModeFlags::MIN), 0b10);
        assert_eq!(depth_resolve_mode_mask(vk::ResolveModeFlags::MAX), 0b100);
        assert_eq!(
            depth_resolve_mode_mask(
                vk::ResolveModeFlags::SAMPLE_ZERO
                    | vk::ResolveModeFlags::MIN
                    | vk::ResolveModeFlags::MAX
                    | vk::ResolveModeFlags::AVERAGE
            ),
            0b111
        );
        assert_eq!(depth_resolve_mode_mask(vk::ResolveModeFlags::AVERAGE), 0);
        assert_eq!(depth_resolve_mode_mask(vk::ResolveModeFlags::empty()), 0);
    }

    #[test]
    fn stencil_resolve_mode_mask_admits_sample_zero_alone_and_drops_min_max() {
        // The contract's stencil family is Sample0 / DepthResolvedSample, and
        // Vulkan has no mode for the latter: SAMPLE_ZERO maps onto bit 0, while
        // MIN and MAX reduce stencil values themselves and are dropped rather
        // than folded onto a filter the caller did not ask for
        // (`research/docs/23` §3.3, v60).
        assert_eq!(
            stencil_resolve_mode_mask(vk::ResolveModeFlags::SAMPLE_ZERO),
            0b1
        );
        assert_eq!(
            stencil_resolve_mode_mask(
                vk::ResolveModeFlags::SAMPLE_ZERO
                    | vk::ResolveModeFlags::MIN
                    | vk::ResolveModeFlags::MAX
            ),
            0b1
        );
        assert_eq!(stencil_resolve_mode_mask(vk::ResolveModeFlags::MIN), 0);
        assert_eq!(stencil_resolve_mode_mask(vk::ResolveModeFlags::MAX), 0);
        assert_eq!(stencil_resolve_mode_mask(vk::ResolveModeFlags::empty()), 0);
    }
}
