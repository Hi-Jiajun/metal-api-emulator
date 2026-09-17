//! Vulkan-to-neutral provider contract mapping.
//!
//! This module translates reflection and selected-device limits into the pure
//! values owned by `metal-api-core::provider`. It deliberately does not expose
//! Vulkan descriptors, region plans, handles, or guest memory pointers.

use ash::vk;
use metal2vulkan::reflect::{
    BufferFootprint, KernelDispatch, ResourceAccess, ResourceKind, ShaderReflection,
};
use metal_api_core::provider::{
    AffineAccess, AffineTerm, AliasMode, AttachmentFormat, BufferAccess, BufferBindingContract,
    DepthResolveFilter, DispatchKind, FootprintProof, IndirectCommandKind, PipelineContract,
    PresentMode, ProviderCapabilities, SemanticDigest, StencilResolveFilter, StorageMode,
    TextureFormat, MAX_COLOR_ATTACHMENTS, MAX_PRESENT_IMAGE_COUNT, MAX_PRESENT_TARGETS,
    MAX_RENDER_TEXTURES,
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
        // count and per format (the single-output, dual, triple and quad 8-bit
        // modules, the single-channel float module, and the depth-only stage
        // beside an empty colour list).
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
        max_vertex_buffers: metal_api_core::provider::MAX_VERTEX_BUFFERS as u32,
        supported_vertex_formats: metal_api_core::provider::VertexFormat::ADMITTED.to_vec(),
        supported_index_formats: metal_api_core::provider::IndexFormat::ADMITTED.to_vec(),
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
        // one binding, one `rgba8_unorm` format, one texture of the render
        // area's own extent — so a wider request is refused by core admission
        // or by the rail's shape gates rather than silently narrowed.
        supports_render_texture_sampling: true,
        max_render_textures: MAX_RENDER_TEXTURES as u32,
        supported_render_texture_formats: vec![TextureFormat::Rgba8Unorm],
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

    let mut buffer_bindings = Vec::with_capacity(reflection.bindings.len());
    for binding in &reflection.bindings {
        // Sampled textures are execution resources, not contract buffer
        // bindings; the provider's reflection validation already admitted them
        // (`research/docs/16` §4.7).
        if binding.kind == ResourceKind::Texture {
            continue;
        }
        if binding.kind != ResourceKind::Buffer {
            return Err(failure(format!(
                "provider contract only maps Metal buffers, found {:?} at {}",
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

    buffer_bindings.sort_by_key(|binding| binding.metal_binding);
    let contract = PipelineContract {
        dispatch_kind,
        required_local_size,
        fixed_grid,
        push_constant_offset,
        push_constant_bytes,
        buffer_bindings,
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
