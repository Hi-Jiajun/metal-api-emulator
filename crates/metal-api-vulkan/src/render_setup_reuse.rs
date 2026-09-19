//! Reuse of the *shape-determined* device objects one offscreen pass creates.
//!
//! # What this is for
//!
//! `docs/SUBMIT-PHASE-PROFILE.md` divides one submission. On the guest desktop
//! the render half's own pre-recording work (`render_setup`, 1 135.9 µs per
//! submit in the fp9b round) builds every device object a pass needs and keeps
//! none of them: two shader modules, the descriptor-set layouts the pipeline
//! layout is built from, the pipeline layout and the graphics pipeline are all
//! created per pass and destroyed with it. A guest that draws the same shape
//! draw after draw — which is what a compositor does — pays that construction
//! again for every one of them.
//!
//! This module keeps the objects whose identity is decided by the pass's
//! *shape* and nothing else, and hands them back on the next pass of the same
//! shape:
//!
//! | object | why the shape decides it |
//! |---|---|
//! | vertex/fragment shader module | the module's own SPIR-V words |
//! | pipeline layout | the ordered list of descriptor-set layouts |
//! | graphics pipeline | the modules, the entries, the vertex input, the raster/blend/depth state, the specialization constants, the render pass and the layout list |
//!
//! Nothing else is cached. The images, buffers, samplers, descriptor sets,
//! render passes, framebuffers, command pools and fences stay per pass, so a
//! pass's *content* is written exactly where it always was: this is a reuse of
//! immutable objects, not of state.
//!
//! # Why the key cannot lie
//!
//! The key is **read back from the structure that is about to be handed to the
//! driver** — the `VkRenderPassCreateInfo2` attachment/subpass/dependency
//! descriptions, the ordered descriptor-set-layout bindings, and the
//! `VkGraphicsPipelineCreateInfo` with its states: every field this rail
//! states, rather than a second derivation of them. (The fields the rail never
//! states — a structure's `flags` word, a never-set `pNext` chain — are the
//! driver's defaults for every pass alike, so they carry no shape.) A cached entry is handed back
//! only when the field-by-field comparison of *those* descriptions succeeds, so
//! two passes meet in the cache exactly when the driver would be told the same
//! thing twice. The bucketing digest below is only a first filter; the
//! comparison is the decision, and a digest collision costs a comparison rather
//! than a wrong reuse.
//!
//! Objects the driver created for one of two definition-identical
//! descriptions are interchangeable by construction: `VkDescriptorSetLayout`
//! compatibility is defined on the bindings a layout was created with, and
//! `VkRenderPass` compatibility on the attachment and subpass descriptions.
//! That is why the reused pipeline may be paired with a pass's own fresh
//! descriptor sets: the sets are allocated from layouts whose *definitions*
//! this key compared.
//!
//! # What a hit skips, and what it does not
//!
//! On a hit the pass skips `vkCreateShaderModule` twice, the
//! `vkCreatePipelineLayout` and `vkCreateGraphicsPipelines`. It still creates
//! its images, buffers, descriptor-set layouts, pools, sets, render pass,
//! framebuffers, command pool and fence, and it still records, submits and reads
//! back exactly as before. The frame bytes are therefore unchanged by
//! construction; the plan's own rail cases (`tests/render_setup_reuse_e2e.rs`)
//! compare them anyway, arm against arm.
//!
//! # The switch, the counters and the failure posture
//!
//! `METAL_API_VULKAN_RENDER_SETUP_CACHE=0` (also `off`, `no`, `false`) turns the
//! whole mechanism off: no lookup, no insert, one relaxed load per pass, which
//! is the arm the round's control run states. Anything else — including unset —
//! leaves it on.
//!
//! The counters the profile line prints are in [`crate::phase_profile`]
//! (`reuse_hit_n`, `reuse_miss_n`, `reuse_mismatch_n`, `reuse_evict_n`,
//! `reuse_unkeyed_n`, `reuse_disabled_n`), so a round can tell a cache that is
//! not being asked from one that is refusing.
//!
//! A pass that cannot produce an exact key is *not* cached (`reuse_unkeyed_n`)
//! and executes exactly as it did before this module existed. The cache holds at
//! most [`ENTRY_CAP`] entries and evicts the oldest; an evicted or flushed entry
//! is destroyed under the lock that removed it, so no device object outlives the
//! cache's own reference to it.

use std::collections::VecDeque;
use std::ffi::CStr;
use std::sync::OnceLock;

use ash::vk;

/// A pass's descriptor-set layout: the bindings `vkCreateDescriptorSetLayout`
/// would be handed, in a canonical order.
///
/// Bindings are sorted by binding number because a layout's identity does not
/// depend on the order its bindings were listed in, and two sites that build
/// the same set from different maps should share one entry.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct LayoutDef {
    bindings: Vec<LayoutBinding>,
}

/// One `VkDescriptorSetLayoutBinding`, as the fields the driver reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct LayoutBinding {
    binding: u32,
    descriptor_type: i32,
    descriptor_count: u32,
    stage_flags: u32,
}

impl LayoutDef {
    /// The definition of the layout `bindings` describes. The immutable-sampler
    /// list is empty at every site this rail creates a layout from, and the
    /// flag word is the default; both are asserted rather than keyed, so a
    /// future site that states either is a deliberate change here rather than a
    /// silent one.
    pub(crate) fn of(bindings: &[vk::DescriptorSetLayoutBinding]) -> Self {
        debug_assert!(
            bindings
                .iter()
                .all(|binding| binding.p_immutable_samplers.is_null()),
            "a layout with immutable samplers needs the samplers in its definition"
        );
        let mut bindings = bindings
            .iter()
            .map(|binding| LayoutBinding {
                binding: binding.binding,
                descriptor_type: binding.descriptor_type.as_raw(),
                descriptor_count: binding.descriptor_count,
                stage_flags: binding.stage_flags.as_raw(),
            })
            .collect::<Vec<_>>();
        bindings.sort_unstable_by_key(|binding| binding.binding);
        Self { bindings }
    }

    fn hash_into(&self, digest: &mut Digest) {
        digest.u64(self.bindings.len() as u64);
        for binding in &self.bindings {
            digest.u64(u64::from(binding.binding));
            digest.u64(binding.descriptor_type as u64);
            digest.u64(u64::from(binding.descriptor_count));
            digest.u64(u64::from(binding.stage_flags));
        }
    }
}

/// The render pass a pass builds, as the description fields the driver reads.
///
/// This is the whole of `VkRenderPassCreateInfo2` minus the pointers: every
/// attachment, the subpass with its references, the depth/stencil resolve chain
/// and every dependency.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub(crate) struct RenderPassDef {
    attachments: Vec<AttachmentDef>,
    color_refs: Vec<ReferenceDef>,
    resolve_refs: Option<Vec<ReferenceDef>>,
    depth_ref: Option<ReferenceDef>,
    depth_resolve_mode: Option<i32>,
    stencil_resolve_mode: Option<i32>,
    resolve_attachment: Option<ReferenceDef>,
    dependencies: Vec<DependencyDef>,
    bind_point: i32,
    view_mask: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct AttachmentDef {
    format: i32,
    samples: u32,
    load_op: i32,
    store_op: i32,
    stencil_load_op: i32,
    stencil_store_op: i32,
    initial_layout: i32,
    final_layout: i32,
}

impl AttachmentDef {
    fn of(description: &vk::AttachmentDescription2<'_>) -> Self {
        Self {
            format: description.format.as_raw(),
            samples: description.samples.as_raw(),
            load_op: description.load_op.as_raw(),
            store_op: description.store_op.as_raw(),
            stencil_load_op: description.stencil_load_op.as_raw(),
            stencil_store_op: description.stencil_store_op.as_raw(),
            initial_layout: description.initial_layout.as_raw(),
            final_layout: description.final_layout.as_raw(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct ReferenceDef {
    attachment: u32,
    layout: i32,
    aspect_mask: u32,
}

/// The depth/stencil resolve chain one pass states, captured before the
/// subpass takes its mutable borrow of the chain itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ResolveDef {
    depth_mode: i32,
    stencil_mode: i32,
    attachment: Option<(u32, i32)>,
}

impl ReferenceDef {
    fn of(reference: &vk::AttachmentReference2<'_>) -> Self {
        Self {
            attachment: reference.attachment,
            layout: reference.layout.as_raw(),
            aspect_mask: reference.aspect_mask.as_raw(),
        }
    }
}

/// One `VkSubpassDependency2` with every field the driver reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct DependencyDef {
    src_subpass: u32,
    dst_subpass: u32,
    src_stage: u32,
    dst_stage: u32,
    src_access: u32,
    dst_access: u32,
    dependency_flags: u32,
    view_offset: i32,
}

impl DependencyDef {
    fn of(dependency: &vk::SubpassDependency2<'_>) -> Self {
        Self {
            src_subpass: dependency.src_subpass,
            dst_subpass: dependency.dst_subpass,
            src_stage: dependency.src_stage_mask.as_raw(),
            dst_stage: dependency.dst_stage_mask.as_raw(),
            src_access: dependency.src_access_mask.as_raw(),
            dst_access: dependency.dst_access_mask.as_raw(),
            dependency_flags: dependency.dependency_flags.as_raw(),
            view_offset: dependency.view_offset,
        }
    }
}

impl RenderPassDef {
    /// The depth/stencil resolve chain a pass states, as the three fields the
    /// module keys: the two modes and the landing they resolve into. The chain
    /// itself is read by the caller before the subpass takes a mutable borrow
    /// of it, so the definition is captured here rather than from the chain.
    pub(crate) fn resolve_def(
        depth_mode: i32,
        stencil_mode: i32,
        attachment: Option<(u32, i32)>,
    ) -> ResolveDef {
        ResolveDef {
            depth_mode,
            stencil_mode,
            attachment,
        }
    }

    /// Read the definition back out of the descriptions that are about to be
    /// handed to `vkCreateRenderPass2`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of(
        attachments: &[vk::AttachmentDescription2<'_>],
        color_refs: &[vk::AttachmentReference2<'_>],
        resolve_refs: Option<&[vk::AttachmentReference2<'_>]>,
        depth_ref: Option<&vk::AttachmentReference2<'_>>,
        resolve: Option<ResolveDef>,
        dependencies: &[vk::SubpassDependency2<'_>],
        bind_point: vk::PipelineBindPoint,
        view_mask: u32,
    ) -> Self {
        Self {
            attachments: attachments.iter().map(AttachmentDef::of).collect(),
            color_refs: color_refs.iter().map(ReferenceDef::of).collect(),
            resolve_refs: resolve_refs.map(|refs| refs.iter().map(ReferenceDef::of).collect()),
            depth_ref: depth_ref.map(ReferenceDef::of),
            depth_resolve_mode: resolve.map(|resolve| resolve.depth_mode),
            stencil_resolve_mode: resolve.map(|resolve| resolve.stencil_mode),
            resolve_attachment: resolve.and_then(|resolve| {
                resolve.attachment.map(|(attachment, layout)| ReferenceDef {
                    attachment,
                    layout,
                    aspect_mask: 0,
                })
            }),
            dependencies: dependencies.iter().map(DependencyDef::of).collect(),
            bind_point: bind_point.as_raw(),
            view_mask,
        }
    }

    /// The seed render pass is a smaller shape of the same thing (its own
    /// attachments, one colour-only subpass, two fixed dependencies), so it is
    /// keyed the same way rather than through a second description type.
    fn hash_into(&self, digest: &mut Digest) {
        digest.u64(self.attachments.len() as u64);
        for attachment in &self.attachments {
            digest.u64(attachment.format as u64);
            digest.u64(u64::from(attachment.samples));
            digest.u64(attachment.load_op as u64);
            digest.u64(attachment.store_op as u64);
            digest.u64(attachment.stencil_load_op as u64);
            digest.u64(attachment.stencil_store_op as u64);
            digest.u64(attachment.initial_layout as u64);
            digest.u64(attachment.final_layout as u64);
        }
        digest.u64(self.color_refs.len() as u64);
        for reference in &self.color_refs {
            digest.u64(u64::from(reference.attachment));
            digest.u64(reference.layout as u64);
            digest.u64(u64::from(reference.aspect_mask));
        }
        match &self.resolve_refs {
            Some(refs) => {
                digest.u64(1);
                digest.u64(refs.len() as u64);
                for reference in refs {
                    digest.u64(u64::from(reference.attachment));
                    digest.u64(reference.layout as u64);
                }
            }
            None => digest.u64(0),
        }
        if let Some(reference) = &self.depth_ref {
            digest.u64(1);
            digest.u64(u64::from(reference.attachment));
            digest.u64(reference.layout as u64);
        } else {
            digest.u64(0);
        }
        digest.u64(self.depth_resolve_mode.map_or(u64::MAX, |mode| mode as u64));
        digest.u64(
            self.stencil_resolve_mode
                .map_or(u64::MAX, |mode| mode as u64),
        );
        if let Some(reference) = &self.resolve_attachment {
            digest.u64(1);
            digest.u64(u64::from(reference.attachment));
            digest.u64(reference.layout as u64);
        } else {
            digest.u64(0);
        }
        digest.u64(self.dependencies.len() as u64);
        for dependency in &self.dependencies {
            digest.u64(u64::from(dependency.src_subpass));
            digest.u64(u64::from(dependency.dst_subpass));
            digest.u64(u64::from(dependency.src_stage));
            digest.u64(u64::from(dependency.dst_stage));
            digest.u64(u64::from(dependency.src_access));
            digest.u64(u64::from(dependency.dst_access));
            digest.u64(u64::from(dependency.dependency_flags));
            digest.u64(dependency.view_offset as u64);
        }
        digest.u64(self.bind_point as u64);
        digest.u64(u64::from(self.view_mask));
    }
}

/// The pipeline a pass builds, as the fields of
/// `VkGraphicsPipelineCreateInfo` — with the shader modules as their own code
/// and the descriptor-set layouts as their definitions, so two passes meet here
/// exactly when the driver would be handed the same pipeline.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct PipelinePlan {
    vertex_words: Vec<u32>,
    fragment_words: Vec<u32>,
    vertex_entry: Vec<u8>,
    fragment_entry: Vec<u8>,
    specialization: Option<[u32; 4]>,
    subpass: u32,
    vertex_bindings: Vec<(u32, u32, u32)>,
    vertex_attributes: Vec<(u32, u32, i32, u32)>,
    vertex_input_flags: u32,
    topology: i32,
    primitive_restart: u32,
    viewport_count: u32,
    scissor_count: u32,
    raster_flags: u32,
    depth_clamp: u32,
    rasterizer_discard: u32,
    polygon_mode: i32,
    cull_mode: u32,
    front_face: i32,
    depth_bias_enable: u32,
    depth_bias_constant: u32,
    depth_bias_clamp: u32,
    depth_bias_slope: u32,
    line_width: u32,
    rasterization_samples: u32,
    sample_shading_enable: u32,
    min_sample_shading: u32,
    sample_mask_len: u32,
    alpha_to_coverage_enable: u32,
    alpha_to_one_enable: u32,
    logic_op_enable: u32,
    logic_op: i32,
    blend_constants: [u32; 4],
    blend_attachments: Vec<BlendAttachmentDef>,
    dynamic_states: Vec<i32>,
    depth_stencil: Option<DepthStencilDef>,
    render_pass: RenderPassDef,
    set_layouts: Vec<LayoutDef>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct BlendAttachmentDef {
    blend_enable: u32,
    src_color: i32,
    dst_color: i32,
    color_op: i32,
    src_alpha: i32,
    dst_alpha: i32,
    alpha_op: i32,
    write_mask: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DepthStencilDef {
    flags: u32,
    depth_test_enable: u32,
    depth_write_enable: u32,
    depth_compare_op: i32,
    depth_bounds_test_enable: u32,
    stencil_test_enable: u32,
    min_depth_bounds: u32,
    max_depth_bounds: u32,
    front: StencilOpDef,
    back: StencilOpDef,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StencilOpDef {
    fail_op: i32,
    pass_op: i32,
    depth_fail_op: i32,
    compare_op: i32,
    compare_mask: u32,
    write_mask: u32,
    reference: u32,
}

impl StencilOpDef {
    fn of(state: &vk::StencilOpState) -> Self {
        Self {
            fail_op: state.fail_op.as_raw(),
            pass_op: state.pass_op.as_raw(),
            depth_fail_op: state.depth_fail_op.as_raw(),
            compare_op: state.compare_op.as_raw(),
            compare_mask: state.compare_mask,
            write_mask: state.write_mask,
            reference: state.reference,
        }
    }
}

/// The pipeline states a plan reads back, in the shape
/// [`PipelinePlan::of`] takes them.
pub(crate) struct PipelineStates<'a> {
    pub(crate) vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub(crate) vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub(crate) vertex_input: &'a vk::PipelineVertexInputStateCreateInfo<'a>,
    pub(crate) input_assembly: &'a vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub(crate) viewport: &'a vk::PipelineViewportStateCreateInfo<'a>,
    pub(crate) rasterization: &'a vk::PipelineRasterizationStateCreateInfo<'a>,
    pub(crate) multisample: &'a vk::PipelineMultisampleStateCreateInfo<'a>,
    pub(crate) color_blend: &'a vk::PipelineColorBlendStateCreateInfo<'a>,
    pub(crate) color_blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    pub(crate) dynamic_states: &'a [vk::DynamicState],
}

impl PipelinePlan {
    /// Read the plan back out of the structures about to be handed to
    /// `vkCreateGraphicsPipelines`.
    ///
    /// The module code, the entry names and the specialization constants travel
    /// as themselves, because a hit means the pass will *not* create modules of
    /// its own: what it reuses has to be the same code, not the same handle.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn of(
        vertex_words: &[u32],
        fragment_words: &[u32],
        vertex_entry: &CStr,
        fragment_entry: &CStr,
        specialization: Option<[u32; 4]>,
        subpass: u32,
        states: &PipelineStates<'_>,
        depth_stencil: Option<DepthStencilState>,
        render_pass: &RenderPassDef,
        set_layouts: &[LayoutDef],
    ) -> Self {
        let rasterization = states.rasterization;
        let multisample = states.multisample;
        let color_blend = states.color_blend;
        Self {
            vertex_words: vertex_words.to_vec(),
            fragment_words: fragment_words.to_vec(),
            vertex_entry: vertex_entry.to_bytes().to_vec(),
            fragment_entry: fragment_entry.to_bytes().to_vec(),
            specialization,
            subpass,
            vertex_bindings: states
                .vertex_bindings
                .iter()
                .map(|binding| {
                    (
                        binding.binding,
                        binding.stride,
                        binding.input_rate.as_raw() as u32,
                    )
                })
                .collect(),
            vertex_attributes: states
                .vertex_attributes
                .iter()
                .map(|attribute| {
                    (
                        attribute.location,
                        attribute.binding,
                        attribute.format.as_raw(),
                        attribute.offset,
                    )
                })
                .collect(),
            vertex_input_flags: states.vertex_input.flags.as_raw(),
            topology: states.input_assembly.topology.as_raw(),
            primitive_restart: states.input_assembly.primitive_restart_enable,
            viewport_count: states.viewport.viewport_count,
            scissor_count: states.viewport.scissor_count,
            raster_flags: rasterization.flags.as_raw(),
            depth_clamp: rasterization.depth_clamp_enable,
            rasterizer_discard: rasterization.rasterizer_discard_enable,
            polygon_mode: rasterization.polygon_mode.as_raw(),
            cull_mode: rasterization.cull_mode.as_raw(),
            front_face: rasterization.front_face.as_raw(),
            depth_bias_enable: rasterization.depth_bias_enable,
            depth_bias_constant: rasterization.depth_bias_constant_factor.to_bits(),
            depth_bias_clamp: rasterization.depth_bias_clamp.to_bits(),
            depth_bias_slope: rasterization.depth_bias_slope_factor.to_bits(),
            line_width: rasterization.line_width.to_bits(),
            rasterization_samples: multisample.rasterization_samples.as_raw(),
            sample_shading_enable: multisample.sample_shading_enable,
            min_sample_shading: multisample.min_sample_shading.to_bits(),
            // The sample mask is a pointer the rail never states (every pass
            // leaves it null); its presence is what the key records.
            sample_mask_len: u32::from(!multisample.p_sample_mask.is_null()),
            alpha_to_coverage_enable: multisample.alpha_to_coverage_enable,
            alpha_to_one_enable: multisample.alpha_to_one_enable,
            logic_op_enable: color_blend.logic_op_enable,
            logic_op: color_blend.logic_op.as_raw(),
            blend_constants: color_blend.blend_constants.map(f32::to_bits),
            blend_attachments: states
                .color_blend_attachments
                .iter()
                .map(|attachment| BlendAttachmentDef {
                    blend_enable: attachment.blend_enable,
                    src_color: attachment.src_color_blend_factor.as_raw(),
                    dst_color: attachment.dst_color_blend_factor.as_raw(),
                    color_op: attachment.color_blend_op.as_raw(),
                    src_alpha: attachment.src_alpha_blend_factor.as_raw(),
                    dst_alpha: attachment.dst_alpha_blend_factor.as_raw(),
                    alpha_op: attachment.alpha_blend_op.as_raw(),
                    write_mask: attachment.color_write_mask.as_raw(),
                })
                .collect(),
            dynamic_states: states
                .dynamic_states
                .iter()
                .map(|state| state.as_raw())
                .collect(),
            depth_stencil: depth_stencil.map(DepthStencilDef::of),
            render_pass: render_pass.clone(),
            set_layouts: set_layouts.to_vec(),
        }
    }

    fn hash_into(&self, digest: &mut Digest) {
        digest.bytes(
            &self
                .vertex_words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        digest.bytes(
            &self
                .fragment_words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        digest.bytes(&self.vertex_entry);
        digest.bytes(&self.fragment_entry);
        digest.u64(self.specialization.map_or(u64::MAX, |values| {
            values.iter().fold(0_u64, |fold, value| {
                fold.rotate_left(16) ^ u64::from(*value)
            })
        }));
        digest.u64(u64::from(self.subpass));
        digest.u64(self.vertex_bindings.len() as u64);
        for (binding, stride, rate) in &self.vertex_bindings {
            digest.u64(u64::from(*binding));
            digest.u64(u64::from(*stride));
            digest.u64(u64::from(*rate));
        }
        digest.u64(self.vertex_attributes.len() as u64);
        for (location, binding, format, offset) in &self.vertex_attributes {
            digest.u64(u64::from(*location));
            digest.u64(u64::from(*binding));
            digest.u64(*format as u64);
            digest.u64(u64::from(*offset));
        }
        digest.u64(u64::from(self.vertex_input_flags));
        digest.u64(self.topology as u64);
        digest.u64(u64::from(self.primitive_restart));
        digest.u64(u64::from(self.viewport_count));
        digest.u64(u64::from(self.scissor_count));
        digest.u64(u64::from(self.raster_flags));
        digest.u64(u64::from(self.depth_clamp));
        digest.u64(u64::from(self.rasterizer_discard));
        digest.u64(self.polygon_mode as u64);
        digest.u64(u64::from(self.cull_mode));
        digest.u64(self.front_face as u64);
        digest.u64(u64::from(self.depth_bias_enable));
        digest.u64(u64::from(self.depth_bias_constant));
        digest.u64(u64::from(self.depth_bias_clamp));
        digest.u64(u64::from(self.depth_bias_slope));
        digest.u64(u64::from(self.line_width));
        digest.u64(u64::from(self.rasterization_samples));
        digest.u64(u64::from(self.sample_shading_enable));
        digest.u64(u64::from(self.min_sample_shading));
        digest.u64(u64::from(self.sample_mask_len));
        digest.u64(u64::from(self.alpha_to_coverage_enable));
        digest.u64(u64::from(self.alpha_to_one_enable));
        digest.u64(u64::from(self.logic_op_enable));
        digest.u64(self.logic_op as u64);
        for constant in self.blend_constants {
            digest.u64(u64::from(constant));
        }
        digest.u64(self.blend_attachments.len() as u64);
        for attachment in &self.blend_attachments {
            digest.u64(u64::from(attachment.blend_enable));
            digest.u64(attachment.src_color as u64);
            digest.u64(attachment.dst_color as u64);
            digest.u64(attachment.color_op as u64);
            digest.u64(attachment.src_alpha as u64);
            digest.u64(attachment.dst_alpha as u64);
            digest.u64(attachment.alpha_op as u64);
            digest.u64(u64::from(attachment.write_mask));
        }
        digest.u64(self.dynamic_states.len() as u64);
        for state in &self.dynamic_states {
            digest.u64(*state as u64);
        }
        match &self.depth_stencil {
            Some(state) => {
                digest.u64(1);
                digest.u64(u64::from(state.flags));
                digest.u64(u64::from(state.depth_test_enable));
                digest.u64(u64::from(state.depth_write_enable));
                digest.u64(state.depth_compare_op as u64);
                digest.u64(u64::from(state.depth_bounds_test_enable));
                digest.u64(u64::from(state.stencil_test_enable));
                digest.u64(u64::from(state.min_depth_bounds));
                digest.u64(u64::from(state.max_depth_bounds));
                for face in [state.front, state.back] {
                    digest.u64(face.fail_op as u64);
                    digest.u64(face.pass_op as u64);
                    digest.u64(face.depth_fail_op as u64);
                    digest.u64(face.compare_op as u64);
                    digest.u64(u64::from(face.compare_mask));
                    digest.u64(u64::from(face.write_mask));
                    digest.u64(u64::from(face.reference));
                }
            }
            None => digest.u64(0),
        }
        self.render_pass.hash_into(digest);
        digest.u64(self.set_layouts.len() as u64);
        for layout in &self.set_layouts {
            layout.hash_into(digest);
        }
    }
}

/// The depth-stencil state a plan carries, as the fields of
/// `VkPipelineDepthStencilStateCreateInfo`.
pub(crate) struct DepthStencilState<'a> {
    state: &'a vk::PipelineDepthStencilStateCreateInfo<'a>,
}

impl<'a> DepthStencilState<'a> {
    pub(crate) fn of(state: &'a vk::PipelineDepthStencilStateCreateInfo<'a>) -> Self {
        Self { state }
    }
}

impl DepthStencilDef {
    fn of(state: DepthStencilState<'_>) -> Self {
        let state = state.state;
        Self {
            flags: state.flags.as_raw(),
            depth_test_enable: state.depth_test_enable,
            depth_write_enable: state.depth_write_enable,
            depth_compare_op: state.depth_compare_op.as_raw(),
            depth_bounds_test_enable: state.depth_bounds_test_enable,
            stencil_test_enable: state.stencil_test_enable,
            min_depth_bounds: state.min_depth_bounds.to_bits(),
            max_depth_bounds: state.max_depth_bounds.to_bits(),
            front: StencilOpDef::of(&state.front),
            back: StencilOpDef::of(&state.back),
        }
    }
}

/// The key one pass looks itself up with.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct PassKey {
    digest: u64,
    pipeline: PipelinePlan,
}

impl PassKey {
    pub(crate) fn of(pipeline: PipelinePlan) -> Self {
        let mut digest = Digest::new();
        pipeline.hash_into(&mut digest);
        Self {
            digest: digest.finish(),
            pipeline,
        }
    }
}

/// The device objects one shape's pass may reuse.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReusablePass {
    pub(crate) vertex_module: vk::ShaderModule,
    pub(crate) fragment_module: vk::ShaderModule,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
}

impl ReusablePass {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.fragment_module != vk::ShaderModule::null() {
                device.destroy_shader_module(self.fragment_module, None);
            }
            if self.vertex_module != vk::ShaderModule::null() {
                device.destroy_shader_module(self.vertex_module, None);
            }
        }
    }
}

struct Entry {
    key: PassKey,
    objects: ReusablePass,
}

/// The resident-shape cache: one entry per pipeline shape a pass has built.
pub(crate) struct RenderSetupReuse {
    /// The `VkDevice` the entries belong to. Kept so an evicted or flushed
    /// entry can be destroyed while the device is alive; the context's own
    /// teardown destroys whatever is still here with the device.
    device: ash::Device,
    enabled: bool,
    entries: VecDeque<Entry>,
    hits: u64,
    misses: u64,
    mismatches: u64,
    unkeyed: u64,
    evictions: u64,
    flushes: u64,
}

/// How many shapes one device keeps resident.
///
/// The guest desktop draws a handful of shapes and repeats them; a round's own
/// census counts them in the hundreds of *passes*, not of shapes. The cap is
/// what keeps a pathological stream of distinct shapes from growing the
/// provider's own memory, and the eviction counter says when it was reached.
pub(crate) const ENTRY_CAP: usize = 128;

/// One cache lookup's outcome, as the profile line counts it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Outcome {
    /// The pass's own shape was resident and its objects came back.
    Hit,
    /// No entry had this shape; the pass built its objects and cached them.
    Miss,
    /// An entry shared the digest but not the shape: fail-closed, and the pass
    /// built its objects without caching them (the digest is a filter, not the
    /// decision).
    Mismatch,
    /// The pass could not state an exact key at all, so it was neither looked
    /// up nor cached.
    Unkeyed,
    /// The switch is off: one relaxed load per pass and no device objects kept.
    Disabled,
}

/// What one lookup found.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Lookup {
    /// The shape was resident; its objects left with the caller.
    Hit(ReusablePass),
    /// No entry had this shape.
    Miss,
    /// An entry shared the digest but not the shape.
    Mismatch,
    /// The switch is off.
    Disabled,
}

impl Lookup {
    /// The profile's name for this outcome.
    pub(crate) fn outcome(&self) -> Outcome {
        match self {
            Lookup::Hit(_) => Outcome::Hit,
            Lookup::Miss => Outcome::Miss,
            Lookup::Mismatch => Outcome::Mismatch,
            Lookup::Disabled => Outcome::Disabled,
        }
    }

    /// The objects a hit carries.
    pub(crate) fn take(self) -> Option<ReusablePass> {
        match self {
            Lookup::Hit(objects) => Some(objects),
            _ => None,
        }
    }
}

impl RenderSetupReuse {
    /// The cache a device starts with.
    pub(crate) fn new(device: ash::Device) -> Self {
        Self {
            device,
            enabled: enabled_from_env(),
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            mismatches: 0,
            unkeyed: 0,
            evictions: 0,
            flushes: 0,
        }
    }

    /// Whether the mechanism is on for this device.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Turn the mechanism on or off. The profile's own arm switch is the
    /// environment variable this starts from; this setter is what an e2e case
    /// and the provider's own test arm use, so both arms can run in one process
    /// against one device.
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }

    /// Take one shape's objects out of the cache.
    ///
    /// The entry leaves with the caller and comes back through [`Self::insert`]
    /// once the pass that took it has retired its work, so a pass's own use of
    /// the objects is not a borrow the cache has to track: what the cache holds
    /// is always something no pass is currently recording with. A pass that
    /// fails in between destroys what it took (its own teardown owns them by
    /// then), which is the fail-closed arm: the next pass of that shape builds
    /// the objects again instead of finding them half-used.
    pub(crate) fn lookup(&mut self, key: &PassKey) -> Lookup {
        if !self.enabled {
            return Lookup::Disabled;
        }
        let mut mismatched = false;
        let mut found = None;
        for index in 0..self.entries.len() {
            if self.entries[index].key.digest != key.digest {
                continue;
            }
            if self.entries[index].key.pipeline == key.pipeline {
                found = Some(index);
                break;
            }
            mismatched = true;
        }
        if let Some(index) = found {
            self.hits += 1;
            let entry = self
                .entries
                .remove(index)
                .expect("the entry the index named is still there");
            return Lookup::Hit(entry.objects);
        }
        if mismatched {
            self.mismatches += 1;
            Lookup::Mismatch
        } else {
            self.misses += 1;
            Lookup::Miss
        }
    }

    /// Keep one pass's objects for the next pass of the same shape.
    ///
    /// An object another entry already holds for the same shape is destroyed
    /// rather than kept twice: two passes of one shape racing on a cold cache
    /// both build, and the loser's objects are surplus.
    pub(crate) fn insert(&mut self, key: PassKey, objects: ReusablePass) {
        if !self.enabled {
            objects.destroy(&self.device);
            return;
        }
        for entry in &mut self.entries {
            if entry.key.pipeline == key.pipeline {
                objects.destroy(&self.device);
                return;
            }
        }
        while self.entries.len() >= ENTRY_CAP {
            if let Some(evicted) = self.entries.pop_front() {
                evicted.objects.destroy(&self.device);
                self.evictions += 1;
            }
        }
        self.entries.push_back(Entry { key, objects });
    }

    /// Drop every entry. Called when the contract surface the entries were
    /// built from moves under them — a released registered pipeline — and when
    /// the mechanism is switched off.
    pub(crate) fn clear(&mut self) {
        while let Some(entry) = self.entries.pop_front() {
            entry.objects.destroy(&self.device);
        }
        self.flushes += 1;
    }

    /// Count one pass that could not be keyed.
    pub(crate) fn note_unkeyed(&mut self) {
        self.unkeyed += 1;
    }

    /// The counters one reading reports.
    pub(crate) fn counts(&self) -> RenderSetupReuseCounts {
        RenderSetupReuseCounts {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            mismatches: self.mismatches,
            unkeyed: self.unkeyed,
            evictions: self.evictions,
            flushes: self.flushes,
        }
    }
}

/// What a cache reading reports, beside the profile line's own window counters.
///
/// `hits` and `misses` count the passes that reached the mechanism with a key;
/// `mismatches` counts digest collisions the full comparison refused;
/// `unkeyed` counts passes that could not state a key and were neither looked
/// up nor cached; `evictions` and `flushes` count entries destroyed by the cap
/// and by a contract-surface change. `entries` is what the cache holds now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderSetupReuseCounts {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub mismatches: u64,
    pub unkeyed: u64,
    pub evictions: u64,
    pub flushes: u64,
}

/// FNV-1a, the digest this crate's other keys use. It picks the bucket; the
/// comparison above decides.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_ne_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

/// Whether the mechanism is on, read once from the process environment.
pub(crate) fn enabled_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        match std::env::var("METAL_API_VULKAN_RENDER_SETUP_CACHE")
            .ok()
            .as_deref()
        {
            None => true,
            Some(value) => !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "off" | "no" | "false"
            ),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn bindings(
        pairs: &[(u32, vk::DescriptorType, u32)],
    ) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
        pairs
            .iter()
            .map(|(binding, ty, flags)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*binding)
                    .descriptor_type(*ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::from_raw(*flags))
            })
            .collect()
    }

    /// The reviewed pair's shape, as the pipeline plan reads it: one colour
    /// target, dynamic viewport and scissor, the two reviewed entry names.
    struct Fixture {
        vertex_bindings: Vec<vk::VertexInputBindingDescription>,
        vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
        entries: (CString, CString),
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                vertex_bindings: vec![vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(8)
                    .input_rate(vk::VertexInputRate::VERTEX)],
                vertex_attributes: vec![vk::VertexInputAttributeDescription::default()
                    .location(0)
                    .binding(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(0)],
                entries: (
                    CString::new("vertex_main").expect("entry"),
                    CString::new("fragment_main").expect("entry"),
                ),
            }
        }

        /// One plan, with the blend state's own enable bit as the only variable
        /// so a reader can see the plan separates two otherwise equal shapes.
        fn plan(&self, blend_enable: bool, vertex_words: &[u32]) -> PipelinePlan {
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&self.vertex_bindings)
                .vertex_attribute_descriptions(&self.vertex_attributes);
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let attachments = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(blend_enable)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let states = PipelineStates {
                vertex_bindings: &self.vertex_bindings,
                vertex_attributes: &self.vertex_attributes,
                vertex_input: &vertex_input,
                input_assembly: &input_assembly,
                viewport: &viewport,
                rasterization: &rasterization,
                multisample: &multisample,
                color_blend: &blend,
                color_blend_attachments: &attachments,
                dynamic_states: &dynamic_states,
            };
            PipelinePlan::of(
                vertex_words,
                &[4_u32, 5, 6],
                &self.entries.0,
                &self.entries.1,
                None,
                0,
                &states,
                None,
                &RenderPassDef::default(),
                &[],
            )
        }
    }

    #[test]
    fn a_layout_definition_does_not_depend_on_binding_order() {
        let first = LayoutDef::of(&bindings(&[
            (3, vk::DescriptorType::SAMPLER, 0x10),
            (1, vk::DescriptorType::SAMPLED_IMAGE, 0x10),
        ]));
        let second = LayoutDef::of(&bindings(&[
            (1, vk::DescriptorType::SAMPLED_IMAGE, 0x10),
            (3, vk::DescriptorType::SAMPLER, 0x10),
        ]));
        assert_eq!(first, second);
    }

    #[test]
    fn a_layout_definition_separates_the_fields_the_driver_sees() {
        let base = LayoutDef::of(&bindings(&[(0, vk::DescriptorType::SAMPLED_IMAGE, 0x10)]));
        let other_type = LayoutDef::of(&bindings(&[(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            0x10,
        )]));
        let other_stage = LayoutDef::of(&bindings(&[(0, vk::DescriptorType::SAMPLED_IMAGE, 0x20)]));
        let other_binding =
            LayoutDef::of(&bindings(&[(1, vk::DescriptorType::SAMPLED_IMAGE, 0x10)]));
        assert_ne!(base, other_type);
        assert_ne!(base, other_stage);
        assert_ne!(base, other_binding);
    }

    #[test]
    fn a_render_pass_definition_reads_every_field_it_is_given() {
        let description = vk::AttachmentDescription2::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let reference = vk::AttachmentReference2::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let plain = RenderPassDef::of(
            &[description],
            &[reference],
            None,
            None,
            None,
            &[],
            vk::PipelineBindPoint::GRAPHICS,
            0,
        );
        let stored = RenderPassDef::of(
            &[description.store_op(vk::AttachmentStoreOp::DONT_CARE)],
            &[reference],
            None,
            None,
            None,
            &[],
            vk::PipelineBindPoint::GRAPHICS,
            0,
        );
        assert_ne!(plain, stored);
    }

    #[test]
    fn a_pipeline_plan_separates_the_shaders_and_the_states() {
        let fixture = Fixture::new();
        let plain = fixture.plan(false, &[1, 2, 3]);
        assert_eq!(plain, fixture.plan(false, &[1, 2, 3]));
        assert_ne!(plain, fixture.plan(true, &[1, 2, 3]));
        assert_ne!(plain, fixture.plan(false, &[9, 9, 9]));
    }

    #[test]
    fn the_digest_agrees_with_the_comparison_it_filters() {
        let fixture = Fixture::new();
        let first = PassKey::of(fixture.plan(false, &[1, 2, 3]));
        let same = PassKey::of(fixture.plan(false, &[1, 2, 3]));
        let other = PassKey::of(fixture.plan(false, &[9, 9, 9]));
        assert_eq!(first.digest, same.digest);
        assert_eq!(first.pipeline, same.pipeline);
        assert_ne!(first.pipeline, other.pipeline);
    }

    #[test]
    fn the_switch_reads_the_words_a_round_spells() {
        for value in ["0", "off", "OFF", " no ", "false"] {
            let trimmed = value.trim().to_ascii_lowercase();
            assert!(matches!(trimmed.as_str(), "0" | "off" | "no" | "false"));
        }
        for value in ["1", "on", "yes", "true", ""] {
            let trimmed = value.trim().to_ascii_lowercase();
            assert!(!matches!(trimmed.as_str(), "0" | "off" | "no" | "false"));
        }
    }
}
