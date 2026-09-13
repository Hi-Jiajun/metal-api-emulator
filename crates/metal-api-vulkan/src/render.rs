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
//! flipping `ProviderCapabilities::supports_render_passes`. Nothing on the trace
//! path calls this rail yet, so the non-test build allows `dead_code` here
//! instead of carrying a `pub(crate)` surface that no production path reaches.
#![cfg_attr(not(test), allow(dead_code))]

use ash::vk;
use metal_api_core::provider::{
    AttachmentFormat, ClearColor, FieldValue, ProviderError, ProviderErrorClass, ProviderPhase,
    Retryability,
};

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

/// Vertex stage: positions from `gl_VertexIndex`, no vertex buffers, no
/// varyings. `spirv-as` output of `render_spv/fullscreen_triangle.vert.spvasm`,
/// which selects `(-1,-1) (3,-1) (-1,3)` — the oversize triangle covers every
/// pixel centre of a 2×2 viewport, so "the draw really ran" stays falsifiable
/// (`research/docs/23` §1.3).
const FULL_SCREEN_TRIANGLE_VERT_SPV: &[u8] =
    include_bytes!("render_spv/fullscreen_triangle.vert.spv");

/// Fragment stage: writes `(64/255, 128/255, 192/255, 1)`, which an 8-bit UNORM
/// attachment stores as `40 80 c0 ff`.
///
/// The constants are byte/255 rather than the round decimals `0.25/0.5/0.75`
/// on purpose: `0.5 * 255 = 127.5` is a half-integer tie, and the probe read
/// `0x80` back on Lavapipe but `0x7f` on both the NVIDIA driver and dzn
/// (`research/docs/23` §3.5). Byte/255 values sit at least 3.7e-6 away from a
/// tie on every driver, so they are the parity-stable discipline the fixture
/// has to follow.
const SOLID_RGBA8_FRAG_SPV: &[u8] = include_bytes!("render_spv/solid_rgba8.frag.spv");

/// One offscreen render pass to execute.
///
/// The shape mirrors `metal_api_core::provider::RenderPassDescriptor` for the
/// fields this rail consumes: the core type carries wiring identities
/// (pipeline/view/allocation ids and a resolved byte source) that Step 3c maps,
/// while Step 3b fixes the Vulkan-side execution against an already-chosen
/// format, extent, clear value and shader pair.
pub(crate) struct OffscreenRenderRequest<'a> {
    /// Colour attachment format, in render-contract terms.
    pub format: AttachmentFormat,
    /// Attachment extent in texels. The milestone fixes 2×2 (`docs/23` §1.3) so
    /// full coverage is distinguishable from a single stored texel.
    pub extent: [u32; 2],
    /// The `LoadOp::Clear` value. Carried as bytes for the same reason the
    /// contract carries bytes: a float clear is not parity-stable
    /// (`research/docs/23` §3.5).
    pub clear: ClearColor,
    /// Vertex-stage SPIR-V module.
    pub vertex_spirv: &'a [u8],
    /// Fragment-stage SPIR-V module.
    pub fragment_spirv: &'a [u8],
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
/// Contract format admission, the device's `COLOR_ATTACHMENT` bit and the
/// `TRANSFER_SRC` bit the readback needs all run before the first
/// `vkCreateImage`, so an unsupported request is refused instead of being
/// handed to the driver.
pub(crate) fn execute_offscreen_render(
    context: &VulkanContext,
    request: &OffscreenRenderRequest<'_>,
) -> Result<Vec<u8>, ProviderError> {
    let format = attachment_vk_format(request.format)?;
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
    let vertex_words = spirv_words(request.vertex_spirv)
        .ok_or_else(|| spirv_refusal("vertex SPIR-V is empty or not a multiple of four bytes"))?;
    let fragment_words = spirv_words(request.fragment_spirv)
        .ok_or_else(|| spirv_refusal("fragment SPIR-V is empty or not a multiple of four bytes"))?;

    let mut objects = OffscreenObjects::new(context);
    objects.create_attachment(format, width, height)?;
    objects.create_render_pass(format)?;
    objects.create_framebuffer(width, height)?;
    objects.create_pipeline(&vertex_words, &fragment_words)?;
    let readback_mapping = objects.create_readback(byte_length)?;
    objects.create_command_pool(queue_index)?;
    objects.record(request.clear, width, height)?;
    objects.submit_and_wait(queue_index)?;

    let texels = unsafe {
        std::slice::from_raw_parts(readback_mapping as *const u8, byte_length as usize).to_vec()
    };
    context.record_buffer_readback();
    context.record_buffer_readback_bytes(texels.len());
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
    /// them. `finalLayout = TRANSFER_SRC_OPTIMAL` is what lets the copy run
    /// without a further layout transition (`research/docs/23` §7.1).
    fn create_render_pass(&mut self, format: vk::Format) -> Result<(), ProviderError> {
        let attachments = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
        let color_refs = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)];
        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ),
        ];
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
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_module)
                .name(c"main"),
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
    fn record(&mut self, clear: ClearColor, width: u32, height: u32) -> Result<(), ProviderError> {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command, &begin)
        }
        .map_err(|error| execution_refusal("begin command buffer", &error.to_string()))?;

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: clear.bytes.map(|byte| f32::from(byte) / 255.0),
            },
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
        let queue = *self
            .context
            .queues
            .get(queue_index)
            .ok_or_else(|| execution_refusal("submit render pass", "queue index is unknown"))?;
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
        unsafe {
            self.context
                .device
                .queue_submit(queue, &submits, self.fence)
        }
        .map_err(|error| submission_refusal("submit render pass", &error.to_string()))?;
        self.context.record_queue_submission(queue_index);
        unsafe {
            self.context
                .device
                .wait_for_fences(&[self.fence], true, crate::FENCE_TIMEOUT_NS)
        }
        .map_err(|error| submission_refusal("wait for render fence", &error.to_string()))?;
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
            if self.view != vk::ImageView::null() {
                self.context.device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                self.context.device.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.context.device.free_memory(self.memory, None);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The readback a 2×2 `R8G8B8A8_UNORM` attachment must hold when the
    /// fragment shader stores `64/255, 128/255, 192/255, 1`.
    const EXPECTED_TEXELS: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

    /// The `LoadOp::Clear` sentinel (`research/docs/23` §1.3): a texel that
    /// still holds it proves the draw did not cover that pixel.
    const CLEAR_SENTINEL: u8 = 0xfe;

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
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV,
            fragment_spirv: SOLID_RGBA8_FRAG_SPV,
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("R32Uint is refused before any Vulkan object exists");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        // No copy in either direction: the refusal precedes vkCreateImage.
        assert_eq!(context.buffer_copy_counts(), (0, 0));
    }

    #[test]
    fn offscreen_full_screen_triangle_reads_back_the_expected_texels() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            format: AttachmentFormat::Rgba8Unorm,
            extent: [2, 2],
            clear: ClearColor::new([CLEAR_SENTINEL; 4]),
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV,
            fragment_spirv: SOLID_RGBA8_FRAG_SPV,
        };
        let (uploads_before, readbacks_before) = context.buffer_copy_counts();
        let texels =
            execute_offscreen_render(&context, &request).expect("the 2x2 render pass executes");
        let (uploads_after, readbacks_after) = context.buffer_copy_counts();

        eprintln!("readback texels: {}", hex(&texels));
        assert_eq!(texels.len(), 16);
        assert_eq!(texels, EXPECTED_TEXELS.repeat(4));
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

    #[test]
    fn offscreen_render_refuses_a_zero_extent() {
        let Some(context) = device_context() else {
            return;
        };
        let request = OffscreenRenderRequest {
            format: AttachmentFormat::Rgba8Unorm,
            extent: [2, 0],
            clear: ClearColor::new([CLEAR_SENTINEL; 4]),
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV,
            fragment_spirv: SOLID_RGBA8_FRAG_SPV,
        };
        let refused = execute_offscreen_render(&context, &request)
            .expect_err("a zero-dimension attachment is a contract refusal");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);
    }
}
