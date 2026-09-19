//! The render sampler's second 8-bit byte order (`research/docs/23` §3.3,
//! §107), the shape census v13 named as its first remaining door: a draw whose
//! `[[texture(0)]]` bind is a `B8G8R8A8_UNORM` guest view.
//!
//! The census sentence (`evidence/gate3-census-v13-2026-09-17/`, 6455 lines /
//! 60.35% of the boot's class exits) states the canonical pass's texture as one
//! single-sample, non-arrayed 2D view with one descriptor, `rgba8_unorm` texels
//! and an identity channel mapping, while the bind is a `B8G8R8A8_UNORM` view —
//! the same four-byte 8-bit UNORM texel in the other byte order. This test is
//! the Vulkan rail's executable half of the widening:
//!
//! * the same fragment stage, whose four `air.sample_texture_2d` calls read
//!   four texel centres and return one whole channel of each, lands the *same*
//!   frame from the two byte-order siblings of one colour pattern — the
//!   `bgra8_unorm` texture's uploaded bytes are the `rgba8_unorm` sibling's
//!   with the red and blue halves swapped, so a rail that uploaded the bytes
//!   under the wrong name would land the swap instead of the colours;
//! * the trace rail and the object rail over one provider land that frame byte
//!   for byte;
//! * another texture moves the frame, so the reading measures the upload rather
//!   than the run;
//! * a format outside the window, an arrayed view and a multisampled view each
//!   keep their named refusal, with the fields in hand;
//! * and the `rgba8_unorm` sibling's own bytes are unmoved (`v70`'s window
//!   still executes exactly as it did).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SamplerAddressMode,
    SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess, TextureBindingContract,
    TextureFormat, TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment, and which forwards no varying at all — the fragment fixture
/// samples at constant texel centres, so it needs none.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const FRAGMENT_ENTRY: &str = "render_sample_texture_2d_bgra_channels";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_2d_bgra_channels.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(940);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(941);
const SCRATCH_VIEW: ViewId = ViewId::new(942);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(943);
const TEXTURE_VIEW: ViewId = ViewId::new(944);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(945);

/// 4x4, the extent of both the attachment and the sampled texture: the rail's
/// window requires the two to agree, and every sample stands on a texel centre.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The sampler state the fragment fixture's AIR carries (nearest +
/// clamp-to-edge), the state the registration has to repeat.
const MODULE_SAMPLER: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// Which colour pattern the fixture's four sampled texels carry: the same
/// colours in two byte orders, and then a second pattern that moves them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    /// Texel `(i, j)` is the colour `(0x10 + 0x10i, 0x50 + 0x10j, 0x90,
    /// 0xc0 + 0x10(i + j))`.
    Primary,
    /// The same shape with the red, green and blue bases moved, so the frame's
    /// first three bytes all change.
    Moved,
}

/// The colour the four sampled texels carry under one pattern, in the
/// fragment stage's own channel order.
fn colour(pattern: Pattern, i: u32, j: u32) -> [u8; 4] {
    let (red, green, blue) = match pattern {
        Pattern::Primary => (0x10 + 0x10 * i, 0x50 + 0x10 * j, 0x90),
        Pattern::Moved => (0x20 + 0x10 * i, 0x30 + 0x10 * j, 0x70),
    };
    [
        red as u8,
        green as u8,
        blue as u8,
        (0xc0 + 0x10 * (i + j)) as u8,
    ]
}

/// The frame the fixture lands: red is texel (0, 0)'s red, green is texel
/// (1, 0)'s green, blue is texel (0, 1)'s blue and alpha is texel (1, 1)'s
/// alpha — the four sampled channels of the pattern.
fn expected_frame(pattern: Pattern) -> [u8; 4] {
    [
        colour(pattern, 0, 0)[0],
        colour(pattern, 1, 0)[1],
        colour(pattern, 0, 1)[2],
        colour(pattern, 1, 1)[3],
    ]
}

/// The tightly packed texel bytes one format's memory layout spells for one
/// colour: `rgba8_unorm` is `(r, g, b, a)`, `bgra8_unorm` is `(b, g, r, a)`.
fn texel_bytes(format: TextureFormat, colour: [u8; 4]) -> [u8; 4] {
    match format {
        TextureFormat::Rgba8Unorm => colour,
        TextureFormat::Bgra8Unorm => [colour[2], colour[1], colour[0], colour[3]],
        other => panic!("the fixture's colours are 8-bit four-component; not {other:?}"),
    }
}

/// The 4x4 texture the fixture samples, in the memory layout the view names.
fn texture_bytes(format: TextureFormat, pattern: Pattern) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((EXTENT * EXTENT * 4) as usize);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            bytes.extend_from_slice(&texel_bytes(format, colour(pattern, i, j)));
        }
    }
    bytes
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest")
}

fn executor_and_provider() -> Option<(Arc<VulkanExecutor>, Arc<VulkanComputeProvider>)> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let provider = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context"),
    );
    Some((executor, provider))
}

/// Translate one fragment fixture the way a host feeding guest AIR would, and
/// report what the module states about its own texture binding.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_air: &str,
    entry: &str,
) -> (TranslatedRenderStage, TranslatedRenderStage) {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(VERTEX_AIR)
        .expect("the vertex fixture loads");
    let function = library
        .function(VERTEX_ENTRY)
        .expect("the vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &function)
        .expect("the vertex stage translates");
    let library = device
        .new_library_with_air(fragment_air)
        .expect("the fragment fixture loads");
    let function = library.function(entry).expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated {entry}: {} bytes, reflection entry {:?}, bindings {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
        fragment
            .reflection()
            .bindings
            .iter()
            .map(|binding| (
                format!("{:?}", binding.kind),
                binding.metal_index,
                binding.descriptor.map(|descriptor| (
                    descriptor.set,
                    descriptor.binding,
                    descriptor.count
                )),
                binding.access.map(|access| format!("{access:?}")),
            ))
            .collect::<Vec<_>>(),
    );
    (vertex, fragment)
}

fn compile_declaring_kernel(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    provider
        .compile_pipeline(&function, digest(b"bgra-texture-compute"))
        .expect("the compute pipeline registers")
}

/// The contract a sampled-texture registration states: the fixture's own two
/// entries, one `rgba8_unorm` attachment, no vertex stream, and one sampled
/// texture at `[[texture(0)]]` whose format is the caller's.
fn contract(format: TextureFormat) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract::sampled(0, format, MODULE_SAMPLER)],
    }
}

fn sampled_texture_view(format: TextureFormat, pattern: Pattern) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(texture_bytes(format, pattern)),
    }
}

fn render_pass(pipeline: PipelineId, textures: Vec<TextureView>) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
        stage_buffers: Vec::new(),
        blend: None,
        cull: None,
        depth: None,
        depth_test: None,
        depth_resolve: None,
        multisample: None,
        stencil: None,
        stencil_test: None,
        stencil_resolve: None,
        base_vertex: 0,
        pipeline,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(EXTENT),
            height: u64::from(EXTENT),
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, EXTENT, EXTENT],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures,
        present: None,
    }
}

/// The trace the declaring compute pass and the sampling pass share
/// (`research/docs/23` §3.6): the compute pass states the attachment's own
/// bytes, so the readback is the pass's declaration rather than a driver's
/// initial contents.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(41),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: compute.pipeline_id,
                buffers: vec![
                    BufferView {
                        view_id: ATTACHMENT_VIEW,
                        metal_binding: 0,
                        allocation_id: ATTACHMENT_ALLOCATION,
                        offset: 0,
                        length: attachment_bytes,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(
                            ATTACHMENT_WORD.repeat(attachment_bytes as usize / 4),
                        ),
                    },
                    BufferView {
                        view_id: SCRATCH_VIEW,
                        metal_binding: 1,
                        allocation_id: SCRATCH_ALLOCATION,
                        offset: 0,
                        length: 4,
                        access: BufferAccess::Write,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0xab; 4]),
                    },
                ],
                textures: Vec::new(),
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: [1, 1, 1],
                    threads_per_threadgroup: [1, 1, 1],
                },
            }),
            TracePass::Render(render_pass(render.pipeline_id, textures)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: attachment_bytes,
        })
        .expect("attachment allocation");
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: SCRATCH_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 8,
        })
        .expect("scratch allocation");
    (trace, resources)
}

/// The frame the trace rail lands for one texture, or the refusal that stopped
/// it before the device ever ran.
fn trace_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    texture: TextureView,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, vec![texture]);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)?;
    let submitted = provider.submit(admitted)?;
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    Ok(submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback"))
}

/// The frame the object rail lands for the same registration and texture: the
/// object API's own command buffer over the very provider the trace rail used.
fn object_readback(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    format: TextureFormat,
    pattern: Pattern,
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the registration wraps for the object API");
    // The declaring pass and the two buffers the trace rail's own trace
    // carries: the attachment's bytes are the pass's own declaration, exactly
    // as the trace rail's declaring compute pass is, and the scratch write is
    // what makes the declaration observable.
    let attachment = device
        .new_buffer_with_bytes(vec![0xfe; 64])
        .expect("the attachment buffer is declared");
    let view = attachment
        .view(0, (u64::from(EXTENT) * u64::from(EXTENT) * 4) as usize)
        .expect("the attachment view is declared");
    let scratch = device
        .new_buffer_with_bytes(vec![0xab; 4])
        .expect("the scratch buffer is declared");
    let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
    let texture = device
        .new_texture_with_bytes(
            format,
            u64::from(EXTENT),
            u64::from(EXTENT),
            texture_bytes(format, pattern),
        )
        .expect("the sampled texture is declared");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"bgra-texture-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(&declaring)
            .expect("compute pipeline state");
        encoder
            .set_buffer(0, &view)
            .expect("the attachment's bytes are the pass's own declaration");
        encoder
            .set_buffer(1, &scratch_view)
            .expect("the kernel's write slot");
        encoder
            .dispatch_threads(
                Size::new(1, 1, 1).expect("grid"),
                Size::new(1, 1, 1).expect("local size"),
            )
            .expect("the declaring dispatch records");
        encoder.end_encoding().expect("end encoding");
    }
    {
        let mut render_encoder = command
            .render_command_encoder()
            .expect("the render encoder opens");
        render_encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the sampling pipeline is bound");
        render_encoder
            .set_fragment_texture(0, &texture)
            .expect("the sampled texture is bound at its own index");
        render_encoder
            .draw_render_pass(
                &view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the full-screen triangle is drawn");
        render_encoder
            .end_encoding()
            .expect("the render encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    attachment
        .read()
        .expect("the attachment bytes are readable")
}

/// One texel of the readback: the fixture reads the same four texels for every
/// fragment, so the whole attachment carries one colour.
fn uniform_texel(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(
        bytes.len() as u64,
        u64::from(EXTENT) * u64::from(EXTENT) * 4
    );
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(
            chunk,
            texel,
            "every fragment samples the same four texels, so every texel has to carry the same \
             colour: {}",
            hex(bytes)
        );
    }
    texel
}

fn register(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    format: TextureFormat,
    what: &str,
) -> CompiledComputePipeline {
    let (vertex, fragment) = translated_pair(executor, FRAGMENT_AIR, FRAGMENT_ENTRY);
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(format),
            vertex,
            fragment,
            logical_digest: digest(what.as_bytes()),
        })
        .expect("the sampled declaration registers")
}

/// Reading 1 (`research/docs/23` §107): the BGRA8 bind enters the provider, the
/// trace rail and the object rail land byte-identical frames, and the frame is
/// the *colours* the guest's view states — the same frame the `rgba8_unorm`
/// sibling of the same module lands from the red/blue-swapped bytes.
#[test]
fn the_bgra8_bind_executes_and_the_two_rails_land_the_same_colours() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let bgra = register(
        &provider,
        &executor,
        TextureFormat::Bgra8Unorm,
        "bgra texture",
    );
    let rgba = register(
        &provider,
        &executor,
        TextureFormat::Rgba8Unorm,
        "rgba sibling",
    );

    let expected = expected_frame(Pattern::Primary);
    let uploaded = texture_bytes(TextureFormat::Bgra8Unorm, Pattern::Primary);
    eprintln!(
        "bgra8 upload: texel (0,0) {}, whole texture {} bytes; expected frame {}",
        hex(&uploaded[0..4]),
        uploaded.len(),
        hex(&expected)
    );
    assert_ne!(
        uploaded[0..4],
        expected[..],
        "the uploaded BGRA8 bytes are not the frame's RGBA bytes, so the reading measures the \
         view's own byte order"
    );

    let trace = trace_readback(
        &provider,
        &compute,
        &bgra,
        sampled_texture_view(TextureFormat::Bgra8Unorm, Pattern::Primary),
    )
    .expect("the BGRA8 bind executes");
    let objects = object_readback(
        &provider,
        &bgra,
        TextureFormat::Bgra8Unorm,
        Pattern::Primary,
    );
    eprintln!(
        "frames: trace {} object {} (expected {})",
        hex(&trace),
        hex(&objects),
        hex(&expected)
    );
    assert_eq!(
        uniform_texel(&trace),
        expected,
        "trace rail: {}",
        hex(&trace)
    );
    assert_eq!(
        uniform_texel(&objects),
        expected,
        "object rail: {}",
        hex(&objects)
    );
    assert_eq!(trace, objects, "the two rails land one frame");

    // The sibling: the same colours in the other memory layout. Its uploaded
    // bytes are the BGRA8 texture's with the red and blue halves swapped, and
    // the frame is the same — which is what "the byte order is the view's own
    // fact, the colours are the observation" means in bytes.
    let sibling_upload = texture_bytes(TextureFormat::Rgba8Unorm, Pattern::Primary);
    for (bgra_texel, rgba_texel) in uploaded.chunks_exact(4).zip(sibling_upload.chunks_exact(4)) {
        assert_eq!(
            [bgra_texel[2], bgra_texel[1], bgra_texel[0], bgra_texel[3]],
            [rgba_texel[0], rgba_texel[1], rgba_texel[2], rgba_texel[3]],
            "the two siblings differ by the red/blue swap alone"
        );
    }
    let sibling = trace_readback(
        &provider,
        &compute,
        &rgba,
        sampled_texture_view(TextureFormat::Rgba8Unorm, Pattern::Primary),
    )
    .expect("the rgba8 sibling executes");
    assert_eq!(
        uniform_texel(&sibling),
        expected,
        "the rgba8 sibling lands the same colours: {}",
        hex(&sibling)
    );
}

/// Reading 2 (`research/docs/23` §107): another texture moves the frame, so the
/// reading is the upload and not the run — and the frame moves on both rails.
#[test]
fn another_texture_moves_the_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let bgra = register(
        &provider,
        &executor,
        TextureFormat::Bgra8Unorm,
        "bgra texture",
    );

    let primary = trace_readback(
        &provider,
        &compute,
        &bgra,
        sampled_texture_view(TextureFormat::Bgra8Unorm, Pattern::Primary),
    )
    .expect("the primary texture executes");
    let moved = trace_readback(
        &provider,
        &compute,
        &bgra,
        sampled_texture_view(TextureFormat::Bgra8Unorm, Pattern::Moved),
    )
    .expect("the moved texture executes");
    let moved_objects =
        object_readback(&provider, &bgra, TextureFormat::Bgra8Unorm, Pattern::Moved);

    let expected_moved = expected_frame(Pattern::Moved);
    eprintln!(
        "moved frames: trace {} object {} (expected {})",
        hex(&moved),
        hex(&moved_objects),
        hex(&expected_moved)
    );
    assert_eq!(
        uniform_texel(&moved),
        expected_moved,
        "moved trace rail: {}",
        hex(&moved)
    );
    assert_eq!(
        uniform_texel(&moved_objects),
        expected_moved,
        "moved object rail: {}",
        hex(&moved_objects)
    );
    assert_eq!(moved, moved_objects, "the two rails land one frame");
    assert_ne!(
        uniform_texel(&primary),
        expected_moved,
        "the moved texture has to move the frame: {} vs {}",
        hex(&primary),
        hex(&moved)
    );
    eprintln!(
        "frames: primary {}, moved {} (expected {})",
        hex(&primary),
        hex(&moved),
        hex(&expected_moved)
    );
}

/// Reading 3 (`research/docs/23` §3.3, §107): the shapes beside the widened
/// window keep their named refusals, with the fields in hand.
#[test]
fn the_shapes_beside_the_window_keep_their_named_refusals() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let bgra = register(
        &provider,
        &executor,
        TextureFormat::Bgra8Unorm,
        "bgra texture",
    );

    // Another format: the single-component `r32_float` texel, the lane the
    // window never widened to (`rgba16_float` joined it with the eight-byte
    // lane). The registration's own format walk refuses it before any device
    // object exists — the declaration names a format this rail does not
    // upload — and the fields name the binding and the format.
    let (vertex, fragment) = translated_pair(&executor, FRAGMENT_AIR, FRAGMENT_ENTRY);
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFormat::R32Float),
            vertex,
            fragment,
            logical_digest: digest(b"wide texture"),
        })
        .expect_err("the rail names the sampled lanes and no other format");
    eprintln!("wide format refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_format_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("format"),
        Some(&FieldValue::Text("R32Float".to_owned()))
    );
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );

    // An arrayed view: the declaration repeats the view's own type (the pair
    // rules hold the two to each other) and the rail refuses the shape by
    // name once the trace reaches it.
    let arrayed = sampled_texture_view(TextureFormat::Bgra8Unorm, Pattern::Primary);
    let mut arrayed_contract = contract(TextureFormat::Bgra8Unorm);
    arrayed_contract.textures[0].texture_type = TextureType::D2Array;
    let mut arrayed_view = arrayed;
    arrayed_view.texture_type = TextureType::D2Array;
    arrayed_view.array_length = 4;
    arrayed_view.source = TextureSource::OwnedBytes(
        texture_bytes(TextureFormat::Bgra8Unorm, Pattern::Primary).repeat(4),
    );
    let (vertex, fragment) = translated_pair(&executor, FRAGMENT_AIR, FRAGMENT_ENTRY);
    let arrayed_render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: arrayed_contract,
            vertex,
            fragment,
            logical_digest: digest(b"arrayed texture"),
        })
        .expect("the arrayed declaration is structurally valid");
    let (trace, resources) = trace_for(&provider, &compute, &arrayed_render, vec![arrayed_view]);
    let admitted = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect("the snapshot's format list says nothing about the shape");
    let refused = provider
        .submit(admitted)
        .expect_err("the rail refuses an arrayed sampled view by name");
    eprintln!("arrayed view refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_shape_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("texture_type"),
        Some(&FieldValue::Text("D2Array".to_owned()))
    );
    assert_eq!(
        refused.fields.get("array_length"),
        Some(&FieldValue::Unsigned(4))
    );
    assert_eq!(
        refused.fields.get("sample_count"),
        Some(&FieldValue::Unsigned(1))
    );

    // A multisampled view: the contract's own binding rule refuses it before
    // admission, under the shape slug the texture views' shared validation
    // owns.
    let mut multisampled = sampled_texture_view(TextureFormat::Bgra8Unorm, Pattern::Primary);
    multisampled.texture_type = TextureType::D2Multisample;
    multisampled.sample_count = 4;
    // A multisampled view's own byte extent counts its samples, so the source
    // has to be whole before the contract's sample-count rule can be the one
    // that answers.
    multisampled.source = TextureSource::OwnedBytes(vec![0x5a; (EXTENT * EXTENT * 4 * 4) as usize]);
    let (trace, resources) = trace_for(&provider, &compute, &bgra, vec![multisampled]);
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a multisampled sampled texture is not a texel the sampler can reduce");
    eprintln!("multisampled view refused: {refused:?}");
    assert_eq!(refused.slug, "texture_shape_mismatch");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    assert!(
        refused
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("carries 4 samples")),
        "the refusal names the sample count it read: {refused:?}"
    );
}
