//! The render sampler's narrow channel lanes (`research/docs/23` §3.3, §113),
//! the shape census v25b/v26 named as a contract-level door: a draw whose
//! `[[texture(0)]]` bind is an `R8_UNORM` guest view.
//!
//! The census sentence (`evidence/gate3-census-v25b-2026-09-18/`, the
//! `texture_bind` bucket) states the canonical pass's texture as one
//! single-sample, non-arrayed 2D view with one descriptor and an identity
//! channel mapping, while the bind is one `R8_UNORM` texel — the same
//! *sampled-format* question one byte wide instead of four. This test is the
//! Vulkan rail's executable half of the widening, and it is deliberately
//! sharper than "the frame changed":
//!
//! * the fragment stage reads one whole channel of each of four texels, so a
//!   texture whose sixteen single-byte texels are all different makes every
//!   channel of the frame falsifiable at once — red is texel (0,0)'s byte and
//!   green/blue are the *fill* channels the format does not carry, which a
//!   rail that logged the byte into the wrong lane would land elsewhere;
//! * the trace rail and the object rail over one provider land that frame byte
//!   for byte;
//! * another texture moves the frame, so the reading measures the upload rather
//!   than the run;
//! * the `rgba8_unorm` sibling of the *same* geometry lands the same red byte
//!   with its own three channels beside it, so the narrow lane is the same
//!   texel vocabulary rather than a second colour space;
//! * and the eight-byte format the render sampler still does not admit keeps
//!   its named refusal.

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

/// The same fragment stage the byte-order increment reads through: it samples
/// four texels of its own texture and returns one *whole channel* of each
/// (`red = (0,0).x`, `green = (1,0).y`, `blue = (0,1).z`, `alpha = (1,1).w`).
/// The narrow lane's reading uses the first three of those to show what the
/// format carries and what the API fills.
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

/// Which bytes the fixture's sixteen narrow texels carry: one pattern whose
/// every texel is distinct, and a second that moves them all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    /// Texel `(i, j)` is the byte `0x10 * (1 + i + 4 * j)`, so no two texels of
    /// the 4x4 surface share a value and every sampled channel is
    /// distinguishable from every fill channel.
    Primary,
    /// The same shape one step up: `0x20 + 0x10 * (i + 4 * j)`.
    Moved,
}

impl Pattern {
    /// The byte one texel of the 4x4 surface carries under this pattern.
    fn byte(self, i: u32, j: u32) -> u8 {
        match self {
            Self::Primary => (0x10 * (1 + i + 4 * j)) as u8,
            Self::Moved => (0x20 + 0x10 * (i + 4 * j)) as u8,
        }
    }
}

/// The frame a `r8_unorm` texture lands under one pattern: red is texel (0,0)'s
/// own byte, while green and blue are the channels the format does not carry —
/// Vulkan's sampling rule fills them with zero — and alpha is the same rule's
/// one. Every fragment samples the same four texels, so the whole attachment
/// carries this one colour.
fn expected_frame(pattern: Pattern) -> [u8; 4] {
    [pattern.byte(0, 0), 0x00, 0x00, 0xff]
}

/// The 4x4 texture the fixture samples, in the memory layout the view names:
/// one tightly packed byte per texel for `r8_unorm`, four for `rgba8_unorm`.
fn texture_bytes(format: TextureFormat, pattern: Pattern) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity((EXTENT * EXTENT) as usize * format.bytes_per_texel() as usize);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let byte = pattern.byte(i, j);
            match format {
                TextureFormat::R8Unorm => bytes.push(byte),
                TextureFormat::Rgba8Unorm => bytes.extend_from_slice(&[byte, 0x00, 0x00, 0xff]),
                other => {
                    panic!("the fixture's textures are one and four byte texels; not {other:?}")
                }
            }
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

/// Translate one fragment fixture the way a host feeding guest AIR would.
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
        "translated {entry}: {} bytes, reflection entry {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
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
        .compile_pipeline(&function, digest(b"narrow-texture-compute"))
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
                logical_digest: digest(b"narrow-texture-object-declaring"),
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

/// Reading 1 (`research/docs/23` §113): the R8 bind enters the provider, the
/// trace rail and the object rail land byte-identical frames, and the frame is
/// the narrow texel's own byte in red with the format's fill in the other three
/// lanes.
#[test]
fn the_r8_bind_executes_and_the_two_rails_land_the_same_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let narrow = register(&provider, &executor, TextureFormat::R8Unorm, "r8 texture");

    let expected = expected_frame(Pattern::Primary);
    let uploaded = texture_bytes(TextureFormat::R8Unorm, Pattern::Primary);
    eprintln!(
        "r8 upload: {} bytes, texel (0,0) {}, texel (1,0) {}; expected frame {}",
        uploaded.len(),
        uploaded[0],
        uploaded[1],
        hex(&expected)
    );
    assert_eq!(uploaded.len(), (EXTENT * EXTENT) as usize);
    assert_eq!(
        uploaded[4],
        Pattern::Primary.byte(0, 1),
        "the upload is tightly packed, one byte per texel"
    );
    assert_ne!(
        uploaded[0], 0x00,
        "the red lane has to carry a value the fill lanes do not"
    );

    let trace = trace_readback(
        &provider,
        &compute,
        &narrow,
        sampled_texture_view(TextureFormat::R8Unorm, Pattern::Primary),
    )
    .expect("the R8 bind executes");
    let objects = object_readback(&provider, &narrow, TextureFormat::R8Unorm, Pattern::Primary);
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
}

/// Reading 2 (`research/docs/23` §113): another texture moves the frame, so the
/// reading is the upload and not the run — the fill lanes cannot move, because
/// nothing in the source states them.
#[test]
fn another_narrow_texture_moves_the_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let narrow = register(&provider, &executor, TextureFormat::R8Unorm, "r8 texture");

    let primary = trace_readback(
        &provider,
        &compute,
        &narrow,
        sampled_texture_view(TextureFormat::R8Unorm, Pattern::Primary),
    )
    .expect("the primary texture executes");
    let moved = trace_readback(
        &provider,
        &compute,
        &narrow,
        sampled_texture_view(TextureFormat::R8Unorm, Pattern::Moved),
    )
    .expect("the moved texture executes");
    let moved_objects = object_readback(&provider, &narrow, TextureFormat::R8Unorm, Pattern::Moved);

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
}

/// Reading 3 (`research/docs/23` §113): the lane is the same sampled-texel
/// vocabulary as the four-component formats — the `rgba8_unorm` sibling of the
/// same geometry lands the same red byte, with its own three channels beside it
/// — while the format the render sampler still refuses keeps its named refusal.
#[test]
fn the_narrow_lane_sits_beside_the_four_component_sibling() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let narrow = register(&provider, &executor, TextureFormat::R8Unorm, "r8 texture");
    let wide = register(
        &provider,
        &executor,
        TextureFormat::Rgba8Unorm,
        "rgba sibling",
    );

    let narrow_frame = trace_readback(
        &provider,
        &compute,
        &narrow,
        sampled_texture_view(TextureFormat::R8Unorm, Pattern::Primary),
    )
    .expect("the R8 bind executes");
    let sibling_frame = trace_readback(
        &provider,
        &compute,
        &wide,
        sampled_texture_view(TextureFormat::Rgba8Unorm, Pattern::Primary),
    )
    .expect("the rgba8 sibling executes");
    eprintln!(
        "narrow {} sibling {}",
        hex(&narrow_frame),
        hex(&sibling_frame)
    );
    // Both stages read texel (0,0)'s red at their first sample, so the two
    // frames share that byte; the sibling's other three lanes come from its own
    // texture bytes while the narrow one's are the format's fill. Only the
    // first three samples reach a lane this fixture writes — its fourth sample
    // reads the alpha lane, which a narrow format's rule pins at one — so the
    // narrow frame's three visible lanes are the byte and the two zero fills.
    let narrow_colour = uniform_texel(&narrow_frame);
    let sibling_colour = uniform_texel(&sibling_frame);
    eprintln!(
        "narrow colour {} sibling colour {}",
        hex(&narrow_colour),
        hex(&sibling_colour)
    );
    assert_eq!(
        narrow_colour[0], sibling_colour[0],
        "the red lane of the two frames is one texel's red byte"
    );
    assert_eq!(
        narrow_colour[1..3],
        [0x00, 0x00],
        "the narrow frame's fill lanes are the format's own zero rule"
    );
    assert_eq!(
        sibling_colour[1..3],
        [0x00, 0x00],
        "the sibling's own texture bytes carry the same three lanes here"
    );

    // The eight-byte texel the render sampler still does not admit: the
    // registration's own format walk refuses it before any device object
    // exists, and the fields name the binding and the format.
    let (vertex, fragment) = translated_pair(&executor, FRAGMENT_AIR, FRAGMENT_ENTRY);
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFormat::Rgba16Float),
            vertex,
            fragment,
            logical_digest: digest(b"wide texture"),
        })
        .expect_err("the rail admits the 8-bit lanes and no other sampled format");
    eprintln!("wide format refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_format_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("format"),
        Some(&FieldValue::Text("Rgba16Float".to_owned()))
    );
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
}
