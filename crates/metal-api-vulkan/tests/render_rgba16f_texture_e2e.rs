//! The render sampler's eight-byte half-float lane (`research/docs/23` §3.3,
//! §107), the shape census v44's `texture_bind` bucket named as the last
//! contract-level door: a draw whose `[[texture(3)]]` bind is a
//! `R16G16B16A16_SFLOAT` guest view.
//!
//! The census sentence (`evidence/gate3-census-v44-2026-09-19/`, 285 records,
//! every one of them a `256x1` view) states the canonical pass's texture as one
//! single-sample, non-arrayed 2D view with one descriptor, an identity channel
//! mapping and one of the provider's own frame's texel formats, while the bind
//! is eight bytes a texel. This test is the Vulkan rail's executable half of
//! the widening, and it is deliberately sharper than "the frame changed":
//!
//! * the fragment stage reads one whole channel of each of four texels, so a
//!   4x4 texture whose sixteen texels carry four chosen values makes every
//!   channel of the frame falsifiable at once;
//! * the values are half floats rather than bytes: red is **2.5**, which the
//!   8-bit attachment clamps to `0xff`, while the first byte of that half's own
//!   encoding is `0x00` — a rail that uploaded the texel as four bytes, or read
//!   its leading byte as a normalised one, lands a different frame;
//! * blue is **-0.5**, the same falsifier in the other direction (clamped to
//!   `0x00` while its leading byte is `0xb8`);
//! * the trace rail and the object rail over one provider land that frame byte
//!   for byte;
//! * another texture moves every sampled channel, so the reading measures the
//!   upload rather than the run;
//! * and the capability frame the consumer reads is asserted to carry the lane,
//!   so the widening the class gate answers from is the one this rail executes.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
    StoreOp, TextureAccess, TextureBindingContract, TextureFormat, TextureSource, TextureType,
    TextureView, TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::provider_api as objects;
use metal_api_core::{ComputeExecutor, Device};
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

/// The same fragment stage the other lane tests read through: it samples four
/// texels of its own texture and returns one *whole channel* of each
/// (`red = (0,0).x`, `green = (1,0).y`, `blue = (0,1).z`, `alpha = (1,1).w`).
/// The format is the bind's own fact, so the same reviewed module reads an
/// `r8_unorm` byte, an `rgba8_unorm` texel and this eight-byte one.
const FRAGMENT_ENTRY: &str = "render_sample_texture_2d_bgra_channels";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_2d_bgra_channels.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(950);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(951);
const SCRATCH_VIEW: ViewId = ViewId::new(952);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(953);
const TEXTURE_VIEW: ViewId = ViewId::new(954);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(955);

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

/// Which values the fixture's 4x4 texture carries, and what the attachment
/// lands for them.
///
/// One texel is eight bytes of four half floats, so a pattern is stated as the
/// four values the fragment stage *reads* plus the value every other channel of
/// the texture carries. The half encodings and the expected 8-bit bytes are
/// written out beside the values: the fixture's expectation is the API's own
/// conversion of the half's exact value, and each byte below is that conversion
/// (`round(clamp(value, 0, 1) * 255)`, ties to even) with no tie in the set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    /// red 2.5, green 0.6, blue -0.5, alpha 0.25; every other channel 0.8.
    Primary,
    /// The same shape one step elsewhere: red 0.9, green 0.4, blue 1.5,
    /// alpha 0.0.
    Moved,
}

impl Pattern {
    /// The four values the fragment stage's four samples reach, in the order
    /// they reach it: red, green, blue, alpha.
    fn sampled(self) -> [f32; 4] {
        match self {
            Self::Primary => [2.5, 0.6, -0.5, 0.25],
            Self::Moved => [0.9, 0.4, 1.5, 0.0],
        }
    }

    /// The half-float bytes of [`Self::sampled`], as
    /// `VK_FORMAT_R16G16B16A16_SFLOAT` spells them (IEEE 754 binary16, little
    /// endian in memory). Written out rather than computed because the standard
    /// library has no stable `f16`; the assertion beside them pins each pair to
    /// the value it encodes.
    fn sampled_half(self) -> [[u8; 2]; 4] {
        match self {
            // 2.5 = 0x4100, 0.6 = 0x38cd, -0.5 = 0xb800, 0.25 = 0x3400.
            Self::Primary => [[0x00, 0x41], [0xcd, 0x38], [0x00, 0xb8], [0x00, 0x34]],
            // 0.9 = 0x3b33, 0.4 = 0x3666, 1.5 = 0x3e00, 0.0 = 0x0000.
            Self::Moved => [[0x33, 0x3b], [0x66, 0x36], [0x00, 0x3e], [0x00, 0x00]],
        }
    }

    /// The channel value every texel carries where the fragment stage does not
    /// read it: 0.8, whose 8-bit conversion (`0xcc`) is distinct from all four
    /// sampled lanes of either pattern, so a channel or texel mix-up cannot
    /// land the same frame.
    fn fill(self) -> f32 {
        0.8
    }

    /// The frame the attachment lands: the four sampled values' own 8-bit
    /// conversion, in the attachment's channel order.
    fn expected(self) -> [u8; 4] {
        match self {
            // 2.5 clamps to 1.0 = 0xff; 0.6 = 153 = 0x99; -0.5 clamps to 0.0 =
            // 0x00; 0.25 = 64 = 0x40.
            Self::Primary => [0xff, 0x99, 0x00, 0x40],
            // 0.9 = 229 = 0xe5; 0.4 = 102 = 0x66; 1.5 clamps to 1.0 = 0xff;
            // 0.0 = 0x00.
            Self::Moved => [0xe5, 0x66, 0xff, 0x00],
        }
    }
}

/// The 4x4 texture the fixture samples, in the memory layout the view names:
/// four half floats per texel, sixteen texels, tightly packed. The four texels
/// the fragment stage reads are `(0,0).x`, `(1,0).y`, `(0,1).z` and `(1,1).w`;
/// every other channel carries the pattern's fill value.
fn texture_bytes(pattern: Pattern) -> Vec<u8> {
    let half = pattern.sampled_half();
    let fill = pattern.fill();
    // 0.8 = 0x3a66.
    let fill_half: [u8; 2] = [0x66, 0x3a];
    assert_eq!(fill, 0.8, "the fill's own encoding below is 0.8's");
    let mut bytes = Vec::with_capacity((EXTENT * EXTENT) as usize * 8);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let mut texel = [fill_half, fill_half, fill_half, fill_half];
            match (i, j) {
                (0, 0) => texel[0] = half[0],
                (1, 0) => texel[1] = half[1],
                (0, 1) => texel[2] = half[2],
                (1, 1) => texel[3] = half[3],
                _ => {}
            }
            for channel in texel {
                bytes.extend_from_slice(&channel);
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
        .compile_pipeline(&function, digest(b"rgba16f-texture-compute"))
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
        source: TextureSource::OwnedBytes(texture_bytes(pattern)),
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
) -> Result<Vec<u8>, metal_api_core::provider::ProviderError> {
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
            texture_bytes(pattern),
        )
        .expect("the sampled texture is declared");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"rgba16f-texture-object-declaring"),
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

/// Reading 1 (`research/docs/23` §107): the eight-byte bind enters the
/// provider, the trace rail and the object rail land byte-identical frames, and
/// the frame is the four half floats' own conversion — the two values outside
/// the 8-bit range prove the texels were read as floats and not as bytes.
#[test]
fn the_rgba16_float_bind_executes_and_the_two_rails_land_the_same_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The frame the consumer's own class gate reads must carry the lane: the
    // widening is answered from *this* list, not from a second table.
    assert!(
        provider
            .capabilities()
            .supported_render_texture_formats
            .contains(&TextureFormat::Rgba16Float),
        "the provider's capability frame states the eight-byte lane"
    );
    let compute = compile_declaring_kernel(&provider, &executor);
    let wide = register(
        &provider,
        &executor,
        TextureFormat::Rgba16Float,
        "rgba16f texture",
    );

    // The fixture's own bytes are the values the expectation derives from, and
    // the two out-of-range values' leading bytes are the falsifier: red 2.5
    // encodes as `00 41`, so a rail that read a texel's first byte as a
    // normalised component would land `0x00` where the frame must carry `0xff`.
    assert_eq!(
        Pattern::Primary.sampled_half(),
        [[0x00, 0x41], [0xcd, 0x38], [0x00, 0xb8], [0x00, 0x34]],
        "the half encodings are the values beside them"
    );
    assert_eq!(Pattern::Primary.sampled()[0], 2.5);
    assert_eq!(Pattern::Primary.sampled()[2], -0.5);

    let expected = Pattern::Primary.expected();
    let uploaded = texture_bytes(Pattern::Primary);
    eprintln!(
        "rgba16f upload: {} bytes, texel (0,0) {}, texel (0,1) {}; expected frame {}",
        uploaded.len(),
        hex(&uploaded[0..8]),
        hex(&uploaded[32..40]),
        hex(&expected)
    );
    assert_eq!(
        uploaded.len(),
        (EXTENT * EXTENT) as usize * 8,
        "the upload is tightly packed, eight bytes per texel"
    );
    assert_eq!(
        &uploaded[32..40],
        &[0x66, 0x3a, 0x66, 0x3a, 0x00, 0xb8, 0x66, 0x3a],
        "texel (0,1) carries the fill in its first two channels and the sampled blue beside them"
    );

    let trace = trace_readback(
        &provider,
        &compute,
        &wide,
        sampled_texture_view(TextureFormat::Rgba16Float, Pattern::Primary),
    )
    .expect("the rgba16_float bind executes");
    let objects = object_readback(
        &provider,
        &wide,
        TextureFormat::Rgba16Float,
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
    assert_ne!(
        uniform_texel(&trace)[0],
        uploaded[0],
        "the red lane is the half's converted value, not the texel's first byte"
    );
}

/// Reading 2 (`research/docs/23` §107): another texture moves every sampled
/// channel, so the reading is the upload and not the run.
#[test]
fn another_rgba16_float_texture_moves_the_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let wide = register(
        &provider,
        &executor,
        TextureFormat::Rgba16Float,
        "rgba16f texture",
    );

    let primary = trace_readback(
        &provider,
        &compute,
        &wide,
        sampled_texture_view(TextureFormat::Rgba16Float, Pattern::Primary),
    )
    .expect("the primary texture executes");
    let moved = trace_readback(
        &provider,
        &compute,
        &wide,
        sampled_texture_view(TextureFormat::Rgba16Float, Pattern::Moved),
    )
    .expect("the moved texture executes");
    let moved_objects =
        object_readback(&provider, &wide, TextureFormat::Rgba16Float, Pattern::Moved);

    let expected_moved = Pattern::Moved.expected();
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
