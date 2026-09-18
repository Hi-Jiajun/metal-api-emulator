//! Trace-produced sampled textures (`research/docs/23` §110, E-TX3).
//!
//! The census's new head gate is the shape this file pins:
//! `render_provider_out_of_class_texture_source`, 31,016 rows in the v15 boot,
//! says that a draw samples a texture whose texels come from the GPU rather
//! than from a copy the request carries. The canonical contract's answer is
//! `TextureSource::TraceView`: the sampled declaration names the view the
//! trace's own earlier pass stores, and the bytes it samples are the ones that
//! production landed — no second name, no bytes in the declaration.
//!
//! The readings are three, all on one Lavapipe device:
//!
//! * the arm **executes**: the producer pass stores a frame, the consumer pass
//!   samples exactly those bytes, and its frame is byte for byte the frame the
//!   same declaration lands when it carries the producer's bytes itself;
//! * the frame **follows the production**: changing the bytes the producer
//!   samples moves the consumer's frame with them;
//! * the shapes the arm cannot express are **refusals by name**: an identity
//!   no pass stores, a store that follows the read, a store that lands no host
//!   bytes, and a declaration that restates another shape.
//!
//! The trace rail and the object rail land the same consumer frame, because
//! the object API's `Device::new_trace_view_texture` states the same arm over
//! the same identity.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
    SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess,
    TextureBindingContract, TextureFootprintProof, TextureFormat, TextureSource, TextureType,
    TextureView, TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of an
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const FRAGMENT_ENTRY: &str = "render_sample_texture_2d";
/// The reviewed sampling module: four `air.sample_texture_2d` reads at the four
/// texel centres, output as `(0,0).x / (1,0).y / (0,1).z / (1,1).w`. Every
/// fragment samples the same coordinates, so a frame it lands is uniform.
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");

const PRODUCER_VIEW: ViewId = ViewId::new(970);
const PRODUCER_ALLOCATION: AllocationId = AllocationId::new(971);
const CONSUMER_VIEW: ViewId = ViewId::new(972);
const CONSUMER_ALLOCATION: AllocationId = AllocationId::new(973);
const INPUT_VIEW: ViewId = ViewId::new(974);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(975);
const SCRATCH_VIEW: ViewId = ViewId::new(976);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(977);

/// 4x4: the extent of both attachments and of the producer's sampled input,
/// because the rail's reviewed window holds every sampled texture to the
/// render area's own extent.
const EXTENT: u32 = 4;
/// The tightly packed byte extent of one 4x4 `Rgba8Unorm` surface.
const FRAME_BYTES: u64 = EXTENT as u64 * EXTENT as u64 * 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

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

/// The producer's sampled input: column `i` holds `64 * i` in red, `64 * j` in
/// green, zero in blue and full alpha. The fragment stage reads one channel of
/// each of the four corner texels, so the colour it stores is decided by these
/// bytes.
fn input_bytes(descending: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FRAME_BYTES as usize);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let column = if descending { EXTENT - 1 - i } else { i };
            bytes.extend_from_slice(&[(64 * column) as u8, (64 * j) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

/// The producer's own sampled texture: the request's copy, which is the arm
/// every earlier increment carried.
fn input_texture(descending: bool) -> TextureView {
    TextureView {
        view_id: INPUT_VIEW,
        metal_binding: 0,
        allocation_id: INPUT_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(input_bytes(descending)),
    }
}

/// The consumer's declaration: the trace's own production of the producer's
/// view, or — for the sibling reading — the bytes that production lands.
fn trace_view_texture(source: TextureSource) -> TextureView {
    TextureView {
        view_id: PRODUCER_VIEW,
        metal_binding: 0,
        allocation_id: PRODUCER_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source,
    }
}

fn declaration(binding: u32, footprint: TextureFootprintProof) -> TextureBindingContract {
    TextureBindingContract {
        metal_binding: binding,
        access: TextureAccess::Sampled,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        sampler: Some(NEAREST_CLAMP),
        runtime_sampler: None,
        footprint,
    }
}

fn contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![declaration(0, TextureFootprintProof::WholeView)],
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
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
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
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
        .compile_pipeline(&function, digest(b"trace-view-declaring-compute"))
        .expect("the compute pipeline registers")
}

fn register_sampling_pipeline(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let (vertex, fragment) = translated_pair(executor);
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"trace-view-sampling-pipeline"),
        })
        .expect("the sampling declaration registers")
}

/// One declaring compute pass: the kernel's read binding carries the
/// attachment view's bytes and its write binding the scratch slot, exactly as
/// every render fixture declares its targets (`research/docs/23` §3.6).
fn declaring_pass(
    compute: &CompiledComputePipeline,
    view_id: ViewId,
    allocation: AllocationId,
) -> ComputePass {
    ComputePass {
        pipeline: compute.pipeline_id,
        buffers: vec![
            BufferView {
                view_id,
                metal_binding: 0,
                allocation_id: allocation,
                offset: 0,
                length: FRAME_BYTES,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(FRAME_BYTES as usize / 4)),
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
    }
}

fn render_pass(
    pipeline: &CompiledComputePipeline,
    view_id: ViewId,
    allocation: AllocationId,
    store: StoreOp,
    textures: Vec<TextureView>,
) -> RenderPassDescriptor {
    RenderPassDescriptor {
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
        pipeline: pipeline.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id,
            allocation_id: allocation,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(EXTENT),
            height: u64::from(EXTENT),
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store,
        }],
        viewport: [0, 0, EXTENT, EXTENT],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures,
        samplers: Vec::new(),
        present: None,
    }
}

/// How the trace orders its two render passes.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    ProducerFirst,
    ConsumerFirst,
}

/// The whole fixture: two declaring compute passes, then the producer and the
/// consumer render passes in the order the caller names.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    input: TextureView,
    sampled: TextureView,
    producer_store: StoreOp,
    order: Order,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let producer = TracePass::Render(render_pass(
        render,
        PRODUCER_VIEW,
        PRODUCER_ALLOCATION,
        producer_store,
        vec![input],
    ));
    let consumer = TracePass::Render(render_pass(
        render,
        CONSUMER_VIEW,
        CONSUMER_ALLOCATION,
        StoreOp::Store,
        vec![sampled],
    ));
    let mut passes = vec![
        TracePass::Compute(declaring_pass(compute, PRODUCER_VIEW, PRODUCER_ALLOCATION)),
        TracePass::Compute(declaring_pass(compute, CONSUMER_VIEW, CONSUMER_ALLOCATION)),
    ];
    match order {
        Order::ProducerFirst => {
            passes.push(producer);
            passes.push(consumer);
        }
        Order::ConsumerFirst => {
            passes.push(consumer);
            passes.push(producer);
        }
    }
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(45),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes,
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation_id, size) in [
        (PRODUCER_ALLOCATION, FRAME_BYTES),
        (CONSUMER_ALLOCATION, FRAME_BYTES),
        (SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("allocation");
    }
    (trace, resources)
}

/// Admit and submit the fixture, and return both frames: the producer's own
/// landing and the consumer's.
fn admit_and_submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    input: TextureView,
    sampled: TextureView,
    producer_store: StoreOp,
    order: Order,
) -> Result<(Vec<u8>, Vec<u8>), metal_api_core::provider::ProviderError> {
    let (trace, resources) = trace_for(
        provider,
        compute,
        render,
        input,
        sampled,
        producer_store,
        order,
    );
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
    let frame = |view: ViewId| {
        submitted
            .writebacks
            .iter()
            .find(|writeback| writeback.view_id == view)
            .map(|writeback| writeback.bytes.clone())
            .expect("the attachment has a writeback")
    };
    Ok((frame(PRODUCER_VIEW), frame(CONSUMER_VIEW)))
}

/// The refusal one trace gets before any device object exists.
fn admit_refusal(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    input: TextureView,
    sampled: TextureView,
    producer_store: StoreOp,
    order: Order,
) -> metal_api_core::provider::ProviderError {
    let (trace, resources) = trace_for(
        provider,
        compute,
        render,
        input,
        sampled,
        producer_store,
        order,
    );
    provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the shape is refused before any device object exists")
}

/// One texel of a readback, which is what the fragment's fixed-coordinate
/// samples land: every texel of the 4x4 frame carries the same colour.
fn uniform_texel(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(bytes.len() as u64, FRAME_BYTES);
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(
            chunk,
            texel,
            "every fragment samples the same coordinates, so every texel has to carry the same \
             colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// The frame the reviewed sampling module lands for one input texture.
///
/// Its AIR names two samples: `(1.375, 0.125)` clamps to the edge texel (3),
/// and `(0.3125, 0.125)` reads texel (1). Both readings are the red channel of
/// the texel they land on — `64 * column` — so an ascending texture lands
/// `(192, 64)` and a descending one `(0, 128)` in the module's red and green
/// halves (`render_sample_texture_2d_nearest_clamp.frag.ll`).
fn producer_frame(descending: bool) -> [u8; 4] {
    if descending {
        [0x00, 0x80, 0x00, 0xff]
    } else {
        [0xc0, 0x40, 0x00, 0xff]
    }
}

/// The consumer samples a texture whose every texel carries the producer's own
/// colour, so both of the module's readings are that colour's red half.
fn consumer_frame(descending: bool) -> [u8; 4] {
    let red = producer_frame(descending)[0];
    [red, red, 0x00, 0xff]
}

/// Reading 1 (`research/docs/23` §110): the trace-produced arm executes in the
/// provider, its frame is byte for byte the frame the same declaration lands
/// when it carries the producer's bytes itself, and the object rail lands that
/// same frame over the same identity.
#[test]
fn a_trace_view_texture_samples_the_bytes_an_earlier_pass_wrote() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_sampling_pipeline(&provider, &executor);

    let (producer, consumer) = admit_and_submit(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Store,
        Order::ProducerFirst,
    )
    .expect("the trace-produced arm executes");
    eprintln!(
        "producer frame {} consumer frame {} (expected {})",
        hex(&producer),
        hex(&consumer),
        hex(&consumer_frame(false))
    );
    assert_eq!(
        uniform_texel(&producer),
        producer_frame(false),
        "the producer's own landing is the frame its input bytes state"
    );
    assert_eq!(
        uniform_texel(&consumer),
        consumer_frame(false),
        "the consumer samples the producer's own frame, so its reading is that frame's red half"
    );

    // The sibling: the same declaration carrying those bytes as its own copy.
    // The arm is a *source*, not a second execution: the two frames agree
    // exactly, which is what "the trace produced them" has to mean in bytes.
    let (_, carried) = admit_and_submit(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::OwnedBytes(producer.clone())),
        StoreOp::Store,
        Order::ProducerFirst,
    )
    .expect("the carried-bytes sibling executes");
    eprintln!("carried-bytes sibling frame {}", hex(&carried));
    assert_eq!(
        consumer, carried,
        "the trace-produced arm lands the frame the carried bytes land"
    );

    // The object rail over the same provider: the same producer pass, the same
    // identity, the same sampled declaration through the object API's own
    // trace-view texture.
    let object_frame = object_readback(&provider, &render, false);
    eprintln!("object rail frame {}", hex(&object_frame));
    assert_eq!(
        uniform_texel(&object_frame),
        consumer_frame(false),
        "object rail: {}",
        hex(&object_frame)
    );
    assert_eq!(
        consumer, object_frame,
        "the trace rail and the object rail land one frame"
    );
}

/// Reading 2: changing the bytes the producer samples moves the consumer's
/// frame with them, because it samples the production and not a copy taken
/// when the trace was declared.
#[test]
fn the_consumer_frame_follows_the_bytes_the_producer_writes() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_sampling_pipeline(&provider, &executor);

    let (first_producer, first_consumer) = admit_and_submit(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Store,
        Order::ProducerFirst,
    )
    .expect("the ascending fixture executes");
    let (second_producer, second_consumer) = admit_and_submit(
        &provider,
        &compute,
        &render,
        input_texture(true),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Store,
        Order::ProducerFirst,
    )
    .expect("the descending fixture executes");
    eprintln!(
        "ascending producer {} consumer {}; descending producer {} consumer {}",
        hex(&first_producer),
        hex(&first_consumer),
        hex(&second_producer),
        hex(&second_consumer)
    );
    assert_ne!(
        first_producer, second_producer,
        "the producer's own landing has to move with its input"
    );
    assert_eq!(uniform_texel(&first_producer), producer_frame(false));
    assert_eq!(uniform_texel(&second_producer), producer_frame(true));
    assert_eq!(uniform_texel(&first_consumer), consumer_frame(false));
    assert_eq!(
        uniform_texel(&second_consumer),
        consumer_frame(true),
        "the consumer's frame follows the bytes the producer wrote"
    );
    assert_ne!(first_consumer, second_consumer);
    // The reading that makes the change *the production's*: each consumer
    // frame's red half is the producer frame's own red half.
    assert_eq!(first_consumer[0], first_producer[0]);
    assert_eq!(second_consumer[0], second_producer[0]);
    assert_eq!(first_producer[0], 0xc0);
    assert_eq!(second_producer[0], 0x00);

    let object_frame = object_readback(&provider, &render, true);
    eprintln!("object rail descending frame {}", hex(&object_frame));
    assert_eq!(object_frame, second_consumer);
}

/// Reading 3: every shape the arm cannot express answers with its own name,
/// before any device object exists.
#[test]
fn the_shapes_the_trace_view_arm_cannot_express_are_refused_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_sampling_pipeline(&provider, &executor);

    // (a) The identity has no producer: the consumer samples bytes no pass of
    // this trace ever stores.
    let mut unwritten = trace_for(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Store,
        Order::ProducerFirst,
    );
    unwritten
        .0
        .passes
        .retain(|pass| !matches!(pass, TracePass::Render(render) if render.color_attachments[0].view_id == PRODUCER_VIEW));
    let refusal = provider
        .capabilities()
        .validate_trace(unwritten.0, unwritten.1)
        .expect_err("an unwritten identity is refused");
    eprintln!("unwritten: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_source_unwritten");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Resource
    );

    // (b) The store follows the read: this increment's order is write-then-read.
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Store,
        Order::ConsumerFirst,
    );
    eprintln!("order: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_source_order_unsupported");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );

    // (c) The producer's most recent store lands no host bytes: the resident
    // arm is the next increment's.
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        input_texture(false),
        trace_view_texture(TextureSource::TraceView),
        StoreOp::Resident,
        Order::ProducerFirst,
    );
    eprintln!("unlanded: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_source_unlanded");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );

    // (d) The declaration restates a shape the stored surface never had: 2x2
    // is not the 4x4 the producer stores, and the arm is not an alias.
    let mut restated = trace_view_texture(TextureSource::TraceView);
    restated.width = 2;
    restated.height = 2;
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        input_texture(false),
        restated,
        StoreOp::Store,
        Order::ProducerFirst,
    );
    eprintln!("shape: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_source_shape_mismatch");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Args
    );
}

/// The frame the object rail lands for the same shape: one command buffer
/// records the producer pass over the input texture and the consumer pass over
/// a texture handle that shares the producer's identity and declares the
/// trace-produced arm.
fn object_readback(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    descending: bool,
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the registration wraps for the object API");
    // The two attachments, one buffer each, exactly as the trace rail's
    // declaring compute passes state them.
    let producer = device
        .new_buffer_with_bytes(ATTACHMENT_WORD.repeat(FRAME_BYTES as usize / 4))
        .expect("the producer's landing buffer is declared");
    let producer_view = producer
        .view(0, FRAME_BYTES as usize)
        .expect("the producer view is declared");
    let consumer = device
        .new_buffer_with_bytes(vec![0x00; FRAME_BYTES as usize])
        .expect("the consumer's landing buffer is declared");
    let consumer_view = consumer
        .view(0, FRAME_BYTES as usize)
        .expect("the consumer view is declared");
    let input = device
        .new_texture_with_bytes(
            TextureFormat::Rgba8Unorm,
            u64::from(EXTENT),
            u64::from(EXTENT),
            input_bytes(descending),
        )
        .expect("the producer's input texture is declared");
    let sampled = device
        .new_trace_view_texture(
            &producer_view,
            TextureFormat::Rgba8Unorm,
            u64::from(EXTENT),
            u64::from(EXTENT),
        )
        .expect("the trace-produced texture is declared over the producer's identity");
    assert_eq!(sampled.allocation_id(), producer_view.allocation_id());
    assert_eq!(sampled.view_id(), producer_view.view_id());

    let command = device.new_command_queue().command_buffer();
    // The declaring side, exactly as the trace rail's compute passes state it:
    // every render attachment's view has to be declared by a compute binding
    // of the same trace (`research/docs/23` §3.6).
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"trace-view-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let scratch = device
            .new_buffer_with_bytes(vec![0xab; 4])
            .expect("the scratch buffer is declared");
        let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
        for view in [&producer_view, &consumer_view] {
            let mut encoder = command.compute_command_encoder().expect("compute encoder");
            encoder
                .set_compute_pipeline_state(&declaring)
                .expect("compute pipeline state");
            encoder
                .set_buffer(0, view)
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
    }
    {
        let mut encoder = command
            .render_command_encoder()
            .expect("the producer's render encoder opens");
        encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the sampling pipeline is bound");
        encoder
            .set_fragment_texture(0, &input)
            .expect("the input texture is bound");
        encoder
            .draw_render_pass(
                &producer_view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the producer pass records");
        encoder.end_encoding().expect("the producer encoder closes");
    }
    {
        let mut encoder = command
            .render_command_encoder()
            .expect("the consumer's render encoder opens");
        encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the sampling pipeline is bound");
        encoder
            .set_fragment_texture(0, &sampled)
            .expect("the trace-produced texture is bound");
        encoder
            .draw_render_pass(
                &consumer_view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the consumer pass records");
        encoder.end_encoding().expect("the consumer encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    consumer
        .read()
        .expect("the consumer's landing bytes are readable")
}
