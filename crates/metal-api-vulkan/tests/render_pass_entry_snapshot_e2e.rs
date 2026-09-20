//! The pass-entry snapshot arm (`research/docs/23` §118, E-TX15).
//!
//! The census's frozen boundary is the shape this file pins:
//! `render_provider_out_of_class_texture_source_order` (5,847 rows in the v43
//! boot) refuses every draw that samples the very attachment it renders into,
//! while the engine draws the same records through its feedback arm (1,183 in
//! the same boot) or — where the feedback extension cannot express the shape —
//! through its *fallback*: "capture the prior resident content into a
//! same-format GPU image before changing the attachment"
//! (`crates/reims-vgpu/src/backend/vulkan/engine/exec.rs`).
//!
//! The canonical answer to that fallback is `TextureSource::PassEntrySnapshot`:
//! a sampled declaration that names one of *this pass's* colour attachments and
//! reads the bytes that attachment holds when the pass opens. The readings are
//! four, all on one Lavapipe device:
//!
//! * the arm **executes**: a pass whose attachment is loaded from caller bytes
//!   samples exactly those bytes, and its frame is byte for byte the frame the
//!   same declaration lands when the same bytes travel through the byte arm;
//! * the frame **follows the entry bytes**: changing the attachment's load
//!   bytes moves the frame with them;
//! * the arm reads the **resident** attachment a previous pass kept, so the
//!   deferred-store chain (the census's own shape) is expressible without a
//!   host round trip: the frame carries the kept pass's drawn column beside its
//!   clear, which is what a "the snapshot was the live frame" reading would not
//!   land;
//! * the provenance reading is the counter, not the frame: one copy of the
//!   attachment's tightly packed extent per snapshot declaration, and zero for
//!   the byte arm;
//! * the shapes the arm cannot express are **refusals by name**: a view that is
//!   not this pass's attachment, a declaration that restates another shape, a
//!   clear or discarded load arm, a resident load with no image the provider
//!   holds, and every other source arm on an attachment's own view.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LoadOp,
    OperationId, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
    StoreOp, TextureAccess, TextureBindingContract, TextureFootprintProof, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel, used to declare the attachment's own bytes in
/// the trace's serial view pool.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const FRAGMENT_ENTRY: &str = "render_sample_texture_2d";
/// The reviewed sampling module: two `air.sample_texture_2d` reads at fixed
/// coordinates, output in the module's red and green halves. Every fragment
/// samples the same coordinates, so a frame it lands is uniform.
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");

/// The identity the snapshot declaration names — the pass's own attachment.
const ATTACHMENT_VIEW: ViewId = ViewId::new(990);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(991);
/// The byte arm's sibling: the same bytes under a view that is *not* the
/// attachment.
const INPUT_VIEW: ViewId = ViewId::new(992);
const INPUT_ALLOCATION: AllocationId = AllocationId::new(993);
const SCRATCH_VIEW: ViewId = ViewId::new(994);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(995);
const VERTEX_VIEW: ViewId = ViewId::new(996);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(997);
const INDEX_VIEW: ViewId = ViewId::new(998);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(999);

/// 4x4: the extent of the attachment and of both sampled declarations.
const EXTENT: u32 = 4;
/// The tightly packed byte extent of one 4x4 `Rgba8Unorm` surface.
const FRAME_BYTES: u64 = EXTENT as u64 * EXTENT as u64 * 4;
/// The clear the solid pass opens with, and the solid fragment's own output.
const CLEAR_TEXEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

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

/// The attachment's entry bytes for the load-arm readings: column `i` holds
/// `64 * i` in red and row `j` holds `64 * j` in green. The sampling module's
/// two fixed readings land on texel (3, 0) and texel (1, 0), so the frame it
/// stores is `(64 * 3, 64 * 1, 0, 255)` for the ascending pattern and
/// `(0, 64 * 2, 0, 255)` for the descending one — two numbers this file
/// states beside the bytes it hands in.
fn gradient_bytes(descending: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FRAME_BYTES as usize);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let column = if descending { EXTENT - 1 - i } else { i };
            bytes.extend_from_slice(&[(64 * column) as u8, (64 * j) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

/// The frame the sampling module lands for one entry pattern, at every texel.
fn sampled_frame(descending: bool) -> [u8; 4] {
    if descending {
        [0x00, 0x80, 0x00, 0xff]
    } else {
        [0xc0, 0x40, 0x00, 0xff]
    }
}

/// The frame the sampling module lands for the kept pass's own frame: texel
/// (3, 0) is the cleared right half and texel (1, 0) the drawn left half.
fn resident_snapshot_frame() -> [u8; 4] {
    [CLEAR_TEXEL[0], QUAD_TEXEL[0], 0x00, 0xff]
}

/// The reviewed stream's four vertices moved into the attachment's left half:
/// on a 4x4 extent the two triangles cover columns 0 and 1 and leave columns 2
/// and 3 for the clear, which is what makes "the frame is the pass's" readable
/// per texel.
fn left_half_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [(-1.0_f32, -1.0_f32), (0.0, -1.0), (-1.0, 1.0), (0.0, 1.0)] {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}

fn quad_index_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12);
    for index in [0_u16, 1, 2, 1, 3, 2] {
        bytes.extend_from_slice(&index.to_ne_bytes());
    }
    bytes
}

fn quad_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: 8,
        step: metal_api_core::provider::VertexStep::PerVertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32x2,
        }],
    }])
}

/// The frame the kept pass produces: the drawn left half beside the clear, in
/// row-major texel order.
fn kept_frame_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FRAME_BYTES as usize);
    for _row in 0..EXTENT {
        for column in 0..EXTENT {
            bytes.extend_from_slice(if column < EXTENT / 2 {
                &QUAD_TEXEL
            } else {
                &CLEAR_TEXEL
            });
        }
    }
    bytes
}

struct Fixture {
    executor: Arc<VulkanExecutor>,
    provider: Arc<VulkanComputeProvider>,
    compute: metal_api_core::provider::CompiledComputePipeline,
    render: metal_api_core::provider::CompiledComputePipeline,
    solid: metal_api_core::provider::CompiledComputePipeline,
}

fn fixture() -> Option<Fixture> {
    let (executor, provider) = executor_and_provider()?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let compute = {
        let function = device
            .new_library_with_air(COPY_WORD_AIR)
            .expect("the fixture library loads")
            .function("copy_word")
            .expect("the fixture entry exists");
        provider
            .compile_pipeline(&function, digest(b"pass-entry-snapshot-declaring"))
            .expect("the declaring pipeline registers")
    };
    let render = {
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
        provider
            .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
                contract: RenderPipelineContract {
                    stage_buffers: Vec::new(),
                    vertex_entry: VERTEX_ENTRY.to_owned(),
                    fragment_entry: FRAGMENT_ENTRY.to_owned(),
                    color_formats: vec![AttachmentFormat::Rgba8Unorm],
                    vertex_layout: VertexLayout::None,
                    textures: vec![TextureBindingContract {
                        metal_binding: 0,
                        access: TextureAccess::Sampled,
                        texture_type: TextureType::D2,
                        format: TextureFormat::Rgba8Unorm,
                        sampler: Some(NEAREST_CLAMP),
                        runtime_sampler: None,
                        footprint: TextureFootprintProof::WholeView,
                    }],
                },
                vertex,
                fragment,
                logical_digest: digest(b"pass-entry-snapshot-sampling"),
            })
            .expect("the sampling pipeline registers")
    };
    let solid = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_buffer_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: quad_layout(),
                textures: Vec::new(),
            },
            vertex_spirv: QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: digest(b"pass-entry-snapshot-solid"),
        })
        .expect("the quad pipeline registers");
    Some(Fixture {
        executor,
        provider,
        compute,
        render,
        solid,
    })
}

/// One declaring pass: the kernel's read binding carries the attachment view's
/// bytes and its write binding the scratch slot, exactly as every render
/// fixture declares its targets (`research/docs/23` §3.6).
fn declaring_pass(
    compute: &metal_api_core::provider::CompiledComputePipeline,
    view_id: ViewId,
    allocation: AllocationId,
    bytes: Vec<u8>,
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
                source: BufferSource::OwnedBytes(bytes),
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

/// The pass fields no reading in this file varies.
fn render_pass_defaults(pipeline: metal_api_core::provider::PipelineId) -> RenderPassDescriptor {
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
        pipeline,
        color_attachments: Vec::new(),
        viewport: [0, 0, EXTENT, EXTENT],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        samplers: Vec::new(),
        present: None,
    }
}

/// One colour attachment of the reading's extent, with the load and store arm
/// the reading varies.
fn attachment(load: LoadOp, store: StoreOp) -> RenderAttachment {
    RenderAttachment {
        view_id: ATTACHMENT_VIEW,
        allocation_id: ATTACHMENT_ALLOCATION,
        format: AttachmentFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        load,
        store,
    }
}

/// The snapshot declaration: the pass's own attachment, through the new arm.
fn snapshot_texture(source: TextureSource) -> TextureView {
    TextureView {
        view_id: ATTACHMENT_VIEW,
        metal_binding: 0,
        allocation_id: ATTACHMENT_ALLOCATION,
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

/// The byte arm's sibling declaration: the same bytes under a view that is not
/// the attachment, so the pass samples them without a same-pass conflict.
fn input_texture(source: TextureSource) -> TextureView {
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
        source,
    }
}

/// The solid pass that keeps its frame in the provider's image: the quad over
/// the left half, opening from the clear sentinel.
fn keeping_pass(fixture: &Fixture) -> RenderPassDescriptor {
    let vertex_bytes = left_half_vertex_bytes();
    let index_bytes = quad_index_bytes();
    RenderPassDescriptor {
        pipeline: fixture.solid.pipeline_id,
        color_attachments: vec![attachment(
            LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
            StoreOp::Resident,
        )],
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
        vertex_buffers: vec![BufferView {
            view_id: VERTEX_VIEW,
            metal_binding: 0,
            allocation_id: VERTEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(vertex_bytes.len()).expect("stream length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vertex_bytes),
        }],
        indices: Some(IndexBufferBinding {
            view: BufferView {
                view_id: INDEX_VIEW,
                metal_binding: 0,
                allocation_id: INDEX_ALLOCATION,
                offset: 0,
                length: u64::try_from(index_bytes.len()).expect("index length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(index_bytes),
            },
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(fixture.solid.pipeline_id)
    }
}

/// One submission's whole trace: the declaring passes, then the reading's own
/// render passes.
fn trace_for(
    fixture: &Fixture,
    declaring: Vec<(ViewId, AllocationId, Vec<u8>)>,
    render_passes: Vec<RenderPassDescriptor>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut passes = declaring
        .into_iter()
        .map(|(view, allocation, bytes)| {
            TracePass::Compute(declaring_pass(&fixture.compute, view, allocation, bytes))
        })
        .collect::<Vec<_>>();
    passes.extend(render_passes.into_iter().map(TracePass::Render));
    // The trace's pipeline table states exactly the pipelines its passes use,
    // in first-use order — an entry no pass names is the `unused pipeline`
    // refusal, so each reading declares only what it runs.
    let mut pipelines = Vec::new();
    for pass in &passes {
        let pipeline = match pass {
            TracePass::Compute(pass) => pass.pipeline,
            TracePass::Render(pass) => pass.pipeline,
            // The fixture builds single-draw passes: the list arm names its
            // pipelines in its own draws, and this walk has none to declare.
            TracePass::RenderDraws(_) | TracePass::Landing(_) => continue,
        };
        if pipelines.iter().any(
            |declared: &metal_api_core::provider::CompiledComputePipeline| {
                declared.pipeline_id == pipeline
            },
        ) {
            continue;
        }
        let declared = if pipeline == fixture.compute.pipeline_id {
            fixture.compute.clone()
        } else if pipeline == fixture.render.pipeline_id {
            fixture.render.clone()
        } else {
            fixture.solid.clone()
        };
        pipelines.push(declared);
    }
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: fixture.provider.device_epoch(),
        operation_id: OperationId::new(46),
        pipelines,
        encoder_dispatch_type: DispatchType::Serial,
        passes,
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation_id, size) in [
        (ATTACHMENT_ALLOCATION, FRAME_BYTES),
        (INPUT_ALLOCATION, FRAME_BYTES),
        (SCRATCH_ALLOCATION, 8),
        (VERTEX_ALLOCATION, 32),
        (INDEX_ALLOCATION, 12),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id,
                owner_epoch: fixture.provider.device_epoch(),
                size,
            })
            .expect("allocation");
    }
    (trace, resources)
}

/// Admit and submit one trace, and return the frame the last render pass lands
/// under the attachment's identity.
fn submit(fixture: &Fixture, trace: ComputeTrace, resources: ResourceTableSnapshot) -> Vec<u8> {
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace is admitted");
    let submitted = fixture
        .provider
        .submit(admitted)
        .expect("the trace submits");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    submitted
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes.clone())
        .expect("the attachment has a writeback")
}

/// The refusal one trace gets before any device object exists.
fn refusal(
    fixture: &Fixture,
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
) -> metal_api_core::provider::ProviderError {
    fixture
        .provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the shape is refused before any device object exists")
}

/// The refusal one trace gets once the rail builds its plan: shapes the
/// contract admits but the rail cannot execute (a resident arm with no image
/// the provider holds) are refused here rather than during admission.
fn submit_refusal(
    fixture: &Fixture,
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
) -> metal_api_core::provider::ProviderError {
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect("the contract admits the shape; the rail refuses it");
    fixture
        .provider
        .submit(admitted)
        .expect_err("the rail refuses the shape")
}

/// Every texel of a uniform frame, checked: the module's two readings land on
/// the same entry bytes, so the whole frame carries one colour.
fn uniform_frame(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(bytes.len() as u64, FRAME_BYTES);
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(
            chunk,
            texel,
            "the module's two samples are fixed coordinates, so every texel has to carry the \
             same colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// Reading 1 (`research/docs/23` §118): the pass-entry snapshot arm executes in
/// the provider, its frame is byte for byte the frame the same bytes land
/// through the byte arm, and the provenance counter shows the copy ran.
#[test]
fn a_pass_entry_snapshot_reads_the_attachment_the_pass_opens() {
    let Some(fixture) = fixture() else {
        return;
    };
    let entry = gradient_bytes(false);
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
            textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    let before = fixture.executor.attachment_snapshot_counts();
    let frame = submit(&fixture, trace, resources);
    let after = fixture.executor.attachment_snapshot_counts();
    eprintln!(
        "pass-entry snapshot frame {} (expected {})",
        hex(&frame),
        hex(&sampled_frame(false).repeat(EXTENT as usize * EXTENT as usize))
    );
    assert_eq!(uniform_frame(&frame), sampled_frame(false));
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, FRAME_BYTES as usize),
        "one snapshot declaration takes one copy of the attachment's tightly packed extent"
    );

    // The sibling reading: the same bytes under a view that is not the
    // attachment. The frame has to be identical, because the arm's whole claim
    // is that the sampled bytes are the attachment's entry contents.
    let (trace, resources) = trace_for(
        &fixture,
        vec![
            (ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone()),
            (INPUT_VIEW, INPUT_ALLOCATION, entry),
        ],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
            textures: vec![input_texture(TextureSource::OwnedBytes(gradient_bytes(
                false,
            )))],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    let before = fixture.executor.attachment_snapshot_counts();
    let sibling = submit(&fixture, trace, resources);
    let after = fixture.executor.attachment_snapshot_counts();
    assert_eq!(
        hex(&sibling),
        hex(&frame),
        "the byte arm and the pass-entry snapshot arm land the same frame"
    );
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (0, 0),
        "the byte arm takes no snapshot copy"
    );
}

/// Reading 2: the frame follows the attachment's entry bytes. The declaration
/// is unchanged; only the bytes the pass loads move, so a rail that sampled
/// anything else (the pass's own output, a stale image, a clear) would not move
/// with them.
#[test]
fn the_snapshot_frame_follows_the_attachments_entry_bytes() {
    let Some(fixture) = fixture() else {
        return;
    };
    let ascending = {
        let (trace, resources) = trace_for(
            &fixture,
            vec![(
                ATTACHMENT_VIEW,
                ATTACHMENT_ALLOCATION,
                gradient_bytes(false),
            )],
            vec![RenderPassDescriptor {
                color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
                textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
                ..render_pass_defaults(fixture.render.pipeline_id)
            }],
        );
        submit(&fixture, trace, resources)
    };
    let descending = {
        let (trace, resources) = trace_for(
            &fixture,
            vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, gradient_bytes(true))],
            vec![RenderPassDescriptor {
                color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
                textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
                ..render_pass_defaults(fixture.render.pipeline_id)
            }],
        );
        submit(&fixture, trace, resources)
    };
    assert_eq!(uniform_frame(&ascending), sampled_frame(false));
    assert_eq!(uniform_frame(&descending), sampled_frame(true));
    assert_ne!(
        hex(&ascending),
        hex(&descending),
        "moving the entry bytes has to move the frame"
    );
}

/// Reading 3: a resident attachment. One pass keeps its frame in the
/// provider's image (`StoreOp::Resident`), and the next pass opens that image
/// (`LoadOp::Resident`) and samples its pass-entry content — the census's own
/// deferred-store shape, with no host round trip anywhere. The frame carries
/// the kept pass's drawn left half beside its clear, per texel.
#[test]
fn a_resident_pass_entry_snapshot_reads_the_frame_a_pass_kept() {
    let Some(fixture) = fixture() else {
        return;
    };
    let (trace, resources) = trace_for(
        &fixture,
        vec![(
            ATTACHMENT_VIEW,
            ATTACHMENT_ALLOCATION,
            vec![0x7e; FRAME_BYTES as usize],
        )],
        vec![
            keeping_pass(&fixture),
            RenderPassDescriptor {
                color_attachments: vec![attachment(LoadOp::Resident, StoreOp::Store)],
                textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
                ..render_pass_defaults(fixture.render.pipeline_id)
            },
        ],
    );
    let before = fixture.executor.attachment_snapshot_counts();
    let frame = submit(&fixture, trace, resources);
    let after = fixture.executor.attachment_snapshot_counts();
    assert_eq!(
        hex(&frame),
        hex(&resident_snapshot_frame().repeat(EXTENT as usize * EXTENT as usize)),
        "the snapshot reads the kept frame: the drawn left half and the pass's own clear, not \
         the clear alone and not the reading pass's output"
    );
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, FRAME_BYTES as usize),
        "the resident arm takes the same one device-side copy"
    );
    // The kept frame itself is what the reading is about: the clear sentinel
    // alone would land `(0xfe, 0xfe, 0, 0xff)`.
    assert_ne!(
        uniform_frame(&frame),
        [CLEAR_TEXEL[0], CLEAR_TEXEL[0], 0x00, 0xff],
        "a snapshot of the clear alone is not the kept frame"
    );
    assert_eq!(
        kept_frame_bytes().len() as u64,
        FRAME_BYTES,
        "the kept frame and the snapshot share one extent"
    );
}

/// Reading 4: the shapes the arm cannot express are refused by name, before any
/// device object exists.
#[test]
fn the_pass_entry_snapshot_shapes_are_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    let entry = gradient_bytes(false);

    // A declaration that names a view this pass does not open as an
    // attachment: the arm has no entry content to read.
    let mut unattached = snapshot_texture(TextureSource::PassEntrySnapshot);
    unattached.view_id = INPUT_VIEW;
    unattached.allocation_id = INPUT_ALLOCATION;
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
            textures: vec![unattached],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        refusal(&fixture, trace, resources).slug,
        "render_pass_entry_snapshot_unattached"
    );

    // A declaration that restates another extent.
    let mut mismatched = snapshot_texture(TextureSource::PassEntrySnapshot);
    mismatched.width = 2;
    mismatched.height = 2;
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
            textures: vec![mismatched],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        refusal(&fixture, trace, resources).slug,
        "render_pass_entry_snapshot_shape_mismatch"
    );

    // A clear load arm establishes the clear colour, not prior contents.
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(
                LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
                StoreOp::Store,
            )],
            textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        refusal(&fixture, trace, resources).slug,
        "render_pass_entry_snapshot_load_unsupported"
    );

    // A discarded load arm establishes nothing at all.
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::DontCare, StoreOp::Store)],
            textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        refusal(&fixture, trace, resources).slug,
        "render_pass_entry_snapshot_load_unsupported"
    );

    // The resident arm with no image the provider holds: the copy's source
    // does not exist, so the pass is refused rather than served whatever bytes
    // a fresh image happens to hold.
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Resident, StoreOp::Store)],
            textures: vec![snapshot_texture(TextureSource::PassEntrySnapshot)],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        submit_refusal(&fixture, trace, resources).slug,
        "resident_target_unavailable",
        "the arm's source is the provider's own image, so a load the provider holds no image \
         for is refused by name instead of copying whatever a fresh image holds"
    );

    // Every other source arm on an attachment's own view stays the conflict
    // refusal it always was — the new arm is the *only* declaration that names
    // the same-pass read, and it does not widen the others.
    let (trace, resources) = trace_for(
        &fixture,
        vec![(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, entry.clone())],
        vec![RenderPassDescriptor {
            color_attachments: vec![attachment(LoadOp::Load, StoreOp::Store)],
            textures: vec![snapshot_texture(TextureSource::OwnedBytes(entry))],
            ..render_pass_defaults(fixture.render.pipeline_id)
        }],
    );
    assert_eq!(
        refusal(&fixture, trace, resources).slug,
        "trace_contract_invalid",
        "the byte arm on the attachment's own view keeps the attachment-conflict refusal"
    );
}
