//! One submission scope for a run of resident-chain render passes
//! (`REIMS_VGPU_RENDER_BATCH`, G3-B/B-1).
//!
//! The rename rail's chain-middle handoff (`research/docs/23` §76, R42) already
//! keeps a record's frame in the provider's own image and has the next record
//! load it — but today every record is its own trace, so the two passes are two
//! `vkQueueSubmit`s, two fences and two waits. This file measures the shape that
//! carries both passes in **one** trace and one submission scope: the keeping
//! pass (`LoadOp::Clear` + `StoreOp::Resident`) and the consuming pass
//! (`LoadOp::Resident` + `StoreOp::Store`) share the queue lock, the submit, the
//! fence and the wait.
//!
//! Every reading here is falsifiable:
//!
//! * the frame the one-submission run publishes is byte for byte the frame the
//!   same two passes publish as two submissions — and it is the frame the
//!   *second* pass composited over the first, so a load that read the image
//!   before the first pass's store was visible shows up as the clear colour in
//!   the first column rather than the drawn texel;
//! * the run costs exactly one queue submission where the same passes cost two,
//!   read from the executor's own submission counter.
//!
//! The control arm is the pre-batch shape itself: two traces, one pass each,
//! submitted through the same provider. The switch is read once per process, so
//! this binary sets it before the first provider exists.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch,
    DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LoadOp, OperationId,
    RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
    SemanticDigest, StoreOp, TracePass, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel, used to declare the attachment's view in the
/// trace's serial pool.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The identity both passes render into, and the one the batch's chain crosses.
const ATTACHMENT_VIEW: ViewId = ViewId::new(971);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(981);
const KEEPING_VERTEX_VIEW: ViewId = ViewId::new(972);
const KEEPING_VERTEX_ALLOCATION: AllocationId = AllocationId::new(982);
const CONSUMING_VERTEX_VIEW: ViewId = ViewId::new(973);
const CONSUMING_VERTEX_ALLOCATION: AllocationId = AllocationId::new(983);
const INDEX_VIEW: ViewId = ViewId::new(974);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(984);
/// An identity no pass ever stores into, named by the drift oracle: the
/// consuming pass that loads it must be refused rather than merged into a batch
/// with the pass that kept a *different* frame.
const DRIFTED_VIEW: ViewId = ViewId::new(976);
const DRIFTED_ALLOCATION: AllocationId = AllocationId::new(986);
/// The declaring kernel's own scratch write, which its contract requires.
const SCRATCH_VIEW: ViewId = ViewId::new(975);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(985);

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The clear the keeping pass opens with. The consuming pass draws over the
/// other column, so a load that missed the keeping pass's store leaves this
/// texel where the drawn one belongs.
const CLEAR_TEXEL: [u8; 4] = [0x0a, 0x0b, 0x0c, 0x0d];

/// The frame the run publishes: both columns drawn, in row-major texel order.
fn batched_frame_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for _row in 0..2 {
        bytes.extend_from_slice(&QUAD_TEXEL);
        bytes.extend_from_slice(&QUAD_TEXEL);
    }
    bytes
}

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

/// The reviewed stream's four vertices over one half of the attachment: the two
/// triangles cover one column and leave the other for whoever drew it before.
fn half_column_vertex_bytes(right: bool) -> Vec<u8> {
    let (left, right_x) = if right {
        (0.0_f32, 1.0_f32)
    } else {
        (-1.0_f32, 0.0_f32)
    };
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [
        (left, -1.0_f32),
        (right_x, -1.0),
        (left, 1.0),
        (right_x, 1.0),
    ] {
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

fn render_pass_defaults(pipeline: metal_api_core::provider::PipelineId) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
        stage_buffers: Vec::new(),
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
        pipeline,
        color_attachments: Vec::new(),
        viewport: [0, 0, 2, 2],
        scissor: None,
        vertices: 6,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// The passes, pipelines and declarations both arms share.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    /// The declaring kernel, so a reading can declare another identity in the
    /// same trace (the drift oracle).
    compute: metal_api_core::provider::CompiledComputePipeline,
    /// The batch arm's trace: the keeping pass and the consuming pass in one
    /// trace, which is one submission once the switch is on.
    batched: ComputeTrace,
    /// The control arm's two traces, one pass each: the pre-batch shape.
    keeping: ComputeTrace,
    consuming: ComputeTrace,
    resources: ResourceTableSnapshot,
}

fn fixture(executor: Arc<VulkanExecutor>) -> Option<Fixture> {
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_batch_declaring"))
        .expect("the declaring pipeline registers");
    let render = provider
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
            logical_digest: digest(b"render_batch_stages"),
        })
        .expect("the vertex-input render pipeline registers");

    let index_bytes = quad_index_bytes();
    let vertex_bytes = |right: bool| half_column_vertex_bytes(right);
    let vertex_buffer = |view: ViewId, allocation: AllocationId, right: bool| BufferView {
        view_id: view,
        metal_binding: 0,
        allocation_id: allocation,
        offset: 0,
        length: u64::try_from(vertex_bytes(right).len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vertex_bytes(right)),
    };
    let index_buffer = BufferView {
        view_id: INDEX_VIEW,
        metal_binding: 0,
        allocation_id: INDEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(index_bytes.len()).expect("index length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(index_bytes.clone()),
    };
    let index_count = u32::try_from(index_bytes.len() / 2).expect("uint16 index count");
    let attachment = |load: LoadOp, store: StoreOp| RenderAttachment {
        view_id: ATTACHMENT_VIEW,
        allocation_id: ATTACHMENT_ALLOCATION,
        format: AttachmentFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        load,
        store,
    };
    // The keeping pass: it clears, draws one column and keeps the frame in the
    // provider's own image. Nothing about it publishes the frame.
    let keeping_pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![attachment(
            LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
            StoreOp::Resident,
        )],
        vertices: index_count,
        vertex_buffers: vec![vertex_buffer(
            KEEPING_VERTEX_VIEW,
            KEEPING_VERTEX_ALLOCATION,
            false,
        )],
        indices: Some(IndexBufferBinding {
            view: index_buffer.clone(),
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(render.pipeline_id)
    };
    // The consuming pass: it keeps the image's own bytes and draws the other
    // column over them, then publishes the frame through the writeback channel.
    let consuming_pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![attachment(LoadOp::Resident, StoreOp::Store)],
        vertices: index_count,
        vertex_buffers: vec![vertex_buffer(
            CONSUMING_VERTEX_VIEW,
            CONSUMING_VERTEX_ALLOCATION,
            true,
        )],
        indices: Some(IndexBufferBinding {
            view: index_buffer.clone(),
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(render.pipeline_id)
    };
    // The attachment's own declaration. A resident store publishes nothing and
    // a resident load reads nothing out of it, but the identity has to be in
    // the trace's serial view list for the pass that stores it to be admitted.
    let declaring = |view: ViewId, allocation: AllocationId| {
        TracePass::Compute(ComputePass {
            pipeline: compute.pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: view,
                    metal_binding: 0,
                    allocation_id: allocation,
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0x7e; 16]),
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
        })
    };
    let trace = |operation: u64, passes: Vec<TracePass>| ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(operation),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes,
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let declared = || declaring(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION);
    let batched = trace(
        41,
        vec![
            declared(),
            TracePass::Render(keeping_pass.clone()),
            TracePass::Render(consuming_pass.clone()),
        ],
    );
    let keeping = trace(42, vec![declared(), TracePass::Render(keeping_pass)]);
    let consuming = trace(43, vec![declared(), TracePass::Render(consuming_pass)]);

    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, 16_u64),
        (KEEPING_VERTEX_ALLOCATION, 32),
        (CONSUMING_VERTEX_ALLOCATION, 32),
        (INDEX_ALLOCATION, 12),
        (SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    Some(Fixture {
        provider: Arc::new(provider),
        compute: compute.clone(),
        batched,
        keeping,
        consuming,
        resources,
    })
}

fn submit(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<metal_api_core::provider::BufferWriteback> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the fixture trace is admitted");
    let submitted = provider.submit(admitted).expect("the trace executes");
    submitted
        .validate_for_trace(trace)
        .expect("the trace's writebacks are what it declared");
    submitted.writebacks
}

fn frame_of(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<u8> {
    submit(provider, trace, resources)
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the consuming pass publishes the frame it read")
}

fn submissions(executor: &VulkanExecutor) -> usize {
    executor.queue_submission_counts().into_iter().sum()
}

#[test]
fn a_run_of_two_kept_passes_is_one_submission_and_the_same_frame() {
    // The switch is read once per process, before the first provider exists.
    // `REIMS_VGPU_RENDER_BATCH=off` in the environment runs this same test as
    // the control arm, which is what makes the two submission counts below a
    // two-armed reading rather than one.
    if std::env::var("REIMS_VGPU_RENDER_BATCH").is_err() {
        std::env::set_var("REIMS_VGPU_RENDER_BATCH", "1");
    }
    let batching = metal_api_vulkan::render_batch_enabled();
    let Some(executor) = executor() else {
        return;
    };
    let Some(fixture) = fixture(Arc::clone(&executor)) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);

    // The control arm: the same two passes as two traces, which is the
    // pre-batch shape. Each trace's declaring compute half is one submission of
    // its own, so the two traces cost four: two compute halves and two render
    // passes.
    let before = submissions(&executor);
    submit(&provider, &fixture.keeping, &fixture.resources);
    let reference = frame_of(&provider, &fixture.consuming, &fixture.resources);
    let two_submissions = submissions(&executor) - before;
    assert_eq!(
        two_submissions, 4,
        "the pre-batch shape submits each pass's render half on its own"
    );

    // The batch arm: both passes in one trace and one submission scope, so the
    // one declaring half plus the one render batch is what is left.
    let before = submissions(&executor);
    let batched = frame_of(&provider, &fixture.batched, &fixture.resources);
    let run_submissions = submissions(&executor) - before;
    assert_eq!(
        run_submissions,
        if batching { 2 } else { 3 },
        "one declaring half plus one render batch when the switch is on, one \
         declaring half plus one submission per pass when it is off"
    );
    assert_eq!(
        hex(&batched),
        hex(&reference),
        "the batched run publishes the same frame as the two-submission shape"
    );
    assert_eq!(
        hex(&batched),
        hex(&batched_frame_bytes()),
        "the frame is the keeping pass's column beside the consuming pass's own: \
         a load that missed the keeping pass's store would leave the clear texel"
    );
}

#[test]
fn a_consuming_pass_that_names_another_identity_is_refused_by_name() {
    // The identity drift the keep plan exists to catch: the consuming pass
    // loads a resident image the keeping pass never stored into. A batch that
    // merged the two — claiming the drifted pair as defined because the pass
    // before it kept *something* — would render into a fresh image full of
    // uninitialised bytes and publish them; the rail's own name is what says it
    // did not.
    let Some(executor) = executor() else {
        return;
    };
    let Some(fixture) = fixture(Arc::clone(&executor)) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let mut drifted = fixture.batched.clone();
    drifted.operation_id = OperationId::new(44);
    let mut rebuilt = Vec::new();
    for entry in drifted.passes {
        match entry {
            TracePass::Render(mut pass) if pass.color_attachments[0].load == LoadOp::Resident => {
                // The drifted identity is declared to the trace — that is what
                // makes this the *rail's* refusal rather than admission's — but
                // no pass ever stores into it.
                rebuilt.push(declaring_pass(
                    &fixture.compute,
                    DRIFTED_VIEW,
                    DRIFTED_ALLOCATION,
                ));
                pass.color_attachments[0].view_id = DRIFTED_VIEW;
                pass.color_attachments[0].allocation_id = DRIFTED_ALLOCATION;
                rebuilt.push(TracePass::Render(pass));
            }
            other => rebuilt.push(other),
        }
    }
    drifted.passes = rebuilt;
    let mut resources = fixture.resources.clone();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: DRIFTED_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 16,
        })
        .expect("the drifted identity's allocation record");
    let admitted = provider
        .capabilities()
        .validate_trace(drifted.clone(), resources.clone())
        .expect("the drift shape is admitted: the refusal is the rail's own");
    let error = provider
        .submit(admitted)
        .expect_err("an identity no pass stored is refused by name");
    assert_eq!(error.slug, "resident_target_unavailable");
    assert_eq!(
        error.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );
}

/// One declaring compute pass for an identity the trace names but no render
/// pass owns (`research/docs/23` §76, R7: the trace's serial pool is what an
/// attachment view is resolved against).
fn declaring_pass(
    compute: &metal_api_core::provider::CompiledComputePipeline,
    view: ViewId,
    allocation: AllocationId,
) -> TracePass {
    TracePass::Compute(ComputePass {
        pipeline: compute.pipeline_id,
        buffers: vec![
            BufferView {
                view_id: view,
                metal_binding: 0,
                allocation_id: allocation,
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0x7e; 16]),
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
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}
