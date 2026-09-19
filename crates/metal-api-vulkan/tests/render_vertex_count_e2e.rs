//! The layout-free vertex count above the milestone's three vertices
//! (2026-09-19, census v45's `vertex_span` bucket).
//!
//! The census's population is one shape: `1920x1080`, no declared vertex
//! layout, a bound-but-unread stream set, and draws that name **six** vertices
//! (`290x`) or twenty-four (`4x`). The contract admitted exactly three of them
//! before this widening — a triangle list's count is bounded *below* by three,
//! not fixed at it — so the whole population stayed on the engine.
//!
//! This test is the Vulkan rail's executable half of the widening, and it is
//! deliberately sharper than "the widened count is accepted":
//!
//! * the vertex fixture is a six-vertex quad whose second triangle is the half
//!   the first one does not cover, so a rail that kept drawing the milestone's
//!   three vertices lands a *different* frame from one that issues the count
//!   the trace named — the reading is the covered texels themselves;
//! * counts that are not multiples of three (`4`, `5`) are admitted and land
//!   the three-vertex frame, because the incomplete second triangle rasterizes
//!   nothing: the arm's rule is the lower bound, not divisibility;
//! * the provider's own capability frame states the bit the reims class gate
//!   reads, and a snapshot that does not declare it refuses the very same trace
//!   by name (`render_vertex_count_window_unsupported`) instead of handing a
//!   rail a `vertex_id` its module need not carry;
//! * a count below the triangle's three stays the contract's own refusal
//!   (`DrawVertexCountMismatch`), which is the rule this widening does *not*
//!   touch.
//!
//! The width of the frame is the attachment's own 4x4 grid: pixel centres sit
//! at ±0.25 and ±0.75 in NDC, and every edge the fixture states passes between
//! them, so both expectations are the fixture's geometry rather than a fill
//! rule's answer.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, ContractError, Dispatch, DispatchKind, DispatchType, FieldValue,
    LoadOp, OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The six-vertex quad's vertex stage, written as the AIR the translator
/// consumes: positions from `[[vertex_id]]` alone, no vertex stream
/// (`VertexLayout::None`), and a geometry whose covered texels depend on how
/// many vertices the draw names.
const VERTEX_AIR: &str = include_str!("fixtures/render_vertex_id_quad.vert.ll");
const VERTEX_ENTRY: &str = "render_vertex_id_quad";
/// The reviewed solid fragment stage, as the same AIR form: the census's
/// layout-free draws write one colour per covered texel.
const FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2.frag.ll");
const FRAGMENT_ENTRY: &str = "render_solid_rgba8";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(960);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(962);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(963);

/// The attachment's own extent, and the viewport that covers it.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
/// The reviewed fragment stage's own output, which an 8-bit UNORM attachment
/// stores as these bytes (`render_offscreen_2x2.frag.ll`).
const FRAGMENT_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The census shape's own count, and the three-vertex triangle the arm always
/// carried (`290x` of v45's `vertex_span` sentences name six).
const QUAD_VERTICES: u32 = 6;
const TRIANGLE_VERTICES: u32 = 3;

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

/// The layout-free registration the fixture states: the quad's vertex stage,
/// the reviewed solid fragment stage, one `rgba8_unorm` attachment and no
/// vertex stream at all.
fn register_quad(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
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
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: VERTEX_ENTRY.to_owned(),
                fragment_entry: FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: Vec::new(),
            },
            vertex,
            fragment,
            logical_digest: digest(b"vertex-id-quad-e2e"),
        })
        .expect("the layout-free quad registers")
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
        .compile_pipeline(&function, digest(b"vertex-count-compute"))
        .expect("the compute pipeline registers")
}

fn render_pass(pipeline: PipelineId, vertices: u32) -> RenderPassDescriptor {
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
        vertices,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// The trace the declaring compute pass and the draw share
/// (`research/docs/23` §3.6): the compute pass states the attachment's own
/// bytes, so the readback is the pass's declaration rather than a driver's
/// initial contents.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    vertices: u32,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(63),
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
            TracePass::Render(render_pass(render.pipeline_id, vertices)),
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

/// The frame the rail lands for one count, or the refusal that stopped it
/// before the device ever ran.
fn frame_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    vertices: u32,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, vertices);
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

/// The texel at one attachment coordinate, in the frame's row-major order.
fn texel(frame: &[u8], column: u32, row: u32) -> [u8; 4] {
    let offset = (row * EXTENT + column) as usize * 4;
    let slice = &frame[offset..offset + 4];
    [slice[0], slice[1], slice[2], slice[3]]
}

/// Reading 1 (2026-09-19, census v45's `vertex_span` bucket): the six-vertex
/// layout-free draw enters the provider and the frame is the *count's* own —
/// the two triangles cover the whole left half of the attachment, which a rail
/// that kept drawing the milestone's three vertices cannot land.
#[test]
fn the_layout_free_draw_executes_the_count_the_trace_names() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The bit the reims class gate reads is on this provider's own frame.
    let capabilities = provider.capabilities();
    assert!(
        capabilities.supports_render_vertex_count_above_triangle
            && capabilities.declares_render_vertex_count_above_triangle(),
        "the provider's capability frame declares the widened count"
    );

    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register_quad(&provider, &executor);

    let six = frame_for(&provider, &compute, &pipeline, QUAD_VERTICES)
        .expect("the six-vertex layout-free draw executes");
    let three = frame_for(&provider, &compute, &pipeline, TRIANGLE_VERTICES)
        .expect("the three-vertex draw keeps executing");
    assert_ne!(
        six,
        three,
        "the fixture's second triangle is the half the first one leaves clear, so the two counts \
         cannot land the same frame (six: {}, three: {})",
        hex(&six),
        hex(&three)
    );

    // The six-vertex pair covers the attachment's left half, texel for texel;
    // the right half keeps the clear sentinel the load op wrote.
    for row in 0..EXTENT {
        for column in 0..EXTENT {
            let covered = column < EXTENT / 2;
            let expected = if covered {
                FRAGMENT_TEXEL
            } else {
                CLEAR_SENTINEL
            };
            assert_eq!(
                texel(&six, column, row),
                expected,
                "six vertices: texel ({column}, {row}) of {}",
                hex(&six)
            );
        }
    }
    // The three-vertex draw rasterizes the first triangle alone: four texels,
    // all inside the left half, and at least one of the left half's stays
    // clear. The covered *count* is the fixture's own geometry — the vertex
    // stage's y flip decides which four they are, not whether there are four.
    let covered = (0..EXTENT)
        .flat_map(|row| (0..EXTENT).map(move |column| (column, row)))
        .filter(|(column, row)| texel(&three, *column, *row) == FRAGMENT_TEXEL)
        .collect::<Vec<_>>();
    assert_eq!(
        covered.len(),
        4,
        "the three-vertex triangle covers four of the sixteen texels: {}",
        hex(&three)
    );
    assert!(
        covered.iter().all(|(column, _)| *column < EXTENT / 2),
        "all four covered texels sit in the half the quad's first triangle reaches: {covered:?}"
    );

    // Counts that are not multiples of three are admitted, and the incomplete
    // triangle they leave over rasterizes nothing: four and five vertices land
    // the three-vertex frame, seven and eight land the six-vertex one. The
    // arm's rule is the triangle list's lower bound, not divisibility.
    for count in [4, 5] {
        let frame = frame_for(&provider, &compute, &pipeline, count)
            .unwrap_or_else(|error| panic!("the {count}-vertex draw executes: {error:?}"));
        assert_eq!(
            frame,
            three,
            "a {count}-vertex draw lands the three-vertex frame: {}",
            hex(&frame)
        );
    }
    for count in [7, 8] {
        let frame = frame_for(&provider, &compute, &pipeline, count)
            .unwrap_or_else(|error| panic!("the {count}-vertex draw executes: {error:?}"));
        assert_eq!(
            frame,
            six,
            "a {count}-vertex draw lands the quad's own frame: {}",
            hex(&frame)
        );
    }
}

/// Reading 2: the widening moves the *upper* end of the arm only. A count
/// below the triangle's three stays the contract's own refusal, and the
/// refusal is the one the arm has always stated.
#[test]
fn a_count_below_the_triangle_stays_the_contract_refusal() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register_quad(&provider, &executor);

    let (trace, _resources) = trace_for(&provider, &compute, &pipeline, TRIANGLE_VERTICES - 1);
    assert_eq!(
        trace.validate(),
        Err(ContractError::DrawVertexCountMismatch {
            expected: TRIANGLE_VERTICES,
            actual: TRIANGLE_VERTICES - 1,
        })
    );
}

/// Reading 3: the snapshot is the gate. A provider whose capability frame does
/// not declare the widened count refuses the very same six-vertex trace by
/// name instead of executing a `vertex_id` its module need not carry.
#[test]
fn a_snapshot_without_the_bit_refuses_the_widened_count_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register_quad(&provider, &executor);
    let (trace, resources) = trace_for(&provider, &compute, &pipeline, QUAD_VERTICES);

    let mut without = provider.capabilities();
    without.supports_render_vertex_count_above_triangle = false;
    assert!(!without.declares_render_vertex_count_above_triangle());
    let refusal = without
        .validate_trace(trace.clone(), resources.clone())
        .expect_err("a snapshot that declares no widened count refuses the six-vertex draw");
    assert_eq!(refusal.slug, "render_vertex_count_window_unsupported");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert_eq!(
        refusal.fields.get("vertices"),
        Some(&FieldValue::Unsigned(u64::from(QUAD_VERTICES)))
    );

    // The three-vertex shape keeps admitting under the same snapshot: the
    // widening never narrows what every earlier increment published.
    let (three, three_resources) = trace_for(&provider, &compute, &pipeline, TRIANGLE_VERTICES);
    without
        .validate_trace(three, three_resources)
        .expect("the three-vertex shape keeps admitting without the bit");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
