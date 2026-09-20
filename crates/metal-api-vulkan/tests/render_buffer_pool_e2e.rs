//! The pooled host-visible upload buffers' own rail cases
//! (`crate::render_buffer_pool`).
//!
//! The increment hands a creation of a given shape — a `VkBuffer` of one size
//! and one usage — the buffer and memory a creation of the *same shape* built
//! before. That is only allowed to be a timing change, so this file is the
//! byte-level oracle the increment's own report reads. The draw the fixture
//! runs binds three of them in one pass: the position stream, the scalar
//! stream (two `float32x1` attributes at offsets zero and four) and the index
//! buffer.
//!
//! * one draw runs twice against one provider: the second pass must be served
//!   from the pool (one hit per upload buffer) and must publish the *same frame
//!   bytes*;
//! * the second arm states *different bytes* for the same shapes and must land
//!   *its own* frame — the one failure direction a pooled upload buffer has
//!   ("the next pass binds what the last one wrote there");
//! * the same passes must publish the same bytes with the mechanism switched
//!   **off**, so the two arms of the reading are compared with each other and
//!   not only with themselves;
//! * switching the mechanism off must drop what it held, and the counters that
//!   name the directions (`hits`, `misses`, `disabled`, `returns`) must be
//!   readable from the provider, so a round that shows no reuse can tell "the
//!   shapes never repeated" from "the pool refused them".

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LoadOp,
    OperationId, PipelineId, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, VertexStep, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderBufferPoolCounts, RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed pair the scalar-lane fixture states: a vertex stage whose
/// `[[stage_in]]` interface is a `float32x2` position beside two scalar
/// `float32x1` lanes, and a fragment stage that stores the forwarded tint.
const VERTEX_AIR: &str = include_str!("fixtures/render_scalar_quad.vert.ll");
const VERTEX_ENTRY: &str = "render_scalar_quad_vertex";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_scalar_tint.frag.ll");
const FRAGMENT_ENTRY: &str = "render_scalar_tint";

/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
const COPY_WORD_ENTRY: &str = "copy_word";

const ATTACHMENT_VIEW: ViewId = ViewId::new(6170);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(6171);
const SCRATCH_VIEW: ViewId = ViewId::new(6172);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(6173);
const POSITION_VIEW: ViewId = ViewId::new(6174);
const POSITION_ALLOCATION: AllocationId = AllocationId::new(6175);
const SCALAR_VIEW: ViewId = ViewId::new(6176);
const SCALAR_ALLOCATION: AllocationId = AllocationId::new(6177);
const INDEX_VIEW: ViewId = ViewId::new(6178);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(6179);

/// 4x4, the extent both rectangles split down the middle.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const POSITION_STRIDE: usize = 8;
const POSITION_RECORDS: [(f32, f32); 12] = [
    (-1.0, -1.0),
    (0.0, -1.0),
    (-1.0, 1.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (0.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (0.0, 1.0),
];

const SCALAR_STRIDE: usize = 8;
const RED_OFFSET: u64 = 0;
const GREEN_OFFSET: u64 = 4;

/// The scalars the two rectangles carry, and the texels they land. `0.2f` is
/// `0x3e4ccccd`, whose `× 255` is `51.000001`; `0.6f` is `0x3f19999a`, whose
/// `× 255` is `153.000006` — both unambiguous under a rounding *and* under a
/// truncating float-to-unorm conversion, so the comparison stays a byte
/// comparison.
const LEFT_SCALARS: [f32; 2] = [0.2, 0.6];
const RIGHT_SCALARS: [f32; 2] = [0.6, 0.2];
const LEFT_TEXEL: [u8; 4] = [0x33, 0x99, 0x00, 0xff];
const RIGHT_TEXEL: [u8; 4] = [0x99, 0x33, 0x00, 0xff];

/// The 12 indices naming the two rectangles' six vertices each.
const INDICES: [u8; 24] = [
    0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00,
    0x08, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x0b, 0x00,
];

/// How many upload buffers one pass creates: the position stream, the scalar
/// stream and the index buffer. The three differ in size or usage, so they are
/// three keys rather than one.
const UPLOADS_PER_PASS: u64 = 3;

fn half(scalars: [f32; 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(SCALAR_STRIDE * 6);
    for _ in 0..6 {
        for component in scalars {
            bytes.extend_from_slice(&component.to_ne_bytes());
        }
    }
    bytes
}

/// The scalar stream both rectangles' vertices read: the left half's scalars
/// first, then the right half's.
fn scalars(left: [f32; 2], right: [f32; 2]) -> Vec<u8> {
    let mut bytes = half(left);
    bytes.extend_from_slice(&half(right));
    bytes
}

fn positions() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(POSITION_STRIDE * POSITION_RECORDS.len());
    for (x, y) in POSITION_RECORDS {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("render-buffer-pool-fixture-v1", case.to_vec()).expect("digest")
}

fn provider_with_device() -> Option<(Arc<VulkanExecutor>, VulkanComputeProvider)> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    Some((executor, provider))
}

fn scalar_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![
        VertexBufferLayout {
            stride: POSITION_STRIDE as u64,
            step: VertexStep::PerVertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        },
        VertexBufferLayout {
            stride: SCALAR_STRIDE as u64,
            step: VertexStep::PerVertex,
            attributes: vec![
                VertexAttribute {
                    location: 1,
                    offset: RED_OFFSET,
                    format: VertexFormat::Float32x1,
                },
                VertexAttribute {
                    location: 2,
                    offset: GREEN_OFFSET,
                    format: VertexFormat::Float32x1,
                },
            ],
        },
    ])
}

fn scalar_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: scalar_layout(),
        textures: Vec::new(),
    }
}

fn translate_stage(
    executor: &Arc<VulkanExecutor>,
    provider: &VulkanComputeProvider,
    stage: RenderStage,
    source: &str,
    entry: &str,
) -> TranslatedRenderStage {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .unwrap_or_else(|error| panic!("the fixture library loads: {error:?}"));
    let function = library
        .function(entry)
        .unwrap_or_else(|error| panic!("the fixture entry {entry} exists: {error:?}"));
    TranslatedRenderStage::translate_with_policy(stage, &function, provider.spirv_feature_policy())
        .unwrap_or_else(|error| panic!("{entry} translates: {error:?}"))
}

fn compile_declaring_kernel(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function(COPY_WORD_ENTRY)
        .expect("the fixture entry exists");
    provider
        .compile_pipeline(&function, digest(b"render-buffer-pool-compute"))
        .expect("the compute pipeline registers")
}

fn register(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let vertex = translate_stage(
        executor,
        provider,
        RenderStage::Vertex,
        VERTEX_AIR,
        VERTEX_ENTRY,
    );
    let fragment = translate_stage(
        executor,
        provider,
        RenderStage::Fragment,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
    );
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: scalar_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"pooled render input buffers"),
        })
        .expect("the reviewed pair registers")
}

/// One render pass over the 4x4 attachment, binding the two streams and the
/// index buffer — the three upload buffers the pool is asked about.
fn render_pass(pipeline: PipelineId, scalars: Vec<u8>) -> RenderPassDescriptor {
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
        vertices: u32::try_from(INDICES.len() / 2).expect("index count"),
        vertex_buffers: vec![
            BufferView {
                view_id: POSITION_VIEW,
                metal_binding: 0,
                allocation_id: POSITION_ALLOCATION,
                offset: 0,
                length: u64::try_from(positions().len()).expect("position length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(positions()),
            },
            BufferView {
                view_id: SCALAR_VIEW,
                metal_binding: 1,
                allocation_id: SCALAR_ALLOCATION,
                offset: 0,
                length: u64::try_from(scalars.len()).expect("stream length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(scalars),
            },
        ],
        indices: Some(IndexBufferBinding {
            view: BufferView {
                view_id: INDEX_VIEW,
                metal_binding: 0,
                allocation_id: INDEX_ALLOCATION,
                offset: 0,
                length: INDICES.len() as u64,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(INDICES.to_vec()),
            },
            format: IndexFormat::Uint16,
        }),
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

fn trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    scalars: Vec<u8>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let pass = render_pass(render.pipeline_id, scalars.clone());
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(61),
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
                        length: 4 * 16,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(16)),
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
            TracePass::Render(pass),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, 4_u64 * 16),
        (SCRATCH_ALLOCATION, 8),
        (
            POSITION_ALLOCATION,
            u64::try_from(positions().len()).expect("position length"),
        ),
        (
            SCALAR_ALLOCATION,
            u64::try_from(scalars.len()).expect("stream length"),
        ),
        (INDEX_ALLOCATION, INDICES.len() as u64),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    (trace, resources)
}

/// Submit one scalar-stream trace and return the attachment's readback.
fn submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    scalars: Vec<u8>,
) -> Vec<u8> {
    let (trace, resources) = trace(provider, compute, render, scalars);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the scalar trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback")
}

/// The frame the two rectangles land, column by column: the left half carries
/// the left rectangle's texel and the right half the right one's.
fn expected_frame(left: [u8; 4], right: [u8; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 * 4);
    for _row in 0..EXTENT {
        bytes.extend_from_slice(&left);
        bytes.extend_from_slice(&left);
        bytes.extend_from_slice(&right);
        bytes.extend_from_slice(&right);
    }
    bytes
}

/// The two arms one reading compares, registered once.
struct Fixture {
    provider: VulkanComputeProvider,
    compute: CompiledComputePipeline,
    render: CompiledComputePipeline,
}

fn fixture() -> Option<Fixture> {
    let (executor, provider) = provider_with_device()?;
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(&provider, &executor);
    Some(Fixture {
        provider,
        compute,
        render,
    })
}

impl Fixture {
    /// Submit one pass whose scalar stream carries `left` and `right`.
    fn run_pass(&self, left: [f32; 2], right: [f32; 2]) -> Vec<u8> {
        submit(
            &self.provider,
            &self.compute,
            &self.render,
            scalars(left, right),
        )
    }
}

/// One draw runs twice: the second pass takes every upload buffer the first one
/// built, and the frame it lands is the frame the fresh path landed.
#[test]
fn a_repeated_draw_is_served_from_the_pool_and_lands_the_same_bytes() {
    let Some(fixture) = fixture() else {
        return;
    };
    let provider = &fixture.provider;

    // Arm one: the mechanism on, which is its default.
    provider.set_render_buffer_pool(true);
    let before: RenderBufferPoolCounts = provider.render_buffer_pool_counts();
    let first = fixture.run_pass(LEFT_SCALARS, RIGHT_SCALARS);
    let after_first = provider.render_buffer_pool_counts();
    let second = fixture.run_pass(LEFT_SCALARS, RIGHT_SCALARS);
    let after_second = provider.render_buffer_pool_counts();

    eprintln!(
        "pool on: first {} second {}; hits {} -> {} -> {}, misses {}, entries {}, held {} bytes",
        hex(&first),
        hex(&second),
        before.hits,
        after_first.hits,
        after_second.hits,
        after_second.misses,
        after_second.entries,
        after_second.held_bytes,
    );
    let expected = expected_frame(LEFT_TEXEL, RIGHT_TEXEL);
    assert_eq!(
        first,
        expected,
        "the frame is the fetched floats' own bytes: {}",
        hex(&first)
    );
    assert_eq!(
        second, first,
        "a pass served from the pool lands exactly the frame the fresh path landed"
    );
    assert_eq!(
        after_first.hits - before.hits,
        0,
        "the first pass of a shape has nothing to take"
    );
    assert_eq!(
        after_first.misses - before.misses,
        UPLOADS_PER_PASS,
        "the first pass builds one buffer per upload binding"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        UPLOADS_PER_PASS,
        "the second pass takes one buffer per upload binding"
    );
    assert_eq!(
        after_second.misses - after_first.misses,
        0,
        "a served creation is never also counted as a build"
    );
    assert_eq!(
        after_second.returns - after_first.returns,
        UPLOADS_PER_PASS,
        "a completed pass hands every buffer it took back"
    );
    assert_eq!(
        after_second.entries, UPLOADS_PER_PASS as usize,
        "the pool holds one buffer per shape this draw states"
    );
    assert!(
        after_second.held_bytes > 0,
        "a held buffer is an allocation the pool kept"
    );
    assert_eq!(
        (after_second.evictions, after_second.flushes),
        (0, 0),
        "one draw's shapes are far below the cap"
    );

    // Arm two: the same two passes with the mechanism switched off. The switch
    // drops what it held, and the fresh path lands the same bytes.
    provider.set_render_buffer_pool(false);
    let off_before = provider.render_buffer_pool_counts();
    assert_eq!(
        off_before.entries, 0,
        "switching the mechanism off drops what it held"
    );
    let third = fixture.run_pass(LEFT_SCALARS, RIGHT_SCALARS);
    let fourth = fixture.run_pass(LEFT_SCALARS, RIGHT_SCALARS);
    let off_after = provider.render_buffer_pool_counts();

    eprintln!(
        "pool off: third {} fourth {}; hits {} -> {}, disabled {} -> {}, entries {}",
        hex(&third),
        hex(&fourth),
        off_before.hits,
        off_after.hits,
        off_before.disabled,
        off_after.disabled,
        off_after.entries,
    );
    assert_eq!(
        (third, fourth),
        (first.clone(), second),
        "the two arms of the reading land the same bytes"
    );
    assert_eq!(
        off_after.hits, off_before.hits,
        "nothing is served while the switch is off"
    );
    assert_eq!(
        off_after.disabled - off_before.disabled,
        2 * UPLOADS_PER_PASS,
        "every creation of both passes reports the switch, not a miss"
    );
    assert_eq!(off_after.entries, 0, "the switch off holds nothing");
    assert_eq!(
        off_after.returns - off_before.returns,
        0,
        "nothing is handed back while the switch is off"
    );
}

/// The second arm states different bytes for the same shapes: the served buffer
/// must carry the *new* payload, not the one the previous pass left there.
#[test]
fn a_served_buffer_carries_the_new_arms_bytes() {
    let Some(fixture) = fixture() else {
        return;
    };
    let provider = &fixture.provider;
    provider.set_render_buffer_pool(true);

    let first = fixture.run_pass(LEFT_SCALARS, RIGHT_SCALARS);
    let before = provider.render_buffer_pool_counts();
    let second = fixture.run_pass(RIGHT_SCALARS, LEFT_SCALARS);
    let after = provider.render_buffer_pool_counts();

    assert_eq!(first, expected_frame(LEFT_TEXEL, RIGHT_TEXEL));
    assert_eq!(
        second,
        expected_frame(RIGHT_TEXEL, LEFT_TEXEL),
        "the served buffer holds this arm's bytes: {}",
        hex(&second)
    );
    assert_eq!(
        after.hits - before.hits,
        UPLOADS_PER_PASS,
        "the second arm is served rather than rebuilt"
    );
    assert_eq!(
        after.misses - before.misses,
        0,
        "the second arm builds nothing of its own"
    );
}
