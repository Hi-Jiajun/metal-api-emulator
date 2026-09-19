//! The scalar `float32` vertex lane, end to end (2026-09-20, census v46's
//! `vertex_format` bucket).
//!
//! The falsifiable claim is the census's own equality: for the two pipes that
//! carry the bucket, `store_routes`' `vertex_format_float` count equals
//! `out_of_class_vertex_format` window by window (8 = 8, 16 = 16), while every
//! other declared storage is already inside the canonical set — so the one
//! format the refusals sit in is `MTL_VERTEX_FORMAT_FLOAT`, the scalar
//! `float32x1` lane the contract did not name.
//!
//! One translated vertex stage reads three attributes: a `float32x2` position
//! and *two* scalar `float32x1` storages that share a stream at offsets zero
//! and four. It forwards the two scalars as red and green (blue zero, alpha the
//! sentinel), so the attachment's texels are the fetched floats' own 8-bit
//! spellings with no arithmetic in between:
//!
//! * the left half of a 4x4 `Rgba8Unorm` attachment carries `33 99 00 ff` and
//!   the right half `99 33 00 ff` — `0.2f * 255` is `51.000001` and
//!   `0.6f * 255` is `153.000006`, so both readings of the float-to-unorm
//!   conversion land the same byte and the comparison stays a byte comparison;
//! * the second scalar's *offset* is load-bearing: a rail that fetched both
//!   storages from offset zero would land `33 33` on the left half instead;
//! * swapping the two rectangles' scalar records leaves the positions alone and
//!   swaps the frame, which is what "the frame follows the vertex bytes" means
//!   as a reading rather than a claim;
//! * a contract that declares a *vector* storage where the shader's own AIR
//!   member is scalar is refused by name at registration
//!   (`render_stage_reflection_mismatch`, fields `location`, `format_code`,
//!   `type_name`), and a snapshot that does not declare the lane refuses the
//!   trace at admission (`vertex_format_unsupported`) — the two named refusals
//!   that keep "not this storage" from being read as "any storage".

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LoadOp,
    OperationId, PipelineId, ProviderCapabilities, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed pair: a vertex stage whose `[[stage_in]]` interface is the
/// three attributes the increment's layout declares, and a fragment stage that
/// stores the forwarded tint.
const VERTEX_AIR: &str = include_str!("fixtures/render_scalar_quad.vert.ll");
const VERTEX_ENTRY: &str = "render_scalar_quad_vertex";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_scalar_tint.frag.ll");
const FRAGMENT_ENTRY: &str = "render_scalar_tint";

/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
const COPY_WORD_ENTRY: &str = "copy_word";

const ATTACHMENT_VIEW: ViewId = ViewId::new(970);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(971);
const SCRATCH_VIEW: ViewId = ViewId::new(972);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(973);
const POSITION_VIEW: ViewId = ViewId::new(974);
const POSITION_ALLOCATION: AllocationId = AllocationId::new(975);
const SCALAR_VIEW: ViewId = ViewId::new(976);
const SCALAR_ALLOCATION: AllocationId = AllocationId::new(977);
const INDEX_VIEW: ViewId = ViewId::new(978);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(979);

/// 4x4, the extent both rectangles split down the middle.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The position stream's stride and the two corners each rectangle is built
/// from: `x` spans one half of the clip-space square, `y` the whole height, so
/// the two rectangles cover disjoint texel columns whatever the y orientation.
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

/// The scalar stream's stride and the two attributes' offsets: two four-byte
/// scalar lanes inside one eight-byte record.
const SCALAR_STRIDE: usize = 8;
const RED_OFFSET: u64 = 0;
const GREEN_OFFSET: u64 = 4;

/// The scalars the two rectangles carry, and the texels they land. `0.2f` is
/// `0x3e4ccccd`, whose `× 255` is `51.000001`; `0.6f` is `0x3f19999a`, whose
/// `× 255` is `153.000006` — both unambiguous under a rounding *and* under a
/// truncating float-to-unorm conversion.
const LEFT_SCALARS: [f32; 2] = [0.2, 0.6];
const RIGHT_SCALARS: [f32; 2] = [0.6, 0.2];
const LEFT_TEXEL: [u8; 4] = [0x33, 0x99, 0x00, 0xff];
const RIGHT_TEXEL: [u8; 4] = [0x99, 0x33, 0x00, 0xff];

/// The 12 indices naming the two rectangles' six vertices each.
const INDICES: [u8; 24] = [
    0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00,
    0x08, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x0b, 0x00,
];

/// Six copies of one rectangle's scalar record, in the vertex order
/// [`POSITION_RECORDS`] names.
fn half(scalars: [f32; 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(SCALAR_STRIDE * 6);
    for _ in 0..6 {
        for component in scalars {
            bytes.extend_from_slice(&component.to_ne_bytes());
        }
    }
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
    SemanticDigest::new("render-scalar-vertex-fixture-v1", case.to_vec()).expect("digest")
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

/// The layout the increment admits: the position stream beside the scalar
/// stream, whose two attributes share one record at offsets zero and four.
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

fn scalar_contract(layout: VertexLayout) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: layout,
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
        .compile_pipeline(&function, digest(b"render-scalar-vertex-compute"))
        .expect("the compute pipeline registers")
}

/// Register the pair under one contract, so a caller can state a layout the
/// fixture's own reflection disagrees with (the refusal arm) as easily as the
/// reviewed one.
fn register(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    layout: VertexLayout,
    what: &[u8],
) -> Result<CompiledComputePipeline, metal_api_core::provider::ProviderError> {
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
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: scalar_contract(layout),
        vertex,
        fragment,
        logical_digest: digest(what),
    })
}

/// One render pass over the 4x4 attachment, with the two streams and the index
/// buffer bound. `scalars` is the scalar stream's bytes, so an arm states what
/// the *vertices* carry rather than which case it is.
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
        operation_id: OperationId::new(31),
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
    what: &str,
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
    let bytes = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback");
    eprintln!("{what} attachment readback: {}", hex(&bytes));
    bytes
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

/// The scalar lane is fetched, from each attribute's own offset, and the
/// attachment's bytes are the stored floats' own spellings.
#[test]
fn the_scalar_lane_lands_its_own_bytes() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    // The widening the trace is admitted against: the consumer's snapshot
    // carries the lane, which is what makes the admission below a reading of
    // the capability list rather than of a hard-coded window.
    assert!(
        provider
            .capabilities()
            .supported_vertex_formats
            .contains(&VertexFormat::Float32x1),
        "the rail's own snapshot declares the scalar lane"
    );
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(&provider, &executor, scalar_layout(), b"scalar-reviewed")
        .expect("the reviewed scalar pair registers");

    let mut scalars = half(LEFT_SCALARS);
    scalars.extend_from_slice(&half(RIGHT_SCALARS));
    let bytes = submit(&provider, &compute, &render, scalars, "reviewed");
    let expected = expected_frame(LEFT_TEXEL, RIGHT_TEXEL);
    assert_eq!(
        bytes,
        expected,
        "the frame is the fetched floats' own bytes: {}",
        hex(&bytes)
    );
    eprintln!(
        "two float32x1 attributes (offsets 0 and 4 of one record) landed [{}] on {}",
        hex(&bytes),
        executor.device_name()
    );
}

/// Swapping the two rectangles' vertex bytes swaps the frame while every
/// position stays where it was — the reading that says the *streams* are what
/// the texels come from.
#[test]
fn swapping_the_scalar_bytes_swaps_the_frame() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(&provider, &executor, scalar_layout(), b"scalar-swapped")
        .expect("the reviewed scalar pair registers");

    let mut swapped = half(RIGHT_SCALARS);
    swapped.extend_from_slice(&half(LEFT_SCALARS));
    let bytes = submit(&provider, &compute, &render, swapped, "swapped");
    let expected = expected_frame(RIGHT_TEXEL, LEFT_TEXEL);
    assert_eq!(
        bytes,
        expected,
        "the swapped streams land the swapped frame: {}",
        hex(&bytes)
    );
    assert_ne!(
        bytes,
        expected_frame(LEFT_TEXEL, RIGHT_TEXEL),
        "the two arms are different frames, so the comparison above is a reading"
    );
}

/// A storage whose component shape the shader's own AIR member does not read is
/// refused at registration, by name, with the location and the storage that
/// disagreed — the direction that keeps the scalar lane from being read as a
/// relabelling of its two-component sibling.
#[test]
fn a_storage_the_shader_cannot_read_is_refused_by_name() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let mut layout = scalar_layout();
    let VertexLayout::Buffers(buffers) = &mut layout else {
        unreachable!("the fixture is a buffer layout")
    };
    // Location 1 is the scalar member: declaring the two-component `float32x2`
    // there is a storage whose component shape the shader's own scalar read
    // does not cover.
    buffers[1].attributes[0].format = VertexFormat::Float32x2;
    let error = match register(&provider, &executor, layout, b"scalar-shape-mismatch") {
        Ok(_) => panic!("a shape the fixture's reflection does not read registers"),
        Err(error) => error,
    };
    eprintln!("shape mismatch refusal: {error:?}");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(error.slug, "render_stage_reflection_mismatch");
    let text = |name: &str| match error.fields.get(name) {
        Some(metal_api_core::provider::FieldValue::Text(value)) => value.clone(),
        other => panic!("{name} is not text: {other:?}"),
    };
    assert_eq!(
        error.fields.get("field"),
        Some(&metal_api_core::provider::FieldValue::Text(
            "vertex_attributes".to_owned()
        ))
    );
    assert_eq!(
        error.fields.get("location"),
        Some(&metal_api_core::provider::FieldValue::Unsigned(1)),
        "the refusal names the attribute"
    );
    assert_eq!(
        error.fields.get("format_code"),
        Some(&metal_api_core::provider::FieldValue::Unsigned(u64::from(
            VertexFormat::Float32x2.code()
        ))),
        "the refusal names the storage the contract declared"
    );
    assert_eq!(
        text("type_name"),
        "float",
        "the refusal names the shape the shader reads"
    );
    assert!(
        error
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("not the component shape")),
        "the refusal states what disagreed: {:?}",
        error.detail
    );
}

/// A snapshot that does not declare the scalar lane refuses the trace at
/// admission under the contract's own slug — the fail-closed arm a rail keeps
/// while its device has no observation for the storage (`metal-api-native`'s
/// window), and the arm the census's own pre-increment provider is on.
#[test]
fn a_snapshot_without_the_scalar_lane_refuses_the_trace_by_name() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(&provider, &executor, scalar_layout(), b"scalar-closed")
        .expect("the reviewed scalar pair registers");

    let mut narrow: ProviderCapabilities = provider.capabilities();
    narrow.supported_vertex_formats = vec![
        VertexFormat::Float32x2,
        VertexFormat::Float32x3,
        VertexFormat::Float32x4,
        VertexFormat::Uint32,
        VertexFormat::Unorm8x2,
        VertexFormat::Unorm8x4,
        VertexFormat::Unorm16x2,
        VertexFormat::Unorm16x4,
    ];
    let mut scalars = half(LEFT_SCALARS);
    scalars.extend_from_slice(&half(RIGHT_SCALARS));
    let (trace, resources) = trace(&provider, &compute, &render, scalars);
    let error = match narrow.validate_trace(trace, resources) {
        Ok(_) => panic!("a snapshot without the scalar lane admits the trace"),
        Err(error) => error,
    };
    eprintln!("narrowed-snapshot refusal: {error:?}");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(error.slug, "vertex_format_unsupported");
}
