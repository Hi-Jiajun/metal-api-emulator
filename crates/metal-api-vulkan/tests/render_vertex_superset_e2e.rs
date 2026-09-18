//! The declared-superset vertex interface (`research/docs/23` §3.3, E-TX11).
//!
//! Census v27b's largest first-failure bucket is `vertex_interface`: 5,241
//! refused draws (50.2% of the round's refusals), and the R-VI1 direction split
//! reads *all* of it as `declared ⊇ reflected` — a contract whose vertex layout
//! names attribute locations the translated vertex module never reads. Metal
//! allows exactly that shape (`MTLVertexDescriptor` may name a location the
//! function ignores, and the stream is bound and ignored), and so does this
//! rail's execution: the pipeline's vertex input state is built from the
//! *contract's* layout, one `VkVertexInputAttributeDescription` per declared
//! attribute, while the module consumes only the locations it declares.
//!
//! The readings this file pins, all on one device and with the fixture's own
//! definitional frame as the oracle:
//!
//! * a registration whose layout declares **four** attributes on one stream
//!   while the module reads **two** lands exactly the frame the two-attribute
//!   control lands — same stride, same bytes, same buffer view, so the only
//!   difference between the two runs is the declared attribute set;
//! * replacing the two ignored attributes' bytes with other values does **not**
//!   move the frame: they are bound with their stream and ignored;
//! * replacing the bytes of an attribute the module *does* read moves the frame
//!   in exactly the column that attribute decides;
//! * the same frame comes out of the **object rail** byte for byte;
//! * the reverse direction — a location the module reads that no declared
//!   attribute covers — is still refused **by name**, with the location in its
//!   fields and no pipeline identity consumed.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The translated pair: the two-stream AIR module of R6 (`research/docs/26`
/// §14), whose reflection reads a `float2` position at location 0 and a `float2`
/// offset at location 1, plus the milestone's solid fragment module. The
/// superset shape is stated around this pair, because the module is what makes
/// the ignored attributes falsifiable: it reads two locations out of the four
/// its contract declares.
const VERTEX_AIR: &str = include_str!("fixtures/reims_indexed_tri_two_stream.ll");
const VERTEX_ENTRY: &str = "reims_two_stream_vertex";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2.frag.ll");
const FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// The declaring compute kernel (`research/docs/23` §3.6): the trace has to
/// declare the attachment view, and a compute pass that only reads it is the
/// sharing core admission admits.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The covering triangle the fixture draws, before the offset is added.
const POSITIONS: [(f32, f32); 3] = [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)];

/// The offset the module reads at location 1. It drops the triangle so that it
/// covers three of the 2x2 texels and misses the second one of the first row,
/// so a rail that dropped the stream — or bound the wrong attribute there —
/// cannot land the expected bytes, and the frame names both the fragment's
/// texel and the clear colour.
const OFFSET: (f32, f32) = (0.0, -1.75);

/// The offset that moves the frame: shifting the triangle to the right covers
/// the second column and leaves the first one clear, which is a different byte
/// string on every row.
const MOVED_OFFSET: (f32, f32) = (0.75, 0.0);

/// The bytes the *ignored* attributes carry. The values are far outside the
/// clip space and pairwise different, so a rail that read location 2 or 3 as an
/// input — or bound one of them to the wrong location — rasterizes a different
/// frame instead of the expected one.
const IGNORED_SENTINELS: [(f32, f32); 2] = [(1000.0, 1000.0), (-1000.0, -1000.0)];
const OTHER_IGNORED_SENTINELS: [(f32, f32); 2] = [(-512.0, 512.0), (256.0, -256.0)];

/// The `LoadOp::Clear` sentinel: a texel still holding it proves the draw did
/// not cover that pixel.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// `(64/255, 128/255, 192/255, 1)` as an 8-bit UNORM attachment stores it:
/// `40 80 c0 ff`, the fragment fixture's own texel.
const EXPECTED_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const ATTACHMENT_VIEW: ViewId = ViewId::new(971);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(972);
const SCRATCH_VIEW: ViewId = ViewId::new(973);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(974);
const STREAM_VIEW: ViewId = ViewId::new(975);
const STREAM_ALLOCATION: AllocationId = AllocationId::new(976);

/// One vertex record: the position, the offset the module reads, and the two
/// `float2` attributes the module never reads. One stream, four attributes,
/// stride 32.
const RECORD_BYTES: usize = 32;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("render-vertex-superset-fixture-v1", case.to_vec()).expect("digest")
}

fn provider_with_device() -> Option<(Arc<VulkanExecutor>, Arc<VulkanComputeProvider>)> {
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

/// One stream's bytes: three records of a position, the offset, and the two
/// ignored attributes.
fn stream_bytes(offset: (f32, f32), ignored: [(f32, f32); 2]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(POSITIONS.len() * RECORD_BYTES);
    for position in POSITIONS {
        for (x, y) in [position, offset, ignored[0], ignored[1]] {
            bytes.extend_from_slice(&x.to_ne_bytes());
            bytes.extend_from_slice(&y.to_ne_bytes());
        }
    }
    bytes
}

/// The superset contract's layout: one stream, stride 32, and **four**
/// attributes — the two the module reads at offsets 0 and 8, and the two it
/// ignores at offsets 16 and 24.
fn superset_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: RECORD_BYTES as u64,
        step: VertexStep::PerVertex,
        attributes: (0..4)
            .map(|location| VertexAttribute {
                location,
                offset: u64::from(location) * 8,
                format: VertexFormat::Float32x2,
            })
            .collect(),
    }])
}

/// The control contract's layout: the same stream and the same stride, with
/// only the two attributes the module reads. Everything else about the two runs
/// — the bytes, the view, the pass, the module — is identical, so the frames
/// differing would be the declared attribute set's doing.
fn read_only_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: RECORD_BYTES as u64,
        step: VertexStep::PerVertex,
        attributes: vec![
            VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            },
            VertexAttribute {
                location: 1,
                offset: 8,
                format: VertexFormat::Float32x2,
            },
        ],
    }])
}

fn contract(layout: VertexLayout) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: layout,
        textures: Vec::new(),
    }
}

/// Register the fixture's pair under `layout`, translating both stages through
/// the rail's own entry point the way a host feeding guest AIR would.
fn register(
    executor: &Arc<VulkanExecutor>,
    provider: &VulkanComputeProvider,
    layout: VertexLayout,
    case: &[u8],
) -> Result<
    metal_api_core::provider::CompiledComputePipeline,
    metal_api_core::provider::ProviderError,
> {
    let policy = provider.spirv_feature_policy();
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(VERTEX_AIR)
        .expect("the vertex fixture loads");
    let function = library
        .function(VERTEX_ENTRY)
        .expect("the vertex entry exists");
    let vertex =
        TranslatedRenderStage::translate_with_policy(RenderStage::Vertex, &function, policy)
            .expect("the vertex stage translates");
    let library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment =
        TranslatedRenderStage::translate_with_policy(RenderStage::Fragment, &function, policy)
            .expect("the fragment stage translates");
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(layout),
        vertex,
        fragment,
        logical_digest: digest(case),
    })
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
        .compile_pipeline(&function, digest(b"vertex-superset-declaring"))
        .expect("the compute pipeline registers")
}

fn render_pass(pipeline: metal_api_core::provider::PipelineId) -> RenderPassDescriptor {
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
            width: 2,
            height: 2,
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, 2, 2],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// One trace carrying the declaring compute pass and one render pass that binds
/// `bytes` as the pipeline's own stream.
fn trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    bytes: Vec<u8>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id);
    let length = u64::try_from(bytes.len()).expect("stream length");
    pass.vertex_buffers = vec![BufferView {
        view_id: STREAM_VIEW,
        metal_binding: 0,
        allocation_id: STREAM_ALLOCATION,
        offset: 0,
        length,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(bytes),
    }];
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(23),
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
                        length: 16,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(4)),
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
        (ATTACHMENT_ALLOCATION, 16_u64),
        (SCRATCH_ALLOCATION, 8),
        (STREAM_ALLOCATION, length),
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

/// Submit one trace over `bytes` and return the attachment's readback bytes.
fn submit_frame(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    bytes: Vec<u8>,
    what: &str,
) -> Vec<u8> {
    let (trace, resources) = trace(provider, compute, render, bytes);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the superset trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let frame = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback");
    eprintln!("{what} superset attachment readback: {}", hex(&frame));
    frame
}

/// The coverage the fixture's own triangle decides, as a row-major table over
/// the 2x2 attachment: `#` is a texel the fragment covers, `.` one it misses.
///
/// The module adds the offset at location 1 to every position, so the triangle
/// is `(-1,-2.75) (3,-2.75) (-1,1.25)` in Metal's +y-up clip space: a right
/// triangle whose vertical edge is `x = -1` and whose hypotenuse runs
/// `x = 0.25 - y`. Sampling at pixel centres puts three texels inside it — the
/// whole second row and the first texel of the first — and leaves the second
/// texel of the first row outside, where the hypotenuse sits at `x = -0.25`
/// against the pixel's `x = 0.5`. The translated vertex stage negates y for
/// Vulkan (E-TX11 keeps the alignment the earlier increments measured), so the
/// image keeps Metal's own orientation and this table is the frame.
const COVERAGE: [&str; 2] = ["#.", "##"];

/// The frame the coverage table describes, in the fixture's own bytes: the
/// fragment's texel where covered, the pass's clear sentinel elsewhere. Stated
/// from the fixture's geometry rather than from any rail's readback.
fn expected_frame() -> Vec<u8> {
    let mut frame = Vec::with_capacity(16);
    for row in COVERAGE {
        for texel in row.chars() {
            if texel == '#' {
                frame.extend_from_slice(&EXPECTED_TEXEL);
            } else {
                frame.extend_from_slice(&CLEAR_SENTINEL);
            }
        }
    }
    frame
}

/// The object rail's frame over the same registration and the same bytes: one
/// buffer holding the stream, one attachment holding the render area, and the
/// same pipeline wrapped for the object API.
fn object_frame(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    bytes: Vec<u8>,
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the superset registration wraps for the object API");
    let attachment = device
        .new_buffer_with_bytes(vec![0x00; 16])
        .expect("the attachment's landing buffer is declared");
    let attachment_view = attachment.view(0, 16).expect("the attachment view");
    let stream = device
        .new_buffer_with_bytes(bytes)
        .expect("the vertex stream is declared");
    let stream_view = stream
        .view(0, POSITIONS.len() * RECORD_BYTES)
        .expect("the stream view is declared at the record stride's own length");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"vertex-superset-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let scratch = device
            .new_buffer_with_bytes(vec![0xab; 4])
            .expect("the scratch buffer is declared");
        let scratch_view = scratch.view(0, 4).expect("the scratch view");
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(&declaring)
            .expect("compute pipeline state");
        encoder
            .set_buffer(0, &attachment_view)
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
        let mut encoder = command.render_command_encoder().expect("render encoder");
        encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the superset pipeline is bound");
        encoder
            .set_vertex_buffer(0, &stream_view)
            .expect("the stream binds at binding 0");
        encoder
            .draw_primitives(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                u32::try_from(POSITIONS.len()).expect("three vertices"),
                None,
            )
            .expect("the superset pass records");
        encoder.end_encoding().expect("the render encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    attachment
        .read()
        .expect("the attachment's landing bytes are readable")
}

/// The superset registration lands the read streams' frame, and so does the
/// two-attribute control over the same bytes and the same buffer: the declared
/// attributes the module ignores are bound and ignored.
#[test]
fn a_declared_superset_layout_lands_the_read_streams_frame() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let superset = register(
        &executor,
        &provider,
        superset_layout(),
        b"vertex-superset-four-attributes",
    )
    .expect("a layout declaring more attributes than the module reads registers");
    let control = register(
        &executor,
        &provider,
        read_only_layout(),
        b"vertex-superset-two-attributes",
    )
    .expect("the two-attribute control registers");

    let bytes = stream_bytes(OFFSET, IGNORED_SENTINELS);
    let expected = expected_frame();
    let superset_frame = submit_frame(&provider, &compute, &superset, bytes.clone(), "superset");
    let control_frame = submit_frame(&provider, &compute, &control, bytes, "control");
    assert_eq!(
        superset_frame, control_frame,
        "the declared attributes the module does not read must not move the frame"
    );
    assert_eq!(
        superset_frame,
        expected,
        "the superset frame is the fixture's own two-attribute definition: {}",
        hex(&superset_frame)
    );
    eprintln!(
        "superset and control agree on [{}] on {}",
        hex(&superset_frame),
        executor.device_name()
    );
}

/// The ignored attributes are *bound* rather than dropped: replacing their
/// bytes leaves the frame exactly where it was, while replacing the bytes of an
/// attribute the module reads moves it — and moves it in the column that
/// attribute decides.
#[test]
fn the_ignored_attributes_bytes_do_not_move_the_frame() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(
        &executor,
        &provider,
        superset_layout(),
        b"vertex-superset-ignored-bytes",
    )
    .expect("the superset pair registers");

    let expected = expected_frame();
    let first = submit_frame(
        &provider,
        &compute,
        &render,
        stream_bytes(OFFSET, IGNORED_SENTINELS),
        "sentinels",
    );
    let second = submit_frame(
        &provider,
        &compute,
        &render,
        stream_bytes(OFFSET, OTHER_IGNORED_SENTINELS),
        "other sentinels",
    );
    assert_eq!(
        first, expected,
        "the first sentinel run is the fixture's frame"
    );
    assert_eq!(
        second,
        first,
        "the ignored attributes' bytes are not an input: {} vs {}",
        hex(&second),
        hex(&first)
    );

    // The offset at location 1 is an input: the other direction covers the left
    // column's texels the first run left clear, so the frame moves.
    let moved = submit_frame(
        &provider,
        &compute,
        &render,
        stream_bytes(MOVED_OFFSET, IGNORED_SENTINELS),
        "moved offset",
    );
    assert_ne!(
        moved, first,
        "the offset the module reads decides the frame"
    );
    // The moved offset puts the triangle's left edge at `-0.25`, which leaves
    // the whole first column clear and covers the second one.
    let mut moved_expected = Vec::with_capacity(16);
    for _row in 0..2 {
        moved_expected.extend_from_slice(&CLEAR_SENTINEL);
        moved_expected.extend_from_slice(&EXPECTED_TEXEL);
    }
    assert_eq!(
        moved,
        moved_expected,
        "the moved offset leaves the first column clear and covers the second: {}",
        hex(&moved)
    );
    eprintln!(
        "ignored bytes left [{}] alone; the read stream moved it to [{}]",
        hex(&first),
        hex(&moved)
    );
}

/// The object rail lands the superset frame byte for byte, over the same
/// registration and the same stream bytes.
#[test]
fn the_object_rail_lands_the_superset_frame_byte_for_byte() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register(
        &executor,
        &provider,
        superset_layout(),
        b"vertex-superset-object-rail",
    )
    .expect("the superset pair registers");
    let bytes = stream_bytes(OFFSET, IGNORED_SENTINELS);
    let trace_frame = submit_frame(&provider, &compute, &render, bytes.clone(), "trace");
    let objects_frame = object_frame(&provider, &render, bytes);
    assert_eq!(
        objects_frame, trace_frame,
        "the object rail lands the trace rail's frame, byte for byte"
    );
    assert_eq!(objects_frame, expected_frame());
}

/// The reverse direction keeps its refusal by name
/// (`research/docs/23` §3.3, E-TX11): a location the module reads that no
/// declared attribute covers would leave that vertex input undefined, so the
/// registration is refused with the location in its fields and no pipeline
/// identity is consumed.
#[test]
fn a_location_the_layout_does_not_declare_is_refused_by_name() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let refused = register(
        &executor,
        &provider,
        VertexLayout::Buffers(vec![VertexBufferLayout {
            stride: RECORD_BYTES as u64,
            step: VertexStep::PerVertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        }]),
        b"vertex-superset-undeclared-location",
    )
    .expect_err("the module reads location 1 and the layout does not declare it");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("stage"),
        Some(&FieldValue::Text("vertex".to_owned()))
    );
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("vertex_attributes".to_owned()))
    );
    assert_eq!(
        refused.fields.get("location"),
        Some(&FieldValue::Unsigned(1)),
        "the refusal names the location the layout does not declare"
    );

    // The refusal happens before any device object exists, so the next
    // registration takes the identity this one did not consume.
    let registered = register(
        &executor,
        &provider,
        read_only_layout(),
        b"vertex-superset-identity",
    )
    .expect("the direction the layout does declare registers");
    assert_eq!(registered.pipeline_id.get(), 1);
}
