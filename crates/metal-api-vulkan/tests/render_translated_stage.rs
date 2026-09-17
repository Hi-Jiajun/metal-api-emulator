//! Translated render stages, end to end (`research/docs/23`, R2 increment).
//!
//! The rail's second registration arm: a render pipeline whose two stage
//! modules came out of metal2vulkan instead of the reviewed `render_spv/` set.
//! The case is the v16 full-screen triangle shape
//! (`conformance/suite-v16.json`'s `render_offscreen_2x2`, here as the owned AIR
//! pair under `tests/fixtures/`): one 2x2 `Rgba8Unorm` attachment, cleared to a
//! sentinel, then covered by a 3-vertex draw whose fragment stage stores
//! `(64/255, 128/255, 192/255, 1)`.
//!
//! What this file measures:
//!
//! * the translated pair executes through the whole chain — `ComputeTrace` with a
//!   render entry -> `ProviderCapabilities::validate_trace` ->
//!   `ComputeProvider::submit` -> the attachment's bytes in the writebacks — and
//!   lands `40 80 c0 ff` per texel, byte for byte the same readback the reviewed
//!   module pair lands on the same device;
//! * the registration gate refuses a translation that does not describe the
//!   contract it is registered under (`render_stage_reflection_mismatch`), one
//!   that names interface the rail does not execute
//!   (`render_stage_unsupported_interface`), and a stage module the rail has no
//!   account of at all (`render_stage_translation_unavailable`);
//! * every refusal happens before the provider mints a pipeline id, so nothing
//!   refused here can be submitted at all — the id sequence is what makes that
//!   measurable from the outside.

use metal_api_core::provider::{
    AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource,
    BufferView, ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy,
    ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue,
    InitialState, LoadOp, OperationId, PipelineId, PresentDescriptor, PresentMode, PresentTarget,
    ProviderError, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The owned AIR pair of the v16 shape: the reviewed milestone MSL's own two
/// stages (`conformance/shaders/render_offscreen_2x2.metal`), written as the AIR
/// the translator consumes.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2.frag.ll");

/// The strictly asymmetric fixture the NDC-y alignment is measured on: the
/// triangle `(-1,0) (1,0) (-1,1)`, which Metal's +y-up clip space maps to the
/// attachment's top half and Vulkan's +y-down clip space mirrors to its
/// bottom half (`research/docs/23` §40).
const ASYMMETRIC_VERTEX_AIR: &str = include_str!("fixtures/render_ndc_y_asymmetric.vert.ll");
const ASYMMETRIC_VERTEX_ENTRY: &str = "render_ndc_y_asymmetric";

/// The counterexample's fragment stage: the same render target, plus one Metal
/// buffer binding the rail's render stages do not bind.
const BUFFERED_FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2_buffered.frag.ll");

/// The linkage counterexample's fragment stage: it consumes a `stage_in`
/// varying the fixture's vertex stage never produces.
const VARYING_FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2_varying.frag.ll");

/// The AIR function names the two fixtures declare. The contract names these:
/// for a translated registration the contract's entries are the AIR entries the
/// reflection has to report, while the pipeline binds each module's own SPIR-V
/// entry point.
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const FRAGMENT_ENTRY: &str = "render_solid_rgba8";
const BUFFERED_FRAGMENT_ENTRY: &str = "render_buffered_rgba8";
const VARYING_FRAGMENT_ENTRY: &str = "render_varying_rgba8";

/// The reviewed pair the same shape is measured against: `spirv-as` output of
/// `render_spv/fullscreen_triangle.vert.spvasm` and
/// `render_spv/solid_unorm8.frag.spvasm`.
const REVIEWED_VERTEX_SPV: &[u8] = include_bytes!("../src/render_spv/fullscreen_triangle.vert.spv");
const REVIEWED_FRAGMENT_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
const REVIEWED_VERTEX_ENTRY: &str = "vertex_main";
const REVIEWED_FRAGMENT_ENTRY: &str = "fragment_main";

/// The reviewed compute kernel the declaring pass runs: the trace has to
/// declare the attachment view, and a compute pass that only reads it is the
/// sharing core admission admits (`AttachmentComputeConflict` refuses the
/// writable half).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The `LoadOp::Clear` sentinel: a texel still holding it proves the draw did
/// not cover that pixel.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// The R6 boundary pair, copied from reims-vgpu `d4ebd12`
/// (`crates/reims-vgpu/tests/fixtures/air/`): the same two-stream vertex module
/// with the `fadd`'s `fast` flag run present and removed. The pinned translator
/// decorates the removed-flag module with `FPFastMathMode` and demands
/// `FloatControls2` + `SPV_KHR_float_controls2` for it (`docs/23` §77, R8).
const TWO_STREAM_VERTEX_AIR: &str = include_str!("fixtures/reims_indexed_tri_two_stream.ll");
const TWO_STREAM_PRECISE_VERTEX_AIR: &str =
    include_str!("fixtures/reims_indexed_tri_two_stream_precise.ll");

/// The entry both halves of the pair declare.
const TWO_STREAM_VERTEX_ENTRY: &str = "reims_two_stream_vertex";

/// The vertices the pair draws: the two-stream positions of R6's fixture
/// `(-1,-1) (3,-1) (-1,3)` — the covering triangle — with one `float2` offset
/// added to every vertex.
const TWO_STREAM_POSITIONS: [(f32, f32); 3] = [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)];

/// The offset both rails have to read out of the *second* stream. Shifting the
/// triangle's left edge to `-0.25` leaves the 2x2 attachment's left column
/// uncovered, so a rail that dropped the stream (or bound the first stream
/// twice) cannot land the expected bytes.
const TWO_STREAM_OFFSET: (f32, f32) = (0.75, 0.0);

/// The `InitialState::Sentinel` bytes the present tail pre-fills its target
/// with: distinct from both the clear sentinel and the fragment output, so a
/// target the pass never rendered into stays falsifiable (`docs/24` §3.1).
const PRESENT_SENTINEL: [u8; 4] = [0xfd; 4];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// `(64/255, 128/255, 192/255, 1)` as an 8-bit UNORM attachment stores it:
/// `40 80 c0 ff`, four texels of a 2x2 attachment.
const EXPECTED_RGBA8_TEXELS: [u8; 16] = [
    0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff,
];

/// The coverage the asymmetric fixture's `(-1,0) (1,0) (-1,1)` triangle has
/// under Metal's +y-up NDC on an 8x4 attachment, derived from the vertex
/// positions and the pixel-centre rule rather than from any rail's readback:
/// two texels on the top row (left of the hypotenuse) and six on the second.
/// The shape is asymmetric under the y flip, so the mirrored frame is a
/// different byte string — which is what lets this fixture see a lost
/// alignment.
const ASYMMETRIC_METAL_COVERAGE: [&str; 4] = ["##......", "######..", "........", "........"];

/// The 8x4 frame the coverage above describes, in the fixture's own bytes: the
/// fragment stage's `40 80 c0 ff` where covered, the clear sentinel elsewhere.
fn asymmetric_metal_frame() -> Vec<u8> {
    let mut frame = Vec::with_capacity(8 * 4 * 4);
    for row in ASYMMETRIC_METAL_COVERAGE {
        for texel in row.chars() {
            if texel == '#' {
                frame.extend_from_slice(&EXPECTED_RGBA8_TEXELS[..4]);
            } else {
                frame.extend_from_slice(&CLEAR_SENTINEL);
            }
        }
    }
    frame
}

/// Flip a `width x 4` frame's rows: the frame a rail lands when it forgets the
/// Metal-to-Vulkan y alignment.
fn mirror_rows(frame: &[u8], width: usize) -> Vec<u8> {
    let row_bytes = width * 4;
    let rows = frame.len() / row_bytes;
    let mut mirrored = Vec::with_capacity(frame.len());
    for row in (0..rows).rev() {
        mirrored.extend_from_slice(&frame[row * row_bytes..(row + 1) * row_bytes]);
    }
    mirrored
}

const ATTACHMENT_VIEW: ViewId = ViewId::new(901);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(902);
const SCRATCH_VIEW: ViewId = ViewId::new(903);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(904);
const POSITION_VIEW: ViewId = ViewId::new(905);
const POSITION_ALLOCATION: AllocationId = AllocationId::new(906);
const OFFSET_VIEW: ViewId = ViewId::new(907);
const OFFSET_ALLOCATION: AllocationId = AllocationId::new(908);

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
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

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("render-translated-stage-fixture-v1", case.to_vec()).expect("digest")
}

/// The contract the translated pair registers under: the AIR entries the
/// translations report, one `Rgba8Unorm` attachment, and no vertex stream
/// (`VertexLayout::None` — the fixture's positions come from `[[vertex_id]]`
/// alone).
fn translated_contract(color_formats: Vec<AttachmentFormat>) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats,
        vertex_layout: VertexLayout::None,
    }
}

/// The contract the reviewed pair registers under: the entries the reviewed
/// modules declare, same attachment shape.
fn reviewed_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: REVIEWED_VERTEX_ENTRY.to_owned(),
        fragment_entry: REVIEWED_FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
    }
}

fn register_reviewed(
    provider: &VulkanComputeProvider,
) -> Result<CompiledComputePipeline, ProviderError> {
    provider.register_render_pipeline(RenderPipelineRequest {
        contract: reviewed_contract(),
        vertex_spirv: REVIEWED_VERTEX_SPV.to_vec(),
        fragment_spirv: REVIEWED_FRAGMENT_SPV.to_vec(),
        logical_digest: digest(b"reviewed-offscreen-2x2"),
    })
}

/// Translate the fixture's two stages through the rail's own entry point, the
/// way a host feeding guest AIR would.
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
    eprintln!(
        "translated vertex: {} bytes, reflection entry {:?}, attributes {}, varyings {}, \
         builtins {:?}",
        vertex.spirv().len(),
        vertex.reflection().entry_point,
        vertex.reflection().vertex_attributes.len(),
        vertex.reflection().varyings.len(),
        vertex.reflection().vertex_builtins,
    );
    let library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated fragment: {} bytes, reflection entry {:?}, render targets {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
        fragment.reflection().render_targets,
    );
    (vertex, fragment)
}

fn translate_fragment(
    executor: &Arc<VulkanExecutor>,
    source: &str,
    entry: &str,
) -> TranslatedRenderStage {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .expect("the fragment fixture loads");
    let function = library.function(entry).expect("the fragment entry exists");
    TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates")
}

/// Compile the declaring compute kernel (`copy_word`) on this provider.
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
        .compile_pipeline(&function, digest(b"render-translated-stage-compute"))
        .expect("the compute pipeline registers")
}

fn render_pass(pipeline: PipelineId, present: Option<PresentDescriptor>) -> RenderPassDescriptor {
    render_pass_sized(pipeline, present, 2, 2)
}

/// One render pass over a `width x height` attachment: the pass the 2x2
/// milestone fixtures use, and the larger one the asymmetric NDC-y fixture
/// needs to see which rows are covered.
fn render_pass_sized(
    pipeline: PipelineId,
    present: Option<PresentDescriptor>,
    width: u32,
    height: u32,
) -> RenderPassDescriptor {
    RenderPassDescriptor {
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
            width: u64::from(width),
            height: u64::from(height),
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, width, height],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present,
    }
}

/// The present tail the R4a case attaches to the fixture's render pass: the
/// pass's own attachment view is the target, handed on once in `Fifo` mode
/// with a blocking acquire (`research/docs/24` §3.6).
fn present_tail() -> PresentDescriptor {
    PresentDescriptor {
        target: PresentTarget {
            allocation_id: ATTACHMENT_ALLOCATION,
            view_id: ATTACHMENT_VIEW,
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            image_count: 1,
            initial: InitialState::Sentinel(PRESENT_SENTINEL.to_vec()),
        },
        source: ATTACHMENT_VIEW,
        mode: PresentMode::Fifo,
        acquire: AcquirePolicy::Blocking,
    }
}

/// One trace carrying the declaring compute pass and one render pass that names
/// `render`, plus the resource table it has to be admitted against. The
/// declaring compute pass reads the attachment's bytes, whose length follows
/// the `width x height` attachment the render pass covers.
fn trace_for_sized(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    present: Option<PresentDescriptor>,
    width: u32,
    height: u32,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(width) * u64::from(height) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(21),
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
            TracePass::Render(render_pass_sized(
                render.pipeline_id,
                present,
                width,
                height,
            )),
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

/// Submit one trace and return the attachment's readback bytes, printing them.
fn submit_for_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    present: Option<PresentDescriptor>,
    what: &str,
) -> Vec<u8> {
    submit_sized_for_readback(provider, compute, render, present, 2, 2, what)
}

/// The same submission over a `width x height` attachment.
#[allow(clippy::too_many_arguments)]
fn submit_sized_for_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    present: Option<PresentDescriptor>,
    width: u32,
    height: u32,
    what: &str,
) -> Vec<u8> {
    let (trace, resources) = trace_for_sized(provider, compute, render, present, width, height);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the render-bearing trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let bytes = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback");
    eprintln!("{what} attachment readback: {}", hex(&bytes));
    bytes
}

/// The milestone's own bytes, from the reviewed module pair, are the reference
/// the translated pair has to match.
#[test]
fn translated_stages_land_the_same_bytes_as_the_reviewed_pair() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let reviewed = register_reviewed(&provider).expect("the reviewed pair registers");
    let (vertex, fragment) = translated_pair(&executor);
    let translated = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: translated_contract(vec![AttachmentFormat::Rgba8Unorm]),
            vertex,
            fragment,
            logical_digest: digest(b"translated-offscreen-2x2"),
        })
        .expect("the translated pair registers");

    let reviewed_bytes = submit_for_readback(&provider, &compute, &reviewed, None, "reviewed");
    let translated_bytes =
        submit_for_readback(&provider, &compute, &translated, None, "translated");
    eprintln!("expected: [{}] x4", hex(&EXPECTED_RGBA8_TEXELS[..4]));
    assert_eq!(reviewed_bytes, EXPECTED_RGBA8_TEXELS);
    assert_eq!(translated_bytes, EXPECTED_RGBA8_TEXELS);
    assert_eq!(
        translated_bytes, reviewed_bytes,
        "the translated pair has to land byte for byte what the reviewed pair lands"
    );
}

/// The translated path's Metal-to-Vulkan y alignment (`research/docs/23` §40).
///
/// Metal's clip space is +y up, Vulkan's is +y down, so a guest AIR vertex
/// module that writes its position unchanged lands a vertically mirrored frame
/// on the Vulkan rail. Before this increment the asymmetric fixture's coverage
/// sat on rows 2 and 3 of the 8x4 attachment instead of rows 0 and 1. The
/// translation now negates the position's y for `RenderStage::Vertex` — the
/// same `OpFNegate` the reviewed `render_spv/*.vert.spvasm` modules carry by
/// hand (`research/docs/23` §32, v38) — so the translated path lands the Metal
/// mapping. The expected frame is derived from the fixture's vertex positions,
/// not from this rail's readback, and the fixture is asymmetric under the y
/// flip, so a rail that lost the alignment lands different bytes.
#[test]
fn translated_vertex_stages_land_the_metal_ndc_mapping() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let vertex_library = device
        .new_library_with_air(ASYMMETRIC_VERTEX_AIR)
        .expect("the asymmetric fixture loads");
    let vertex_function = vertex_library
        .function(ASYMMETRIC_VERTEX_ENTRY)
        .expect("the asymmetric entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &vertex_function)
        .expect("the asymmetric vertex stage translates");
    let fragment_library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the fragment fixture loads");
    let fragment_function = fragment_library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &fragment_function)
        .expect("the fragment stage translates");
    let pipeline = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: ASYMMETRIC_VERTEX_ENTRY.to_owned(),
                fragment_entry: FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex,
            fragment,
            logical_digest: digest(b"translated-ndc-y-asymmetric-8x4"),
        })
        .expect("the asymmetric pair registers");

    let bytes =
        submit_sized_for_readback(&provider, &compute, &pipeline, None, 8, 4, "translated 8x4");
    let expected = asymmetric_metal_frame();
    eprintln!("expected (Metal mapping): {}", hex(&expected));
    assert_ne!(
        expected,
        mirror_rows(&expected, 8),
        "the fixture has to be asymmetric under the y flip, or the test could not see one"
    );
    assert_eq!(
        bytes, expected,
        "the translated vertex stage has to land the Metal NDC mapping, not its mirror"
    );
}

/// R4a (E side): a **translated** registration executes a present tail
/// (`research/docs/24` §3.6). Before this increment the present rail refused
/// any translated stage by name
/// (`render_present_translated_stage_unsupported`); the rail now binds the
/// fragment module the registration named, exactly as the offscreen rail
/// does, and the bytes, the acquire/present counters and the reused target
/// identity are the ones the reviewed rail lands.
#[test]
fn translated_stages_land_the_same_bytes_through_a_present_tail() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let reviewed = register_reviewed(&provider).expect("the reviewed pair registers");
    let (vertex, fragment) = translated_pair(&executor);
    let translated = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: translated_contract(vec![AttachmentFormat::Rgba8Unorm]),
            vertex,
            fragment,
            logical_digest: digest(b"translated-present-2x2"),
        })
        .expect("the translated pair registers");
    let (acquires_before, presents_before) = provider.present_counts();

    let reviewed_bytes = submit_for_readback(
        &provider,
        &compute,
        &reviewed,
        Some(present_tail()),
        "reviewed present",
    );
    let translated_bytes = submit_for_readback(
        &provider,
        &compute,
        &translated,
        Some(present_tail()),
        "translated present",
    );

    assert_eq!(reviewed_bytes, EXPECTED_RGBA8_TEXELS);
    assert_eq!(
        translated_bytes, reviewed_bytes,
        "the translated pair has to land byte for byte what the reviewed pair lands, \
         present tail included"
    );
    assert!(
        !translated_bytes
            .chunks_exact(4)
            .any(|texel| texel == PRESENT_SENTINEL),
        "the present target lands the fragment output, not its preset sentinel: {}",
        hex(&translated_bytes)
    );
    // One acquire and one present per present action (`docs/24` §5.3), and
    // both presents name the same (allocation, view) identity, so the target
    // is reused rather than recreated.
    assert_eq!(
        provider.present_counts(),
        (acquires_before + 2, presents_before + 2)
    );
    assert_eq!(
        provider.present_target_count(),
        1,
        "both present tails name one target identity"
    );
    eprintln!(
        "present counters after the two present tails: {:?}, target count={}",
        provider.present_counts(),
        provider.present_target_count()
    );
}

/// A translation that reads no vertex stream cannot describe a contract whose
/// vertex layout declares one — and the refusal is what keeps the pipeline from
/// being built with a vertex input state the shader never consumes.
#[test]
fn a_translation_missing_a_declared_vertex_attribute_is_refused() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let (vertex, fragment) = translated_pair(&executor);
    let mut contract = translated_contract(vec![AttachmentFormat::Rgba8Unorm]);
    contract.vertex_layout = VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: 8,
        step: VertexStep::PerVertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32x2,
        }],
    }]);
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest: digest(b"translated-missing-attribute"),
        })
        .expect_err("a contract attribute the shader does not read is a different interface");
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
        refused.fields.get("declared_attributes"),
        Some(&FieldValue::Unsigned(1))
    );
    assert_eq!(
        refused.fields.get("reflected_attributes"),
        Some(&FieldValue::Unsigned(0))
    );

    // Nothing was registered: the refusal happens before the provider mints a
    // pipeline id, so the next registration takes the id this one would have
    // taken.
    let reviewed = register_reviewed(&provider).expect("the reviewed pair registers");
    eprintln!("reviewed pipeline id: {}", reviewed.pipeline_id.get());
    assert_eq!(
        reviewed.pipeline_id.get(),
        1,
        "the refused registration consumed no pipeline identity"
    );
}

/// A translation that stores one location cannot describe a two-attachment
/// contract: the second attachment would read back bytes the stage never wrote.
#[test]
fn a_translation_with_fewer_render_targets_than_the_contract_is_refused() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let (vertex, fragment) = translated_pair(&executor);
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: translated_contract(vec![
                AttachmentFormat::Rgba8Unorm,
                AttachmentFormat::Rgba8Unorm,
            ]),
            vertex,
            fragment,
            logical_digest: digest(b"translated-target-count"),
        })
        .expect_err("one reflected render target cannot describe two attachments");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("render_targets".to_owned()))
    );
    assert_eq!(
        refused.fields.get("declared_targets"),
        Some(&FieldValue::Unsigned(2))
    );
    assert_eq!(
        refused.fields.get("reflected_targets"),
        Some(&FieldValue::Unsigned(1))
    );
}

/// The rail's render stages bind no descriptor set, so a translation that names
/// a Metal buffer is refused by name rather than executed with the binding
/// silently dropped.
#[test]
fn a_translation_with_a_buffer_binding_is_refused() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let (vertex, _) = translated_pair(&executor);
    let fragment = translate_fragment(&executor, BUFFERED_FRAGMENT_AIR, BUFFERED_FRAGMENT_ENTRY);
    eprintln!(
        "buffered fragment reflection bindings: {:?}",
        fragment
            .reflection()
            .bindings
            .iter()
            .map(|binding| (binding.metal_index, binding.kind))
            .collect::<Vec<_>>()
    );
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: VERTEX_ENTRY.to_owned(),
                fragment_entry: BUFFERED_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex,
            fragment,
            logical_digest: digest(b"translated-buffer-binding"),
        })
        .expect_err("a render stage that binds a buffer is outside this rail's interface");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_unsupported_interface");
    assert_eq!(
        refused.fields.get("stage"),
        Some(&FieldValue::Text("fragment".to_owned()))
    );
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("bindings".to_owned()))
    );
}

/// A module the rail has no account of — the translated vertex module handed to
/// the reviewed registration, which carries no reflection — is refused by name
/// instead of being executed on the strength of its bytes.
#[test]
fn an_untranslated_module_is_refused_by_name() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let (vertex, _) = translated_pair(&executor);
    let refused = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: reviewed_contract(),
            vertex_spirv: vertex.spirv().to_vec(),
            fragment_spirv: REVIEWED_FRAGMENT_SPV.to_vec(),
            logical_digest: digest(b"untranslated-vertex-module"),
        })
        .expect_err("a vertex module outside the reviewed set is not a reviewed stage");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_translation_unavailable");
    assert_eq!(
        refused.fields.get("stage"),
        Some(&FieldValue::Text("vertex".to_owned()))
    );
    assert_eq!(
        refused.fields.get("entry"),
        Some(&FieldValue::Text(REVIEWED_VERTEX_ENTRY.to_owned()))
    );
}

/// A fragment stage that consumes a varying the vertex stage never produces is
/// a linkage Vulkan would only discover at draw time, so the pair is refused at
/// registration: the two reflections have to name the same varying locations.
#[test]
fn a_translation_consuming_an_unproduced_varying_is_refused() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let (vertex, _) = translated_pair(&executor);
    let fragment = translate_fragment(&executor, VARYING_FRAGMENT_AIR, VARYING_FRAGMENT_ENTRY);
    eprintln!(
        "varying fragment reflection varyings: {:?}",
        fragment.reflection().varyings
    );
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: VERTEX_ENTRY.to_owned(),
                fragment_entry: VARYING_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex,
            fragment,
            logical_digest: digest(b"translated-unproduced-varying"),
        })
        .expect_err("a consumed varying has to be produced by the vertex stage");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("varyings".to_owned()))
    );
    assert_eq!(
        refused.fields.get("produced_varyings"),
        Some(&FieldValue::Text(String::new()))
    );
    assert_eq!(
        refused.fields.get("consumed_varyings"),
        Some(&FieldValue::Text("0".to_owned()))
    );
}

/// Translate one stage under the device's own capability policy.
///
/// The policy is the provider's answer, so the module a caller registers is the
/// module the device that will execute it validated (`docs/23` §77, R8).
fn translate_stage_with_policy(
    executor: &Arc<VulkanExecutor>,
    stage: RenderStage,
    source: &str,
    entry: &str,
    policy: metal_api_vulkan::SpirvFeaturePolicy,
) -> Result<TranslatedRenderStage, metal_api_core::ExecutorError> {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(source)
        .expect("the fixture library loads");
    let function = library.function(entry).expect("the fixture entry exists");
    TranslatedRenderStage::translate_with_policy(stage, &function, policy)
}

/// Whether one module declares `OpCapability FloatControls2` *and* the
/// extension name the translator emits beside it — the pair R6 measured on the
/// decline. Both halves are read out of the instruction stream, word by word,
/// so the assertion is about the module's own bytes rather than about the
/// translator's promise.
fn declares_float_controls2(module: &[u8]) -> bool {
    let words = module
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    let mut capability = false;
    let mut extension = false;
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let word_count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        if word_count == 0 || cursor + word_count > words.len() {
            break;
        }
        if opcode == spirv::Op::Capability as u32
            && word_count == 2
            && words[cursor + 1] == spirv::Capability::FloatControls2 as u32
        {
            capability = true;
        }
        if opcode == spirv::Op::Extension as u32 {
            let mut bytes = Vec::new();
            for word in &words[cursor + 1..cursor + word_count] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            if let Some(end) = bytes.iter().position(|byte| *byte == 0) {
                extension = bytes[..end] == *b"SPV_KHR_float_controls2";
            }
        }
        cursor += word_count;
    }
    capability && extension
}

/// The two-stream layout the pair's interface states: one `float32x2` attribute
/// per stream, at locations 0 and 1.
fn two_stream_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![
        VertexBufferLayout {
            stride: 8,
            step: VertexStep::PerVertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        },
        VertexBufferLayout {
            stride: 8,
            step: VertexStep::PerVertex,
            attributes: vec![VertexAttribute {
                location: 1,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        },
    ])
}

/// The contract the pair registers under: the AIR entry both halves declare,
/// the format list the shared fragment fixture stores, and the two-stream
/// layout.
fn two_stream_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: TWO_STREAM_VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: two_stream_layout(),
    }
}

/// `float2` records in stream order: little-endian pairs.
fn f32x2(records: &[(f32, f32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(records.len() * 8);
    for (x, y) in records {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}

fn two_stream_view(
    view_id: ViewId,
    allocation_id: AllocationId,
    binding: u32,
    bytes: Vec<u8>,
) -> BufferView {
    BufferView {
        view_id,
        metal_binding: binding,
        allocation_id,
        offset: 0,
        length: u64::try_from(bytes.len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(bytes),
    }
}

/// One trace carrying the declaring compute pass and one render pass that binds
/// both vertex streams and draws the pair's three vertices.
fn two_stream_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let positions = f32x2(&TWO_STREAM_POSITIONS);
    let offsets = f32x2(&[TWO_STREAM_OFFSET; 3]);
    let mut pass = render_pass(render.pipeline_id, None);
    pass.vertex_buffers = vec![
        two_stream_view(POSITION_VIEW, POSITION_ALLOCATION, 0, positions.clone()),
        two_stream_view(OFFSET_VIEW, OFFSET_ALLOCATION, 1, offsets.clone()),
    ];
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(22),
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
        (
            POSITION_ALLOCATION,
            u64::try_from(positions.len()).expect("stream length"),
        ),
        (
            OFFSET_ALLOCATION,
            u64::try_from(offsets.len()).expect("stream length"),
        ),
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

/// Submit one two-stream trace and return the attachment's readback bytes.
fn submit_two_stream(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    what: &str,
) -> Vec<u8> {
    let (trace, resources) = two_stream_trace(provider, compute, render);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the two-stream trace is admitted");
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
    eprintln!("{what} two-stream attachment readback: {}", hex(&bytes));
    bytes
}

/// R8: the R6 boundary pair, translated, registered and executed.
///
/// R6 pinned a boundary rather than a feature: the two-stream vertex module
/// beside this test carries the `fast` flag run a Metal module compiled with the
/// default math mode carries, while the same module with that run removed makes
/// the pinned translator decorate its float result with `FPFastMathMode` and
/// demand `FloatControls2` + `SPV_KHR_float_controls2` — which the Phase-1
/// capability subset did not admit. The canonical provider answered a *typed
/// decline* (`pipeline_compile`, `SPIR-V capability 6029 …`) and the draw ended.
///
/// The device now answers for the capability, so the withheld-permission module
/// translates, registers and executes, and it lands byte for byte what its
/// `fast` sibling lands: the left column keeps the clear sentinel (the second
/// stream's offset was read) and the right column carries the fragment's own
/// texel. The test also asserts the fixture is the module it is named for — the
/// `fast` sibling must *not* declare the capability, the precise one must — and
/// keeps the fail-closed arm for a device that reports no `shaderFloatControls2`
/// (the same sentence R6 recorded).
#[test]
fn a_withheld_float_permission_lands_the_pairs_own_bytes() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let policy = provider.spirv_feature_policy();
    let support = provider.float_controls2_support();
    eprintln!(
        "device float-controls2: extension={} feature={} enabled={}",
        support.extension_present(),
        support.feature_reported(),
        support.enabled()
    );
    let fast = translate_stage_with_policy(
        &executor,
        RenderStage::Vertex,
        TWO_STREAM_VERTEX_AIR,
        TWO_STREAM_VERTEX_ENTRY,
        policy,
    )
    .expect("the fast sibling translates");
    assert!(
        !declares_float_controls2(fast.spirv()),
        "the fast sibling grants every relaxation, so the translator emits no FPFastMathMode \
         for it"
    );
    let precise = match translate_stage_with_policy(
        &executor,
        RenderStage::Vertex,
        TWO_STREAM_PRECISE_VERTEX_AIR,
        TWO_STREAM_VERTEX_ENTRY,
        policy,
    ) {
        Ok(precise) => precise,
        Err(error) => {
            // Fail-closed arm: a device that does not answer for the capability
            // refuses the module with the sentence R6 recorded, and the `fast`
            // sibling is still the shape that executes.
            assert!(
                !policy.float_controls2(),
                "the device answered for FloatControls2 and the module still failed: {error}"
            );
            assert!(
                error.message().contains("capability 6029"),
                "the refusal keeps the capability number: {error}"
            );
            eprintln!("this device does not answer for FloatControls2: {error}");
            return;
        }
    };
    assert!(
        policy.float_controls2() && declares_float_controls2(precise.spirv()),
        "the device answered for FloatControls2, so the precise fixture has to be the module \
         that demands it"
    );
    let compute = compile_declaring_kernel(&provider, &executor);
    let register = |vertex: TranslatedRenderStage, what: &[u8]| {
        let fragment = translate_stage_with_policy(
            &executor,
            RenderStage::Fragment,
            FRAGMENT_AIR,
            FRAGMENT_ENTRY,
            policy,
        )
        .expect("the shared fragment fixture translates");
        provider
            .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
                contract: two_stream_contract(),
                vertex,
                fragment,
                logical_digest: digest(what),
            })
            .expect("the two-stream pair registers")
    };
    let fast_pipeline = register(fast, b"two-stream-fast");
    let precise_pipeline = register(precise, b"two-stream-precise");

    let fast_bytes = submit_two_stream(&provider, &compute, &fast_pipeline, "fast");
    let precise_bytes = submit_two_stream(&provider, &compute, &precise_pipeline, "precise");
    assert_eq!(
        precise_bytes, fast_bytes,
        "the withheld-permission module has to land byte for byte what its fast sibling lands"
    );
    // The offset the second stream carries is read: the triangle's left edge
    // sits at -0.25, so the 2x2 attachment's left column keeps the clear
    // sentinel while the right column carries the fragment's own texel.
    let texel = EXPECTED_RGBA8_TEXELS[..4].to_vec();
    let mut expected = Vec::new();
    for _row in 0..2 {
        expected.extend_from_slice(&CLEAR_SENTINEL);
        expected.extend_from_slice(&texel);
    }
    assert_eq!(
        precise_bytes,
        expected,
        "left column keeps the clear sentinel, right column the fragment's texel: {}",
        hex(&precise_bytes)
    );
    eprintln!(
        "withheld-permission module landed [{}] on {}",
        hex(&precise_bytes),
        executor.device_name()
    );
}
