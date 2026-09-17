//! End-to-end render rail: one trace that carries a render pass, submitted
//! through the provider's own admission and submit path.
//!
//! The case is the milestone of `research/docs/23`: a 2×2 attachment cleared to
//! a sentinel and then covered by the full-screen triangle, whose fragment stage
//! stores `(64/255, 128/255, 192/255, 1)`. What this test measures is the whole
//! chain rather than the rail alone — a `ComputeTrace` with a render entry →
//! `ProviderCapabilities::validate_trace` → `ComputeProvider::submit` → the
//! attachment's bytes in the returned writebacks — so a rail that runs but lands
//! nothing cannot pass.
//!
//! The rail builds the fragment stage from the attachment format, so the same
//! trace shape is measured once per admitted format: `Rgba8Unorm`
//! (`40 80 c0 ff` per texel), `Bgra8Unorm` (the same colour in a B,G,R,A layout,
//! `c0 80 40 ff`) and `R32Float` (one `float`, `64/255`, `81 80 80 3e`). A host
//! hands the rail compiled stages, so a registration is also where the pairing is
//! checked: a format registered with another format's fragment stage is refused
//! (`render_fragment_stage_mismatch`) instead of being run and read back as bytes
//! the format claim does not cover.
//!
//! The attachment view is declared by the trace's compute pass because that is
//! what the render contract requires: `validate_serial_buffer_reuse` resolves
//! every attachment against the views the trace declares, and a compute pass
//! that only *reads* the view is exactly the sharing core admission admits
//! (`AttachmentComputeConflict` refuses the writable half). The compute pass
//! therefore does real work of its own — `copy_word` writes the scratch view —
//! and one submission carries both rails' writebacks.

use metal_api_core::provider::{
    AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess,
    BufferLease, BufferSource, BufferView, ClearColor, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch,
    DispatchKind, DispatchType, FieldValue, IndexBufferBinding, IndexFormat,
    IndirectCommandBufferDescriptor, IndirectCommandDescriptor, IndirectCommandKind,
    IndirectCommandPayload, IndirectCommandRange, InitialState, LeaseId, LeaseImporter,
    LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId, PipelineId, PresentDescriptor,
    PresentMode, PresentTarget, ProviderCapabilities, ProviderError, ProviderErrorClass,
    ProviderPhase, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StagedLease, StoreOp, TextureAccess, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor, PRESENT_TARGET_BUDGET,
};
use std::sync::Arc;

/// Vertex stage of the milestone: `spirv-as` output of the reviewed
/// `render_spv/fullscreen_triangle.vert.spvasm` (entry `vertex_main`).
const FULL_SCREEN_TRIANGLE_VERT_SPV: &[u8] =
    include_bytes!("../src/render_spv/fullscreen_triangle.vert.spv");

/// The reviewed 8-bit UNORM fragment stage: `spirv-as` output of
/// `render_spv/solid_unorm8.frag.spvasm` (entry `fragment_main`). One module
/// serves both UNORM layouts, because the channel order is the image format's
/// decision rather than the shader's.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");

/// The reviewed dual-output fragment stage for the `[Rgba8Unorm, Rgba8Unorm]`
/// MRT shape: `Location 0` stores `(64/255, 128/255, 192/255, 1)` and
/// `Location 1` stores `(1, 128/255, 64/255, 192/255)`.
const SOLID_UNORM8_DUAL_FRAG_SPV: &[u8] =
    include_bytes!("../src/render_spv/solid_unorm8_dual.frag.spv");

/// The reviewed single-channel float fragment stage: `spirv-as` output of
/// `render_spv/solid_r32f.frag.spvasm` (entry `fragment_main`).
const SOLID_R32F_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_r32f.frag.spv");

/// The reviewed compute fixture the declaring pass runs.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The `LoadOp::Clear` sentinel (`research/docs/23` §1.3). A texel still holding
/// it proves the draw did not cover that pixel, so "the pass ran" is falsifiable
/// rather than assumed.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const ATTACHMENT_VIEW: ViewId = ViewId::new(701);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(801);
const SECOND_ATTACHMENT_VIEW: ViewId = ViewId::new(703);
const SECOND_ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(803);
const SCRATCH_VIEW: ViewId = ViewId::new(702);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(802);
const SECOND_SCRATCH_VIEW: ViewId = ViewId::new(704);
const SECOND_SCRATCH_ALLOCATION: AllocationId = AllocationId::new(804);

/// The colour formats this file measures: the contract's admitted set.
const ADMITTED_FORMATS: [AttachmentFormat; 3] = AttachmentFormat::ADMITTED;

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The fragment stage a host registers for `format`.
///
/// The test names these modules itself because a registration is exactly where a
/// host hands the rail compiled stages. The rail then checks the pairing against
/// its own per-format map, which
/// `registering_a_format_with_another_formats_fragment_stage_is_refused` observes
/// from the outside.
fn reviewed_fragment_spirv(format: AttachmentFormat) -> &'static [u8] {
    match format {
        AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm => SOLID_UNORM8_FRAG_SPV,
        AttachmentFormat::R32Float => SOLID_R32F_FRAG_SPV,
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// The bytes one texel holds when the fragment stage stores
/// `(64/255, 128/255, 192/255, 1)` into an attachment of `format`.
fn expected_texels(format: AttachmentFormat) -> [u8; 4] {
    match format {
        // The stage's components in the order it writes them.
        AttachmentFormat::Rgba8Unorm => [0x40, 0x80, 0xc0, 0xff],
        // The same colour with that layout's blue/red exchange applied: the
        // stored red `0x40` lands third, behind the stored blue `0xc0`.
        AttachmentFormat::Bgra8Unorm => [0xc0, 0x80, 0x40, 0xff],
        // A float attachment quantises nothing, so the texel is the stage's
        // `float 64/255` (`0x3e808081`) in little-endian byte order.
        AttachmentFormat::R32Float => [0x81, 0x80, 0x80, 0x3e],
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// The bytes a *surviving clear* leaves in an attachment of `format`.
///
/// The rail's clear components are `byte/255` as floats at every format, so the
/// single-channel float attachment's sentinel is the float `254/255` rather than
/// the four sentinel bytes themselves.
fn clear_bytes(format: AttachmentFormat) -> [u8; 4] {
    match format {
        AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm => CLEAR_SENTINEL,
        AttachmentFormat::R32Float => (f32::from(CLEAR_SENTINEL[0]) / 255.0).to_le_bytes(),
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// One provider context with both pipelines registered and one render-bearing
/// trace built against it.
struct Fixture {
    provider: VulkanComputeProvider,
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
    render_pipeline: PipelineId,
    format: AttachmentFormat,
}

fn render_pass(
    pipeline: PipelineId,
    format: AttachmentFormat,
    width: u64,
    height: u64,
) -> RenderPassDescriptor {
    RenderPassDescriptor {
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
            format,
            width,
            height,
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, width as u32, height as u32],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
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

/// Register one render pipeline for `format` and return its id.
///
/// `fragment_spirv` is the caller's choice, which is what makes the mismatch case
/// observable: the rail answers whether that choice is the format's own stage.
fn register_render(
    provider: &VulkanComputeProvider,
    format: AttachmentFormat,
    fragment_spirv: &[u8],
    digest: SemanticDigest,
) -> Result<CompiledComputePipeline, ProviderError> {
    provider.register_render_pipeline(RenderPipelineRequest {
        contract: RenderPipelineContract {
            vertex_entry: "vertex_main".to_owned(),
            fragment_entry: "fragment_main".to_owned(),
            color_formats: vec![format],
            vertex_layout: VertexLayout::None,
        },
        vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
        fragment_spirv: fragment_spirv.to_vec(),
        logical_digest: digest,
    })
}

fn fixture_with_stage(format: AttachmentFormat, fragment_spirv: &[u8]) -> Option<Fixture> {
    let executor = executor()?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let fixture_digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");

    // The declaring compute pass runs a reviewed kernel; the render pass names
    // a pipeline registered from the two stage modules of the milestone.
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, fixture_digest(b"render_e2e_compute"))
        .expect("the compute pipeline registers");
    let render = register_render(
        &provider,
        format,
        fragment_spirv,
        fixture_digest(b"render_e2e_stages"),
    )
    .expect("the render pipeline registers");

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(11),
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
                        // 2×2 texels of four bytes: exactly the extent the
                        // render attachment restates, so the declaration covers
                        // it whatever the compute kernel reads out of it.
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
            TracePass::Render(render_pass(render.pipeline_id, format, 2, 2)),
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
            size: 16,
        })
        .expect("attachment allocation");
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: SCRATCH_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 8,
        })
        .expect("scratch allocation");

    Some(Fixture {
        provider,
        trace,
        resources,
        render_pipeline: render.pipeline_id,
        format,
    })
}

fn fixture(format: AttachmentFormat) -> Option<Fixture> {
    fixture_with_stage(format, reviewed_fragment_spirv(format))
}

fn submit_fixture(fixture: &Fixture) -> Vec<(ViewId, Vec<u8>)> {
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect("the render-bearing trace is admitted");
    let submitted = fixture
        .provider
        .submit(admitted)
        .expect("the submission completes");
    submitted
        .validate_for_trace(&fixture.trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect()
}

fn readback(writebacks: &[(ViewId, Vec<u8>)], view: ViewId) -> Vec<u8> {
    writebacks
        .iter()
        .find(|(id, _)| *id == view)
        .map(|(_, bytes)| bytes.clone())
        .unwrap_or_else(|| panic!("view {view:?} has no writeback"))
}

/// The attachment's readback for one submitted fixture, with the raw bytes
/// printed so the run's log carries the evidence the assertions are about.
fn attachment_readback(fixture: &Fixture, writebacks: &[(ViewId, Vec<u8>)]) -> Vec<u8> {
    let attachment = readback(writebacks, ATTACHMENT_VIEW);
    eprintln!(
        "{:?} attachment readback: {} ({} bytes, first texel: {})",
        fixture.format,
        hex(&attachment),
        attachment.len(),
        hex(&attachment[..4])
    );
    eprintln!(
        "{:?} expected: [{}] x4, clear sentinel: [{}]",
        fixture.format,
        hex(&expected_texels(fixture.format)),
        hex(&clear_bytes(fixture.format))
    );
    attachment
}

/// The whole chain for the milestone's original case, plus the registration's
/// lifecycle.
#[test]
fn render_pass_trace_executes_and_lands_attachment_bytes_through_writeback() {
    let Some(fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let writebacks = submit_fixture(&fixture);
    let attachment = attachment_readback(&fixture, &writebacks);

    assert_eq!(attachment.len(), 16);
    assert_eq!(attachment, expected_texels(fixture.format).repeat(4));
    // The attachment is cleared to `fe fe fe fe` and the draw then covers every
    // texel of the 2×2 viewport, so the clear's own bytes are observable only
    // here: "the readback is the draw's bytes" rather than the clear value.
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == clear_bytes(fixture.format).as_slice()),
        "a surviving clear sentinel means the triangle did not cover every texel, so the \
         readback would be the ClearColor rather than the fragment stage's bytes: {}",
        hex(&attachment)
    );
    assert_ne!(clear_bytes(fixture.format), expected_texels(fixture.format));

    // The compute rail's own writeback rides the same submission, so the render
    // pass was not traded for the compute result.
    let scratch = readback(&writebacks, SCRATCH_VIEW);
    eprintln!("scratch readback: {}", hex(&scratch));
    assert_eq!(scratch, ATTACHMENT_WORD.to_vec());

    // A render pipeline is a provider-side registration, released the way a
    // registered compute pipeline is.
    let metadata = fixture
        .trace
        .pipeline(fixture.render_pipeline)
        .expect("the trace carries the render pipeline entry")
        .clone();
    fixture
        .provider
        .release_render_pipeline(&metadata)
        .expect("the registration is released");
    let refused = fixture
        .provider
        .release_render_pipeline(&metadata)
        .expect_err("a released registration cannot be released twice");
    assert_eq!(refused.slug, "unknown_render_pipeline");
}

/// The same trace shape through every admitted colour format: the format selects
/// the fragment stage, and the attachment lands the bytes that stage's colour
/// has in that format's layout.
#[test]
fn every_admitted_colour_format_lands_its_own_attachment_bytes() {
    for format in ADMITTED_FORMATS {
        let Some(fixture) = fixture(format) else {
            return;
        };
        let writebacks = submit_fixture(&fixture);
        let attachment = attachment_readback(&fixture, &writebacks);

        assert_eq!(attachment.len(), 16);
        assert_eq!(attachment, expected_texels(format).repeat(4));
        assert!(
            !attachment
                .chunks_exact(4)
                .any(|texel| texel == clear_bytes(format).as_slice()),
            "a surviving clear sentinel means the triangle did not cover every texel, so the \
             {format:?} readback would be the ClearColor: {}",
            hex(&attachment)
        );
        assert_ne!(clear_bytes(format), expected_texels(format));
        // The two UNORM layouts must not land the same bytes: a rail that wrote
        // the same byte order at both formats would pass an R,G,B,A-only check
        // and still be wrong about one of them.
        if format == AttachmentFormat::Bgra8Unorm {
            assert_ne!(
                attachment,
                expected_texels(AttachmentFormat::Rgba8Unorm).repeat(4),
                "a B,G,R,A attachment cannot read back the R,G,B,A bytes"
            );
        }

        let scratch = readback(&writebacks, SCRATCH_VIEW);
        assert_eq!(scratch, ATTACHMENT_WORD.to_vec());
    }
}

/// The reviewed MRT shape through the whole chain: two 2×2 `Rgba8Unorm`
/// attachments drawn in one pass, each landing its own location's bytes in its
/// own view writeback, with `copy_out == 2` (one image→buffer copy per
/// attachment). Each attachment view needs its own declaring compute pass
/// (`copy_word` reads binding 0, writes binding 1), so the executor's readback
/// delta is the render rail's two copies plus two compute scratch copies.
#[test]
fn dual_attachments_land_both_locations_through_writeback() {
    let Some(executor) = executor() else {
        return;
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");

    // The declaring compute pass runs the reviewed kernel; the render pass
    // names a pipeline registered from the reviewed dual-output module.
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_e2e_mrt_compute"))
        .expect("the compute pipeline registers");
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_DUAL_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_mrt_stages"),
        })
        .expect("the dual render pipeline registers");

    let attachment = |view: ViewId, allocation: AllocationId| RenderAttachment {
        view_id: view,
        allocation_id: allocation,
        format: AttachmentFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
        store: StoreOp::Store,
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(12),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            // The reviewed kernel reads binding 0 and writes binding 1, so each
            // attachment view needs its own declaring compute pass: binding 0
            // is the only read slot, and core refuses a compute write of
            // attachment bytes (`AttachmentComputeConflict`). Each pass writes
            // its own scratch view so both declarations do real work.
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
            TracePass::Compute(ComputePass {
                pipeline: compute.pipeline_id,
                buffers: vec![
                    BufferView {
                        view_id: SECOND_ATTACHMENT_VIEW,
                        metal_binding: 0,
                        allocation_id: SECOND_ATTACHMENT_ALLOCATION,
                        offset: 0,
                        length: 16,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(4)),
                    },
                    BufferView {
                        view_id: SECOND_SCRATCH_VIEW,
                        metal_binding: 1,
                        allocation_id: SECOND_SCRATCH_ALLOCATION,
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
            TracePass::Render(RenderPassDescriptor {
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
                pipeline: render.pipeline_id,
                color_attachments: vec![
                    attachment(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION),
                    attachment(SECOND_ATTACHMENT_VIEW, SECOND_ATTACHMENT_ALLOCATION),
                ],
                viewport: [0, 0, 2, 2],
                scissor: None,
                vertices: 3,
                vertex_buffers: Vec::new(),
                indices: None,
                instance_count: 1,
                textures: Vec::new(),
                present: None,
            }),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };

    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, 16),
        (SECOND_ATTACHMENT_ALLOCATION, 16),
        (SCRATCH_ALLOCATION, 8),
        (SECOND_SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("attachment allocation");
    }

    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the dual-attachment trace is admitted");
    let (_, readbacks_before) = executor.buffer_copy_counts();
    let submitted = provider.submit(admitted).expect("the submission completes");
    let (_, readbacks_after) = executor.buffer_copy_counts();
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();

    let first = readback(&writebacks, ATTACHMENT_VIEW);
    let second = readback(&writebacks, SECOND_ATTACHMENT_VIEW);
    eprintln!("location 0 readback: {}", hex(&first));
    eprintln!("location 1 readback: {}", hex(&second));
    assert_eq!(first, [0x40, 0x80, 0xc0, 0xff].repeat(4));
    assert_eq!(second, [0xff, 0x80, 0x40, 0xc0].repeat(4));

    // One copy-out per attachment (`copy_out == 2` on the render rail) plus
    // each declaring compute pass's scratch readback, so the executor's
    // readback counter advances by exactly four.
    let scratch = readback(&writebacks, SCRATCH_VIEW);
    assert_eq!(scratch, ATTACHMENT_WORD.to_vec());
    let second_scratch = readback(&writebacks, SECOND_SCRATCH_VIEW);
    assert_eq!(second_scratch, ATTACHMENT_WORD.to_vec());
    assert_eq!(
        readbacks_after - readbacks_before,
        4,
        "two render attachment copies plus two compute scratch copies"
    );
}

/// The v19 discard shape through the whole chain (`docs/23` §3.6): location 0
/// stores, location 1 is `DontCare`, so only location 0 lands a writeback and
/// the readback counter advances by the one stored attachment instead of two.
/// The declaring compute passes still touch both attachment views, so the
/// upload counter keeps covering both (`copy_in` unchanged), while `copy_out`
/// is the two declaring scratch writes plus the one stored attachment.
#[test]
fn a_discarded_attachment_lands_no_writeback_but_the_stored_one_does() {
    let Some(executor) = executor() else {
        return;
    };
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
        .compile_pipeline(&function, digest(b"render_e2e_discard_compute"))
        .expect("the compute pipeline registers");
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_DUAL_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_discard_stages"),
        })
        .expect("the dual render pipeline registers");

    let attachment = |view: ViewId, allocation: AllocationId, store: StoreOp| RenderAttachment {
        view_id: view,
        allocation_id: allocation,
        format: AttachmentFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
        store,
    };
    let declaring = |attachment_view: ViewId,
                     attachment_allocation: AllocationId,
                     scratch_view: ViewId,
                     scratch_allocation: AllocationId| {
        ComputePass {
            pipeline: compute.pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: attachment_view,
                    metal_binding: 0,
                    allocation_id: attachment_allocation,
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(4)),
                },
                BufferView {
                    view_id: scratch_view,
                    metal_binding: 1,
                    allocation_id: scratch_allocation,
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
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(13),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(declaring(
                ATTACHMENT_VIEW,
                ATTACHMENT_ALLOCATION,
                SCRATCH_VIEW,
                SCRATCH_ALLOCATION,
            )),
            TracePass::Compute(declaring(
                SECOND_ATTACHMENT_VIEW,
                SECOND_ATTACHMENT_ALLOCATION,
                SECOND_SCRATCH_VIEW,
                SECOND_SCRATCH_ALLOCATION,
            )),
            TracePass::Render(RenderPassDescriptor {
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
                pipeline: render.pipeline_id,
                color_attachments: vec![
                    attachment(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION, StoreOp::Store),
                    attachment(
                        SECOND_ATTACHMENT_VIEW,
                        SECOND_ATTACHMENT_ALLOCATION,
                        StoreOp::DontCare,
                    ),
                ],
                viewport: [0, 0, 2, 2],
                scissor: None,
                vertices: 3,
                vertex_buffers: Vec::new(),
                indices: None,
                instance_count: 1,
                textures: Vec::new(),
                present: None,
            }),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };

    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, 16),
        (SECOND_ATTACHMENT_ALLOCATION, 16),
        (SCRATCH_ALLOCATION, 8),
        (SECOND_SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("attachment allocation");
    }

    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the store-plus-discard trace is admitted");
    let (uploads_before, readbacks_before) = executor.buffer_copy_counts();
    let submitted = provider.submit(admitted).expect("the submission completes");
    let (uploads_after, readbacks_after) = executor.buffer_copy_counts();
    submitted
        .validate()
        .expect("the writeback list is well formed");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();

    let stored = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("stored location readback: {}", hex(&stored));
    assert_eq!(stored, [0x40, 0x80, 0xc0, 0xff].repeat(4));
    assert!(
        !writebacks
            .iter()
            .any(|(view, _)| *view == SECOND_ATTACHMENT_VIEW),
        "the discarded location disappears from the observable surface"
    );

    // Both declaring passes still do their own work and upload their two
    // owned views each — the attachment view and the scratch view — exactly as
    // the dual-`Store` baseline does, so `copy_in` covers both attachments and
    // the discard changes nothing on the upload side (`touched` counts
    // everything). `copy_out` is the two scratch writebacks plus the one
    // stored attachment copy.
    let scratch = readback(&writebacks, SCRATCH_VIEW);
    assert_eq!(scratch, ATTACHMENT_WORD.to_vec());
    let second_scratch = readback(&writebacks, SECOND_SCRATCH_VIEW);
    assert_eq!(second_scratch, ATTACHMENT_WORD.to_vec());
    assert_eq!(
        uploads_after - uploads_before,
        4,
        "two declaring passes upload two owned views each, unchanged by the discard"
    );
    assert_eq!(
        readbacks_after - readbacks_before,
        3,
        "one stored attachment copy plus two compute scratch copies"
    );
}

/// A registration pairs a compiled fragment stage with a format, and the rail
/// refuses a pairing that is not the format's own stage. This is the end-to-end
/// witness that the fragment stage really is a function of the format: the
/// caller cannot choose one the format was not built for.
#[test]
fn registering_a_format_with_another_formats_fragment_stage_is_refused() {
    let Some(executor) = executor() else {
        return;
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");

    // The pairings the review filed as I2: the 8-bit module's `vec4` store on the
    // one-component float attachment, and the float module on each 8-bit layout.
    for (format, wrong_stage) in [
        (AttachmentFormat::R32Float, SOLID_UNORM8_FRAG_SPV),
        (AttachmentFormat::Rgba8Unorm, SOLID_R32F_FRAG_SPV),
        (AttachmentFormat::Bgra8Unorm, SOLID_R32F_FRAG_SPV),
    ] {
        let refused = register_render(&provider, format, wrong_stage, digest(b"mismatch"))
            .expect_err("another format's fragment stage is refused");
        eprintln!("{format:?} with another format's stage: refused: {refused:?}");
        assert_eq!(refused.slug, "render_fragment_stage_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("format_code"),
            Some(&FieldValue::Unsigned(u64::from(format.code())))
        );
    }

    // Every format's own stage registers, so the refusal above is about the
    // pairing rather than about render registration itself.
    for (index, format) in ADMITTED_FORMATS.into_iter().enumerate() {
        let case = format!("render_e2e_reviewed_{index}");
        register_render(
            &provider,
            format,
            reviewed_fragment_spirv(format),
            digest(case.as_bytes()),
        )
        .unwrap_or_else(|error| panic!("{format:?} registers with its own stage: {error:?}"));
    }
}

#[test]
fn the_same_trace_is_refused_when_the_provider_declares_no_render_support() {
    let Some(fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let declared = fixture.provider.capabilities();
    assert!(declared.supports_render_passes);
    declared
        .admit(&fixture.trace, &fixture.resources)
        .expect("the declared capability bits admit the fixture");

    // The pre-render snapshot: same trace, same provider, same resources, only
    // the render bits differ.
    let without_render = without_render_bits(&declared);
    let refused = admit_error(&without_render, &fixture.trace, &fixture.resources);
    eprintln!("render bits off: refused: {refused:?}");
    assert_eq!(refused.slug, "render_passes_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);

    // The dimension bit is load-bearing as well: the probe trace's 3×3
    // attachment outgrows the declaring view's 16 bytes, so admission refuses
    // the extent agreement. (The rail executes up to four texels per axis from
    // v27 on, so the dimension ceiling itself is no longer what this probe
    // trips; the extent/view agreement is.)
    let oversized = oversized_trace(&fixture);
    let refused = admit_error(&declared, &oversized, &fixture.resources);
    eprintln!("3x3 attachment: refused: {refused:?}");
    assert_eq!(refused.slug, "attachment_extent_mismatch");
}

/// The registered entry is what core admission reads (review item I3,
/// 2026-09-14).
///
/// The registration hands the owner the render half beside the compute half, so
/// a trace's own table carries the colour format the stages were compiled for.
/// The two halves of this test are the two ways a trace can get that wrong: an
/// entry with no render half at all, and an entry whose render half names
/// another admitted format. Both are refused by admission — before the provider
/// reserves anything — and the third shape, the registered entry itself, keeps
/// executing.
#[test]
fn admission_reads_the_render_contract_from_the_registered_entry() {
    for format in ADMITTED_FORMATS {
        let Some(fixture) = fixture(format) else {
            return;
        };
        // The clone below keeps the table order, so one index serves both.
        let render_entry = fixture
            .trace
            .pipelines
            .iter()
            .position(|pipeline| pipeline.pipeline_id == fixture.render_pipeline)
            .expect("the fixture carries the registered render entry");
        let registered = &fixture.trace.pipelines[render_entry];
        assert_eq!(
            registered
                .render
                .as_ref()
                .map(|contract| contract.color_formats.as_slice()),
            Some([format].as_slice()),
            "the registration hands the owner the half admission reads"
        );

        let mut dropped = fixture.trace.clone();
        dropped.pipelines[render_entry].render = None;
        let refused = admit_error(
            &fixture.provider.capabilities(),
            &dropped,
            &fixture.resources,
        );
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);
        assert!(
            refused
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("no render contract")),
            "the refusal has to name the missing half, got {:?}",
            refused.detail
        );

        let mut retargeted = fixture.trace.clone();
        let other = ADMITTED_FORMATS
            .into_iter()
            .find(|candidate| *candidate != format)
            .expect("the admitted set has more than one format");
        retargeted.pipelines[render_entry]
            .render
            .as_mut()
            .expect("the fixture entry carries the half")
            .color_formats = vec![other];
        let refused = admit_error(
            &fixture.provider.capabilities(),
            &retargeted,
            &fixture.resources,
        );
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert!(
            refused
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("does not match attachment format")),
            "the refusal has to name both formats, got {:?}",
            refused.detail
        );

        // The registered entry as it was handed out still admits and runs.
        assert_eq!(
            attachment_readback(&fixture, &submit_fixture(&fixture)),
            expected_texels(format).repeat(4)
        );
    }
}

fn without_render_bits(declared: &ProviderCapabilities) -> ProviderCapabilities {
    let mut capabilities = declared.clone();
    capabilities.supports_render_passes = false;
    capabilities.max_color_attachments = 0;
    capabilities.max_attachment_dimension = [0, 0];
    capabilities.supported_color_formats.clear();
    capabilities
}

fn oversized_trace(fixture: &Fixture) -> ComputeTrace {
    let mut trace = fixture.trace.clone();
    trace.passes[1] = TracePass::Render(render_pass(fixture.render_pipeline, fixture.format, 3, 3));
    trace
}

fn admit_error(
    capabilities: &ProviderCapabilities,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> ProviderError {
    capabilities
        .admit(trace, resources)
        .expect_err("the trace is refused")
}

/// The `InitialState::Sentinel` bytes a present fixture pre-fills its target
/// with. Distinct from both the rendered colour and the clear sentinel, so
/// "the present never happened" (a readback still showing these bytes) is
/// falsifiable (`docs/24` §3.1).
const PRESENT_SENTINEL: [u8; 4] = [0xfe; 4];

/// Attach the first increment's present action to a fixture's render pass: the
/// target is the attachment's own allocation/view, handed on once in `Fifo`
/// mode with a blocking acquire.
fn attach_present(
    trace: &mut ComputeTrace,
    view: ViewId,
    allocation: AllocationId,
    format: AttachmentFormat,
    sentinel: [u8; 4],
) {
    let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    // The target restates the attachment it presents, extent included
    // (`PresentDescriptor::validate_against`).
    let width = pass.color_attachments[0].width;
    let height = pass.color_attachments[0].height;
    pass.present = Some(PresentDescriptor {
        target: PresentTarget {
            allocation_id: allocation,
            view_id: view,
            format,
            width,
            height,
            image_count: 1,
            initial: InitialState::Sentinel(sentinel.to_vec()),
        },
        source: view,
        mode: PresentMode::Fifo,
        acquire: AcquirePolicy::Blocking,
    });
}

fn presenting_fixture() -> Option<Fixture> {
    let mut fixture = fixture(AttachmentFormat::Rgba8Unorm)?;
    attach_present(
        &mut fixture.trace,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        fixture.format,
        PRESENT_SENTINEL,
    );
    Some(fixture)
}

/// The milestone's present case end to end: one render pass hands its own 2×2
/// attachment on as a present target, the target lands the fragment output
/// (not the sentinel), and the submission counts exactly one acquire and one
/// present (`research/docs/24` §6 Step 3).
#[test]
fn present_target_lands_rendered_bytes_and_counts_one_acquire_one_present() {
    let Some(fixture) = presenting_fixture() else {
        return;
    };
    let (acquires_before, presents_before) = fixture.provider.present_counts();

    let writebacks = submit_fixture(&fixture);
    let attachment = attachment_readback(&fixture, &writebacks);

    assert_eq!(attachment.len(), 16);
    assert_eq!(attachment, expected_texels(fixture.format).repeat(4));
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == PRESENT_SENTINEL),
        "a surviving present sentinel means the present never overwrote the target: {}",
        hex(&attachment)
    );
    assert_ne!(PRESENT_SENTINEL, expected_texels(fixture.format));

    // Exactly one acquire and one present for the single present action
    // (`docs/24` §5.3).
    let (acquires_after, presents_after) = fixture.provider.present_counts();
    assert_eq!(
        acquires_after - acquires_before,
        1,
        "one acquire per present action"
    );
    assert_eq!(
        presents_after - presents_before,
        1,
        "one present per present action"
    );

    // The target is provider-owned, so it survives the submission (`docs/24`
    // §3.3 rule 2) and stays reusable after `wait`.
    assert_eq!(
        fixture.provider.present_target_count(),
        1,
        "one present target stays alive after the submission"
    );
}

/// A second present of the same allocation/view reuses the provider-owned
/// target rather than recreating it: the target image survives across
/// submissions until the lease is released (`docs/24` §5.2).
/// The indirect payload one replay carries: one non-indexed draw of the
/// milestone's full-screen triangle (`research/docs/25` §6 Step 4).
fn indirect_draw() -> IndirectCommandPayload {
    IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::Draw],
        },
        command: IndirectCommandDescriptor::Draw {
            vertex_count: 3,
            instance_count: 1,
        },
        range: IndirectCommandRange { start: 0, count: 1 },
    }
}

/// The indirect payload one indexed replay carries: one `draw_indexed` of the
/// milestone's full-screen triangle through the rail's own `[0, 1, 2]` index
/// buffer (`research/docs/25` §6 Step 4).
fn indirect_draw_indexed() -> IndirectCommandPayload {
    IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::DrawIndexed],
        },
        command: IndirectCommandDescriptor::DrawIndexed {
            index_count: 3,
            instance_count: 1,
        },
        range: IndirectCommandRange { start: 0, count: 1 },
    }
}

/// The first indirect increment replays the pass's full-screen triangle from a
/// CPU-encoded `VkDrawIndirectCommand`. The falsifiable claim is byte equality
/// with the direct draw: the same attachment, the same fragment output, and
/// neither the clear sentinel nor an uninitialised image.
#[test]
fn an_indirect_draw_replays_the_same_attachment_bytes_as_a_direct_draw() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let direct_bytes = attachment_readback(&direct, &submit_fixture(&direct));
    // One texel's bytes replicated over the 2x2 attachment: the fixture covers
    // every texel with the same fragment output.
    let expected = expected_texels(AttachmentFormat::Rgba8Unorm).repeat(4);
    assert_eq!(direct_bytes, expected);

    let mut trace = direct.trace.clone();
    trace.indirect = Some(Box::new(indirect_draw()));
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("the indirect-bearing trace is admitted");
    let submitted = direct
        .provider
        .submit(admitted)
        .expect("the indirect replay completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the replay lands the attachment writeback");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    let indirect_bytes = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("indirect draw readback: {}", hex(&indirect_bytes));
    assert_eq!(indirect_bytes, expected);
    assert_eq!(indirect_bytes, direct_bytes);
    assert_ne!(indirect_bytes, CLEAR_SENTINEL.repeat(4));
}

/// The indexed sibling of the draw replay: the same attachment bytes come back
/// from a `vkCmdDrawIndexedIndirect` replay that binds the rail's `[0, 1, 2]`
/// index buffer. The falsifiable claim stays byte equality with the direct
/// draw, so an index buffer that selected a different vertex set could not
/// pass.
#[test]
fn an_indirect_indexed_draw_replays_the_same_attachment_bytes_as_a_direct_draw() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let direct_bytes = attachment_readback(&direct, &submit_fixture(&direct));
    let expected = expected_texels(AttachmentFormat::Rgba8Unorm).repeat(4);
    assert_eq!(direct_bytes, expected);

    let mut trace = direct.trace.clone();
    trace.indirect = Some(Box::new(indirect_draw_indexed()));
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("the indexed-indirect-bearing trace is admitted");
    let submitted = direct
        .provider
        .submit(admitted)
        .expect("the indexed indirect replay completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the replay lands the attachment writeback");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    let indirect_bytes = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("indexed indirect draw readback: {}", hex(&indirect_bytes));
    assert_eq!(indirect_bytes, expected);
    assert_eq!(indirect_bytes, direct_bytes);
    assert_ne!(indirect_bytes, CLEAR_SENTINEL.repeat(4));
}

/// The first indexed increment replays exactly the milestone's three indices.
/// A command that names another index count clears admission (the command's own
/// validation only rejects zero), so the provider's own guard is what refuses
/// the un-reviewed shape before any render object is created.
#[test]
fn an_indirect_indexed_draw_with_an_unreviewed_index_count_is_refused() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let mut trace = direct.trace.clone();
    trace.indirect = Some(Box::new(IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::DrawIndexed],
        },
        command: IndirectCommandDescriptor::DrawIndexed {
            index_count: 2,
            instance_count: 1,
        },
        range: IndirectCommandRange { start: 0, count: 1 },
    }));
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("admission validates the indexed command's nonzero counts");
    let refused = direct
        .provider
        .submit(admitted)
        .expect_err("the reviewed indexed shape is three indices only");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// A dispatch command is now in the admitted set, so it clears admission; the
/// provider then refuses the shape mismatch — a dispatch cannot replay into a
/// render pass — with the capability slug the contract publishes, before any
/// render object is created.
#[test]
fn an_indirect_dispatch_into_a_render_trace_is_refused() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let mut trace = direct.trace.clone();
    trace.indirect = Some(Box::new(IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: 1,
            kinds: vec![IndirectCommandKind::Dispatch],
        },
        command: IndirectCommandDescriptor::Dispatch {
            threadgroups: [1, 1, 1],
        },
        range: IndirectCommandRange { start: 0, count: 1 },
    }));
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("the dispatch command is in the first increment's admitted set");
    let refused = direct
        .provider
        .submit(admitted)
        .expect_err("a dispatch command cannot replay into a render pass");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// An indirect command with no render pass to replay into is refused, not
/// dropped: admission admits the payload (it validates the command, not the
/// pass list), so the provider's own guard is what keeps the submission from
/// reporting success while never replaying the draw.
#[test]
fn an_indirect_command_without_a_render_pass_is_refused() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let mut trace = direct.trace.clone();
    trace.passes.truncate(1);
    // The truncated render pass is gone, so its pipeline cannot stay in the
    // table: admission refuses unused pipeline metadata before it reaches the
    // provider.
    trace
        .pipelines
        .retain(|pipeline| pipeline.pipeline_id != direct.render_pipeline);
    trace.indirect = Some(Box::new(indirect_draw()));
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("admission validates the indirect payload, not the pass list");
    let refused = direct
        .provider
        .submit(admitted)
        .expect_err("a replayed command needs a render pass");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// An indirect command cannot be replayed into a presenting pass in the first
/// increment: the present tail owns the pass's terminal layout, so the
/// combination is refused rather than silently running one of the two.
#[test]
fn an_indirect_presenting_pass_is_refused() {
    let Some(presenting) = presenting_fixture() else {
        return;
    };
    let mut trace = presenting.trace.clone();
    trace.indirect = Some(Box::new(indirect_draw()));
    let admitted = presenting
        .provider
        .capabilities()
        .validate_trace(trace.clone(), presenting.resources.clone())
        .expect("the payload itself is admissible");
    let refused = presenting
        .provider
        .submit(admitted)
        .expect_err("an indirect replay into a presenting pass is outside the increment");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// One indirect command replays into exactly one render pass: a trace with two
/// render passes is refused rather than replaying the command into an ambiguous
/// pass (or into one of them and silently dropping the other).
#[test]
fn an_indirect_command_with_two_render_passes_is_refused() {
    let Some(direct) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let mut trace = direct.trace.clone();
    let Some(render) = trace.passes.last().cloned() else {
        return;
    };
    trace.passes.push(render);
    trace.indirect = Some(Box::new(indirect_draw()));
    // Admission admits the two-pass shape; the provider owns the one-pass
    // indirect guard, so the refusal has to come from `submit`.
    let admitted = direct
        .provider
        .capabilities()
        .validate_trace(trace.clone(), direct.resources.clone())
        .expect("admission admits the two-pass trace");
    let refused = direct
        .provider
        .submit(admitted)
        .expect_err("one indirect command needs exactly one render pass");
    assert_eq!(refused.slug, "icb_command_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

#[test]
fn a_second_present_reuses_the_same_target_image() {
    let Some(fixture) = presenting_fixture() else {
        return;
    };
    let first = submit_fixture(&fixture);
    let second = submit_fixture(&fixture);

    for writebacks in [first, second] {
        let attachment = readback(&writebacks, ATTACHMENT_VIEW);
        assert_eq!(attachment, expected_texels(fixture.format).repeat(4));
        assert!(
            !attachment
                .chunks_exact(4)
                .any(|texel| texel == PRESENT_SENTINEL),
            "the reused target still lands the fragment output, not the sentinel: {}",
            hex(&attachment)
        );
    }

    // Two presents, one target: reuse is observable as a stable target count.
    assert_eq!(fixture.provider.present_target_count(), 1);
    assert_eq!(fixture.provider.present_counts(), (2, 2));
}

/// One presenting trace whose attachment (and therefore present target)
/// identity is the caller's, so a registry-wide test can present many distinct
/// targets through one provider.
fn presenting_trace_for(
    fixture: &Fixture,
    view: ViewId,
    allocation: AllocationId,
    sentinel: [u8; 4],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut trace = fixture.trace.clone();
    if let Some(TracePass::Compute(pass)) = trace.passes.first_mut() {
        pass.buffers[0].view_id = view;
        pass.buffers[0].allocation_id = allocation;
    } else {
        panic!("the fixture opens with the declaring compute pass");
    }
    if let Some(TracePass::Render(pass)) = trace.passes.last_mut() {
        pass.color_attachments[0].view_id = view;
        pass.color_attachments[0].allocation_id = allocation;
    } else {
        panic!("the fixture ends in a render pass");
    }
    attach_present(&mut trace, view, allocation, fixture.format, sentinel);
    let mut resources = fixture.resources.clone();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: allocation,
            owner_epoch: fixture.provider.device_epoch(),
            size: 16,
        })
        .expect("the presenting attachment's allocation");
    (trace, resources)
}

/// Submit one hand-built trace and return its writebacks, asserting admission
/// and completion on the way so a registry case only measures the registry.
fn submit_presenting_trace(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<(ViewId, Vec<u8>)> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the presenting trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(trace)
        .expect("the writebacks cover the trace");
    submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect()
}

/// The present registry's budget and eviction order (`research/docs/24`
/// §5.2): the contract bounds the targets one trace names, the provider bounds
/// the identities it keeps resident, and the victim is the least recently used
/// one — which the eviction counter makes observable, because re-presenting a
/// live target evicts nothing while re-presenting an evicted one does.
#[test]
fn present_targets_are_bounded_and_evicted_least_recently_used() {
    let Some(fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let provider = &fixture.provider;
    let budget = PRESENT_TARGET_BUDGET;
    assert!(
        budget >= 2,
        "the LRU case needs more than one resident target"
    );

    // `budget + 2` distinct identities: the registry has to evict exactly the
    // two oldest ones, keeping its count at the budget.
    let identities = (0..budget + 2)
        .map(|index| {
            (
                ViewId::new(9000 + index as u64),
                AllocationId::new(9500 + index as u64),
            )
        })
        .collect::<Vec<_>>();
    for (index, (view, allocation)) in identities.iter().enumerate() {
        let (trace, resources) =
            presenting_trace_for(&fixture, *view, *allocation, PRESENT_SENTINEL);
        let writebacks = submit_presenting_trace(provider, &trace, &resources);
        assert_eq!(
            readback(&writebacks, *view),
            expected_texels(fixture.format).repeat(4),
            "the presenting target at index {index} lands the drawn texels"
        );
    }

    let over_budget = (identities.len() - budget) as u64;
    assert_eq!(
        provider.present_target_count(),
        budget,
        "the registry keeps at most the budget's worth of targets resident"
    );
    let evictions = provider.present_target_evictions();
    assert_eq!(
        evictions, over_budget,
        "one eviction per identity beyond the budget"
    );
    eprintln!(
        "present registry after {} distinct identities: count={} budget={} evictions={}",
        identities.len(),
        provider.present_target_count(),
        budget,
        evictions
    );

    // The two oldest identities were the victims: presenting them again has to
    // recreate the target and evict the next least recently used entry, which
    // is what makes the order — not just the count — observable.
    for (view, allocation) in identities.iter().take(over_budget as usize) {
        let (trace, resources) =
            presenting_trace_for(&fixture, *view, *allocation, PRESENT_SENTINEL);
        let writebacks = submit_presenting_trace(provider, &trace, &resources);
        assert_eq!(
            readback(&writebacks, *view),
            expected_texels(fixture.format).repeat(4),
            "the re-created target lands the drawn texels"
        );
    }
    assert_eq!(
        provider.present_target_count(),
        budget,
        "re-creating an evicted target evicts a live one instead of growing"
    );
    assert_eq!(
        provider.present_target_evictions(),
        evictions + over_budget,
        "each re-created target evicts the least recently used live target"
    );
    eprintln!(
        "present registry after re-presenting the {} evicted identities: count={} evictions={}",
        over_budget,
        provider.present_target_count(),
        provider.present_target_evictions()
    );
}

/// The normal-path retirement surface that pairs with the budget: a present
/// target reserved for an allocation is retired when the staged lease that
/// allocation was imported under is released — the same rule the native rail's
/// `drop_present_targets_for` states (`research/docs/24` §6 Step 7).
#[test]
fn releasing_a_staged_lease_retires_the_present_target_of_its_allocation() {
    let Some(mut fixture) = presenting_fixture() else {
        return;
    };
    let provider = &fixture.provider;
    let lease = LeaseId::new(11);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: 16,
    };
    provider
        .import_staged_lease(
            StagedLease::new(reservation, ATTACHMENT_WORD.repeat(4))
                .expect("the staged attachment carries one byte per declared byte"),
        )
        .expect("the provider stages the attachment's bytes");
    if let Some(TracePass::Compute(pass)) = fixture.trace.passes.first_mut() {
        pass.buffers[0].source = BufferSource::StagedLease(lease);
    } else {
        panic!("the fixture opens with the declaring compute pass");
    }
    let mut resources = fixture.resources.clone();
    resources
        .insert_lease(reservation)
        .expect("the reservation covers the attachment view");

    let writebacks = submit_presenting_trace(provider, &fixture.trace, &resources);
    assert_eq!(
        readback(&writebacks, ATTACHMENT_VIEW),
        expected_texels(fixture.format).repeat(4)
    );
    assert_eq!(
        provider.present_target_count(),
        1,
        "the present target of a leased allocation stays alive while the lease does"
    );
    let evictions = provider.present_target_evictions();
    eprintln!(
        "leased present: target count={} evictions={}",
        provider.present_target_count(),
        evictions
    );

    provider
        .release_staged_lease(lease)
        .expect("the staged lease is released");
    assert_eq!(
        provider.present_target_count(),
        0,
        "releasing the lease retires the allocation's present target in the same call"
    );
    assert_eq!(
        provider.present_target_evictions(),
        evictions + 1,
        "a lease-driven retirement is the same observable as a budget eviction"
    );
    eprintln!(
        "present target count after the lease release: {} evictions={}",
        provider.present_target_count(),
        provider.present_target_evictions()
    );
}

/// A texture beside the reviewed solid pair is refused by the slug the
/// offscreen rail states (`render_texture_stage_unsupported`) rather than
/// executed by a present pass that drops the binding: the present rail's
/// reviewed arm binds the format's solid module, which names no image binding
/// at all (`research/docs/23` §3.3, v70; R4a increment).
#[test]
fn a_presenting_pass_with_a_render_texture_is_refused_by_name() {
    let Some(fixture) = presenting_fixture() else {
        return;
    };
    let mut trace = fixture.trace.clone();
    let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    pass.textures = vec![TextureView {
        view_id: SAMPLED_TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: SAMPLED_TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(sampled_texels()[..16].to_vec()),
    }];
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), fixture.resources.clone())
        .expect("the declaration stays well formed");
    let refused = fixture
        .provider
        .submit(admitted)
        .expect_err("the present rail binds the format's solid module, which samples nothing");
    eprintln!("texture beside a solid present pass refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_stage_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// The typed refusal for an unsupported snapshot is unchanged by the execution
/// rail: a snapshot without the present bits still refuses a present-bearing
/// trace with `present_targets_unsupported` before any resource action
/// (`docs/24` §4.2).
#[test]
fn a_presenting_trace_is_refused_when_the_snapshot_declares_no_presentation() {
    let Some(fixture) = presenting_fixture() else {
        return;
    };
    let declared = fixture.provider.capabilities();
    assert!(declared.supports_presentation);
    declared
        .admit(&fixture.trace, &fixture.resources)
        .expect("the declared present bits admit the fixture");

    let mut without_presentation = declared.clone();
    without_presentation.supports_presentation = false;
    without_presentation.max_present_targets = 0;
    without_presentation.supported_present_modes.clear();
    without_presentation.max_present_image_count = 0;
    let refused = without_presentation
        .admit(&fixture.trace, &fixture.resources)
        .unwrap_err();
    eprintln!("present bits off: refused: {refused:?}");
    assert_eq!(refused.slug, "present_targets_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

// ---------------------------------------------------------------------------
// Vertex input: caller-held vertex and index buffers (`research/docs/23`
// §3.3). The rail uploads each bound pool view, builds the pipeline's vertex
// input state from the contract layout and issues `vkCmdBindVertexBuffers` +
// `vkCmdDrawIndexed`.
// ---------------------------------------------------------------------------

/// Vertex stage of the vertex-input rail: `spirv-as` output of
/// `render_spv/quad_indexed.vert.spvasm` (entry `vertex_buffer_main`). It reads
/// `vec2` position from location 0 — i.e. from the caller's vertex buffer —
/// rather than generating positions from `gl_VertexIndex`.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");

const VERTEX_VIEW: ViewId = ViewId::new(703);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(803);
const INDEX_VIEW: ViewId = ViewId::new(704);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(804);

/// The fragment output the reviewed quad stores, in the attachment's own byte
/// order: `(64/255, 128/255, 192/255, 1)` as tightly packed `Rgba8Unorm`.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// Two `float32` components per vertex, four vertices: the reviewed stream.
fn quad_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}

/// The same stream collapsed onto one NDC corner: a draw that reads the
/// caller's bytes covers no pixel centre, while a `vertex_id` triangle would
/// still cover all four texels. This is what makes "the stream was read"
/// falsifiable.
fn collapsed_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for _ in 0..4 {
        bytes.extend_from_slice(&(-1.0_f32).to_ne_bytes());
        bytes.extend_from_slice(&(-1.0_f32).to_ne_bytes());
    }
    bytes
}

/// The reviewed stream's four vertices moved into the left column: `(-1,-1)
/// (0,-1) (-1,1) (0,1)`. With the six reviewed indices the two triangles cover
/// exactly two of the four texels, which leaves the other two for the loaded
/// bytes to show through.
///
/// The band is chosen to be *symmetric under the NDC y flip* the two rails
/// disagree about — Vulkan's y points down, Metal's points up — so both rails
/// cover the same texel pair `(row 0, col 0)` and `(row 1, col 0)` and the byte
/// expectation stays identical. A quadrant-shaped band covered the top-left
/// texel on Lavapipe and the bottom-left one on Apple Paravirtual (CI run
/// `34870722991`), which no single expectation can describe.
fn left_column_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [(-1.0_f32, -1.0_f32), (0.0, -1.0), (-1.0, 1.0), (0.0, 1.0)] {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}
fn quad_index_bytes() -> Vec<u8> {
    // Two `uint16` triangles over the four corners: `0,1,2` and `1,3,2`.
    let mut bytes = Vec::with_capacity(12);
    for index in [0_u16, 1, 2, 1, 3, 2] {
        bytes.extend_from_slice(&index.to_ne_bytes());
    }
    bytes
}

/// The reviewed vertex layout: one `float32x2` position stream, stride eight.
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

/// A provider with the reviewed vertex-input pipeline registered plus the trace
/// and resource table that bind `vertex_bytes` and `index_bytes`.
fn vertex_input_fixture(
    vertex_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
) -> Option<(VulkanComputeProvider, ComputeTrace, ResourceTableSnapshot)> {
    vertex_input_fixture_with_load(vertex_bytes, index_bytes, false)
}

/// The same fixture with the attachment's load op selected: `clear` for the
/// milestone's shape, `load` for the pass that uploads the declaring view's
/// bytes first (`research/docs/23` §3.3).
fn vertex_input_fixture_with_load(
    vertex_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    load: bool,
) -> Option<(VulkanComputeProvider, ComputeTrace, ResourceTableSnapshot)> {
    let executor = executor()?;
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let fixture_digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, fixture_digest(b"render_e2e_quad_compute"))
        .expect("the compute pipeline registers");
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_buffer_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: quad_layout(),
            },
            vertex_spirv: QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: fixture_digest(b"render_e2e_quad_stages"),
        })
        .expect("the vertex-input render pipeline registers");

    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    if load {
        // The declaring compute case's attachment view carries the previous
        // bytes; the rail uploads them before the draw.
        pass.color_attachments[0].load = LoadOp::Load;
    }
    // The draw's count is the index count: six for the reviewed quad, three for
    // the load fixture's single triangle.
    pass.vertices = u32::try_from(index_bytes.len() / 2).expect("uint16 index count");
    pass.vertex_buffers = vec![BufferView {
        view_id: VERTEX_VIEW,
        metal_binding: 0,
        allocation_id: VERTEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(vertex_bytes.len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vertex_bytes.clone()),
    }];
    pass.indices = Some(IndexBufferBinding {
        view: BufferView {
            view_id: INDEX_VIEW,
            metal_binding: 0,
            allocation_id: INDEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(index_bytes.len()).expect("index length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(index_bytes.clone()),
        },
        format: IndexFormat::Uint16,
    });

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(12),
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
        (VERTEX_ALLOCATION, 32),
        (INDEX_ALLOCATION, 12),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }

    Some((provider, trace, resources))
}

/// Submit one vertex-input trace through admission and collect its writebacks.
fn submit_vertex_input(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<(ViewId, Vec<u8>)> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the vertex-input trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(trace)
        .expect("the writebacks cover the trace");
    submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect()
}

#[test]
fn indexed_quad_reads_the_caller_streams_and_lands_the_attachment() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture(quad_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("indexed quad readback: {}", hex(&attachment));
    assert_eq!(attachment.len(), 16);
    for texel in attachment.chunks_exact(4) {
        assert_eq!(texel, QUAD_TEXEL, "every texel is the fragment output");
    }
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == CLEAR_SENTINEL),
        "the clear sentinel is fully covered: {}",
        hex(&attachment)
    );
}

#[test]
fn the_draw_reads_the_caller_bytes_rather_than_vertex_id() {
    // The falsification for "the rail still draws the milestone triangle": the
    // same pipeline, the same index buffer, but a vertex stream collapsed onto
    // one corner. Every texel keeps the clear sentinel, which a `vertex_id`
    // triangle could not produce.
    let Some((provider, trace, resources)) =
        vertex_input_fixture(collapsed_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("collapsed quad readback: {}", hex(&attachment));
    for texel in attachment.chunks_exact(4) {
        assert_eq!(
            texel,
            CLEAR_SENTINEL,
            "a degenerate draw leaves the sentinel: {}",
            hex(&attachment)
        );
    }
}

#[test]
fn a_loading_pass_keeps_the_bytes_the_draw_does_not_cover() {
    // The `LoadOp::Load` shape (`research/docs/23` §3.3): the same reviewed
    // layout and pipeline, but a left-column stream that leaves half the pixel
    // centres uncovered, drawn into an attachment the rail first fills with the
    // declaring case's own bytes. The uncovered texels keep those bytes, which
    // is what makes "the upload happened" falsifiable: a clearing pass would
    // leave the clear colour there instead.
    let Some((provider, trace, resources)) =
        vertex_input_fixture_with_load(left_column_vertex_bytes(), quad_index_bytes(), true)
    else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("loading triangle readback: {}", hex(&attachment));
    assert_eq!(attachment.len(), 16);
    let covered = attachment
        .chunks_exact(4)
        .filter(|texel| *texel == QUAD_TEXEL)
        .count();
    let previous = attachment
        .chunks_exact(4)
        .filter(|texel| *texel == ATTACHMENT_WORD)
        .count();
    assert_eq!(
        (covered, previous),
        (2, 2),
        "the left column is drawn and the other two texels keep the uploaded bytes: {}",
        hex(&attachment)
    );

    // The counter-shape: the identical geometry and index buffer with
    // `LoadOp::Clear` leaves the clear sentinel everywhere the draw missed,
    // so the two runs differ in exactly the texel the load is about.
    let Some((provider, trace, resources)) =
        vertex_input_fixture(left_column_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let cleared = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("clearing triangle readback: {}", hex(&cleared));
    assert!(
        cleared.chunks_exact(4).any(|texel| texel == CLEAR_SENTINEL),
        "a clearing pass leaves its own colour where the draw missed: {}",
        hex(&cleared)
    );
    assert_ne!(cleared, attachment);
}

#[test]
fn vertex_input_refusals_name_the_stream_that_cannot_be_read() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture(quad_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };

    // A lease-backed stream now resolves through the shared registry: a lease
    // no owner ever imported is refused by name at resolve time, before any
    // device object exists, instead of being read as if the trace owned it
    // (`research/docs/23` §71, R3c).
    let mut leased = trace.clone();
    if let Some(TracePass::Render(pass)) = leased.passes.last_mut() {
        pass.vertex_buffers[0].source = BufferSource::StagedLease(LeaseId::new(7));
    }
    let admitted = provider
        .capabilities()
        .validate_trace(leased, resources.clone())
        .expect("a staged lease is a well-formed declaration");
    let refused = provider
        .submit(admitted)
        .expect_err("a lease no owner ever imported cannot be read");
    eprintln!("unimported staged lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    assert_eq!(refused.phase, ProviderPhase::Resolve);

    // An index view too short for the draw's index count: the rail proves the
    // footprint before touching the device.
    let mut short_index = trace.clone();
    if let Some(TracePass::Render(pass)) = short_index.passes.last_mut() {
        let indices = pass.indices.as_mut().expect("the fixture is indexed");
        indices.view.length = 4;
        indices.view.source = BufferSource::OwnedBytes(vec![0; 4]);
    }
    let refused = match provider
        .capabilities()
        .validate_trace(short_index, resources.clone())
    {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("a four-byte index view cannot cover six indices"),
        Err(error) => error,
    };
    eprintln!("short index view refused: {refused:?}");
    assert_eq!(refused.slug, "render_index_buffer_footprint_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);

    // An index that names a vertex the bound stream does not cover: the
    // footprint proof is what makes the read safe rather than merely bound.
    let mut out_of_range = quad_index_bytes();
    out_of_range[0..2].copy_from_slice(&4_u16.to_ne_bytes());
    let Some((provider, trace, resources)) =
        vertex_input_fixture(quad_vertex_bytes(), out_of_range)
    else {
        return;
    };
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the trace is well formed");
    let refused = provider
        .submit(admitted)
        .expect_err("index 4 names a fifth vertex the 32-byte stream does not hold");
    eprintln!("out-of-range index refused: {refused:?}");
    assert_eq!(refused.slug, "render_vertex_buffer_footprint_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// One owner allocation the provider imports without copying
/// (`research/docs/23` §71, R3c).
///
/// The provider binds this address directly, so the allocation has to stay
/// alive — and at this address — until the owner releases the import. The
/// alignment is the device's own import alignment, which the caller reads from
/// the provider before allocating.
struct AlignedBuffer {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, alignment)
            .expect("the import alignment is a valid allocation alignment");
        let pointer = unsafe { std::alloc::alloc(layout) };
        let pointer = std::ptr::NonNull::new(pointer).expect("aligned allocation failed");
        Self { pointer, layout }
    }

    fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// The staged half of the render-input lease channel (`research/docs/23` §71,
/// R3c): the vertex and index bytes arrive as staged leases instead of
/// trace-owned bytes, and the attachment is byte-for-byte the one the owned
/// fixture lands.
#[test]
fn a_staged_lease_vertex_stream_renders_the_same_bytes_as_owned_bytes() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture(quad_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };
    let owned = readback(
        &submit_vertex_input(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );

    let epoch = provider.device_epoch();
    let vertex_lease = LeaseId::new(31);
    let index_lease = LeaseId::new(32);
    let vertex_reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: vertex_lease,
            allocation_id: VERTEX_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 32,
    };
    let index_reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: index_lease,
            allocation_id: INDEX_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 12,
    };
    provider
        .import_staged_lease(
            StagedLease::new(vertex_reservation, quad_vertex_bytes())
                .expect("the staged stream carries one byte per declared byte"),
        )
        .expect("the provider stages the owner's vertex bytes");
    provider
        .import_staged_lease(
            StagedLease::new(index_reservation, quad_index_bytes())
                .expect("the staged stream carries one byte per declared byte"),
        )
        .expect("the provider stages the owner's index bytes");

    let mut leased = trace.clone();
    if let Some(TracePass::Render(pass)) = leased.passes.last_mut() {
        pass.vertex_buffers[0].source = BufferSource::StagedLease(vertex_lease);
        pass.indices
            .as_mut()
            .expect("the fixture is indexed")
            .view
            .source = BufferSource::StagedLease(index_lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_lease(vertex_reservation)
        .expect("the vertex reservation covers its view");
    leased_resources
        .insert_lease(index_reservation)
        .expect("the index reservation covers its view");
    let attachment = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("staged lease attachment: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the staged lease channel uploads the same bytes the trace-owned fixture carries"
    );

    // The staged bytes are the provider's copy: releasing them is what the
    // owner's ledger drives, and a submission after the release is refused by
    // name instead of silently falling back to a copy of its own.
    provider
        .release_staged_lease(vertex_lease)
        .expect("the staged vertex lease is released");
    provider
        .release_staged_lease(index_lease)
        .expect("the staged index lease is released");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released staged lease cannot be read");
    eprintln!("released staged lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
}

/// The no-copy half of the render-input lease channel (`research/docs/23` §71,
/// R3c): the vertex and index bytes stay in the owner's own mapping, the device
/// reads that mapping, and the registry's hold is retired once the pass's fence
/// has signalled.
///
/// The falsifications are the point: a rail that snapshotted at import would
/// still draw the original quad after the owner rewrites its pages, and a rail
/// whose footprint proof read stale bytes would not see an index the owner
/// wrote past the stream's coverage.
#[test]
fn a_borrowed_lease_vertex_stream_reads_the_owners_pages() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture(quad_vertex_bytes(), quad_index_bytes())
    else {
        return;
    };
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let owned = readback(
        &submit_vertex_input(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );

    let mut owner_vertices = AlignedBuffer::new(32, alignment as usize);
    owner_vertices
        .as_mut_slice()
        .copy_from_slice(&quad_vertex_bytes());
    let mut owner_indices = AlignedBuffer::new(12, alignment as usize);
    owner_indices
        .as_mut_slice()
        .copy_from_slice(&quad_index_bytes());

    let epoch = provider.device_epoch();
    let vertex_lease = LeaseId::new(41);
    let index_lease = LeaseId::new(42);
    let vertex_reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: vertex_lease,
            allocation_id: VERTEX_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 32,
    };
    let index_reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: index_lease,
            allocation_id: INDEX_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 12,
    };
    // SAFETY: both owner allocations outlive every submission below and the
    // provider's release of the two imports.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(vertex_reservation, owner_vertices.as_ptr() as usize)
                    .expect("the owner's vertex window is a valid reservation"),
            )
            .expect("the provider imports the owner's vertex window");
        provider
            .import_borrowed_lease(
                BorrowedLease::new(index_reservation, owner_indices.as_ptr() as usize)
                    .expect("the owner's index window is a valid reservation"),
            )
            .expect("the provider imports the owner's index window");
    }

    let mut leased = trace.clone();
    if let Some(TracePass::Render(pass)) = leased.passes.last_mut() {
        pass.vertex_buffers[0].source = BufferSource::BorrowedNoCopy(vertex_lease);
        pass.indices
            .as_mut()
            .expect("the fixture is indexed")
            .view
            .source = BufferSource::BorrowedNoCopy(index_lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_lease(vertex_reservation)
        .expect("the vertex reservation covers its view");
    leased_resources
        .insert_lease(index_reservation)
        .expect("the index reservation covers its view");

    let attachment = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("borrowed lease attachment: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the no-copy channel binds the owner's window and lands the owned fixture's bytes"
    );

    // The pass is synchronous, so its fence is the retirement evidence: both
    // holds are back to zero by the time this submission returns.
    let registry = provider.borrowed_registry();
    assert_eq!(
        registry.outstanding(vertex_lease),
        Some(0),
        "the vertex hold is retired once the fence signals"
    );
    assert_eq!(
        registry.outstanding(index_lease),
        Some(0),
        "the index hold is retired once the fence signals"
    );

    // A device that had snapshotted the owner's pages at import would still
    // draw the original quad; collapsing the stream onto one corner changes
    // every texel to the clear sentinel instead.
    owner_vertices
        .as_mut_slice()
        .copy_from_slice(&collapsed_vertex_bytes());
    let collapsed = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!(
        "owner-rewritten vertex window readback: {}",
        hex(&collapsed)
    );
    assert!(
        collapsed
            .chunks_exact(4)
            .all(|texel| texel == CLEAR_SENTINEL),
        "a write into the owner's mapping after the import reaches the draw: {}",
        hex(&collapsed)
    );

    // The index half is read from the owner's pages as well: every index
    // rewritten to vertex 0 degenerates both triangles, which leaves the clear
    // sentinel everywhere the quad covered.
    owner_vertices
        .as_mut_slice()
        .copy_from_slice(&quad_vertex_bytes());
    owner_indices.as_mut_slice().fill(0);
    let degenerated = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!(
        "owner-rewritten index window readback: {}",
        hex(&degenerated)
    );
    assert!(
        degenerated
            .chunks_exact(4)
            .all(|texel| texel == CLEAR_SENTINEL),
        "the draw follows the index values the owner wrote after the import: {}",
        hex(&degenerated)
    );

    // The footprint proof reads the same window: an index the owner writes past
    // the stream's coverage is refused by name before any device object exists.
    owner_indices
        .as_mut_slice()
        .copy_from_slice(&quad_index_bytes());
    owner_indices.as_mut_slice()[0..2].copy_from_slice(&4_u16.to_ne_bytes());
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources.clone())
        .expect("the declaration stays well formed");
    let refused = provider
        .submit(admitted)
        .expect_err("index 4 names a vertex the owner's stream does not cover");
    eprintln!("owner-written out-of-range index refused: {refused:?}");
    assert_eq!(refused.slug, "render_vertex_buffer_footprint_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);

    // Once the owner releases both imports, the same declaration is refused by
    // name instead of being read through a mapping the provider no longer owns.
    registry
        .release(vertex_lease)
        .expect("no retain is outstanding after the fence");
    registry
        .release(index_lease)
        .expect("no retain is outstanding after the fence");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released no-copy lease cannot be read");
    eprintln!("released borrowed lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
}

/// The staged half of the attachment-load lease channel (`research/docs/23`
/// §74, R5b): the previous contents a `LoadOp::Load` attachment uploads arrive
/// as a staged lease instead of trace-owned bytes, and the pass lands
/// byte-for-byte the attachment the owned fixture lands.
#[test]
fn a_staged_lease_attachment_load_renders_the_same_bytes_as_owned_bytes() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture_with_load(left_column_vertex_bytes(), quad_index_bytes(), true)
    else {
        return;
    };
    let owned = readback(
        &submit_vertex_input(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );

    let epoch = provider.device_epoch();
    let lease = LeaseId::new(51);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 16,
    };
    provider
        .import_staged_lease(
            StagedLease::new(reservation, ATTACHMENT_WORD.repeat(4))
                .expect("the staged window carries one byte per declared byte"),
        )
        .expect("the provider stages the owner's previous bytes");

    // The attachment's previous contents are the declaring view's own bytes, so
    // the lease is named by the binding that declares that view — the compute
    // pass's read — rather than by the render pass, which restates only the
    // attachment's identity (`research/docs/23` §3.3).
    let mut leased = trace.clone();
    if let Some(TracePass::Compute(pass)) = leased.passes.first_mut() {
        pass.buffers[0].source = BufferSource::StagedLease(lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_lease(reservation)
        .expect("the attachment reservation covers its view");
    let attachment = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("staged attachment load: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the staged lease channel uploads the previous contents the owned fixture carries"
    );

    // The staged bytes are the provider's copy: releasing them is what the
    // owner's ledger drives, and a submission after the release is refused by
    // name instead of silently uploading the last copy it saw.
    provider
        .release_staged_lease(lease)
        .expect("the staged attachment lease is released");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released staged lease cannot be read");
    eprintln!("released staged attachment lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
}

/// The no-copy half of the attachment-load lease channel (`research/docs/23`
/// §74, R5b): the previous contents stay in the owner's own mapping, the
/// device reads that mapping as the copy's transfer source, and the registry's
/// hold is retired once the pass's fence has signalled.
///
/// The falsifications are the point: a rail that snapshotted the owner's window
/// when the view was declared would keep uploading the first word after the
/// owner rewrites the pages, and a rail whose upload read stale bytes could not
/// show the owner's new word where the draw leaves a texel alone.
#[test]
fn a_borrowed_lease_attachment_load_reads_the_owners_pages() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture_with_load(left_column_vertex_bytes(), quad_index_bytes(), true)
    else {
        return;
    };
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let owned = readback(
        &submit_vertex_input(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );

    let mut owner_attachment = AlignedBuffer::new(16, alignment as usize);
    owner_attachment
        .as_mut_slice()
        .copy_from_slice(&ATTACHMENT_WORD.repeat(4));

    let epoch = provider.device_epoch();
    let lease = LeaseId::new(52);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 16,
    };
    // SAFETY: the owner allocation outlives every submission below and the
    // provider's release of the import.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(reservation, owner_attachment.as_ptr() as usize)
                    .expect("the owner's attachment window is a valid reservation"),
            )
            .expect("the provider imports the owner's attachment window");
    }

    let mut leased = trace.clone();
    if let Some(TracePass::Compute(pass)) = leased.passes.first_mut() {
        pass.buffers[0].source = BufferSource::BorrowedNoCopy(lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_lease(reservation)
        .expect("the attachment reservation covers its view");
    let attachment = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("borrowed attachment load: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the no-copy channel uploads the owner's window and lands the owned fixture's bytes"
    );

    // The pass is synchronous, so its fence is the retirement evidence: the
    // hold the attachment's own window took is back to zero.
    let registry = provider.borrowed_registry();
    assert_eq!(
        registry.outstanding(lease),
        Some(0),
        "the attachment hold is retired once the fence signals"
    );

    // A device that had snapshotted the owner's pages at import would keep
    // uploading the first word; the owner's rewrite reaches the two texels the
    // draw leaves alone instead.
    let rewritten_word = [0x55_u8, 0x66, 0x77, 0x88];
    owner_attachment
        .as_mut_slice()
        .copy_from_slice(&rewritten_word.repeat(4));
    let rewritten = readback(
        &submit_vertex_input(&provider, &leased, &leased_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!(
        "owner-rewritten attachment window readback: {}",
        hex(&rewritten)
    );
    let covered = rewritten
        .chunks_exact(4)
        .filter(|texel| *texel == QUAD_TEXEL)
        .count();
    let loaded = rewritten
        .chunks_exact(4)
        .filter(|texel| *texel == rewritten_word)
        .count();
    assert_eq!(
        (covered, loaded),
        (2, 2),
        "the owner's rewritten window is what the uncovered texels upload: {}",
        hex(&rewritten)
    );
    assert!(
        !rewritten
            .chunks_exact(4)
            .any(|texel| texel == ATTACHMENT_WORD),
        "no texel keeps the pre-rewrite word a snapshot would have pinned: {}",
        hex(&rewritten)
    );

    // Once the owner releases the import, the same declaration is refused by
    // name instead of being read through a mapping the provider no longer owns.
    registry
        .release(lease)
        .expect("no retain is outstanding after the fence");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released no-copy lease cannot be read");
    eprintln!("released borrowed attachment lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
}

/// The reviewed sampling pair (`research/docs/23` §3.3, v70): the full-screen
/// geometry with its `Location 0` uv varying, and the fragment stage that
/// samples `DescriptorSet 0 / Binding 0` with it.
const SAMPLED_QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/sampled_quad.vert.spv");
const SAMPLED_UNORM8_FRAG_SPV: &[u8] =
    include_bytes!("../src/render_spv/solid_unorm8_sampled.frag.spv");

const SAMPLED_TEXTURE_VIEW: ViewId = ViewId::new(712);
const SAMPLED_TEXTURE_ALLOCATION: AllocationId = AllocationId::new(812);

/// The sixteen texels the sampled texture holds, row-major: four distinct
/// channels per texel, so a transposed, flipped or filtered read lands bytes
/// no expectation of this fixture contains.
fn sampled_texels() -> Vec<u8> {
    (0..4u8)
        .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
        .collect()
}

/// The render sampler's fixture: the reviewed sampling pair, a 4×4 attachment,
/// and the 4×4 texture the fragment stage samples at texel centres. Shared by
/// the milestone case and the present-rail refusal beside it.
fn sampled_fixture() -> Option<(
    VulkanComputeProvider,
    ComputeTrace,
    ResourceTableSnapshot,
    Vec<u8>,
)> {
    let executor = executor()?;
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
        .compile_pipeline(&function, digest(b"render_e2e_sampled_compute"))
        .expect("the compute pipeline registers");
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: SAMPLED_QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SAMPLED_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_sampled_stages"),
        })
        .expect("the sampled pair registers");

    let texels = sampled_texels();
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 4, 4);
    pass.textures = vec![TextureView {
        view_id: SAMPLED_TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: SAMPLED_TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 4,
        height: 4,
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(texels.clone()),
    }];

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(11),
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
                        // 4×4 texels of four bytes: exactly the extent the
                        // render attachment restates.
                        length: 64,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0; 64]),
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
    for (allocation, size) in [(ATTACHMENT_ALLOCATION, 64), (SCRATCH_ALLOCATION, 8)] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    Some((provider, trace, resources, texels))
}

/// The whole chain for the render sampler: a 4×4 attachment, one texture the
/// fragment stage samples at texel centres, and the bytes that land in the
/// writeback channel. The falsification is the point: a rail that ignores the
/// texture reads back the clear sentinel, one that filters reads a neighbour's
/// texel, and one that flips or transposes the uv reads another row or column.
#[test]
fn a_render_pass_samples_its_texture_and_lands_the_texels() {
    let Some((provider, trace, resources, texels)) = sampled_fixture() else {
        return;
    };
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the sampled trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!(
        "sampled attachment readback: {} ({} bytes)",
        hex(&attachment),
        attachment.len()
    );
    eprintln!("uploaded texels: {}", hex(&texels));

    assert_eq!(attachment.len(), 64);
    assert_eq!(
        attachment, texels,
        "the fragment stage's sample over a texel-aligned texture is the uploaded texel itself"
    );
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == CLEAR_SENTINEL),
        "a surviving clear sentinel means the sampled pass did not cover every texel: {}",
        hex(&attachment)
    );

    // The falsification control: the same pass without its texture binding is
    // refused by name instead of executed with an unbound descriptor (which
    // would read the clear, not the texels).
    let mut unbound = trace.clone();
    if let Some(TracePass::Render(pass)) = unbound.passes.last_mut() {
        pass.textures = Vec::new();
    }
    let refused = match provider
        .capabilities()
        .validate_trace(unbound, resources.clone())
    {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("the reviewed sampling pair needs a texture binding"),
        Err(error) => error,
    };
    eprintln!("sampling without a texture refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_binding_required");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// The present rail binds the format's solid fragment module rather than the
/// reviewed sampling pair, so a present tail beside a sampled pass is refused
/// by the same slugs the offscreen rail states for that pair
/// (`render_texture_stage_unsupported` for a bound texture,
/// `render_texture_binding_required` for a missing one) instead of being
/// executed with the binding silently dropped (`research/docs/23` §3.3, v70;
/// R4a increment).
#[test]
fn a_presenting_pass_beside_the_sampling_pair_is_refused_by_name() {
    let Some((provider, trace, resources, _texels)) = sampled_fixture() else {
        return;
    };
    let mut presenting = trace.clone();
    attach_present(
        &mut presenting,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );

    // The bound-texture shape is the offscreen rail's, not the present rail's.
    let admitted = provider
        .capabilities()
        .validate_trace(presenting.clone(), resources.clone())
        .expect("the sampled declaration stays well formed");
    let refused = provider
        .submit(admitted)
        .expect_err("the present rail binds no descriptor set for the sampling pair");
    eprintln!("texture beside a present pass refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_stage_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);

    // The same pass without its binding: the sampling pair needs its texture,
    // and the present rail still refuses rather than sampling nothing.
    let mut unbound = presenting.clone();
    if let Some(TracePass::Render(pass)) = unbound.passes.last_mut() {
        pass.textures = Vec::new();
    }
    let admitted = provider
        .capabilities()
        .validate_trace(unbound, resources)
        .expect("the unbound declaration stays well formed");
    let refused = provider
        .submit(admitted)
        .expect_err("the sampling pair without its texture is not the present rail's shape");
    eprintln!("sampling pair without a texture refused beside a present pass: {refused:?}");
    assert_eq!(refused.slug, "render_texture_binding_required");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// Submit one sampled trace exactly as the milestone case does and land the
/// attachment's bytes (`research/docs/23` §75, R5c).
fn submit_sampled(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<u8> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the sampled declaration is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The sixteen texels the owner's rewritten window holds
/// (`research/docs/23` §75, R5c): four distinct channels per texel, none of
/// them equal to the fixture's own, so "the device read the owner's pages after
/// the rewrite" is falsifiable per texel.
fn rewritten_texels() -> Vec<u8> {
    (0..4u8)
        .flat_map(|y| (0..4u8).flat_map(move |x| [0x80 | x, 0x40 | y, x ^ y, 0xff]))
        .collect()
}

/// The staged half of the render-texture lease channel (`research/docs/23`
/// §75, R5c): the sampled texels arrive as a staged lease instead of
/// trace-owned bytes, and the identity-sampling fixture lands the same
/// attachment bytes.
///
/// The reservation is the page-aligned window a real owner has to hand out —
/// four kilobytes for a sixty-four-byte surface — so this case also pins the
/// window rule: the texture is the reservation's first sixty-four bytes, and
/// the padding behind them is not part of any declaration.
#[test]
fn a_staged_lease_render_texture_samples_the_providers_copy() {
    let Some((provider, trace, resources, texels)) = sampled_fixture() else {
        return;
    };
    let owned = submit_sampled(&provider, &trace, &resources);

    let epoch = provider.device_epoch();
    let lease = LeaseId::new(61);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: SAMPLED_TEXTURE_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 4096,
    };
    let mut staged_bytes = vec![0x5a_u8; 4096];
    staged_bytes[..texels.len()].copy_from_slice(&texels);
    provider
        .import_staged_lease(
            StagedLease::new(reservation, staged_bytes)
                .expect("the staged window carries one byte per reserved byte"),
        )
        .expect("the provider imports the owner's staged window");

    let mut leased = trace.clone();
    if let Some(TracePass::Render(pass)) = leased.passes.last_mut() {
        pass.textures[0].source = TextureSource::StagedLease(lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_allocation(AllocationRecord {
            allocation_id: SAMPLED_TEXTURE_ALLOCATION,
            owner_epoch: epoch,
            size: 4096,
        })
        .expect("the owner's window allocation is well formed");
    leased_resources
        .insert_lease(reservation)
        .expect("the staged reservation covers the texture");
    let attachment = submit_sampled(&provider, &leased, &leased_resources);
    eprintln!("staged lease texture: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the staged lease channel uploads the texels the trace-owned fixture carries"
    );

    // The staged bytes are the provider's copy: releasing them is what the
    // owner's ledger drives, and the same declaration is refused afterwards.
    provider
        .release_staged_lease(lease)
        .expect("the staged texture lease is released");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released staged lease cannot be read");
    eprintln!("released staged texture lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
}

/// The no-copy half of the render-texture lease channel (`research/docs/23`
/// §75, R5c): the sampled texels stay in the owner's own mapping, the device
/// reads that mapping as the copy's transfer source, and the registry's hold is
/// retired once the pass's fence has signalled.
///
/// The falsifications are the point: a rail that snapshotted the owner's window
/// when the texture was declared — or that uploaded it into its own image while
/// building the pass — would keep sampling the first texels after the owner
/// rewrites the pages, and a rail whose copy read stale bytes could not show
/// the owner's new texels at all.
#[test]
fn a_borrowed_lease_render_texture_reads_the_owners_pages() {
    let Some((provider, trace, resources, texels)) = sampled_fixture() else {
        return;
    };
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let owned = submit_sampled(&provider, &trace, &resources);

    // The owner's window is the page-aligned reservation a real backing needs;
    // the texture is its first sixty-four bytes.
    let mut owner_texture = AlignedBuffer::new(4096, alignment as usize);
    owner_texture.as_mut_slice().fill(0x5a);
    owner_texture.as_mut_slice()[..texels.len()].copy_from_slice(&texels);

    let epoch = provider.device_epoch();
    let lease = LeaseId::new(62);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: SAMPLED_TEXTURE_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 4096,
    };
    // SAFETY: the owner allocation outlives every submission below and the
    // provider's release of the import.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(reservation, owner_texture.as_ptr() as usize)
                    .expect("the owner's texture window is a valid reservation"),
            )
            .expect("the provider imports the owner's texture window");
    }

    let mut leased = trace.clone();
    if let Some(TracePass::Render(pass)) = leased.passes.last_mut() {
        pass.textures[0].source = TextureSource::BorrowedNoCopy(lease);
    }
    let mut leased_resources = resources.clone();
    leased_resources
        .insert_allocation(AllocationRecord {
            allocation_id: SAMPLED_TEXTURE_ALLOCATION,
            owner_epoch: epoch,
            size: 4096,
        })
        .expect("the owner's window allocation is well formed");
    leased_resources
        .insert_lease(reservation)
        .expect("the borrowed reservation covers the texture");
    let attachment = submit_sampled(&provider, &leased, &leased_resources);
    eprintln!("borrowed lease texture: {}", hex(&attachment));
    assert_eq!(
        attachment, owned,
        "the no-copy channel samples the owner's window and lands the owned fixture's bytes"
    );

    // The pass is synchronous, so its fence is the retirement evidence: the
    // hold the texture's own window took is back to zero.
    let registry = provider.borrowed_registry();
    assert_eq!(
        registry.outstanding(lease),
        Some(0),
        "the texture hold is retired once the fence signals"
    );

    // A device that had snapshotted the owner's pages at import would keep
    // sampling the first texels; the owner's rewrite reaches every texel the
    // draw reads instead.
    let rewritten = rewritten_texels();
    owner_texture.as_mut_slice()[..rewritten.len()].copy_from_slice(&rewritten);
    let attachment = submit_sampled(&provider, &leased, &leased_resources);
    eprintln!(
        "owner-rewritten texture window readback: {}",
        hex(&attachment)
    );
    assert_eq!(
        attachment,
        rewritten,
        "the owner's rewritten window is what the fragment stage samples: {}",
        hex(&attachment)
    );
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == &texels[..4]),
        "no texel keeps the pre-rewrite value a snapshot would have pinned: {}",
        hex(&attachment)
    );

    // Once the owner releases the import, the same declaration is refused by
    // name instead of being read through a mapping the provider no longer owns.
    registry
        .release(lease)
        .expect("no retain is outstanding after the fence");
    let admitted = provider
        .capabilities()
        .validate_trace(leased.clone(), leased_resources)
        .expect("the declaration stays well formed after the release");
    let refused = provider
        .submit(admitted)
        .expect_err("a released no-copy lease cannot be read");
    eprintln!("released borrowed texture lease refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
}
