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
    AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource,
    BufferView, ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy,
    ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue,
    IndirectCommandBufferDescriptor, IndirectCommandDescriptor, IndirectCommandKind,
    IndirectCommandPayload, IndirectCommandRange, InitialState, LoadOp, OperationId, PipelineId,
    PresentDescriptor, PresentMode, PresentTarget, ProviderCapabilities, ProviderError,
    ProviderErrorClass, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
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
const SCRATCH_VIEW: ViewId = ViewId::new(702);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(802);

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
        vertices: 3,
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
            color_format: format,
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

    // The dimension bit is load-bearing as well: 3×3 is outside the rail's
    // window, and that refusal comes before the extent agreement behind it.
    let oversized = oversized_trace(&fixture);
    let refused = admit_error(&declared, &oversized, &fixture.resources);
    eprintln!("3x3 attachment: refused: {refused:?}");
    assert_eq!(refused.slug, "attachment_dimension_limit");
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
                .map(|contract| contract.color_format),
            Some(format),
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
            .color_format = other;
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
fn presenting_fixture() -> Option<Fixture> {
    let mut fixture = fixture(AttachmentFormat::Rgba8Unorm)?;
    let Some(TracePass::Render(pass)) = fixture.trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    pass.present = Some(PresentDescriptor {
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
    });
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
