//! End-to-end render rail: one trace that carries a render pass, submitted
//! through the provider's own admission and submit path.
//!
//! The case is the milestone of `research/docs/23`: a 2×2 `Rgba8Unorm`
//! attachment cleared to a sentinel and then covered by the full-screen
//! triangle, whose fragment stage stores `40 80 c0 ff` per texel. What this
//! test measures is the whole chain rather than the rail alone — a
//! `ComputeTrace` with a render entry → `ProviderCapabilities::validate_trace`
//! → `ComputeProvider::submit` → the attachment's bytes in the returned
//! writebacks — so a rail that runs but lands nothing cannot pass.
//!
//! The attachment view is declared by the trace's compute pass because that is
//! what the render contract requires: `validate_serial_buffer_reuse` resolves
//! every attachment against the views the trace declares, and a compute pass
//! that only *reads* the view is exactly the sharing core admission admits
//! (`AttachmentComputeConflict` refuses the writable half). The compute pass
//! therefore does real work of its own — `copy_word` writes the scratch view —
//! and one submission carries both rails' writebacks.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompletionDisposition, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId, PipelineId,
    ProviderCapabilities, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// Vertex stage of the milestone: `spirv-as` output of the reviewed
/// `render_spv/fullscreen_triangle.vert.spvasm` (entry `vertex_main`).
const FULL_SCREEN_TRIANGLE_VERT_SPV: &[u8] =
    include_bytes!("../src/render_spv/fullscreen_triangle.vert.spv");

/// Fragment stage of the milestone: `spirv-as` output of the reviewed
/// `render_spv/solid_rgba8.frag.spvasm` (entry `fragment_main`).
const SOLID_RGBA8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_rgba8.frag.spv");

/// The reviewed compute fixture the declaring pass runs.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// What the fragment stage stores, read back one texel at a time: a 2×2
/// `R8G8B8A8_UNORM` attachment holds `40 80 c0 ff` four times.
const EXPECTED_TEXELS: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The `LoadOp::Clear` sentinel (`research/docs/23` §1.3). A texel still
/// holding it proves the draw did not cover that pixel, so "the pass ran" is
/// falsifiable rather than assumed.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const ATTACHMENT_VIEW: ViewId = ViewId::new(701);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(801);
const SCRATCH_VIEW: ViewId = ViewId::new(702);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(802);

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One provider context with both pipelines registered and one render-bearing
/// trace built against it.
struct Fixture {
    provider: VulkanComputeProvider,
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
    render_pipeline: PipelineId,
}

fn render_pass(pipeline: PipelineId, width: u64, height: u64) -> RenderPassDescriptor {
    RenderPassDescriptor {
        pipeline,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width,
            height,
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, width as u32, height as u32],
        vertices: 3,
    }
}

fn fixture() -> Option<Fixture> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
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
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_format: AttachmentFormat::Rgba8Unorm,
                vertex_layout: VertexLayout::None,
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_RGBA8_FRAG_SPV.to_vec(),
            logical_digest: fixture_digest(b"render_e2e_stages"),
        })
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
            TracePass::Render(render_pass(render.pipeline_id, 2, 2)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
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
    })
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

#[test]
fn render_pass_trace_executes_and_lands_attachment_bytes_through_writeback() {
    let Some(fixture) = fixture() else {
        return;
    };
    let writebacks = submit_fixture(&fixture);

    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!(
        "attachment readback: {} ({} bytes)",
        hex(&attachment),
        attachment.len()
    );
    eprintln!("expected: [{}] x4", hex(&EXPECTED_TEXELS));
    assert_eq!(attachment.len(), 16);
    assert_eq!(attachment, EXPECTED_TEXELS.repeat(4));
    // The attachment is cleared to `fe fe fe fe` and the draw then covers every
    // texel of the 2×2 viewport, so the clear's own bytes are observable only
    // where the draw did *not* land. Asserting that no texel still holds the
    // sentinel is what keeps "the readback is the draw's bytes" falsifiable:
    // a surviving sentinel would mean the coverage claim is false, and the
    // readback a clear value rather than the fragment stage's output.
    assert!(
        !attachment
            .chunks_exact(4)
            .any(|texel| texel == CLEAR_SENTINEL),
        "a surviving clear sentinel means the triangle did not cover every texel, so the \
         readback would be the ClearColor rather than the fragment stage's bytes: {}",
        hex(&attachment)
    );
    assert_ne!(CLEAR_SENTINEL, EXPECTED_TEXELS);

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

#[test]
fn the_same_trace_is_refused_when_the_provider_declares_no_render_support() {
    let Some(fixture) = fixture() else {
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
    trace.passes[1] = TracePass::Render(render_pass(fixture.render_pipeline, 3, 3));
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
