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
    AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat, BlendAttachment, BlendFactor,
    BlendOperation, BorrowedLease, BufferAccess, BufferLease, BufferSource, BufferView, ClearColor,
    ColorWriteMask, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue,
    FootprintProof, IndexBufferBinding, IndexFormat, IndirectCommandBufferDescriptor,
    IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload, IndirectCommandRange,
    InitialState, LeaseId, LeaseImporter, LeaseReservation, LoadOp, MultisampleState,
    NoCopyLeaseImporter, OperationId, PipelineId, PresentDescriptor, PresentMode, PresentTarget,
    ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderPhase, RenderAttachment,
    RenderPassBlend, RenderPassDescriptor, RenderPipelineContract, RenderPipelineStage,
    ResourceTableSnapshot, SampleCount, SamplerAddressMode, SamplerFilter, SamplerPolicy,
    SemanticDigest, StageBufferBinding, StageBufferView, StagedLease, StoreOp, TextureAccess,
    TextureBindingContract, TextureFormat, TextureSource, TextureType, TextureView, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    DeviceLossPoint, RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest,
    TranslatedRenderStage, VulkanComputeProvider, VulkanExecutor, PRESENT_TARGET_BUDGET,
    RESIDENT_TARGET_BUDGET,
};
use std::sync::Arc;

use metal2vulkan::reflect::DescriptorLayout;

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
const ADMITTED_FORMATS: [AttachmentFormat; 4] = AttachmentFormat::ADMITTED;

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
        // The rail's own map: one `vec4` module for the four-component formats
        // (both 8-bit UNORM layouts and the eight-byte float format,
        // `research/docs/23` §78), the single-component module for `R32Float`.
        AttachmentFormat::Rgba8Unorm
        | AttachmentFormat::Bgra8Unorm
        | AttachmentFormat::Rgba16Float => SOLID_UNORM8_FRAG_SPV,
        AttachmentFormat::R32Float => SOLID_R32F_FRAG_SPV,
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// The bytes one texel holds when the fragment stage stores
/// `(64/255, 128/255, 192/255, 1)` into an attachment of `format`.
///
/// One texel is the format's own width (`research/docs/23` §78): four bytes for
/// the 8-bit UNORM and single-channel formats, eight for `Rgba16Float`.
fn expected_texels(format: AttachmentFormat) -> Vec<u8> {
    match format {
        // The stage's components in the order it writes them.
        AttachmentFormat::Rgba8Unorm => vec![0x40, 0x80, 0xc0, 0xff],
        // The same colour with that layout's blue/red exchange applied: the
        // stored red `0x40` lands third, behind the stored blue `0xc0`.
        AttachmentFormat::Bgra8Unorm => vec![0xc0, 0x80, 0x40, 0xff],
        // A float attachment quantises nothing, so the texel is the stage's
        // `float 64/255` (`0x3e808081`) in little-endian byte order.
        AttachmentFormat::R32Float => vec![0x81, 0x80, 0x80, 0x3e],
        // The eight-byte format rounds each stored `float` to its nearest half
        // (round-to-nearest-even). Not one of the stage's three colour
        // constants is on the half grid, so the rounding is visible in the
        // bytes: `64/255 = 0x3e808081` → `0x3404` (up), `128/255` → `0x3804`
        // (up), `192/255 = 0x3f40c0c1` → `0x3a06` (down), and `1.0` stays
        // exactly `0x3c00`. That is the point of this case: a rail that stored
        // the four `f32` values, quantised them as 8-bit UNORM, or read the
        // texel as four bytes cannot land these bytes
        // (`research/docs/23` §78).
        AttachmentFormat::Rgba16Float => vec![
            0x04, 0x34, // red: half(64/255) = 0.2509765625
            0x04, 0x38, // green: half(128/255) = 0.501953125
            0x06, 0x3a, // blue: half(192/255) = 0.7529296875
            0x00, 0x3c, // alpha: 1.0 is exact in half
        ],
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// The bytes a *surviving clear* leaves in an attachment of `format`.
///
/// The rail's clear components are `byte/255` as floats at every format, so the
/// single-channel float attachment's sentinel is the float `254/255` rather than
/// the four sentinel bytes themselves.
fn clear_bytes(format: AttachmentFormat) -> Vec<u8> {
    match format {
        AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm => CLEAR_SENTINEL.to_vec(),
        AttachmentFormat::R32Float => (f32::from(CLEAR_SENTINEL[0]) / 255.0)
            .to_le_bytes()
            .to_vec(),
        // The eight-byte format's sentinel is four half floats, each the half
        // nearest `254/255 = 0.996078431` — `0x3bf8`, whose value is
        // `0.99609375`. The readback must be those very bytes: the four
        // sentinel bytes *widened* would be a different payload, and four `f32`
        // components would be eight bytes per channel (`research/docs/23` §78).
        AttachmentFormat::Rgba16Float => [0xf8, 0x3b].repeat(4),
        AttachmentFormat::R32Uint => panic!("R32Uint is outside the first render increment"),
    }
}

/// The clear one attachment of `format` is opened with: the four sentinel bytes
/// for the four-byte class, four sentinel halves for the eight-byte one.
fn attachment_clear(format: AttachmentFormat) -> ClearColor {
    ClearColor::from_bytes(&clear_bytes(format)).expect("a clear is one texel of its format")
}

/// The attachment view's declared length in bytes: 2×2 texels of `format`.
fn attachment_length(format: AttachmentFormat) -> usize {
    4 * usize::try_from(format.bytes_per_texel()).expect("a texel width fits a host usize")
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
            format,
            width,
            height,
            load: LoadOp::Clear(attachment_clear(format)),
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
            stage_buffers: Vec::new(),
            vertex_entry: "vertex_main".to_owned(),
            fragment_entry: "fragment_main".to_owned(),
            color_formats: vec![format],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
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
                        // 2×2 texels of the format under test: exactly the
                        // extent the render attachment restates, so the
                        // declaration covers it whatever the compute kernel
                        // reads out of it.
                        length: attachment_length(format) as u64,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(
                            ATTACHMENT_WORD.repeat(attachment_length(format) / 4),
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
            size: attachment_length(format) as u64,
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

/// The render pass of a fixture's trace, for the tests that state one of its
/// fields (`research/docs/23` §3.3, v100).
fn render_pass_of(fixture: &mut Fixture) -> &mut RenderPassDescriptor {
    match fixture.trace.passes.last_mut() {
        Some(TracePass::Render(pass)) => pass,
        _ => panic!("the fixture's last pass is its render pass"),
    }
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
    let expected = expected_texels(fixture.format);
    eprintln!(
        "{:?} attachment readback: {} ({} bytes, first texel: {})",
        fixture.format,
        hex(&attachment),
        attachment.len(),
        hex(&attachment[..expected.len()])
    );
    eprintln!(
        "{:?} expected: [{}] x4, clear sentinel: [{}]",
        fixture.format,
        hex(&expected),
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

/// A viewport of the pass's own (`research/docs/23` §3.1, v100): the rect NDC
/// maps onto. The covering default is measured first in the same run, so the
/// two readings differ only in the viewport: the declared rect paints exactly
/// the texels it covers and leaves the rest holding the load op's bytes.
#[test]
fn a_declared_viewport_moves_and_shrinks_the_raster() {
    let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    // Control: the covering default every earlier increment published.
    let writebacks = submit_fixture(&fixture);
    let covering = attachment_readback(&fixture, &writebacks);
    let covered = expected_texels(fixture.format);
    assert_eq!(covering, covered.repeat(4));

    // The declared rect: origin (1, 1), one texel wide and one tall. Texel
    // (1, 1) is the only one whose centre lies inside the rect, so the other
    // three keep the clear's bytes — the frame the rect states rather than the
    // frame the attachment covers.
    render_pass_of(&mut fixture).viewport = [1, 1, 1, 1];
    let writebacks = submit_fixture(&fixture);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    let clear = clear_bytes(fixture.format);
    let expected = [
        clear.as_slice(),
        clear.as_slice(),
        clear.as_slice(),
        covered.as_slice(),
    ]
    .concat();
    eprintln!(
        "viewport [1, 1, 1, 1] readback: {} (clear sentinel per texel: {}, fragment texel: {})",
        hex(&attachment),
        hex(&clear),
        hex(&covered)
    );
    assert_eq!(attachment.len(), 16);
    assert_eq!(attachment, expected);
    assert_eq!(
        &attachment[12..16],
        covered.as_slice(),
        "the covered texel is the one whose centre lies inside the declared rect"
    );
}

/// The write mask is not part of the blend (`research/docs/23` §3.3, v100): an
/// attachment that does not blend still writes only the channels its mask
/// names, and the channels it leaves out keep the load op's own bytes.
#[test]
fn a_write_mask_keeps_the_unwritten_channels_in_their_load_bytes() {
    let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let pass = render_pass_of(&mut fixture);
    pass.blend = Some(RenderPassBlend {
        attachments: vec![BlendAttachment {
            enabled: false,
            source_rgb: BlendFactor::One,
            destination_rgb: BlendFactor::Zero,
            source_alpha: BlendFactor::One,
            destination_alpha: BlendFactor::Zero,
            operation: BlendOperation::Add,
            alpha_operation: BlendOperation::Add,
            write_mask: ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
        }],
    });
    let writebacks = submit_fixture(&fixture);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    let covered = expected_texels(fixture.format);
    let clear = clear_bytes(fixture.format);
    // Red and alpha are the draw's own channels; green and blue keep the
    // clear. One texel's worth of bytes, repeated over the four texels the
    // covering viewport paints.
    let mut texel = clear.clone();
    texel[0] = covered[0];
    texel[3] = covered[3];
    eprintln!(
        "write mask red|alpha readback: {} (cover-all texel would be {}, clear texel is {})",
        hex(&attachment),
        hex(&covered),
        hex(&clear)
    );
    assert_eq!(attachment, texel.repeat(4));
    assert_ne!(texel, covered);
}

/// A blend entry reads the attachment's own contents (`research/docs/23` §3.3,
/// v100): the colour pair adds the fragment output to the load's bytes while
/// the alpha pair keeps the destination. Two different equations in one entry
/// are exactly what the separate alpha operation and factor pair state, and the
/// load is a colour of its own so both halves are observable in the readback.
#[test]
fn a_blend_entry_mixes_the_fragment_output_with_the_load() {
    let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let pass = render_pass_of(&mut fixture);
    pass.color_attachments[0].load = LoadOp::Clear(ClearColor::new([0x10, 0x10, 0x10, 0x10]));
    pass.blend = Some(RenderPassBlend {
        attachments: vec![BlendAttachment {
            enabled: true,
            source_rgb: BlendFactor::One,
            destination_rgb: BlendFactor::One,
            source_alpha: BlendFactor::Zero,
            destination_alpha: BlendFactor::One,
            operation: BlendOperation::Add,
            alpha_operation: BlendOperation::Add,
            write_mask: ColorWriteMask::ALL,
        }],
    });
    let writebacks = submit_fixture(&fixture);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    // rgb: src + dst = 0x40 + 0x10, 0x80 + 0x10, 0xc0 + 0x10; alpha: dst.
    let expected = [0x50_u8, 0x90, 0xd0, 0x10].repeat(4);
    eprintln!(
        "blend rgb = source + destination, alpha = destination, load = 10 10 10 10: {}",
        hex(&attachment)
    );
    assert_eq!(attachment, expected);

    // The operation's other half: subtracting the load from the output lands
    // the same texels one subtraction down, which the sum above cannot be
    // mistaken for.
    render_pass_of(&mut fixture)
        .blend
        .as_mut()
        .expect("the entry is stated")
        .attachments[0]
        .operation = BlendOperation::Subtract;
    let writebacks = submit_fixture(&fixture);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    let expected = [0x30_u8, 0x70, 0xb0, 0x10].repeat(4);
    eprintln!(
        "blend rgb = source - destination, alpha = destination, load = 10 10 10 10: {}",
        hex(&attachment)
    );
    assert_eq!(attachment, expected);
}

/// The viewport is a rect inside the raster (`research/docs/23` §3.1, v100):
/// one that reaches outside is refused by name with the extent the rule
/// measured it against, rather than clipped to a frame neither rail declared.
#[test]
fn a_viewport_that_reaches_outside_the_attachment_is_refused_by_name() {
    let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    render_pass_of(&mut fixture).viewport = [1, 0, 2, 2];
    let refused = fixture
        .provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect_err("a rect that reaches outside the attachment is refused");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "viewport_extent_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.detail.as_deref(),
        Some(
            "viewport [origin_x, origin_y, width, height] [1, 0, 2, 2] is not inside the pass's \
             attachment extent [2, 2]"
        )
    );
}

/// The blend families this increment cannot execute are refused by name
/// (`research/docs/23` §3.3, v100): the blend constant, the fragment shader's
/// second colour output, and `SourceAlphaSaturated` in a destination slot.
#[test]
fn the_blend_families_this_increment_cannot_execute_are_refused_by_name() {
    let entry = |factor: BlendFactor| BlendAttachment {
        enabled: true,
        source_rgb: factor,
        destination_rgb: BlendFactor::One,
        source_alpha: BlendFactor::One,
        destination_alpha: BlendFactor::Zero,
        operation: BlendOperation::Add,
        alpha_operation: BlendOperation::Add,
        write_mask: ColorWriteMask::ALL,
    };
    let cases = [
        (
            entry(BlendFactor::BlendColor),
            "blend_constant_unsupported",
            ProviderErrorClass::Capability,
            "colour attachment 0's source rgb blend factor BlendColor reads the blend constant, \
             which this pass does not carry",
        ),
        (
            entry(BlendFactor::Source1Alpha),
            "blend_dual_source_unsupported",
            ProviderErrorClass::Capability,
            "colour attachment 0's source rgb blend factor Source1Alpha reads the fragment \
             shader's second colour output, which the reviewed stages do not declare",
        ),
    ];
    for (attachment, slug, class, detail) in cases {
        let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
            return;
        };
        render_pass_of(&mut fixture).blend = Some(RenderPassBlend {
            attachments: vec![attachment],
        });
        let refused = fixture
            .provider
            .capabilities()
            .validate_trace(fixture.trace.clone(), fixture.resources.clone())
            .expect_err("the blend family is refused");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, slug);
        assert_eq!(refused.class, class);
        assert_eq!(refused.detail.as_deref(), Some(detail));
    }

    // The destination slot keeps its own rule, and it is the caller's shape
    // rather than a device limit.
    let Some(mut fixture) = fixture(AttachmentFormat::Rgba8Unorm) else {
        return;
    };
    let mut attachment = entry(BlendFactor::One);
    attachment.destination_alpha = BlendFactor::SourceAlphaSaturated;
    render_pass_of(&mut fixture).blend = Some(RenderPassBlend {
        attachments: vec![attachment],
    });
    let refused = fixture
        .provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect_err("a destination-slot sourceAlphaSaturated is refused");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "blend_factor_slot_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    assert_eq!(
        refused.detail.as_deref(),
        Some(
            "colour attachment 0's destination alpha blend factor SourceAlphaSaturated is not one \
             the two APIs define in a destination slot"
        )
    );
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

        let texel = expected_texels(format);
        // The extent is the format's own: four texels of four bytes, or of
        // eight (`research/docs/23` §78). A rail that sized the readback from
        // one rail-wide constant would truncate this one to half its bytes.
        assert_eq!(attachment.len(), texel.len() * 4);
        assert_eq!(attachment, texel.repeat(4));
        assert!(
            !attachment
                .chunks_exact(texel.len())
                .any(|texel| texel == clear_bytes(format).as_slice()),
            "a surviving clear sentinel means the triangle did not cover every texel, so the \
             {format:?} readback would be the ClearColor: {}",
            hex(&attachment)
        );
        assert_ne!(clear_bytes(format), texel);
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
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: Vec::new(),
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
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: Vec::new(),
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

    let texel = expected_texels(fixture.format);
    assert_eq!(attachment.len(), texel.len() * 4);
    assert_eq!(attachment, texel.repeat(4));
    assert!(
        !attachment
            .chunks_exact(texel.len())
            .any(|texel| texel == PRESENT_SENTINEL),
        "a surviving present sentinel means the present never overwrote the target: {}",
        hex(&attachment)
    );
    assert_ne!(PRESENT_SENTINEL.as_slice(), texel.as_slice());

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
    // The declaration travels with the pipeline (`research/docs/23` §3.3,
    // v100), so a trace that binds a texture names a registration that states
    // one. This test registers that registration — the reviewed solid pair's
    // own two modules, under a declaration neither of them reads — and points
    // the pass at it, so what it measures is the *present rail*'s own refusal
    // rather than the contract's pair rules.
    let declaring = fixture
        .provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: vec![TextureBindingContract::sampled(
                    0,
                    TextureFormat::Rgba8Unorm,
                    SamplerPolicy {
                        filter: SamplerFilter::Nearest,
                        address: SamplerAddressMode::ClampToEdge,
                    },
                )],
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: SemanticDigest::new(
                "metal-smoke-fixture-v1",
                b"render_e2e_declaring_solid_stages".to_vec(),
            )
            .expect("digest"),
        })
        .expect("the declaring solid registration is well formed");
    let named = pass.pipeline;
    for pipeline in trace.pipelines.iter_mut() {
        if pipeline.pipeline_id == named {
            *pipeline = declaring.clone();
        }
    }
    pass.pipeline = declaring.pipeline_id;
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
    vertex_input_fixture_with_raster(vertex_bytes, index_bytes, load, None)
}

/// The same fixture drawn into a four-sample raster the attachment is seeded
/// into (`research/docs/23` §82, v82): the shape the multisampled `Load` route
/// exists for.
fn vertex_input_fixture_with_multisample(
    vertex_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    load: bool,
) -> Option<(VulkanComputeProvider, ComputeTrace, ResourceTableSnapshot)> {
    vertex_input_fixture_with_raster(vertex_bytes, index_bytes, load, Some(SampleCount::Four))
}

/// The body of the vertex-input fixtures: one pass over a 2×2 attachment whose
/// load op and raster sample count the caller selects.
fn vertex_input_fixture_with_raster(
    vertex_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    load: bool,
    samples: Option<SampleCount>,
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
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_buffer_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: quad_layout(),
                textures: Vec::new(),
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
    if let Some(sample_count) = samples {
        pass.multisample = Some(MultisampleState { sample_count });
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

/// The v82 tail channel: a multisampled attachment a pass opens with
/// `LoadOp::Load`.
///
/// A multisampled image cannot receive its previous bytes through
/// `vkCmdCopyBufferToImage` — the command's own valid usage holds its
/// destination to one sample (`VUID-vkCmdCopyBufferToImage-dstImage-07973`) —
/// so the rail seeds every sample of the image with the declaring view's own
/// texel through a `CLEAR`-opened render pass it records before the measured
/// one, which then opens the image with `LOAD` (`research/docs/23` §82). The
/// left-column triangle covers the left texel of the 2×2 attachment completely
/// and the right one not at all, so the resolve is deterministic on any device:
/// the covered texel carries the fragment output and the uncovered one the
/// seed. What this measures that the clearing fixture cannot: a rail that
/// dropped the declared bytes would land the driver's own undefined contents
/// in the uncovered texel, and a rail that refused the shape would land
/// nothing.
#[test]
fn a_multisampled_load_pass_resolves_the_seeded_samples() {
    let Some((provider, trace, resources)) =
        vertex_input_fixture_with_multisample(left_column_vertex_bytes(), quad_index_bytes(), true)
    else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let attachment = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("seeded multisample readback: {}", hex(&attachment));
    assert_eq!(attachment.len(), 16);
    let covered = attachment
        .chunks_exact(4)
        .filter(|texel| *texel == QUAD_TEXEL)
        .count();
    let seeded = attachment
        .chunks_exact(4)
        .filter(|texel| *texel == ATTACHMENT_WORD)
        .count();
    assert_eq!(
        (covered, seeded),
        (2, 2),
        "the covered texels resolve the fragment output and the uncovered ones the seed: {}",
        hex(&attachment)
    );

    // The counter-shape: the identical raster opened from a clear leaves the
    // clear sentinel where the draw missed, which is exactly the byte the seed
    // replaces. The two runs therefore differ in the texels the load is about.
    let Some((provider, trace, resources)) = vertex_input_fixture_with_multisample(
        left_column_vertex_bytes(),
        quad_index_bytes(),
        false,
    ) else {
        return;
    };
    let writebacks = submit_vertex_input(&provider, &trace, &resources);
    let cleared = readback(&writebacks, ATTACHMENT_VIEW);
    eprintln!("cleared multisample readback: {}", hex(&cleared));
    assert!(
        cleared.chunks_exact(4).any(|texel| texel == CLEAR_SENTINEL),
        "a clearing multisampled pass leaves its own colour where the draw missed: {}",
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
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                // The pair's declaration (`research/docs/23` §3.3, v100): the
                // sampling module reads the pass's one texture at binding 0,
                // with the state its MSL sibling's `constexpr sampler` carries.
                textures: vec![TextureBindingContract::sampled(
                    0,
                    TextureFormat::Rgba8Unorm,
                    SamplerPolicy {
                        filter: SamplerFilter::Nearest,
                        address: SamplerAddressMode::ClampToEdge,
                    },
                )],
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
    // The refusal the pair rules state is the contract's own
    // (`research/docs/23` §3.3, v100): the registration declared a texture the
    // pass does not bind, so no rail ever sees the unbound descriptor.
    assert_eq!(refused.slug, "trace_contract_invalid");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    assert!(refused
        .detail
        .as_deref()
        .is_some_and(|detail| detail.contains("declares texture binding 0")));
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

    // The same pass without its binding: the pair rules refuse it
    // (`research/docs/23` §3.3, v100) before the present rail's own arm runs,
    // because the registration declared the very texture this pass dropped.
    let mut unbound = presenting.clone();
    if let Some(TracePass::Render(pass)) = unbound.passes.last_mut() {
        pass.textures = Vec::new();
    }
    let refused = provider
        .capabilities()
        .validate_trace(unbound, resources)
        .expect_err("the sampling pair's declaration names a texture this pass no longer binds");
    eprintln!("sampling pair without a texture refused beside a present pass: {refused:?}");
    assert_eq!(refused.slug, "trace_contract_invalid");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    assert!(refused
        .detail
        .as_deref()
        .is_some_and(|detail| detail.contains("declares texture binding 0")));
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

/// The `R7` increment's own bytes (`research/docs/23` §76).
///
/// The seeding pass clears the attachment to the *declaring view's* own bytes —
/// [`ATTACHMENT_WORD`], the four bytes the fixture's compute pass reads out of
/// it — so a resident image and a trace-declared previous-contents upload carry
/// the same bytes and the two halves are comparable byte for byte. The two
/// other byte patterns the case measures with are the draw's own output
/// ([`expected_texels`]) and the milestone's clear sentinel, both of which
/// differ from it (and from each other), so "the resident load was skipped and
/// executed as a clear" cannot pass as "the frame stayed where it was".
const RESIDENT_CLEAR: [u8; 4] = ATTACHMENT_WORD;

/// One trace over the reviewed 2×2 quad pipeline with the attachment's load and
/// store replaced, and the vertex stream the caller names
/// (`research/docs/23` §76, R7).
///
/// The three halves of the resident case are the same trace shape with those
/// two decisions changed: the seeding pass keeps its raster resident, the pass
/// that follows loads the provider's image, and the baseline loads the trace's
/// own declared bytes.
fn resident_trace(
    base: &ComputeTrace,
    load: LoadOp,
    store: StoreOp,
    vertex_bytes: Vec<u8>,
) -> ComputeTrace {
    let mut trace = base.clone();
    let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    pass.color_attachments[0].load = load;
    pass.color_attachments[0].store = store;
    pass.vertex_buffers[0].source = BufferSource::OwnedBytes(vertex_bytes.clone());
    pass.vertex_buffers[0].length = u64::try_from(vertex_bytes.len()).expect("stream length");
    // The reviewed stream's six indices select four vertices, whatever shape
    // those four vertices describe.
    pass.vertices = 6;
    trace
}

/// Point one trace at another attachment identity, in both places the contract
/// names it: the declaring compute pass's read-only view and the render pass's
/// attachment (`validate_serial_buffer_reuse` resolves the attachment against
/// the views the trace declares).
fn retarget_resident_trace(trace: &mut ComputeTrace, view: ViewId, allocation: AllocationId) {
    let Some(TracePass::Compute(pass)) = trace.passes.first_mut() else {
        panic!("the fixture opens with the declaring compute pass");
    };
    pass.buffers[0].view_id = view;
    pass.buffers[0].allocation_id = allocation;
    let Some(TracePass::Render(render)) = trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    render.color_attachments[0].view_id = view;
    render.color_attachments[0].allocation_id = allocation;
}

/// The resource table one retargeted trace needs: the attachment's own
/// allocation beside the fixture's scratch allocation.
fn resident_resources(
    epoch: metal_api_core::provider::DeviceEpoch,
    allocation: AllocationId,
    size: u64,
) -> ResourceTableSnapshot {
    let mut resources = ResourceTableSnapshot::new();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: allocation,
            owner_epoch: epoch,
            size,
        })
        .expect("the retargeted attachment's allocation");
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: SCRATCH_ALLOCATION,
            owner_epoch: epoch,
            size: 8,
        })
        .expect("the fixture's scratch allocation");
    resources
}

/// Submit one trace the provider is expected to refuse *after* its shape was
/// admitted, returning that refusal (`research/docs/23` §76, R7).
fn resident_refusal(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> ProviderError {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the trace shape is admitted");
    provider
        .submit(admitted)
        .expect_err("the provider refuses the submission")
}

/// The resident chain end to end (`research/docs/23` §76, R7): two submissions
/// on one provider, and the second one's bytes come from the image the first
/// left behind — no guest writeback, no re-declaration.
///
/// The comparison is against the closest shape that already existed: one
/// submission whose loading attachment takes its previous bytes from the
/// trace's own view declaration. The two have to agree byte for byte, which is
/// what makes the claim falsifiable — a rail that skipped the load and cleared
/// instead lands the sentinel in the two texels this draw does not cover, a
/// rail that re-used a stale image lands the wrong four bytes, and a rail that
/// published a writeback for the resident store would show a second attachment
/// writeback where the trace declared none.
#[test]
fn a_resident_store_keeps_the_frame_a_later_pass_loads() {
    // The baseline first, because it is the comparison every seed shape below
    // is measured against: one submission whose loading attachment takes its
    // previous bytes from the trace's own declaration.
    let Some((baseline_provider, baseline, baseline_resources)) =
        vertex_input_fixture_with_load(left_column_vertex_bytes(), quad_index_bytes(), true)
    else {
        return;
    };
    let baseline_bytes = readback(
        &submit_presenting_trace(&baseline_provider, &baseline, &baseline_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("trace-owned baseline readback: {}", hex(&baseline_bytes));

    // The expectation itself: the drawn left column beside the seeded right
    // one, both of which differ from the clear sentinel this second pass never
    // writes.
    let drawn = expected_texels(AttachmentFormat::Rgba8Unorm);
    let mut expected = Vec::with_capacity(16);
    for _row in 0..2 {
        expected.extend_from_slice(&drawn);
        expected.extend_from_slice(&RESIDENT_CLEAR);
    }
    assert_eq!(
        baseline_bytes, expected,
        "the baseline the chain is compared against is the drawn column beside the declared bytes"
    );

    // Both ways a resident image is defined: a clear the pass keeps resident,
    // and the trace's own bytes uploaded into the provider's image. They have to
    // produce the same chain, because what the second submission reads is in
    // both cases the provider's image.
    for seed_load in [LoadOp::Clear(ClearColor::new(RESIDENT_CLEAR)), LoadOp::Load] {
        let Some((provider, base, resources)) =
            vertex_input_fixture_with_load(collapsed_vertex_bytes(), quad_index_bytes(), false)
        else {
            return;
        };

        // 1. The seeding pass: define the attachment's bytes (a clear, or the
        //    declaring compute view's own bytes uploaded), draw nothing, and
        //    keep the raster in the provider's image. It publishes no writeback
        //    at all — the bytes stay where the provider can load them
        //    (`StoreOp::Resident`).
        let seed = resident_trace(
            &base,
            seed_load,
            StoreOp::Resident,
            collapsed_vertex_bytes(),
        );
        let seed_writebacks = submit_presenting_trace(&provider, &seed, &resources);
        assert_eq!(
            seed_writebacks
                .iter()
                .filter(|(view, _)| *view == ATTACHMENT_VIEW)
                .count(),
            0,
            "a resident store keeps the frame in the provider's image and publishes no writeback: \
             {seed_writebacks:?} (seed load {seed_load:?})"
        );
        assert_eq!(
            provider.resident_target_count(),
            1,
            "the seeding pass leaves exactly the identity it named resident"
        );
        assert_eq!(provider.resident_target_evictions(), 0);
        assert!(provider.resident_target_is_live(ATTACHMENT_ALLOCATION, ATTACHMENT_VIEW));

        // 2. The second submission loads the provider's own bytes and overwrites
        //    the left column of the 2×2 raster, publishing the mixed result.
        let chained = resident_trace(
            &base,
            LoadOp::Resident,
            StoreOp::Store,
            left_column_vertex_bytes(),
        );
        let chained_writebacks = submit_presenting_trace(&provider, &chained, &resources);
        let chained_bytes = readback(&chained_writebacks, ATTACHMENT_VIEW);
        eprintln!(
            "resident chain readback after a {seed_load:?} seed: {} ({} bytes)",
            hex(&chained_bytes),
            chained_bytes.len()
        );
        assert_eq!(
            chained_bytes, baseline_bytes,
            "the resident load lands the same bytes the trace-declared previous contents do"
        );
        assert_eq!(
            chained_bytes, expected,
            "the uncovered half keeps the resident image's own bytes"
        );
        assert!(
            !chained_bytes
                .chunks_exact(4)
                .any(|texel| texel == clear_bytes(AttachmentFormat::Rgba8Unorm)),
            "no texel keeps the clear sentinel: a skipped resident load cannot pass as a kept frame"
        );
        assert_eq!(
            provider.resident_target_count(),
            1,
            "the chain reuses one image"
        );
        assert_eq!(provider.resident_target_evictions(), 0);
    }
}

/// The resident registry's budget and eviction order
/// (`research/docs/23` §76, R7), the R4a rule generalised to the render rail:
/// the provider bounds the identities it keeps, the victim is the least
/// recently used one, and a load of an identity the budget evicted is refused
/// by name instead of being served the bytes the image used to hold.
#[test]
fn resident_targets_are_bounded_and_evicted_least_recently_used() {
    let Some((provider, base, _resources)) =
        vertex_input_fixture_with_load(quad_vertex_bytes(), quad_index_bytes(), false)
    else {
        return;
    };
    let budget = RESIDENT_TARGET_BUDGET;
    assert!(budget >= 2, "the LRU case needs more than one identity");

    let identities = (0..budget + 2)
        .map(|index| {
            (
                ViewId::new(9100 + index as u64),
                AllocationId::new(9600 + index as u64),
            )
        })
        .collect::<Vec<_>>();
    for (view, allocation) in &identities {
        let mut trace = resident_trace(
            &base,
            LoadOp::Clear(ClearColor::new(RESIDENT_CLEAR)),
            StoreOp::Resident,
            quad_vertex_bytes(),
        );
        retarget_resident_trace(&mut trace, *view, *allocation);
        let table = resident_resources(provider.device_epoch(), *allocation, 16);
        let writebacks = submit_presenting_trace(&provider, &trace, &table);
        assert_eq!(
            writebacks
                .iter()
                .filter(|(writeback, _)| *writeback == *view)
                .count(),
            0,
            "every seeding pass keeps its bytes resident"
        );
    }

    let over_budget = (identities.len() - budget) as u64;
    assert_eq!(
        provider.resident_target_count(),
        budget,
        "the registry keeps at most the budget's worth of identities"
    );
    assert_eq!(
        provider.resident_target_evictions(),
        over_budget,
        "one eviction per identity beyond the budget"
    );
    eprintln!(
        "resident registry after {} identities: count={} budget={} evictions={}",
        identities.len(),
        provider.resident_target_count(),
        budget,
        provider.resident_target_evictions()
    );

    // The two oldest identities were the victims: loading one of them states
    // the rule by name, and the registry did not grow to serve it.
    for (index, (view, allocation)) in identities.iter().take(over_budget as usize).enumerate() {
        assert!(
            !provider.resident_target_is_live(*allocation, *view),
            "identity {index} was the least recently used entry and is gone"
        );
        let mut trace = resident_trace(
            &base,
            LoadOp::Resident,
            StoreOp::Store,
            left_column_vertex_bytes(),
        );
        retarget_resident_trace(&mut trace, *view, *allocation);
        let table = resident_resources(provider.device_epoch(), *allocation, 16);
        let refused = resident_refusal(&provider, &trace, &table);
        eprintln!("evicted resident load refused: {refused:?}");
        assert_eq!(refused.slug, "resident_target_evicted");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("retired_by"),
            Some(&FieldValue::Text("budget".to_owned()))
        );
        assert_eq!(
            refused.fields.get("view"),
            Some(&FieldValue::Unsigned(view.get()))
        );
    }
    assert_eq!(
        provider.resident_target_count(),
        budget,
        "a refused load does not re-create the identity it could not read"
    );
    assert_eq!(provider.resident_target_evictions(), over_budget);
}

/// The normal-path retirement surface that pairs with the budget: a resident
/// target reserved for an allocation is retired when the staged lease that
/// allocation was imported under is released, and a later resident load names
/// that release instead of reading an image whose backing the owner took back
/// (`research/docs/23` §76, R7).
#[test]
fn releasing_a_staged_lease_retires_the_resident_target_of_its_allocation() {
    let Some((provider, base, resources)) =
        vertex_input_fixture_with_load(quad_vertex_bytes(), quad_index_bytes(), false)
    else {
        return;
    };
    let lease = LeaseId::new(12);
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
    let mut trace = resident_trace(
        &base,
        LoadOp::Clear(ClearColor::new(RESIDENT_CLEAR)),
        StoreOp::Resident,
        quad_vertex_bytes(),
    );
    let Some(TracePass::Compute(pass)) = trace.passes.first_mut() else {
        panic!("the fixture opens with the declaring compute pass");
    };
    pass.buffers[0].source = BufferSource::StagedLease(lease);
    let mut table = resources.clone();
    table
        .insert_lease(reservation)
        .expect("the reservation covers the attachment view");

    let _ = submit_presenting_trace(&provider, &trace, &table);
    assert_eq!(
        provider.resident_target_count(),
        1,
        "the resident target of a leased allocation stays alive while the lease does"
    );

    provider
        .release_staged_lease(lease)
        .expect("the staged lease is released");
    assert_eq!(
        provider.resident_target_count(),
        0,
        "releasing the lease retires the allocation's resident target in the same call"
    );
    assert_eq!(
        provider.resident_target_evictions(),
        1,
        "a lease-driven retirement is the same observable as a budget eviction"
    );
    assert!(!provider.resident_target_is_live(ATTACHMENT_ALLOCATION, ATTACHMENT_VIEW));

    let load = resident_trace(
        &base,
        LoadOp::Resident,
        StoreOp::Store,
        left_column_vertex_bytes(),
    );
    let refused = resident_refusal(&provider, &load, &resources);
    eprintln!("released resident load refused: {refused:?}");
    assert_eq!(refused.slug, "resident_target_released");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("retired_by"),
        Some(&FieldValue::Text("lease_released".to_owned()))
    );
}

/// One identity names one image, so a pass that declares a different shape for
/// it is refused by name rather than rendered into an image of the wrong
/// extent (`research/docs/23` §76, R7).
#[test]
fn a_resident_load_of_a_different_shape_is_refused_by_name() {
    let Some((provider, base, resources)) =
        vertex_input_fixture_with_load(quad_vertex_bytes(), quad_index_bytes(), false)
    else {
        return;
    };
    let seed = resident_trace(
        &base,
        LoadOp::Clear(ClearColor::new(RESIDENT_CLEAR)),
        StoreOp::Resident,
        quad_vertex_bytes(),
    );
    let _ = submit_presenting_trace(&provider, &seed, &resources);
    assert_eq!(provider.resident_target_count(), 1);

    // The same identity, declared as a 4×4 raster: the shape the registry holds
    // is 2×2, so the two cannot be one image.
    let mut trace = resident_trace(&base, LoadOp::Resident, StoreOp::Store, quad_vertex_bytes());
    let Some(TracePass::Compute(pass)) = trace.passes.first_mut() else {
        panic!("the fixture opens with the declaring compute pass");
    };
    pass.buffers[0].length = 64;
    pass.buffers[0].source = BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(16));
    let Some(TracePass::Render(render)) = trace.passes.last_mut() else {
        panic!("the fixture ends in a render pass");
    };
    render.color_attachments[0].width = 4;
    render.color_attachments[0].height = 4;
    render.viewport = [0, 0, 4, 4];
    let mut table = ResourceTableSnapshot::new();
    table
        .insert_allocation(AllocationRecord {
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 64,
        })
        .expect("the widened attachment allocation");
    table
        .insert_allocation(AllocationRecord {
            allocation_id: SCRATCH_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 8,
        })
        .expect("the fixture's scratch allocation");

    let refused = resident_refusal(&provider, &trace, &table);
    eprintln!("reshaped resident load refused: {refused:?}");
    assert_eq!(refused.slug, "resident_target_shape_changed");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(refused.fields.get("width"), Some(&FieldValue::Unsigned(4)));
    assert_eq!(refused.fields.get("height"), Some(&FieldValue::Unsigned(4)));
    assert_eq!(
        refused.fields.get("expected_width"),
        Some(&FieldValue::Unsigned(2))
    );
    assert_eq!(
        refused.fields.get("expected_height"),
        Some(&FieldValue::Unsigned(2))
    );
    assert_eq!(
        provider.resident_target_count(),
        1,
        "the resident image the pass could not use stays as it was"
    );
}

/// A device-loss rebuild retires every resident image of the dead device, and a
/// later resident load is refused as stale instead of being served bytes from a
/// device that no longer exists (`research/docs/23` §76, R7).
#[test]
fn a_device_loss_rebuild_retires_resident_targets() {
    let Some(executor) = executor() else {
        return;
    };
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
    let compile_compute = |provider: &VulkanComputeProvider| {
        provider
            .compile_pipeline(
                &function,
                fixture_digest(b"render_e2e_resident_loss_compute"),
            )
            .expect("the compute pipeline registers")
    };
    let register_render = |provider: &VulkanComputeProvider| {
        provider
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
                logical_digest: fixture_digest(b"render_e2e_resident_loss_stages"),
            })
            .expect("the render pipeline registers")
    };
    let allocations = |provider: &VulkanComputeProvider| {
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
        resources
    };

    // 1. Seed one resident identity on the original device.
    let compute = compile_compute(&provider);
    let render = register_render(&provider);
    let seed = resident_loss_trace(
        provider.device_epoch(),
        &compute,
        &render,
        quad_pass(
            render.pipeline_id,
            LoadOp::Clear(ClearColor::new(RESIDENT_CLEAR)),
            StoreOp::Resident,
            quad_vertex_bytes(),
        ),
    );
    let resources = allocations(&provider);
    let _ = submit_presenting_trace(&provider, &seed, &resources);
    assert_eq!(
        provider.resident_target_count(),
        1,
        "the seeding pass leaves one resident identity"
    );

    // 2. The substituted driver answer fails the submission through the real
    //    loss path, so the provider's own lifecycle reaches `DeviceLost`.
    executor.inject_driver_device_loss_for_test(DeviceLossPoint::Submit);
    let error = provider
        .submit(
            provider
                .capabilities()
                .validate_trace(seed.clone(), resources.clone())
                .expect("the trace shape is admitted"),
        )
        .expect_err("the substituted driver answer refuses the submission");
    assert_eq!(error.class, ProviderErrorClass::DeviceLost);

    // 3. The rebuild retires every image of the dead device with the device.
    provider
        .rebuild_after_device_loss()
        .expect("the lost provider rebuilds in place");
    assert_eq!(
        provider.resident_target_count(),
        0,
        "every image of the dead device is gone with it"
    );
    assert!(!provider.resident_target_is_live(ATTACHMENT_ALLOCATION, ATTACHMENT_VIEW));
    assert_eq!(
        provider.resident_target_evictions(),
        0,
        "the epoch advance is observable through the epoch, not through the budget counter"
    );

    // 4. Re-register on the fresh device and ask for the image again: the
    //    identity is refused as stale rather than served from the dead epoch.
    let compute = compile_compute(&provider);
    let render = register_render(&provider);
    let load = resident_loss_trace(
        provider.device_epoch(),
        &compute,
        &render,
        quad_pass(
            render.pipeline_id,
            LoadOp::Resident,
            StoreOp::Store,
            left_column_vertex_bytes(),
        ),
    );
    let refused = resident_refusal(&provider, &load, &allocations(&provider));
    eprintln!("stale resident load refused: {refused:?}");
    assert_eq!(refused.slug, "resident_target_stale");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("retired_by"),
        Some(&FieldValue::Text("epoch_advance".to_owned()))
    );
}

/// The declaring compute pass plus the caller's render pass, over the reviewed
/// compute kernel and one registered render pipeline, on the epoch's own
/// registrations (`research/docs/23` §76, R7).
fn resident_loss_trace(
    epoch: metal_api_core::provider::DeviceEpoch,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    pass: RenderPassDescriptor,
) -> ComputeTrace {
    ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: epoch,
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
    }
}

/// One 2×2 render pass over the reviewed quad pipeline, with the attachment's
/// load/store and the vertex stream the caller names
/// (`research/docs/23` §76, R7).
fn quad_pass(
    pipeline: PipelineId,
    load: LoadOp,
    store: StoreOp,
    vertex_bytes: Vec<u8>,
) -> RenderPassDescriptor {
    let mut pass = render_pass(pipeline, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.color_attachments[0].load = load;
    pass.color_attachments[0].store = store;
    pass.vertices = 6;
    pass.vertex_buffers = vec![BufferView {
        view_id: VERTEX_VIEW,
        metal_binding: 0,
        allocation_id: VERTEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(vertex_bytes.len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vertex_bytes),
    }];
    pass.indices = Some(IndexBufferBinding {
        view: BufferView {
            view_id: INDEX_VIEW,
            metal_binding: 0,
            allocation_id: INDEX_ALLOCATION,
            offset: 0,
            length: 12,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(quad_index_bytes()),
        },
        format: IndexFormat::Uint16,
    });
    pass
}

/// The reviewed stage-buffer pair (`research/docs/23` §3.3, v83): the vertex
/// stage reads its three positions from `DescriptorSet 1 / Binding 0` and the
/// fragment stage reads its colour from `DescriptorSet 2 / Binding 0`, so both
/// stages' bytes move the pass's observable output.
const STAGE_BUFFER_POSITIONS_VERT_SPV: &[u8] =
    include_bytes!("../src/render_spv/stage_buffer_positions.vert.spv");
const STAGE_BUFFER_TINT_FRAG_SPV: &[u8] =
    include_bytes!("../src/render_spv/stage_buffer_tint.frag.spv");

const STAGE_BUFFER_POSITION_VIEW: ViewId = ViewId::new(713);
const STAGE_BUFFER_POSITION_ALLOCATION: AllocationId = AllocationId::new(813);
const STAGE_BUFFER_TINT_VIEW: ViewId = ViewId::new(714);
const STAGE_BUFFER_TINT_ALLOCATION: AllocationId = AllocationId::new(814);

/// The three Metal-NDC `vec2` positions the reviewed vertex stage reads: a
/// triangle whose vertices are `(-0.9, 0.9)`, `(0.0, 0.9)` and `(-0.9, 0.0)`.
/// The module's y flip lands them in Vulkan NDC as `(-0.9, -0.9)`,
/// `(0.0, -0.9)`, `(-0.9, 0.0)` — an area inside the top-left texel of a 2×2
/// render area that contains that texel's centre, so exactly one texel carries
/// the fragment stage's bytes and the other three keep the clear sentinel.
fn stage_buffer_positions() -> Vec<u8> {
    [-0.9_f32, 0.9, 0.0, 0.9, -0.9, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The same buffer's coverage control: the milestone's full-screen triangle in
/// Metal NDC (`(-1, -1)`, `(3, -1)`, `(-1, 3)`), which covers every texel of
/// the 2×2 render area. A rail that read no bytes at all — or a fixture whose
/// positions were silently replaced by the milestone geometry — would fill all
/// four texels, which is what this control states.
fn stage_buffer_full_cover_positions() -> Vec<u8> {
    [-1.0_f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The one `vec4` the reviewed fragment stage reads: the same `(64/255,
/// 128/255, 192/255, 1)` constants the solid stages store, so the covered
/// texel's readback is `40 80 c0 ff` — the stage buffer's own payload with the
/// format's quantisation an identity, rather than a new expectation.
fn stage_buffer_tint() -> Vec<u8> {
    [64.0_f32 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The one view a pass binds at `stage`/`index`: the trace's own bytes, the
/// shape every render input's declaring arm has.
fn stage_buffer_view(
    stage: RenderPipelineStage,
    index: u32,
    view_id: ViewId,
    allocation_id: AllocationId,
    bytes: &[u8],
) -> StageBufferView {
    StageBufferView {
        stage,
        view: BufferView {
            view_id,
            metal_binding: index,
            allocation_id,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("buffer length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes.to_vec()),
        },
    }
}

/// The two slots the reviewed pair declares: the vertex stage's 24-byte
/// positions and the fragment stage's 16-byte tint, each read once with a
/// static footprint.
fn stage_buffer_declarations() -> Vec<StageBufferBinding> {
    vec![
        StageBufferBinding {
            stage: RenderPipelineStage::Vertex,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 24 },
        },
        StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 16 },
        },
    ]
}

/// The stage-buffer fixture's registrations: the declaring pass's compute
/// kernel, the reviewed pair, and one milestone registration that declares the
/// same slots without a module that reads them — the pairing the rail refuses
/// by name.
fn stage_buffer_fixture() -> Option<(
    VulkanComputeProvider,
    CompiledComputePipeline,
    CompiledComputePipeline,
    CompiledComputePipeline,
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
        .compile_pipeline(&function, digest(b"render_e2e_stage_buffer_compute"))
        .expect("the compute pipeline registers");
    let reviewed = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "stage_buffer_positions_main".to_owned(),
                fragment_entry: "stage_buffer_tint_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: stage_buffer_declarations(),
                textures: Vec::new(),
            },
            vertex_spirv: STAGE_BUFFER_POSITIONS_VERT_SPV.to_vec(),
            fragment_spirv: STAGE_BUFFER_TINT_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_stage_buffer_stages"),
        })
        .expect("the stage-buffer pair registers");
    let other = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: stage_buffer_declarations(),
                textures: Vec::new(),
            },
            vertex_spirv: FULL_SCREEN_TRIANGLE_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_stage_buffer_other"),
        })
        .expect("the milestone pair registers beside the stage-buffer declaration");
    Some((provider, compute, reviewed, other))
}

/// One render-bearing trace whose pass binds the two stage buffers the
/// pipeline declares. The declaring compute pass is the sampled-texture
/// fixture's shape: it is what puts the attachment view in the trace's
/// resource namespace, and the render pass draws three vertices through the
/// reviewed pair.
fn stage_buffer_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
    tint: &[u8],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = vec![
        stage_buffer_view(
            RenderPipelineStage::Vertex,
            0,
            STAGE_BUFFER_POSITION_VIEW,
            STAGE_BUFFER_POSITION_ALLOCATION,
            positions,
        ),
        stage_buffer_view(
            RenderPipelineStage::Fragment,
            0,
            STAGE_BUFFER_TINT_VIEW,
            STAGE_BUFFER_TINT_ALLOCATION,
            tint,
        ),
    ];
    stage_buffer_trace_with_pass(provider, compute, render, pass)
}

/// The same trace with a caller-built pass: the shape the refusal controls
/// need, where the pass deliberately binds fewer (or other) stage buffers than
/// the pipeline declares.
fn stage_buffer_trace_with_pass(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    pass: RenderPassDescriptor,
) -> (ComputeTrace, ResourceTableSnapshot) {
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
                        // 2×2 texels of four bytes: exactly the extent the
                        // render attachment restates.
                        length: attachment_length(AttachmentFormat::Rgba8Unorm) as u64,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0; 16]),
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
    for (allocation, size) in [(ATTACHMENT_ALLOCATION, 16), (SCRATCH_ALLOCATION, 8)] {
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

/// Submit one stage-buffer trace and return the attachment's readback.
fn stage_buffer_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
    tint: &[u8],
) -> Vec<u8> {
    let (trace, resources) = stage_buffer_trace(provider, compute, render, positions, tint);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the stage-buffer trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The whole chain for the stage buffers: a 2×2 attachment the two stages read
/// their own bytes from. The falsification is the point — the fragment
/// stage's payload is the covered texel, the vertex stage's positions decide
/// *which* texel is covered, and the clear sentinel survives where the draw
/// does not reach.
#[test]
fn a_render_pass_reads_its_stage_buffers_and_lands_their_bytes() {
    let Some((provider, compute, reviewed, other)) = stage_buffer_fixture() else {
        return;
    };
    let positions = stage_buffer_positions();
    let tint = stage_buffer_tint();
    let attachment = stage_buffer_readback(&provider, &compute, &reviewed, &positions, &tint);
    eprintln!("stage-buffer attachment readback: {}", hex(&attachment));
    eprintln!("positions: {}", hex(&positions));
    eprintln!("tint: {}", hex(&tint));
    assert_eq!(attachment.len(), 16);
    assert_eq!(
        &attachment[..4],
        &[0x40, 0x80, 0xc0, 0xff],
        "the covered texel is the fragment stage buffer's own payload"
    );
    assert!(
        attachment[4..]
            .chunks_exact(4)
            .all(|texel| texel == CLEAR_SENTINEL),
        "the positions buffer's triangle covers exactly one texel: {}",
        hex(&attachment)
    );

    // Control one: the same pass with another fragment payload lands that
    // payload's bytes in the covered texel, and nothing else moves.
    let other_tint: Vec<u8> = [0.0_f32, 1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let green = stage_buffer_readback(&provider, &compute, &reviewed, &positions, &other_tint);
    eprintln!("green readback: {}", hex(&green));
    assert_eq!(&green[..4], &[0x00, 0xff, 0x00, 0xff]);
    assert!(green[4..]
        .chunks_exact(4)
        .all(|texel| texel == CLEAR_SENTINEL));

    // Control two: the same fragment payload with the full-cover positions
    // fills all four texels, so the vertex stage's bytes — not the module's
    // own geometry — are what selected the covered texel above.
    let full = stage_buffer_readback(
        &provider,
        &compute,
        &reviewed,
        &stage_buffer_full_cover_positions(),
        &tint,
    );
    eprintln!("full-cover readback: {}", hex(&full));
    assert_eq!(full, [0x40, 0x80, 0xc0, 0xff].repeat(4));

    // Control three: the same bindings under a pipeline whose modules read no
    // stage buffer are refused by name instead of executed with the bindings
    // dropped.
    let (trace, resources) = stage_buffer_trace(&provider, &compute, &other, &positions, &tint);
    let refused = match provider.capabilities().validate_trace(trace, resources) {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("the milestone pair reads no stage buffer"),
        Err(error) => error,
    };
    eprintln!("stage buffers under the milestone pair refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_stage_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

/// A registration whose contract declares the stage-buffer pair's modules but
/// no slots is refused at execution by name: the reviewed pair reads its own
/// bindings, so a pass that binds none would draw from descriptors nobody
/// filled (`research/docs/23` §3.3, v83).
#[test]
fn the_reviewed_stage_buffer_pair_requires_its_two_bindings() {
    let Some(executor) = executor() else {
        return;
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let unbound = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "stage_buffer_positions_main".to_owned(),
                fragment_entry: "stage_buffer_tint_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: Vec::new(),
                textures: Vec::new(),
            },
            vertex_spirv: STAGE_BUFFER_POSITIONS_VERT_SPV.to_vec(),
            fragment_spirv: STAGE_BUFFER_TINT_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_stage_buffer_unbound"),
        })
        .expect("the pair registers: the contract declares no slot");
    let function = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>)
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(
            &function,
            digest(b"render_e2e_stage_buffer_unbound_compute"),
        )
        .expect("the compute pipeline registers");
    let pass = render_pass(unbound.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    let (trace, resources) = stage_buffer_trace_with_pass(&provider, &compute, &unbound, pass);
    let refused = match provider.capabilities().validate_trace(trace, resources) {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("the reviewed pair needs its two bindings"),
        Err(error) => error,
    };
    eprintln!("unbound stage-buffer pair refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_binding_required");
}

/// The translated stage-buffer fixture (`research/docs/23` §3.3, v84): the
/// milestone's own two AIR stages, with the fragment half reading its colour
/// from its `[[buffer(0)]]` argument instead of an immediate. Both stages are
/// translations, so the descriptor slot is the one the fragment reflection
/// names — the translator's default layout puts it at `DescriptorSet 0 /
/// Binding 0` — rather than one of the reviewed pair's fixed sets.
const TRANSLATED_STAGE_BUFFER_VERTEX_AIR: &str =
    include_str!("fixtures/render_offscreen_2x2.vert.ll");
const TRANSLATED_STAGE_BUFFER_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_stage_buffer_tint.frag.ll");
const TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const TRANSLATED_STAGE_BUFFER_FRAGMENT_ENTRY: &str = "render_stage_buffer_rgba8";

const TRANSLATED_STAGE_BUFFER_TINT_VIEW: ViewId = ViewId::new(715);
const TRANSLATED_STAGE_BUFFER_TINT_ALLOCATION: AllocationId = AllocationId::new(815);

/// The translated stage-buffer fixture's registrations: the declaring pass's
/// compute kernel and the translated pair whose fragment stage reads the slot
/// the contract declares.
///
/// `set` is the descriptor set the fragment stage is translated into: the
/// translator's default (set 0, every Metal resource in one set) or the
/// fragment set of the reviewed pair's arrangement (set 2, the vertex stage's
/// buffers would be set 1). The rail binds whichever set the reflection names
/// (`research/docs/23` §3.3, v84).
fn translated_stage_buffer_fixture(
    set: u32,
) -> Option<(
    VulkanComputeProvider,
    CompiledComputePipeline,
    CompiledComputePipeline,
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
        .compile_pipeline(
            &function,
            digest(b"render_e2e_translated_stage_buffer_compute"),
        )
        .expect("the compute pipeline registers");
    let vertex_library = device
        .new_library_with_air(TRANSLATED_STAGE_BUFFER_VERTEX_AIR)
        .expect("the translated vertex fixture loads");
    let vertex_function = vertex_library
        .function(TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY)
        .expect("the translated vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &vertex_function)
        .expect("the vertex stage translates");
    let fragment_library = device
        .new_library_with_air(TRANSLATED_STAGE_BUFFER_FRAGMENT_AIR)
        .expect("the translated fragment fixture loads");
    let fragment_function = fragment_library
        .function(TRANSLATED_STAGE_BUFFER_FRAGMENT_ENTRY)
        .expect("the translated fragment entry exists");
    let fragment = TranslatedRenderStage::translate_with_policy_and_layout(
        RenderStage::Fragment,
        &fragment_function,
        executor.spirv_feature_policy(),
        DescriptorLayout {
            set,
            ..DescriptorLayout::default()
        },
    )
    .expect("the fragment stage translates");
    let binding = fragment
        .reflection()
        .bindings
        .iter()
        .find(|binding| binding.metal_index == 0)
        .expect("the fixture declares one Metal buffer");
    eprintln!(
        "translated stage-buffer reflection: descriptor {:?} access {:?} footprint {:?}",
        binding.descriptor, binding.access, binding.footprint
    );
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
                fragment_entry: TRANSLATED_STAGE_BUFFER_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: vec![StageBufferBinding {
                    stage: RenderPipelineStage::Fragment,
                    index: 0,
                    access: BufferAccess::Read,
                    // One `float4` load is the whole reach of the fixture.
                    footprint: FootprintProof::Static { max_bytes: 16 },
                }],
                textures: Vec::new(),
            },
            vertex,
            fragment,
            logical_digest: digest(b"render_e2e_translated_stage_buffer"),
        })
        .expect("the translated stage-buffer pair registers");
    Some((provider, compute, render))
}

/// One render-bearing trace whose pass binds the translated fragment stage's
/// stage buffer: the same declaring-compute + render shape the reviewed
/// stage-buffer trace has, with the slot bound at the fragment stage alone.
fn translated_stage_buffer_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    tint: &[u8],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = vec![stage_buffer_view(
        RenderPipelineStage::Fragment,
        0,
        TRANSLATED_STAGE_BUFFER_TINT_VIEW,
        TRANSLATED_STAGE_BUFFER_TINT_ALLOCATION,
        tint,
    )];
    stage_buffer_trace_with_pass(provider, compute, render, pass)
}

/// Submit one translated stage-buffer trace and return the attachment's
/// readback.
fn translated_stage_buffer_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    tint: &[u8],
) -> Vec<u8> {
    let (trace, resources) = translated_stage_buffer_trace(provider, compute, render, tint);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the translated stage-buffer trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The translated arm of the stage-buffer face (`research/docs/23` §3.3,
/// v84): the module's own reflection names the descriptor slot, the contract
/// declares the bytes, and the pass's view is what the attachment lands. The
/// falsification is the buffer's own payload — the full-screen triangle covers
/// every texel of the 2×2 area, so all four readback texels are the stage
/// buffer's bytes through the format's quantisation, and swapping the payload
/// swaps the frame. The same stage is executed twice: once translated into the
/// translator's default set 0 and once into set 2 — the reviewed pair's
/// fragment set — so the slot the rail binds is the module's own in both
/// arrangements and the two land the same bytes.
#[test]
fn a_translated_stage_reads_its_stage_buffer_and_lands_its_bytes() {
    let Some((provider, compute, render)) = translated_stage_buffer_fixture(0) else {
        return;
    };
    let tint = stage_buffer_tint();
    let attachment = translated_stage_buffer_readback(&provider, &compute, &render, &tint);
    eprintln!(
        "translated stage-buffer attachment readback: {}",
        hex(&attachment)
    );
    eprintln!("stage buffer payload: {}", hex(&tint));
    assert_eq!(attachment.len(), 16);
    assert_eq!(
        attachment,
        [0x40, 0x80, 0xc0, 0xff].repeat(4),
        "every texel is the stage buffer's own payload: {}",
        hex(&attachment)
    );

    // The mutation control: the same pass with another payload lands that
    // payload's bytes, so the readback is the buffer's content rather than a
    // constant the module carries.
    let green: Vec<u8> = [0.0_f32, 1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mutated = translated_stage_buffer_readback(&provider, &compute, &render, &green);
    eprintln!("mutated stage-buffer readback: {}", hex(&mutated));
    assert_eq!(mutated, [0x00, 0xff, 0x00, 0xff].repeat(4));

    // The arrangement control: the same stage translated into set 2 is bound
    // through that set and lands the same bytes, so the slot really is the
    // reflection's own rather than the reviewed pair's fixed one.
    let Some((set2_provider, set2_compute, set2_render)) = translated_stage_buffer_fixture(2)
    else {
        return;
    };
    let set2 = translated_stage_buffer_readback(&set2_provider, &set2_compute, &set2_render, &tint);
    eprintln!("fragment-set stage-buffer readback: {}", hex(&set2));
    assert_eq!(
        set2, attachment,
        "the fragment set's translated stage lands the same bytes as the default set's"
    );
}

/// The widened stage-buffer shape (`research/docs/23` §3.3, §108, E-SB1): the
/// milestone's own vertex stage beside a fragment stage that reads six
/// `[[buffer(n)]]` arguments — the pipeline-level list census v13's deep tail
/// states (`stage_buffer_shape_gt4`, 334 rows), one past the first
/// increment's four.
///
/// The six declarations are `Static { max_bytes: 16 }` each, all on the
/// fragment stage, so the whole list fills one descriptor set — the
/// arrangement whose per-set floor (eight storage buffers) is what the
/// widened ceiling rests on.
const WIDENED_STAGE_BUFFER_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_stage_buffer_six.frag.ll");
const WIDENED_STAGE_BUFFER_FRAGMENT_ENTRY: &str = "render_stage_buffer_six_rgba8";

/// The six slots the widened list declares: one view and one allocation each,
/// so a rail that folds two declarations into one slot cannot pass.
const WIDENED_STAGE_BUFFER_VIEW_BASE: u64 = 720;
const WIDENED_STAGE_BUFFER_ALLOCATION_BASE: u64 = 820;

/// The six payloads: five slots carry `32/255` in every channel, the last
/// `64/255`, so the summed colour is `224/255` on every channel — far from a
/// quantisation tie, and a rail that binds four (or five) of the six
/// declarations lands `128/255` (or `160/255`) instead.
fn widened_stage_buffer_payloads() -> Vec<Vec<u8>> {
    (0..6)
        .map(|index| {
            let value = if index == 5 {
                64.0_f32 / 255.0
            } else {
                32.0_f32 / 255.0
            };
            [value; 4].into_iter().flat_map(f32::to_le_bytes).collect()
        })
        .collect()
}

/// The six declarations the widened pipeline states, canonical by construction
/// (one stage, ascending indices).
fn widened_stage_buffer_declarations() -> Vec<StageBufferBinding> {
    (0..6)
        .map(|index| StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 16 },
        })
        .collect()
}

/// The widened fixture's registrations: the declaring pass's compute kernel
/// and the translated pair whose fragment stage reads six contract-declared
/// slots.
///
/// `set` is the descriptor set the fragment stage is translated into, the same
/// arrangement control the two-slot translated fixture runs: the translator's
/// default (set 0) or the reviewed pair's fragment set (set 2).
fn widened_stage_buffer_fixture(
    set: u32,
) -> Option<(
    VulkanComputeProvider,
    CompiledComputePipeline,
    CompiledComputePipeline,
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
        .compile_pipeline(
            &function,
            digest(b"render_e2e_widened_stage_buffer_compute"),
        )
        .expect("the compute pipeline registers");
    let vertex_library = device
        .new_library_with_air(TRANSLATED_STAGE_BUFFER_VERTEX_AIR)
        .expect("the translated vertex fixture loads");
    let vertex_function = vertex_library
        .function(TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY)
        .expect("the translated vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &vertex_function)
        .expect("the vertex stage translates");
    let fragment_library = device
        .new_library_with_air(WIDENED_STAGE_BUFFER_FRAGMENT_AIR)
        .expect("the widened fragment fixture loads");
    let fragment_function = fragment_library
        .function(WIDENED_STAGE_BUFFER_FRAGMENT_ENTRY)
        .expect("the widened fragment entry exists");
    let fragment = TranslatedRenderStage::translate_with_policy_and_layout(
        RenderStage::Fragment,
        &fragment_function,
        executor.spirv_feature_policy(),
        DescriptorLayout {
            set,
            ..DescriptorLayout::default()
        },
    )
    .expect("the widened fragment stage translates");
    let descriptors = fragment
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
        .map(|binding| (binding.metal_index, binding.descriptor))
        .collect::<Vec<_>>();
    eprintln!("widened stage-buffer reflection: buffers={descriptors:?}");
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
                fragment_entry: WIDENED_STAGE_BUFFER_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: widened_stage_buffer_declarations(),
                textures: Vec::new(),
            },
            vertex,
            fragment,
            logical_digest: digest(b"render_e2e_widened_stage_buffer"),
        })
        .expect("the widened stage-buffer pair registers");
    Some((provider, compute, render))
}

/// One render-bearing trace whose pass binds all six declarations the widened
/// pipeline states.
fn widened_stage_buffer_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    payloads: &[Vec<u8>],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = payloads
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            stage_buffer_view(
                RenderPipelineStage::Fragment,
                u32::try_from(index).expect("six slots"),
                ViewId::new(WIDENED_STAGE_BUFFER_VIEW_BASE + index as u64),
                AllocationId::new(WIDENED_STAGE_BUFFER_ALLOCATION_BASE + index as u64),
                bytes,
            )
        })
        .collect();
    stage_buffer_trace_with_pass(provider, compute, render, pass)
}

/// Submit the widened trace and return the attachment's readback.
fn widened_stage_buffer_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    payloads: &[Vec<u8>],
) -> Vec<u8> {
    let (trace, resources) = widened_stage_buffer_trace(provider, compute, render, payloads);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the widened stage-buffer trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The widened face's execution reading (E-SB1, `research/docs/23` §108): six
/// declarations — past the first increment's four, inside the widened eight —
/// enter the provider, the submission completes, and every texel of the 2×2
/// attachment is the six payloads' sum through the format's quantisation. The
/// arrangement control runs the same list translated into set 0 and set 2: the
/// two frames have to be byte-identical, so the six slots really are the
/// module's own rather than the reviewed pair's fixed ones.
#[test]
fn a_widened_stage_buffer_shape_enters_the_rail_and_lands_its_bytes() {
    let payloads = widened_stage_buffer_payloads();
    let expected = [0xe0_u8; 4].repeat(4);
    let mut frames = Vec::new();
    for set in [0_u32, 2] {
        let Some((provider, compute, render)) = widened_stage_buffer_fixture(set) else {
            return;
        };
        let frame = widened_stage_buffer_readback(&provider, &compute, &render, &payloads);
        eprintln!("widened stage-buffer readback (set {set}): {}", hex(&frame));
        assert_eq!(
            frame, expected,
            "every texel is the six payloads' sum (224/255) through the format's quantisation"
        );
        frames.push(frame);
    }
    assert_eq!(
        frames[0], frames[1],
        "the two descriptor arrangements land the same bytes"
    );
}

/// The widened ceiling's refusal (E-SB1, `research/docs/23` §108): a pass that
/// binds nine slots — one past `MAX_RENDER_STAGE_BUFFERS` — is refused by name
/// with the count and the ceiling on the error, instead of running with a
/// declaration dropped. Both halves of the pair state nine (the pass's list and
/// the pipeline's), so the refusal is the count rule rather than a pairing
/// disagreement; the slug, class, fields and detail are the reading.
#[test]
fn a_stage_buffer_list_above_the_widened_ceiling_is_refused_by_name() {
    let Some((provider, compute, render, _other)) = stage_buffer_fixture() else {
        return;
    };
    let ceiling = metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS;
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = (0..=u32::try_from(ceiling).expect("ceiling"))
        .map(|index| {
            stage_buffer_view(
                RenderPipelineStage::Fragment,
                index,
                ViewId::new(730 + u64::from(index)),
                AllocationId::new(830 + u64::from(index)),
                &[0_u8; 16],
            )
        })
        .collect();
    let (mut trace, resources) = stage_buffer_trace_with_pass(&provider, &compute, &render, pass);
    let wide = trace
        .pipelines
        .iter_mut()
        .find(|pipeline| pipeline.pipeline_id == render.pipeline_id)
        .expect("the render pipeline is in the trace");
    wide.render = Some(RenderPipelineContract {
        vertex_entry: "stage_buffer_positions_main".to_owned(),
        fragment_entry: "stage_buffer_tint_main".to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        stage_buffers: (0..=u32::try_from(ceiling).expect("ceiling"))
            .map(|index| StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 16 },
            })
            .collect(),
        textures: Vec::new(),
    });
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("nine stage-buffer slots exceed the widened ceiling");
    eprintln!(
        "widened stage-buffer refusal: phase={:?} class={:?} slug={} fields={:?} detail={:?}",
        refusal.phase, refusal.class, refusal.slug, refusal.fields, refusal.detail
    );
    assert_eq!(refusal.slug, "render_stage_buffer_limit");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
}

/// The vertex half of the *per-stage* shape (E-SB2, `research/docs/23` §117):
/// seven `[[buffer(n)]]` arguments — positions in `b0` and six offsets folded
/// into the clip position — translated into the canonical namespace set 1.
const PER_STAGE_CEILING_VERTEX_AIR: &str =
    include_str!("fixtures/render_stage_buffer_seven.vert.ll");
const PER_STAGE_CEILING_VERTEX_ENTRY: &str = "render_stage_buffer_seven_positions";

/// The fragment half is the widened increment's own six-argument module
/// (`render_stage_buffer_six.frag.ll`), so the pair declares thirteen slots:
/// seven on the vertex stage, six on the fragment stage — neither past the
/// contract's per-stage ceiling, the pair past the old list bound of eight.
const PER_STAGE_CEILING_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_stage_buffer_six.frag.ll");
const PER_STAGE_CEILING_FRAGMENT_ENTRY: &str = "render_stage_buffer_six_rgba8";

/// The thirteen views and allocations the fixture binds: one each, so a rail
/// that folds two declarations into one slot cannot pass.
const PER_STAGE_CEILING_VERTEX_VIEW_BASE: u64 = 780;
const PER_STAGE_CEILING_VERTEX_ALLOCATION_BASE: u64 = 880;
const PER_STAGE_CEILING_FRAGMENT_VIEW_BASE: u64 = 800;
const PER_STAGE_CEILING_FRAGMENT_ALLOCATION_BASE: u64 = 900;

/// The three `float2` clip positions the vertex stage's `b0` carries: the
/// oversize triangle that covers every pixel centre of a 2x2 viewport.
fn per_stage_ceiling_positions() -> Vec<u8> {
    [(-1.0_f32, -1.0_f32), (3.0, -1.0), (-1.0, 3.0)]
        .into_iter()
        .flat_map(|(x, y)| [x, y])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The vertex stage's six offset payloads: zeros, so the position the module
/// returns is exactly `b0`'s triangle. A mutation replaces one of them with a
/// shift the 2x2 viewport cannot see the triangle through.
fn per_stage_ceiling_offsets() -> Vec<Vec<u8>> {
    (0..6).map(|_| vec![0_u8; 16]).collect()
}

/// The fragment stage's six payloads: five slots of `32/255` and one of
/// `64/255`, the widened increment's own sum (`224/255` on every channel).
fn per_stage_ceiling_payloads() -> Vec<Vec<u8>> {
    (0..6)
        .map(|index| {
            let value = if index == 5 {
                64.0_f32 / 255.0
            } else {
                32.0_f32 / 255.0
            };
            [value; 4].into_iter().flat_map(f32::to_le_bytes).collect()
        })
        .collect()
}

/// The thirteen declarations the pair's contract states, canonical by
/// construction: the vertex stage's seven ascending, then the fragment
/// stage's six.
fn per_stage_ceiling_declarations() -> Vec<StageBufferBinding> {
    let declare = |stage: RenderPipelineStage, index: u32, max_bytes: u64| StageBufferBinding {
        stage,
        index,
        access: BufferAccess::Read,
        footprint: FootprintProof::Static { max_bytes },
    };
    // The vertex stage's `b0` carries the three `float2` positions (24 bytes);
    // its six offsets are one `float4` each, exactly as the module loads them.
    std::iter::once(declare(RenderPipelineStage::Vertex, 0, 24))
        .chain((1..7).map(|index| declare(RenderPipelineStage::Vertex, index, 16)))
        .chain((0..6).map(|index| declare(RenderPipelineStage::Fragment, index, 16)))
        .collect()
}

/// The per-stage fixture's registrations: the declaring compute kernel and the
/// translated pair, the vertex stage in the canonical namespace set 1 and the
/// fragment stage in the translator's own set 0 (E-TX9's arrangement).
fn per_stage_ceiling_fixture() -> Option<(
    VulkanComputeProvider,
    CompiledComputePipeline,
    CompiledComputePipeline,
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
        .compile_pipeline(&function, digest(b"render_e2e_per_stage_ceiling_compute"))
        .expect("the compute pipeline registers");
    let vertex_library = device
        .new_library_with_air(PER_STAGE_CEILING_VERTEX_AIR)
        .expect("the per-stage vertex fixture loads");
    let vertex_function = vertex_library
        .function(PER_STAGE_CEILING_VERTEX_ENTRY)
        .expect("the per-stage vertex entry exists");
    let vertex = TranslatedRenderStage::translate_with_policy_and_layout(
        RenderStage::Vertex,
        &vertex_function,
        executor.spirv_feature_policy(),
        metal_api_vulkan::stage_buffer_namespace_layout(),
    )
    .expect("the per-stage vertex stage translates into the namespace set");
    let fragment_library = device
        .new_library_with_air(PER_STAGE_CEILING_FRAGMENT_AIR)
        .expect("the per-stage fragment fixture loads");
    let fragment_function = fragment_library
        .function(PER_STAGE_CEILING_FRAGMENT_ENTRY)
        .expect("the per-stage fragment entry exists");
    let fragment = TranslatedRenderStage::translate_with_policy(
        RenderStage::Fragment,
        &fragment_function,
        executor.spirv_feature_policy(),
    )
    .expect("the per-stage fragment stage translates");
    eprintln!(
        "per-stage ceiling reflection: vertex={:?} fragment={:?}",
        vertex_slots(&vertex),
        vertex_slots(&fragment)
    );
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: PER_STAGE_CEILING_VERTEX_ENTRY.to_owned(),
                fragment_entry: PER_STAGE_CEILING_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: per_stage_ceiling_declarations(),
                textures: Vec::new(),
            },
            vertex,
            fragment,
            logical_digest: digest(b"render_e2e_per_stage_ceiling"),
        })
        .expect("the thirteen-slot pair registers");
    Some((provider, compute, render))
}

/// The reflected `(metal index, descriptor slot)` pairs one translated stage
/// states, as the fixture's own reading of where its slots landed.
fn vertex_slots(
    stage: &TranslatedRenderStage,
) -> Vec<(u32, Option<metal2vulkan::reflect::DescriptorLocation>)> {
    stage
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
        .map(|binding| (binding.metal_index, binding.descriptor))
        .collect()
}

/// One render-bearing trace whose pass binds all thirteen declarations: the
/// vertex stage's seven views and the fragment stage's six.
fn per_stage_ceiling_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
    offsets: &[Vec<u8>],
    payloads: &[Vec<u8>],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = std::iter::once(stage_buffer_view(
        RenderPipelineStage::Vertex,
        0,
        ViewId::new(PER_STAGE_CEILING_VERTEX_VIEW_BASE),
        AllocationId::new(PER_STAGE_CEILING_VERTEX_ALLOCATION_BASE),
        positions,
    ))
    .chain(offsets.iter().enumerate().map(|(index, bytes)| {
        stage_buffer_view(
            RenderPipelineStage::Vertex,
            u32::try_from(index).expect("six offsets") + 1,
            ViewId::new(PER_STAGE_CEILING_VERTEX_VIEW_BASE + 1 + index as u64),
            AllocationId::new(PER_STAGE_CEILING_VERTEX_ALLOCATION_BASE + 1 + index as u64),
            bytes,
        )
    }))
    .chain(payloads.iter().enumerate().map(|(index, bytes)| {
        stage_buffer_view(
            RenderPipelineStage::Fragment,
            u32::try_from(index).expect("six payloads"),
            ViewId::new(PER_STAGE_CEILING_FRAGMENT_VIEW_BASE + index as u64),
            AllocationId::new(PER_STAGE_CEILING_FRAGMENT_ALLOCATION_BASE + index as u64),
            bytes,
        )
    }))
    .collect();
    stage_buffer_trace_with_pass(provider, compute, render, pass)
}

/// Submit the per-stage trace and return the attachment's readback.
fn per_stage_ceiling_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
    offsets: &[Vec<u8>],
    payloads: &[Vec<u8>],
) -> Vec<u8> {
    let (trace, resources) =
        per_stage_ceiling_trace(provider, compute, render, positions, offsets, payloads);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the thirteen-slot trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The per-stage ceiling's execution reading (E-SB2, `research/docs/23` §117).
///
/// Census v39 still reads 92 class exits under
/// `render_provider_out_of_class_stage_buffer_shape`
/// (`evidence/gate3-census-v39-2026-09-19/v39-summary.txt`), and the shape
/// behind them is a pair that declares more than the old *list* bound between
/// its two stages. This fixture is that shape: thirteen declarations, seven on
/// the vertex stage and six on the fragment stage, each stage inside the
/// contract's own ceiling.
///
/// Three readings are the three things the increment states: the pair enters
/// the rail, submits and lands every texel of the 2x2 attachment as the
/// fragment half's six payloads summed through the format's quantisation (a
/// rail that dropped a declaration lands a different colour); the *vertex*
/// half's bytes move the frame, so its seven declarations are bound bytes
/// rather than declarations on paper; and the snapshot's own window is what
/// admits the shape, with the list bound beside it being the pair's sum rather
/// than the old single-list eight.
#[test]
fn a_per_stage_stage_buffer_shape_enters_the_rail_and_lands_its_bytes() {
    let Some((provider, compute, render)) = per_stage_ceiling_fixture() else {
        return;
    };
    let capabilities = provider.capabilities();
    eprintln!(
        "per-stage ceiling window: per_stage={} list={}",
        capabilities.max_render_stage_buffers_per_stage, capabilities.max_render_stage_buffers
    );
    assert!(capabilities.declares_render_stage_buffer_per_stage_ceiling());
    assert_eq!(
        capabilities.max_render_stage_buffers_per_stage,
        metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS as u32
    );
    assert_eq!(
        capabilities.max_render_stage_buffers,
        metal_api_core::provider::MAX_RENDER_STAGE_BUFFER_DECLARATIONS as u32
    );

    let positions = per_stage_ceiling_positions();
    let offsets = per_stage_ceiling_offsets();
    let payloads = per_stage_ceiling_payloads();
    let expected = [0xe0_u8; 4].repeat(4);
    let frame = per_stage_ceiling_readback(
        &provider, &compute, &render, &positions, &offsets, &payloads,
    );
    eprintln!("per-stage ceiling readback: {}", hex(&frame));
    assert_eq!(
        frame, expected,
        "every texel is the six fragment payloads' sum (224/255) through the format's quantisation"
    );

    // The fragment half's last slot is really bound: with it zeroed the sum is
    // five slots of 32/255 — 160/255 rather than the fixture's 224/255.
    let mut moved_payloads = payloads.clone();
    moved_payloads[5] = [0.0_f32; 4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let dropped = per_stage_ceiling_readback(
        &provider,
        &compute,
        &render,
        &positions,
        &offsets,
        &moved_payloads,
    );
    eprintln!("per-stage ceiling fragment mutation: {}", hex(&dropped));
    assert_eq!(dropped, [0xa0_u8; 4].repeat(4));
    assert_ne!(dropped, frame);

    // The vertex half's slots are bound bytes too: an offset that pushes the
    // triangle past the viewport leaves the clear sentinel behind, so the
    // seven declarations the contract states are read rather than dropped.
    let mut shifted_offsets = offsets.clone();
    shifted_offsets[5] = [2.0_f32, 0.0, 0.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let shifted = per_stage_ceiling_readback(
        &provider,
        &compute,
        &render,
        &positions,
        &shifted_offsets,
        &payloads,
    );
    eprintln!("per-stage ceiling vertex mutation: {}", hex(&shifted));
    assert_eq!(
        shifted,
        CLEAR_SENTINEL.repeat(4),
        "a vertex offset past the viewport leaves the draw outside every pixel centre"
    );
    assert_ne!(shifted, frame);
}

/// The per-stage ceiling's refusal (E-SB2, `research/docs/23` §117): the newer
/// list bound is the *pair's* sum, so a single stage that names nine slots is
/// still refused by name — the count rule never stopped being the stage's.
#[test]
fn a_stage_list_above_the_per_stage_ceiling_is_refused_by_name() {
    let Some((provider, compute, render)) = per_stage_ceiling_fixture() else {
        return;
    };
    let positions = per_stage_ceiling_positions();
    let offsets = per_stage_ceiling_offsets();
    let payloads = per_stage_ceiling_payloads();
    let (mut trace, resources) = per_stage_ceiling_trace(
        &provider, &compute, &render, &positions, &offsets, &payloads,
    );
    let ceiling = metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS;
    {
        let wide = trace
            .pipelines
            .iter_mut()
            .find(|pipeline| pipeline.pipeline_id == render.pipeline_id)
            .expect("the render pipeline is in the trace");
        wide.render = Some(RenderPipelineContract {
            vertex_entry: PER_STAGE_CEILING_VERTEX_ENTRY.to_owned(),
            fragment_entry: PER_STAGE_CEILING_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            stage_buffers: (0..=u32::try_from(ceiling).expect("ceiling"))
                .map(|index| StageBufferBinding {
                    stage: RenderPipelineStage::Fragment,
                    index,
                    access: BufferAccess::Read,
                    footprint: FootprintProof::Static { max_bytes: 16 },
                })
                .collect(),
            textures: Vec::new(),
        });
    }
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("nine fragment declarations cross the per-stage ceiling");
    eprintln!(
        "per-stage ceiling refusal: phase={:?} class={:?} slug={} fields={:?} detail={:?}",
        refusal.phase, refusal.class, refusal.slug, refusal.fields, refusal.detail
    );
    assert_eq!(refusal.slug, "render_stage_buffer_limit");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
}

/// R9i (E side): the present rail executes the reviewed stage-buffer pair.
///
/// Both rails render the same two modules into their own target, and the
/// present rail's target bytes have to be the offscreen rail's bytes: the
/// fragment stage's payload in the texel the positions buffer covers, the
/// clear sentinel elsewhere. The counters are the present action's own
/// bookkeeping, so a rail that presented the sentinel (or presented twice)
/// cannot pass as the pair's readback.
#[test]
fn a_presenting_pass_reads_its_stage_buffers_and_lands_their_bytes() {
    let Some((provider, compute, reviewed, _other)) = stage_buffer_fixture() else {
        return;
    };
    let positions = stage_buffer_positions();
    let tint = stage_buffer_tint();
    let offscreen = stage_buffer_readback(&provider, &compute, &reviewed, &positions, &tint);

    let (mut trace, resources) =
        stage_buffer_trace(&provider, &compute, &reviewed, &positions, &tint);
    attach_present(
        &mut trace,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );
    let (acquires_before, presents_before) = provider.present_counts();
    let presented = readback(
        &submit_presenting_trace(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );
    eprintln!("presented stage-buffer readback: {}", hex(&presented));
    eprintln!("offscreen stage-buffer readback: {}", hex(&offscreen));
    eprintln!("stage buffer payload: {}", hex(&tint));

    assert_eq!(
        presented, offscreen,
        "the present rail lands the offscreen rail's own bytes for the same pass"
    );
    assert_eq!(
        &presented[..4],
        &[0x40, 0x80, 0xc0, 0xff],
        "the covered texel is the fragment stage buffer's own payload"
    );
    assert!(
        presented[4..]
            .chunks_exact(4)
            .all(|texel| texel == CLEAR_SENTINEL),
        "the positions buffer's triangle covers exactly one texel: {}",
        hex(&presented)
    );
    assert_ne!(
        PRESENT_SENTINEL.as_slice(),
        &presented[..4],
        "a surviving present sentinel means the pass never rendered into the target"
    );
    assert_eq!(
        provider.present_counts(),
        (acquires_before + 1, presents_before + 1),
        "one acquire and one present per present action"
    );
}

/// R9i (E side): the present rail executes a **translated** stage's stage
/// buffer.
///
/// This is the arm the R9 tail was about: a registration whose fragment half
/// came out of metal2vulkan declares its `[[buffer(0)]]` slot, and the
/// presenting pass binds it. The rail binds the set the reflection names
/// (here the translator's default set 0) and the target lands the payload —
/// byte for byte the offscreen rail's readback of the same pass.
#[test]
fn a_translated_presenting_pass_reads_its_stage_buffer_and_lands_its_bytes() {
    let Some((provider, compute, render)) = translated_stage_buffer_fixture(0) else {
        return;
    };
    let tint = stage_buffer_tint();
    let offscreen = translated_stage_buffer_readback(&provider, &compute, &render, &tint);

    let (mut trace, resources) = translated_stage_buffer_trace(&provider, &compute, &render, &tint);
    attach_present(
        &mut trace,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );
    let (acquires_before, presents_before) = provider.present_counts();
    let presented = readback(
        &submit_presenting_trace(&provider, &trace, &resources),
        ATTACHMENT_VIEW,
    );
    eprintln!(
        "translated presenting stage-buffer readback: {}",
        hex(&presented)
    );
    eprintln!(
        "translated offscreen stage-buffer readback: {}",
        hex(&offscreen)
    );
    eprintln!("stage buffer payload: {}", hex(&tint));
    assert_eq!(
        presented,
        [0x40, 0x80, 0xc0, 0xff].repeat(4),
        "every texel is the stage buffer's own payload: {}",
        hex(&presented)
    );
    assert_eq!(
        presented, offscreen,
        "the present rail lands the offscreen rail's own bytes for the same pass"
    );
    assert_eq!(
        provider.present_counts(),
        (acquires_before + 1, presents_before + 1),
        "one acquire and one present per present action"
    );

    // The mutation control: the same pass with another payload presents that
    // payload's bytes, so the target is the buffer's content rather than a
    // constant the module (or the format list) carries.
    let green: Vec<u8> = [0.0_f32, 1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let (mut mutated, mutated_resources) =
        translated_stage_buffer_trace(&provider, &compute, &render, &green);
    attach_present(
        &mut mutated,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );
    let mutated = readback(
        &submit_presenting_trace(&provider, &mutated, &mutated_resources),
        ATTACHMENT_VIEW,
    );
    eprintln!(
        "mutated presenting stage-buffer readback: {}",
        hex(&mutated)
    );
    assert_eq!(mutated, [0x00, 0xff, 0x00, 0xff].repeat(4));
}

/// R9i (E side): the present rail's refusal for a stage buffer its modules do
/// not read keeps the offscreen rail's name and fields.
///
/// The milestone pair declares the two slots with modules that read no
/// `[[buffer(n)]]` argument, so presenting that pass would drop the bindings.
/// The refusal is the same slug and the same two fields the offscreen rail
/// reports for the identical shape — a rail that refused with a different name
/// (or with no field) would not be the same boundary.
#[test]
fn a_presenting_pass_beside_a_reviewed_stage_buffer_is_refused_by_name() {
    let Some((provider, compute, _reviewed, other)) = stage_buffer_fixture() else {
        return;
    };
    let positions = stage_buffer_positions();
    let tint = stage_buffer_tint();
    let (mut trace, resources) = stage_buffer_trace(&provider, &compute, &other, &positions, &tint);
    attach_present(
        &mut trace,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );
    let refused = match provider.capabilities().validate_trace(trace, resources) {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("the milestone pair reads no stage buffer"),
        Err(error) => error,
    };
    eprintln!("stage buffers under the milestone pair beside a present refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_stage_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("stage"),
        Some(&FieldValue::Text("vertex".to_owned()))
    );
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
}

/// R9i (E side): the reviewed stage-buffer pair needs both of its slots beside
/// a present action too.
///
/// The registration declares no slot while its module reads one, so a
/// presenting pass that binds nothing would draw from a descriptor nobody
/// filled. The present rail refuses it with the offscreen rail's slug rather
/// than presenting the shape.
#[test]
fn a_presenting_stage_buffer_pair_requires_its_two_bindings() {
    let Some(executor) = executor() else {
        return;
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let unbound = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "stage_buffer_positions_main".to_owned(),
                fragment_entry: "stage_buffer_tint_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: Vec::new(),
                textures: Vec::new(),
            },
            vertex_spirv: STAGE_BUFFER_POSITIONS_VERT_SPV.to_vec(),
            fragment_spirv: STAGE_BUFFER_TINT_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_present_stage_buffer_unbound"),
        })
        .expect("the pair registers: the contract declares no slot");
    let function = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>)
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(
            &function,
            digest(b"render_e2e_present_stage_buffer_unbound_compute"),
        )
        .expect("the compute pipeline registers");
    let pass = render_pass(unbound.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    let (mut trace, resources) = stage_buffer_trace_with_pass(&provider, &compute, &unbound, pass);
    attach_present(
        &mut trace,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        AttachmentFormat::Rgba8Unorm,
        PRESENT_SENTINEL,
    );
    let refused = match provider.capabilities().validate_trace(trace, resources) {
        Ok(admitted) => provider
            .submit(admitted)
            .expect_err("the reviewed pair needs its two bindings"),
        Err(error) => error,
    };
    eprintln!("unbound stage-buffer pair beside a present refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_binding_required");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
}

// ---------------------------------------------------------------------------
// R9f (`research/docs/23` §3.3, v86): writable stage buffers and affine
// footprints.
//
// The two halves are measured apart. The write half is one translated fragment
// stage that copies its read-only slot into its writable one, so a submission
// has two falsifiable readings: the frame is the source's payload, and the
// sink's *writeback* is that same payload — moving the payload moves both, and a
// rail that bound the sink read-only or dropped its landing leaves one of them
// behind. The affine half is one translated vertex stage that reads
// `positions[vertex_id]`, so the draw's own vertex count is the bound the
// contract's affine declaration is proven against.
// ---------------------------------------------------------------------------

/// The translated fragment stage of the write half: a read-only `source` at
/// `[[buffer(0)]]` and a writable `sink` at `[[buffer(1)]]`.
const WRITE_STAGE_BUFFER_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_stage_buffer_write.frag.ll");
const WRITE_STAGE_BUFFER_FRAGMENT_ENTRY: &str = "render_stage_buffer_write_rgba8";

/// The translated vertex stage of the affine half: `positions[vertex_id]`.
const AFFINE_STAGE_BUFFER_VERTEX_AIR: &str =
    include_str!("fixtures/render_stage_buffer_positions.vert.ll");
const AFFINE_STAGE_BUFFER_VERTEX_ENTRY: &str = "render_vertex_positions";

/// The affine half's fragment stage: the solid 8-bit UNORM module that reads no
/// buffer at all, so the draw's only stage buffer is the vertex one.
const AFFINE_STAGE_BUFFER_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_offscreen_2x2.frag.ll");
const AFFINE_STAGE_BUFFER_FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// The declaring pass's kernel for the write half: three bindings, so the
/// pass can declare the attachment (read), its own copy landing (write) and the
/// writable stage buffer's sink (read) in one go. The witness binding is what
/// keeps the sink's pool entry a *read*: a compute pass may not write bytes the
/// draw writes (`RenderInputComputeConflict`), and the pool entry the
/// writeback lands in is what the trace has to declare.
const COPY_WORD_WITH_WITNESS_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word_with_witness.ll");

const WRITE_STAGE_BUFFER_SOURCE_VIEW: ViewId = ViewId::new(716);
const WRITE_STAGE_BUFFER_SOURCE_ALLOCATION: AllocationId = AllocationId::new(816);
const WRITE_STAGE_BUFFER_SINK_VIEW: ViewId = ViewId::new(717);
const WRITE_STAGE_BUFFER_SINK_ALLOCATION: AllocationId = AllocationId::new(817);
const WRITE_STAGE_BUFFER_WITNESS_VIEW: ViewId = ViewId::new(719);
const WRITE_STAGE_BUFFER_WITNESS_ALLOCATION: AllocationId = AllocationId::new(819);
const AFFINE_STAGE_BUFFER_VIEW: ViewId = ViewId::new(718);
const AFFINE_STAGE_BUFFER_ALLOCATION: AllocationId = AllocationId::new(818);

/// The write half's declaration pair: the source is read with a static extent,
/// the sink written with one (`research/docs/23` §3.3, v86).
fn write_stage_buffer_declarations(
    source_access: BufferAccess,
    sink_access: BufferAccess,
) -> Vec<StageBufferBinding> {
    vec![
        StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: source_access,
            footprint: FootprintProof::Static { max_bytes: 16 },
        },
        StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 1,
            access: sink_access,
            footprint: FootprintProof::Static { max_bytes: 16 },
        },
    ]
}

/// Register the write half's two pipelines: the declaring pass's kernel and
/// one translated pair whose fragment stage is the copy fixture.
fn write_stage_buffer_registration(
    declarations: Vec<StageBufferBinding>,
) -> Result<
    (
        VulkanComputeProvider,
        CompiledComputePipeline,
        CompiledComputePipeline,
    ),
    ProviderError,
> {
    let Some(executor) = executor() else {
        return Err(ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "render_e2e_no_device",
        )
        .expect("static slug"));
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .expect("the provider context builds");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_WITH_WITNESS_AIR)
        .expect("the witness fixture loads")
        .function("copy_word_with_witness")
        .expect("the witness entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_e2e_r9f_write_compute"))
        .expect("the compute pipeline registers");
    let vertex_library = device
        .new_library_with_air(TRANSLATED_STAGE_BUFFER_VERTEX_AIR)
        .expect("the translated vertex fixture loads");
    let vertex_function = vertex_library
        .function(TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY)
        .expect("the translated vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &vertex_function)
        .expect("the vertex stage translates");
    let fragment_library = device
        .new_library_with_air(WRITE_STAGE_BUFFER_FRAGMENT_AIR)
        .expect("the write fixture loads");
    let fragment_function = fragment_library
        .function(WRITE_STAGE_BUFFER_FRAGMENT_ENTRY)
        .expect("the write entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &fragment_function)
        .expect("the fragment stage translates");
    let render = provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: RenderPipelineContract {
            vertex_entry: TRANSLATED_STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
            fragment_entry: WRITE_STAGE_BUFFER_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            stage_buffers: declarations,
            textures: Vec::new(),
        },
        vertex,
        fragment,
        logical_digest: digest(b"render_e2e_r9f_write_stages"),
    })?;
    Ok((provider, compute, render))
}

/// The refusal of one registration that is expected to fail: `expect_err`
/// needs `Debug` on the success half, and a provider context is not `Debug`.
fn registration_refusal(
    result: Result<
        (
            VulkanComputeProvider,
            CompiledComputePipeline,
            CompiledComputePipeline,
        ),
        ProviderError,
    >,
) -> ProviderError {
    match result {
        Ok(_) => panic!("the registration was expected to be refused"),
        Err(error) => error,
    }
}

/// The three registrations one positive R9f fixture expects, with "this box has
/// no device" the only refusal that silently skips the test.
fn expect_registration(
    result: Result<
        (
            VulkanComputeProvider,
            CompiledComputePipeline,
            CompiledComputePipeline,
        ),
        ProviderError,
    >,
) -> Option<(
    VulkanComputeProvider,
    CompiledComputePipeline,
    CompiledComputePipeline,
)> {
    match result {
        Ok(value) => Some(value),
        Err(error) if error.slug == "render_e2e_no_device" => None,
        Err(error) => panic!("the fixture registration was refused: {error:?}"),
    }
}

/// One trace of the write half: the declaring compute pass and the render pass
/// whose fragment stage copies `source` into `sink`.
///
/// `sink_in_pool` is the trace's own declaration choice, and it is the shape
/// the no-landing refusal is about: with the sink declared by the compute pass
/// the pool holds its view and the writeback lands there; without it the view
/// is not part of the trace's pool at all.
fn write_stage_buffer_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    source: &[u8],
    sink_in_pool: bool,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = vec![
        StageBufferView {
            stage: RenderPipelineStage::Fragment,
            view: BufferView {
                view_id: WRITE_STAGE_BUFFER_SOURCE_VIEW,
                metal_binding: 0,
                allocation_id: WRITE_STAGE_BUFFER_SOURCE_ALLOCATION,
                offset: 0,
                length: u64::try_from(source.len()).expect("source length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(source.to_vec()),
            },
        },
        StageBufferView {
            stage: RenderPipelineStage::Fragment,
            view: BufferView {
                view_id: WRITE_STAGE_BUFFER_SINK_VIEW,
                metal_binding: 1,
                allocation_id: WRITE_STAGE_BUFFER_SINK_ALLOCATION,
                offset: 0,
                length: 16,
                access: BufferAccess::Write,
                attribute_stride: None,
                // The sink starts as zeros: a rail that executes the pass but
                // lands nothing would publish these bytes.
                source: BufferSource::OwnedBytes(vec![0; 16]),
            },
        },
    ];
    let mut buffers = vec![
        BufferView {
            view_id: ATTACHMENT_VIEW,
            metal_binding: 0,
            allocation_id: ATTACHMENT_ALLOCATION,
            offset: 0,
            length: attachment_length(AttachmentFormat::Rgba8Unorm) as u64,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
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
    ];
    if sink_in_pool {
        buffers.push(BufferView {
            view_id: WRITE_STAGE_BUFFER_SINK_VIEW,
            metal_binding: 2,
            allocation_id: WRITE_STAGE_BUFFER_SINK_ALLOCATION,
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
        });
    } else {
        // The same three-binding kernel with its witness slot filled by a view
        // of its own: the compute pass stays legal, and the writable stage
        // buffer's view is what the trace does *not* declare.
        buffers.push(BufferView {
            view_id: WRITE_STAGE_BUFFER_WITNESS_VIEW,
            metal_binding: 2,
            allocation_id: WRITE_STAGE_BUFFER_WITNESS_ALLOCATION,
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
        });
    }
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(14),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: compute.pipeline_id,
                buffers,
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
        (ATTACHMENT_ALLOCATION, 16),
        (SCRATCH_ALLOCATION, 8),
        (WRITE_STAGE_BUFFER_SOURCE_ALLOCATION, 16),
        (WRITE_STAGE_BUFFER_SINK_ALLOCATION, 16),
        (WRITE_STAGE_BUFFER_WITNESS_ALLOCATION, 16),
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

/// Submit one write-half trace and return `(frame, sink bytes)`.
fn write_stage_buffer_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    source: &[u8],
    sink_in_pool: bool,
) -> (Vec<u8>, Vec<u8>) {
    let (trace, resources) =
        write_stage_buffer_trace(provider, compute, render, source, sink_in_pool);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the write-half trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    (
        readback(&writebacks, ATTACHMENT_VIEW),
        readback(&writebacks, WRITE_STAGE_BUFFER_SINK_VIEW),
    )
}

/// The write half of `research/docs/23` §3.3 v86 on the Vulkan rail: the
/// declaration says the fragment stage writes its `[[buffer(1)]]`, the module's
/// reflection says the same, the pass binds it writable, and the bytes land
/// through the same writeback channel the attachments use.
#[test]
fn a_writable_stage_buffer_lands_its_bytes_through_the_writeback_channel() {
    let Some((provider, compute, render)) = expect_registration(write_stage_buffer_registration(
        write_stage_buffer_declarations(BufferAccess::Read, BufferAccess::Write),
    )) else {
        return;
    };
    let payload = stage_buffer_tint();
    let (frame, sink) = write_stage_buffer_readback(&provider, &compute, &render, &payload, true);
    eprintln!("write-half frame: {}", hex(&frame));
    eprintln!("write-half source payload: {}", hex(&payload));
    eprintln!("write-half sink writeback: {}", hex(&sink));
    assert_eq!(frame.len(), 16);
    assert_eq!(
        frame,
        [0x40, 0x80, 0xc0, 0xff].repeat(4),
        "the frame is the source payload through the format's quantisation: {}",
        hex(&frame)
    );
    assert_eq!(
        sink,
        payload,
        "the sink's writeback is the stage's own write: {}",
        hex(&sink)
    );
    assert_ne!(
        sink,
        vec![0_u8; 16],
        "the sink starts as zeros, so a rail that landed nothing would publish them"
    );

    // The mutation control, in the other direction: another payload lands
    // another frame *and* another sink, so both readings are the stage's own
    // bytes rather than a constant either rail carries.
    let green: Vec<u8> = [0.0_f32, 1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let (mutated_frame, mutated_sink) =
        write_stage_buffer_readback(&provider, &compute, &render, &green, true);
    eprintln!("write-half mutated frame: {}", hex(&mutated_frame));
    eprintln!("write-half mutated sink: {}", hex(&mutated_sink));
    assert_eq!(mutated_frame, [0x00, 0xff, 0x00, 0xff].repeat(4));
    assert_eq!(mutated_sink, green);
}

/// The write half's refusal face: every shape that would execute a write the
/// declaration and the module do not agree on stays on this side of the device
/// and is refused by name.
#[test]
fn the_writable_stage_buffer_face_refuses_what_it_cannot_land() {
    // One: the declaration says the source is written, the module only reads
    // it (`research/docs/23` §3.3, v86). Refused at registration, before any
    // device object exists.
    let refusal = registration_refusal(write_stage_buffer_registration(
        write_stage_buffer_declarations(BufferAccess::Write, BufferAccess::Write),
    ));
    eprintln!("write-half refusal, written source: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refusal.fields.get("declared_access"),
        Some(&FieldValue::Text("write".to_owned()))
    );
    assert_eq!(
        refusal.fields.get("reflected_access"),
        Some(&FieldValue::Text("read".to_owned()))
    );

    // Two: a read-only declaration for the slot the module writes — the same
    // disagreement from the other side.
    let refusal = registration_refusal(write_stage_buffer_registration(
        write_stage_buffer_declarations(BufferAccess::Read, BufferAccess::Read),
    ));
    eprintln!("write-half refusal, read sink: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refusal.fields.get("declared_access"),
        Some(&FieldValue::Text("read".to_owned()))
    );
    assert_eq!(
        refusal.fields.get("reflected_access"),
        Some(&FieldValue::Text("write".to_owned()))
    );

    // Three: this rail's *reviewed* stage-buffer modules read their slots, so a
    // writable declaration has no writer behind it.
    let Some(executor) = executor() else {
        return;
    };
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .expect("the provider context builds");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let refusal = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                vertex_entry: "stage_buffer_positions_main".to_owned(),
                fragment_entry: "stage_buffer_tint_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                stage_buffers: vec![StageBufferBinding {
                    stage: RenderPipelineStage::Vertex,
                    index: 0,
                    access: BufferAccess::Write,
                    footprint: FootprintProof::Static { max_bytes: 24 },
                }],
                textures: Vec::new(),
            },
            vertex_spirv: STAGE_BUFFER_POSITIONS_VERT_SPV.to_vec(),
            fragment_spirv: STAGE_BUFFER_TINT_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_e2e_r9f_reviewed_write"),
        })
        .expect_err("a writable declaration under a reviewed module is refused");
    eprintln!("write-half refusal, reviewed module: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_buffer_write_unsupported");
    assert_eq!(
        refusal.fields.get("stage"),
        Some(&FieldValue::Text("vertex".to_owned()))
    );
    assert_eq!(refusal.fields.get("index"), Some(&FieldValue::Unsigned(0)));
    assert_eq!(
        refusal.fields.get("access"),
        Some(&FieldValue::Text("write".to_owned()))
    );

    // Four: a writable stage buffer whose view the trace does not declare has
    // no pool entry the writeback could land in.
    let Some((provider, compute, render)) = expect_registration(write_stage_buffer_registration(
        write_stage_buffer_declarations(BufferAccess::Read, BufferAccess::Write),
    )) else {
        return;
    };
    let payload = stage_buffer_tint();
    let (trace, resources) =
        write_stage_buffer_trace(&provider, &compute, &render, &payload, false);
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a write with no landing view is refused");
    eprintln!("write-half refusal, no landing: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_buffer_writeback_unknown");
    assert_eq!(refusal.class, ProviderErrorClass::Resource);
    assert_eq!(refusal.phase, ProviderPhase::Resolve);

    // Five: the pass's own half of the pair. A read-only declaration beside a
    // *writable* view is the disagreement the pair rules answer, one layer
    // below the module pairing that cases one and two measured.
    let Some((provider, compute, reviewed, _other)) = stage_buffer_fixture() else {
        return;
    };
    let mut pass = render_pass(reviewed.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = vec![
        stage_buffer_view(
            RenderPipelineStage::Vertex,
            0,
            STAGE_BUFFER_POSITION_VIEW,
            STAGE_BUFFER_POSITION_ALLOCATION,
            &stage_buffer_positions(),
        ),
        StageBufferView {
            stage: RenderPipelineStage::Fragment,
            view: BufferView {
                access: BufferAccess::Write,
                ..stage_buffer_view(
                    RenderPipelineStage::Fragment,
                    0,
                    STAGE_BUFFER_TINT_VIEW,
                    STAGE_BUFFER_TINT_ALLOCATION,
                    &stage_buffer_tint(),
                )
                .view
            },
        },
    ];
    let (trace, resources) = stage_buffer_trace_with_pass(&provider, &compute, &reviewed, pass);
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a writable view under a read-only declaration is refused");
    eprintln!("write-half refusal, writable view: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("fragment stage buffer 0 is bound Write")),
        "the detail names the bound access and the declared one: {:?}",
        refusal.detail
    );
}

/// The affine declaration of the vertex fixture: the two `float2` component
/// loads the reflection reports, each `base + vertex_id * 8` (`fields` are the
/// reflection's own numbers — the declaration is a second measurement of one
/// module, not a guess).
fn affine_stage_buffer_declaration() -> StageBufferBinding {
    StageBufferBinding {
        stage: RenderPipelineStage::Vertex,
        index: 0,
        access: BufferAccess::Read,
        footprint: FootprintProof::Affine {
            accesses: vec![
                metal_api_core::provider::AffineAccess {
                    base_offset: 0,
                    access_size: 4,
                    terms: vec![metal_api_core::provider::AffineTerm { axis: 0, stride: 8 }],
                },
                metal_api_core::provider::AffineAccess {
                    base_offset: 4,
                    access_size: 4,
                    terms: vec![metal_api_core::provider::AffineTerm { axis: 0, stride: 8 }],
                },
            ],
        },
    }
}

/// Register the affine half's pair: the translated vertex stage that reads
/// `positions[vertex_id]` beside the translated solid fragment stage.
fn affine_stage_buffer_registration(
    declaration: StageBufferBinding,
) -> Result<
    (
        VulkanComputeProvider,
        CompiledComputePipeline,
        CompiledComputePipeline,
    ),
    ProviderError,
> {
    let Some(executor) = executor() else {
        return Err(ProviderError::new(
            ProviderPhase::Resolve,
            ProviderErrorClass::Capability,
            "render_e2e_no_device",
        )
        .expect("static slug"));
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .expect("the provider context builds");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_e2e_r9f_affine_compute"))
        .expect("the compute pipeline registers");
    let vertex_library = device
        .new_library_with_air(AFFINE_STAGE_BUFFER_VERTEX_AIR)
        .expect("the affine vertex fixture loads");
    let vertex_function = vertex_library
        .function(AFFINE_STAGE_BUFFER_VERTEX_ENTRY)
        .expect("the affine vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &vertex_function)
        .expect("the affine vertex stage translates");
    let fragment_library = device
        .new_library_with_air(AFFINE_STAGE_BUFFER_FRAGMENT_AIR)
        .expect("the affine fragment fixture loads");
    let fragment_function = fragment_library
        .function(AFFINE_STAGE_BUFFER_FRAGMENT_ENTRY)
        .expect("the affine fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &fragment_function)
        .expect("the fragment stage translates");
    let render = provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: RenderPipelineContract {
            vertex_entry: AFFINE_STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
            fragment_entry: AFFINE_STAGE_BUFFER_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            stage_buffers: vec![declaration],
            textures: Vec::new(),
        },
        vertex,
        fragment,
        logical_digest: digest(b"render_e2e_r9f_affine_stages"),
    })?;
    Ok((provider, compute, render))
}

/// One affine-half trace: the declaring compute pass and one draw whose vertex
/// stage reads its positions out of the bound view.
fn affine_stage_buffer_trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
) -> (ComputeTrace, ResourceTableSnapshot) {
    let mut pass = render_pass(render.pipeline_id, AttachmentFormat::Rgba8Unorm, 2, 2);
    pass.stage_buffers = vec![StageBufferView {
        stage: RenderPipelineStage::Vertex,
        view: BufferView {
            view_id: AFFINE_STAGE_BUFFER_VIEW,
            metal_binding: 0,
            allocation_id: AFFINE_STAGE_BUFFER_ALLOCATION,
            offset: 0,
            length: u64::try_from(positions.len()).expect("positions length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(positions.to_vec()),
        },
    }];
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(15),
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
                        length: attachment_length(AttachmentFormat::Rgba8Unorm) as u64,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0; 16]),
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
        (ATTACHMENT_ALLOCATION, 16),
        (SCRATCH_ALLOCATION, 8),
        (
            AFFINE_STAGE_BUFFER_ALLOCATION,
            u64::try_from(positions.len()).expect("positions length"),
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

/// Submit one affine-half trace and return the frame's readback.
fn affine_stage_buffer_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: &[u8],
) -> Vec<u8> {
    let (trace, resources) = affine_stage_buffer_trace(provider, compute, render, positions);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the affine-half trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let writebacks = submitted
        .writebacks
        .into_iter()
        .map(|writeback| (writeback.view_id, writeback.bytes))
        .collect::<Vec<_>>();
    readback(&writebacks, ATTACHMENT_VIEW)
}

/// The affine half of `research/docs/23` §3.3 v86: the declaration states the
/// module's reflected `0/4 + vertex_id * 8` reach, the draw's own vertex count
/// (three) is what bounds it, and the frame is a function of the bytes the
/// vertex stage reads.
#[test]
fn an_affine_stage_buffer_footprint_is_bounded_by_the_draw() {
    let Some((provider, compute, render)) = expect_registration(affine_stage_buffer_registration(
        affine_stage_buffer_declaration(),
    )) else {
        return;
    };
    let positions = stage_buffer_positions();
    let frame = affine_stage_buffer_readback(&provider, &compute, &render, &positions);
    eprintln!("affine-half frame: {}", hex(&frame));
    eprintln!("affine-half positions: {}", hex(&positions));
    assert_eq!(frame.len(), 16);
    assert_eq!(
        &frame[..4],
        &[0x40, 0x80, 0xc0, 0xff],
        "the covered texel is the solid fragment stage's constant: {}",
        hex(&frame)
    );
    assert!(
        frame[4..]
            .chunks_exact(4)
            .all(|texel| texel == CLEAR_SENTINEL),
        "the buffer's triangle covers exactly one texel: {}",
        hex(&frame)
    );

    // The falsification: moving the buffer's own vertices moves the covered
    // texel, so the frame is what the vertex stage read rather than a constant
    // or a geometry the rail computed itself.
    let shifted: Vec<u8> = [-0.9_f32, -0.9, 0.0, -0.9, -0.9, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let moved = affine_stage_buffer_readback(&provider, &compute, &render, &shifted);
    eprintln!("affine-half shifted frame: {}", hex(&moved));
    assert_ne!(moved, frame, "the frame follows the buffer's own vertices");

    // The bound is the draw's, so a view that stops short of
    // `(vertices - 1) * stride + access_size` is refused at admission: three
    // vertices of eight bytes are 24 bytes, and this view declares 16.
    let (trace, resources) =
        affine_stage_buffer_trace(&provider, &compute, &render, &positions[..16]);
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a view shorter than the affine bound is refused");
    eprintln!("affine-half refusal, short view: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_buffer_footprint_unsupported");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);

    // And the other shape of the same gate: a declaration that states a static
    // extent where the module's reach is affine is a disagreement between the
    // declaration and the module, refused at registration.
    let refusal = registration_refusal(affine_stage_buffer_registration(StageBufferBinding {
        stage: RenderPipelineStage::Vertex,
        index: 0,
        access: BufferAccess::Read,
        footprint: FootprintProof::Static { max_bytes: 24 },
    }));
    eprintln!("affine-half refusal, static declaration: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_reflection_mismatch");
    assert_eq!(refusal.fields.get("index"), Some(&FieldValue::Unsigned(0)));
    assert_eq!(
        refusal.fields.get("field"),
        Some(&FieldValue::Text("bindings".to_owned()))
    );
}
