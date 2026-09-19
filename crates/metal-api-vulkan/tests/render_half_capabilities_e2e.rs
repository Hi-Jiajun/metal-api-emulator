//! The 16-bit shader capability pair (2026-09-20, census v48's LPF pipeline).
//!
//! Census v48's remaining LPF pipeline (`fixed_vert_lpf_gen` x
//! `fixed_frag_lpf_cpf`) translates into a fragment module whose own words
//! declare `OpCapability Float16` beside `OpCapability Int16` — the narrowing
//! shape its index arithmetic carries — and the rail's capability gate refused
//! both by name, so the pipeline could not even be translated
//! (`SPIR-V capability 9 requires a Vulkan feature outside the Phase 1
//! subset`). Vulkan's rule is the device's: `Float16` is admissible exactly on
//! a device created with `shaderFloat16`, `Int16` exactly with `shaderInt16`.
//!
//! What this file measures, on one device and with the fixture's own
//! arithmetic as the oracle:
//!
//! * the device's two readings travel apart and compose one policy
//!   (`VulkanExecutor::half_shader_support`), the capability snapshot publishes
//!   their conjunction as its own one bit
//!   (`ProviderCapabilities::supports_render_half_capabilities`), and the
//!   registration gate checks the module against that same policy;
//! * a module that declares the pair **translates, registers and executes** on
//!   a device that enabled both features, landing `40 80 c0 ff` per texel —
//!   byte for byte the module the rail executed before it could declare them;
//! * the frame is a function of the narrowing path rather than a constant: the
//!   twin module, identical except for the `i16` value its own comparison
//!   states, lands `ff 80 c0 00`, so a rail that lost the half conversion, the
//!   `i16` shift or that comparison cannot land the first module's frame;
//! * the phase-1 policy keeps the census's own sentence, which is what a device
//!   that enabled neither feature answers (pinned without a device in `lib.rs`'s
//!   `the_16_bit_pair_rides_the_device_policy_one_capability_at_a_time`, whose
//!   injected readings are the ones a device here may not have).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, SpirvFeaturePolicy, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The fixture's vertex stage: the milestone's `[[vertex_id]]` triangle, which
/// covers every pixel centre of the 2x2 attachment and binds no stream
/// (`VertexLayout::None`).
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// The module this increment is about: a float narrowed to `half`, the half's
/// bits read back through an `i16` shift, and the texel a function of that
/// path.
const FRAGMENT_AIR: &str = include_str!("fixtures/render_half_truncated_rgba8.frag.ll");
const FRAGMENT_ENTRY: &str = "render_half_truncated_rgba8";

/// The twin: the same module with a different `i16` value stated in its own
/// comparison, so the derived truncation no longer matches what the fixture's
/// arithmetic says.
const OFF_FRAGMENT_AIR: &str = include_str!("fixtures/render_half_truncated_off_rgba8.frag.ll");
const OFF_FRAGMENT_ENTRY: &str = "render_half_truncated_off_rgba8";

/// The declaring compute kernel (`research/docs/23` §3.6): the trace has to
/// declare the attachment view, and a compute pass that only reads it is the
/// sharing core admission admits.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The texel the module's own arithmetic lands: `half(64/255)`,
/// `half(128/255)` and `half(192/255)` round back inside the same 8-bit step as
/// the reviewed offscreen fixture's texel, which an `Rgba8Unorm` attachment
/// reads back as these four bytes.
const EXPECTED_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The twin's texel: its comparison states a truncation the module's own
/// arithmetic does not produce, so the red channel and the alpha are the
/// module's other arm (`1.0` and `0.0`) while the two channels the half round
/// trip still feeds keep their bytes.
const OFF_TEXEL: [u8; 4] = [0xff, 0x80, 0xc0, 0x00];

/// The `LoadOp::Clear` sentinel: a texel still holding it proves the draw did
/// not cover that pixel.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const ATTACHMENT_VIEW: ViewId = ViewId::new(1501);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(1502);
const SCRATCH_VIEW: ViewId = ViewId::new(1503);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(1504);

/// The sentence the census's own probe recorded for this module
/// (`evidence/texture-sampler-d94d8da-2026-09-20/11-pipe58-translatability-probe.log`).
const PHASE1_REFUSAL: &str =
    "SPIR-V capability 9 requires a Vulkan feature outside the Phase 1 subset";

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
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

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("render-half-capabilities-fixture-v1", case.to_vec()).expect("digest")
}

/// The contract the translated pair registers under: the AIR entries the
/// translations report, one `Rgba8Unorm` attachment and no vertex stream.
fn contract(fragment_entry: &str) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: Vec::new(),
    }
}

/// Translate the pair under the device's own policy and register it — the same
/// ask the provider repeats at registration, so a module this device did not
/// enable the features for cannot reach a pipeline.
fn register(
    executor: &Arc<VulkanExecutor>,
    provider: &VulkanComputeProvider,
    fragment_air: &str,
    fragment_entry: &str,
    case: &[u8],
) -> Result<CompiledComputePipeline, metal_api_core::provider::ProviderError> {
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
        .new_library_with_air(fragment_air)
        .expect("the fragment fixture loads");
    let function = library
        .function(fragment_entry)
        .expect("the fragment entry exists");
    let declared =
        TranslatedRenderStage::declared_shader_capabilities(RenderStage::Fragment, &function)
            .expect("the half fixture translates under the admitting policy");
    eprintln!(
        "declared fragment capabilities {fragment_entry}: float_controls2={} float16={} int16={} \
         half={}",
        declared.float_controls2(),
        declared.float16(),
        declared.int16(),
        declared.declares_half()
    );
    let fragment =
        TranslatedRenderStage::translate_with_policy(RenderStage::Fragment, &function, policy)
            .expect("the fragment stage translates under this device's policy");
    eprintln!(
        "translated fragment {fragment_entry}: {} bytes, {} render targets",
        fragment.spirv().len(),
        fragment.reflection().render_targets.len(),
    );
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(fragment_entry),
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
        .compile_pipeline(&function, digest(b"half-capabilities-declaring"))
        .expect("the compute pipeline registers")
}

/// One attachment of the 2x2 pass.
fn attachment() -> RenderAttachment {
    RenderAttachment {
        view_id: ATTACHMENT_VIEW,
        allocation_id: ATTACHMENT_ALLOCATION,
        format: AttachmentFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
        store: StoreOp::Store,
    }
}

fn render_pass(pipeline: PipelineId) -> RenderPassDescriptor {
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
        color_attachments: vec![attachment()],
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

/// The declaring pass: the reviewed `copy_word` kernel reads the attachment's
/// view and writes a scratch word of its own.
fn declaring_pass(pipeline: PipelineId) -> TracePass {
    TracePass::Compute(ComputePass {
        pipeline,
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
    })
}

/// Submit one trace and return the attachment's readback bytes.
fn submit_frame(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    what: &str,
) -> Vec<u8> {
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(42),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            declaring_pass(compute.pipeline_id),
            TracePass::Render(render_pass(render.pipeline_id)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [(ATTACHMENT_ALLOCATION, 16_u64), (SCRATCH_ALLOCATION, 8)] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace is admitted");
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
        .iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes.clone())
        .expect("the attachment has a writeback");
    eprintln!("{what} attachment readback: {}", hex(&frame));
    frame
}

/// The frame the fixture's own geometry states: a covering triangle, so every
/// texel is the fragment's texel — one 2x2 attachment, four identical texels.
fn expected_frame(texel: [u8; 4]) -> Vec<u8> {
    texel.repeat(4)
}

/// The device's own answers, the snapshot the frame publishes, the phase-1
/// sentence — then the frame itself and the twin's.
#[test]
fn a_module_that_declares_the_16_bit_pair_lands_the_texel_its_half_path_computes() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let support = executor.half_shader_support();
    let policy = provider.spirv_feature_policy();
    eprintln!(
        "device {}: shaderFloat16={} shaderInt16={} enabled={}",
        executor.device_name(),
        support.float16_reported(),
        support.int16_reported(),
        support.enabled()
    );
    assert_eq!(policy.float16(), support.float16_reported());
    assert_eq!(policy.int16(), support.int16_reported());
    assert_eq!(policy.half(), support.enabled());
    // The snapshot and the gate are one pair of readings: the frame's one bit
    // is the conjunction the device answered with.
    let capabilities = provider.capabilities();
    assert_eq!(
        capabilities.supports_render_half_capabilities,
        support.enabled(),
        "the capability snapshot publishes the device's own pair"
    );
    assert_eq!(
        capabilities.declares_render_half_capabilities(),
        support.enabled()
    );

    // The phase-1 policy keeps the census's sentence, on this device too.
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the half fixture loads");
    let function = library.function(FRAGMENT_ENTRY).expect("the entry exists");
    let phase1 = TranslatedRenderStage::translate_with_policy(
        RenderStage::Fragment,
        &function,
        SpirvFeaturePolicy::PHASE1,
    )
    .err()
    .expect("the phase-1 subset does not contain this module");
    assert_eq!(
        phase1.message(),
        PHASE1_REFUSAL,
        "the phase-1 arm is the census's own sentence"
    );
    // The device's own policy is the phase-1 answer exactly when the device did
    // not enable the pair: one condition, one reading.
    let device_translation =
        TranslatedRenderStage::translate_with_policy(RenderStage::Fragment, &function, policy);
    assert_eq!(
        device_translation.is_ok(),
        support.enabled(),
        "the device's policy admits the module exactly when it enabled both features"
    );
    if !support.enabled() {
        eprintln!(
            "this device did not enable the pair: {}",
            device_translation
                .err()
                .expect("the pair is not enabled")
                .message()
        );
        return;
    }

    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &executor,
        &provider,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        b"half-capabilities-fixture",
    )
    .expect("a module that declares the pair registers on this device");
    let frame = submit_frame(&provider, &compute, &pipeline, "half narrowing");
    assert_eq!(frame.len(), 16);
    assert_eq!(
        frame,
        expected_frame(EXPECTED_TEXEL),
        "the module's half path lands the attachment's own texel: {}",
        hex(&frame)
    );
    assert_ne!(
        frame,
        expected_frame(CLEAR_SENTINEL),
        "the draw covered the attachment"
    );

    // The twin module's comparison states a truncation the module's own
    // arithmetic does not produce, and the frame follows it: the red channel
    // and the alpha are the module's other arm while the two channels the half
    // round trip still feeds keep their bytes.
    let off = register(
        &executor,
        &provider,
        OFF_FRAGMENT_AIR,
        OFF_FRAGMENT_ENTRY,
        b"half-capabilities-off-fixture",
    )
    .expect("the twin registers under the same contract");
    let off_frame = submit_frame(&provider, &compute, &off, "half narrowing, twin");
    assert_eq!(
        off_frame,
        expected_frame(OFF_TEXEL),
        "the frame is a function of the derived i16, not a constant: {}",
        hex(&off_frame)
    );
    eprintln!(
        "16-bit shader pair agrees on [{}] and [{}] on {}",
        hex(&frame),
        hex(&off_frame),
        executor.device_name()
    );
}
