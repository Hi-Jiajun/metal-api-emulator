//! The superset fragment interface (2026-09-20, the third door behind census
//! v46's `stage_buffer_footprint` bucket).
//!
//! Census v46's remaining `stage_buffer_footprint` records are one LPF
//! pipeline (`fixed_vert_lpf_gen` × `fixed_frag_lpf_cpf`) whose fragment stage
//! stores three colour locations while the draw attaches one. The rail
//! executes that shape — the pipeline is built from the *contract's* own
//! `color_formats`, and a Vulkan fragment store whose location has no
//! attachment behind it is discarded rather than refused — but the
//! registration gate used to hold the declared list and the reflection to the
//! same count, so the shape came back a provider refusal the class never got
//! to answer (the census red line `draws_skipped_after_engine_refusal`).
//!
//! The readings this file pins, all on one device and with the fixture's own
//! definitional frame as the oracle:
//!
//! * a registration whose fragment module **stores two locations** under a
//!   contract that attaches **one** lands the attached location's texel in
//!   every covered pixel — `40 80 c0 ff`, the reviewed offscreen fixture's own
//!   texel, so the two modules agree on the attachment they share;
//! * the twin module, identical except for the *dropped* location's texel,
//!   lands **the same four bytes per pixel**: the extra store is discarded, not
//!   folded in and not bound to the attached location;
//! * the same module under a contract that attaches **both** locations does
//!   store the second one, which is what makes "discarded" a statement about
//!   the attachment list rather than about the module;
//! * the object rail lands the same frame byte for byte;
//! * the attached locations keep the shape rule they always had — a
//!   `float4` store under an `R32Float` attachment is still refused by name —
//!   and the reverse direction (an attachment the module never stores) keeps
//!   its count refusal.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The fixture's vertex stage: the milestone's `[[vertex_id]]` triangle, which
/// covers every pixel centre of the 2x2 attachment and binds no stream
/// (`VertexLayout::None`).
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// The module this increment is about: two unconditional `Location` stores, the
/// second with no attachment behind it.
const FRAGMENT_AIR: &str = include_str!("fixtures/render_two_output_rgba8.frag.ll");
const FRAGMENT_ENTRY: &str = "render_two_output_rgba8";

/// The twin: the same module with the dropped location's texel changed.
const ALT_FRAGMENT_AIR: &str = include_str!("fixtures/render_two_output_rgba8_alt.frag.ll");
const ALT_FRAGMENT_ENTRY: &str = "render_two_output_rgba8_alt";

/// The reviewed single-output module, used as the reverse direction's
/// counterexample: one store, two attached locations.
const SINGLE_FRAGMENT_AIR: &str = include_str!("fixtures/render_offscreen_2x2.frag.ll");
const SINGLE_FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// The declaring compute kernel (`research/docs/23` §3.6): the trace has to
/// declare the attachment view, and a compute pass that only reads it is the
/// sharing core admission admits.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// `(64/255, 128/255, 192/255, 1)` as an 8-bit UNORM attachment stores it: the
/// fragment fixture's own Location 0 texel, and the reviewed module's.
const EXPECTED_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The Location 1 texel of [`FRAGMENT_AIR`] — the one a single-attachment
/// registration has no destination for.
const DROPPED_TEXEL: [u8; 4] = [0xff, 0x00, 0x00, 0xff];

/// The `LoadOp::Clear` sentinel: a texel still holding it proves the draw did
/// not cover that pixel.
const CLEAR_SENTINEL: [u8; 4] = [0xfe; 4];

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const ATTACHMENT_VIEW: ViewId = ViewId::new(1401);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(1402);
const SECOND_VIEW: ViewId = ViewId::new(1403);
const SECOND_ALLOCATION: AllocationId = AllocationId::new(1404);
const SCRATCH_VIEW: ViewId = ViewId::new(1405);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(1406);
const SECOND_SCRATCH_VIEW: ViewId = ViewId::new(1407);
const SECOND_SCRATCH_ALLOCATION: AllocationId = AllocationId::new(1408);
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("render-fragment-output-superset-fixture-v1", case.to_vec())
        .expect("digest")
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

/// The contract the class states for this draw: the attached lanes only.
///
/// The list is exactly the pass's own attachment list — one entry per attached
/// location — which is what the increment makes registerable beside a module
/// that declares more. Nothing about it mentions the module's extra locations.
fn contract(
    formats: Vec<AttachmentFormat>,
    fragment_entry: &str,
    vertex_layout: VertexLayout,
) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        color_formats: formats,
        vertex_layout,
        textures: Vec::new(),
    }
}

/// The provider-path contract: the milestone's layout-free pass.
fn layout_free_contract(
    formats: Vec<AttachmentFormat>,
    fragment_entry: &str,
) -> RenderPipelineContract {
    contract(formats, fragment_entry, VertexLayout::None)
}

/// Register a fixture pair under `formats`, translating both stages through the
/// rail's own entry point the way a host feeding guest AIR would.
fn register(
    executor: &Arc<VulkanExecutor>,
    provider: &VulkanComputeProvider,
    build_contract: &dyn Fn(Vec<AttachmentFormat>, &str) -> RenderPipelineContract,
    formats: Vec<AttachmentFormat>,
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
    let fragment =
        TranslatedRenderStage::translate_with_policy(RenderStage::Fragment, &function, policy)
            .expect("the fragment stage translates");
    eprintln!(
        "translated fragment {fragment_entry}: {} bytes, {} render targets {:?}",
        fragment.spirv().len(),
        fragment.reflection().render_targets.len(),
        fragment
            .reflection()
            .render_targets
            .iter()
            .map(|target| (target.location, target.type_name.clone()))
            .collect::<Vec<_>>(),
    );
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: build_contract(formats, fragment_entry),
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
        .compile_pipeline(&function, digest(b"fragment-output-superset-declaring"))
        .expect("the compute pipeline registers")
}

/// One attachment of the 2x2 pass.
fn attachment(view_id: ViewId, allocation_id: AllocationId) -> RenderAttachment {
    RenderAttachment {
        view_id,
        allocation_id,
        format: AttachmentFormat::Rgba8Unorm,
        width: 2,
        height: 2,
        load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
        store: StoreOp::Store,
    }
}

fn render_pass(pipeline: PipelineId, attachments: Vec<RenderAttachment>) -> RenderPassDescriptor {
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
        color_attachments: attachments,
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

/// The declaring pass for one attachment: the reviewed `copy_word` kernel
/// reads the attachment's view and writes a scratch word of its own. The
/// compute pass is what declares the attachment view inside the trace, so a
/// two-attachment pass carries one declaring pass per attachment.
fn declaring_pass(
    pipeline: PipelineId,
    view_id: ViewId,
    allocation_id: AllocationId,
    scratch_id: ViewId,
    scratch_allocation: AllocationId,
) -> TracePass {
    TracePass::Compute(ComputePass {
        pipeline,
        buffers: vec![
            BufferView {
                view_id,
                metal_binding: 0,
                allocation_id,
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(ATTACHMENT_WORD.repeat(4)),
            },
            BufferView {
                view_id: scratch_id,
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
    })
}

/// One trace carrying one declaring compute pass per attachment and one render
/// pass that attaches `attachments`.
fn trace(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    attachments: Vec<RenderAttachment>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let pass = render_pass(render.pipeline_id, attachments.clone());
    let mut allocations = vec![(ATTACHMENT_ALLOCATION, 16_u64), (SCRATCH_ALLOCATION, 8)];
    let mut passes = vec![declaring_pass(
        compute.pipeline_id,
        ATTACHMENT_VIEW,
        ATTACHMENT_ALLOCATION,
        SCRATCH_VIEW,
        SCRATCH_ALLOCATION,
    )];
    for extra in &attachments {
        if extra.view_id == ATTACHMENT_VIEW {
            continue;
        }
        passes.push(declaring_pass(
            compute.pipeline_id,
            extra.view_id,
            extra.allocation_id,
            SECOND_SCRATCH_VIEW,
            SECOND_SCRATCH_ALLOCATION,
        ));
        allocations.push((extra.allocation_id, 16));
        allocations.push((SECOND_SCRATCH_ALLOCATION, 8));
    }
    passes.push(TracePass::Render(pass));
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(41),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes,
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in allocations {
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

/// Submit one trace and return every colour attachment's readback bytes, in the
/// pass's own attachment order.
fn submit_frame(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    attachments: Vec<RenderAttachment>,
    what: &str,
) -> Vec<Vec<u8>> {
    let expected: Vec<ViewId> = attachments.iter().map(|item| item.view_id).collect();
    let (trace, resources) = trace(provider, compute, render, attachments);
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
    let frames = expected
        .iter()
        .map(|view_id| {
            submitted
                .writebacks
                .iter()
                .find(|writeback| writeback.view_id == *view_id)
                .map(|writeback| writeback.bytes.clone())
                .expect("the attachment has a writeback")
        })
        .collect::<Vec<_>>();
    eprintln!(
        "{what} attachment readbacks: {}",
        frames
            .iter()
            .map(|frame| hex(frame))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    frames
}

/// The frame the fixture's own geometry states: a covering triangle, so every
/// texel is the fragment's texel — one 2x2 attachment, four identical texels.
fn expected_frame(texel: [u8; 4]) -> Vec<u8> {
    texel.repeat(4)
}

/// The object rail's frame over the same registration: one buffer holding the
/// attachment, and the same pipeline wrapped for the object API. The pass is
/// layout-free, so it records through `draw_render_pass` — the entry that binds
/// no stream, exactly as the milestone's `vertex_id` shape does.
fn object_frame(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
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
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"fragment-output-superset-object-declaring"),
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
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
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

/// The increment's own reading: the attached location lands its texel, the
/// dropped location's bytes do not move the frame, and the store that is
/// dropped under one attachment really is a store.
#[test]
fn a_module_that_declares_two_locations_lands_the_attached_one_and_drops_the_other() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let two_output = register(
        &executor,
        &provider,
        &layout_free_contract,
        vec![AttachmentFormat::Rgba8Unorm],
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        b"fragment-output-superset-two-outputs",
    )
    .expect("a module that stores more locations than the contract attaches registers");
    let twin = register(
        &executor,
        &provider,
        &layout_free_contract,
        vec![AttachmentFormat::Rgba8Unorm],
        ALT_FRAGMENT_AIR,
        ALT_FRAGMENT_ENTRY,
        b"fragment-output-superset-twin",
    )
    .expect("the twin registers under the same contract");
    let both_attached = register(
        &executor,
        &provider,
        &layout_free_contract,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        b"fragment-output-superset-both-attached",
    )
    .expect("the same module registers under the two-attachment contract");

    let one = vec![attachment(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION)];
    let two = vec![
        attachment(ATTACHMENT_VIEW, ATTACHMENT_ALLOCATION),
        attachment(SECOND_VIEW, SECOND_ALLOCATION),
    ];

    let frame = submit_frame(
        &provider,
        &compute,
        &two_output,
        one.clone(),
        "one attached",
    );
    assert_eq!(frame.len(), 1);
    assert_eq!(
        frame[0],
        expected_frame(EXPECTED_TEXEL),
        "the attached location's texel is the frame: {}",
        hex(&frame[0])
    );

    let twin_frame = submit_frame(&provider, &compute, &twin, one, "twin, one attached");
    assert_eq!(
        frame[0], twin_frame[0],
        "the dropped location's texel must not move the frame"
    );

    let both = submit_frame(&provider, &compute, &both_attached, two, "both attached");
    assert_eq!(both.len(), 2);
    assert_eq!(
        both[0],
        expected_frame(EXPECTED_TEXEL),
        "the first attachment keeps the Location 0 texel"
    );
    assert_eq!(
        both[1],
        expected_frame(DROPPED_TEXEL),
        "the second attachment is the very store the single-attachment arm drops: {}",
        hex(&both[1])
    );

    let object = object_frame(&provider, &two_output);
    assert_eq!(
        object, frame[0],
        "the object rail lands the same frame byte for byte"
    );
    eprintln!(
        "fragment output superset agrees on [{}] on {}",
        hex(&frame[0]),
        executor.device_name()
    );
}

/// The reverse direction keeps its refusal: a contract that attaches a location
/// the module never stores would read back bytes nothing wrote.
#[test]
fn a_contract_that_attaches_a_location_the_module_never_stores_is_still_refused() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let refused = register(
        &executor,
        &provider,
        &layout_free_contract,
        vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
        SINGLE_FRAGMENT_AIR,
        SINGLE_FRAGMENT_ENTRY,
        b"fragment-output-superset-reverse",
    )
    .expect_err("one stored location cannot describe two attachments");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
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

/// The widened count rule does not widen the *attached* locations' shape rule:
/// every attached position still has to carry the component shape its format
/// stores.
#[test]
fn the_attached_locations_shape_rule_is_not_widened_by_the_extra_store() {
    let Some((executor, provider)) = provider_with_device() else {
        return;
    };
    let refused = register(
        &executor,
        &provider,
        &layout_free_contract,
        vec![AttachmentFormat::R32Float],
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        b"fragment-output-superset-shape",
    )
    .expect_err("a float4 store is not the component shape an R32Float attachment reads");
    eprintln!("refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("render_targets".to_owned()))
    );
    assert_eq!(
        refused.fields.get("location"),
        Some(&FieldValue::Unsigned(0))
    );
    assert_eq!(
        refused.fields.get("type_name"),
        Some(&FieldValue::Text("float4".to_owned()))
    );
}

/// The registration is the rail's own gate, and the capability snapshot says
/// the same thing the gate does: the two are one declaration, so a device that
/// answers for the shape is a device whose provider registers it.
#[test]
fn the_capability_snapshot_declares_the_shape_the_gate_registers() {
    let Some((_executor, provider)) = provider_with_device() else {
        return;
    };
    let capabilities = provider.capabilities();
    assert!(
        capabilities.supports_render_fragment_output_superset,
        "the Vulkan rail registers the superset fragment interface"
    );
    assert!(capabilities.declares_render_fragment_output_superset_support());
    // The gate's own arm (`render::EXECUTES_FRAGMENT_OUTPUT_SUPERSET`) is read
    // against this same snapshot inside the crate (`provider.rs`'s capability
    // test), because the constant is the rail's private answer rather than API.
}
