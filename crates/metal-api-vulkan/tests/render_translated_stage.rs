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
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
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

/// The word `copy_word` reads out of the attachment view's first four bytes.
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// `(64/255, 128/255, 192/255, 1)` as an 8-bit UNORM attachment stores it:
/// `40 80 c0 ff`, four texels of a 2x2 attachment.
const EXPECTED_RGBA8_TEXELS: [u8; 16] = [
    0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff, 0x40, 0x80, 0xc0, 0xff,
];

const ATTACHMENT_VIEW: ViewId = ViewId::new(901);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(902);
const SCRATCH_VIEW: ViewId = ViewId::new(903);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(904);

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

fn render_pass(pipeline: PipelineId) -> RenderPassDescriptor {
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
        present: None,
    }
}

/// One trace carrying the declaring compute pass and one render pass that names
/// `render`, plus the resource table it has to be admitted against.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
) -> (ComputeTrace, ResourceTableSnapshot) {
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
            TracePass::Render(render_pass(render.pipeline_id)),
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
    (trace, resources)
}

/// Submit one trace and return the attachment's readback bytes, printing them.
fn submit_for_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    what: &str,
) -> Vec<u8> {
    let (trace, resources) = trace_for(provider, compute, render);
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

    let reviewed_bytes = submit_for_readback(&provider, &compute, &reviewed, "reviewed");
    let translated_bytes = submit_for_readback(&provider, &compute, &translated, "translated");
    eprintln!("expected: [{}] x4", hex(&EXPECTED_RGBA8_TEXELS[..4]));
    assert_eq!(reviewed_bytes, EXPECTED_RGBA8_TEXELS);
    assert_eq!(translated_bytes, EXPECTED_RGBA8_TEXELS);
    assert_eq!(
        translated_bytes, reviewed_bytes,
        "the translated pair has to land byte for byte what the reviewed pair lands"
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
