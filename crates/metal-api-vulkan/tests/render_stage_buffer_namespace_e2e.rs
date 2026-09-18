//! A translated render pair whose two stages read the *same* `[[buffer(0)]]`
//! index (`research/docs/23` §3.3, E-TX9).
//!
//! Metal's `setVertexBuffer(_:offset:index:)` and
//! `setFragmentBuffer(_:offset:index:)` name independent index spaces, so a
//! pair of translated stages that both read index 0 is two declarations of two
//! different slots that the translator's default descriptor layout folds into
//! one: every `[[buffer(n)]]` lands at `(set 0, binding n)`, and the rail
//! refuses the pair by name rather than letting one stage read the other's
//! bytes (`render_stage_buffer_layout_unsupported`).
//!
//! The increment under test publishes the arrangement such a pair is
//! executable in — [`stage_buffer_namespace_layout`], the vertex stage's whole
//! layout moved to set 1 — and this file is its evidence:
//!
//! * the translated pair executes through the whole chain (`ComputeTrace` →
//!   `ProviderCapabilities::validate_trace` → `ComputeProvider::submit` →
//!   the attachment's bytes in the writebacks);
//! * the frame is the *reviewed* `stage_buffer_borrowed_tint_2x2` reading
//!   (`conformance/suite-v31.json`, Apple device reading): the same position
//!   and tint bytes land `40 80 c0 ff` in the covered texel and the clear
//!   sentinel in the other three, so "each stage read its own bytes" is
//!   falsifiable per texel rather than asserted from the pass being accepted;
//! * each payload owns its own half of the frame — swapping the vertex bytes
//!   moves which texels are covered, swapping the tint bytes moves only the
//!   covered texel's colour;
//! * the same declaration translated the other way (the translator's default
//!   layout) is still refused by name, which is what makes the published layout
//!   the thing that made the frame above possible;
//! * the reviewed pair — the module pair whose Apple reading the frame is
//!   pinned to — executes beside the translated one on the same device and
//!   lands the same bytes, so the new shape neither replaced nor perturbed it.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FootprintProof, LoadOp,
    OperationId, PipelineId, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    RenderPipelineStage, ResourceTableSnapshot, SemanticDigest, StageBufferBinding,
    StageBufferView, StoreOp, TracePass, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    stage_buffer_namespace_layout, RenderPipelineRequest, RenderStage,
    TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The vertex stage that reads its clip positions out of `[[buffer(0)]]`
/// instead of computing them from `[[vertex_id]]`: one `float2` per vertex, the
/// exact affine reach the contract declares.
const POSITIONS_AIR: &str = include_str!("fixtures/render_stage_buffer_positions.vert.ll");
const POSITIONS_ENTRY: &str = "render_vertex_positions";

/// The fragment stage that reads its colour out of its own `[[buffer(0)]]`: one
/// `float4`, which the attachment stores through the format's quantisation and
/// nothing else. The two stages therefore name the *same* Metal index with
/// different bytes behind it — the shape this file is about.
const TINT_AIR: &str = include_str!("fixtures/render_stage_buffer_tint.frag.ll");
const TINT_ENTRY: &str = "render_stage_buffer_rgba8";

/// The reviewed module pair the translated one is measured against: the
/// `spirv-as` output of `render_spv/stage_buffer_positions.vert.spvasm` and
/// `render_spv/stage_buffer_tint.frag.spvasm`, whose Apple device reading
/// (`stage_buffer_borrowed_tint_2x2`) the frame below is pinned to.
const REVIEWED_VERTEX_SPV: &[u8] =
    include_bytes!("../src/render_spv/stage_buffer_positions.vert.spv");
const REVIEWED_FRAGMENT_SPV: &[u8] = include_bytes!("../src/render_spv/stage_buffer_tint.frag.spv");
const REVIEWED_VERTEX_ENTRY: &str = "stage_buffer_positions_main";
const REVIEWED_FRAGMENT_ENTRY: &str = "stage_buffer_tint_main";

/// The declaring compute pass's kernel: it is what puts the attachment view in
/// the trace's resource namespace (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(961);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(962);
const SCRATCH_VIEW: ViewId = ViewId::new(963);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(964);
const POSITIONS_VIEW: ViewId = ViewId::new(965);
const POSITIONS_ALLOCATION: AllocationId = AllocationId::new(966);
const TINT_VIEW: ViewId = ViewId::new(967);
const TINT_ALLOCATION: AllocationId = AllocationId::new(968);

/// The reviewed case's render area: 2×2 texels of four bytes each.
const EXTENT: u32 = 2;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];

/// The position bytes the reviewed `stage_buffer_borrowed_tint_2x2` case
/// carries (`conformance/suite-v31.json`): the three Metal-NDC `vec2`s
/// `(-0.9, 0.9)`, `(0.0, 0.9)`, `(-0.9, 0.0)`, which the rail's y flip lands
/// inside the top-left texel of a 2×2 render area.
const REVIEWED_POSITIONS_HEX: &str = "666666bf6666663f000000006666663f666666bf00000000";

/// The tint bytes the same case carries: `(64/255, 128/255, 192/255, 1)` as
/// four `f32`s, the payload whose quantisation is the frame's colour.
const REVIEWED_TINT_HEX: &str = "8180803e8180003fc1c0403f0000803f";

/// The frame the reviewed case's own `expected_hex` states: the covered texel
/// is the fragment stage's payload and the other three keep the clear sentinel.
const REVIEWED_FRAME_HEX: &str = "4080c0fffefefefefefefefefefefefe";

/// The same buffer's coverage control: the milestone's full-screen triangle in
/// Metal NDC, which covers every texel of the render area. A rail that read no
/// position bytes at all — or a fixture whose positions were replaced by the
/// milestone geometry — would fill all four texels, which is what this states.
const FULL_COVER_POSITIONS_HEX: &str = "000080bf000080bf00004040000080bf000080bf00004040";

fn unhex(literal: &str) -> Vec<u8> {
    literal
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("the fixture's hex is ASCII");
            u8::from_str_radix(text, 16).expect("the fixture's hex is well formed")
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest")
}

fn executor_and_provider() -> Option<(Arc<VulkanExecutor>, VulkanComputeProvider)> {
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

/// The contract both stages of the folded pair register under: the vertex
/// stage's affine `[[buffer(0)]]` reach (`0 / 4 + vertex_id * 8`, the shape
/// [`POSITIONS_AIR`] states) and the fragment stage's one 16-byte `float4` —
/// two declarations, one Metal index each.
fn contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: vec![
            StageBufferBinding {
                stage: RenderPipelineStage::Vertex,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Affine {
                    accesses: vec![
                        metal_api_core::provider::AffineAccess {
                            base_offset: 0,
                            access_size: 4,
                            terms: vec![metal_api_core::provider::AffineTerm {
                                axis: 0,
                                stride: 8,
                            }],
                        },
                        metal_api_core::provider::AffineAccess {
                            base_offset: 4,
                            access_size: 4,
                            terms: vec![metal_api_core::provider::AffineTerm {
                                axis: 0,
                                stride: 8,
                            }],
                        },
                    ],
                },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 16 },
            },
        ],
        vertex_entry: POSITIONS_ENTRY.to_owned(),
        fragment_entry: TINT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: metal_api_core::provider::VertexLayout::None,
        textures: Vec::new(),
    }
}

/// The reviewed pair's contract: the same two slots, declared with the static
/// ceilings the reviewed modules' own arguments state (`stage_buffer_positions`
/// reads 24 bytes, `stage_buffer_tint` 16).
fn reviewed_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: vec![
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
        ],
        vertex_entry: REVIEWED_VERTEX_ENTRY.to_owned(),
        fragment_entry: REVIEWED_FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: metal_api_core::provider::VertexLayout::None,
        textures: Vec::new(),
    }
}

/// Translate the folded pair the two ways this file measures: the vertex stage
/// under `vertex_layout`, the fragment stage under the translator's default.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    vertex_layout: metal2vulkan::reflect::DescriptorLayout,
) -> (TranslatedRenderStage, TranslatedRenderStage) {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(POSITIONS_AIR)
        .expect("the positions fixture loads");
    let function = library
        .function(POSITIONS_ENTRY)
        .expect("the positions entry exists");
    let vertex = TranslatedRenderStage::translate_with_policy_and_layout(
        RenderStage::Vertex,
        &function,
        executor.spirv_feature_policy(),
        vertex_layout,
    )
    .expect("the positions stage translates");
    let library = device
        .new_library_with_air(TINT_AIR)
        .expect("the tint fixture loads");
    let function = library.function(TINT_ENTRY).expect("the tint entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the tint stage translates");
    (vertex, fragment)
}

/// The slots a stage's reflection names for its own buffers.
fn buffer_slots(stage: &TranslatedRenderStage) -> Vec<(u32, u32)> {
    stage
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
        .filter_map(|binding| {
            binding
                .descriptor
                .map(|descriptor| (descriptor.set, descriptor.binding))
        })
        .collect()
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
        .compile_pipeline(&function, digest(b"render-stage-buffer-namespace-compute"))
        .expect("the compute pipeline registers")
}

/// One stage buffer's view: the caller's bytes, read-only, at the stage's own
/// slot.
fn stage_buffer_view(
    stage: RenderPipelineStage,
    view_id: ViewId,
    allocation_id: AllocationId,
    bytes: Vec<u8>,
) -> StageBufferView {
    StageBufferView {
        stage,
        view: BufferView {
            view_id,
            metal_binding: 0,
            allocation_id,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the view fits a u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        },
    }
}

/// One render pass over the reviewed 2×2 `Rgba8Unorm` attachment, with the two
/// stages' slots bound to the caller's bytes.
fn render_pass(pipeline: PipelineId, positions: Vec<u8>, tint: Vec<u8>) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
        stage_buffers: vec![
            stage_buffer_view(
                RenderPipelineStage::Vertex,
                POSITIONS_VIEW,
                POSITIONS_ALLOCATION,
                positions,
            ),
            stage_buffer_view(
                RenderPipelineStage::Fragment,
                TINT_VIEW,
                TINT_ALLOCATION,
                tint,
            ),
        ],
        blend: None,
        cull: None,
        depth: None,
        depth_test: None,
        depth_resolve: None,
        multisample: None,
        stencil: None,
        stencil_test: None,
        stencil_resolve: None,
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
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// The trace the pass runs in: a declaring compute pass that carries the
/// attachment view, then the render pass.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: Vec<u8>,
    tint: Vec<u8>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(43),
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
                        source: BufferSource::OwnedBytes(vec![0x00; attachment_bytes as usize]),
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
            TracePass::Render(render_pass(render.pipeline_id, positions, tint)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, attachment_bytes),
        (SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("the declaration's allocation");
    }
    (trace, resources)
}

/// Submit one trace through `render` and return the attachment's readback.
fn readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    positions: Vec<u8>,
    tint: Vec<u8>,
) -> Vec<u8> {
    let (trace, resources) = trace_for(provider, compute, render, positions, tint);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace's declarations are the contract's own");
    let submitted = provider.submit(admitted).expect("the pass executes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback")
}

/// Register the translated folded pair under `vertex_layout`.
fn register_folded_pair(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    vertex_layout: metal2vulkan::reflect::DescriptorLayout,
    what: &str,
) -> CompiledComputePipeline {
    let (vertex, fragment) = translated_pair(executor, vertex_layout);
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(what.as_bytes()),
        })
        .expect("each stage's own pairing holds")
}

#[test]
fn the_folded_pair_executes_and_lands_the_reviewed_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    // The slots the two stages were translated into are separate: the vertex
    // stage's `[[buffer(0)]]` sits in the published set, the fragment stage's in
    // the translator's own.
    let (vertex, fragment) = translated_pair(&executor, stage_buffer_namespace_layout());
    eprintln!(
        "translated slots: vertex={:?} fragment={:?}",
        buffer_slots(&vertex),
        buffer_slots(&fragment)
    );
    assert_ne!(
        buffer_slots(&vertex),
        buffer_slots(&fragment),
        "the two stages' Metal index spaces stay apart"
    );
    let render = register_folded_pair(
        &provider,
        &executor,
        stage_buffer_namespace_layout(),
        "folded-pair-reviewed-frame",
    );
    let frame = readback(
        &provider,
        &compute,
        &render,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    eprintln!("translated folded frame: {}", hex(&frame));
    assert_eq!(
        hex(&frame),
        REVIEWED_FRAME_HEX,
        "the translated pair lands the reviewed case's own frame"
    );
}

#[test]
fn each_payload_owns_its_half_of_the_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_folded_pair(
        &provider,
        &executor,
        stage_buffer_namespace_layout(),
        "folded-pair-payload-controls",
    );
    let baseline = readback(
        &provider,
        &compute,
        &render,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    // Control one: swapping the vertex payload covers the whole render area and
    // leaves the fragment payload's colour alone.
    let covered = readback(
        &provider,
        &compute,
        &render,
        unhex(FULL_COVER_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    // Control two: swapping the fragment payload moves only the covered texel's
    // colour, and the vertex payload still decides where that texel is.
    let green = readback(
        &provider,
        &compute,
        &render,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex("000000000000803f000000000000803f"),
    );
    eprintln!(
        "folded frame controls: baseline {} covered {} green {}",
        hex(&baseline),
        hex(&covered),
        hex(&green)
    );
    assert_eq!(hex(&baseline), REVIEWED_FRAME_HEX);
    let covered_texel = unhex(REVIEWED_FRAME_HEX)[..4].repeat(EXTENT as usize * EXTENT as usize);
    assert_eq!(
        covered, covered_texel,
        "the vertex payload moves the coverage"
    );
    assert_eq!(
        &green[..4],
        &[0x00, 0xff, 0x00, 0xff],
        "the fragment payload owns the covered texel's colour"
    );
    assert_eq!(
        &green[4..],
        &baseline[4..],
        "the fragment payload moves nothing outside the covered texel"
    );
}

#[test]
fn the_same_pair_under_the_default_layout_is_still_refused() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    // The same declaration, both stages translated the way the translator's
    // default layout puts them: one `(set, binding)` for two index spaces.
    let render = register_folded_pair(
        &provider,
        &executor,
        metal2vulkan::reflect::DescriptorLayout::default(),
        "folded-pair-default-layout",
    );
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    let admitted = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect("the trace's declarations are the contract's own");
    let refused = provider
        .submit(admitted)
        .expect_err("the two stages' slots fold onto one descriptor");
    eprintln!("folded pair under the default layout refused: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_layout_unsupported");
    let detail = refused
        .detail
        .as_deref()
        .expect("the refusal carries a detail");
    assert!(
        detail.contains("stage_buffer_namespace_layout"),
        "the refusal names the published layout: {detail}"
    );
}

/// The reviewed pair — the modules whose Apple reading the frame above is
/// pinned to — runs beside the translated pair on one device and lands the same
/// bytes, so the new shape neither replaced the reviewed one nor perturbed it
/// (`research/docs/23` §3.3, E-TX9; §92).
#[test]
fn the_folded_translation_and_the_reviewed_pair_share_one_device() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let reviewed = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: reviewed_contract(),
            vertex_spirv: REVIEWED_VERTEX_SPV.to_vec(),
            fragment_spirv: REVIEWED_FRAGMENT_SPV.to_vec(),
            logical_digest: digest(b"folded-pair-reviewed-modules"),
        })
        .expect("the reviewed pair registers");
    let translated = register_folded_pair(
        &provider,
        &executor,
        stage_buffer_namespace_layout(),
        "folded-pair-beside-the-reviewed-pair",
    );
    let reviewed_frame = readback(
        &provider,
        &compute,
        &reviewed,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    let translated_frame = readback(
        &provider,
        &compute,
        &translated,
        unhex(REVIEWED_POSITIONS_HEX),
        unhex(REVIEWED_TINT_HEX),
    );
    eprintln!(
        "one device, two registrations: reviewed {} translated {}",
        hex(&reviewed_frame),
        hex(&translated_frame)
    );
    assert_eq!(hex(&reviewed_frame), REVIEWED_FRAME_HEX);
    assert_eq!(reviewed_frame, translated_frame);
}
