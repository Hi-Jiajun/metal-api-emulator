//! The stage-buffer whole-binding arm, end to end (`research/docs/23` §3.3,
//! E-SB3).
//!
//! A translated module can reach into a `[[buffer(n)]]` argument at an index
//! the translator cannot express — a value read out of memory, a float→half→
//! bitfield chain, anything whose byte address is not an affine function of the
//! draw's own invocation indices. The reflection says so in the translator's own
//! words (`BufferFootprint::has_unbounded_access`): *the complete caller-provided
//! buffer window must remain available*. That is a statement about the binding
//! rather than about a byte count, and this increment is the arm that carries it
//! into the contract (`FootprintProof::BindingRange`).
//!
//! What this file measures:
//!
//! * the registration pairs the arm with the reflection's own "nothing states a
//!   reach" reading, and the pass pairing admits the declaration *without* a
//!   byte comparison — the pass binds the slot, the access agrees, and the
//!   provider executes the caller's window whole;
//! * the draw executes through the whole chain (`ComputeTrace` →
//!   `ProviderCapabilities::validate_trace` → `ComputeProvider::submit` → the
//!   attachment's bytes in the writebacks) and the frame follows *both* buffers:
//!   the selector's four bytes choose the table entry, so flipping either buffer
//!   moves the landed colour;
//! * the arm is not a widening of `Unbounded`: a declaration that states a
//!   static ceiling for a reach the reflection could not express is still
//!   refused by name, and so is a pass that leaves the declared slot unbound;
//! * the capability snapshot publishes the bit exactly where the device carries
//!   the reading the arm rests on (`robustBufferAccess`), and a device without it
//!   refuses the registration by name rather than executing the arm.

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
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The vertex stage of the pair: the milestone's full-screen triangle, which
/// covers every texel of the 2×2 render area and declares no `[[buffer(n)]]`
/// argument of its own — so the fragment stage below is the only half this file
/// has to watch.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// The fragment stage whose second argument is read at a data-dependent index:
/// `[[buffer(0)]]` carries the index, `[[buffer(1)]]` the table it selects
/// from.
const RANGE_AIR: &str = include_str!("fixtures/render_stage_buffer_range.frag.ll");
const RANGE_ENTRY: &str = "render_stage_buffer_range_rgba8";

/// The declaring compute pass's kernel: it is what puts the attachment view in
/// the trace's resource namespace (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(981);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(982);
const SCRATCH_VIEW: ViewId = ViewId::new(983);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(984);
const SELECTOR_VIEW: ViewId = ViewId::new(985);
const SELECTOR_ALLOCATION: AllocationId = AllocationId::new(986);
const TABLE_VIEW: ViewId = ViewId::new(987);
const TABLE_ALLOCATION: AllocationId = AllocationId::new(988);

/// The render area: 2×2 texels of four bytes each, every one of them covered by
/// the full-screen triangle.
const EXTENT: u32 = 2;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];

/// The table's four entries, as the four colours the attachment can land. Entry
/// 0 is the milestone's own `(64/255, 128/255, 192/255, 1)`, so the first
/// reading below is the same frame the reviewed pair lands.
const TABLE_COLOURS: [[u8; 4]; 4] = [
    [0x40, 0x80, 0xc0, 0xff],
    [0xff, 0x00, 0x00, 0xff],
    [0x00, 0xff, 0x00, 0xff],
    [0x00, 0x00, 0xff, 0xff],
];

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

/// One `[[buffer(1)]]` table: four `float4` entries, one per colour above.
fn table_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 * 16);
    for colour in TABLE_COLOURS {
        for channel in colour {
            bytes.extend_from_slice(&(f32::from(channel) / 255.0).to_le_bytes());
        }
    }
    bytes
}

/// The `[[buffer(0)]]` selector: one `u32`, the index the module reads.
fn selector_bytes(index: u32) -> Vec<u8> {
    index.to_le_bytes().to_vec()
}

/// The contract the pair registers under: the fragment stage's selector as a
/// static four-byte ceiling beside the whole-binding arm for the table it
/// indexes.
fn contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: vec![
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 4 },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 1,
                access: BufferAccess::Read,
                footprint: FootprintProof::BindingRange,
            },
        ],
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: RANGE_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: metal_api_core::provider::VertexLayout::None,
        textures: Vec::new(),
    }
}

/// The same contract with the table declared as a static ceiling instead: the
/// declaration the reflection cannot be paired with.
fn static_table_contract() -> RenderPipelineContract {
    let mut contract = contract();
    contract.stage_buffers[1].footprint = FootprintProof::Static { max_bytes: 64 };
    contract
}

/// Translate the pair through the rail's own entry point, the way a host
/// feeding guest AIR would.
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
    let library = device
        .new_library_with_air(RANGE_AIR)
        .expect("the range fixture loads");
    let function = library
        .function(RANGE_ENTRY)
        .expect("the range entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the range stage translates");
    (vertex, fragment)
}

/// The reflection the arm pairs with, printed so the reading is falsifiable
/// rather than assumed: `[[buffer(0)]]` states a four-byte static range and
/// `[[buffer(1)]]` states no reach at all.
fn print_range_reflection(fragment: &TranslatedRenderStage) {
    for binding in fragment
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
    {
        let footprint = binding
            .footprint
            .as_ref()
            .expect("the fixture's buffers carry footprints");
        eprintln!(
            "range fragment binding {}: access {:?} descriptor {:?} unbounded {} strided {} \
             static ranges {:?}",
            binding.metal_index,
            binding.access,
            binding.descriptor,
            footprint.has_unbounded_access,
            footprint.strided_accesses.len(),
            footprint.static_ranges,
        );
    }
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
        .compile_pipeline(
            &function,
            digest(b"render-stage-buffer-binding-range-compute"),
        )
        .expect("the compute pipeline registers")
}

fn stage_buffer_view(
    view_id: ViewId,
    allocation_id: AllocationId,
    metal_binding: u32,
    bytes: Vec<u8>,
) -> StageBufferView {
    StageBufferView {
        stage: RenderPipelineStage::Fragment,
        view: BufferView {
            view_id,
            metal_binding,
            allocation_id,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the view fits a u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        },
    }
}

/// One render pass over the 2×2 `Rgba8Unorm` attachment, with the fragment
/// stage's two slots bound to the caller's bytes.
fn render_pass(
    pipeline: PipelineId,
    selector: Option<Vec<u8>>,
    table: Option<Vec<u8>>,
) -> RenderPassDescriptor {
    let mut stage_buffers = Vec::new();
    if let Some(selector) = selector {
        stage_buffers.push(stage_buffer_view(
            SELECTOR_VIEW,
            SELECTOR_ALLOCATION,
            0,
            selector,
        ));
    }
    if let Some(table) = table {
        stage_buffers.push(stage_buffer_view(TABLE_VIEW, TABLE_ALLOCATION, 1, table));
    }
    RenderPassDescriptor {
        samplers: Vec::new(),
        stage_buffers,
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
    selector: Option<Vec<u8>>,
    table: Option<Vec<u8>>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(97),
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
            TracePass::Render(render_pass(render.pipeline_id, selector, table)),
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
    selector: Option<Vec<u8>>,
    table: Option<Vec<u8>>,
) -> Vec<u8> {
    let (trace, resources) = trace_for(provider, compute, render, selector, table);
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

/// The frame every texel of the render area carries when the table's entry `i`
/// is the one the selector names: the colour, quantised by the attachment's
/// format, repeated four times.
fn frame_of(index: usize) -> String {
    TABLE_COLOURS[index]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
        .repeat(4)
}

#[test]
fn the_whole_binding_arm_executes_and_the_frame_follows_both_buffers() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The device half first: the snapshot publishes the arm exactly where the
    // device carries the `robustBufferAccess` reading it rests on.
    let robust = executor.supports_robust_buffer_access();
    let published = provider
        .capabilities()
        .supports_render_stage_buffer_binding_range;
    eprintln!("device robustBufferAccess={robust} arm published={published}");
    assert_eq!(
        published, robust,
        "the capability snapshot publishes the arm exactly on the devices that carry the reading"
    );

    let (vertex, fragment) = translated_pair(&executor);
    print_range_reflection(&fragment);
    if !robust {
        // A device without robustness refuses the registration by name — the
        // arm is never executed where its own sentence about out-of-range
        // accesses would not hold.
        let refused = provider
            .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
                contract: contract(),
                vertex,
                fragment,
                logical_digest: digest(b"whole-binding-without-robustness"),
            })
            .expect_err("a device without robustness refuses the arm by name");
        eprintln!("refused without robustness: {refused:?}");
        assert_eq!(
            refused.slug,
            "render_stage_buffer_binding_range_unsupported"
        );
        return;
    }
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"whole-binding-arm"),
        })
        .expect("the whole-binding arm pairs with the reflection that states no reach");
    let compute = compile_declaring_kernel(&provider, &executor);

    let base = readback(
        &provider,
        &compute,
        &render,
        Some(selector_bytes(0)),
        Some(table_bytes()),
    );
    eprintln!("selector 0 frame: {}", hex(&base));
    assert_eq!(
        hex(&base),
        frame_of(0),
        "the selector's own bytes choose the table entry the attachment lands"
    );

    // The selector owns the choice: the same table, another index.
    let second = readback(
        &provider,
        &compute,
        &render,
        Some(selector_bytes(2)),
        Some(table_bytes()),
    );
    eprintln!("selector 2 frame: {}", hex(&second));
    assert_eq!(
        hex(&second),
        frame_of(2),
        "the index buffer's bytes move the frame"
    );
    assert_ne!(hex(&second), hex(&base));

    // The table owns the bytes: the same index, another payload behind it.
    let mut edited = table_bytes();
    let entry = 2 * 16;
    for channel in 0..4 {
        let value = f32::from(TABLE_COLOURS[1][channel]) / 255.0;
        edited[entry + channel * 4..entry + channel * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    let edited_frame = readback(
        &provider,
        &compute,
        &render,
        Some(selector_bytes(2)),
        Some(edited),
    );
    eprintln!("edited table frame: {}", hex(&edited_frame));
    assert_eq!(
        hex(&edited_frame),
        frame_of(1),
        "the table's own bytes are what the attachment lands"
    );
}

/// The arm is not a widening of `Unbounded`, and it is not a licence to drop a
/// binding (`research/docs/23` §3.3, E-SB3): a declaration that states a static
/// ceiling for the same reach is still refused by the pairing, and a pass that
/// leaves the declared slot unbound is still refused by the pair rules.
#[test]
fn the_arm_keeps_the_static_pairing_and_the_bound_slot_rules() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    if !executor.supports_robust_buffer_access() {
        eprintln!("SKIP: this device does not carry the reading the arm rests on");
        return;
    }
    let (vertex, fragment) = translated_pair(&executor);
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: static_table_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"static-ceiling-over-an-unstated-reach"),
        })
        .expect_err("a static ceiling is not what an unstated reach is paired with");
    eprintln!("static ceiling over an unstated reach: {refused:?}");
    assert_eq!(refused.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refused.fields.get("index"),
        Some(&metal_api_core::provider::FieldValue::Unsigned(1)),
        "the refusal names the table's own slot: {refused:?}"
    );

    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"whole-binding-arm-slot-rules"),
        })
        .expect("the arm registers");
    let compute = compile_declaring_kernel(&provider, &executor);
    // The table is declared and not bound: the pair rules refuse the pass by
    // name (`MissingStageBufferBinding`), exactly as they do for every other
    // footprint arm.
    let (trace, resources) = trace_for(&provider, &compute, &render, Some(selector_bytes(1)), None);
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the declared slot has no binding");
    eprintln!("unbound table slot: {refused:?}");
    assert_eq!(refused.slug, "trace_contract_invalid");
    assert!(
        refused
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("but the pass binds none there")),
        "the refusal names the missing binding: {refused:?}"
    );
}
