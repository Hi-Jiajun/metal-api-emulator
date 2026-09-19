//! A fragment stage that declares thirteen sampled textures (E-TC1).
//!
//! The falsifiable claim is the census's own shape: the preview round behind
//! this increment read 42 class exits under
//! `render_provider_out_of_class_texture_count`, and every one of them was a
//! fragment stage declaring **thirteen** sampled textures while the canonical
//! contract stated eight for the pass's whole list
//! (`/mnt/c/tmp/reims-vgpu-fail.log`). The contract now states one *stage's*
//! window — sixteen, Vulkan's own per-stage floor — and the pass's list is the
//! pair's sum beside it, so this file pins the three readings that follow:
//!
//! * the thirteen-declaration shape **executes** — the module's thirteen
//!   `[[texture(n)]]` arguments reach the provider, its descriptor set carries
//!   thirteen combined image samplers, and the frame it lands is the thirteen
//!   uploaded red channels summed (`0x5b`), with the first and the last
//!   argument pinned in the channels beside it (`0x01`, `0x0d`);
//! * the frame follows the **uploaded texels**, not the table's shape: moving
//!   one texture's bytes — a middle argument and the last one — moves the frame
//!   both times, and the same payload twice lands the same bytes;
//! * the device's window is what **admits** the shape, and a narrower window is
//!   a refusal by name: the snapshot states the per-stage window beside the
//!   list bound, the thirteen enter the rail, and a twelve-slot window answers
//!   `render_texture_limit` with the stage, the count and the window on it.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, ProviderError, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
    StoreOp, TextureAccess, TextureBindingContract, TextureFootprintProof, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The shape this increment answers: thirteen sampled `texture2d<float,
/// sample>` arguments, one per Metal index 0 through 12.
const THIRTEEN_AIR: &str = include_str!("fixtures/render_sample_thirteen_textures.frag.ll");
const THIRTEEN_ENTRY: &str = "render_sample_thirteen_textures";
/// The number of declarations the fixture states, and the number the census's
/// own rows carry.
const TEXTURE_COUNT: u32 = 13;

const ATTACHMENT_VIEW: ViewId = ViewId::new(1200);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(1201);
const SCRATCH_VIEW: ViewId = ViewId::new(1202);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(1203);
/// The thirteen sampled textures' identities: one view and one allocation per
/// Metal index, because a pass that samples one view twice is the
/// duplicate-identity refusal rather than the shape this file measures.
const TEXTURE_VIEW_BASE: u64 = 1210;
const TEXTURE_ALLOCATION_BASE: u64 = 1250;

/// 4x4, the extent of both the attachment and every sampled texture: the rail's
/// reviewed window requires the two to agree, so the stage's fixed-coordinate
/// samples stand on texel centres of both.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// The frame the thirteen-declaration module lands when every texture carries
/// its own payload: `red` is `1 + 2 + ... + 13 = 91`, `green` is texture 0's
/// reading and `blue` is texture 12's, so the first and the last argument of
/// the table are pinned beside the sum.
const THIRTEEN_FRAME: [u8; 4] = [0x5b, 0x01, 0x0d, 0xff];

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

/// Texture `k`'s sixteen texels. Texel `(1, 0)` — the one the module samples —
/// holds `k + 1` in red, and the other texels repeat it, so the reading is the
/// texture's own payload rather than a texel the coordinate picked.
fn texture_bytes(index: u32) -> Vec<u8> {
    let red = u8::try_from(index + 1).expect("thirteen payloads fit one byte");
    let mut bytes = Vec::with_capacity(64);
    for _ in 0..(EXTENT * EXTENT) {
        bytes.extend_from_slice(&[red, 0x00, 0x00, 0xff]);
    }
    bytes
}

/// One sampled texture bound at `index`, carrying texture `index`'s payload.
fn sampled_texture_view(index: u32, payload: Option<Vec<u8>>) -> TextureView {
    TextureView {
        view_id: ViewId::new(TEXTURE_VIEW_BASE + u64::from(index)),
        metal_binding: index,
        allocation_id: AllocationId::new(TEXTURE_ALLOCATION_BASE + u64::from(index)),
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(payload.unwrap_or_else(|| texture_bytes(index))),
    }
}

/// The thirteen declarations the module's own argument table states, each
/// carrying the Nearest + ClampToEdge state the fixture's AIR sampler holds.
fn contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: THIRTEEN_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: (0..TEXTURE_COUNT)
            .map(|index| TextureBindingContract {
                metal_binding: index,
                access: TextureAccess::Sampled,
                texture_type: TextureType::D2,
                format: TextureFormat::Rgba8Unorm,
                sampler: Some(NEAREST_CLAMP),
                runtime_sampler: None,
                footprint: TextureFootprintProof::WholeView,
            })
            .collect(),
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would.
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
        .new_library_with_air(THIRTEEN_AIR)
        .expect("the thirteen-texture fragment fixture loads");
    let function = library
        .function(THIRTEEN_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    // The reflection has to name the thirteen textures the fixture declares,
    // one at each Metal index: the rail's slots come from this walk, so a
    // fixture that failed to declare an argument would be a smaller shape than
    // the one this file measures.
    let texture_indexes = fragment
        .reflection()
        .bindings
        .iter()
        .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Texture)
        .map(|binding| binding.metal_index)
        .collect::<Vec<_>>();
    eprintln!(
        "translated fragment: {} bytes, reflection entry {:?}, texture bindings {texture_indexes:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
    );
    assert_eq!(
        texture_indexes,
        (0..TEXTURE_COUNT).collect::<Vec<_>>(),
        "the module declares one sampled texture per Metal index"
    );
    (vertex, fragment)
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
        .compile_pipeline(&function, digest(b"render-texture-count-compute"))
        .expect("the compute pipeline registers")
}

fn render_pass(pipeline: PipelineId, textures: Vec<TextureView>) -> RenderPassDescriptor {
    RenderPassDescriptor {
        stage_buffers: Vec::new(),
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
        textures,
        samplers: Vec::new(),
        present: None,
    }
}

fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(44),
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
            TracePass::Render(render_pass(render.pipeline_id, textures)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation_id, size) in [
        (ATTACHMENT_ALLOCATION, attachment_bytes),
        (SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("allocation");
    }
    (trace, resources)
}

/// Admit one pass through the registered pipeline and return its attachment's
/// readback bytes.
fn admit_and_submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, textures);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)?;
    let submitted = provider.submit(admitted)?;
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
    Ok(bytes)
}

/// One texel of the readback: the stage's samples stand on fixed coordinates,
/// so every texel of the 4x4 attachment carries the same colour.
fn uniform_texel(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(
        bytes.len() as u64,
        u64::from(EXTENT) * u64::from(EXTENT) * 4
    );
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(
            chunk,
            texel,
            "every fragment samples the same coordinates, so every texel has to carry the same \
             colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// The thirteen-declaration shape reaches the provider and lands its bytes.
///
/// The frame's three colour channels are one reading each of the table's own
/// shape: the sum over all thirteen arguments (a dropped contribution would
/// move it by at least one unorm step), the first argument's own payload and
/// the last one's.
#[test]
fn thirteen_sampled_textures_enter_the_rail_and_land_their_bytes() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"thirteen sampled textures"),
        })
        .expect("the thirteen-declaration registration is well formed");
    let bytes = admit_and_submit(
        &provider,
        &compute,
        &render,
        (0..TEXTURE_COUNT)
            .map(|index| sampled_texture_view(index, None))
            .collect(),
    )
    .expect("the thirteen-texture pass executes");
    let texel = uniform_texel(&bytes);
    eprintln!(
        "thirteen sampled textures: {} ({} texels)",
        hex(&texel),
        bytes.len() / 4
    );
    assert_eq!(texel, THIRTEEN_FRAME);
}

/// The frame is the thirteen uploaded payloads, and every argument of the table
/// is one of them: moving a middle argument's bytes and the last argument's
/// bytes moves the frame both times, and the same payloads land the same bytes.
#[test]
fn the_thirteen_texture_frame_follows_the_uploaded_texels() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"thirteen texture payloads"),
        })
        .expect("the registration is well formed");
    let payloads = |moved: Option<u32>| {
        (0..TEXTURE_COUNT)
            .map(|index| {
                if moved == Some(index) {
                    // The moved texture keeps its shape and drops its payload to
                    // zero, so the reading that changes is its own.
                    sampled_texture_view(index, Some(vec![0x00; 64]))
                } else {
                    sampled_texture_view(index, None)
                }
            })
            .collect::<Vec<_>>()
    };

    let first = admit_and_submit(&provider, &compute, &render, payloads(None))
        .expect("the thirteen-texture pass executes");
    let again = admit_and_submit(&provider, &compute, &render, payloads(None))
        .expect("the same shape executes twice");
    assert_eq!(
        first, again,
        "one registration, one shape, one payload: the same bytes land the same frame"
    );

    // A middle argument of the table: texture 7's payload is 8, so the sum
    // falls to 83 and the channel beside it does not move.
    let middle = admit_and_submit(&provider, &compute, &render, payloads(Some(7)))
        .expect("the shape with a moved middle payload executes");
    let middle_texel = uniform_texel(&middle);
    eprintln!("texture 7 zeroed: {}", hex(&middle_texel));
    assert_eq!(middle_texel, [0x53, 0x01, 0x0d, 0xff]);
    assert_ne!(middle_texel, THIRTEEN_FRAME);

    // The last argument of the table: texture 12's payload is 13, so both the
    // sum and the channel that pins the last argument move.
    let last = admit_and_submit(&provider, &compute, &render, payloads(Some(12)))
        .expect("the shape with a moved last payload executes");
    let last_texel = uniform_texel(&last);
    eprintln!("texture 12 zeroed: {}", hex(&last_texel));
    assert_eq!(last_texel, [0x4e, 0x01, 0x00, 0xff]);
    assert_ne!(last_texel, THIRTEEN_FRAME);
}

/// The device's window is what admits the shape, and a narrower one is a
/// refusal by name (`E-TC1`).
///
/// The snapshot the rail publishes states the per-stage window beside the list
/// bound — `min(MAX_RENDER_TEXTURES, maxPerStageDescriptorSampledImages)` and
/// twice it — and the thirteen declarations are inside it on the device this
/// test runs on. A snapshot whose window is twelve refuses the very same trace
/// with `render_texture_limit` and the stage, the count and the window on the
/// refusal, which is the arm a device narrower than the review ceiling answers.
#[test]
fn the_device_window_is_what_admits_the_thirteenth_texture() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"thirteen texture window"),
        })
        .expect("the registration is well formed");
    let capabilities = provider.capabilities();
    eprintln!(
        "device window: per_stage={} list={} declared={}",
        capabilities.max_render_textures_per_stage,
        capabilities.max_render_textures,
        capabilities.declares_render_texture_per_stage_ceiling()
    );
    assert!(
        capabilities.declares_render_texture_per_stage_ceiling(),
        "the rail publishes the window the shape is weighed against"
    );
    assert!(
        capabilities.max_render_textures_per_stage >= TEXTURE_COUNT,
        "the device's window admits the census's thirteen declarations"
    );
    assert_eq!(
        capabilities.max_render_textures,
        capabilities.max_render_textures_per_stage.saturating_mul(2),
        "the list bound beside it is the pair's sum"
    );

    let textures = || {
        (0..TEXTURE_COUNT)
            .map(|index| sampled_texture_view(index, None))
            .collect::<Vec<_>>()
    };
    let (trace, resources) = trace_for(&provider, &compute, &render, textures());
    capabilities
        .validate_trace(trace.clone(), resources)
        .expect("the device's own window admits the shape");

    let mut narrower = capabilities.clone();
    narrower.max_render_textures_per_stage = TEXTURE_COUNT - 1;
    let (trace, resources) = trace_for(&provider, &compute, &render, textures());
    let refusal = narrower
        .validate_trace(trace, resources)
        .expect_err("twelve is one below the thirteen the pass declares");
    eprintln!("narrowed window refusal: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_limit");
    assert_eq!(
        refusal
            .fields
            .get("stage")
            .map(|field| format!("{field:?}")),
        Some("Text(\"fragment\")".to_owned()),
        "the refusal names the stage whose own list crossed the window"
    );
    assert_eq!(
        refusal
            .fields
            .get("requested")
            .map(|field| format!("{field:?}")),
        Some("Unsigned(13)".to_owned())
    );
    assert_eq!(
        refusal
            .fields
            .get("maximum")
            .map(|field| format!("{field:?}")),
        Some("Unsigned(12)".to_owned())
    );
}
