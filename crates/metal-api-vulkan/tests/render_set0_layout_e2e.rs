//! One translated render pass whose fragment stage samples a texture *and*
//! reads a stage buffer from the same descriptor set 0 (`research/docs/23`
//! §3.3, v112).
//!
//! The shape is the guest's own: a translated fragment module reads
//! `[[texture(0)]]` through its AIR constexpr sampler and `[[buffer(0)]]`
//! beside it, and the translator's descriptor layout puts both in set 0 — the
//! buffer band `0..32` and the sampled-texture band `32..160`, with the AIR
//! static sampler in `160..192`. The rail used to refuse the pair of faces by
//! name; the increment under test is that they are one layout, one pool and one
//! set, not two claimants for one slot.
//!
//! What this file measures, on one device:
//!
//! * the pass enters the provider and lands the bytes both faces name — red is
//!   the texel the module's own nearest + clamp-to-edge sampler returns at
//!   `(1.375, 0.125)` (the clamped edge texel, `0xc0` against a texture whose
//!   column `i` holds `64 * i` in red), green and blue are the tint's second
//!   and third components (`0x80`/`0xc0`) — and the object rail's frame for the
//!   same declaration is the same bytes;
//! * swapping either payload moves the frame in the channels that payload owns
//!   and in no others, so both descriptors are really read;
//! * the boundaries that remain are refused by name: a combined translation
//!   whose layout sits above the rail's set ceiling, and the two stages'
//!   buffers folding onto one set-0 slot — the arbitration the merge must not
//!   replace with one write overwriting the other.

use metal2vulkan::reflect::DescriptorLayout;
use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue,
    FootprintProof, LoadOp, OperationId, PipelineId, ProviderError, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, RenderPipelineStage, ResourceTableSnapshot,
    SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, StageBufferBinding,
    StageBufferView, StoreOp, TextureAccess, TextureBindingContract, TextureFootprintProof,
    TextureFormat, TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    stage_buffer_namespace_layout, RenderStage, TranslatedRenderPipelineRequest,
    TranslatedRenderStage, VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment, and which forwards no varying at all.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// The vertex stage that reads its clip positions out of `[[buffer(0)]]`
/// instead, the fixture the two-namespaces-one-slot reading is built from.
const POSITIONS_AIR: &str = include_str!("fixtures/render_stage_buffer_positions.vert.ll");
const POSITIONS_ENTRY: &str = "render_vertex_positions";

/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The fixture under test: one fragment stage that samples a texture and reads
/// a `[[buffer(0)]]` argument beside it.
const FRAGMENT_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_stage_buffer_tint.frag.ll");
const FRAGMENT_ENTRY: &str = "render_sample_texture_2d_stage_buffer_tint";

const ATTACHMENT_VIEW: ViewId = ViewId::new(951);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(952);
const SCRATCH_VIEW: ViewId = ViewId::new(953);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(954);
const TEXTURE_VIEW: ViewId = ViewId::new(955);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(956);
const STAGE_VIEW: ViewId = ViewId::new(957);
const STAGE_ALLOCATION: AllocationId = AllocationId::new(958);
const POSITIONS_VIEW: ViewId = ViewId::new(959);
const POSITIONS_ALLOCATION: AllocationId = AllocationId::new(960);

/// 4x4, the extent of both the attachment and the sampled texture.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The state the new fragment fixture's AIR carries: the declaration has to
/// repeat it, so the contract below names the same policy.
const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// The tint the stage buffer carries: `(64/255, 128/255, 192/255, 1)`, whose
/// second and third components are the frame's green and blue, `0x80` and
/// `0xc0`. Every component is a multiple of `1/255`, so the eight-bit
/// quantization is exact on any driver.
const TINT: [f32; 4] = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0];

/// The second tint, the same three components permuted: green `0x40`, blue
/// `0x80`.
const OTHER_TINT: [f32; 4] = [192.0 / 255.0, 64.0 / 255.0, 128.0 / 255.0, 1.0];

/// The frame the fixture's own two payloads name: red from the clamped edge
/// texel of the ascending texture, green and blue from [`TINT`].
const EXPECTED: [u8; 4] = [0xc0, 0x80, 0xc0, 0xff];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest")
}

fn executor_and_provider() -> Option<(Arc<VulkanExecutor>, Arc<VulkanComputeProvider>)> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    Some((executor, Arc::new(provider)))
}

/// The sampled texture's sixteen texels: column `i` holds `64 * i` in red,
/// `64 * j` in green, zero in blue and full alpha. The fixture's one sample
/// sits outside the unit square, so clamped addressing reads the edge column
/// (192 ascending, 0 descending) and the payload is what the frame's red is.
fn texture_bytes(descending: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let column = if descending { EXTENT - 1 - i } else { i };
            bytes.extend_from_slice(&[(64 * column) as u8, (64 * j) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

fn sampled_texture_view(descending: bool) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(texture_bytes(descending)),
    }
}

/// Four `f32`s as the little-endian bytes a stage buffer's view carries.
fn tint_bytes(tint: [f32; 4]) -> Vec<u8> {
    tint.iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// The contract the fixture registers under: the fragment stage's one stage
/// buffer, the sampled texture with the module's own state repeated, one
/// `Rgba8Unorm` attachment and no vertex stream.
fn contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: vec![StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 0,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 16 },
        }],
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract {
            metal_binding: 0,
            access: TextureAccess::Sampled,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            sampler: Some(NEAREST_CLAMP),
            runtime_sampler: None,
            footprint: TextureFootprintProof::WholeView,
        }],
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would, and
/// report the slots the reflection names: the reading the whole increment rests
/// on is that both faces landed in set 0 and in different bindings.
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
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the sampling stage-buffer fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated fragment: {} bytes, reflection entry {:?}, bindings {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
        fragment
            .reflection()
            .bindings
            .iter()
            .map(|binding| (
                format!("{:?}", binding.kind),
                binding.metal_index,
                binding.access,
                binding.param_index,
                binding.descriptor.map(|descriptor| (
                    descriptor.set,
                    descriptor.binding,
                    descriptor.count
                )),
            ))
            .collect::<Vec<_>>(),
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
        .compile_pipeline(&function, digest(b"render-set0-layout-compute"))
        .expect("the compute pipeline registers")
}

/// The stage buffer's view: the tint's sixteen bytes, read-only.
fn stage_buffer_view(tint: [f32; 4]) -> StageBufferView {
    StageBufferView {
        stage: RenderPipelineStage::Fragment,
        view: BufferView {
            view_id: STAGE_VIEW,
            metal_binding: 0,
            allocation_id: STAGE_ALLOCATION,
            offset: 0,
            length: 16,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(tint_bytes(tint)),
        },
    }
}

fn render_pass(
    pipeline: PipelineId,
    textures: Vec<TextureView>,
    stage_buffers: Vec<StageBufferView>,
) -> RenderPassDescriptor {
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
        textures,
        present: None,
    }
}

fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
    stage_buffers: Vec<StageBufferView>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(41),
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
            TracePass::Render(render_pass(render.pipeline_id, textures, stage_buffers)),
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

/// Register the fixture's translated pair under the declaring contract, then
/// submit one trace and return the attachment's readback bytes.
fn readback(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    compute: &CompiledComputePipeline,
    descending: bool,
    tint: [f32; 4],
    what: &str,
) -> Result<Vec<u8>, ProviderError> {
    let (vertex, fragment) = translated_pair(executor);
    let render = provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(),
        vertex,
        fragment,
        logical_digest: digest(what.as_bytes()),
    })?;
    let (trace, resources) = trace_for(
        provider,
        compute,
        &render,
        vec![sampled_texture_view(descending)],
        vec![stage_buffer_view(tint)],
    );
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

/// One texel of the readback: every fragment samples the same coordinate and
/// reads the same tint, so the attachment is uniform.
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
            "every fragment lands the same colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// The object rail's frame for the same declaration: one sampled texture and
/// one caller-held stage-buffer view recorded through the same translated
/// pipeline, so the two rails state one interface through two APIs.
fn object_frame(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    descending: bool,
    tint: [f32; 4],
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, PipelineProvider, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    // The object rail wraps the *same* provider: the registration above is the
    // provider's own, and the object API only re-states it as objects.
    let handle: Arc<dyn PipelineProvider> = Arc::clone(provider) as Arc<dyn PipelineProvider>;
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the registration wraps for the object API");
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let attachment = device
        .new_buffer_with_bytes(vec![0x00; attachment_bytes as usize])
        .expect("the attachment's landing buffer is declared");
    let attachment_view = attachment
        .view(0, attachment_bytes as usize)
        .expect("the attachment view is declared");
    let texture = device
        .new_texture_with_bytes(
            TextureFormat::Rgba8Unorm,
            u64::from(EXTENT),
            u64::from(EXTENT),
            texture_bytes(descending),
        )
        .expect("the sampled texture is declared");
    let tint = device
        .new_buffer_with_bytes(tint_bytes(tint))
        .expect("the stage buffer's bytes are declared");
    let tint_view = tint.view(0, 16).expect("the stage buffer view is declared");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"render_set0_layout_object_declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let scratch = device
            .new_buffer_with_bytes(vec![0xab; 4])
            .expect("the scratch buffer is declared");
        let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
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
            .expect("the sampling pipeline is bound");
        encoder
            .set_fragment_texture(0, &texture)
            .expect("the texture is bound at its own binding");
        encoder
            .set_stage_buffer(RenderPipelineStage::Fragment, 0, &tint_view)
            .expect("the stage buffer is bound at its own slot");
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the sampling pass records");
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

#[test]
fn a_set_zero_sampler_and_stage_buffer_enter_the_provider() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let trace = readback(
        &provider,
        &executor,
        &compute,
        false,
        TINT,
        "set-0 sampler + buffer",
    )
    .expect("the shared set-0 pass executes");
    let texel = uniform_texel(&trace);
    eprintln!("trace rail frame: {}", hex(&trace));
    assert_eq!(texel, EXPECTED);

    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"set-0-sampler-and-buffer-object"),
        })
        .expect("the pair registers for the object rail");
    let object = object_frame(&provider, &render, false, TINT);
    eprintln!("object rail frame: {}", hex(&object));
    assert_eq!(
        object, trace,
        "the two rails state one interface and have to land one frame"
    );
}

#[test]
fn the_frame_follows_the_texture_and_the_stage_buffers_bytes() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let baseline = readback(
        &provider,
        &executor,
        &compute,
        false,
        TINT,
        "payload baseline",
    )
    .expect("the baseline executes");
    let other_texture = readback(
        &provider,
        &executor,
        &compute,
        true,
        TINT,
        "swapped texture bytes",
    )
    .expect("the swapped texture executes");
    let other_tint = readback(
        &provider,
        &executor,
        &compute,
        false,
        OTHER_TINT,
        "swapped stage buffer bytes",
    )
    .expect("the swapped tint executes");
    eprintln!(
        "baseline {}; swapped texture {}; swapped stage buffer {}",
        hex(&baseline),
        hex(&other_texture),
        hex(&other_tint)
    );
    let baseline = uniform_texel(&baseline);
    let other_texture = uniform_texel(&other_texture);
    let other_tint = uniform_texel(&other_tint);
    // The sampled texture owns red alone: the descending payload's clamped edge
    // texel is column 0, and every other channel stands still.
    assert_eq!(baseline, EXPECTED);
    assert_eq!(other_texture, [0x00, EXPECTED[1], EXPECTED[2], 0xff]);
    // The stage buffer owns green and blue: the second tint's own components
    // land there and red stands still.
    assert_eq!(other_tint, [EXPECTED[0], 0x40, 0x80, 0xff]);
}

/// The ceiling arm of the merged shape (`research/docs/23` §3.3, v84/v112): a
/// combined translation whose layout names a set above the rail's own is
/// refused by name at registration — the merge is not a loophole for a module
/// the rail has no pipeline layout for.
#[test]
fn a_combined_translation_above_the_set_ceiling_is_refused() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(VERTEX_AIR)
        .expect("the vertex fixture loads");
    let function = library
        .function(VERTEX_ENTRY)
        .expect("the vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &function)
        .expect("the vertex stage translates");
    let library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the combined fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate_with_policy_and_layout(
        RenderStage::Fragment,
        &function,
        executor.spirv_feature_policy(),
        DescriptorLayout {
            set: 3,
            ..DescriptorLayout::default()
        },
    )
    .expect("the combined fragment translates under the layout");
    let refused = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"above-ceiling-combined-set-0"),
        })
        .expect_err("the rail's pipeline layout does not name set 3");
    eprintln!("refused the out-of-rail combined layout: {refused:?}");
    assert_eq!(refused.slug, "render_stage_unsupported_interface");
    assert_eq!(
        refused.fields.get("field"),
        Some(&FieldValue::Text("bindings".to_owned()))
    );
    assert_eq!(refused.fields.get("set"), Some(&FieldValue::Unsigned(3)));
    assert_eq!(
        refused.fields.get("set_ceiling"),
        Some(&FieldValue::Unsigned(2))
    );
}

/// The coexistence boundary that stays (`research/docs/23` §3.3, v84/v112): the
/// two stages' Metal buffer namespaces are independent, so a pair whose modules
/// each read `[[buffer(0)]]` in set 0 folds both writes onto one descriptor —
/// refused by name beside the texture rather than executed with one stage
/// reading the other's bytes.
///
/// The companion reading (`a_two_stage_fold_executes_under_the_namespace_layout`)
/// is the same request translated another way: the vertex stage's own layout
/// moved to the set the provider publishes for this shape, so the two index
/// spaces no longer collide. The pair is what makes the refusal's remedy
/// falsifiable — the same declaration, the same bytes, one axis of difference.
#[test]
fn two_stage_buffers_on_one_set_zero_slot_are_still_refused() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment, declaration) = folded_pair(&executor, DescriptorLayout::default());
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: declaration,
            vertex,
            fragment,
            logical_digest: digest(b"two-stage-buffers-one-set-0-slot"),
        })
        .expect("each stage's own pairing holds");
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(false)],
        vec![
            // Canonical order again: the vertex stage's slot first, then the
            // fragment stage's.
            positions_view(vec![0x00; 32]),
            stage_buffer_view(TINT),
        ],
    );
    let admitted = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect("the trace's declarations are the contract's own");
    let refused = provider
        .submit(admitted)
        .expect_err("the two stages' slots fold onto one descriptor");
    eprintln!("refused the folded stage-buffer slots: {refused:?}");
    assert_eq!(refused.slug, "render_stage_buffer_layout_unsupported");
    assert_eq!(refused.fields.get("set"), Some(&FieldValue::Unsigned(0)));
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
    // The remedy the refusal names is the layout the companion reading uses
    // (`research/docs/23` §3.3, E-TX9): the detail points at the published
    // arrangement rather than leaving the caller to guess one.
    let detail = refused
        .detail
        .as_deref()
        .expect("the refusal carries a detail");
    assert!(
        detail.contains("stage_buffer_namespace_layout"),
        "the refusal points at the canonical namespace layout: {detail}"
    );
}

/// The folded request both tests in this pair state (`research/docs/23` §3.3,
/// v112/E-TX9): the vertex stage reads its positions from `[[buffer(0)]]`, the
/// fragment stage reads its tint from `[[buffer(0)]]` and samples a texture
/// from the same set 0, and one declaration names both slots. The only axis the
/// pair varies is the layout the *vertex* module is translated against.
fn folded_pair(
    executor: &Arc<VulkanExecutor>,
    vertex_layout: DescriptorLayout,
) -> (
    TranslatedRenderStage,
    TranslatedRenderStage,
    RenderPipelineContract,
) {
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
    .expect("the positions stage translates under the layout");
    let library = device
        .new_library_with_air(FRAGMENT_AIR)
        .expect("the combined fragment fixture loads");
    let function = library
        .function(FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    let mut declaration = contract();
    // Canonical order: the vertex stage's slots first, then the fragment
    // stage's — the order the contract's own binding walk states.
    declaration.stage_buffers.insert(
        0,
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
        },
    );
    declaration.vertex_entry = POSITIONS_ENTRY.to_owned();
    (vertex, fragment, declaration)
}

/// The vertex stage's own slot as a pass binds it: the three `vec2` positions
/// the fixture reads, at `[[buffer(0)]]`.
fn positions_view(bytes: Vec<u8>) -> StageBufferView {
    StageBufferView {
        stage: RenderPipelineStage::Vertex,
        view: BufferView {
            view_id: POSITIONS_VIEW,
            metal_binding: 0,
            allocation_id: POSITIONS_ALLOCATION,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the positions fit a u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        },
    }
}

/// The three Metal-NDC `vec2` positions the reviewed stage-buffer fixture
/// reads (`research/docs/23` §3.3, v83): `(-0.9, 0.9)`, `(0.0, 0.9)`,
/// `(-0.9, 0.0)`. Translated, the rail flips y into Vulkan's clip space, so
/// the triangle covers the render area's top-left corner.
fn reviewed_positions() -> Vec<u8> {
    [-0.9_f32, 0.9, 0.0, 0.9, -0.9, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The same request, executed (`research/docs/23` §3.3, E-TX9).
///
/// The vertex stage's `[[buffer(0)]]` arguments are translated into the
/// provider's published namespace layout, so they land in set 1 while the
/// fragment stage keeps set 0 — the two stages' Metal index spaces stay
/// separate, the sampled texture still shares set 0 with the fragment half, and
/// the frame is therefore both payloads' own: the vertex bytes decide which
/// texels are covered and the texture and tint decide what they hold.
#[test]
fn a_two_stage_fold_executes_under_the_namespace_layout() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment, declaration) = folded_pair(&executor, stage_buffer_namespace_layout());
    // The reading the whole test rests on: the two stages' buffer descriptors
    // are in different sets, so nothing folds.
    eprintln!(
        "namespace pair slots: vertex={:?} fragment={:?}",
        vertex
            .reflection()
            .bindings
            .iter()
            .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
            .map(|binding| binding
                .descriptor
                .map(|descriptor| (descriptor.set, descriptor.binding)))
            .collect::<Vec<_>>(),
        fragment
            .reflection()
            .bindings
            .iter()
            .filter(|binding| binding.kind == metal2vulkan::reflect::ResourceKind::Buffer)
            .map(|binding| binding
                .descriptor
                .map(|descriptor| (descriptor.set, descriptor.binding)))
            .collect::<Vec<_>>(),
    );
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: declaration,
            vertex,
            fragment,
            logical_digest: digest(b"two-stage-fold-namespace-layout"),
        })
        .expect("each stage's own pairing holds");
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(false)],
        vec![
            positions_view(reviewed_positions()),
            stage_buffer_view(TINT),
        ],
    );
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace's declarations are the contract's own");
    let submitted = provider
        .submit(admitted)
        .expect("the two stages' slots are separate, so the pair executes");
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
    eprintln!("namespace-layout frame: {}", hex(&bytes));
    // The vertex bytes cover the top-left corner of the 4×4 render area: three
    // texels (`(0,0)`, `(1,0)`, `(0,1)`) carry the fragment stage's own bytes —
    // the texture's clamped edge texel in red, the tint's second and third
    // components in green and blue — and every other texel keeps the clear
    // sentinel. A rail that folded the two slots would land one stage's bytes
    // under the other's, which this frame cannot read as.
    assert_eq!(bytes.len(), 64, "four texels per row, four rows");
    let mut expected = Vec::with_capacity(64);
    for row in 0..EXTENT {
        for column in 0..EXTENT {
            let covered = matches!((column, row), (0, 0) | (1, 0) | (0, 1));
            expected.extend_from_slice(if covered { &EXPECTED } else { &CLEAR_SENTINEL });
        }
    }
    assert_eq!(bytes, expected);
}
