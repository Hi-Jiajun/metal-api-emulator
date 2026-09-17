//! The render face's sampler-free texel-fetch arm (`research/docs/23` §3.3,
//! v105).
//!
//! The falsifiable claims are the census's own shape, made executable:
//!
//! * a fragment stage whose `[[texture(0)]]` is read with `texture.read()` —
//!   an `access::read` texture, `OpImageFetch` and no `OpSampledImage` —
//!   registers and executes under a declaration that states `Fetched`, with no
//!   `VkSampler` anywhere in the binding, and lands the texels the module's own
//!   integer coordinates name;
//! * the same module and declaration land *different* bytes when the uploaded
//!   texels change and when only the coordinates change, so the reading
//!   measures the fetch rather than the run;
//! * the two rails that execute the class — the trace rail and the object rail
//!   over one provider — land byte-identical frames;
//! * a declaration that states the other access, or one that states a sampler
//!   form a fetch-only module cannot back, is refused by name with both halves
//!   in its fields (`render_texture_access_unsupported`).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SamplerAddressMode,
    SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess, TextureBindingContract,
    TextureFootprintProof, TextureFormat, TextureSource, TextureType, TextureView, TracePass,
    VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment, and which forwards no varying at all.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The fetch-only fixture and its far-coordinate sibling (`v105`), and the
/// sampling fixture the refusal arm declares `Fetched` under.
const FRAGMENT_ENTRY: &str = "render_fetch_texture_2d";
const FETCH_AIR: &str = include_str!("fixtures/render_fetch_texture_2d.frag.ll");
const FETCH_FAR_AIR: &str = include_str!("fixtures/render_fetch_texture_2d_far.frag.ll");
const SAMPLING_ENTRY: &str = "render_sample_texture_2d";
const SAMPLING_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(940);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(941);
const SCRATCH_VIEW: ViewId = ViewId::new(942);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(943);
const TEXTURE_VIEW: ViewId = ViewId::new(944);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(945);

/// 4x4, the extent of both the attachment and the fetched texture: the rail's
/// reviewed window requires the two to agree.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The frame each reading lands, one texel of the 4x4 attachment: every
/// fragment fetches the same two texels, so the whole attachment carries one
/// colour.
///
/// * the fixture's own coordinates (1, 0) and (0, 1) over a texture whose texel
///   `(i, j)` holds `(64 * i, 64 * j, 0, 255)`: red 64, green 64;
/// * the same module over the descending texture (column `i` holds
///   `64 * (3 - i)` and row `j` holds `64 * (3 - j)`): texel (1, 0).x is 128
///   and texel (0, 1).y is 128;
/// * the far fixture (3, 0) and (0, 3) over the ascending texture: 192 and 192.
const FETCHED_FRAME: [u8; 4] = [0x40, 0x40, 0x00, 0xff];
const DESCENDING_FRAME: [u8; 4] = [0x80, 0x80, 0x00, 0xff];
const FAR_FRAME: [u8; 4] = [0xc0, 0xc0, 0x00, 0xff];

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
    let provider = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context"),
    );
    Some((executor, provider))
}

/// The fetched texture's sixteen texels: texel `(i, j)` holds `(64 * i,
/// 64 * j, 0, 255)` ascending, and `(64 * (3 - i), 64 * (3 - j), 0, 255)`
/// descending, so the one payload moves both halves the fixture reads. Every
/// reading is a multiple of sixteen, so the 8-bit unorm quantisation is exact
/// on any driver.
fn texture_bytes(descending: bool) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            let column = if descending { EXTENT - 1 - i } else { i };
            let row = if descending { EXTENT - 1 - j } else { j };
            bytes.extend_from_slice(&[(64 * column) as u8, (64 * row) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

fn fetched_texture_view(descending: bool) -> TextureView {
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
        access: TextureAccess::Fetched,
        source: TextureSource::OwnedBytes(texture_bytes(descending)),
    }
}

/// The contract a fetch-only registration states: the fixture's own two
/// entries, one `rgba8_unorm` attachment, no vertex stream, and the
/// sampler-free texture declaration the registration repeats (`v105`).
fn contract(entry: &str, footprint: TextureFootprintProof) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: entry.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract {
            metal_binding: 0,
            access: TextureAccess::Fetched,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            sampler: None,
            runtime_sampler: None,
            footprint,
        }],
    }
}

/// Translate one fragment fixture the way a host feeding guest AIR would, and
/// report what the module states about its own texture binding.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_air: &str,
    entry: &str,
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
        .new_library_with_air(fragment_air)
        .expect("the fragment fixture loads");
    let function = library.function(entry).expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated {entry}: {} bytes, reflection entry {:?}, bindings {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
        fragment
            .reflection()
            .bindings
            .iter()
            .map(|binding| (
                format!("{:?}", binding.kind),
                binding.metal_index,
                binding.descriptor.map(|descriptor| (
                    descriptor.set,
                    descriptor.binding,
                    descriptor.count
                )),
                binding.access.map(|access| format!("{access:?}")),
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
        .compile_pipeline(&function, digest(b"fetch-texture-compute"))
        .expect("the compute pipeline registers")
}

fn render_pass(pipeline: PipelineId, textures: Vec<TextureView>) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
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
            TracePass::Render(render_pass(render.pipeline_id, textures)),
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
            size: attachment_bytes,
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

/// The frame the trace rail lands for one fixture and payload.
fn trace_readback(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    compute: &CompiledComputePipeline,
    fragment_air: &str,
    entry: &str,
    descending: bool,
    what: &str,
) -> Result<Vec<u8>, ProviderError> {
    let (vertex, fragment) = translated_pair(executor, fragment_air, entry);
    let render = provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(entry, TextureFootprintProof::WholeView),
        vertex,
        fragment,
        logical_digest: digest(what.as_bytes()),
    })?;
    let (trace, resources) = trace_for(
        provider,
        compute,
        &render,
        vec![fetched_texture_view(descending)],
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
    Ok(submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback"))
}

/// The frame the object rail lands for the same registration and payload: the
/// object API's own command buffer over the very provider the trace rail used.
fn object_readback(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    descending: bool,
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the registration wraps for the object API");
    // The declaring pass and the two buffers the trace rail's own trace
    // carries: the attachment's bytes are the pass's own declaration, exactly
    // as the trace rail's declaring compute pass is, and the scratch write is
    // what makes the declaration observable.
    let attachment = device
        .new_buffer_with_bytes(vec![0xfe; 64])
        .expect("the attachment buffer is declared");
    let view = attachment
        .view(0, (u64::from(EXTENT) * u64::from(EXTENT) * 4) as usize)
        .expect("the attachment view is declared");
    let scratch = device
        .new_buffer_with_bytes(vec![0xab; 4])
        .expect("the scratch buffer is declared");
    let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
    let texture = device
        .new_fetched_texture_with_bytes(
            TextureFormat::Rgba8Unorm,
            u64::from(EXTENT),
            u64::from(EXTENT),
            texture_bytes(descending),
        )
        .expect("the fetched texture is declared");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"fetch-texture-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(&declaring)
            .expect("compute pipeline state");
        encoder
            .set_buffer(0, &view)
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
        let mut render_encoder = command
            .render_command_encoder()
            .expect("the render encoder opens");
        render_encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the fetch pipeline is bound");
        render_encoder
            .set_fragment_texture(0, &texture)
            .expect("the fetch texture is bound at its own index");
        render_encoder
            .draw_render_pass(
                &view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the full-screen triangle is drawn");
        render_encoder
            .end_encoding()
            .expect("the render encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    attachment
        .read()
        .expect("the attachment bytes are readable")
}

/// One texel of the readback: the fixture fetches the same two texels for every
/// fragment, so the whole attachment carries one colour.
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
            "every fragment fetches the same texels, so every texel has to carry the same \
             colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// Reading 1 (`research/docs/23` §3.3, v105): the fetch-only declaration
/// executes, lands the texels the module's own integer coordinates name, and
/// the trace rail and the object rail land byte-identical frames.
#[test]
fn the_sampler_free_declaration_executes_the_modules_own_fetches() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, FETCH_AIR, FRAGMENT_ENTRY);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(FRAGMENT_ENTRY, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"fetch-only declaration"),
        })
        .expect("the fetch-only declaration registers");

    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        vec![fetched_texture_view(false)],
    );
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the fetch-only pass is admitted");
    let submitted = provider.submit(admitted).expect("the trace executes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let trace_bytes = submitted
        .writebacks
        .iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes.clone())
        .expect("the attachment has a writeback");
    let trace_texel = uniform_texel(&trace_bytes);
    eprintln!(
        "trace rail: expected {} landed {}",
        hex(&FETCHED_FRAME),
        hex(&trace_texel)
    );
    assert_eq!(trace_texel, FETCHED_FRAME);

    // The second rail: the object API's own command buffer over the same
    // registration and the same payload. A fetch binds the image alone, so the
    // object rail's texel-fetch handle is the access every view it declares
    // carries (`v105`), and the two rails' frames have to be one frame.
    let object_bytes = object_readback(&provider, &render, false);
    let object_texel = uniform_texel(&object_bytes);
    eprintln!(
        "object rail: {} (trace rail {})",
        hex(&object_texel),
        hex(&trace_texel)
    );
    assert_eq!(
        object_bytes, trace_bytes,
        "the trace rail and the object rail execute one registration and one payload, so their \
         frames have to agree byte for byte"
    );

    provider
        .release_render_pipeline(&render)
        .expect("the registration is released");
}

/// Reading 2 (`research/docs/23` §3.3, v105): the frame follows the payload and
/// the coordinates under one and the same declaration, so the reading above
/// measures the fetch.
#[test]
fn the_fetched_bytes_follow_the_uploaded_texels_and_the_coordinates() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let ascending = trace_readback(
        &provider,
        &executor,
        &compute,
        FETCH_AIR,
        FRAGMENT_ENTRY,
        false,
        "fetch ascending texture",
    )
    .expect("the ascending payload executes");
    let descending = trace_readback(
        &provider,
        &executor,
        &compute,
        FETCH_AIR,
        FRAGMENT_ENTRY,
        true,
        "fetch descending texture",
    )
    .expect("the descending payload executes");
    let far = trace_readback(
        &provider,
        &executor,
        &compute,
        FETCH_FAR_AIR,
        "render_fetch_texture_2d_far",
        false,
        "fetch far coordinates",
    )
    .expect("the far fixture executes");
    let ascending = uniform_texel(&ascending);
    let descending = uniform_texel(&descending);
    let far = uniform_texel(&far);
    eprintln!(
        "ascending {} descending {} far {}",
        hex(&ascending),
        hex(&descending),
        hex(&far)
    );
    assert_eq!(ascending, FETCHED_FRAME);
    assert_eq!(descending, DESCENDING_FRAME);
    assert_eq!(far, FAR_FRAME);
    for (index, first) in [ascending, descending, far].iter().enumerate() {
        for second in [ascending, descending, far].iter().skip(index + 1) {
            assert_ne!(
                first, second,
                "two readings of one declaration that differ in payload or coordinates cannot \
                 land the same texels"
            );
        }
    }

    // Control: the very same request run twice lands the very same bytes, so
    // the readings above measure the payload and the coordinates rather than
    // the run.
    let again = trace_readback(
        &provider,
        &executor,
        &compute,
        FETCH_AIR,
        FRAGMENT_ENTRY,
        false,
        "fetch ascending texture",
    )
    .expect("the ascending payload executes again");
    eprintln!("ascending, second run: {}", hex(&uniform_texel(&again)));
    assert_eq!(uniform_texel(&again), ascending);
}

/// Reading 3 (`research/docs/23` §3.3, v105): a declaration that states the
/// other access is refused by name, with both halves in its fields, before any
/// device object exists.
#[test]
fn a_declaration_that_does_not_repeat_the_module_is_refused_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };

    // The census's own mismatch: the pass declares the sampler-free fetch under
    // a module whose instructions sample the texture through an AIR constexpr
    // sampler. `canonical` used to answer this shape with
    // `render_stage_reflection_mismatch`; the fetch arm names it.
    let (vertex, fragment) = translated_pair(&executor, SAMPLING_AIR, SAMPLING_ENTRY);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(SAMPLING_ENTRY, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"fetch declaration under a sampling module"),
        })
        .expect_err("the module samples the texture the declaration only fetches");
    eprintln!("sampling module under a Fetched declaration: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_access_unsupported");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert_eq!(
        refusal.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
    assert_eq!(
        refusal.fields.get("declared_access"),
        Some(&FieldValue::Text("Fetched".to_owned()))
    );
    assert_eq!(
        refusal.fields.get("module_access"),
        Some(&FieldValue::Text("sampled".to_owned()))
    );

    // The other direction: the module only texel-fetches, and the declaration
    // states the sampled access. The sampler state it names has no
    // `OpSampledImage` behind it, so it is refused with the module's own half
    // in the fields rather than executed with a sampler nothing reads through.
    let (vertex, fragment) = translated_pair(&executor, FETCH_AIR, FRAGMENT_ENTRY);
    let mut sampled = contract(FRAGMENT_ENTRY, TextureFootprintProof::WholeView);
    sampled.textures[0].access = TextureAccess::Sampled;
    sampled.textures[0].sampler = Some(SamplerPolicy {
        filter: SamplerFilter::Nearest,
        address: SamplerAddressMode::ClampToEdge,
    });
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: sampled,
            vertex,
            fragment,
            logical_digest: digest(b"sampled declaration under a fetch module"),
        })
        .expect_err("the module fetches the texture the declaration samples");
    eprintln!("fetch module under a Sampled declaration: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_access_unsupported");
    assert_eq!(
        refusal.fields.get("declared_access"),
        Some(&FieldValue::Text("Sampled".to_owned()))
    );
    assert_eq!(
        refusal.fields.get("module_access"),
        Some(&FieldValue::Text("fetched".to_owned()))
    );

    // Control: the module's own arm registers, so the refusals above are about
    // the access rather than about the pair.
    let (vertex, fragment) = translated_pair(&executor, FETCH_AIR, FRAGMENT_ENTRY);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(FRAGMENT_ENTRY, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"fetch declaration control"),
        })
        .expect("the module's own arm registers");
    provider
        .release_render_pipeline(&render)
        .expect("the registration is released");
}
