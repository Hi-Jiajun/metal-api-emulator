//! Sparse render texture declarations (`research/docs/23` §3.3, v104).
//!
//! The falsifiable claim is the census v11 shape: a fragment stage whose
//! sampled texture is `[[texture(3)]]` — the draw binds the texture at Metal
//! index 3 and the stage's argument table has nothing at 0, 1 or 2 — used to be
//! the engine's own answer (`render_provider_out_of_class_texture_binding`,
//! 5156 = 49.2% of that boot's first-failure lines) because the canonical
//! contract's texture list was *positional*: entry `i` had to be
//! `[[texture(i)]]`, and a list that would have to skip an index was not one
//! the class could state.
//!
//! Since `v104` the list is *indexed*: each entry's own `metal_binding` is the
//! fragment stage's `[[texture(n)]]` argument, the list stays canonical
//! (ascending, unique, inside its own index bound) and the pipeline's
//! declaration pairs with the pass's binding by that index. This file pins the
//! three readings the increment claims:
//!
//! * the sparse declaration **executes** — one `[[texture(3)]]` and nothing
//!   below it reaches the provider, and the frame it lands is byte for byte the
//!   frame the same body at `[[texture(0)]]` lands;
//! * the frame follows the **uploaded texels**, not the index: swapping the
//!   texture's columns moves the readback under one and the same registration;
//! * the list's own rules stay **refusals by name** — an index at the bound, a
//!   repeated index and both directions of the pairing each answer with their
//!   own slug and the binding they disagreed about.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, ProviderError, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
    StoreOp, TextureAccess, TextureBindingContract, TextureFootprintProof, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    MAX_RENDER_TEXTURE_INDEX, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
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

const FRAGMENT_ENTRY: &str = "render_sample_texture_2d";
/// The dense sibling: the same body with its one texture at `[[texture(0)]]`.
const DENSE_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");
/// The census's own shape: the same body with its one texture at
/// `[[texture(3)]]`.
const SPARSE_AIR: &str = include_str!("fixtures/render_sample_texture_2d_index3.frag.ll");

/// The Metal index the sparse fixture's fragment stage reads.
const SPARSE_BINDING: u32 = 3;

const ATTACHMENT_VIEW: ViewId = ViewId::new(960);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(962);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(963);
const TEXTURE_VIEW: ViewId = ViewId::new(964);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(965);

/// 4x4, the extent of both the attachment and the sampled texture: the rail's
/// reviewed window requires the two to agree.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

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

/// The sampled texture's sixteen texels: column `i` holds `64 * i` in red,
/// `64 * j` in green, zero in blue and full alpha. Both readings the fixture
/// takes are multiples of sixteen, so the linear blend quantizes exactly.
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

/// One pass binding at the index the caller names (`v104`): the view's own
/// `metal_binding` is the fragment stage's `[[texture(n)]]` argument.
fn sampled_texture_view(binding: u32, descending: bool) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: binding,
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

/// One declaration, at the index the caller names, carrying the state the
/// fixtures' own AIR constexpr sampler carries.
fn declaration(binding: u32, footprint: TextureFootprintProof) -> TextureBindingContract {
    TextureBindingContract {
        metal_binding: binding,
        access: TextureAccess::Sampled,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        sampler: Some(NEAREST_CLAMP),
        runtime_sampler: None,
        footprint,
    }
}

fn contract(binding: u32, footprint: TextureFootprintProof) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![declaration(binding, footprint)],
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_air: &str,
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
        .compile_pipeline(&function, digest(b"render-texture-index-compute"))
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

/// The refusal one trace gets before any device object exists.
fn admit_refusal(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
) -> ProviderError {
    let (trace, resources) = trace_for(provider, compute, render, textures);
    provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the shape is refused before any device object exists")
}

/// One texel of the readback, which is what the fragment's fixed-coordinate
/// samples land: every texel of the 4x4 attachment carries the same colour,
/// because the stage samples the same coordinates for every fragment.
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

/// The sparse declaration executes: one binding at `[[texture(3)]]`, nothing
/// below it, and the frame it lands is byte for byte the frame the same body
/// lands when the texture sits at `[[texture(0)]]`.
#[test]
fn a_sparse_texture_declaration_lands_the_dense_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);

    // The dense arm: the pre-v104 shape, which the contract stated as a
    // position. Its frame is what "the index moved, the texels did not" is
    // measured against.
    let (vertex, fragment) = translated_pair(&executor, DENSE_AIR);
    let dense = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(0, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"dense texture index"),
        })
        .expect("the dense registration is well formed");
    let dense_bytes = admit_and_submit(
        &provider,
        &compute,
        &dense,
        vec![sampled_texture_view(0, false)],
    )
    .expect("the dense pass executes");

    // The census's shape: the fragment stage declares `[[texture(3)]]` and the
    // pass binds the texture there. Nothing sits at 0, 1 or 2 — on either side
    // of the pairing.
    let (vertex, fragment) = translated_pair(&executor, SPARSE_AIR);
    let sparse = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(SPARSE_BINDING, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"sparse texture index"),
        })
        .expect("the sparse registration is well formed");
    let sparse_bytes = admit_and_submit(
        &provider,
        &compute,
        &sparse,
        vec![sampled_texture_view(SPARSE_BINDING, false)],
    )
    .expect("the sparse pass executes");

    let dense_texel = uniform_texel(&dense_bytes);
    let sparse_texel = uniform_texel(&sparse_bytes);
    eprintln!(
        "[[texture(0)]] frame: {}  [[texture(3)]] frame: {}",
        hex(&dense_texel),
        hex(&sparse_texel)
    );
    assert_eq!(dense_texel, [0xc0, 0x40, 0x00, 0xff]);
    assert_eq!(
        sparse_texel, dense_texel,
        "the index is a binding, not a texel: the same body at [[texture(3)]] lands the frame \
         [[texture(0)]] lands"
    );
    assert_eq!(sparse_bytes, dense_bytes);
}

/// The frame follows the uploaded texels, not the index it is bound at: one
/// registration, one pass shape, two payloads — and the very same payload twice
/// lands the very same bytes.
#[test]
fn the_sparse_binding_follows_the_uploaded_texels() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, SPARSE_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(SPARSE_BINDING, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"sparse texture payloads"),
        })
        .expect("the sparse registration is well formed");

    let ascending = admit_and_submit(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false)],
    )
    .expect("the ascending payload executes");
    let reversed = admit_and_submit(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, true)],
    )
    .expect("the descending payload executes");
    let ascending = uniform_texel(&ascending);
    let reversed = uniform_texel(&reversed);
    eprintln!(
        "ascending texture: {} descending texture: {}",
        hex(&ascending),
        hex(&reversed)
    );
    assert_eq!(ascending, [0xc0, 0x40, 0x00, 0xff]);
    // The clamped sample reads the edge texel — column 3 ascending, column 0
    // descending — while the second sample stays in texel 1, whose descending
    // column carries `64 * 2` instead of `64 * 1`.
    assert_eq!(reversed, [0x00, 0x80, 0x00, 0xff]);
    assert_ne!(ascending, reversed);

    // Control: the very same request run twice lands the very same bytes, so
    // the two readings above measure the payload rather than the run.
    let again = admit_and_submit(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false)],
    )
    .expect("the ascending payload executes again");
    let again = uniform_texel(&again);
    eprintln!("ascending texture, second run: {}", hex(&again));
    assert_eq!(again, ascending);
}

/// The list's own rules stay refusals by name, and each one arrives before any
/// device object exists: the index bound, a repeated index, and both directions
/// of the declaration-versus-binding pairing.
#[test]
fn the_sparse_shape_refuses_the_index_bound_duplicates_and_unpaired_bindings() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);

    // The declaration's own bound: an index at [`MAX_RENDER_TEXTURE_INDEX`] is
    // refused when the registration states it, because the fragment stage's
    // argument table has a width of its own and the *count* cap cannot state
    // it.
    let (vertex, fragment) = translated_pair(&executor, SPARSE_AIR);
    let out_of_range =
        provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(MAX_RENDER_TEXTURE_INDEX, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"texture index at the bound"),
        });
    let refusal = match out_of_range {
        Err(error) => error,
        Ok(_) => panic!("the contract's index bound refuses this declaration"),
    };
    eprintln!("declaration at the index bound: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_pipeline_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains(
                "is at or past the render contract's own \
                                                   texture index bound"
            )),
        "{refusal:?}"
    );

    let (vertex, fragment) = translated_pair(&executor, SPARSE_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(SPARSE_BINDING, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"sparse texture refusals"),
        })
        .expect("the sparse registration is well formed");

    // The pass's own bound: a view at the index the fragment stage's table ends
    // at is refused by the pass's own validator, under the index slug.
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(MAX_RENDER_TEXTURE_INDEX, false)],
    );
    eprintln!("binding at the index bound: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_index_unsupported");
    assert_eq!(
        refusal.class,
        metal_api_core::provider::ProviderErrorClass::Capability
    );

    // One index bound twice is one descriptor with two candidate contents, so
    // the pass's own list refuses it.
    let mut duplicate = sampled_texture_view(SPARSE_BINDING, false);
    duplicate.view_id = ViewId::new(966);
    duplicate.allocation_id = AllocationId::new(967);
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false), duplicate],
    );
    eprintln!("repeated index: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("duplicate Metal binding 3")),
        "{refusal:?}"
    );

    // A declaration the pass never binds leaves the descriptor the module reads
    // undefined.
    let refusal = admit_refusal(&provider, &compute, &render, Vec::new());
    eprintln!("declaration without a binding: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("declares texture binding 3")),
        "{refusal:?}"
    );

    // The other direction: a binding the declaration never names would fill a
    // slot the module said nothing about. The declaration's own binding is
    // bound beside it, so this is the reverse walk's refusal rather than the
    // one above.
    let mut undeclared = sampled_texture_view(5, false);
    undeclared.view_id = ViewId::new(968);
    undeclared.allocation_id = AllocationId::new(969);
    let refusal = admit_refusal(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false), undeclared],
    );
    eprintln!("binding without a declaration: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("pass binds texture 5")),
        "{refusal:?}"
    );

    // Control: the same registration with the binding it declares lands its
    // bytes, so the refusals above are about the bindings rather than the
    // shape.
    let bytes = admit_and_submit(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false)],
    )
    .expect("the declared binding executes");
    assert_eq!(uniform_texel(&bytes), [0xc0, 0x40, 0x00, 0xff]);
}

/// The *other* track's reading (`research/docs/23` §3.3, v104): the same sparse
/// binding recorded through the object API — the encoder binds the texture at
/// index 3 and the pass carries that index rather than a position — lands the
/// very same frame the trace rail lands, byte for byte.
#[test]
fn the_sparse_binding_lands_the_same_frame_on_the_object_rail() {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::{Device as ObjectDevice, RenderAttachmentLoad};
    use metal_api_core::Size;

    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, SPARSE_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(SPARSE_BINDING, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"sparse texture object rail"),
        })
        .expect("the sparse registration is well formed");
    let trace_bytes = admit_and_submit(
        &provider,
        &compute,
        &render,
        vec![sampled_texture_view(SPARSE_BINDING, false)],
    )
    .expect("the trace rail executes the sparse pass");

    // The object rail over the same provider and the same registration: one
    // declaring compute pass (the trace's own statement of the attachment's
    // bytes), one attachment buffer, one sampled texture and one binding at
    // index 3.
    let device = ObjectDevice::new(Arc::new(provider));
    let pipeline = device
        .render_pipeline(&render)
        .expect("the registered metadata is a render pipeline");
    let attachment = device
        .new_buffer_with_bytes(vec![0xfe; 64])
        .expect("attachment buffer");
    let attachment_view = attachment.view(0, 64).expect("attachment view");
    let scratch = device
        .new_buffer_with_bytes(vec![0xab; 4])
        .expect("scratch buffer");
    let scratch_view = scratch.view(0, 4).expect("scratch view");
    let texture = device
        .new_texture_with_bytes(TextureFormat::Rgba8Unorm, 4, 4, texture_bytes(false))
        .expect("sampled texture");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"render-texture-index-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
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
            .expect("pipeline state");
        encoder
            .set_fragment_texture(SPARSE_BINDING, &texture)
            .expect("the sparse binding records");
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                4,
                4,
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the sparse draw records");
        encoder.end_encoding().expect("end encoding");
    }
    command.commit().expect("commit");
    command.wait_until_completed().expect("completion");
    let object_bytes = attachment.read().expect("attachment readback");

    let trace_texel = uniform_texel(&trace_bytes);
    let object_texel = uniform_texel(&object_bytes);
    eprintln!(
        "trace rail frame: {}  object rail frame: {}",
        hex(&trace_texel),
        hex(&object_texel)
    );
    assert_eq!(trace_texel, [0xc0, 0x40, 0x00, 0xff]);
    assert_eq!(
        object_texel, trace_texel,
        "both tracks execute one indexed binding, so both land one frame"
    );
    assert_eq!(object_bytes, trace_bytes);
}
