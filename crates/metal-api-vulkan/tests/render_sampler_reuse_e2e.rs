//! Sampled-sampler-reuse probe: one fragment stage whose four sampled textures
//! read through **two** AIR `constexpr samplers`, three of them through one
//! (2026-09-20, census v48's fifth door; `research/docs/23` §3.3).
//!
//! The census's own shape is thirteen sampled textures against four sampler
//! descriptors, and the question this probe settles is whether the rail
//! executes a module that samples *several* textures through one AIR static
//! sampler — the registration used to pair one AIR sampler with one sampled
//! texture by position and refuse the counts
//! (`render_stage_reflection_mismatch`, "this rail pairs one AIR static sampler
//! with one sampled texture").
//!
//! The fixture answers it in the attachment's bytes. All four textures hold the
//! same 4x4 surface (column `i` holds `64 * i` in red) and every sample is at
//! `u = 1.375`, one half column past the surface's right edge:
//!
//! * the module's Nearest + Repeat state reads texel `(1, 3)` — `0x40`;
//! * the module's Linear + ClampToEdge state reads texel `(3, 3)` — `0xc0`.
//!
//! so the frame is `40 40 c0 40` with red, green and alpha carrying the reused
//! state's reading and blue the other one's. A rail that paired the two states
//! with the textures by position would land blue as `0x40` and green as `0xc0`,
//! and one that refused the counts would land no frame at all.
//!
//! The three arms beside the positive one are the refusals that have to survive
//! the reuse being admitted: a contract that declares a texture through the
//! state the module does *not* name for it
//! (`render_texture_sampler_unsupported`), a contract that omits a texture the
//! module reads (`render_stage_reflection_mismatch`), and a contract whose
//! runtime sampler no declaration pairs with (`render_runtime_sampler_unpaired`
//! — this fixture declares no runtime argument, so that arm is the same module
//! with a `[[sampler(0)]]` declaration on one texture).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, RenderSamplerBinding, ResourceTableSnapshot,
    SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess,
    TextureBindingContract, TextureFormat, TextureSource, TextureType, TextureView, TracePass,
    VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage: a full-screen triangle whose vertex_id
/// positions cover the whole attachment and which forwards no varying.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");
/// The reuse fixture: four sampled textures, two AIR static samplers, one of
/// them read through by three of the four.
const REUSE_FRAGMENT_ENTRY: &str = "render_sample_texture_2d_sampler_reuse";
const REUSE_AIR: &str = include_str!("fixtures/render_sample_texture_2d_sampler_reuse.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(970);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(971);
const SCRATCH_VIEW: ViewId = ViewId::new(972);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(973);
const TEXTURE_VIEW_BASE: u64 = 974;
const TEXTURE_ALLOCATION_BASE: u64 = 984;

/// 4x4, the extent of the attachment and of all four sampled textures.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The module's two AIR states: the census's own pair
/// (`smpl=4[s832:gNNnee,s833:gNNnee,s834:cLLnee,s835:cNNnrr]`).
const REPEAT_POLICY: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::Repeat,
};
const LINEAR_CLAMP_POLICY: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::ClampToEdge,
};

/// The one reading the positive arm asserts: the reused state's texel `(1, 3)`
/// for three textures and the clamped state's texel `(3, 3)` for the third.
const REUSE_READBACK: [u8; 4] = [0x40, 0x40, 0xc0, 0x40];

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

/// The sampled textures' sixteen texels: column `i` holds `64 * i` in red,
/// `64 * j` in green, zero in blue and full alpha.
fn texture_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            bytes.extend_from_slice(&[(64 * i) as u8, (64 * j) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

fn texture_view(view_id: ViewId, allocation_id: AllocationId, metal_binding: u32) -> TextureView {
    TextureView {
        view_id,
        metal_binding,
        allocation_id,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(EXTENT),
        height: u64::from(EXTENT),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(texture_bytes()),
    }
}

/// The reuse module's contract: every one of the four textures stated through
/// the state the module's own sample sites name for it — `[[texture(0)]]`,
/// `[[texture(1)]]` and `[[texture(3)]]` through the repeated state,
/// `[[texture(2)]]` through the linear one.
fn reuse_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: REUSE_FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![
            TextureBindingContract::sampled(0, TextureFormat::Rgba8Unorm, REPEAT_POLICY),
            TextureBindingContract::sampled(1, TextureFormat::Rgba8Unorm, REPEAT_POLICY),
            TextureBindingContract::sampled(2, TextureFormat::Rgba8Unorm, LINEAR_CLAMP_POLICY),
            TextureBindingContract::sampled(3, TextureFormat::Rgba8Unorm, REPEAT_POLICY),
        ],
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
        .new_library_with_air(REUSE_AIR)
        .expect("the fragment fixture loads");
    let function = library
        .function(REUSE_FRAGMENT_ENTRY)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated fragment {REUSE_FRAGMENT_ENTRY}: {} bytes, bindings {:?}",
        fragment.spirv().len(),
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
        .compile_pipeline(&function, digest(b"render-sampler-reuse-compute"))
        .expect("the compute pipeline registers")
}

fn render_pass(
    pipeline: PipelineId,
    textures: Vec<TextureView>,
    samplers: Vec<RenderSamplerBinding>,
) -> RenderPassDescriptor {
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
        samplers,
        present: None,
    }
}

fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
    samplers: Vec<RenderSamplerBinding>,
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
            TracePass::Render(render_pass(render.pipeline_id, textures, samplers)),
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

/// Admit and submit one pass, returning the attachment's readback bytes.
fn submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
    samplers: Vec<RenderSamplerBinding>,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, textures, samplers);
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

/// Every fragment samples the same fixed coordinates, so every texel of the
/// attachment carries the same colour.
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

/// The four sampled textures the draw binds: one at each `[[texture(i)]]` the
/// declarations name.
///
/// No sampler state travels with them: every texture of this module reads
/// through one of the module's *own* AIR static samplers, and the rail creates
/// each descriptor's `VkSampler` from that state — a pass states a state only
/// for a runtime `[[sampler(n)]]` argument, and this module binds none.
fn reuse_resources() -> Vec<TextureView> {
    (0..4)
        .map(|index| {
            texture_view(
                ViewId::new(TEXTURE_VIEW_BASE + u64::from(index)),
                AllocationId::new(TEXTURE_ALLOCATION_BASE + u64::from(index)),
                index,
            )
        })
        .collect::<Vec<_>>()
}

/// The falsifiable claim: one stage whose four sampled textures read through two
/// AIR static samplers — one of them reused by three textures — registers,
/// executes, and lands each texture's own reading.
#[test]
fn the_reused_air_static_sampler_is_bound_for_every_texture_that_reads_through_it() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: reuse_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"render sampler reuse"),
        })
        .expect("the reuse registration is well formed");
    let bytes = submit(&provider, &compute, &render, reuse_resources(), Vec::new())
        .expect("the reuse shape executes");
    let texel = uniform_texel(&bytes);
    eprintln!(
        "sampler reuse: expected {} landed {}",
        hex(&REUSE_READBACK),
        hex(&texel)
    );
    assert_eq!(texel, REUSE_READBACK);
}

/// The three refusals that have to survive the reuse being admitted.
#[test]
fn the_shapes_beside_the_reuse_are_refused_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);

    // A declaration that states the state the module does *not* name for that
    // texture: `[[texture(1)]]`'s own sample site reads through the repeated
    // state, and this contract hands it the linear one.
    let (vertex, fragment) = translated_pair(&executor);
    let mut contract = reuse_contract();
    contract.textures[1] =
        TextureBindingContract::sampled(1, TextureFormat::Rgba8Unorm, LINEAR_CLAMP_POLICY);
    let error = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest: digest(b"render sampler reuse, wrong state"),
        })
        .expect_err("a declaration the module does not back is refused");
    eprintln!("wrong-state refusal: {error:?}");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(error.slug, "render_texture_sampler_unsupported");
    assert_eq!(error.fields.get("binding"), Some(&FieldValue::Unsigned(1)));

    // A contract that omits a texture the module reads.
    let (vertex, fragment) = translated_pair(&executor);
    let mut contract = reuse_contract();
    contract.textures.truncate(3);
    let error = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest: digest(b"render sampler reuse, missing texture"),
        })
        .expect_err("a module the contract does not cover is refused");
    eprintln!("missing-texture refusal: {error:?}");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(error.slug, "render_stage_reflection_mismatch");

    // A declaration that names a runtime `[[sampler(n)]]` argument the module
    // does not bind: this module's reflection carries no `[[sampler(i)]]`
    // argument at all, so the pairing has nothing to fill.
    let (vertex, fragment) = translated_pair(&executor);
    let mut contract = reuse_contract();
    contract.textures[3] = TextureBindingContract::sampled_runtime(3, TextureFormat::Rgba8Unorm, 0);
    let error = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract,
            vertex,
            fragment,
            logical_digest: digest(b"render sampler reuse, unpaired runtime sampler"),
        })
        .expect_err("a runtime sampler the module never binds is refused");
    eprintln!("unpaired-runtime refusal: {error:?}");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(error.slug, "render_runtime_sampler_unpaired");

    // The positive registration still executes on the same provider, so the
    // refusals above are not a provider that refuses everything.
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: reuse_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"render sampler reuse, after the refusals"),
        })
        .expect("the reuse registration is well formed");
    let bytes = submit(&provider, &compute, &render, reuse_resources(), Vec::new())
        .expect("the reuse shape still executes");
    assert_eq!(uniform_texel(&bytes), REUSE_READBACK);
}
