//! Sampler-family falsification probe: one fragment stage that carries *both*
//! sampler forms at once (`research/docs/23` §3.3, v100/v102).
//!
//! The census's `texture_sampler_family` bucket is one shape and one shape
//! only: a stage with **one AIR static sampler** (a `constexpr sampler`) beside
//! **one runtime `[[sampler(n)]]`** argument. The contract admits each binding
//! in exactly one of the two forms and nowhere states that a *stage* is limited
//! to one form; the question the probe settles is whether the *rail* executes
//! the two forms side by side in one stage or refuses the combination by name.
//!
//! The fixture answers it in the attachment's bytes. One translated fragment
//! stage samples its first texture through the module's own AIR constexpr
//! sampler (Nearest + ClampToEdge) and its second through the runtime
//! `[[sampler(0)]]` argument, storing the static sample as red and the two
//! runtime samples as green and blue. Against a texture whose column `i` holds
//! `64 * i` in red:
//!
//! * runtime nearest + clamp: `c0 c0 40 ff`;
//! * runtime nearest + repeat: `c0 40 40 ff`;
//! * runtime linear + clamp: `c0 c0 30 ff`.
//!
//! The red channel is the control: it is the static half's reading, and it is
//! the value the static-only sibling fixture lands for the same sample under
//! the same pipeline shape. A rail that folded one form into the other — or
//! that resolved the runtime half through the module's AIR state — would move
//! it.
//!
//! The remaining arms are the refusals that must survive the mix being
//! admitted: a pass that states no state for the paired runtime sampler
//! (`render_runtime_sampler_missing`), a contract that declares the mixed
//! module's runtime texture as an AIR-static one
//! (`render_stage_reflection_mismatch`), and a contract that pairs the second
//! texture with a `[[sampler(n)]]` the module never binds
//! (`render_runtime_sampler_unpaired`).

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

/// The mixed fixture: one stage, one AIR static sampler, one runtime
/// `[[sampler(0)]]`.
const MIXED_FRAGMENT_ENTRY: &str = "render_sample_texture_2d_mixed";
const MIXED_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_static_and_runtime.frag.ll");
/// The static-only baseline: the same static sampler state, one texture, no
/// runtime sampler argument at all.
const STATIC_FRAGMENT_ENTRY: &str = "render_sample_texture_2d";
const STATIC_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(960);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(962);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(963);
const STATIC_VIEW: ViewId = ViewId::new(964);
const STATIC_ALLOCATION: AllocationId = AllocationId::new(965);
const RUNTIME_VIEW: ViewId = ViewId::new(966);
const RUNTIME_ALLOCATION: AllocationId = AllocationId::new(967);

/// 4x4, the extent of the attachment and of both sampled textures.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The module's own AIR state: the reviewed milestone's Nearest + ClampToEdge.
const STATIC_POLICY: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};
const NEAREST_CLAMP: SamplerPolicy = STATIC_POLICY;
const NEAREST_REPEAT: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::Repeat,
};
const LINEAR_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::ClampToEdge,
};

/// The three states the *pass* states for the one runtime sampler, and the
/// readback each lands through the one mixed registration.
const ARMS: [(&str, SamplerPolicy, [u8; 4]); 3] = [
    (
        "runtime nearest + clamp-to-edge",
        NEAREST_CLAMP,
        [0xc0, 0xc0, 0x40, 0xff],
    ),
    (
        "runtime nearest + repeat",
        NEAREST_REPEAT,
        [0xc0, 0x40, 0x40, 0xff],
    ),
    (
        "runtime linear + clamp-to-edge",
        LINEAR_CLAMP,
        [0xc0, 0xc0, 0x30, 0xff],
    ),
];

/// The static-only sibling's own readback for the same two samples: red is the
/// clamp-to-edge reading and green the nearest one.
const STATIC_ONLY_READBACK: [u8; 4] = [0xc0, 0x40, 0x00, 0xff];

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

/// The mixed module's contract: binding 0 through the module's own AIR state,
/// binding 1 through the runtime `[[sampler(0)]]` argument — the two forms in
/// one stage, which is the shape under test.
fn mixed_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: MIXED_FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![
            TextureBindingContract::sampled(0, TextureFormat::Rgba8Unorm, STATIC_POLICY),
            TextureBindingContract::sampled_runtime(1, TextureFormat::Rgba8Unorm, 0),
        ],
    }
}

/// The static-only baseline's contract: one binding, the module's own state,
/// no runtime sampler anywhere in the stage.
fn static_contract() -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: STATIC_FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract::sampled(
            0,
            TextureFormat::Rgba8Unorm,
            STATIC_POLICY,
        )],
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would, and
/// report the reflection the rail will pair the contract's declarations with.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_entry: &str,
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
        .function(fragment_entry)
        .expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated fragment {fragment_entry}: {} bytes, bindings {:?}, runtime sampler \
         specializations {:?}",
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
        fragment.reflection().runtime_sampler_specializations,
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
        .compile_pipeline(&function, digest(b"render-sampler-family-compute"))
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
        operation_id: OperationId::new(42),
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

/// One texel of the readback: every fragment samples the same fixed
/// coordinates, so every texel of the attachment carries the same colour.
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

/// The static-only sibling, registered and submitted on its own pipeline: the
/// red channel of its readback is the reading the mixed fixture's static half
/// must reproduce.
fn static_only_baseline(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    compute: &CompiledComputePipeline,
) -> [u8; 4] {
    let (vertex, fragment) = translated_pair(executor, STATIC_FRAGMENT_ENTRY, STATIC_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: static_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"static-only sampler family baseline"),
        })
        .expect("the static-only registration is well formed");
    let bytes = submit(
        provider,
        compute,
        &render,
        vec![texture_view(STATIC_VIEW, STATIC_ALLOCATION, 0)],
        Vec::new(),
    )
    .expect("the static-only baseline executes");
    let texel = uniform_texel(&bytes);
    eprintln!(
        "static-only baseline: expected {} landed {}",
        hex(&STATIC_ONLY_READBACK),
        hex(&texel)
    );
    assert_eq!(texel, STATIC_ONLY_READBACK);
    texel
}

/// The falsifiable claim: one stage that carries an AIR static sampler *and* a
/// runtime `[[sampler(n)]]` registers, executes, and lands both halves' own
/// readings — the static half against the static-only baseline, the runtime
/// half against the state the pass states.
#[test]
fn one_stage_may_carry_an_air_static_sampler_and_a_runtime_sampler() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let baseline = static_only_baseline(&provider, &executor, &compute);

    let (vertex, fragment) = translated_pair(&executor, MIXED_FRAGMENT_ENTRY, MIXED_AIR);
    let registered =
        provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: mixed_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"mixed sampler family registration"),
        });
    let render = match registered {
        Ok(render) => render,
        Err(refusal) => panic!(
            "the mixed stage is refused by name: slug={} class={:?} fields={:?} detail={:?}",
            refusal.slug, refusal.class, refusal.fields, refusal.detail
        ),
    };

    let mut landed = Vec::new();
    for (what, policy, expected) in ARMS {
        let bytes = submit(
            &provider,
            &compute,
            &render,
            vec![
                texture_view(STATIC_VIEW, STATIC_ALLOCATION, 0),
                texture_view(RUNTIME_VIEW, RUNTIME_ALLOCATION, 1),
            ],
            vec![RenderSamplerBinding::new(0, policy)],
        )
        .unwrap_or_else(|error| panic!("{what} executes: {error:?}"));
        let texel = uniform_texel(&bytes);
        eprintln!("{what}: expected {} landed {}", hex(&expected), hex(&texel));
        assert_eq!(texel, expected, "{what}");
        // The red channel is the static half's reading, and the mix may not
        // move it: the AIR constexpr sampler, not the pass's state, decides it.
        assert_eq!(
            texel[0],
            baseline[0],
            "the static half's reading is the module's own AIR state, so the mixed stage's red \
             channel has to equal the static-only baseline's: {} vs {}",
            hex(&texel),
            hex(&baseline)
        );
        landed.push(texel);
    }
    // The runtime half is the pass's fact: three requests that differ only in
    // the state they state cannot land the same texels.
    for (index, first) in landed.iter().enumerate() {
        for second in &landed[index + 1..] {
            assert_ne!(
                first, second,
                "three states the pass states cannot land the same texels: {landed:?}"
            );
        }
    }

    // Control: the very same request run twice lands the very same bytes, so
    // the arms above measure the stated states rather than the run.
    let again = submit(
        &provider,
        &compute,
        &render,
        vec![
            texture_view(STATIC_VIEW, STATIC_ALLOCATION, 0),
            texture_view(RUNTIME_VIEW, RUNTIME_ALLOCATION, 1),
        ],
        vec![RenderSamplerBinding::new(0, NEAREST_CLAMP)],
    )
    .expect("the fully stated pass executes again");
    assert_eq!(uniform_texel(&again), landed[0]);
}

/// The mix being admissible does not widen the rail's named refusals: a pass
/// that leaves the paired runtime sampler unstated, a contract that declares
/// the runtime texture as an AIR-static one, and a contract that names a
/// `[[sampler(n)]]` the module does not bind each keep their own slug.
#[test]
fn the_mixed_stage_still_answers_each_disagreement_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, MIXED_FRAGMENT_ENTRY, MIXED_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: mixed_contract(),
            vertex,
            fragment,
            logical_digest: digest(b"mixed sampler family refusals"),
        })
        .expect("the mixed registration is well formed");

    // The request's half: the module samples through `[[sampler(0)]]`, so a
    // pass that states no state for it would fill the descriptor with a
    // sampler nobody named.
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        vec![
            texture_view(STATIC_VIEW, STATIC_ALLOCATION, 0),
            texture_view(RUNTIME_VIEW, RUNTIME_ALLOCATION, 1),
        ],
        Vec::new(),
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the paired runtime sampler is unstated");
    eprintln!("unstated runtime sampler beside a static one: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_runtime_sampler_missing");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);

    // The contract's half, direction one: the module's second texture is read
    // through the runtime argument, so declaring it against the module's AIR
    // static sampler is the declaration disagreeing with the module.
    let (vertex, fragment) = translated_pair(&executor, MIXED_FRAGMENT_ENTRY, MIXED_AIR);
    let mut statically_declared = mixed_contract();
    statically_declared.textures[1] =
        TextureBindingContract::sampled(1, TextureFormat::Rgba8Unorm, STATIC_POLICY);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: statically_declared,
            vertex,
            fragment,
            logical_digest: digest(b"mixed module declared all static"),
        })
        .expect_err("the module carries one AIR static sampler, not two");
    eprintln!("runtime texture declared static: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_reflection_mismatch");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert_eq!(refusal.fields.get("index"), Some(&FieldValue::Unsigned(1)));

    // The contract's half, direction two: the module binds one runtime
    // sampler, and a declaration naming another index pairs with nothing.
    let (vertex, fragment) = translated_pair(&executor, MIXED_FRAGMENT_ENTRY, MIXED_AIR);
    let mut unpaired = mixed_contract();
    unpaired.textures[1].runtime_sampler = Some(1);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: unpaired,
            vertex,
            fragment,
            logical_digest: digest(b"mixed module with an unpaired runtime sampler"),
        })
        .expect_err("the module binds no `[[sampler(1)]]`");
    eprintln!("unpaired runtime sampler: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_runtime_sampler_unpaired");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert_eq!(
        refusal.fields.get("sampler_binding"),
        Some(&FieldValue::Unsigned(1))
    );

    // Control: the same registration and the same pass, stated, lands its
    // bytes — so the three refusals above are about the disagreement rather
    // than about the mixed shape itself.
    let bytes = submit(
        &provider,
        &compute,
        &render,
        vec![
            texture_view(STATIC_VIEW, STATIC_ALLOCATION, 0),
            texture_view(RUNTIME_VIEW, RUNTIME_ALLOCATION, 1),
        ],
        vec![RenderSamplerBinding::new(0, NEAREST_CLAMP)],
    )
    .expect("the fully stated mixed pass executes");
    assert_eq!(uniform_texel(&bytes), [0xc0, 0xc0, 0x40, 0xff]);
}
