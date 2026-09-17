//! Translated render-side sampling through runtime `[[sampler(n)]]` arguments
//! (`research/docs/23` §3.3, v102).
//!
//! The falsifiable claim is the census's first blocking shape, one face over
//! from E-RS1: a fragment stage that samples its textures through *runtime*
//! sampler arguments — the state Metal binds when the draw is encoded, not the
//! `constexpr sampler` an AIR module carries — is executed with the state the
//! **pass** states, and two textures can read through two samplers of different
//! states in one draw.
//!
//! One translated fragment stage takes two `[[texture(n)]]` arguments and two
//! `[[sampler(n)]]` arguments, samples the left texture once and the right
//! texture twice, and stores the three component-zero readings as red, green
//! and blue. Against a texture whose column `i` holds `64 * i` in red:
//!
//! * nearest+clamp on both: `c0 c0 40 ff` — `u = 1.375` clamps to the edge
//!   texel in red and green, and `u = 0.3125` lands in texel 1 for blue;
//! * nearest+repeat on both: `40 40 40 ff` — `u = 1.375` wraps back to texel 1;
//! * nearest+clamp on the first, linear+clamp on the second: `c0 c0 30 ff` —
//!   the second sampler's filtering blends a quartile of texel 0 into texel 1.
//!
//! The three arms use **one registration**: the state is a request fact, so the
//! same pipeline executes under three states and the attachment's bytes are the
//! only thing that changes. The remaining arms pin the refusals: a declaration
//! that pairs a texture with a `[[sampler(n)]]` the module never binds
//! (`render_runtime_sampler_unpaired`), a module whose sampler argument no
//! declaration pairs with (`render_runtime_sampler_undeclared`), a pass that
//! leaves a paired sampler unstated (`render_runtime_sampler_missing`, with the
//! descriptor slot), and the contract's own ceiling on the declaration list
//! (`render_texture_limit`).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, RenderSamplerBinding, ResourceTableSnapshot,
    SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess,
    TextureBindingContract, TextureFootprintProof, TextureFormat, TextureSource, TextureType,
    TextureView, TracePass, VertexLayout, ViewId, MAX_RENDER_TEXTURES, PROVIDER_SCHEMA_VERSION,
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

const FRAGMENT_ENTRY: &str = "render_sample_texture_2d_runtime";
const RUNTIME_AIR: &str = include_str!("fixtures/render_sample_texture_2d_runtime.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(950);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(951);
const SCRATCH_VIEW: ViewId = ViewId::new(952);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(953);
const LEFT_VIEW: ViewId = ViewId::new(954);
const LEFT_ALLOCATION: AllocationId = AllocationId::new(955);
const RIGHT_VIEW: ViewId = ViewId::new(956);
const RIGHT_ALLOCATION: AllocationId = AllocationId::new(957);

/// 4x4, the extent of the attachment and of both sampled textures: the rail's
/// reviewed window requires a texture to share the render area's extent, and
/// the fixture's fixed sample coordinates are stated for that grid.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};
const NEAREST_REPEAT: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::Repeat,
};
const LINEAR_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::ClampToEdge,
};

/// The three state pairs the request states, and the readback each one lands
/// through the one registration.
const ARMS: [(&str, SamplerPolicy, SamplerPolicy, [u8; 4]); 3] = [
    (
        "both samplers nearest + clamp-to-edge",
        NEAREST_CLAMP,
        NEAREST_CLAMP,
        [0xc0, 0xc0, 0x40, 0xff],
    ),
    (
        "both samplers nearest + repeat",
        NEAREST_REPEAT,
        NEAREST_REPEAT,
        [0x40, 0x40, 0x40, 0xff],
    ),
    (
        "edge sampler clamped, right sampler linear",
        NEAREST_CLAMP,
        LINEAR_CLAMP,
        [0xc0, 0xc0, 0x30, 0xff],
    ),
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

/// The sampled textures' sixteen texels: column `i` holds `64 * i` in red,
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

fn texture_view(
    view_id: ViewId,
    allocation_id: AllocationId,
    metal_binding: u32,
    descending: bool,
) -> TextureView {
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
        source: TextureSource::OwnedBytes(texture_bytes(descending)),
    }
}

/// One declaration that pairs a texture with the runtime `[[sampler(n)]]`
/// argument it reads through (`research/docs/23` §3.3, v102).
fn runtime_declaration(
    metal_binding: u32,
    sampler_binding: u32,
    footprint: TextureFootprintProof,
) -> TextureBindingContract {
    TextureBindingContract {
        metal_binding,
        access: TextureAccess::Sampled,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        sampler: None,
        runtime_sampler: Some(sampler_binding),
        footprint,
    }
}

/// The contract the fixture's own reflection backs: two sampled textures, each
/// paired with its own runtime sampler argument.
fn contract(footprint: TextureFootprintProof) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![
            runtime_declaration(0, 0, footprint),
            runtime_declaration(1, 1, footprint),
        ],
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
        "translated fragment: {} bytes, reflection entry {:?}, bindings {:?}, runtime sampler \
         specializations {:?}",
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
        .compile_pipeline(&function, digest(b"render-runtime-sampler-compute"))
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

/// Submit one pass through the registered pipeline and return the attachment's
/// readback bytes.
fn submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    left: TextureView,
    right: TextureView,
    samplers: Vec<RenderSamplerBinding>,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, vec![left, right], samplers);
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

fn sampled_pair(left_descending: bool, right_descending: bool) -> (TextureView, TextureView) {
    (
        texture_view(LEFT_VIEW, LEFT_ALLOCATION, 0, left_descending),
        texture_view(RIGHT_VIEW, RIGHT_ALLOCATION, 1, right_descending),
    )
}

fn sampler_bindings(first: SamplerPolicy, second: SamplerPolicy) -> Vec<RenderSamplerBinding> {
    vec![
        RenderSamplerBinding::new(0, first),
        RenderSamplerBinding::new(1, second),
    ]
}

/// One registration, three states: the two textures read through the two
/// runtime samplers the *request* states, and each state pair lands the texels
/// its own filtering and addressing rule names.
#[test]
fn two_textures_read_through_the_runtime_samplers_the_pass_states() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"runtime sampler registration"),
        })
        .expect("the runtime sampler registration is well formed");
    let (left, right) = sampled_pair(false, false);
    let mut landed = Vec::new();
    for (what, first, second, expected) in ARMS {
        let bytes = submit(
            &provider,
            &compute,
            &render,
            left.clone(),
            right.clone(),
            sampler_bindings(first, second),
        )
        .unwrap_or_else(|error| panic!("{what} executes: {error:?}"));
        let texel = uniform_texel(&bytes);
        eprintln!("{what}: expected {} landed {}", hex(&expected), hex(&texel));
        assert_eq!(texel, expected, "{what}");
        landed.push(texel);
    }
    for (index, first) in landed.iter().enumerate() {
        for second in &landed[index + 1..] {
            assert_ne!(
                first, second,
                "two requests that differ only in the states they state cannot land the same \
                 texels: {landed:?}"
            );
        }
    }
}

/// Changing either texture's own bytes changes the readback under one and the
/// same registration and one and the same sampler states, and running the very
/// same request twice lands the very same bytes — the identity a state
/// difference is measured against.
#[test]
fn the_sampled_bytes_follow_the_uploaded_texels() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"runtime sampler payloads"),
        })
        .expect("the registration is well formed");
    let samplers = sampler_bindings(NEAREST_CLAMP, NEAREST_CLAMP);
    let (left, right) = sampled_pair(false, false);
    let ascending = submit(
        &provider,
        &compute,
        &render,
        left.clone(),
        right.clone(),
        samplers.clone(),
    )
    .expect("the ascending payloads execute");
    // The right texture's columns run the other way, so its two readings move
    // (the clamped sample reads the edge texel, which is column 3 ascending and
    // column 0 descending) while the left texture's reading stays put.
    let (_, descending_right) = sampled_pair(false, true);
    let reversed = submit(
        &provider,
        &compute,
        &render,
        left,
        descending_right,
        samplers.clone(),
    )
    .expect("the descending payload executes");
    let ascending = uniform_texel(&ascending);
    let reversed = uniform_texel(&reversed);
    eprintln!(
        "ascending payloads: {} descending right texture: {}",
        hex(&ascending),
        hex(&reversed)
    );
    assert_eq!(ascending, [0xc0, 0xc0, 0x40, 0xff]);
    assert_eq!(reversed, [0xc0, 0x00, 0x80, 0xff]);

    // Control: the very same request run twice lands the very same bytes, so
    // the arms above measure the payload and the states rather than the run.
    let (left, right) = sampled_pair(false, false);
    let again = submit(&provider, &compute, &render, left, right, samplers)
        .expect("the ascending payloads execute again");
    eprintln!(
        "ascending payloads, second run: {}",
        hex(&uniform_texel(&again))
    );
    assert_eq!(uniform_texel(&again), ascending);
}

/// The registration's own half of the pairing: a declaration that names a
/// `[[sampler(n)]]` the module never binds, and a module whose sampler argument
/// no declaration pairs with, each keep their own named refusal.
#[test]
fn the_pairs_rules_name_each_disagreement() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };

    // The module binds `[[sampler(0)]]` and `[[sampler(1)]]`; a declaration
    // that pairs the second texture with index 2 names a descriptor nobody
    // samples through.
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    let mut unpaired = contract(TextureFootprintProof::WholeView);
    unpaired.textures[1].runtime_sampler = Some(2);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: unpaired,
            vertex,
            fragment,
            logical_digest: digest(b"unpaired runtime sampler"),
        })
        .expect_err("the module binds no sampler at that index");
    eprintln!("unpaired runtime sampler: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_runtime_sampler_unpaired");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert_eq!(
        refusal.fields.get("binding"),
        Some(&FieldValue::Unsigned(1))
    );
    assert_eq!(
        refusal.fields.get("sampler_binding"),
        Some(&FieldValue::Unsigned(2))
    );

    // The other direction: both declarations pair with `[[sampler(0)]]`, which
    // the module does bind, and leave `[[sampler(1)]]` — the argument the
    // second texture's samples actually read through — paired by nothing.
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    let mut undeclared = contract(TextureFootprintProof::WholeView);
    undeclared.textures[1].runtime_sampler = Some(0);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: undeclared,
            vertex,
            fragment,
            logical_digest: digest(b"undeclared runtime sampler"),
        })
        .expect_err("the module binds a sampler no declaration pairs with");
    eprintln!("undeclared runtime sampler: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_runtime_sampler_undeclared");
    assert_eq!(
        refusal.fields.get("sampler_binding"),
        Some(&FieldValue::Unsigned(1))
    );
    assert_eq!(
        refusal.fields.get("descriptor"),
        Some(&FieldValue::Unsigned(161))
    );

    // Control: the pairing the module backs registers, so the two refusals
    // above are about the pairing rather than about the shape.
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"runtime sampler control"),
        })
        .expect("the module's own pairing registers");
}

/// The request's own half: a pass that binds no state for a paired runtime
/// sampler is refused by name — the descriptor would be filled with a sampler
/// nobody stated — and the contract's ceiling refuses a declaration list above
/// [`MAX_RENDER_TEXTURES`].
#[test]
fn a_request_that_leaves_a_runtime_sampler_unstated_is_refused_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"runtime sampler request"),
        })
        .expect("the registration is well formed");

    // One state, two paired samplers: `[[sampler(1)]]` stays unbound.
    let (left, right) = sampled_pair(false, false);
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &render,
        vec![left.clone(), right.clone()],
        vec![RenderSamplerBinding::new(0, NEAREST_CLAMP)],
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the declaration pairs a sampler the pass does not bind");
    eprintln!("unbound runtime sampler: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_runtime_sampler_missing");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("[[sampler(1)]]")),
        "{refusal:?}"
    );

    // The declaration list's own ceiling (`v102`): one entry per texture the
    // fixture names is inside it, and the entry above [`MAX_RENDER_TEXTURES`] is
    // refused before any of them is resolved.
    let mut wide = contract(TextureFootprintProof::WholeView);
    for binding in 2..=MAX_RENDER_TEXTURES as u32 {
        wide.textures.push(runtime_declaration(
            binding,
            binding,
            TextureFootprintProof::WholeView,
        ));
    }
    let mut wide_pipeline = render.clone();
    wide_pipeline
        .render
        .as_mut()
        .expect("the entry renders")
        .textures = wide.textures;
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &wide_pipeline,
        vec![left, right],
        sampler_bindings(NEAREST_CLAMP, NEAREST_CLAMP),
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a declaration list above the contract cap is refused");
    eprintln!("one sampled texture past the cap: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_limit");

    // The declaration's reach has to be provable, exactly as the static
    // half's must (`v100`): an unbounded runtime pair is refused under the
    // contract's own capability slug before any rail runs.
    let mut unbounded = render.clone();
    unbounded
        .render
        .as_mut()
        .expect("the entry renders")
        .textures[1]
        .footprint = TextureFootprintProof::Unbounded;
    let (left, right) = sampled_pair(false, false);
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &unbounded,
        vec![left, right],
        sampler_bindings(NEAREST_CLAMP, NEAREST_CLAMP),
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("an unbounded runtime pair has no provable reach");
    eprintln!("unbounded runtime pair: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_footprint_unsupported");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);

    // Control: the very same pass with both states lands its bytes, so the
    // refusal above is about the missing statement rather than the shape.
    let (left, right) = sampled_pair(false, false);
    let bytes = submit(
        &provider,
        &compute,
        &render,
        left,
        right,
        sampler_bindings(NEAREST_CLAMP, NEAREST_CLAMP),
    )
    .expect("the fully stated pass executes");
    assert_eq!(uniform_texel(&bytes), [0xc0, 0xc0, 0x40, 0xff]);
}
