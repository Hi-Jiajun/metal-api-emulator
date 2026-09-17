//! Translated render-side texture sampling with the module's own sampler state
//! (`research/docs/23` §3.3, v100).
//!
//! The falsifiable claim is C1b's, one face over: the sampler state the
//! *fragment module's own AIR* carries is the state the pass executes with, and
//! it is observable in the attachment's bytes. Three translations of one
//! fragment stage differ only in their `@__air_sampler_state` word — nearest
//! with clamped addressing, nearest with repeat, and linear with clamped
//! addressing — and each samples a 4x4 `rgba8_unorm` texture twice at fixed
//! coordinates, storing the two component-zero readings as red and green.
//!
//! Against a texture whose column `i` holds `64 * i` in red, the three states
//! land three different texels:
//!
//! * nearest + clamp-to-edge: `c0 40 00 ff` — `u = 1.375` clamps to the edge
//!   texel, and `u = 0.3125` falls in texel 1;
//! * nearest + repeat: `40 40 00 ff` — `u = 1.375` wraps to texel 1;
//! * linear + clamp-to-edge: `c0 30 00 ff` — the second sample blends a quartile
//!   of texel 0 into texel 1.
//!
//! The refusal half pins the *declaration*: the registration's contract has to
//! repeat the module's own state, and a request that declares another one is
//! refused by name (`render_texture_sampler_unsupported`) with the binding and
//! both state halves — the rail creates one `VkSampler` per declaration, so a
//! declaration that disagreed would silently change which texels a sample
//! returns.
//!
//! The remaining arms are the contract's own pair rules: a pass that binds a
//! texture its registration never declared, a declaration the pass never binds,
//! a format the two sides disagree on and a declaration whose reach is
//! unbounded each have their own named refusal, and each arrives before any
//! device object exists.

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
const NEAREST_CLAMP_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");
const NEAREST_REPEAT_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_nearest_repeat.frag.ll");
const LINEAR_CLAMP_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_linear_clamp.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(940);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(941);
const SCRATCH_VIEW: ViewId = ViewId::new(942);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(943);
const TEXTURE_VIEW: ViewId = ViewId::new(944);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(945);

/// 4x4, the extent of both the attachment and the sampled texture: the rail's
/// reviewed window requires the two to agree.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The three states the fixtures carry, and the readback each one lands.
const ARMS: [(&str, &str, SamplerPolicy, [u8; 4]); 3] = [
    (
        "nearest + clamp-to-edge",
        NEAREST_CLAMP_AIR,
        SamplerPolicy {
            filter: SamplerFilter::Nearest,
            address: SamplerAddressMode::ClampToEdge,
        },
        [0xc0, 0x40, 0x00, 0xff],
    ),
    (
        "nearest + repeat",
        NEAREST_REPEAT_AIR,
        SamplerPolicy {
            filter: SamplerFilter::Nearest,
            address: SamplerAddressMode::Repeat,
        },
        [0x40, 0x40, 0x00, 0xff],
    ),
    (
        "linear + clamp-to-edge",
        LINEAR_CLAMP_AIR,
        SamplerPolicy {
            filter: SamplerFilter::Linear,
            address: SamplerAddressMode::ClampToEdge,
        },
        [0xc0, 0x30, 0x00, 0xff],
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

/// The contract a declaring registration states: the fixture's own two entries,
/// one `rgba8_unorm` attachment, no vertex stream, and the texture declaration
/// the registration repeats.
fn contract(sampler: SamplerPolicy, footprint: TextureFootprintProof) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract {
            metal_binding: 0,
            access: TextureAccess::Sampled,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            sampler: Some(sampler),
            footprint,
        }],
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
        .compile_pipeline(&function, digest(b"render-texture-sampler-compute"))
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
        operation_id: OperationId::new(31),
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

/// Register the translated pair under a declaring contract, then submit one
/// trace and return the attachment's readback bytes.
// The parameter list is the request's own declaration surface — the pair, the
// declaration, the payload and the label — the same shape the rail's own tests
// spell out rather than wrap.
#[allow(clippy::too_many_arguments)]
fn readback(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    compute: &CompiledComputePipeline,
    fragment_air: &str,
    sampler: SamplerPolicy,
    footprint: TextureFootprintProof,
    descending: bool,
    what: &str,
) -> Result<Vec<u8>, ProviderError> {
    let (vertex, fragment) = translated_pair(executor, fragment_air);
    let render = provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(sampler, footprint),
        vertex,
        fragment,
        logical_digest: digest(what.as_bytes()),
    })?;
    let (trace, resources) = trace_for(
        provider,
        compute,
        &render,
        vec![sampled_texture_view(descending)],
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

/// One texel of the readback, which is what the fragment's fixed-coordinate
/// samples land: every texel of the 4x4 attachment carries the same colour,
/// because the stage samples the same two coordinates for every fragment.
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
            "every fragment samples the same two coordinates, so every texel has to carry the \
             same colour: {}",
            hex(bytes)
        );
    }
    texel
}

/// The three AIR states land three different texels, and each one is the texel
/// the state's own filtering and addressing rule names.
#[test]
fn the_module_s_own_sampler_state_lands_the_texels_it_names() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let mut landed = Vec::new();
    for (what, air, sampler, expected) in ARMS {
        let bytes = readback(
            &provider,
            &executor,
            &compute,
            air,
            sampler,
            TextureFootprintProof::WholeView,
            false,
            what,
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
                "two declarations that differ only in their sampler state cannot land the same \
                 texels: {landed:?}"
            );
        }
    }
}

/// Changing the texture's own bytes changes the readback, under one and the same
/// declaration: the identity the falsification needs before any state question
/// is asked.
#[test]
fn the_sampled_bytes_follow_the_uploaded_texels() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let sampler = ARMS[0].2;
    let ascending = readback(
        &provider,
        &executor,
        &compute,
        NEAREST_CLAMP_AIR,
        sampler,
        TextureFootprintProof::WholeView,
        false,
        "texture ascending",
    )
    .expect("the ascending payload executes");
    let descending = readback(
        &provider,
        &executor,
        &compute,
        NEAREST_CLAMP_AIR,
        sampler,
        TextureFootprintProof::WholeView,
        true,
        "texture descending",
    )
    .expect("the descending payload executes");
    let ascending = uniform_texel(&ascending);
    let descending = uniform_texel(&descending);
    eprintln!(
        "ascending texture: {} descending texture: {}",
        hex(&ascending),
        hex(&descending)
    );
    // The clamped sample reads the last column (192 ascending, 0 descending)
    // and the filtered one texel 1 (64 ascending, 128 descending), so both the
    // red and the green channel move with the payload.
    assert_eq!(ascending, [0xc0, 0x40, 0x00, 0xff]);
    assert_eq!(descending, [0x00, 0x80, 0x00, 0xff]);

    // Control: the very same request run twice lands the very same bytes, so
    // the arm above measures the payload rather than the run.
    let again = readback(
        &provider,
        &executor,
        &compute,
        NEAREST_CLAMP_AIR,
        sampler,
        TextureFootprintProof::WholeView,
        false,
        "texture ascending",
    )
    .expect("the ascending payload executes again");
    eprintln!(
        "ascending texture, second run: {}",
        hex(&uniform_texel(&again))
    );
    assert_eq!(uniform_texel(&again), ascending);
}

/// The declaration has to repeat the module's own state: another filtering or
/// addressing mode is refused by name, with both halves, before any device
/// object exists.
#[test]
fn a_declaration_that_does_not_repeat_the_module_is_refused_by_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    for (what, air, module, declared) in [
        (
            "linear module under a nearest declaration",
            LINEAR_CLAMP_AIR,
            SamplerPolicy {
                filter: SamplerFilter::Linear,
                address: SamplerAddressMode::ClampToEdge,
            },
            SamplerPolicy {
                filter: SamplerFilter::Nearest,
                address: SamplerAddressMode::ClampToEdge,
            },
        ),
        (
            "nearest module under a repeat declaration",
            NEAREST_CLAMP_AIR,
            SamplerPolicy {
                filter: SamplerFilter::Nearest,
                address: SamplerAddressMode::ClampToEdge,
            },
            SamplerPolicy {
                filter: SamplerFilter::Nearest,
                address: SamplerAddressMode::Repeat,
            },
        ),
    ] {
        let (vertex, fragment) = translated_pair(&executor, air);
        let refusal = provider
            .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
                contract: contract(declared, TextureFootprintProof::WholeView),
                vertex,
                fragment,
                logical_digest: digest(what.as_bytes()),
            })
            .expect_err("the declaration does not repeat the module");
        eprintln!("{what}: refused: {refusal:?}");
        assert_eq!(refusal.slug, "render_texture_sampler_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(
            refusal.fields.get("binding"),
            Some(&FieldValue::Unsigned(0))
        );
        assert_eq!(
            refusal.fields.get("filter"),
            Some(&FieldValue::Text(format!("{:?}", declared.filter)))
        );
        assert_eq!(
            refusal.fields.get("address"),
            Some(&FieldValue::Text(format!("{:?}", declared.address)))
        );
        assert_eq!(
            refusal.fields.get("module_filter"),
            Some(&FieldValue::Text(format!("{:?}", module.filter)))
        );
        assert_eq!(
            refusal.fields.get("module_address"),
            Some(&FieldValue::Text(format!("{:?}", module.address)))
        );
        // Control: the module's own state registers, so the refusal above is
        // about the state rather than about the pair.
        let (vertex, fragment) = translated_pair(&executor, air);
        provider
            .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
                contract: contract(module, TextureFootprintProof::WholeView),
                vertex,
                fragment,
                logical_digest: digest(format!("{what} control").as_bytes()),
            })
            .expect("the module's own state registers");
    }
}

/// The contract's own pair rules, each with its own named refusal: a pass that
/// binds a texture the registration never declared, a declaration whose reach
/// is unbounded, a declaration the pass never binds, and a format the two sides
/// disagree on.
#[test]
fn the_pair_rules_name_each_disagreement() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let sampler = ARMS[0].2;

    // Registration refuses an unbounded declaration under the rail's own
    // contract slug, and admission answers with the declaration's own
    // capability slug (`render_texture_footprint_unsupported`): this arm
    // measures the second, which is the one the contract publishes.
    let (vertex, fragment) = translated_pair(&executor, NEAREST_CLAMP_AIR);
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(sampler, TextureFootprintProof::Unbounded),
            vertex,
            fragment,
            logical_digest: digest(b"unbounded declaration"),
        })
        .expect_err("an unbounded render texture declaration is refused");
    eprintln!("unbounded declaration at registration: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_pipeline_contract_invalid");

    // The executable registration the remaining arms are measured against.
    let (vertex, fragment) = translated_pair(&executor, NEAREST_CLAMP_AIR);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(sampler, TextureFootprintProof::WholeView),
            vertex,
            fragment,
            logical_digest: digest(b"declaring registration"),
        })
        .expect("the declaring registration is well formed");

    // The same declaration through admission: the contract's own refusal, which
    // is the slug the trace-level report carries.
    let mut unbounded = render.clone();
    let render_contract = unbounded.render.as_mut().expect("the entry renders");
    render_contract.textures[0].footprint = TextureFootprintProof::Unbounded;
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &unbounded,
        vec![sampled_texture_view(false)],
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("an unbounded reach is refused by name");
    eprintln!("unbounded declaration at admission: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_texture_footprint_unsupported");
    assert_eq!(refusal.class, ProviderErrorClass::Capability);

    // A registration whose contract declares no texture, under a module that
    // reads one, is refused at registration: the module's reflection and the
    // declaration are held to each other before a pipeline id exists.
    let (vertex, fragment) = translated_pair(&executor, NEAREST_CLAMP_AIR);
    let mut undeclaring = contract(sampler, TextureFootprintProof::WholeView);
    undeclaring.textures = Vec::new();
    let refusal = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: undeclaring,
            vertex,
            fragment,
            logical_digest: digest(b"undeclaring registration"),
        })
        .expect_err("the module reads a texture this registration does not declare");
    eprintln!("undeclared module binding: refused: {refusal:?}");
    assert_eq!(refusal.slug, "render_stage_reflection_mismatch");
    assert_eq!(
        refusal.fields.get("field"),
        Some(&FieldValue::Text("textures".to_owned()))
    );
    assert_eq!(refusal.fields.get("index"), Some(&FieldValue::Unsigned(0)));

    // The trace-level half of the same question: a pass that binds a texture
    // whose pipeline entry declares none is refused by the contract's pair
    // rules, before any rail runs.
    let mut silent = render.clone();
    silent
        .render
        .as_mut()
        .expect("the entry renders")
        .textures
        .clear();
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &silent,
        vec![sampled_texture_view(false)],
    );
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the pass binds a texture the registration never declared");
    eprintln!("undeclared binding: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("declares no texture there")),
        "{refusal:?}"
    );

    // The declaration the pass never binds.
    let (trace, resources) = trace_for(&provider, &compute, &render, Vec::new());
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the declaration names a binding the pass never fills");
    eprintln!("unbound declaration: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("declares texture binding 0")),
        "{refusal:?}"
    );

    // The two sides disagree about the format: the declaration states
    // rgba8_unorm and the pass binds bgra8_unorm.
    let mut other = sampled_texture_view(false);
    other.format = TextureFormat::Bgra8Unorm;
    other.source = TextureSource::OwnedBytes(texture_bytes(false));
    let (trace, resources) = trace_for(&provider, &compute, &render, vec![other]);
    let refusal = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("the declaration and the binding state two formats");
    eprintln!("format disagreement: refused: {refusal:?}");
    assert_eq!(refusal.slug, "trace_contract_invalid");
    assert!(
        refusal.detail.as_deref().is_some_and(|detail| {
            detail.contains("Bgra8Unorm") && detail.contains("Rgba8Unorm")
        }),
        "{refusal:?}"
    );
}
