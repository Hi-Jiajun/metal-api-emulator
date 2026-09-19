//! Render-side sampling through a **pixel-coordinate** runtime sampler
//! (2026-09-19, census v43's `texture_state` axis).
//!
//! The falsifiable claim is the census's last unfrozen render state: the guest
//! binds a runtime `[[sampler(n)]]` whose `MTLSamplerDescriptor` states
//! `normalizedCoordinates = NO`, so the coordinates its shader computed are in
//! **texels** — and every earlier increment executed the normalized space or
//! kept the draw on the engine. This file states the texel space as an arm: the
//! pass states [`SamplerCoordinates::Pixel`] on the runtime sampler binding,
//! the rail creates the descriptor's `VkSampler` with
//! `unnormalizedCoordinates`, and it executes the fragment module's
//! **explicit-LOD sibling**: the same module with `OpImageSampleExplicitLod`
//! and a `Lod 0` operand at every sample site, which is what makes the
//! unnormalized sampler a legal use (`VUID-vkCmdDraw-None-08610`/`-08611`).
//!
//! The fixture is the runtime-sampler fixture: one translated fragment stage
//! takes two `[[texture(n)]]` arguments and two `[[sampler(n)]]` arguments and
//! samples at `(1.375, 0.125)` — twice — and `(0.3125, 0.125)`. Against a 4x4
//! `rgba8_unorm` texture whose column `i` holds `64 * i` in red and whose row
//! `j` holds `64 * j` in green, the texel space's own arithmetic is (the
//! translator's pixel lowering, `floor(u - 0.5)` and `frac(u - 0.5)`, which is
//! Vulkan's unnormalized linear filtering verbatim):
//!
//! * linear + clampToZero: `i0 = 0, alpha = 0.875` on both x reads, and
//!   `j0 = -1, beta = 0.625` on every y — the out-of-range row is the
//!   transparent-black border, so only the `(i1, j1)` tap carries a texel and
//!   the first two readings land `0.875 * 0.625 * 64 = 35`; the third lands a
//!   column that is only border and column zero: `23 23 00 ff`;
//! * nearest + clampToZero: `floor(1.375) = 1` and `floor(0.3125) = 0`, so the
//!   frame is `40 40 00 ff`;
//! * the *normalized* reading of the same coordinates is a different frame
//!   (`00 00 30 ff` for linear), which is the fingerprint that makes the arm
//!   falsifiable: a rail that executed the normalized space lands those bytes
//!   and not the texel ones.
//!
//! The same registration answers the sibling's own neutrality: the module the
//! registration carries and its explicit-LOD sibling, executed under one
//! normalized state, land byte-identical frames — the implicit form computes
//! its LOD from derivatives and the family's `minLod = maxLod = 0` pins it to
//! level zero, and every canonical view carries one mip level, so `Lod 0` is
//! that same level.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, ProviderError, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    RenderSamplerBinding, ResourceTableSnapshot, SamplerAddressMode, SamplerCoordinates,
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

const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const RUNTIME_ENTRY: &str = "render_sample_texture_2d_runtime";
const RUNTIME_AIR: &str = include_str!("fixtures/render_sample_texture_2d_runtime.frag.ll");
const OFFSET_ENTRY: &str = "render_sample_texture_2d_offset_sampler";
const OFFSET_AIR: &str = include_str!("fixtures/render_sample_texture_2d_offset_sampler.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(960);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(962);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(963);
const LEFT_VIEW: ViewId = ViewId::new(964);
const LEFT_ALLOCATION: AllocationId = AllocationId::new(965);
const RIGHT_VIEW: ViewId = ViewId::new(966);
const RIGHT_ALLOCATION: AllocationId = AllocationId::new(967);

const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const LINEAR_CLAMP_TO_ZERO: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::ClampToZero,
};
const NEAREST_CLAMP_TO_ZERO: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToZero,
};
const LINEAR_MIP_NEAREST_CLAMP_TO_ZERO: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::LinearMipNearest,
    address: SamplerAddressMode::ClampToZero,
};
const LINEAR_REPEAT: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::Repeat,
};
const NEAREST_CLAMP_TO_EDGE: SamplerPolicy = SamplerPolicy {
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

/// Column `i` holds `64 * i` in red, row `j` holds `64 * j` in green, full
/// alpha. Every reading the fixture takes is a multiple of eight, so the linear
/// blends quantize exactly.
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

fn runtime_declaration(metal_binding: u32, sampler_binding: u32) -> TextureBindingContract {
    TextureBindingContract {
        metal_binding,
        access: TextureAccess::Sampled,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        sampler: None,
        runtime_sampler: Some(sampler_binding),
        footprint: TextureFootprintProof::WholeView,
    }
}

fn contract(fragment_entry: &str) -> RenderPipelineContract {
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: fragment_entry.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![runtime_declaration(0, 0), runtime_declaration(1, 1)],
    }
}

/// Translate one fragment fixture beside the shared vertex stage, the way a
/// host feeding guest AIR would.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_air: &str,
    fragment_entry: &str,
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
        .compile_pipeline(&function, digest(b"render-pixel-sampler-compute"))
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
        operation_id: OperationId::new(73),
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

fn submit(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    samplers: Vec<RenderSamplerBinding>,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(
        provider,
        compute,
        render,
        vec![
            texture_view(LEFT_VIEW, LEFT_ALLOCATION, 0, false),
            texture_view(RIGHT_VIEW, RIGHT_ALLOCATION, 1, false),
        ],
        samplers,
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

/// Every fragment samples the same fixed coordinates, so the whole attachment
/// carries one colour and the four bytes are the fragment's three readings.
fn uniform_texel(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(
        bytes.len() as u64,
        u64::from(EXTENT) * u64::from(EXTENT) * 4
    );
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(chunk, texel, "the frame is uniform: {}", hex(bytes));
    }
    texel
}

fn samplers(policy: SamplerPolicy, coordinates: SamplerCoordinates) -> Vec<RenderSamplerBinding> {
    vec![
        RenderSamplerBinding::with_coordinates(0, policy, coordinates),
        RenderSamplerBinding::with_coordinates(1, policy, coordinates),
    ]
}

/// The sibling is neutral: the registration's own module and its explicit-LOD
/// sibling, under one normalized state, land byte-identical frames — and that
/// frame is the one the normalized reading of the fixture's coordinates names.
#[test]
fn the_explicit_lod_sibling_lands_the_registration_modules_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR, RUNTIME_ENTRY);
    let (sibling_vertex, sibling_stage) = translated_pair(&executor, RUNTIME_AIR, RUNTIME_ENTRY);
    let sibling = sibling_stage
        .explicit_lod_sibling()
        .expect("the runtime fixture has an explicit-LOD sibling");
    if let Ok(dir) = std::env::var("PIXEL_SAMPLER_DUMP") {
        std::fs::write(
            std::path::Path::new(&dir).join("registration.frag.spv"),
            sibling_stage.spirv(),
        )
        .expect("dump the registration module");
        std::fs::write(
            std::path::Path::new(&dir).join("sibling.frag.spv"),
            sibling.spirv(),
        )
        .expect("dump the sibling module");
    }
    assert!(fragment.executes_pixel_coordinate_samplers());
    assert!(sibling.executes_pixel_coordinate_samplers());
    let registered = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(RUNTIME_ENTRY),
            vertex,
            fragment,
            logical_digest: digest(b"pixel sampler registration"),
        })
        .expect("the registration is well formed");
    let explicit = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(RUNTIME_ENTRY),
            vertex: sibling_vertex,
            fragment: sibling,
            logical_digest: digest(b"pixel sampler sibling registration"),
        })
        .expect("the sibling registers under the same contract");

    let implicit_bytes = submit(
        &provider,
        &compute,
        &registered,
        samplers(LINEAR_CLAMP_TO_ZERO, SamplerCoordinates::Normalized),
    )
    .expect("the registration's own module executes");
    let explicit_bytes = submit(
        &provider,
        &compute,
        &explicit,
        samplers(LINEAR_CLAMP_TO_ZERO, SamplerCoordinates::Normalized),
    )
    .expect("the sibling executes");
    assert_eq!(
        hex(&implicit_bytes),
        hex(&explicit_bytes),
        "the implicit and explicit siblings are the same sample program under the family's \
         one-mip, LOD-0 create-info"
    );
    assert_eq!(
        uniform_texel(&implicit_bytes),
        [0x00, 0x00, 0x30, 0xff],
        "the normalized reading of (1.375, 0.125) is out of range — clampToZero answers zero — \
         and (0.3125, 0.125) lands a three-quarter mix of texel 1 in red"
    );
}

/// The texel space: the same registration, the same module, the same sampler
/// state — and the frame the guest's texel coordinates name.
#[test]
fn a_pixel_coordinate_sampler_samples_the_texels_the_guest_coordinates_name() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR, RUNTIME_ENTRY);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(RUNTIME_ENTRY),
            vertex,
            fragment,
            logical_digest: digest(b"pixel sampler registration"),
        })
        .expect("the registration is well formed");

    let linear = submit(
        &provider,
        &compute,
        &render,
        samplers(LINEAR_CLAMP_TO_ZERO, SamplerCoordinates::Pixel),
    )
    .expect("the texel space executes");
    assert_eq!(
        uniform_texel(&linear),
        [0x23, 0x23, 0x00, 0xff],
        "floor(1.375 - 0.5) = 0 with alpha = 0.875, and floor(0.125 - 0.5) = -1 is the \
         transparent-black border row, so the (1, 0) texel's weight is 0.875 * 0.625 = 0.546875 \
         and the reading is 35"
    );

    let nearest = submit(
        &provider,
        &compute,
        &render,
        samplers(NEAREST_CLAMP_TO_ZERO, SamplerCoordinates::Pixel),
    )
    .expect("the texel space executes");
    assert_eq!(
        uniform_texel(&nearest),
        [0x40, 0x40, 0x00, 0xff],
        "floor(1.375) = texel 1 (red 64) and floor(0.3125) = texel 0 (red 0)"
    );

    // The fingerprint: the normalized reading of the very same coordinates is
    // another frame, so an arm that executed the fraction space cannot land the
    // texel space's bytes.
    let normalized = submit(
        &provider,
        &compute,
        &render,
        samplers(NEAREST_CLAMP_TO_ZERO, SamplerCoordinates::Normalized),
    )
    .expect("the normalized space still executes");
    assert_eq!(uniform_texel(&normalized), [0x00, 0x00, 0x40, 0xff]);
    assert_ne!(hex(&nearest), hex(&normalized));
}

/// The state family an unnormalized `VkSampler` can carry: one filter for both
/// halves, no mip filtering, and the two addressing modes that answer an
/// out-of-range texel (`VUID-VkSamplerCreateInfo-unnormalizedCoordinates-01072`
/// … `-01077`). Every other state is refused by name.
#[test]
fn the_texel_space_states_only_the_states_an_unnormalized_sampler_can_carry() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, RUNTIME_AIR, RUNTIME_ENTRY);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(RUNTIME_ENTRY),
            vertex,
            fragment,
            logical_digest: digest(b"pixel sampler registration"),
        })
        .expect("the registration is well formed");

    for (what, policy) in [
        ("a mip-filtering state", LINEAR_MIP_NEAREST_CLAMP_TO_ZERO),
        (
            "an addressing mode an unnormalized sampler cannot state",
            LINEAR_REPEAT,
        ),
    ] {
        let error = submit(
            &provider,
            &compute,
            &render,
            samplers(policy, SamplerCoordinates::Pixel),
        )
        .expect_err(what);
        assert_eq!(error.slug, "render_pixel_sampler_state_unsupported");
        assert_eq!(
            error.fields.get("coordinates"),
            Some(&metal_api_core::provider::FieldValue::Text(
                "Pixel".to_owned()
            )),
            "{what} names the space it read"
        );
    }
}

/// A module with a sample form no unnormalized sampler may be used with has no
/// sibling, so the texel space is refused by name — while the normalized arm
/// keeps executing the very same registration.
#[test]
fn a_module_with_an_offset_sample_has_no_sibling_and_the_texel_space_is_refused() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor, OFFSET_AIR, OFFSET_ENTRY);
    assert!(
        fragment.explicit_lod_sibling().is_none(),
        "an offset-carrying sample is a form an unnormalized sampler may not be used with"
    );
    assert!(!fragment.executes_pixel_coordinate_samplers());
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(OFFSET_ENTRY),
            vertex,
            fragment,
            logical_digest: digest(b"offset sampler registration"),
        })
        .expect("the registration is well formed — the normalized arm is unchanged");

    let error = submit(
        &provider,
        &compute,
        &render,
        samplers(NEAREST_CLAMP_TO_ZERO, SamplerCoordinates::Pixel),
    )
    .expect_err("the texel space has no module behind it");
    assert_eq!(error.slug, "render_pixel_sampler_variant_unavailable");
    assert_eq!(
        error.fields.get("fragment_entry"),
        Some(&metal_api_core::provider::FieldValue::Text(
            OFFSET_ENTRY.to_owned()
        ))
    );

    // The normalized arm of the same registration still lands a frame: the
    // offset shifts every reading one texel to the right, and clamp-to-edge
    // answers the reading that leaves the image with the edge texel.
    let normalized = submit(
        &provider,
        &compute,
        &render,
        samplers(NEAREST_CLAMP_TO_EDGE, SamplerCoordinates::Normalized),
    )
    .expect("the normalized arm executes");
    assert_eq!(
        uniform_texel(&normalized),
        [0xc0, 0xc0, 0x80, 0xff],
        "floor(1.375 * 4 + 1) = 6 clamps to the edge column (192) and \
         floor(0.3125 * 4 + 1) = 2 lands column 2 (128)"
    );
}
