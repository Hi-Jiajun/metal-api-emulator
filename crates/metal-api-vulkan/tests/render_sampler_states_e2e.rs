//! The widened sampler state family on the render rail (`research/docs/23` §109).
//!
//! The census that followed v107 put the render track's next gate on the state
//! a *runtime* `[[sampler(n)]]` argument executes with: 233 of one boot's class
//! exits were a draw whose bind "states another filter, another address mode, a
//! mip filter, unnormalized coordinates, a comparison or anisotropy" — the
//! family the canonical rail created held only `{nearest, linear} x
//! {clamp-to-edge, repeat}` with the not-mipmapped mip filter.
//!
//! v109 widens the family to the filters `{nearest, linear}` minification and
//! magnification, each with the mip filter `{not-mipmapped, nearest, linear}`,
//! crossed with the address modes `{clamp-to-edge, repeat, mirror-clamp-to-edge,
//! mirror-repeat, clamp-to-zero}`. The readings below are one registration and
//! one translated fragment stage (the boundary fixture's three out-of-range
//! coordinates, stored as red, green and blue) against a texture whose columns
//! hold `00, 40, 80, c0` in red:
//!
//! * the five address modes land five distinct frames — `00 c0 c0 ff`,
//!   `40 c0 c0 ff`, `c0 40 40 ff`, `40 c0 40 ff` and `00 00 00 ff` — so the
//!   three names §109 appended are distinguishable from the two that were
//!   already there, and from each other;
//! * the three `Linear` mip names land *one* frame: every canonical view
//!   carries one mip level, so a stated mip filter can only name the mode level
//!   zero is selected under — the moved frame beside them is the one the
//!   address half owes (`LinearMipLinear` + mirror-repeat);
//! * the state nothing can state is refused by name, not substituted: the
//!   crate's own unit readings pin `clampToBorderColor` (no border colour in the
//!   family) and the two `bicubic` filters, and the registration gate refuses a
//!   request naming a state the translation was not lowered against with both
//!   halves.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, LoadOp, OperationId,
    PipelineId, ProviderError, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, RenderSamplerBinding, ResourceTableSnapshot, SamplerAddressMode,
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

/// The reviewed milestone vertex stage: a full-screen triangle whose vertex_id
/// positions cover the whole attachment and which forwards no varying.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const FRAGMENT_ENTRY: &str = "render_sample_texture_2d_boundary";
const BOUNDARY_AIR: &str = include_str!("fixtures/render_sample_texture_2d_boundary.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(970);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(971);
const SCRATCH_VIEW: ViewId = ViewId::new(972);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(973);
const TEXTURE_VIEW: ViewId = ViewId::new(974);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(975);

/// 4x4, the extent of the attachment and of the sampled texture: the rail's
/// reviewed window requires a texture to share the render area's extent, and
/// the fixture's fixed sample coordinates are stated for that grid.
const EXTENT: u32 = 4;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

const fn policy(filter: SamplerFilter, address: SamplerAddressMode) -> SamplerPolicy {
    SamplerPolicy { filter, address }
}

/// The five address modes (`research/docs/23` §109), one registration, one
/// fragment stage, nearest minification and magnification.
const ADDRESS_ARMS: [(&str, SamplerPolicy, [u8; 4]); 5] = [
    (
        "nearest + clamp-to-edge",
        policy(SamplerFilter::Nearest, SamplerAddressMode::ClampToEdge),
        [0x00, 0xc0, 0xc0, 0xff],
    ),
    (
        "nearest + mirror-clamp-to-edge",
        policy(
            SamplerFilter::Nearest,
            SamplerAddressMode::MirrorClampToEdge,
        ),
        [0x40, 0xc0, 0xc0, 0xff],
    ),
    (
        "nearest + repeat",
        policy(SamplerFilter::Nearest, SamplerAddressMode::Repeat),
        [0xc0, 0x40, 0x40, 0xff],
    ),
    (
        "nearest + mirror-repeat",
        policy(SamplerFilter::Nearest, SamplerAddressMode::MirrorRepeat),
        [0x40, 0xc0, 0x40, 0xff],
    ),
    (
        "nearest + clamp-to-zero",
        policy(SamplerFilter::Nearest, SamplerAddressMode::ClampToZero),
        [0x00, 0x00, 0x00, 0xff],
    ),
];

/// The mip half: the three `Linear` names land one frame on a one-mip view, and
/// the fourth arm moves that very state's frame by its address half alone.
const MIP_ARMS: [(&str, SamplerPolicy, [u8; 4]); 4] = [
    (
        "linear, mip not-mipmapped + clamp-to-edge",
        policy(SamplerFilter::Linear, SamplerAddressMode::ClampToEdge),
        [0x00, 0xc0, 0xc0, 0xff],
    ),
    (
        "linear, nearest mip + clamp-to-edge",
        policy(
            SamplerFilter::LinearMipNearest,
            SamplerAddressMode::ClampToEdge,
        ),
        [0x00, 0xc0, 0xc0, 0xff],
    ),
    (
        "linear, linear mip + clamp-to-edge",
        policy(
            SamplerFilter::LinearMipLinear,
            SamplerAddressMode::ClampToEdge,
        ),
        [0x00, 0xc0, 0xc0, 0xff],
    ),
    (
        "linear, linear mip + mirror-repeat",
        policy(
            SamplerFilter::LinearMipLinear,
            SamplerAddressMode::MirrorRepeat,
        ),
        [0x20, 0xa0, 0x20, 0xff],
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
/// `64 * j` in green, zero in blue and full alpha. Every reading the fixture
/// takes is a multiple of sixteen, so a linear blend quantizes exactly.
fn texture_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for j in 0..EXTENT {
        for i in 0..EXTENT {
            bytes.extend_from_slice(&[(64 * i) as u8, (64 * j) as u8, 0x00, 0xff]);
        }
    }
    bytes
}

fn texture_view() -> TextureView {
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
        source: TextureSource::OwnedBytes(texture_bytes()),
    }
}

/// One declaration that pairs the texture with the runtime `[[sampler(0)]]`
/// argument it reads through (`research/docs/23` §3.3, v102).
fn contract() -> RenderPipelineContract {
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
            sampler: None,
            runtime_sampler: Some(0),
            footprint: TextureFootprintProof::WholeView,
        }],
    }
}

/// Translate the fixture pair, the way a host feeding guest AIR would, and
/// record what the fragment stage reflected about its runtime sampler.
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
        "translated fragment: {} bytes, reflection entry {:?}, runtime sampler specializations {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
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
        .compile_pipeline(&function, digest(b"render-sampler-states-compute"))
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
        operation_id: OperationId::new(45),
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
    samplers: Vec<RenderSamplerBinding>,
) -> Result<Vec<u8>, ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, vec![texture_view()], samplers);
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

/// One texel of the readback: every fragment samples the same three
/// coordinates, so every texel of the 4x4 attachment carries the same colour.
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

/// One registration the fragment stage's own reflection backs: one sampled
/// texture paired with the runtime `[[sampler(0)]]` argument it reads through.
fn register_boundary_pipeline(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let (vertex, fragment) = translated_pair(executor, BOUNDARY_AIR);
    provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"sampler state family registration"),
        })
        .expect("the boundary registration is well formed")
}

/// The reading the census asked for: the five address modes the family names,
/// one registration, five distinct frames. The device's own answer to the one
/// feature an address mode needs is printed beside them.
#[test]
fn the_five_address_modes_land_five_distinct_frames() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    eprintln!(
        "device: {} mirror clamp to edge feature enabled={}",
        executor.device_name(),
        executor.supports_sampler_mirror_clamp_to_edge()
    );
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_boundary_pipeline(&provider, &executor);
    let mut landed = Vec::new();
    for (what, policy, expected) in ADDRESS_ARMS {
        let bytes = submit(
            &provider,
            &compute,
            &render,
            vec![RenderSamplerBinding::new(0, policy)],
        )
        .unwrap_or_else(|error| panic!("{what} executes: {error:?}"));
        let texel = uniform_texel(&bytes);
        eprintln!("{what}: expected {} landed {}", hex(&expected), hex(&texel));
        assert_eq!(texel, expected, "{what}");
        landed.push((what, texel));
    }
    for (index, (what, first)) in landed.iter().enumerate() {
        for (other, second) in &landed[index + 1..] {
            assert_ne!(
                first, second,
                "{what} and {other} state different address modes and cannot land the same \
                 texels: {landed:?}"
            );
        }
    }
}

/// The mip half (`research/docs/23` §109): every canonical view carries one mip
/// level, so the three `Linear` mip names land one frame — and the fourth arm
/// shows the frame a difference *can* move, through the address half alone.
#[test]
fn the_mip_names_land_the_frame_the_one_mip_view_owes() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_boundary_pipeline(&provider, &executor);
    for (what, policy, expected) in MIP_ARMS {
        let bytes = submit(
            &provider,
            &compute,
            &render,
            vec![RenderSamplerBinding::new(0, policy)],
        )
        .unwrap_or_else(|error| panic!("{what} executes: {error:?}"));
        let texel = uniform_texel(&bytes);
        eprintln!("{what}: expected {} landed {}", hex(&expected), hex(&texel));
        assert_eq!(texel, expected, "{what}");
    }
}

/// The refusal the widened family keeps by name (`research/docs/23` §109): the
/// declaration pairs the texture with `[[sampler(0)]]`, and a pass that states
/// no state there is refused rather than filled with a provider default. The
/// state's own named refusals — `clampToBorderColor`, which needs a border
/// colour the family does not name, and the two `bicubic` filters — are pinned
/// by the crate's own unit readings beside this file.
#[test]
fn the_family_keeps_its_named_refusals() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let render = register_boundary_pipeline(&provider, &executor);
    // A pass that binds no state at all for the argument the module samples
    // through: the declaration pairs the texture with `[[sampler(0)]]`, so the
    // missing state is refused by name rather than filled with a default.
    let refused = submit(&provider, &compute, &render, Vec::new())
        .expect_err("a runtime sampler the pass never states is a refusal");
    eprintln!("runtime sampler missing: refused: {refused:?}");
    assert_eq!(refused.slug, "render_runtime_sampler_missing");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.detail.as_deref(),
        Some(
            "render texture declaration 0 samples through runtime [[sampler(0)]], but the pass \
             binds no runtime sampler there"
        )
    );
}
