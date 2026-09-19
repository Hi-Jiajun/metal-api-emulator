//! The pooled sampled-texture backing's own rail cases
//! (`crate::render_texture_pool`).
//!
//! The increment hands a sampled declaration the image, its memory and its view
//! that a declaration of the *same shape* built before. That is only allowed to
//! be a timing change, so this file is the byte-level oracle the increment's own
//! report reads:
//!
//! * one sampled shape runs twice against one provider: the second pass must be
//!   served from the pool (every declaration a hit) and must publish the *same
//!   frame bytes*;
//! * the same two passes must publish the same bytes with the mechanism
//!   switched **off**, so the two arms of the reading are compared with each
//!   other and not only with themselves;
//! * a pass whose declarations take backings a *previous* pass uploaded must
//!   still land its own texels: the frame is the payload of the arm that drew
//!   it, which is the one failure direction a pooled host-visible image could
//!   have ("the next pass samples what the last one wrote there");
//! * switching the mechanism off must drop what it held, and the counters that
//!   name the directions (`hits`, `misses`, `disabled`) must be readable from
//!   the provider, so a round that shows no reuse can tell "the shapes never
//!   repeated" from "the pool refused them".

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
    RenderStage, RenderTexturePoolCounts, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
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

/// The sampled shape this file runs: thirteen `texture2d<float, sample>`
/// arguments, one per Metal index 0 through 12 — the largest per-stage table
/// the guest desktop was observed to bind, and therefore the one whose backing
/// the pool has the most to save.
const THIRTEEN_AIR: &str = include_str!("fixtures/render_sample_thirteen_textures.frag.ll");
const THIRTEEN_ENTRY: &str = "render_sample_thirteen_textures";
const TEXTURE_COUNT: u32 = 13;

const ATTACHMENT_VIEW: ViewId = ViewId::new(2000);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(2001);
const SCRATCH_VIEW: ViewId = ViewId::new(2002);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(2003);
/// One view and one allocation per sampled declaration: a pass that samples one
/// view twice is the duplicate-identity refusal rather than the shape this file
/// measures.
const TEXTURE_VIEW_BASE: u64 = 2010;
const TEXTURE_ALLOCATION_BASE: u64 = 2050;

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
/// The frame the same module lands when every texture carries `0x80` instead:
/// `red` sums to `13 * 128 / 255`, which clamps at one, and both pinned
/// channels carry the arm's own payload. A pooled backing that kept the first
/// arm's texels would land `THIRTEEN_FRAME` here instead.
const SECOND_ARM_FRAME: [u8; 4] = [0xff, 0x80, 0x80, 0xff];

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
/// holds `red` in its red channel, and the other texels repeat it, so the
/// reading is the texture's own payload rather than a texel the coordinate
/// picked.
fn texture_bytes(red: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for _ in 0..(EXTENT * EXTENT) {
        bytes.extend_from_slice(&[red, 0x00, 0x00, 0xff]);
    }
    bytes
}

/// One sampled texture bound at `index`, carrying `red` in every texel.
fn sampled_texture_view(index: u32, red: u8) -> TextureView {
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
        source: TextureSource::OwnedBytes(texture_bytes(red)),
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
        .compile_pipeline(&function, digest(b"render-texture-pool-compute"))
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
        operation_id: OperationId::new(91),
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

/// The two arms one reading compares, registered once.
struct Fixture {
    provider: VulkanComputeProvider,
    compute: CompiledComputePipeline,
    render: CompiledComputePipeline,
}

fn fixture() -> Option<Fixture> {
    let (executor, provider) = executor_and_provider()?;
    let compute = compile_declaring_kernel(&provider, &executor);
    let (vertex, fragment) = translated_pair(&executor);
    let render = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: contract(),
            vertex,
            fragment,
            logical_digest: digest(b"pooled sampled texture backing"),
        })
        .expect("the thirteen-declaration registration is well formed");
    Some(Fixture {
        provider,
        compute,
        render,
    })
}

impl Fixture {
    /// Submit one thirteen-texture pass, taking declaration `index`'s red
    /// channel from `red`.
    fn run_pass(&self, red: impl Fn(u32) -> u8) -> [u8; 4] {
        let bytes = admit_and_submit(
            &self.provider,
            &self.compute,
            &self.render,
            (0..TEXTURE_COUNT)
                .map(|index| sampled_texture_view(index, red(index)))
                .collect(),
        )
        .expect("the thirteen-texture pass executes");
        uniform_texel(&bytes)
    }
}

/// One shape runs twice: the second pass takes every backing the first one
/// built, and the frame it lands is the frame the fresh path landed.
#[test]
fn a_repeated_shape_is_served_from_the_pool_and_lands_the_same_bytes() {
    let Some(fixture) = fixture() else {
        return;
    };
    let provider = &fixture.provider;

    // Arm one: the mechanism on, which is its default.
    provider.set_render_texture_pool(true);
    let before: RenderTexturePoolCounts = provider.render_texture_pool_counts();
    let indexed = |index: u32| u8::try_from(index + 1).expect("thirteen payloads fit one byte");
    let first = fixture.run_pass(indexed);
    let after_first = provider.render_texture_pool_counts();
    let second = fixture.run_pass(indexed);
    let after_second = provider.render_texture_pool_counts();

    eprintln!(
        "pool on: first {first:02x?} second {second:02x?}; hits {} -> {} -> {}, misses {}, \
         entries {}, held {} bytes",
        before.hits,
        after_first.hits,
        after_second.hits,
        after_second.misses,
        after_second.entries,
        after_second.held_bytes,
    );
    assert_eq!(first, THIRTEEN_FRAME, "the declared frame is the payload's");
    assert_eq!(
        second, first,
        "a pass served from the pool lands exactly the frame the fresh path landed"
    );
    assert_eq!(
        after_first.hits - before.hits,
        0,
        "the first pass of a shape has nothing to take"
    );
    assert_eq!(
        after_first.misses - before.misses,
        u64::from(TEXTURE_COUNT),
        "the first pass builds one backing per declaration"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        u64::from(TEXTURE_COUNT),
        "the second pass takes one backing per declaration"
    );
    assert_eq!(
        after_second.misses - after_first.misses,
        0,
        "a served declaration is never also counted as a build"
    );
    assert_eq!(
        after_second.entries, TEXTURE_COUNT as usize,
        "the pool holds one backing per declaration of this shape"
    );
    assert!(
        after_second.held_bytes > 0,
        "a held backing is an allocation the pool kept"
    );
    assert_eq!(
        (after_second.evictions, after_second.flushes),
        (0, 0),
        "one shape is far below the cap"
    );

    // Arm two: the same two passes with the mechanism switched off. The switch
    // drops what it held, and the fresh path lands the same bytes.
    provider.set_render_texture_pool(false);
    let off_before = provider.render_texture_pool_counts();
    assert_eq!(
        off_before.entries, 0,
        "switching the mechanism off drops what it held"
    );
    let third = fixture.run_pass(indexed);
    let fourth = fixture.run_pass(indexed);
    let off_after = provider.render_texture_pool_counts();

    eprintln!(
        "pool off: third {third:02x?} fourth {fourth:02x?}; hits {} -> {}, disabled {} -> {}, \
         entries {}",
        off_before.hits, off_after.hits, off_before.disabled, off_after.disabled, off_after.entries,
    );
    assert_eq!(
        (third, fourth),
        (first, second),
        "the two arms of the reading land the same bytes"
    );
    assert_eq!(
        off_after.hits, off_before.hits,
        "no declaration is served while the switch is off"
    );
    assert_eq!(
        off_after.disabled - off_before.disabled,
        2 * u64::from(TEXTURE_COUNT),
        "every declaration of both passes reports the switch, not a miss"
    );
    assert_eq!(off_after.entries, 0, "the switch off holds nothing");
    assert_eq!(
        off_after.returns - off_before.returns,
        0,
        "nothing is handed back while the switch is off"
    );
}

/// A pass whose declarations take backings a previous pass uploaded still lands
/// its own texels.
///
/// This is the one failure a pooled host-visible image could have: the second
/// arm's payload is a single value every declaration carries, so a frame that
/// served the first arm's texels would land `THIRTEEN_FRAME` instead of the
/// second arm's own reading — and the frame moves exactly when the upload does.
#[test]
fn a_pass_that_takes_a_pooled_backing_lands_its_own_texels() {
    let Some(fixture) = fixture() else {
        return;
    };
    let provider = &fixture.provider;
    provider.set_render_texture_pool(true);

    let first =
        fixture.run_pass(|index| u8::try_from(index + 1).expect("thirteen payloads fit one byte"));
    let before = provider.render_texture_pool_counts();
    let second = fixture.run_pass(|_| 0x80);
    let after = provider.render_texture_pool_counts();

    eprintln!(
        "pool reuse across payloads: first {first:02x?} second {second:02x?}; hits {} -> {}, \
         misses {}",
        before.hits, after.hits, after.misses,
    );
    assert_eq!(first, THIRTEEN_FRAME);
    assert_eq!(
        second, SECOND_ARM_FRAME,
        "a pass that took pooled backings uploads its own texels into them"
    );
    assert_eq!(
        after.hits - before.hits,
        u64::from(TEXTURE_COUNT),
        "the second arm is served, so the frame above is the pooled path's"
    );
    assert_eq!(
        after.misses - before.misses,
        0,
        "a served declaration is never also counted as a build"
    );
    // And the third arm moves again, so the reading is not the second arm's
    // frame pinned by an earlier upload either: thirteen texels of `0x11` sum
    // to `13 * 17/255 = 0.867`, which stores as `0xdd` — below the clamp the
    // second arm's `0x80` payload reaches.
    let third = fixture.run_pass(|_| 0x11);
    eprintln!("third arm {third:02x?}");
    assert_eq!(third, [0xdd, 0x11, 0x11, 0xff]);
    assert_ne!(third, second);
}
