//! The shape-decided render objects' own rail cases
//! (`crate::render_setup_reuse`).
//!
//! The increment hands a pass the shader modules, the pipeline layout and the
//! graphics pipeline a pass of the *same shape* built before. That is only
//! allowed to be a timing change, so this file is the byte-level oracle the
//! increment's own report reads:
//!
//! * one shape runs twice against one provider: the second pass must be served
//!   from the cache (`hits` +1) and must publish the *same frame bytes*;
//! * a second shape — one whose pipeline differs while its render pass agrees —
//!   must miss, and the first shape's next pass must hit again;
//! * the same two passes must publish the same bytes with the mechanism
//!   switched **off**, so the two arms of the reading are compared with each
//!   other and not only with themselves;
//! * retiring the registered pipeline the passes drew through must drop what
//!   the cache minted while it was live: the next pass of that shape misses
//!   again, and its frame is unchanged;
//! * the counters that name the failure directions (`mismatches`, `unkeyed`)
//!   must stay zero on these shapes, so a reading that shows no reuse can tell
//!   "the shapes never repeated" from "the cache refused them".

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BlendAttachment, BlendFactor, BlendOperation,
    BufferAccess, BufferSource, BufferView, ClearColor, ColorWriteMask, CompletionPolicy,
    ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType,
    IndexBufferBinding, IndexFormat, LoadOp, OperationId, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, RenderSetupReuseCounts, VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` positions from the caller's own
/// stream, straight in NDC.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel: it admits the attachment's view into the
/// trace's own pool, which is where the render rail resolves its bytes from.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(921);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(931);
const SCRATCH_VIEW: ViewId = ViewId::new(922);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(932);
const VERTEX_VIEW: ViewId = ViewId::new(923);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(933);
const INDEX_VIEW: ViewId = ViewId::new(924);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(934);

/// The attachment's extent: four by four texels, four bytes each.
const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const ATTACHMENT_BYTES: usize = (WIDTH as usize) * (HEIGHT as usize) * 4;

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The clear the pass opens from, stated in the attachment's own format. The
/// byte-exact clear is what the rail's reviewed class admits, and it is what
/// makes every pass of this shape publish the same frame.
const CLEAR: ClearColor = ClearColor::new([0x10, 0x20, 0x30, 0xff]);

fn quad_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [(-1.0_f32, -1.0_f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        bytes.extend_from_slice(&x.to_ne_bytes());
        bytes.extend_from_slice(&y.to_ne_bytes());
    }
    bytes
}

fn quad_index_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12);
    for index in [0_u16, 1, 2, 1, 3, 2] {
        bytes.extend_from_slice(&index.to_ne_bytes());
    }
    bytes
}

fn quad_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: 8,
        step: metal_api_core::provider::VertexStep::PerVertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32x2,
        }],
    }])
}

fn executor() -> Option<Arc<VulkanExecutor>> {
    match VulkanExecutor::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            None
        }
    }
}

/// One shape: what the pass states beyond the shared fixture. `blend` is the
/// axis this file moves when it needs a *second* shape whose render pass agrees
/// with the first one's — the pipeline then differs while the render pass
/// definition is identical, which is exactly the case a pipeline-only cache has
/// to separate.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Plain,
    Blended,
}

impl Shape {
    fn blend(self) -> Option<metal_api_core::provider::RenderPassBlend> {
        match self {
            Shape::Plain => None,
            Shape::Blended => Some(metal_api_core::provider::RenderPassBlend {
                attachments: vec![BlendAttachment {
                    enabled: true,
                    source_rgb: BlendFactor::One,
                    destination_rgb: BlendFactor::One,
                    operation: BlendOperation::Add,
                    source_alpha: BlendFactor::Zero,
                    destination_alpha: BlendFactor::One,
                    alpha_operation: BlendOperation::Add,
                    write_mask: ColorWriteMask::ALL,
                }],
            }),
        }
    }
}

/// One provider and the registered pipelines both arms of a case draw through.
struct Fixture {
    provider: VulkanComputeProvider,
    render: metal_api_core::provider::CompiledComputePipeline,
    /// The declaring kernel that admits the attachment's view into the trace's
    /// own pool, which is where the render rail resolves its bytes from.
    declaring: metal_api_core::provider::CompiledComputePipeline,
}

fn fixture(executor: Arc<VulkanExecutor>) -> Fixture {
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("the provider is built");
    let declaring = provider
        .compile_pipeline(&function, digest(b"render_setup_reuse_declaring"))
        .expect("the declaring pipeline registers");
    let render = register_render(&provider);
    Fixture {
        provider,
        render,
        declaring,
    }
}

/// Register the one reviewed render pipeline this file draws through.
///
/// Registration is a contract-surface act rather than a device one, so the
/// released-registration case re-registers the same modules: the pass after the
/// release draws through an entry the cache has never seen, which is what
/// "the registration moved under the cache" means here.
fn register_render(
    provider: &VulkanComputeProvider,
) -> metal_api_core::provider::CompiledComputePipeline {
    provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_buffer_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: quad_layout(),
                textures: Vec::new(),
            },
            vertex_spirv: QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SOLID_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: SemanticDigest::new(
                "metal-smoke-fixture-v1",
                b"render_setup_reuse_stages".to_vec(),
            )
            .expect("digest"),
        })
        .expect("the vertex-input render pipeline registers")
}

/// Submit one pass of `shape` and return the frame the writeback channel
/// published for the attachment.
fn run_pass(fixture: &Fixture, shape: Shape) -> Vec<u8> {
    let provider = &fixture.provider;
    let vertex_bytes = quad_vertex_bytes();
    let index_bytes = quad_index_bytes();
    let vertex_buffer = BufferView {
        view_id: VERTEX_VIEW,
        metal_binding: 0,
        allocation_id: VERTEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(vertex_bytes.len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vertex_bytes),
    };
    let index_length = u64::try_from(index_bytes.len()).expect("index length");
    let index_buffer = BufferView {
        view_id: INDEX_VIEW,
        metal_binding: 0,
        allocation_id: INDEX_ALLOCATION,
        offset: 0,
        length: index_length,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(index_bytes),
    };
    let attachment_view = BufferView {
        view_id: ATTACHMENT_VIEW,
        metal_binding: 0,
        allocation_id: ATTACHMENT_ALLOCATION,
        offset: 0,
        length: u64::try_from(ATTACHMENT_BYTES).expect("attachment length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vec![0u8; ATTACHMENT_BYTES]),
    };
    let pass = RenderPassDescriptor {
        pipeline: fixture.render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(WIDTH),
            height: u64::from(HEIGHT),
            load: LoadOp::Clear(CLEAR),
            store: StoreOp::Store,
        }],
        vertices: u32::try_from(index_length / 2).expect("uint16 index count"),
        vertex_buffers: vec![vertex_buffer],
        indices: Some(IndexBufferBinding {
            view: index_buffer,
            format: IndexFormat::Uint16,
        }),
        blend: shape.blend(),
        ..render_pass_defaults(fixture.render.pipeline_id)
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(21),
        pipelines: vec![fixture.declaring.clone(), fixture.render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: fixture.declaring.pipeline_id,
                buffers: vec![
                    attachment_view,
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
            TracePass::Render(pass),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (
            ATTACHMENT_ALLOCATION,
            u64::try_from(ATTACHMENT_BYTES).expect("size"),
        ),
        (VERTEX_ALLOCATION, 32),
        (INDEX_ALLOCATION, 12),
        (SCRATCH_ALLOCATION, 8),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment lands a writeback")
}

/// The pass fields no case in this file varies.
fn render_pass_defaults(pipeline: metal_api_core::provider::PipelineId) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
        stage_buffers: Vec::new(),
        blend: None,
        multisample: None,
        depth_resolve: None,
        stencil_resolve: None,
        cull: None,
        depth: None,
        depth_test: None,
        stencil: None,
        stencil_test: None,
        base_vertex: 0,
        pipeline,
        color_attachments: Vec::new(),
        viewport: [0, 0, WIDTH, HEIGHT],
        scissor: None,
        vertices: 6,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

fn counts(fixture: &Fixture) -> RenderSetupReuseCounts {
    fixture.provider.render_setup_reuse_counts()
}

/// The frame one unblended pass publishes: the quad covers the whole four-by-four
/// render area, so every texel is the fragment stage's own output. A reading
/// that published the clear instead would be a frame from a different shape.
fn quad_frame() -> Vec<u8> {
    QUAD_TEXEL
        .iter()
        .copied()
        .cycle()
        .take(ATTACHMENT_BYTES)
        .collect()
}

/// One shape repeated is served from the cache, and its frame does not move.
#[test]
fn a_repeated_shape_is_served_from_the_cache_with_the_same_bytes() {
    let Some(executor) = executor() else {
        return;
    };
    let fixture = fixture(executor);
    let before = counts(&fixture);
    let first = run_pass(&fixture, Shape::Plain);
    let after_first = counts(&fixture);
    let second = run_pass(&fixture, Shape::Plain);
    let after_second = counts(&fixture);

    assert_eq!(first, second, "the reused pipeline moved the frame's bytes");
    assert_eq!(first, quad_frame(), "the frame is the pass's own output");
    assert_eq!(
        after_first.misses - before.misses,
        1,
        "the first pass of a shape builds its objects and caches them"
    );
    assert_eq!(
        after_second.hits - after_first.hits,
        1,
        "the second pass of the same shape is served from the cache"
    );
    assert_eq!(
        (after_second.mismatches, after_second.unkeyed),
        (before.mismatches, before.unkeyed),
        "the shape was keyed exactly: neither arm refuses it"
    );
    assert_eq!(
        after_second.entries - before.entries,
        1,
        "one resident shape"
    );
}

/// A second shape misses, and the first shape's next pass still hits.
#[test]
fn a_second_shape_misses_and_the_first_shape_still_hits() {
    let Some(executor) = executor() else {
        return;
    };
    let fixture = fixture(executor);
    let before = counts(&fixture);
    let plain_first = run_pass(&fixture, Shape::Plain);
    let blended = run_pass(&fixture, Shape::Blended);
    let after_blended = counts(&fixture);
    let plain_second = run_pass(&fixture, Shape::Plain);
    let after_plain = counts(&fixture);

    assert_eq!(
        after_blended.misses - before.misses,
        2,
        "the second shape is a miss even though its render pass agrees"
    );
    assert_eq!(
        after_blended.hits, before.hits,
        "nothing was served to the second shape"
    );
    assert_eq!(
        after_plain.hits - after_blended.hits,
        1,
        "the first shape's next pass is served from its own entry"
    );
    assert_eq!(
        plain_first, plain_second,
        "the hit changed the first shape's frame"
    );
    assert_ne!(
        plain_first, blended,
        "the two shapes are not the same shape: the oracle would be vacuous"
    );
}

/// With the mechanism off every pass builds its objects, and the frames are the
/// ones the on-arm published.
#[test]
fn the_switch_off_builds_every_pass_and_the_frames_agree() {
    let Some(executor) = executor() else {
        return;
    };
    let fixture = fixture(executor);
    assert!(
        fixture.provider.render_setup_reuse_enabled(),
        "the mechanism is on by default"
    );
    let on_first = run_pass(&fixture, Shape::Plain);
    let on_second = run_pass(&fixture, Shape::Plain);
    let on = counts(&fixture);
    assert_eq!(on.hits, 1, "the on arm is served once");

    fixture.provider.set_render_setup_reuse(false);
    assert!(!fixture.provider.render_setup_reuse_enabled());
    assert_eq!(
        counts(&fixture).entries,
        0,
        "switching the mechanism off drops what it held"
    );
    let off_first = run_pass(&fixture, Shape::Plain);
    let off_second = run_pass(&fixture, Shape::Plain);
    let after = counts(&fixture);

    assert_eq!(
        (after.hits, after.misses, after.entries),
        (on.hits, on.misses, 0),
        "nothing is looked up, cached or kept while the switch is off"
    );
    assert_eq!(on_first, on_second);
    assert_eq!(off_first, off_second);
    assert_eq!(
        on_first, off_first,
        "the two arms publish the same frame for one shape"
    );
}

/// Retiring the registered pipeline drops what the cache minted alive, and the
/// next pass of that shape builds again — with the same frame.
#[test]
fn a_released_registration_drops_what_it_minted() {
    let Some(executor) = executor() else {
        return;
    };
    let mut fixture = fixture(executor);
    let before = counts(&fixture);
    let first = run_pass(&fixture, Shape::Plain);
    let second = run_pass(&fixture, Shape::Plain);
    let hit = counts(&fixture);
    assert_eq!(hit.hits - before.hits, 1, "the second pass hits");
    assert_eq!(hit.entries - before.entries, 1, "one resident shape");

    fixture
        .provider
        .release_render_pipeline(&fixture.render)
        .expect("the registration is released");
    let flushed = counts(&fixture);
    assert_eq!(flushed.entries, 0, "the release empties the cache");
    assert!(flushed.flushes > hit.flushes, "and says so");

    // A submission through a retired registration is refused before it reaches
    // the device (`unknown_render_pipeline`), so the arm that draws again states
    // a fresh registration over the same modules: the shape is the same one the
    // released entry held, and the cache must build rather than serve.
    fixture.render = register_render(&fixture.provider);
    let third = run_pass(&fixture, Shape::Plain);
    let rebuilt = counts(&fixture);
    assert_eq!(
        rebuilt.hits, flushed.hits,
        "the pass after the release is not served from the retired entry"
    );
    assert_eq!(
        rebuilt.misses - flushed.misses,
        1,
        "it builds its objects again"
    );
    assert_eq!(first, second);
    assert_eq!(second, third, "the rebuild publishes the same frame");
}
