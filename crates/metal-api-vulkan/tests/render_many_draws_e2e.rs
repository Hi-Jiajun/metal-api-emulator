//! One render pass that carries an **ordered list of draws** (G3-B/B-2,
//! `research/docs/23` §3.3).
//!
//! The engine track has always drawn N `vkCmdDraw*` calls inside one render
//! pass; the canonical track drew one draw per pass, and paid the pass's fixed
//! cost — the render pass object, the framebuffer, the attachments, the
//! command buffer, the submission and the fence — once per draw. This file
//! measures the shape that carries N draws in **one** render pass instance:
//
//! * the frame one pass of N draws publishes is byte for byte the frame the
//!   same N draws publish as N single-draw passes (the head keeps the frame the
//!   later draws load), read for N = 1, 2, 3 and 8;
//! * the draws execute in declared order, read from a pair whose two orders
//!   *cannot* land the same bytes: a plain draw and an additive one over the
//!   same texels;
//! * the multi-draw trace is **one** queue submission, read from the executor's
//!   own counter, where the same draws as separate submissions are N;
//! * a list above the ceiling is refused by name, not truncated.
//!
//! Every reading here is falsifiable: a rail that executed only the head, or
//! that read the list as a single draw, lands a frame with the other draws'
//! scissors still holding the clear colour.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BlendAttachment, BlendFactor, BlendOperation,
    BufferAccess, BufferSource, BufferView, ClearColor, ColorWriteMask, CompletionPolicy,
    ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType,
    IndexBufferBinding, IndexFormat, LoadOp, OperationId, PipelineId, RenderAttachment, RenderDraw,
    RenderDrawsDescriptor, RenderPassBlend, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StageBufferView, StoreOp, TracePass, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexLayout, ViewId, MAX_DRAWS_PER_PASS,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel, used to declare the attachment's view in the
/// trace's serial pool.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The clear the multi-draw pass opens with: a texel no draw covers keeps it,
/// so a draw the rail dropped is a clear texel in the frame.
const CLEAR_TEXEL: [u8; 4] = [0x0a, 0x0b, 0x0c, 0x0d];

/// The attachment both arms render into: four by four texels, so eight draws
/// can each be told apart by the texel their scissor covers.
const ATTACHMENT_EXTENT: u32 = 4;
const ATTACHMENT_VIEW: ViewId = ViewId::new(971);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(981);
/// The declaring kernel's own scratch write, which its contract requires.
const SCRATCH_VIEW: ViewId = ViewId::new(972);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(982);
/// The vertex and index streams one draw binds: `DRAW_VIEW_BASE + i` and
/// `DRAW_INDEX_BASE + i` for draw `i`, so every draw owns its own declaration.
const DRAW_VIEW_BASE: u64 = 1_000;
const DRAW_ALLOCATION_BASE: u64 = 2_000;
const DRAW_INDEX_VIEW_BASE: u64 = 3_000;
const DRAW_INDEX_ALLOCATION_BASE: u64 = 4_000;

/// The frame the one-pass-of-N arm and the N-pass arm both publish: the clear
/// texel everywhere a draw's scissor does not cover, the quad texel where it
/// does.
fn union_frame_bytes(scissors: &[[u32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((ATTACHMENT_EXTENT * ATTACHMENT_EXTENT * 4) as usize);
    for y in 0..ATTACHMENT_EXTENT {
        for x in 0..ATTACHMENT_EXTENT {
            let covered = scissors.iter().any(|[sx, sy, width, height]| {
                width > &0
                    && height > &0
                    && x >= *sx
                    && x < sx + width
                    && y >= *sy
                    && y < sy + height
            });
            bytes.extend_from_slice(if covered { &QUAD_TEXEL } else { &CLEAR_TEXEL });
        }
    }
    bytes
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

/// The reviewed stream's four vertices, spanning the whole attachment.
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

fn render_pass_defaults(pipeline: PipelineId) -> RenderPassDescriptor {
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
        viewport: [0, 0, ATTACHMENT_EXTENT, ATTACHMENT_EXTENT],
        scissor: None,
        vertices: 6,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// One draw of the fixture, stated once and rendered through both arms.
struct DrawSpec {
    vertex_view: ViewId,
    vertex_allocation: AllocationId,
    index_view: ViewId,
    index_allocation: AllocationId,
    /// The texels this draw is clipped to: one texel per draw, so a dropped
    /// draw is a clear texel in the frame.
    scissor: [u32; 4],
    /// The blend state this draw's pipeline is built with, or `None` for "write
    /// the fragment output". The order-observable pair is the two states.
    blend: Option<RenderPassBlend>,
}

impl DrawSpec {
    /// This draw as the list arm states it.
    fn draw(&self, pipeline: PipelineId, index_bytes: &[u8], vertex_bytes: &[u8]) -> RenderDraw {
        RenderDraw {
            pipeline,
            viewport: [0, 0, ATTACHMENT_EXTENT, ATTACHMENT_EXTENT],
            scissor: Some(self.scissor),
            vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
            vertex_buffers: vec![self.vertex_stream(vertex_bytes)],
            indices: Some(IndexBufferBinding {
                view: BufferView {
                    view_id: self.index_view,
                    metal_binding: 0,
                    allocation_id: self.index_allocation,
                    offset: 0,
                    length: u64::try_from(index_bytes.len()).expect("index length"),
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(index_bytes.to_vec()),
                },
                format: IndexFormat::Uint16,
            }),
            base_vertex: 0,
            cull: None,
            blend: self.blend.clone(),
            depth_test: None,
            stencil_test: None,
            instance_count: 1,
            textures: Vec::new(),
            samplers: Vec::new(),
            stage_buffers: Vec::<StageBufferView>::new(),
        }
    }

    /// This draw as one single-draw pass, with the attachment state the caller
    /// states: the reference arm's own shape.
    fn pass(
        &self,
        pipeline: PipelineId,
        attachment: RenderAttachment,
        index_bytes: &[u8],
        vertex_bytes: &[u8],
    ) -> RenderPassDescriptor {
        RenderPassDescriptor {
            pipeline,
            color_attachments: vec![attachment],
            scissor: Some(self.scissor),
            vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
            vertex_buffers: vec![self.vertex_stream(vertex_bytes)],
            indices: Some(IndexBufferBinding {
                view: BufferView {
                    view_id: self.index_view,
                    metal_binding: 0,
                    allocation_id: self.index_allocation,
                    offset: 0,
                    length: u64::try_from(index_bytes.len()).expect("index length"),
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(index_bytes.to_vec()),
                },
                format: IndexFormat::Uint16,
            }),
            blend: self.blend.clone(),
            ..render_pass_defaults(pipeline)
        }
    }

    fn vertex_stream(&self, vertex_bytes: &[u8]) -> BufferView {
        BufferView {
            view_id: self.vertex_view,
            metal_binding: 0,
            allocation_id: self.vertex_allocation,
            offset: 0,
            length: u64::try_from(vertex_bytes.len()).expect("stream length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vertex_bytes.to_vec()),
        }
    }
}

/// The two traces of one reading and the declarations both of them need.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    /// The list arm: one render pass whose head and tail carry every draw.
    draws: ComputeTrace,
    /// The reference arm: the same draws as `draws.len()` single-draw passes in
    /// one trace, chained through the attachment's resident image (the head
    /// clears and keeps, the middle passes keep, the tail stores).
    reference: ComputeTrace,
    resources: ResourceTableSnapshot,
}

/// The order-observable pair's additive state: `source + destination`, which
/// cannot land the same bytes as the plain draw whichever way the two run
/// (`research/docs/23` §3.3, v40).
fn additive_blend() -> RenderPassBlend {
    RenderPassBlend {
        attachments: vec![BlendAttachment {
            enabled: true,
            source_rgb: BlendFactor::One,
            destination_rgb: BlendFactor::One,
            source_alpha: BlendFactor::One,
            destination_alpha: BlendFactor::One,
            operation: BlendOperation::Add,
            alpha_operation: BlendOperation::Add,
            write_mask: ColorWriteMask::ALL,
        }],
    }
}

/// Build the fixture for `specs`: one list-bearing trace and the same draws as
/// single-draw passes.
fn fixture(executor: Arc<VulkanExecutor>, specs: &[DrawSpec]) -> Option<Fixture> {
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_many_draws_declaring"))
        .expect("the declaring pipeline registers");
    let render = provider
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
            logical_digest: digest(b"render_many_draws_stages"),
        })
        .expect("the vertex-input render pipeline registers");
    let index_bytes = quad_index_bytes();
    let vertex_bytes = quad_vertex_bytes();
    let attachment = |load: LoadOp, store: StoreOp| RenderAttachment {
        view_id: ATTACHMENT_VIEW,
        allocation_id: ATTACHMENT_ALLOCATION,
        format: AttachmentFormat::Rgba8Unorm,
        width: u64::from(ATTACHMENT_EXTENT),
        height: u64::from(ATTACHMENT_EXTENT),
        load,
        store,
    };
    // The attachment's own declaration in the trace's serial pool: the pass
    // that stores the frame publishes into this buffer view, and the pass that
    // loads it uploads from it (`research/docs/23` §3.3/§74).
    let declaring = TracePass::Compute(ComputePass {
        pipeline: compute.pipeline_id,
        buffers: vec![
            BufferView {
                view_id: ATTACHMENT_VIEW,
                metal_binding: 0,
                allocation_id: ATTACHMENT_ALLOCATION,
                offset: 0,
                length: u64::from(ATTACHMENT_EXTENT * ATTACHMENT_EXTENT * 4),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![
                    0x7e;
                    (ATTACHMENT_EXTENT * ATTACHMENT_EXTENT * 4)
                        as usize
                ]),
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
    });
    let trace = |operation: u64, passes: Vec<TracePass>| ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(operation),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes,
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    // The list arm: the pass state once (clear the whole raster, store the
    // frame), every draw's own declaration in the list.
    let head = RenderPassDescriptor {
        color_attachments: vec![attachment(
            LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
            StoreOp::Store,
        )],
        pipeline: render.pipeline_id,
        scissor: specs.first().map(|spec| spec.scissor),
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
        vertex_buffers: vec![specs[0].vertex_stream(&vertex_bytes)],
        indices: Some(IndexBufferBinding {
            view: BufferView {
                view_id: specs[0].index_view,
                metal_binding: 0,
                allocation_id: specs[0].index_allocation,
                offset: 0,
                length: u64::try_from(index_bytes.len()).expect("index length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(index_bytes.clone()),
            },
            format: IndexFormat::Uint16,
        }),
        blend: specs[0].blend.clone(),
        ..render_pass_defaults(render.pipeline_id)
    };
    let tail: Vec<RenderDraw> = specs[1..]
        .iter()
        .map(|spec| spec.draw(render.pipeline_id, &index_bytes, &vertex_bytes))
        .collect();
    let draws = trace(
        51,
        vec![
            declaring.clone(),
            TracePass::RenderDraws(RenderDrawsDescriptor { head, tail }),
        ],
    );
    // The reference arm: the same draws as one single-draw pass each, chained
    // through the attachment's own resident image — the head clears and keeps
    // the frame, the middle passes keep it, the tail stores it. The frame the
    // chain ends with is the frame the list arm's one pass lands.
    let mut reference_passes = vec![declaring];
    for (index, spec) in specs.iter().enumerate() {
        let state = match (index + 1, specs.len()) {
            (1, 1) => (LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)), StoreOp::Store),
            (1, _) => (
                LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
                StoreOp::Resident,
            ),
            (last, total) if last == total => (LoadOp::Resident, StoreOp::Store),
            _ => (LoadOp::Resident, StoreOp::Resident),
        };
        reference_passes.push(TracePass::Render(spec.pass(
            render.pipeline_id,
            attachment(state.0, state.1),
            &index_bytes,
            &vertex_bytes,
        )));
    }
    let reference = trace(52, reference_passes);

    let mut resources = ResourceTableSnapshot::new();
    let mut allocation = |record: AllocationRecord| {
        resources
            .insert_allocation(record)
            .expect("fixture allocation");
    };
    allocation(AllocationRecord {
        allocation_id: ATTACHMENT_ALLOCATION,
        owner_epoch: provider.device_epoch(),
        size: u64::from(ATTACHMENT_EXTENT * ATTACHMENT_EXTENT * 4),
    });
    allocation(AllocationRecord {
        allocation_id: SCRATCH_ALLOCATION,
        owner_epoch: provider.device_epoch(),
        size: 8,
    });
    for spec in specs {
        allocation(AllocationRecord {
            allocation_id: spec.vertex_allocation,
            owner_epoch: provider.device_epoch(),
            size: u64::try_from(vertex_bytes.len()).expect("stream length"),
        });
        allocation(AllocationRecord {
            allocation_id: spec.index_allocation,
            owner_epoch: provider.device_epoch(),
            size: u64::try_from(index_bytes.len()).expect("index length"),
        });
    }
    Some(Fixture {
        provider: Arc::new(provider),
        draws,
        reference,
        resources,
    })
}

/// The draws of one reading: draw `i` clipped to texel `i` of the raster.
fn specs(draws: usize, order: Option<&RenderPassBlend>) -> Vec<DrawSpec> {
    (0..draws)
        .map(|index| DrawSpec {
            vertex_view: ViewId::new(DRAW_VIEW_BASE + index as u64),
            vertex_allocation: AllocationId::new(DRAW_ALLOCATION_BASE + index as u64),
            index_view: ViewId::new(DRAW_INDEX_VIEW_BASE + index as u64),
            index_allocation: AllocationId::new(DRAW_INDEX_ALLOCATION_BASE + index as u64),
            scissor: [
                u32::try_from(index).expect("draw index") % ATTACHMENT_EXTENT,
                u32::try_from(index).expect("draw index") / ATTACHMENT_EXTENT,
                1,
                1,
            ],
            // The order-observable pair states its blend on the *second* draw
            // only; every other reading draws plainly.
            blend: match order {
                Some(blend) if index == 1 => Some(blend.clone()),
                _ => None,
            },
        })
        .collect()
}

fn submit(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<u8> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the fixture trace is admitted");
    let submitted = provider.submit(admitted).expect("the trace executes");
    submitted
        .validate_for_trace(trace)
        .expect("the trace's writebacks are what it declared");
    submitted
        .writebacks
        .into_iter()
        .rfind(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the trace publishes the frame the caller reads")
}

fn submissions(executor: &VulkanExecutor) -> usize {
    executor.queue_submission_counts().into_iter().sum()
}

#[test]
fn one_pass_with_n_draws_lands_the_same_frame_as_n_single_draw_passes() {
    let Some(executor) = executor() else {
        return;
    };
    // The snapshot declares the arm: the rail executes it, and the ceiling it
    // publishes is the contract's own.
    let probe = VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("context");
    assert!(probe.capabilities().declares_render_multi_draw_support());
    assert_eq!(
        probe.capabilities().max_draws_per_pass,
        MAX_DRAWS_PER_PASS as u32
    );

    for draws in [1_usize, 2, 3, 8] {
        let specs = specs(draws, None);
        let expected =
            union_frame_bytes(&specs.iter().map(|spec| spec.scissor).collect::<Vec<_>>());
        // The list arm runs in a provider of its own: it states no resident
        // target, and the reference arm below *does* — a provider that has
        // already kept this identity's image refuses a pass that names it
        // without declaring the residency (`resident_target_undeclared`).
        let Some(list) = fixture(Arc::clone(&executor), &specs) else {
            return;
        };
        let before = submissions(&executor);
        let listed = submit(&list.provider, &list.draws, &list.resources);
        let after = submissions(&executor);
        assert_eq!(
            listed, expected,
            "one pass of {draws} draws lands the clear colour beside every draw's own texel"
        );
        // The trace carries the declaring compute pass beside the render pass,
        // so the render half's own count is the difference: one render
        // submission for one pass of N draws, whatever N is.
        assert_eq!(
            after - before,
            2,
            "one pass of {draws} draws is one render submission beside the declaring dispatch"
        );
        // The cross-arm reading: the same draws as single-draw passes in one
        // trace. The fixture provider's own ceiling is eight trace entries
        // (`max_passes`), so the eight-draw reference would need nine — the
        // eight-draw reading above is held to the frame the *spec* states
        // instead, which is at least as strict for draw loss: a draw the rail
        // dropped leaves its texel holding the clear colour.
        if draws < 8 {
            let Some(reference) = fixture(Arc::clone(&executor), &specs) else {
                return;
            };
            let chained = submit(
                &reference.provider,
                &reference.reference,
                &reference.resources,
            );
            assert_eq!(
                chained, expected,
                "{draws} single-draw passes land the clear colour beside every draw's own texel"
            );
            assert_eq!(
                listed, chained,
                "one pass of {draws} draws lands the frame {draws} single-draw passes land"
            );
        }
    }
}

#[test]
fn the_declared_order_decides_which_draw_lands_last() {
    let Some(executor) = executor() else {
        return;
    };
    // Two draws over the **same** texel: the first plainly, the second
    // additively. The additive draw reads what the plain one wrote, so the two
    // orders land different bytes — which is what makes the order observable at
    // all.
    let overlapping = || {
        let mut specs = specs(2, None);
        specs[1].scissor = specs[0].scissor;
        specs
    };
    let blend = additive_blend();
    let mut plain_then_add = overlapping();
    plain_then_add[1].blend = Some(blend.clone());
    let mut add_then_plain = overlapping();
    add_then_plain[0].blend = Some(blend);

    let Some(plain_fixture) = fixture(Arc::clone(&executor), &plain_then_add) else {
        return;
    };
    let plain_frame = submit(
        &plain_fixture.provider,
        &plain_fixture.draws,
        &plain_fixture.resources,
    );
    let Some(plain_reference) = fixture(Arc::clone(&executor), &plain_then_add) else {
        return;
    };
    let plain_chained = submit(
        &plain_reference.provider,
        &plain_reference.reference,
        &plain_reference.resources,
    );
    assert_eq!(
        plain_frame, plain_chained,
        "the list arm's order is the reference arm's order: plain, then additive"
    );

    let Some(add_fixture) = fixture(Arc::clone(&executor), &add_then_plain) else {
        return;
    };
    let add_frame = submit(
        &add_fixture.provider,
        &add_fixture.draws,
        &add_fixture.resources,
    );
    let Some(add_reference) = fixture(Arc::clone(&executor), &add_then_plain) else {
        return;
    };
    let add_chained = submit(
        &add_reference.provider,
        &add_reference.reference,
        &add_reference.resources,
    );
    assert_eq!(
        add_frame, add_chained,
        "the list arm's order is the reference arm's order: additive, then plain"
    );
    assert_ne!(
        plain_frame, add_frame,
        "the two orders land different bytes, so the reading above is about order"
    );
    // The additive draw over the plain one sums the two fragment outputs, and
    // the plain one over it is the fragment output alone.
    assert_eq!(&plain_frame[..4], &[0x80, 0xff, 0xff, 0xff]);
    assert_eq!(&add_frame[..4], &QUAD_TEXEL);
}

#[test]
fn a_draw_list_above_the_ceiling_is_refused_by_name() {
    let Some(executor) = executor() else {
        return;
    };
    let Some(fixture) = fixture(Arc::clone(&executor), &specs(1, None)) else {
        return;
    };
    let Some(TracePass::RenderDraws(list)) = fixture.draws.passes.last().cloned() else {
        panic!("the fixture's last entry is the draw list");
    };
    let draw = list.head.draw();
    let over = RenderDrawsDescriptor {
        head: list.head,
        tail: vec![draw; MAX_DRAWS_PER_PASS],
    };
    let mut trace = fixture.draws.clone();
    trace.passes.pop();
    trace.passes.push(TracePass::RenderDraws(over));
    let error = fixture
        .provider
        .capabilities()
        .validate_trace(trace, fixture.resources)
        .expect_err("a list above the ceiling is refused");
    assert_eq!(error.slug, "render_draw_count_limit");
}
