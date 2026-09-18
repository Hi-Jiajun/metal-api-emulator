//! The owner-window store for a colour attachment's frame (`research/docs/23`
//! §114, E-TX8).
//!
//! Census v21 (`evidence/gate3-census-v21-2026-09-18`) reads `guest_backing` as
//! the eighth first-failure bucket: records whose colour attachment is *backed
//! by the guest's own pages* (`door=mapping`, the mapper-ref-texture surface)
//! and whose frame therefore lands where the guest reads it. The narrow class's
//! refusal states the gap exactly — "the class renders into a provider image,
//! and writing the guest's pages from it is a landing this rail does not
//! carry" — while the *load* side of the same pages has been declarable since
//! R5b/§74 and E-TX6/§113 (a registered window, or a run list).
//!
//! This file measures the arm that closes the store half: a colour attachment
//! whose own view declaration names the owner's window is opened with
//! `LoadOp::Load` from that window and stored with `StoreOp::Borrowed`, so the
//! frame the pass read back lands in the same guest pages. Four readings, each
//! falsifiable:
//!
//! * the window's bytes after the pass are the frame the writeback channel
//!   carries, byte for byte — and that frame is the one the same bytes land
//!   through the pure-writeback rail (`StoreOp::Store`), so a rail that wrote
//!   something else, or a stale copy, cannot pass;
//! * the owner rewriting the window's uncovered words moves the *next* frame
//!   with them (the load reads the owner's live pages) and the pass lands the
//!   result back in those same pages;
//! * a store whose declaration names a copy arm (`OwnedBytes`, `StagedLease`)
//!   rather than a window is refused by name;
//! * a window that is not the attachment's own tightly packed extent is
//!   refused by name.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess, BufferLease,
    BufferSource, BufferView, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, GuestRun, IndexBufferBinding, IndexFormat, LeaseId,
    LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId, ProviderErrorClass,
    RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
    SemanticDigest, StoreOp, TracePass, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel.
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

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The word the surface's head window carries.
const HEAD_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
/// The word the surface's tail window carries.
const TAIL_WORD: [u8; 4] = [0x55, 0x66, 0x77, 0x88];
/// The word the owner rewrites both windows with.
const MOVED_WORD: [u8; 4] = [0x9a, 0xbc, 0xde, 0xf0];

const HEAD_LEASE: LeaseId = LeaseId::new(81);
const TAIL_LEASE: LeaseId = LeaseId::new(82);

/// The bytes the two windows concatenate to: the head word twice, then the
/// tail word.
fn window_words() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&HEAD_WORD.repeat(2));
    bytes.extend_from_slice(&TAIL_WORD.repeat(2));
    bytes
}

/// The guest surface's two registered windows: the head, then the tail.
fn two_runs() -> Vec<GuestRun> {
    vec![
        GuestRun {
            lease_id: HEAD_LEASE,
            offset: 0,
            length: 8,
        },
        GuestRun {
            lease_id: TAIL_LEASE,
            offset: 0,
            length: 8,
        },
    ]
}

/// The owner's own mapping for one window, aligned to the device's import grid.
struct AlignedBuffer {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl AlignedBuffer {
    fn new(len: usize, alignment: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, alignment)
            .expect("the import alignment is a valid allocation alignment");
        let pointer = unsafe { std::alloc::alloc(layout) };
        let pointer = std::ptr::NonNull::new(pointer).expect("aligned allocation failed");
        Self { pointer, layout }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// The fixture: one reviewed vertex-input pipeline over a 2x2 attachment, the
/// declaring compute case, and the resource table the trace is admitted
/// against. The attachment's own declaration carries the window source the
/// reading varies; the pipeline, streams and index bytes never move.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    render: metal_api_core::provider::CompiledComputePipeline,
    trace: ComputeTrace,
    resources: ResourceTableSnapshot,
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

/// The reviewed stream's four vertices moved into the left column: the two
/// triangles cover column 0 and leave column 1 for the loaded words, which is
/// what makes "the load and the landing both happened" readable per texel.
fn left_column_vertex_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    for (x, y) in [(-1.0_f32, -1.0_f32), (0.0, -1.0), (-1.0, 1.0), (0.0, 1.0)] {
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

/// The pass fields no reading in this file varies.
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
        viewport: [0, 0, 2, 2],
        scissor: None,
        vertices: 6,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    }
}

/// Build the fixture with one declaration for the attachment's view and one
/// store arm for its frame.
fn fixture(source: BufferSource, load: LoadOp, store: StoreOp) -> Option<Fixture> {
    let executor = executor()?;
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
        .compile_pipeline(&function, digest(b"render_owner_window_declaring"))
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
            logical_digest: digest(b"render_owner_window_stages"),
        })
        .expect("the vertex-input render pipeline registers");

    let vertex_bytes = left_column_vertex_bytes();
    let index_bytes = quad_index_bytes();
    // The reviewed quad reads its vertices from the caller's own stream and
    // draws them through the caller's index buffer, exactly as the census's
    // records do; the attachment's declaration is the only field the readings
    // vary.
    let vertex_buffer = BufferView {
        view_id: VERTEX_VIEW,
        metal_binding: 0,
        allocation_id: VERTEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(vertex_bytes.len()).expect("stream length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(vertex_bytes.clone()),
    };
    let index_buffer = BufferView {
        view_id: INDEX_VIEW,
        metal_binding: 0,
        allocation_id: INDEX_ALLOCATION,
        offset: 0,
        length: u64::try_from(index_bytes.len()).expect("index length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(index_bytes.clone()),
    };
    let pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load,
            store,
        }],
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
        vertex_buffers: vec![vertex_buffer],
        indices: Some(IndexBufferBinding {
            view: index_buffer,
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(render.pipeline_id)
    };

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(21),
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
                        length: 16,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source,
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
            TracePass::Render(pass),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };

    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, 16_u64),
        (SCRATCH_ALLOCATION, 8),
        (VERTEX_ALLOCATION, 32),
        (INDEX_ALLOCATION, 12),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }

    Some(Fixture {
        provider: Arc::new(provider),
        render,
        trace,
        resources,
    })
}

/// Submit one trace through admission and collect the attachment's writeback.
fn attachment_frame(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Vec<u8> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .expect("the trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(trace)
        .expect("the writebacks cover the trace");
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment lands a writeback")
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn reservations(provider: &VulkanComputeProvider) -> [LeaseReservation; 2] {
    let epoch = provider.device_epoch();
    [HEAD_LEASE, TAIL_LEASE].map(|lease_id| LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 8,
    })
}

/// The same bytes through the pure-writeback rail: the attachment's own view
/// is the trace's bytes and its store is `StoreOp::Store`, so the comparison
/// is between the two *store* arms over one load.
fn writeback_baseline(load: LoadOp, bytes: Vec<u8>) -> Option<Vec<u8>> {
    let fixture = fixture(BufferSource::OwnedBytes(bytes), load, StoreOp::Store)?;
    let provider = Arc::clone(&fixture.provider);
    Some(attachment_frame(
        &provider,
        &fixture.trace,
        &fixture.resources,
    ))
}

#[test]
fn a_borrowed_store_lands_the_frame_in_the_owners_windows() {
    let Some(owned) = writeback_baseline(LoadOp::Load, window_words()) else {
        return;
    };
    eprintln!("writeback-rail frame: {}", hex(&owned));
    assert_eq!(
        owned,
        [QUAD_TEXEL, HEAD_WORD, QUAD_TEXEL, TAIL_WORD].concat(),
        "the left column is drawn and column 1 keeps the loaded words"
    );

    let Some(fixture) = fixture(
        BufferSource::GuestRuns(two_runs()),
        LoadOp::Load,
        StoreOp::Borrowed,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let [head_reservation, tail_reservation] = reservations(&provider);
    let mut head = AlignedBuffer::new(8, alignment as usize);
    let mut tail = AlignedBuffer::new(8, alignment as usize);
    head.as_mut_slice().copy_from_slice(&HEAD_WORD.repeat(2));
    tail.as_mut_slice().copy_from_slice(&TAIL_WORD.repeat(2));
    // Safety: both owner allocations outlive every submission below and the
    // provider's release of the imports.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(head_reservation, head.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's head window");
        provider
            .import_borrowed_lease(
                BorrowedLease::new(tail_reservation, tail.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's tail window");
    }
    let mut resources = fixture.resources.clone();
    for reservation in [head_reservation, tail_reservation] {
        resources
            .insert_lease(reservation)
            .expect("the window's reservation covers it");
    }

    let frame = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("borrowed-store frame: {}", hex(&frame));
    assert_eq!(
        frame,
        owned,
        "the owner-window store lands the frame the writeback rail lands: {}",
        hex(&frame)
    );
    // The landing itself: the guest's own pages hold the frame, per texel, in
    // the declaration's own order. A rail that landed a copy-arm frame, wrote
    // one window and not the other, or left the pre-pass bytes behind cannot
    // produce this.
    eprintln!(
        "owner windows after the pass: {} / {}",
        hex(head.as_mut_slice()),
        hex(tail.as_mut_slice())
    );
    assert_eq!(
        [head.as_mut_slice(), tail.as_mut_slice()].concat(),
        frame,
        "the frame lands in the owner's windows byte for byte"
    );
    assert_eq!(
        head.as_mut_slice(),
        &[QUAD_TEXEL, HEAD_WORD].concat(),
        "the head window holds the drawn texel and the word the load read"
    );
    assert_eq!(
        tail.as_mut_slice(),
        &[QUAD_TEXEL, TAIL_WORD].concat(),
        "the tail window holds the drawn texel and the word the load read"
    );

    // The holds the gather and the landing took are retired by the fences.
    let registry = provider.borrowed_registry();
    for reservation in [head_reservation, tail_reservation] {
        assert_eq!(
            registry.outstanding(reservation.lease.lease_id),
            Some(0),
            "every window's hold is retired once its submission's fence signals"
        );
    }

    let _ = &fixture.render;
    provider
        .release_borrowed_lease(HEAD_LEASE)
        .expect("no retain is outstanding after the submission");
    provider
        .release_borrowed_lease(TAIL_LEASE)
        .expect("no retain is outstanding after the submission");
}

#[test]
fn the_owner_rewriting_the_windows_moves_the_next_frame() {
    let Some(fixture) = fixture(
        BufferSource::GuestRuns(two_runs()),
        LoadOp::Load,
        StoreOp::Borrowed,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let [head_reservation, tail_reservation] = reservations(&provider);
    let mut head = AlignedBuffer::new(8, alignment as usize);
    let mut tail = AlignedBuffer::new(8, alignment as usize);
    head.as_mut_slice().copy_from_slice(&HEAD_WORD.repeat(2));
    tail.as_mut_slice().copy_from_slice(&TAIL_WORD.repeat(2));
    // Safety: both owner allocations outlive every submission below.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(head_reservation, head.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's head window");
        provider
            .import_borrowed_lease(
                BorrowedLease::new(tail_reservation, tail.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's tail window");
    }
    let mut resources = fixture.resources.clone();
    for reservation in [head_reservation, tail_reservation] {
        resources
            .insert_lease(reservation)
            .expect("the window's reservation covers it");
    }

    let first = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("first frame: {}", hex(&first));
    assert_eq!(
        first,
        [QUAD_TEXEL, HEAD_WORD, QUAD_TEXEL, TAIL_WORD].concat()
    );

    // The owner rewrites both uncovered words *and* the covered ones: the
    // pass's load must read the live pages (so the uncovered texels follow),
    // and the pass's own draw must overwrite the covered ones.
    head.as_mut_slice().copy_from_slice(&MOVED_WORD.repeat(2));
    tail.as_mut_slice().copy_from_slice(&MOVED_WORD.repeat(2));
    let moved = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("owner-rewritten frame: {}", hex(&moved));
    assert_eq!(
        moved,
        [QUAD_TEXEL, MOVED_WORD, QUAD_TEXEL, MOVED_WORD].concat(),
        "the owner's rewritten pages are what the uncovered texels upload: {}",
        hex(&moved)
    );
    assert_eq!(
        [head.as_mut_slice(), tail.as_mut_slice()].concat(),
        moved,
        "the rewritten windows are the landed frame again"
    );

    provider
        .release_borrowed_lease(HEAD_LEASE)
        .expect("no retain is outstanding after the submission");
    provider
        .release_borrowed_lease(TAIL_LEASE)
        .expect("no retain is outstanding after the submission");
}

#[test]
fn a_borrowed_store_whose_declaration_names_a_copy_arm_is_refused_by_name() {
    // The declaration states the same bytes through a copy arm: nothing is
    // wrong with the trace, but there is no owner window to land in, and a rail
    // that landed the provider's own copy would be writing memory no owner's
    // ledger holds.
    let Some(fixture) = fixture(
        BufferSource::OwnedBytes(window_words()),
        LoadOp::Load,
        StoreOp::Borrowed,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let admitted = provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect("the declaration is well formed: the arm is what the rail refuses");
    let refused = provider
        .submit(admitted)
        .expect_err("a borrowed store needs an owner window");
    eprintln!("copy-arm declaration refused: {refused:?}");
    assert_eq!(refused.slug, "render_attachment_landing_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused
            .fields
            .get("source")
            .map(|value| format!("{value:?}")),
        Some("Text(\"owned_bytes\")".to_owned()),
        "the refusal names the arm the declaration carries"
    );
}

#[test]
fn a_window_that_is_not_the_attachment_extent_is_refused_by_name() {
    // The pass opens with a clear, so the declaration is the only thing under
    // test: the owner's window is 12 bytes where the attachment's own tightly
    // packed extent is 16. The contract answers first — the attachment's view
    // declaration has to cover exactly the attachment — and the rail carries
    // the same rule as a value-level second line
    // (`render_attachment_landing_mismatch`) for a directly-built request, so
    // no path lands a truncated frame.
    let Some(fixture) = fixture(
        BufferSource::GuestRuns(vec![GuestRun {
            lease_id: HEAD_LEASE,
            offset: 0,
            length: 12,
        }]),
        LoadOp::Clear(metal_api_core::provider::ClearColor::new([0, 0, 0, 0xff])),
        StoreOp::Borrowed,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let [head_reservation, _] = reservations(&provider);
    let head_reservation = LeaseReservation {
        lease: head_reservation.lease,
        offset: 0,
        length: 12,
    };
    let mut head = AlignedBuffer::new(12, alignment as usize);
    head.as_mut_slice().copy_from_slice(&HEAD_WORD.repeat(3));
    // Safety: the allocation outlives the submission and the release below.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(head_reservation, head.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's window");
    }
    let mut resources = fixture.resources.clone();
    resources
        .insert_lease(head_reservation)
        .expect("the window's reservation covers it");
    let mut trace = fixture.trace.clone();
    if let Some(TracePass::Compute(pass)) = trace.passes.first_mut() {
        pass.buffers[0].length = 12;
    }
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a window that is not the attachment's extent cannot receive the frame");
    eprintln!("wrong-extent window refused: {refused:?}");
    assert_eq!(refused.slug, "attachment_extent_mismatch");
    assert_eq!(
        refused.detail.as_deref(),
        Some(
            "render pass 1 attachment view ViewId(921) covers 16 bytes, but the trace declares 12"
        ),
        "the refusal names the attachment's own extent and the declaration's"
    );

    provider
        .release_borrowed_lease(HEAD_LEASE)
        .expect("no retain is outstanding after the refusal");
}

#[test]
fn a_single_registered_window_is_the_same_landing_through_the_borrow_arm() {
    // The other window arm: one reservation whose own view is the attachment's
    // whole extent. The load reads it through the device's host import
    // (`BorrowedNoCopy`), and the store lands the frame back in the same
    // window — which is why the arm is the borrow arm on both halves.
    let Some(owned) = writeback_baseline(LoadOp::Load, window_words()) else {
        return;
    };
    let Some(fixture) = fixture(
        BufferSource::BorrowedNoCopy(HEAD_LEASE),
        LoadOp::Load,
        StoreOp::Borrowed,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: HEAD_LEASE,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: 16,
    };
    let mut window = AlignedBuffer::new(16, alignment as usize);
    window.as_mut_slice().copy_from_slice(&window_words());
    // Safety: the owner allocation outlives every submission below and the
    // provider's release of the import.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(reservation, window.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's window");
    }
    let mut resources = fixture.resources.clone();
    resources
        .insert_lease(reservation)
        .expect("the window's reservation covers the attachment");

    let frame = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("borrow-arm frame: {}", hex(&frame));
    assert_eq!(
        frame,
        owned,
        "one registered window is the same bytes and the same landing: {}",
        hex(&frame)
    );
    assert_eq!(
        window.as_mut_slice(),
        frame,
        "the single window holds the frame the pass landed"
    );
    assert_eq!(
        provider.borrowed_registry().outstanding(HEAD_LEASE),
        Some(0),
        "the window's holds are retired once the fences signal"
    );
    provider
        .release_borrowed_lease(HEAD_LEASE)
        .expect("no retain is outstanding after the submission");
}
