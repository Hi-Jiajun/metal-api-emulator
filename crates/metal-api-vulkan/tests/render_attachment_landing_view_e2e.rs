//! The colour attachment's *landing view* (`research/docs/23` §115 之后的增量,
//! E-TX13).
//!
//! Census v31/v32 read the `guest_backing` bucket as a record whose load source
//! is the frame the exec walk handed on as bytes, while its guest pages hold the
//! frame *before* the packet. E-TX8's `StoreOp::Borrowed` can only take its
//! window from the attachment's **own** view declaration, so that shape had no
//! legal spelling: declaring the window would have swapped the pass's pre-pass
//! bytes to the older frame (right counts, wrong pixels), and declaring the
//! caller's bytes landed the frame nowhere the guest could read.
//!
//! This file measures the arm that separates the two facts. The attachment's
//! own view declares the **caller's bytes** (`fefefefe…`), a second view
//! declares the **owner's window** (`11223344…`), and the store names that second
//! view. Five readings, each falsifiable:
//!
//! * the window the landing view declares holds the frame the writeback channel
//!   carries, byte for byte;
//! * the texels the draw does not cover hold the **caller's bytes**, not the
//!   window's old bytes — which is what "the load stayed the caller's" means per
//!   texel, and the reading a rail that resolved the load from the landing view
//!   would fail;
//! * moving the window's old bytes moves nothing while moving the caller's bytes
//!   moves the frame: the landing is a write direction, the load a read one;
//! * the E-TX8 arm over this same fixture keeps its own refusal, which is why
//!   the second declaration is not a convenience but the only legal spelling;
//! * a landing view the trace never declares, one whose source is a copy arm,
//!   and one that is not the attachment's tight extent are each refused by name.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, AttachmentLandingView, BorrowedLease,
    BufferAccess, BufferLease, BufferSource, BufferView, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, GuestRun,
    IndexBufferBinding, IndexFormat, LeaseId, LeaseReservation, LoadOp, NoCopyLeaseImporter,
    OperationId, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel, used to declare each view in the trace's pool.
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
/// The second declaration this increment is about: the owner's window.
const LANDING_VIEW: ViewId = ViewId::new(925);
const LANDING_ALLOCATION: AllocationId = AllocationId::new(935);
const WINDOW_SCRATCH_VIEW: ViewId = ViewId::new(926);
const WINDOW_SCRATCH_ALLOCATION: AllocationId = AllocationId::new(936);
/// The identity a refusal test names without declaring it.
const UNDECLARED_LANDING_VIEW: ViewId = ViewId::new(927);
const UNDECLARED_LANDING_ALLOCATION: AllocationId = AllocationId::new(937);

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The frame the exec walk handed on as bytes: the attachment's own view.
const CALLER_TEXEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
/// What the owner's window holds *before* the pass — deliberately unlike the
/// caller's bytes, so a rail that read the load from the window would show them.
const WINDOW_TEXEL: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
/// A second window content: the landing is a write direction, so the frame the
/// pass produces must not move when only this changes.
const OTHER_WINDOW_TEXEL: [u8; 4] = [0xaa, 0xbb, 0xcc, 0xdd];
/// The chain value of the second reading: moving the caller's bytes moves the
/// texels the draw does not cover.
const MOVED_CALLER_TEXEL: [u8; 4] = [0x01, 0x02, 0x03, 0x04];

const LANDING_LEASE: LeaseId = LeaseId::new(83);

/// One repeated texel over the attachment's four texels.
fn words(texel: [u8; 4]) -> Vec<u8> {
    texel.repeat(4)
}

/// The owner's own mapping for the landing window, aligned to the device's
/// import grid.
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

/// One reviewed vertex-input pipeline over a 2x2 attachment, the declaring
/// compute passes that put both views in the trace's pool, and the resource
/// table the trace is admitted against.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
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
/// triangles cover column 0 and leave column 1 for the load's own bytes, which
/// is what makes "the load and the landing both happened" readable per texel.
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

/// Build the fixture.
///
/// `caller` are the bytes the attachment's *own* view declares (the walk's
/// chain value), `landing_length` is the length of the second declaration
/// (`None` leaves the landing view out of the trace entirely), and `store` is
/// the arm the pass records. The two declaring compute passes exist so both
/// views are in the trace's serial view list, exactly as a producer's own
/// declaring pass puts them there.
fn fixture(
    caller: Vec<u8>,
    landing: Option<(BufferSource, u64)>,
    store: StoreOp,
) -> Option<Fixture> {
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
        .compile_pipeline(&function, digest(b"render_landing_view_declaring"))
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
            logical_digest: digest(b"render_landing_view_stages"),
        })
        .expect("the vertex-input render pipeline registers");

    let vertex_bytes = left_column_vertex_bytes();
    let index_bytes = quad_index_bytes();
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
            load: LoadOp::Load,
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

    let mut passes = vec![TracePass::Compute(ComputePass {
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
                source: BufferSource::OwnedBytes(caller),
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
    })];
    // The landing view's declaration. A producer's trace puts every view it
    // names in this same serial list, so the rail resolves the store arm's
    // identity against it and nothing else.
    if let Some((source, length)) = landing {
        passes.push(TracePass::Compute(ComputePass {
            pipeline: compute.pipeline_id,
            buffers: vec![
                BufferView {
                    view_id: LANDING_VIEW,
                    metal_binding: 0,
                    allocation_id: LANDING_ALLOCATION,
                    offset: 0,
                    length,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source,
                },
                BufferView {
                    view_id: WINDOW_SCRATCH_VIEW,
                    metal_binding: 1,
                    allocation_id: WINDOW_SCRATCH_ALLOCATION,
                    offset: 0,
                    length: 4,
                    access: BufferAccess::Write,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xcd; 4]),
                },
            ],
            textures: Vec::new(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
        }));
    }
    passes.push(TracePass::Render(pass));

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(23),
        pipelines: vec![compute.clone(), render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes,
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
        (LANDING_ALLOCATION, 64 * 1024),
        (WINDOW_SCRATCH_ALLOCATION, 8),
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
        trace,
        resources,
    })
}

/// The landing view's own reservation: the whole 16-byte extent inside one
/// page-aligned owner window.
fn window_reservation(provider: &VulkanComputeProvider, length: u64) -> LeaseReservation {
    LeaseReservation {
        lease: BufferLease {
            lease_id: LANDING_LEASE,
            allocation_id: LANDING_ALLOCATION,
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length,
    }
}

/// Import one owner window holding `texel` and admit its reservation.
fn import_window(
    provider: &VulkanComputeProvider,
    resources: &mut ResourceTableSnapshot,
    window: &mut AlignedBuffer,
    texel: [u8; 4],
) {
    window.as_mut_slice().fill(0x5a);
    window.as_mut_slice()[..16].copy_from_slice(&words(texel));
    let reservation = window_reservation(provider, 16);
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
    resources
        .insert_lease(reservation)
        .expect("the window's reservation covers the attachment");
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

#[test]
fn a_landing_view_lands_the_frame_without_reading_the_window() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 16)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let alignment = provider.no_copy_alignment() as usize;
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    let mut resources = fixture.resources.clone();
    import_window(&provider, &mut resources, &mut window, WINDOW_TEXEL);

    let frame = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("landing-view frame: {}", hex(&frame));
    // Column 0 is the draw's own output; column 1 keeps the load's bytes, which
    // are the *caller's* — a rail that read the pre-pass bytes from the landing
    // window would land `11223344` there instead.
    assert_eq!(
        frame,
        [QUAD_TEXEL, CALLER_TEXEL, QUAD_TEXEL, CALLER_TEXEL].concat(),
        "the drawn texels are the fragment's and the rest are the caller's bytes"
    );
    // The landing: the owner's own pages hold the frame the pass published.
    assert_eq!(
        &window.as_mut_slice()[..16],
        frame.as_slice(),
        "the landing view's window holds the frame byte for byte"
    );
    assert_eq!(
        provider.borrowed_registry().outstanding(LANDING_LEASE),
        Some(0),
        "the window's holds are retired once the fences signal"
    );
    provider
        .release_borrowed_lease(LANDING_LEASE)
        .expect("no retain is outstanding after the submission");
}

#[test]
fn the_landing_is_a_write_direction_and_the_load_a_read_one() {
    // Two runs that differ only in the window's *old* bytes land the same frame:
    // the window is the frame's destination, never its source.
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(first) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 16)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    let provider = Arc::clone(&first.provider);
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let alignment = provider.no_copy_alignment() as usize;
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    let mut resources = first.resources.clone();
    import_window(&provider, &mut resources, &mut window, WINDOW_TEXEL);
    let unmoved = attachment_frame(&provider, &first.trace, &resources);
    provider
        .release_borrowed_lease(LANDING_LEASE)
        .expect("no retain is outstanding after the submission");

    let mut other_window = AlignedBuffer::new(64 * 1024, alignment);
    let mut other_resources = first.resources.clone();
    import_window(
        &provider,
        &mut other_resources,
        &mut other_window,
        OTHER_WINDOW_TEXEL,
    );
    let moved_window = attachment_frame(&provider, &first.trace, &other_resources);
    eprintln!("frame with the other window: {}", hex(&moved_window));
    assert_eq!(
        unmoved, moved_window,
        "the window's own old bytes never reach the frame"
    );
    provider
        .release_borrowed_lease(LANDING_LEASE)
        .expect("no retain is outstanding after the submission");

    // The same pass over a *different* chain value moves exactly the texels the
    // draw does not cover: the load really is the caller's bytes.
    let Some(moved) = fixture(
        words(MOVED_CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 16)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    let provider = Arc::clone(&moved.provider);
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    let mut resources = moved.resources.clone();
    import_window(&provider, &mut resources, &mut window, WINDOW_TEXEL);
    let frame = attachment_frame(&provider, &moved.trace, &resources);
    eprintln!("frame with the moved caller bytes: {}", hex(&frame));
    assert_eq!(
        frame,
        [
            QUAD_TEXEL,
            MOVED_CALLER_TEXEL,
            QUAD_TEXEL,
            MOVED_CALLER_TEXEL
        ]
        .concat(),
        "the uncovered texels follow the caller's bytes"
    );
    provider
        .release_borrowed_lease(LANDING_LEASE)
        .expect("no retain is outstanding after the submission");
}

#[test]
fn the_borrowed_arm_keeps_its_own_refusal_over_the_callers_view() {
    // The same fixture through E-TX8's arm: the attachment's own view declares
    // the caller's bytes, so there is no window for the frame to land in. This
    // is the refusal the landing view exists to answer — and it is a *different*
    // arm's refusal, not a relaxed one.
    let Some(fixture) = fixture(words(CALLER_TEXEL), None, StoreOp::Borrowed) else {
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
    eprintln!("borrowed arm over the caller's view: {refused:?}");
    assert_eq!(refused.slug, "render_attachment_landing_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused
            .fields
            .get("source")
            .map(|value| format!("{value:?}")),
        Some("Text(\"owned_bytes\")".to_owned()),
        "the refusal names the declaring arm"
    );
}

#[test]
fn a_landing_view_the_trace_never_declares_is_refused_by_name() {
    let undeclared = AttachmentLandingView {
        allocation_id: UNDECLARED_LANDING_ALLOCATION,
        view_id: UNDECLARED_LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::OwnedBytes(words(WINDOW_TEXEL)), 16)),
        StoreOp::BorrowedLanding(undeclared),
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let admitted = provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect("the declaration is well formed: the missing view is what the rail refuses");
    let refused = provider
        .submit(admitted)
        .expect_err("a landing view the trace never declares cannot receive the frame");
    eprintln!("undeclared landing view: {refused:?}");
    assert_eq!(refused.slug, "render_attachment_landing_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused
            .fields
            .get("source")
            .map(|value| format!("{value:?}")),
        Some("Text(\"landing_view_undeclared\")".to_owned()),
        "the refusal names the missing declaration rather than the attachment's own"
    );
    assert_eq!(
        refused
            .fields
            .get("landing_view")
            .map(|value| format!("{value:?}")),
        Some(format!("Unsigned({})", UNDECLARED_LANDING_VIEW.get()))
    );
}

#[test]
fn a_landing_view_whose_source_is_a_copy_arm_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(mut fixture) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 16)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    // The second declaration states the same bytes through a copy arm: nothing
    // is wrong with the trace, but there is no owner window to land in.
    if let Some(TracePass::Compute(pass)) = fixture.trace.passes.get_mut(1) {
        pass.buffers[0].source = BufferSource::OwnedBytes(words(WINDOW_TEXEL));
    }
    let provider = Arc::clone(&fixture.provider);
    let admitted = provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect("the declaration is well formed: the arm is what the rail refuses");
    let refused = provider
        .submit(admitted)
        .expect_err("a landing view needs an owner window");
    eprintln!("copy-arm landing view refused: {refused:?}");
    assert_eq!(refused.slug, "render_attachment_landing_unsupported");
    assert_eq!(
        refused
            .fields
            .get("source")
            .map(|value| format!("{value:?}")),
        Some("Text(\"owned_bytes\")".to_owned()),
        "the refusal names the arm the landing view's declaration carries"
    );
}

#[test]
fn a_landing_view_that_is_not_the_attachment_extent_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    // The pass opens with a clear, so the second declaration is the only thing
    // under test; it covers 12 bytes where the attachment is 16.
    let Some(mut fixture) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 12)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    if let Some(TracePass::Render(pass)) = fixture.trace.passes.last_mut() {
        pass.color_attachments[0].load =
            LoadOp::Clear(metal_api_core::provider::ClearColor::new([0; 4]));
    }
    let provider = Arc::clone(&fixture.provider);
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let alignment = provider.no_copy_alignment() as usize;
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    let mut resources = fixture.resources.clone();
    window.as_mut_slice().fill(0x5a);
    let reservation = window_reservation(&provider, 16);
    // Safety: the owner allocation outlives the submission below and the
    // provider's release of the import.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(reservation, window.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's window");
    }
    resources
        .insert_lease(reservation)
        .expect("the window's reservation covers the view's declared range");
    let admitted = provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), resources.clone())
        .expect("the declaration is well formed: the extent is what the rail refuses");
    let refused = provider
        .submit(admitted)
        .expect_err("a landing view that is not the attachment's extent cannot receive the frame");
    eprintln!("wrong-extent landing view refused: {refused:?}");
    assert_eq!(refused.slug, "render_attachment_landing_mismatch");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    provider
        .release_borrowed_lease(LANDING_LEASE)
        .expect("no retain is outstanding after the refusal");
}

/// The runs a producer may spell instead of one registration
/// (`BufferSource::GuestRuns`): the arm's window list is resolved by the same
/// rule the single-window arm uses, so the landing lands in both runs in order.
#[test]
fn a_run_list_landing_view_is_the_same_landing_in_order() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(mut fixture) = fixture(
        words(CALLER_TEXEL),
        Some((BufferSource::BorrowedNoCopy(LANDING_LEASE), 16)),
        StoreOp::BorrowedLanding(landing),
    ) else {
        return;
    };
    let head_lease = LeaseId::new(84);
    let tail_lease = LeaseId::new(85);
    if let Some(TracePass::Compute(pass)) = fixture.trace.passes.get_mut(1) {
        pass.buffers[0].source = BufferSource::GuestRuns(vec![
            GuestRun {
                lease_id: head_lease,
                offset: 0,
                length: 8,
            },
            GuestRun {
                lease_id: tail_lease,
                offset: 0,
                length: 8,
            },
        ]);
    }
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let mut head = AlignedBuffer::new(8, alignment as usize);
    let mut tail = AlignedBuffer::new(8, alignment as usize);
    head.as_mut_slice().copy_from_slice(&WINDOW_TEXEL.repeat(2));
    tail.as_mut_slice()
        .copy_from_slice(&OTHER_WINDOW_TEXEL.repeat(2));
    let epoch = provider.device_epoch();
    let reservations = [head_lease, tail_lease].map(|lease_id| LeaseReservation {
        lease: BufferLease {
            lease_id,
            allocation_id: LANDING_ALLOCATION,
            owner_epoch: epoch,
        },
        offset: 0,
        length: 8,
    });
    let mut resources = fixture.resources.clone();
    for (reservation, window) in reservations.iter().zip([&mut head, &mut tail]) {
        // Safety: both owner allocations outlive every submission below and the
        // provider's release of the imports.
        unsafe {
            provider
                .import_borrowed_lease(
                    BorrowedLease::new(*reservation, window.as_mut_slice().as_ptr() as usize)
                        .expect("the owner's window is a valid reservation"),
                )
                .expect("the provider imports the owner's window");
        }
        resources
            .insert_lease(*reservation)
            .expect("the run's reservation covers it");
    }

    let frame = attachment_frame(&provider, &fixture.trace, &resources);
    eprintln!("run-list landing frame: {}", hex(&frame));
    assert_eq!(
        frame,
        [QUAD_TEXEL, CALLER_TEXEL, QUAD_TEXEL, CALLER_TEXEL].concat(),
        "the run-list landing lands the same frame the single window does"
    );
    assert_eq!(
        [&head, &tail]
            .map(|window| window.pointer.as_ptr() as *const u8)
            .map(|pointer| unsafe { std::slice::from_raw_parts(pointer, 8) })
            .concat(),
        frame,
        "the two runs hold the frame in the declaration's own order"
    );
    for lease in [head_lease, tail_lease] {
        provider
            .release_borrowed_lease(lease)
            .expect("no retain is outstanding after the submission");
    }
}
