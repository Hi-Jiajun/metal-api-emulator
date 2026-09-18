//! The owner-window store through the object rail (`research/docs/23` §115,
//! E-TX9b).
//!
//! E-TX8/§114 gave the *trace* rail the arm: a colour attachment whose own
//! view declaration names the owner's registered window is opened with
//! `LoadOp::Load` from those pages and stored with `StoreOp::Borrowed`, so the
//! frame the pass read back lands in the guest's own memory. The object rail
//! could not state it: `RenderColorAttachment::store` carries the store tag,
//! but nothing let a recording *name the owner's window* as the attachment's
//! own declaration, and both rails resolve that window from the trace's serial
//! view list.
//!
//! This file measures the object API's closing of that gap — the compute
//! rail's lease arm (`set_buffer_lease`, the render stage-buffer arm's
//! sibling) plus the declared attachment list
//! (`draw_indexed_primitives_with_declared_attachments` with a `Window` entry).
//! Five readings, each falsifiable:
//!
//! * the guest's window after the pass holds the frame the *trace* rail lands
//!   from the same bytes through its own `BufferSource::BorrowedNoCopy`
//!   declaration — one frame, landed twice, byte for byte;
//! * that frame is the census shape's own: the left column drawn, column 1 the
//!   words the load read out of the guest's pages;
//! * an attachment whose window no pass of the command declares is refused by
//!   name (`WindowAttachmentUndeclared`) instead of landing nothing;
//! * a window that is not the attachment's tightly packed extent is refused by
//!   name (`WindowAttachmentExtentMismatch`) rather than truncated — the
//!   padded/multi-registration shape this arm does not carry;
//! * a copy arm (`StagedLease`) and a *writable* lease-bound slot are refused
//!   by name (`WindowAttachmentNamesACopyArm`,
//!   `WritableComputeLeaseUnsupported`).

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess, BufferLease,
    BufferSource, BufferView, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LeaseId,
    LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId, PipelineCompileRequest,
    RenderAttachment, RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot,
    SemanticDigest, ShaderSource, StoreOp, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::provider_api::{
    Device as ObjectDevice, Error as ObjectError, RenderAttachmentDeclaration,
    RenderAttachmentLoad, RenderWindowAttachment, StageBufferLease, StageBufferLeaseArm,
};
use metal_api_core::Size;
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel: one word from binding 0 into binding 1.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The attachment's identity, the one both rails name.
const ATTACHMENT_VIEW: ViewId = ViewId::new(951);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(952);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(962);
const VERTEX_VIEW: ViewId = ViewId::new(953);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(963);
const INDEX_VIEW: ViewId = ViewId::new(954);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(964);

/// The object rail's own lease (the trace rail below uses its own).
const OBJECT_LEASE: LeaseId = LeaseId::new(91);
/// The trace rail's lease.
const TRACE_LEASE: LeaseId = LeaseId::new(92);

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The word the surface's head window carries.
const HEAD_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
/// The word the surface's tail window carries.
const TAIL_WORD: [u8; 4] = [0x55, 0x66, 0x77, 0x88];

/// The 2x2 attachment's tightly packed extent, which is also the window's own
/// length.
const EXTENT: u64 = 16;

/// The bytes the guest's surface holds before the pass: the head word twice,
/// then the tail word twice.
fn window_words() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(EXTENT as usize);
    bytes.extend_from_slice(&HEAD_WORD.repeat(2));
    bytes.extend_from_slice(&TAIL_WORD.repeat(2));
    bytes
}

/// The frame both rails land: the left column drawn, column 1 the loaded words.
fn expected_frame() -> Vec<u8> {
    [QUAD_TEXEL, HEAD_WORD, QUAD_TEXEL, TAIL_WORD].concat()
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

fn digest(label: &str) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", label.as_bytes().to_vec()).expect("digest")
}

fn reservation(provider: &VulkanComputeProvider, lease: LeaseId) -> LeaseReservation {
    LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: EXTENT,
    }
}

fn stage_lease(
    provider: &VulkanComputeProvider,
    lease: LeaseId,
    arm: StageBufferLeaseArm,
) -> StageBufferLease {
    StageBufferLease {
        reservation: reservation(provider, lease),
        allocation_size: EXTENT,
        arm,
    }
}

/// The shared tooling one reading needs: the provider both rails use, the one
/// object device every recording belongs to, and the compiled pipelines both
/// rails name.
///
/// The declaring `Pipeline` *handle* is kept alive for the fixture's lifetime
/// on purpose: the object API's pipeline handle owns its provider-side
/// registration, so dropping it would retire the pipeline the trace rail's
/// submission names.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    device: ObjectDevice,
    compute: metal_api_core::provider_api::Pipeline,
    render: metal_api_core::provider::CompiledComputePipeline,
}

impl Fixture {
    fn compute_metadata(&self) -> &metal_api_core::provider::CompiledComputePipeline {
        self.compute.metadata()
    }
}

fn fixture() -> Option<Fixture> {
    let executor = executor()?;
    let provider = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context"),
    );
    let device = ObjectDevice::new(
        Arc::clone(&provider) as Arc<dyn metal_api_core::provider::PipelineProvider>
    );
    let compute = device
        .compile_pipeline(PipelineCompileRequest {
            entry_name: "copy_word".to_owned(),
            logical_digest: digest("render_object_window_declaring"),
            source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
        })
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
            logical_digest: digest("render_object_window_stages"),
        })
        .expect("the vertex-input render pipeline registers");
    // The declaring kernel reads one buffer and writes one, so the object
    // rail's encoder has to bind both slots: the window at the read slot and a
    // caller-held scratch buffer at the writable one.
    assert_eq!(
        compute.metadata().contract.buffer_bindings.len(),
        2,
        "the declaring kernel's contract is the reviewed pair"
    );
    Some(Fixture {
        provider,
        device,
        compute,
        render,
    })
}

/// The frame the *trace* rail lands for the same window and the same stream.
fn trace_rail_frame(fixture: &Fixture, window: &mut AlignedBuffer) -> Vec<u8> {
    let provider = &fixture.provider;
    let reservation = reservation(provider, TRACE_LEASE);
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
    let vertex_bytes = left_column_vertex_bytes();
    let index_bytes = quad_index_bytes();
    let pass = RenderPassDescriptor {
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
        pipeline: fixture.render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Load,
            store: StoreOp::Borrowed,
        }],
        viewport: [0, 0, 2, 2],
        scissor: None,
        vertices: 6,
        vertex_buffers: vec![BufferView {
            view_id: VERTEX_VIEW,
            metal_binding: 0,
            allocation_id: VERTEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(vertex_bytes.len()).expect("stream length"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vertex_bytes.clone()),
        }],
        indices: Some(IndexBufferBinding {
            view: BufferView {
                view_id: INDEX_VIEW,
                metal_binding: 0,
                allocation_id: INDEX_ALLOCATION,
                offset: 0,
                length: u64::try_from(index_bytes.len()).expect("index length"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(index_bytes),
            },
            format: IndexFormat::Uint16,
        }),
        instance_count: 1,
        textures: Vec::new(),
        present: None,
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(31),
        pipelines: vec![fixture.compute_metadata().clone(), fixture.render.clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: fixture.compute_metadata().pipeline_id,
                buffers: vec![
                    // The attachment's own declaration: the guest's window,
                    // imported without a copy (`research/docs/23` §74, R5b).
                    BufferView {
                        view_id: ATTACHMENT_VIEW,
                        metal_binding: 0,
                        allocation_id: ATTACHMENT_ALLOCATION,
                        offset: 0,
                        length: EXTENT,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::BorrowedNoCopy(TRACE_LEASE),
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
        (ATTACHMENT_ALLOCATION, EXTENT),
        (SCRATCH_ALLOCATION, 4),
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
    resources.insert_lease(reservation).expect("window lease");
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let frame = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment lands a writeback");
    provider
        .release_borrowed_lease(TRACE_LEASE)
        .expect("no retain is outstanding after the submission");
    frame
}

/// Record and submit the census's shape through the object rail, returning the
/// guest window's bytes afterwards.
fn object_rail_window(fixture: &Fixture, window: &mut AlignedBuffer) -> Vec<u8> {
    let provider = &fixture.provider;
    let reservation = reservation(provider, OBJECT_LEASE);
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
    let device = &fixture.device;
    let compute = &fixture.compute;
    let render = device
        .render_pipeline(&fixture.render)
        .expect("the render registration is this device's");
    let scratch = device
        .new_buffer_with_bytes(vec![0_u8; 4])
        .expect("scratch buffer");
    let vertex_stream = device
        .new_buffer_with_bytes(left_column_vertex_bytes())
        .expect("vertex stream");
    let index_buffer = device
        .new_buffer_with_bytes(quad_index_bytes())
        .expect("index buffer");
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    {
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(compute)
            .expect("declaring pipeline state");
        encoder
            .set_buffer_lease(
                0,
                stage_lease(provider, OBJECT_LEASE, StageBufferLeaseArm::BorrowedNoCopy),
            )
            .expect("the window binds at the declaring kernel's read slot");
        encoder
            .set_buffer(1, &scratch.view(0, 4).expect("scratch view"))
            .expect("the scratch binds at the declaring kernel's write slot");
        encoder
            .dispatch_threads(
                Size::new(1, 1, 1).expect("grid"),
                Size::new(1, 1, 1).expect("local size"),
            )
            .expect("the declaring dispatch records");
        encoder.end_encoding().expect("end encoding");
    }
    {
        let mut encoder = command.render_command_encoder().expect("render encoder");
        encoder
            .set_render_pipeline_state(&render)
            .expect("render pipeline state");
        encoder
            .set_vertex_buffer(0, &vertex_stream.view(0, 32).expect("stream view"))
            .expect("vertex stream binds");
        encoder
            .set_index_buffer(
                &index_buffer.view(0, 12).expect("index view"),
                IndexFormat::Uint16,
            )
            .expect("index buffer binds");
        encoder
            .draw_indexed_primitives_with_declared_attachments(
                &[RenderAttachmentDeclaration::Window(
                    RenderWindowAttachment {
                        lease: stage_lease(
                            provider,
                            OBJECT_LEASE,
                            StageBufferLeaseArm::BorrowedNoCopy,
                        ),
                        format: AttachmentFormat::Rgba8Unorm,
                        load: RenderAttachmentLoad::Load,
                    },
                )],
                2,
                2,
                6,
                None,
            )
            .expect("the window attachment records");
        encoder.end_encoding().expect("end encoding");
    }
    command.commit().expect("commit");
    command.wait_until_completed().expect("completion");
    let frame = window.as_mut_slice().to_vec();
    provider
        .release_borrowed_lease(OBJECT_LEASE)
        .expect("no retain is outstanding after the submission");
    frame
}

#[test]
fn the_object_rail_lands_the_same_frame_the_trace_rail_lands_in_the_owners_window() {
    let Some(fixture) = fixture() else {
        return;
    };
    let alignment = fixture.provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let mut trace_window = AlignedBuffer::new(EXTENT as usize, alignment as usize);
    trace_window.as_mut_slice().copy_from_slice(&window_words());
    let trace_frame = trace_rail_frame(&fixture, &mut trace_window);
    assert_eq!(
        trace_frame,
        expected_frame(),
        "the trace rail's borrowed store lands the reviewed frame"
    );
    assert_eq!(
        trace_window.as_mut_slice(),
        trace_frame.as_slice(),
        "the trace rail's window holds the frame it landed"
    );

    let mut object_window = AlignedBuffer::new(EXTENT as usize, alignment as usize);
    object_window
        .as_mut_slice()
        .copy_from_slice(&window_words());
    let object_frame = object_rail_window(&fixture, &mut object_window);
    eprintln!("trace-rail frame: {}", hex(&trace_frame));
    eprintln!("object-rail window: {}", hex(&object_frame));
    assert_eq!(
        object_frame, trace_frame,
        "the object rail's window holds the frame the trace rail lands, byte for byte"
    );
    assert_eq!(
        object_frame,
        expected_frame(),
        "the census shape's own frame: left column drawn, column 1 loaded"
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

/// Which declaration the command's pass names beside the window attachment.
#[derive(Clone, Copy, PartialEq)]
enum DeclaredWindow {
    /// The owner's window, the shape that lands.
    Window,
    /// The caller's own buffer: the serial view list then carries a *copy* arm
    /// for the attachment's identity.
    CallerView,
}

/// Record the census's shape through the object rail with the given window
/// attachment, returning the draw's own result.
///
/// Every step but the draw is a fixture step, so a refusal a reading is about
/// has to be the one the draw (or the commit) answers.
fn record_window_command(
    fixture: &Fixture,
    window: RenderWindowAttachment,
    declared: DeclaredWindow,
) -> Result<metal_api_core::provider_api::CommandBuffer, ObjectError> {
    let device = &fixture.device;
    let compute = &fixture.compute;
    let render = device
        .render_pipeline(&fixture.render)
        .expect("the render registration is this device's");
    let holder = device
        .new_buffer_with_bytes(window_words())
        .expect("holder buffer");
    let scratch = device
        .new_buffer_with_bytes(vec![0_u8; 4])
        .expect("scratch buffer");
    let vertex_stream = device
        .new_buffer_with_bytes(left_column_vertex_bytes())
        .expect("vertex stream");
    let index_buffer = device
        .new_buffer_with_bytes(quad_index_bytes())
        .expect("index buffer");
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    {
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(compute)
            .expect("declaring pipeline state");
        match declared {
            DeclaredWindow::Window => {
                encoder
                    .set_buffer_lease(0, window.lease)
                    .expect("the window binds at the declaring kernel's read slot");
            }
            DeclaredWindow::CallerView => {
                encoder
                    .set_buffer(0, &holder.view(0, EXTENT as usize).expect("holder view"))
                    .expect("the holder binds at the declaring kernel's read slot");
            }
        }
        encoder
            .set_buffer(1, &scratch.view(0, 4).expect("scratch view"))
            .expect("the scratch binds at the declaring kernel's write slot");
        encoder
            .dispatch_threads(
                Size::new(1, 1, 1).expect("grid"),
                Size::new(1, 1, 1).expect("local size"),
            )
            .expect("the declaring dispatch records");
        encoder.end_encoding().expect("end encoding");
    }
    let mut encoder = command.render_command_encoder().expect("render encoder");
    encoder
        .set_render_pipeline_state(&render)
        .expect("render pipeline state");
    encoder
        .set_vertex_buffer(0, &vertex_stream.view(0, 32).expect("stream view"))
        .expect("vertex stream binds");
    encoder
        .set_index_buffer(
            &index_buffer.view(0, 12).expect("index view"),
            IndexFormat::Uint16,
        )
        .expect("index buffer binds");
    encoder.draw_indexed_primitives_with_declared_attachments(
        &[RenderAttachmentDeclaration::Window(window)],
        2,
        2,
        6,
        None,
    )?;
    encoder.end_encoding().expect("end encoding");
    Ok(command)
}

#[test]
fn a_window_no_pass_declares_is_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    if fixture.provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    // The attachment names the owner's window, and the command's own
    // declaring pass names a caller-held buffer for the same identity: the
    // serial view list carries no window, so the landing has nowhere to
    // resolve from and the commit is refused by name.
    let window = RenderWindowAttachment {
        lease: stage_lease(
            &fixture.provider,
            OBJECT_LEASE,
            StageBufferLeaseArm::BorrowedNoCopy,
        ),
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Load,
    };
    let command = record_window_command(&fixture, window, DeclaredWindow::CallerView)
        .expect("the recording itself states a shape the pass can hold");
    match command.commit() {
        Err(ObjectError::WindowAttachmentUndeclared { lease, view }) => {
            assert_eq!(lease, OBJECT_LEASE, "the refusal names the window's lease");
            assert_eq!(
                view,
                ViewId::new(OBJECT_LEASE.get()),
                "the refusal names the window's own view, which is the lease's identity"
            );
        }
        other => panic!("expected WindowAttachmentUndeclared, got {other:?}"),
    }
}

#[test]
fn a_window_that_is_not_the_attachment_extent_is_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    // The padded shape: a window of twelve bytes cannot take a sixteen-byte
    // frame without truncating it, and this arm refuses that by name — at the
    // recorder, before any device object exists.
    let mut lease = stage_lease(
        &fixture.provider,
        OBJECT_LEASE,
        StageBufferLeaseArm::BorrowedNoCopy,
    );
    lease.reservation.length = 12;
    let window = RenderWindowAttachment {
        lease,
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Load,
    };
    match record_window_command(&fixture, window, DeclaredWindow::Window).err() {
        Some(ObjectError::WindowAttachmentExtentMismatch {
            lease,
            expected,
            declared,
        }) => {
            assert_eq!(lease, OBJECT_LEASE);
            assert_eq!(expected, EXTENT, "the attachment's tightly packed extent");
            assert_eq!(
                declared, 12,
                "the window the caller declared, not a truncated one"
            );
        }
        other => panic!("expected WindowAttachmentExtentMismatch, got {other:?}"),
    }
}

#[test]
fn a_window_naming_a_copy_arm_is_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    // A staged lease is the provider's copy of one reservation, not the
    // owner's live pages: the frame would land in a buffer no owner's ledger
    // protects, which the render rail refuses as `staged_lease` and the object
    // rail refuses at the recorder.
    let window = RenderWindowAttachment {
        lease: stage_lease(
            &fixture.provider,
            OBJECT_LEASE,
            StageBufferLeaseArm::StagedLease,
        ),
        format: AttachmentFormat::Rgba8Unorm,
        load: RenderAttachmentLoad::Load,
    };
    match record_window_command(&fixture, window, DeclaredWindow::Window).err() {
        Some(ObjectError::WindowAttachmentNamesACopyArm { lease }) => {
            assert_eq!(lease, OBJECT_LEASE);
        }
        other => panic!("expected WindowAttachmentNamesACopyArm, got {other:?}"),
    }
}

#[test]
fn a_writable_lease_bound_slot_is_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    let provider = &fixture.provider;
    let device = &fixture.device;
    let compute = &fixture.compute;
    let holder = device
        .new_buffer_with_bytes(window_words())
        .expect("holder buffer");
    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    let mut encoder = command.compute_command_encoder().expect("compute encoder");
    encoder
        .set_compute_pipeline_state(compute)
        .expect("declaring pipeline state");
    encoder
        .set_buffer(0, &holder.view(0, EXTENT as usize).expect("holder view"))
        .expect("the holder binds at the read slot");
    // The declaring kernel's second slot is a *write*, and the writeback
    // channel lands a writable slot in the bytes' own host image — which an
    // imported lease does not have. The refusal is by name.
    encoder
        .set_buffer_lease(
            1,
            stage_lease(provider, OBJECT_LEASE, StageBufferLeaseArm::BorrowedNoCopy),
        )
        .expect("the lease binds at the write slot");
    match encoder.dispatch_threads(
        Size::new(1, 1, 1).expect("grid"),
        Size::new(1, 1, 1).expect("local size"),
    ) {
        Err(ObjectError::WritableComputeLeaseUnsupported { index }) => assert_eq!(index, 1),
        other => panic!("expected WritableComputeLeaseUnsupported, got {other:?}"),
    }
}
