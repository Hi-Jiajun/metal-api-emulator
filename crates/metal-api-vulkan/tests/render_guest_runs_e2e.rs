//! The guest-runs arm for a render input's bytes (`research/docs/23` §74,
//! E-TX6).
//!
//! Census v19 (`evidence/gate3-census-v19-2026-09-18` §3) reads `load_seed` as
//! its second-largest first-failure bucket, 798 records (23.4 %): draws whose
//! previous contents are *guest bytes*, printed `seed=bytes` with the bytes
//! living behind a mapper-ref-texture mapping (`door=mapping`) or a linear GVA
//! (`door=gva`). The engine's own carrier for those contents is a run list —
//! `GuestTargetSeed`'s `GuestRunSource`, an ordered set of windows inside the
//! surface's registered pages plus its CPU alias — so the honest provider-side
//! declaration is a list, not one window.
//!
//! This file measures the arm that states that list: a `BufferSource::GuestRuns`
//! declaration whose windows the rail gathers out of the owner's imported
//! mappings at resolution and uploads as its own copy. Three readings, each
//! falsifiable:
//!
//! * the frame a two-run declaration lands is the frame the same bytes land
//!   through the owned arm and through the object rail, byte for byte — and the
//!   *uncovered* texels carry run 1's word then run 2's word, so a rail that
//!   swapped, dropped or snapshotted a run cannot pass;
//! * the owner rewriting both windows moves the frame with them (the gather
//!   reads the owner's pages, not a declaration-time copy);
//! * a run whose lease was never imported, and a run list that does not add up
//!   to the view's own length, are refused by name before device execution.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess, BufferLease,
    BufferSource, BufferView, CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace,
    Dispatch, DispatchKind, DispatchType, GuestRun, IndexBufferBinding, IndexFormat, LeaseId,
    LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId, PipelineProvider,
    ProviderErrorClass, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SemanticDigest, StoreOp, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(901);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(911);
const SCRATCH_VIEW: ViewId = ViewId::new(902);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(912);
const VERTEX_VIEW: ViewId = ViewId::new(903);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(913);
const INDEX_VIEW: ViewId = ViewId::new(904);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(914);

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The word run 1 carries — the *head* of the guest surface's pages.
const HEAD_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
/// The word run 2 carries — the surface's tail.
const TAIL_WORD: [u8; 4] = [0x55, 0x66, 0x77, 0x88];
/// The word the owner rewrites both windows with.
const MOVED_WORD: [u8; 4] = [0x9a, 0xbc, 0xde, 0xf0];

const HEAD_LEASE: LeaseId = LeaseId::new(61);
const TAIL_LEASE: LeaseId = LeaseId::new(62);

/// The bytes the two runs concatenate to: run 1's word twice, then run 2's.
fn run_words() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&HEAD_WORD.repeat(2));
    bytes.extend_from_slice(&TAIL_WORD.repeat(2));
    bytes
}

/// The owner's own mapping for one run, aligned to the device's import grid.
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

/// The fixture: one reviewed vertex-input pipeline over a 2×2 attachment, the
/// declaring compute case, and the resource table the trace is admitted
/// against. The attachment's own previous-contents source is what the caller
/// varies, so both the owned baseline and the guest-runs declaration run
/// through exactly the same pipeline, streams and index bytes.
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
/// triangles cover the texels at column 0 and leave column 1 for the loaded
/// bytes, which is what makes "the upload happened" readable per texel.
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

/// Build the fixture with one previous-contents source for the attachment's
/// view: the declaring compute case binds `source`, and the render pass opens
/// the attachment with `LoadOp::Load`.
fn fixture(source: BufferSource) -> Option<Fixture> {
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
        .compile_pipeline(&function, digest(b"render_guest_runs_declaring"))
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
            logical_digest: digest(b"render_guest_runs_stages"),
        })
        .expect("the vertex-input render pipeline registers");

    let vertex_bytes = left_column_vertex_bytes();
    let index_bytes = quad_index_bytes();
    let pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Load,
            store: StoreOp::Store,
        }],
        viewport: [0, 0, 2, 2],
        scissor: None,
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
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
                source: BufferSource::OwnedBytes(index_bytes.clone()),
            },
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(render.pipeline_id)
    };

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(12),
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

/// The object rail's frame for the same pass: an attachment whose view is the
/// caller's own host bytes, recorded with a declaring compute case and opened
/// with `RenderAttachmentLoad::Load`. The bytes are the two runs' concatenation,
/// so the two rails state the same previous contents through the two
/// declaration arms the canonical contract carries.
fn object_frame(fixture: &Fixture) -> Vec<u8> {
    use metal_api_core::provider::PipelineCompileRequest;
    use metal_api_core::provider::ShaderSource;
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<dyn PipelineProvider> =
        Arc::clone(&fixture.provider) as Arc<dyn PipelineProvider>;
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(&fixture.render)
        .expect("the registration wraps for the object API");
    let attachment = device
        .new_buffer_with_bytes(run_words())
        .expect("the attachment's own bytes are declared");
    let attachment_view = attachment
        .view(0, 16)
        .expect("the attachment view is declared");
    let stream = device
        .new_buffer_with_bytes(left_column_vertex_bytes())
        .expect("the vertex stream is declared");
    let stream_view = stream.view(0, 32).expect("the stream view is declared");
    let indices = device
        .new_buffer_with_bytes(quad_index_bytes())
        .expect("the index buffer is declared");
    let indices_view = indices.view(0, 12).expect("the index view is declared");

    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: SemanticDigest::new(
                    "metal-smoke-fixture-v1",
                    b"render_guest_runs_object_declaring".to_vec(),
                )
                .expect("digest"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let scratch = device
            .new_buffer_with_bytes(vec![0xab; 4])
            .expect("the scratch buffer is declared");
        let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(&declaring)
            .expect("compute pipeline state");
        encoder
            .set_buffer(0, &attachment_view)
            .expect("the attachment's bytes are the pass's declaration");
        encoder
            .set_buffer(1, &scratch_view)
            .expect("the kernel's write slot");
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
            .set_render_pipeline_state(&pipeline)
            .expect("the pipeline is bound");
        encoder
            .set_vertex_buffer(0, &stream_view)
            .expect("the stream is bound");
        encoder
            .set_index_buffer(&indices_view, IndexFormat::Uint16)
            .expect("the index buffer is bound");
        encoder
            .draw_indexed_primitives(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                2,
                2,
                RenderAttachmentLoad::Load,
                6,
                None,
            )
            .expect("the loading pass records");
        encoder.end_encoding().expect("the render encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    attachment
        .read()
        .expect("the attachment's landing bytes are readable")
}

/// The two-run declaration: run 1 is the surface's head, run 2 its tail, and
/// the pair is what the attachment's own 16-byte extent is made of.
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

#[test]
fn a_guest_runs_attachment_load_reads_the_owners_windows_in_order() {
    let Some(fixture) = fixture(BufferSource::OwnedBytes(run_words())) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    // The baseline: the same bytes as one owned declaration.
    let owned = attachment_frame(&provider, &fixture.trace, &fixture.resources);
    eprintln!("owned-bytes attachment frame: {}", hex(&owned));
    assert_eq!(
        owned,
        [QUAD_TEXEL, HEAD_WORD, QUAD_TEXEL, TAIL_WORD].concat(),
        "the left column is drawn and column 1 keeps the loaded words"
    );

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
                    .expect("the owner's run window is a valid reservation"),
            )
            .expect("the provider imports the owner's head window");
        provider
            .import_borrowed_lease(
                BorrowedLease::new(tail_reservation, tail.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's run window is a valid reservation"),
            )
            .expect("the provider imports the owner's tail window");
    }

    let mut gathered = fixture.trace.clone();
    if let Some(TracePass::Compute(pass)) = gathered.passes.first_mut() {
        pass.buffers[0].source = BufferSource::GuestRuns(two_runs());
    }
    let mut gathered_resources = fixture.resources.clone();
    for reservation in [head_reservation, tail_reservation] {
        gathered_resources
            .insert_lease(reservation)
            .expect("the run's reservation covers its window");
    }

    let frame = attachment_frame(&provider, &gathered, &gathered_resources);
    eprintln!("guest-runs attachment frame: {}", hex(&frame));
    assert_eq!(
        frame,
        owned,
        "the two runs gather to exactly the bytes the owned arm carries: {}",
        hex(&frame)
    );
    // The order reading: run 1's second word is the texel at row 0/column 1 and
    // run 2's the one at row 1/column 1, so a rail that read the list backwards
    // — or dropped a run — lands a different frame.
    assert_eq!(
        frame.chunks_exact(4).nth(1),
        Some(HEAD_WORD.as_slice()),
        "the head run is read first: {}",
        hex(&frame)
    );
    assert_eq!(
        frame.chunks_exact(4).nth(3),
        Some(TAIL_WORD.as_slice()),
        "the tail run is read second: {}",
        hex(&frame)
    );

    // The two rails: the object API's own declaration of the same previous
    // contents lands the same frame.
    let object = object_frame(&fixture);
    eprintln!("object-rail attachment frame: {}", hex(&object));
    assert_eq!(
        object, owned,
        "the object rail states the same previous contents and lands the same frame"
    );

    // The holds the gathered read took are retired by the submission's fence.
    let registry = provider.borrowed_registry();
    for reservation in [head_reservation, tail_reservation] {
        assert_eq!(
            registry.outstanding(reservation.lease.lease_id),
            Some(0),
            "the run's hold is retired once the pass's fence signals"
        );
    }

    // The seed reading: the owner rewrites both windows, and the next
    // submission's frame follows them. A rail that snapshotted the windows when
    // the declaration was admitted would keep the old words.
    head.as_mut_slice().copy_from_slice(&MOVED_WORD.repeat(2));
    tail.as_mut_slice().copy_from_slice(&MOVED_WORD.repeat(2));
    let moved = attachment_frame(&provider, &gathered, &gathered_resources);
    eprintln!("owner-rewritten guest-runs frame: {}", hex(&moved));
    assert_eq!(
        moved,
        [QUAD_TEXEL, MOVED_WORD, QUAD_TEXEL, MOVED_WORD].concat(),
        "the owner's rewritten windows are what the uncovered texels upload: {}",
        hex(&moved)
    );
}

#[test]
fn a_guest_runs_run_naming_an_unimported_lease_is_refused_by_name() {
    let Some(fixture) = fixture(BufferSource::GuestRuns(two_runs())) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let alignment = provider.no_copy_alignment();
    if alignment == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let [head_reservation, tail_reservation] = reservations(&provider);
    // Only run 1's lease is imported: the arm resolves the second run through
    // the registry and is refused there, before a device object exists.
    let mut head = AlignedBuffer::new(8, alignment as usize);
    head.as_mut_slice().copy_from_slice(&HEAD_WORD.repeat(2));
    // Safety: the allocation outlives the submission and the release below.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(head_reservation, head.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's run window is a valid reservation"),
            )
            .expect("the provider imports the owner's run window");
    }
    let mut resources = fixture.resources.clone();
    for reservation in [head_reservation, tail_reservation] {
        resources
            .insert_lease(reservation)
            .expect("both reservations are admitted: the arm's own checks are what answer");
    }
    let admitted = provider
        .capabilities()
        .validate_trace(fixture.trace.clone(), resources)
        .expect("the declaration is well formed: admission does not ask which leases are imported");
    let refused = provider
        .submit(admitted)
        .expect_err("a run whose lease was never imported cannot be read");
    eprintln!("unimported guest run refused: {refused:?}");
    assert_eq!(refused.slug, "lease_not_imported");
    assert_eq!(refused.class, ProviderErrorClass::Args);
    // The refusal rolled back the hold the first run had already taken.
    let registry = provider.borrowed_registry();
    assert_eq!(registry.outstanding(HEAD_LEASE), Some(0));

    // A run list that is not the view's own length is refused at admission:
    // the runs are the view's bytes and nothing else.
    let mut short = fixture.trace.clone();
    if let Some(TracePass::Compute(pass)) = short.passes.first_mut() {
        pass.buffers[0].source = BufferSource::GuestRuns(vec![GuestRun {
            lease_id: HEAD_LEASE,
            offset: 0,
            length: 8,
        }]);
    }
    let mut resources = fixture.resources.clone();
    resources
        .insert_lease(head_reservation)
        .expect("the run's reservation is admitted");
    let refused = provider
        .capabilities()
        .validate_trace(short, resources)
        .expect_err("a run list of the wrong length is not the view's declaration");
    eprintln!("short guest-runs list refused: {refused:?}");
    assert_eq!(refused.slug, "buffer_source_length_mismatch");
    assert_eq!(
        refused.detail.as_deref(),
        Some("view ViewId(901) source length 8 does not match declared length 16"),
        "the refusal names the view and the length the runs do not add up to"
    );

    provider
        .release_borrowed_lease(HEAD_LEASE)
        .expect("no retain is outstanding after the refusals");
}

#[test]
fn a_guest_runs_run_outside_its_reservation_is_refused_at_admission() {
    // The run's own window is bounded by the reservation the snapshot admits:
    // a run that reaches past it is refused by the contract's own check
    // (`LeaseRangeOutOfBounds`) rather than read out of range by the rail.
    let Some(fixture) = fixture(BufferSource::OwnedBytes(run_words())) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let [head_reservation, tail_reservation] = reservations(&provider);
    let mut trace = fixture.trace.clone();
    if let Some(TracePass::Compute(pass)) = trace.passes.first_mut() {
        pass.buffers[0].source = BufferSource::GuestRuns(vec![
            GuestRun {
                lease_id: HEAD_LEASE,
                offset: 0,
                length: 8,
            },
            GuestRun {
                lease_id: TAIL_LEASE,
                offset: 4,
                length: 8,
            },
        ]);
    }
    let mut resources = fixture.resources.clone();
    for reservation in [head_reservation, tail_reservation] {
        resources
            .insert_lease(reservation)
            .expect("both reservations are admitted");
    }
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a run past its reservation is refused");
    eprintln!("out-of-reservation guest run refused: {refused:?}");
    assert_eq!(refused.slug, "resource_contract_invalid");
    assert_eq!(
        refused.detail.as_deref(),
        Some("lease LeaseId(62) range end 12 exceeds allocation size 8"),
        "the refusal names the run's own lease and the bound it crossed"
    );
}
