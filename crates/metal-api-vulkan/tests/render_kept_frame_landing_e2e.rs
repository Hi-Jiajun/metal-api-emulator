//! The kept-frame landing entry (`research/docs/23` §115 之后的增量, E-TX14/R4b).
//!
//! fp3's census reads the two *delayed store* tails (`provider_held_store_gva`
//! 9 464/round, `provider_held_store_surface` 547/round) as records whose owner
//! wanted the frame to stay in the provider's image until a real reader asked
//! for it — and whose only way out today is a whole-frame relay back through the
//! engine. `StoreOp::Borrowed` / `BorrowedLanding` both land a *pass's own*
//! frame in the same completion; neither can deliver a frame a *previous* pass
//! kept.
//!
//! This file measures the entry that can. One submission carries a pass that
//! keeps its frame in the provider's image (`StoreOp::Resident`) and then a
//! landing-only entry that delivers that frame into the owner's registered
//! window, with no draw anywhere in the entry. Every reading is falsifiable:
//!
//! * the owner's window holds the frame the keeping pass produced, byte for
//!   byte, and the kept image still holds it afterwards (the entry copies out;
//!   it does not draw);
//! * a second landing of the same identity is refused by name — the first one
//!   consumed it, which is what keeps an old frame from being written twice;
//! * a landing that stands *before* the pass that keeps the frame, and one for
//!   an identity nothing ever kept, are each refused by name rather than
//!   waiting for a later pass;
//! * a landing view the trace never declares, one whose source is a copy arm,
//!   and one whose bytes are not the kept frame's extent are refused by name;
//! * a snapshot that does not declare the entry's capability bit refuses the
//!   whole trace during admission.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, AttachmentLandingView, BorrowedLease,
    BufferAccess, BufferLease, BufferSource, BufferView, ClearColor, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, GuestRun,
    IndexBufferBinding, IndexFormat, KeptFrame, KeptFrameLanding, LeaseId, LeaseReservation,
    LoadOp, NoCopyLeaseImporter, OperationId, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor};
use std::sync::Arc;

/// The reviewed indexed vertex stage: `vec2` position from the caller's stream.
const QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/quad_indexed.vert.spv");
/// The reviewed 8-bit UNORM fragment stage: `(64/255, 128/255, 192/255, 1)`.
const SOLID_UNORM8_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/solid_unorm8.frag.spv");
/// The reviewed declaring kernel, used to declare the window view in the pool.
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The identity whose image the keeping pass leaves the frame in.
const ATTACHMENT_VIEW: ViewId = ViewId::new(941);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(951);
const SCRATCH_VIEW: ViewId = ViewId::new(942);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(952);
const VERTEX_VIEW: ViewId = ViewId::new(943);
const VERTEX_ALLOCATION: AllocationId = AllocationId::new(953);
const INDEX_VIEW: ViewId = ViewId::new(944);
const INDEX_ALLOCATION: AllocationId = AllocationId::new(954);
/// The owner's registered window the entry delivers into.
const LANDING_VIEW: ViewId = ViewId::new(945);
const LANDING_ALLOCATION: AllocationId = AllocationId::new(955);
const WINDOW_SCRATCH_VIEW: ViewId = ViewId::new(946);
const WINDOW_SCRATCH_ALLOCATION: AllocationId = AllocationId::new(956);
/// An identity no pass ever keeps, named by one refusal test.
const NEVER_KEPT_VIEW: ViewId = ViewId::new(947);
const NEVER_KEPT_ALLOCATION: AllocationId = AllocationId::new(957);
/// An identity the trace never declares a view for, named by one refusal test.
const UNDECLARED_LANDING_VIEW: ViewId = ViewId::new(948);
const UNDECLARED_LANDING_ALLOCATION: AllocationId = AllocationId::new(958);

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];
/// The clear the keeping pass opens with: the texels the draw does not cover
/// hold exactly these bytes, so "the frame is the pass's own" is readable.
const CLEAR_TEXEL: [u8; 4] = [0x0a, 0x0b, 0x0c, 0x0d];
/// What the owner's window holds *before* the entry runs — deliberately unlike
/// the frame, so an entry that never ran shows up as these bytes.
const WINDOW_TEXEL: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
/// A second window content: the entry is a write direction, so the frame it
/// delivers must not move when only this changes.
const OTHER_WINDOW_TEXEL: [u8; 4] = [0xaa, 0xbb, 0xcc, 0xdd];

const LANDING_LEASE: LeaseId = LeaseId::new(91);
const SHORT_LANDING_LEASE: LeaseId = LeaseId::new(92);

fn words(texel: [u8; 4]) -> Vec<u8> {
    texel.repeat(4)
}

/// The frame the keeping pass produces: the drawn left column beside the clear
/// the pass opened with, in row-major texel order.
fn kept_frame_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for _row in 0..2 {
        bytes.extend_from_slice(&QUAD_TEXEL);
        bytes.extend_from_slice(&CLEAR_TEXEL);
    }
    bytes
}

/// The owner's own mapping for a landing window, aligned to the device's import
/// grid.
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
/// triangles cover column 0 and leave column 1 for the clear, which is what
/// makes "the frame is the pass's" readable per texel.
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

/// Build a fixture whose trace keeps the frame and then lands it.
///
/// `landing` is the entry's own second declaration (`None` leaves the entry out
/// and leaves the trace with just the keeping pass), `window` is the byte length
/// the window's declaration and its reservation state, and `position` says
/// whether the entry stands after the keeping pass or before it.
fn fixture(
    landing: Option<AttachmentLandingView>,
    window_bytes: u64,
    window_source: BufferSource,
    window_lease: LeaseId,
    entry_first: bool,
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
        .compile_pipeline(&function, digest(b"kept_frame_landing_declaring"))
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
            logical_digest: digest(b"kept_frame_landing_stages"),
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
    // The keeping pass: it clears, draws the left column and keeps the frame in
    // the provider's image. Nothing about it publishes the frame, which is the
    // shape the entry exists to deliver.
    let keeping_pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Clear(ClearColor::new(CLEAR_TEXEL)),
            store: StoreOp::Resident,
        }],
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
        vertex_buffers: vec![vertex_buffer],
        indices: Some(IndexBufferBinding {
            view: index_buffer,
            format: IndexFormat::Uint16,
        }),
        ..render_pass_defaults(render.pipeline_id)
    };

    // The attachment's own declaration. The keeping pass reads nothing out of
    // it (`Clear` presets its contents and a resident store publishes nothing),
    // but the identity has to be in the trace's serial view list for a later
    // pass that reads the kept image back through the writeback channel — the
    // reading the positive case asserts.
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
                source: BufferSource::OwnedBytes(vec![0x7e; 16]),
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
    // The window's own declaration: a producer's trace puts every view it names
    // in this serial list, so the entry's second declaration is resolvable there
    // and nowhere else.
    passes.push(TracePass::Compute(ComputePass {
        pipeline: compute.pipeline_id,
        buffers: vec![
            BufferView {
                view_id: LANDING_VIEW,
                metal_binding: 0,
                allocation_id: LANDING_ALLOCATION,
                offset: 0,
                length: window_bytes,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: window_source,
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
    let entry = landing.map(|landing| {
        TracePass::Landing(KeptFrameLanding {
            frame: KeptFrame {
                allocation_id: ATTACHMENT_ALLOCATION,
                view_id: ATTACHMENT_VIEW,
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
            },
            landing,
        })
    });
    if entry_first {
        passes.extend(entry.clone());
    }
    passes.push(TracePass::Render(keeping_pass));
    if !entry_first {
        passes.extend(entry);
    }

    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(24),
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
    resources
        .insert_lease(LeaseReservation {
            lease: BufferLease {
                lease_id: window_lease,
                allocation_id: LANDING_ALLOCATION,
                owner_epoch: provider.device_epoch(),
            },
            offset: 0,
            length: window_bytes,
        })
        .expect("the window's reservation covers the declaration");

    Some(Fixture {
        provider: Arc::new(provider),
        trace,
        resources,
    })
}

/// The device's host-import alignment, or `None` when it cannot import host
/// memory at all — in which case the owner-window readings in this file skip,
/// exactly as the E-TX8/E-TX13 landings do.
fn import_alignment(provider: &VulkanComputeProvider) -> Option<usize> {
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return None;
    }
    Some(usize::try_from(provider.no_copy_alignment()).expect("the alignment fits usize"))
}

/// Import one owner window holding `texel` over `window_bytes` bytes under the
/// lease the trace's declaration names.
fn import_window(
    provider: &VulkanComputeProvider,
    window: &mut AlignedBuffer,
    window_bytes: u64,
    texel: [u8; 4],
    lease_id: LeaseId,
) {
    window.as_mut_slice().fill(0x5a);
    let prefix = usize::try_from(window_bytes).expect("the window fits the host buffer");
    for texel_slot in window.as_mut_slice()[..prefix].chunks_exact_mut(4) {
        texel_slot.copy_from_slice(&texel);
    }
    // Safety: the owner allocation outlives every submission below and the
    // provider's release of the import.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(
                    LeaseReservation {
                        lease: BufferLease {
                            lease_id,
                            allocation_id: LANDING_ALLOCATION,
                            owner_epoch: provider.device_epoch(),
                        },
                        offset: 0,
                        length: window_bytes,
                    },
                    window.as_mut_slice().as_ptr() as usize,
                )
                .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's window");
    }
}

/// Submit one trace through admission, returning its refusal when there is one.
fn submit(
    provider: &VulkanComputeProvider,
    trace: &ComputeTrace,
    resources: &ResourceTableSnapshot,
) -> Result<metal_api_core::provider::ProviderSubmission, metal_api_core::provider::ProviderError> {
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())?;
    provider.submit(admitted)
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

#[test]
fn a_kept_frame_lands_in_the_window_a_later_entry_names() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    let submitted = submit(&provider, &fixture.trace, &fixture.resources)
        .expect("the keeping pass and the landing entry run");
    submitted
        .validate_for_trace(&fixture.trace)
        .expect("a resident store owes no writeback and the entry owes none either");
    // The declaring compute passes publish their own scratch writes; what the
    // entry must not add is a writeback for the frame or for the window it
    // delivered into — the frame's new home is the owner's pages.
    assert!(
        submitted.writebacks.iter().all(|writeback| {
            writeback.view_id != ATTACHMENT_VIEW && writeback.view_id != LANDING_VIEW
        }),
        "the entry publishes no writeback for the frame: {:?}",
        submitted
            .writebacks
            .iter()
            .map(|writeback| (writeback.view_id, writeback.allocation_id))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        hex(&window.as_mut_slice()[..16]),
        hex(&kept_frame_bytes()),
        "the window holds the frame the keeping pass left in the provider's image"
    );

    // The entry copied the frame out; it did not draw. A later pass that loads
    // the resident contents and stores them through the writeback channel reads
    // exactly the same bytes back, which is the reading a rail that re-rendered
    // or cleared the image would fail.
    let readback = resident_readback(&provider, &fixture);
    assert_eq!(
        hex(&readback),
        hex(&kept_frame_bytes()),
        "the kept image still holds the pass's own frame after the landing"
    );
}

/// Submit a second trace that reads the resident image back through the
/// writeback channel (`LoadOp::Resident` + `StoreOp::Store`).
fn resident_readback(provider: &VulkanComputeProvider, fixture: &Fixture) -> Vec<u8> {
    let mut trace = fixture.trace.clone();
    trace.operation_id = OperationId::new(25);
    let TracePass::Render(pass) = trace
        .passes
        .iter_mut()
        .find(|entry| matches!(entry, TracePass::Render(_)))
        .expect("the fixture keeps a render pass")
    else {
        unreachable!("the find matched a render entry");
    };
    pass.color_attachments[0].load = LoadOp::Resident;
    pass.color_attachments[0].store = StoreOp::Store;
    trace
        .passes
        .retain(|entry| !matches!(entry, TracePass::Landing(_)));
    let resources = fixture.resources.clone();
    let submitted = submit(provider, &trace, &resources)
        .expect("a resident load beside a store reads the kept frame back");
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the resident load publishes the frame it read")
}

#[test]
fn a_second_landing_of_one_kept_frame_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    let mut twice = fixture.trace.clone();
    twice.operation_id = OperationId::new(26);
    let entry = twice
        .passes
        .iter()
        .find(|pass| matches!(pass, TracePass::Landing(_)))
        .cloned()
        .expect("the fixture carries the entry");
    twice.passes.push(entry);
    let error = submit(&provider, &twice, &fixture.resources)
        .expect_err("the first landing consumes the identity, so the second is refused");
    assert_eq!(error.slug, "kept_frame_already_landed");
    assert_eq!(error.class, ProviderErrorClass::Capability);
}

#[test]
fn a_landing_before_its_keeping_pass_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        true,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    let error = submit(&provider, &fixture.trace, &fixture.resources)
        .expect_err("a frame that is not kept yet cannot be landed");
    assert_eq!(error.slug, "kept_frame_not_held");
    assert_eq!(error.class, ProviderErrorClass::Capability);
}

#[test]
fn a_landing_of_a_frame_no_pass_ever_kept_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    // An identity nothing ever kept, in a submission of its own: the registry
    // has no entry and no tombstone, so the refusal is `kept_frame_not_held`.
    let mut trace = fixture.trace.clone();
    trace.operation_id = OperationId::new(27);
    let render_pipeline = trace
        .passes
        .iter()
        .find_map(|entry| entry.as_render())
        .map(|pass| pass.pipeline)
        .expect("the fixture keeps a render pass");
    trace
        .passes
        .retain(|entry| !matches!(entry, TracePass::Render(_)));
    // The table's entries have to be used at least once, so the pass that is
    // gone takes its pipeline registration with it.
    trace
        .pipelines
        .retain(|pipeline| pipeline.pipeline_id != render_pipeline);
    if let Some(TracePass::Landing(entry)) = trace
        .passes
        .iter_mut()
        .find(|pass| matches!(pass, TracePass::Landing(_)))
    {
        entry.frame.view_id = NEVER_KEPT_VIEW;
        entry.frame.allocation_id = NEVER_KEPT_ALLOCATION;
    }
    let mut resources = fixture.resources.clone();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: NEVER_KEPT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 16,
        })
        .expect("the never-kept identity's allocation");
    let error =
        submit(&provider, &trace, &resources).expect_err("the registry never held this identity");
    assert_eq!(error.slug, "kept_frame_not_held");
}

#[test]
fn a_landing_window_without_its_own_declaration_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: UNDECLARED_LANDING_ALLOCATION,
        view_id: UNDECLARED_LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    let mut resources = fixture.resources.clone();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: UNDECLARED_LANDING_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 16,
        })
        .expect("the undeclared window's allocation");
    let error = submit(&provider, &fixture.trace, &resources)
        .expect_err("a landing view the trace never declares has no window to resolve");
    assert_eq!(error.slug, "kept_frame_landing_undeclared");
}

#[test]
fn a_landing_window_of_another_length_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    // A thirty-two-byte window against a sixteen-byte frame: the declaration
    // has to be the frame's own tightly packed extent. (The window is twice the
    // frame rather than half of it because a host import smaller than the
    // device's own granularity is refused by some ICDs at buffer creation —
    // Dozen on the RTX 5060 run does — and that refusal would be a reading about
    // the import, not about this rail's own length rule.)
    let Some(fixture) = fixture(
        Some(landing),
        32,
        BufferSource::BorrowedNoCopy(SHORT_LANDING_LEASE),
        SHORT_LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(
        &provider,
        &mut window,
        32,
        WINDOW_TEXEL,
        SHORT_LANDING_LEASE,
    );

    let error = submit(&provider, &fixture.trace, &fixture.resources)
        .expect_err("a window of another length cannot receive the frame");
    assert_eq!(error.slug, "kept_frame_landing_mismatch");
    assert_eq!(error.class, ProviderErrorClass::Args);
}

#[test]
fn a_landing_window_that_is_a_copy_arm_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::OwnedBytes(words(WINDOW_TEXEL)),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    let error = submit(&provider, &fixture.trace, &fixture.resources)
        .expect_err("a copy arm is not a window an owner's ledger holds");
    assert_eq!(error.slug, "kept_frame_landing_unsupported");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert!(
        error
            .fields
            .get("source")
            .is_some_and(|value| matches!(value, metal_api_core::provider::FieldValue::Text(text) if text == "owned_bytes")),
        "the refusal names the copy arm it found: {:?}",
        error.fields
    );
}

#[test]
fn a_snapshot_without_the_bit_refuses_the_entry_during_admission() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    assert!(
        provider
            .capabilities()
            .declares_render_kept_frame_landing_support(),
        "the Vulkan snapshot declares the entry it executes"
    );
    let mut without = provider.capabilities();
    without.supports_render_kept_frame_landing = false;
    let error = without
        .validate_trace(fixture.trace.clone(), fixture.resources.clone())
        .expect_err("a snapshot that keeps no frames refuses the entry by name");
    assert_eq!(error.slug, "kept_frame_landing_unsupported");
}

#[test]
fn the_window_a_landing_writes_does_not_change_the_frame_it_delivers() {
    // The entry is a write direction: moving the window's old bytes moves
    // nothing, which is the reading a rail that read the frame *out of* the
    // window (rather than out of the provider's image) would fail.
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(
        &provider,
        &mut window,
        16,
        OTHER_WINDOW_TEXEL,
        LANDING_LEASE,
    );

    submit(&provider, &fixture.trace, &fixture.resources).expect("the entry runs");
    assert_eq!(
        hex(&window.as_mut_slice()[..16]),
        hex(&kept_frame_bytes()),
        "the delivered frame is the provider's own, whatever the window held"
    );
}

/// The second owner-window arm (`BufferSource::GuestRuns`): the declaration is
/// a run list whose concatenation is the frame's own extent, and the rail
/// resolves every run out of the owner's imported pages exactly as it does for
/// an attachment landing (`research/docs/23` §115 之后的增量，E-TX14/R4b).
#[test]
fn a_landing_through_a_guest_run_list_lands_the_same_frame() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::GuestRuns(vec![GuestRun {
            lease_id: LANDING_LEASE,
            offset: 0,
            length: 16,
        }]),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    submit(&provider, &fixture.trace, &fixture.resources)
        .expect("the run list lands the kept frame");
    assert_eq!(
        hex(&window.as_mut_slice()[..16]),
        hex(&kept_frame_bytes()),
        "the guest-run arm delivers the same bytes the no-copy arm does"
    );
}

/// The identity a landing names can be *gone*: the resident registry keeps
/// [`metal_api_vulkan::RESIDENT_TARGET_BUDGET`] identities, and the one the
/// budget evicts leaves a tombstone naming the rule. A landing for it is
/// refused by name rather than served from the image it used to hold
/// (`research/docs/23` §115 之后的增量，E-TX14/R4b).
#[test]
fn a_landing_of_an_identity_the_budget_evicted_is_refused_by_name() {
    let landing = AttachmentLandingView {
        allocation_id: LANDING_ALLOCATION,
        view_id: LANDING_VIEW,
    };
    let Some(fixture) = fixture(
        Some(landing),
        16,
        BufferSource::BorrowedNoCopy(LANDING_LEASE),
        LANDING_LEASE,
        false,
    ) else {
        return;
    };
    let provider = Arc::clone(&fixture.provider);
    let Some(alignment) = import_alignment(&provider) else {
        return;
    };
    let mut window = AlignedBuffer::new(64 * 1024, alignment);
    import_window(&provider, &mut window, 16, WINDOW_TEXEL, LANDING_LEASE);

    // Keep one frame per identity, one more than the budget holds: the first
    // identity is the least recently used one, so it is the one evicted.
    let mut keep = fixture.trace.clone();
    keep.passes
        .retain(|entry| !matches!(entry, TracePass::Landing(_)));
    let first = (AllocationId::new(2000), ViewId::new(1000));
    for index in
        0..=u64::try_from(metal_api_vulkan::RESIDENT_TARGET_BUDGET).expect("the budget fits u64")
    {
        let view = ViewId::new(1000 + index);
        let allocation = AllocationId::new(2000 + index);
        let Some(TracePass::Render(pass)) = keep
            .passes
            .iter_mut()
            .find(|entry| matches!(entry, TracePass::Render(_)))
        else {
            panic!("the fixture keeps a render pass");
        };
        pass.color_attachments[0].view_id = view;
        pass.color_attachments[0].allocation_id = allocation;
        // Every render attachment's identity has to be declared by the trace,
        // so the declaring pass moves to the identity this round keeps.
        let Some(TracePass::Compute(declaring)) = keep
            .passes
            .iter_mut()
            .find(|entry| matches!(entry, TracePass::Compute(_)))
        else {
            panic!("the fixture declares the attachment's own view");
        };
        declaring.buffers[0].view_id = view;
        declaring.buffers[0].allocation_id = allocation;
        keep.operation_id = OperationId::new(40 + index);
        let mut resources = fixture.resources.clone();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size: 16,
            })
            .expect("the kept identity's allocation");
        submit(&provider, &keep, &resources).expect("the keeping pass runs");
    }
    assert_eq!(
        provider.resident_target_count(),
        metal_api_vulkan::RESIDENT_TARGET_BUDGET
    );
    assert!(provider.resident_target_evictions() >= 1);
    assert!(!provider.resident_target_is_live(first.0, first.1));

    let mut land = fixture.trace.clone();
    land.operation_id = OperationId::new(70);
    let Some(TracePass::Landing(entry)) = land
        .passes
        .iter_mut()
        .find(|entry| matches!(entry, TracePass::Landing(_)))
    else {
        panic!("the fixture keeps the entry");
    };
    entry.frame.view_id = first.1;
    entry.frame.allocation_id = first.0;
    let mut resources = fixture.resources.clone();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: first.0,
            owner_epoch: provider.device_epoch(),
            size: 16,
        })
        .expect("the evicted identity's allocation");
    let error = submit(&provider, &land, &resources)
        .expect_err("the budget retired this identity, so there is no frame to deliver");
    assert_eq!(error.slug, "kept_frame_evicted");
    assert_eq!(error.class, ProviderErrorClass::Capability);
    assert_eq!(
        error.fields.get("retired_by"),
        Some(&metal_api_core::provider::FieldValue::Text(
            "budget".to_owned()
        )),
        "the refusal names the rule that retired the identity"
    );
}
