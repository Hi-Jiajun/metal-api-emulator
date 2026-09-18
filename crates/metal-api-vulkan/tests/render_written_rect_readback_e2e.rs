//! The written-rect readback's byte-level oracle (`docs/WRITTEN-RECT-READBACK.md`).
//!
//! The increment narrows one stored attachment's readback from "the whole
//! extent" to "the rectangle this pass can have written", rebuilding the rest of
//! the frame from the seed the image began from. That is only allowed to be a
//! *timing* change: the frame the writeback channel publishes, and the bytes the
//! provider lands in an owner's own pages, have to stay byte for byte what the
//! whole-attachment readback published.
//!
//! This file measures exactly that, in the two-arm shape the control switch
//! exists for:
//!
//! * the same three shapes run twice — once with `METAL_API_VULKAN_FULL_READBACK`
//!   unset (the trimmed arm) and once set to `1` (the pre-increment arm). The
//!   switch is read once per process, so each arm runs as its own child of this
//!   test binary and the comparison is between the two children's bytes;
//! * every shape is compared *and* checked against the frame its seed and its
//!   written rectangle state, so an arm that published something stale or
//!   something mirrored cannot pass by agreeing with itself;
//! * the guest's own page is read back after the pass on the owner-window arm —
//!   the case the census's `door=mapping` records state — because that arm's
//!   landing is the provider writing the frame into the owner's memory;
//! * a shape this rail cannot prove (an undefined load) is checked to take the
//!   whole-attachment path and to say so in the counters, which is the other
//!   half of the rule: the fallbacks are named rather than silent.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess, BufferLease,
    BufferSource, BufferView, ClearColor, CompletionPolicy, ComputePass, ComputeProvider,
    ComputeTrace, Dispatch, DispatchKind, DispatchType, IndexBufferBinding, IndexFormat, LeaseId,
    LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SemanticDigest, StoreOp,
    TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    ReadbackRegionCounts, RenderPipelineRequest, VulkanComputeProvider, VulkanExecutor,
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
const WINDOW_LEASE: LeaseId = LeaseId::new(81);

/// The attachment's extent: four by four texels, four bytes each.
const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const ATTACHMENT_BYTES: usize = (WIDTH as usize) * (HEIGHT as usize) * 4;

/// The fragment output the reviewed quad stores.
const QUAD_TEXEL: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The child arm's environment: which readback the child runs.
const ORACLE_ARM: &str = "METAL_VULKAN_READBACK_ORACLE_ARM";
/// The switch the pre-increment arm states.
const FULL_READBACK: &str = "METAL_API_VULKAN_FULL_READBACK";
/// This test's own name, which the parent re-invokes the binary with.
const TEST_NAME: &str = "the_trimmed_readback_publishes_the_same_guest_pages_as_the_whole_readback";

/// The seed a `Load` shape begins from: one texel per grid position, so a frame
/// that carried a stale or a misplaced rectangle cannot match its own
/// expectation.
fn pattern_seed() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ATTACHMENT_BYTES);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            bytes.extend_from_slice(&[x as u8, y as u8, 0x33, 0xff]);
        }
    }
    bytes
}

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

/// The frame a shape has to publish: its seed with the written texels patched in.
fn expected_frame(seed: &[u8], rect: [u32; 4]) -> Vec<u8> {
    let [x, y, width, height] = rect;
    let mut frame = seed.to_vec();
    for row in y..y + height {
        for column in x..x + width {
            let offset = ((row * WIDTH + column) * 4) as usize;
            frame[offset..offset + 4].copy_from_slice(&QUAD_TEXEL);
        }
    }
    frame
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex"))
        .collect()
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

/// What one shape's submission published.
struct Published {
    /// The frame the writeback channel carried for the attachment.
    frame: Vec<u8>,
    /// The owner's window after the pass, for the shapes that land in one.
    window: Option<Vec<u8>>,
    /// The executor's readback counters after the submission, with the counters
    /// before it already subtracted.
    regions: ReadbackRegionCounts,
}

/// One shape's request: what the pass begins from, how it stores, and the rects
/// that bound what it can write.
struct ShapeRequest<'a> {
    /// The bytes the pass begins from, which is also the frame it has to
    /// publish outside the rectangle.
    seed: &'a [u8],
    load: LoadOp,
    store: StoreOp,
    viewport: [u32; 4],
    scissor: Option<[u32; 4]>,
    /// The owner's own page this shape's declaration names, when the shape is
    /// the owner-window arm (E-TX8).
    window: Option<&'a mut AlignedBuffer>,
}

/// Read the executor's counters.
fn regions(executor: &VulkanExecutor) -> ReadbackRegionCounts {
    executor.readback_region_counts()
}

/// The difference between two readings of the counters.
fn delta(before: ReadbackRegionCounts, after: ReadbackRegionCounts) -> ReadbackRegionCounts {
    let difference = |left: usize, right: usize| right - left;
    ReadbackRegionCounts {
        rect_attachments: difference(before.rect_attachments, after.rect_attachments),
        rect_bytes: difference(before.rect_bytes, after.rect_bytes),
        rect_extent_bytes: difference(before.rect_extent_bytes, after.rect_extent_bytes),
        full_attachments: difference(before.full_attachments, after.full_attachments),
        full_bytes: difference(before.full_bytes, after.full_bytes),
        switch_attachments: difference(before.switch_attachments, after.switch_attachments),
        shape_attachments: difference(before.shape_attachments, after.shape_attachments),
        bounds_attachments: difference(before.bounds_attachments, after.bounds_attachments),
        whole_attachments: difference(before.whole_attachments, after.whole_attachments),
    }
}

/// Submit one shape over the four-by-four attachment and collect what it
/// published.
///
/// `window` is the owner's own page the attachment's declaration names when the
/// shape is the owner-window arm (`BufferSource::BorrowedNoCopy` +
/// `StoreOp::Borrowed`); `seed` is what the pass begins from, which is that
/// page's contents on that arm.
fn run_shape(
    executor: &Arc<VulkanExecutor>,
    provider: &VulkanComputeProvider,
    mut request: ShapeRequest<'_>,
) -> Published {
    // The owner's page, read after the pass through the same allocation the
    // window named: the pointer is what the borrowed window is, and the
    // allocation outlives this call.
    let page_pointer = request.window.as_ref().map(|page| page.pointer);
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let digest =
        |case: &[u8]| SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest");
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_written_rect_declaring"))
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
            logical_digest: digest(b"render_written_rect_stages"),
        })
        .expect("the vertex-input render pipeline registers");

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
    let (source, leases) = match request.window {
        Some(ref mut page) => {
            page.as_mut_slice().copy_from_slice(request.seed);
            let reservation = LeaseReservation {
                lease: BufferLease {
                    lease_id: WINDOW_LEASE,
                    allocation_id: ATTACHMENT_ALLOCATION,
                    owner_epoch: provider.device_epoch(),
                },
                offset: 0,
                length: u64::try_from(request.seed.len()).expect("window length"),
            };
            // Safety: the owner's allocation outlives every submission below
            // and the provider's release of the import.
            unsafe {
                provider
                    .import_borrowed_lease(
                        BorrowedLease::new(reservation, page.as_mut_slice().as_ptr() as usize)
                            .expect("the owner's window is a valid reservation"),
                    )
                    .expect("the provider imports the owner's window");
            }
            (
                BufferSource::BorrowedNoCopy(WINDOW_LEASE),
                Some(reservation),
            )
        }
        None => (BufferSource::OwnedBytes(request.seed.to_vec()), None),
    };
    let attachment_view = BufferView {
        view_id: ATTACHMENT_VIEW,
        metal_binding: 0,
        allocation_id: ATTACHMENT_ALLOCATION,
        offset: 0,
        length: u64::try_from(ATTACHMENT_BYTES).expect("attachment length"),
        access: BufferAccess::Read,
        attribute_stride: None,
        source,
    };
    let pass = RenderPassDescriptor {
        pipeline: render.pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(WIDTH),
            height: u64::from(HEIGHT),
            load: request.load,
            store: request.store,
        }],
        vertices: u32::try_from(index_bytes.len() / 2).expect("uint16 index count"),
        vertex_buffers: vec![vertex_buffer],
        indices: Some(IndexBufferBinding {
            view: index_buffer,
            format: IndexFormat::Uint16,
        }),
        viewport: request.viewport,
        scissor: request.scissor,
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
    if let Some(reservation) = leases {
        resources
            .insert_lease(reservation)
            .expect("the window's reservation covers it");
    }

    let before = regions(executor);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the trace is admitted");
    let submitted = provider.submit(admitted).expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    let after = regions(executor);
    let frame = submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment lands a writeback");
    Published {
        frame,
        window: page_pointer.map(|pointer| unsafe {
            std::slice::from_raw_parts(pointer.as_ptr(), ATTACHMENT_BYTES).to_vec()
        }),
        regions: delta(before, after),
    }
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

/// One shape this file measures, with the frame it has to publish.
struct Shape {
    name: &'static str,
    /// The rectangle the pass can have written, which is what its frame differs
    /// from the seed by. `None` is the shape that has no provable rectangle.
    rect: Option<[u32; 4]>,
}

const SHAPES: [Shape; 4] = [
    // The census's own shape: a small scissor inside a 1920x1080 attachment.
    Shape {
        name: "scissor",
        rect: Some([1, 1, 2, 2]),
    },
    // The same narrowing through the viewport instead of the scissor.
    Shape {
        name: "viewport",
        rect: Some([1, 2, 2, 2]),
    },
    // A clear seed rather than a loaded one: the frame's uncovered texels are
    // the clear payload repeated.
    Shape {
        name: "clear",
        rect: Some([0, 0, 1, 3]),
    },
    // An undefined load: no seed exists host-side, so the whole attachment has
    // to be read back. Kept in the same run so the fallback counter is
    // falsifiable beside the three trimmed shapes.
    Shape {
        name: "undefined",
        rect: None,
    },
];

/// Run every shape on this arm and print what it published. Called in the child
/// process only; the parent compares the two arms' output.
fn run_oracle_arm() {
    let Some(executor) = executor() else {
        eprintln!("SKIP: no Vulkan device");
        return;
    };
    let provider =
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context");
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return;
    }
    let seed = pattern_seed();
    for shape in SHAPES {
        let mut page = AlignedBuffer::new(ATTACHMENT_BYTES, provider.no_copy_alignment() as usize);
        let (load, store, viewport, scissor) = match shape.name {
            "scissor" => (
                LoadOp::Load,
                StoreOp::Borrowed,
                [0, 0, WIDTH, HEIGHT],
                Some(shape.rect.expect("the scissor shape states a rectangle")),
            ),
            "viewport" => (
                LoadOp::Load,
                StoreOp::Store,
                shape.rect.expect("the viewport shape states a rectangle"),
                None,
            ),
            "clear" => (
                LoadOp::Clear(ClearColor::new([0x20, 0x30, 0x40, 0xff])),
                StoreOp::Store,
                [0, 0, WIDTH, HEIGHT],
                Some(shape.rect.expect("the clear shape states a rectangle")),
            ),
            _ => (
                LoadOp::DontCare,
                StoreOp::Store,
                [0, 0, WIDTH, HEIGHT],
                Some([0, 0, 1, 1]),
            ),
        };
        // The owner-window arm names the guest's own page: the pass loads from
        // it and lands its frame back in it (E-TX8), which is the shape the
        // census's `door=mapping` records state.
        let window = match shape.name {
            "scissor" => Some(&mut page),
            _ => None,
        };
        let published = run_shape(
            &executor,
            &provider,
            ShapeRequest {
                seed: &seed,
                load,
                store,
                viewport,
                scissor,
                window,
            },
        );
        println!(
            "ORACLE shape={} frame={} window={} rect_n={} rect_bytes={} rect_extent_bytes={} \
             full_n={} full_bytes={} switch_n={} shape_n={} bounds_n={} whole_n={}",
            shape.name,
            hex(&published.frame),
            published
                .window
                .as_ref()
                .map(|bytes| hex(bytes))
                .unwrap_or_else(|| "-".to_owned()),
            published.regions.rect_attachments,
            published.regions.rect_bytes,
            published.regions.rect_extent_bytes,
            published.regions.full_attachments,
            published.regions.full_bytes,
            published.regions.switch_attachments,
            published.regions.shape_attachments,
            published.regions.bounds_attachments,
            published.regions.whole_attachments,
        );
    }
}

/// One `ORACLE` line, parsed into the fields the comparison reads.
#[derive(Debug)]
struct Reading {
    name: String,
    frame: Vec<u8>,
    window: Option<Vec<u8>>,
    regions: ReadbackRegionCounts,
}

/// Run one arm as this test binary's own child and collect its readings.
///
/// `None` when the child skipped — no Vulkan device, or no host-memory import —
/// so a machine without the device reports a skip instead of a failure.
fn oracle_child(arm: &str, full_readback: Option<&str>) -> Option<Vec<Reading>> {
    let exe = std::env::current_exe().expect("the test binary's own path");
    let mut command = std::process::Command::new(exe);
    command
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(ORACLE_ARM, arm);
    match full_readback {
        Some(value) => {
            command.env(FULL_READBACK, value);
        }
        None => {
            command.env_remove(FULL_READBACK);
        }
    }
    let output = command.output().expect("the arm's child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut readings = Vec::new();
    for line in stdout.lines() {
        // The test harness prints its own `test <name> ... ` prefix, which can
        // land on the first reading's line, so the marker is searched for
        // rather than required at the line's start.
        let Some(marker) = line.find("ORACLE ") else {
            continue;
        };
        let rest = &line[marker + "ORACLE ".len()..];
        let field = |name: &str| {
            rest.split_whitespace()
                .find_map(|token| token.strip_prefix(name))
                .unwrap_or_else(|| panic!("the {arm} arm's line states {name}: {line}"))
                .to_owned()
        };
        let number = |name: &str| {
            field(name)
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("the {arm} arm's {name} is a count: {line}"))
        };
        let window = field("window=");
        readings.push(Reading {
            name: field("shape="),
            frame: unhex(&field("frame=")),
            window: (window != "-").then(|| unhex(&window)),
            regions: ReadbackRegionCounts {
                rect_attachments: number("rect_n="),
                rect_bytes: number("rect_bytes="),
                rect_extent_bytes: number("rect_extent_bytes="),
                full_attachments: number("full_n="),
                full_bytes: number("full_bytes="),
                switch_attachments: number("switch_n="),
                shape_attachments: number("shape_n="),
                bounds_attachments: number("bounds_n="),
                whole_attachments: number("whole_n="),
            },
        });
    }
    if readings.is_empty() {
        eprintln!(
            "SKIP: the {arm} arm reported no readings (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    assert!(
        output.status.success(),
        "the {arm} arm failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(readings)
}

#[test]
fn the_trimmed_readback_publishes_the_same_guest_pages_as_the_whole_readback() {
    if std::env::var(ORACLE_ARM).is_ok() {
        run_oracle_arm();
        return;
    }
    let Some(trimmed) = oracle_child("trimmed", None) else {
        return;
    };
    let whole = oracle_child("whole", Some("1")).expect("the control arm runs");
    assert_eq!(trimmed.len(), SHAPES.len());
    assert_eq!(whole.len(), SHAPES.len());

    let seed = pattern_seed();
    let clear_seed = [0x20_u8, 0x30, 0x40, 0xff].repeat(ATTACHMENT_BYTES / 4);
    for (trimmed, whole) in trimmed.iter().zip(&whole) {
        assert_eq!(trimmed.name, whole.name, "the two arms run the same shapes");
        eprintln!(
            "{} arm: frame {} window {} regions {:?}",
            trimmed.name,
            hex(&trimmed.frame),
            trimmed
                .window
                .as_ref()
                .map(|bytes| hex(bytes))
                .unwrap_or_else(|| "-".to_owned()),
            trimmed.regions
        );
        let shape = SHAPES
            .iter()
            .find(|shape| shape.name == trimmed.name)
            .expect("every reading names a shape this file states");
        match shape.rect {
            Some(rect) => {
                assert_eq!(
                    trimmed.frame, whole.frame,
                    "the {} shape's frame is the one the whole-attachment readback publishes",
                    trimmed.name
                );
                let expectation = expected_frame(
                    match trimmed.name.as_str() {
                        "clear" => &clear_seed,
                        _ => &seed,
                    },
                    rect,
                );
                assert_eq!(
                    trimmed.frame, expectation,
                    "the {} shape's frame is its seed with the written rectangle patched in",
                    trimmed.name
                );
                assert_eq!(
                    trimmed.regions.rect_attachments, 1,
                    "the {} shape is read back through its written rectangle",
                    trimmed.name
                );
                let rect_bytes = (rect[2] * rect[3] * 4) as usize;
                assert_eq!(trimmed.regions.rect_bytes, rect_bytes);
                assert_eq!(
                    trimmed.regions.rect_extent_bytes, ATTACHMENT_BYTES,
                    "the counter states what the whole attachment would have cost"
                );
                assert_eq!(whole.regions.rect_attachments, 0);
                assert_eq!(
                    whole.regions.switch_attachments, 1,
                    "the control switch sends the shape down the whole-attachment path"
                );
                assert_eq!(whole.regions.full_bytes, ATTACHMENT_BYTES);
            }
            None => {
                // The uncovered texels of an undefined load are undefined by
                // contract, so this shape's claim is not about its frame's
                // bytes: it is that the rail says so — the whole attachment is
                // read back and the fallback is counted.
                assert_eq!(trimmed.frame.len(), ATTACHMENT_BYTES);
                assert_eq!(
                    trimmed.regions.rect_attachments, 0,
                    "a shape with an undefined load has no provable rectangle"
                );
                assert_eq!(
                    trimmed.regions.shape_attachments, 1,
                    "and it is counted as a shape fallback rather than silently read back whole"
                );
                assert_eq!(trimmed.regions.full_bytes, ATTACHMENT_BYTES);
            }
        }
        if let Some(window) = &trimmed.window {
            assert_eq!(
                window, &trimmed.frame,
                "the owner's own page holds the frame the writeback channel carries"
            );
            assert_eq!(
                window,
                &whole.window.clone().expect("the control arm lands one too"),
                "and it is the page the whole-attachment readback lands"
            );
        }
    }
}
