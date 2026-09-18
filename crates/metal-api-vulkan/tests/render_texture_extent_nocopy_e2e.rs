//! The gathered extent's no-copy arm (`research/docs/23` §111, E-TX12).
//!
//! The census's `texture_extent` bucket is the owner's no-copy window: the
//! engine routes those draws by arm (`texture_extent_host_bytes` /
//! `texture_extent_borrowed_no_copy`) and fail-closes the second one, because no
//! capability bit said whether a snapshot executes it. This file is the rail's
//! own reading of that arm — *without* a host copy of the owner's mapping:
//!
//! * the reviewed pair's *gathered* fragment sibling reads the owner's window
//!   in place: the image keeps the source's own extent, the descriptor carries
//!   the image alone (no `VkSampler`), and the module fetches the texel the
//!   destination grid names — `floor((2 * index + 1) * source / (2 *
//!   destination))`, computed on the device from the fragment's own framebuffer
//!   coordinate with the two extents injected as specialization constants. The
//!   frame is therefore the host-bytes gather's frame, per byte, and the two
//!   arms are compared here instead of being described as equivalent;
//! * a *translated* fragment stage states its own coordinates, so the rail binds
//!   the owner's window at the source's own extent exactly as it binds the
//!   trace's own bytes — the arm the census's shapes arrive through, because the
//!   fork registers the guest's translated stages;
//! * the reviewed pair's *sampling* sibling keeps the arm refused by name (its
//!   sampler would leave the texel choice to the driver's interpolation and
//!   filtering precision), and a registration that declares the gathered sibling
//!   over any other source is refused by name too.
//!
//! Every reading below therefore has a falsifiable frame: a rail that copied the
//! window to the host, a rail that let a sampler pick the texel, or a rail that
//! ignored the declaring module would land another frame — or none.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BorrowedLease, BufferAccess, BufferLease,
    BufferSource, BufferView, ClearColor, CompiledComputePipeline, CompletionDisposition,
    CompletionPolicy, ComputePass, ComputeProvider, ComputeTrace, Dispatch, DispatchKind,
    DispatchType, FieldValue, LeaseId, LeaseReservation, LoadOp, NoCopyLeaseImporter, OperationId,
    ProviderErrorClass, RenderAttachment, RenderPassDescriptor, RenderPipelineContract,
    ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
    StoreOp, TextureAccess, TextureBindingContract, TextureFormat, TextureSource, TextureType,
    TextureView, TracePass, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed sampling pair (`research/docs/23` §3.3, v70): the full-screen
/// triangle with its `Location 0` uv varying, and the fragment module that
/// samples the pass's texture at the fragment's own centre.
const SAMPLED_QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/sampled_quad.vert.spv");
const SAMPLED_UNORM8_FRAG_SPV: &[u8] =
    include_bytes!("../src/render_spv/solid_unorm8_sampled.frag.spv");

/// The same pair's *gathered* sibling (`research/docs/23` §111, E-TX12): the
/// module that reads the destination grid's texel by `OpImageFetch`, with the
/// two extents injected as specialization constants.
const GATHERED_FETCH_FRAG_SPV: &[u8] = include_bytes!("../src/render_spv/gathered_fetch.frag.spv");

/// The census's arm: the guest's translated stages, whose fragment module states
/// two absolute sample coordinates over the source's own extent.
const TRANSLATED_VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const TRANSLATED_VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const TRANSLATED_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");
const TRANSLATED_FRAGMENT_ENTRY: &str = "render_sample_texture_2d";

/// The declaring compute pass's kernel, the trace's own declaration of an
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(991);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(992);
const SCRATCH_VIEW: ViewId = ViewId::new(993);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(994);
const TEXTURE_VIEW: ViewId = ViewId::new(995);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(996);

const REVIEWED_LEASE: LeaseId = LeaseId::new(97);
const TRANSLATED_LEASE: LeaseId = LeaseId::new(98);

const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// Which module reads the case's source.
#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// The reviewed pair's gathered sibling: the destination grid's index,
    /// computed on the device.
    Gathered,
    /// The reviewed pair's sampling sibling: the same source through the host
    /// gather, which is what the no-copy arm's frame has to equal.
    Sampled,
    /// The guest's translated stages, the census's arm.
    Translated,
}

/// One fixture: the sampled texture's own extent, the render area's, and — for
/// the two small reviewed cases — the source texel every destination pixel
/// reads, written out by hand.
#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    arm: Arm,
    source: [u32; 2],
    destination: [u32; 2],
    table: Option<[[(u8, u8); 4]; 4]>,
}

const CASES: [Case; 3] = [
    Case {
        name: "six by four into four by four",
        arm: Arm::Gathered,
        source: [6, 4],
        destination: [4, 4],
        table: Some([
            [(0, 0), (2, 0), (3, 0), (5, 0)],
            [(0, 1), (2, 1), (3, 1), (5, 1)],
            [(0, 2), (2, 2), (3, 2), (5, 2)],
            [(0, 3), (2, 3), (3, 3), (5, 3)],
        ]),
    },
    Case {
        name: "the census scale: thirty-two by thirty-two into forty by thirty-two",
        arm: Arm::Gathered,
        source: [32, 32],
        destination: [40, 32],
        table: None,
    },
    Case {
        name: "the census arm: a translated module over six by four into four by four",
        arm: Arm::Translated,
        source: [6, 4],
        destination: [4, 4],
        table: None,
    },
];

/// The translated fixture's own reading (`research/docs/23` §111, E-TX5).
///
/// Its fragment stage samples `(1.375, 0.125)` and `(0.3125, 0.125)` with
/// nearest + clamp-to-edge over a `6x4` source whose column `x` holds `16 * x`
/// in red, so every fragment lands `50 10 00 ff`. A rail that read the owner's
/// window through the destination grid instead — the reviewed arm's rule — would
/// answer that second sample with column 2 (`32`), which is `50 20 00 ff`.
const TRANSLATED_FRAME: [u8; 4] = [0x50, 0x10, 0x00, 0xff];

fn digest(label: &str) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", label.as_bytes().to_vec()).expect("digest")
}

/// The source texels, row-major: every texel names its own position, so a frame
/// built from a grid is a byte-for-byte statement of which source texel every
/// destination pixel read.
///
/// The translated arm spaces its red channel by sixteen instead, because its
/// fixture samples two absolute coordinates and the *channels* of its output are
/// what separate the two rules.
fn texels(case: &Case) -> Vec<u8> {
    let red = |x: u32| match case.arm {
        Arm::Translated => (16 * x) as u8,
        Arm::Gathered | Arm::Sampled => x as u8,
    };
    (0..case.source[1])
        .flat_map(|y| (0..case.source[0]).flat_map(move |x| [red(x), y as u8, 0x80, 0xff]))
        .collect()
}

/// The fixture's own oracle for one axis of the destination grid — the
/// definition, not the rail's arithmetic: the source texel `i` spans
/// `[i, i + 1)`, so the texel a destination pixel's centre falls in is the count
/// of source boundaries not past it.
fn oracle_axis(destination_index: u32, source: u32, destination: u32) -> u32 {
    let centre = (2 * u128::from(destination_index) + 1) * u128::from(source);
    let mut texel = 0u32;
    for boundary in 1..u128::from(source) {
        if boundary * 2 * u128::from(destination) <= centre {
            texel = u32::try_from(boundary).expect("a boundary inside the source is a u32");
        }
    }
    texel
}

/// The frame one case's pass lands, from the case's own table when it has one
/// and from the fixture's oracle otherwise.
fn expected_frame(case: &Case) -> Vec<u8> {
    if case.arm == Arm::Translated {
        return TRANSLATED_FRAME
            .iter()
            .copied()
            .cycle()
            .take((case.destination[0] * case.destination[1] * 4) as usize)
            .collect();
    }
    let mut frame = Vec::with_capacity((case.destination[0] * case.destination[1] * 4) as usize);
    for row in 0..case.destination[1] {
        for column in 0..case.destination[0] {
            let (x, y) = match case.table {
                Some(table) => table[row as usize][column as usize],
                None => (
                    oracle_axis(column, case.source[0], case.destination[0]) as u8,
                    oracle_axis(row, case.source[1], case.destination[1]) as u8,
                ),
            };
            frame.extend_from_slice(&[x, y, 0x80, 0xff]);
        }
    }
    frame
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

/// One case's declaring kernel, compiled once per fixture.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    /// The reviewed pair's *gathered* sibling, which states the no-copy arm.
    gathered: CompiledComputePipeline,
    /// The reviewed pair's *sampling* sibling, which gathers on the host.
    sampled: CompiledComputePipeline,
    /// The guest's translated stages, the census's arm.
    translated: CompiledComputePipeline,
    compute: CompiledComputePipeline,
}

fn fixture() -> Option<Fixture> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let provider = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context"),
    );
    if provider.no_copy_alignment() == 0 {
        eprintln!("SKIP: the device does not import host memory");
        return None;
    }
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest("render_texture_nocopy_compute"))
        .expect("the compute pipeline registers");
    let contract = |entry: &str| RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: "vertex_main".to_owned(),
        fragment_entry: entry.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![TextureBindingContract::sampled(
            0,
            TextureFormat::Rgba8Unorm,
            NEAREST_CLAMP,
        )],
    };
    let registration = |label: &str, fragment: &[u8]| {
        provider
            .register_render_pipeline(RenderPipelineRequest {
                contract: contract("fragment_main"),
                vertex_spirv: SAMPLED_QUAD_VERT_SPV.to_vec(),
                fragment_spirv: fragment.to_vec(),
                logical_digest: digest(label),
            })
            .unwrap_or_else(|error| panic!("the {label} pair registers: {error:?}"))
    };
    let gathered = registration("render_texture_nocopy_gathered", GATHERED_FETCH_FRAG_SPV);
    let sampled = registration("render_texture_nocopy_sampled", SAMPLED_UNORM8_FRAG_SPV);
    // The translated arm: the same vertex AIR the other sampling fixtures use,
    // translated beside the E-RS1 fragment module whose samples are absolute.
    let translation_device = Device::new(Arc::clone(&executor) as Arc<dyn ComputeExecutor>);
    let library = translation_device
        .new_library_with_air(TRANSLATED_VERTEX_AIR)
        .expect("the translated vertex fixture loads");
    let function = library
        .function(TRANSLATED_VERTEX_ENTRY)
        .expect("the translated vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &function)
        .expect("the translated vertex stage translates");
    let library = translation_device
        .new_library_with_air(TRANSLATED_FRAGMENT_AIR)
        .expect("the translated sampling fixture loads");
    let function = library
        .function(TRANSLATED_FRAGMENT_ENTRY)
        .expect("the translated sampling entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the translated sampling stage translates");
    let translated = provider
        .register_translated_render_pipeline(TranslatedRenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: TRANSLATED_VERTEX_ENTRY.to_owned(),
                fragment_entry: TRANSLATED_FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: vec![TextureBindingContract::sampled(
                    0,
                    TextureFormat::Rgba8Unorm,
                    NEAREST_CLAMP,
                )],
            },
            vertex,
            fragment,
            logical_digest: digest("render_texture_nocopy_translated"),
        })
        .expect("the translated sampling pair registers");
    Some(Fixture {
        provider,
        gathered,
        sampled,
        translated,
        compute,
    })
}

fn pipeline_for<'a>(fixture: &'a Fixture, case: &Case) -> &'a CompiledComputePipeline {
    match case.arm {
        Arm::Gathered => &fixture.gathered,
        Arm::Sampled => &fixture.sampled,
        Arm::Translated => &fixture.translated,
    }
}

/// The sampled texture one reading declares, over one explicit source arm.
fn texture_view(case: &Case, source: TextureSource) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba8Unorm,
        width: u64::from(case.source[0]),
        height: u64::from(case.source[1]),
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source,
    }
}

fn render_pass(fixture: &Fixture, case: &Case, textures: Vec<TextureView>) -> RenderPassDescriptor {
    RenderPassDescriptor {
        samplers: Vec::new(),
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
        pipeline: pipeline_for(fixture, case).pipeline_id,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(case.destination[0]),
            height: u64::from(case.destination[1]),
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, case.destination[0], case.destination[1]],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures,
        present: None,
    }
}

/// The trace one reading submits: the declaring compute pass (which is what
/// makes the attachment view a view of this trace) and the sampling pass.
fn trace_for(
    fixture: &Fixture,
    case: &Case,
    texture: TextureView,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(case.destination[0]) * u64::from(case.destination[1]) * 4;
    let texture_bytes = u64::from(case.source[0]) * u64::from(case.source[1]) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: fixture.provider.device_epoch(),
        operation_id: OperationId::new(61),
        pipelines: vec![fixture.compute.clone(), pipeline_for(fixture, case).clone()],
        encoder_dispatch_type: DispatchType::Serial,
        passes: vec![
            TracePass::Compute(ComputePass {
                pipeline: fixture.compute.pipeline_id,
                buffers: vec![
                    BufferView {
                        view_id: ATTACHMENT_VIEW,
                        metal_binding: 0,
                        allocation_id: ATTACHMENT_ALLOCATION,
                        offset: 0,
                        length: attachment_bytes,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(vec![0; attachment_bytes as usize]),
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
            TracePass::Render(render_pass(fixture, case, vec![texture])),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, attachment_bytes),
        (SCRATCH_ALLOCATION, 8),
        (TEXTURE_ALLOCATION, texture_bytes),
    ] {
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: fixture.provider.device_epoch(),
                size,
            })
            .expect("fixture allocation");
    }
    (trace, resources)
}

/// Submit one trace and read the attachment's frame back.
fn submit_frame(
    fixture: &Fixture,
    trace: &ComputeTrace,
    resources: ResourceTableSnapshot,
) -> Vec<u8> {
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the sampled source is admitted");
    let submitted = fixture
        .provider
        .submit(admitted)
        .expect("the submission completes");
    submitted
        .validate_for_trace(trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback")
}

/// The frame the trace rail lands for one case over the trace's own bytes.
fn owned_frame(fixture: &Fixture, case: &Case) -> Vec<u8> {
    owned_frame_with_source(fixture, case, texels(case))
}

/// The same reading over one explicit source byte grid.
fn owned_frame_with_source(fixture: &Fixture, case: &Case, source: Vec<u8>) -> Vec<u8> {
    let (trace, resources) = trace_for(
        fixture,
        case,
        texture_view(case, TextureSource::OwnedBytes(source)),
    );
    submit_frame(fixture, &trace, resources)
}

/// The frame the trace rail lands for one case out of the *owner's own
/// mapping*: the same bytes, in a window the provider imports instead of
/// copying, with the lease released after the submission.
fn window_frame(fixture: &Fixture, case: &Case, lease: LeaseId, source: Vec<u8>) -> Vec<u8> {
    let provider = &fixture.provider;
    let alignment = provider.no_copy_alignment();
    assert!(
        alignment != 0,
        "the fixture refuses a device without host import before any reading"
    );
    let mut window = AlignedBuffer::new(source.len(), alignment as usize);
    window.as_mut_slice().copy_from_slice(&source);
    let reservation = LeaseReservation {
        lease: BufferLease {
            lease_id: lease,
            allocation_id: TEXTURE_ALLOCATION,
            owner_epoch: provider.device_epoch(),
        },
        offset: 0,
        length: source.len() as u64,
    };
    // Safety: the window outlives the submission and the provider's release.
    unsafe {
        provider
            .import_borrowed_lease(
                BorrowedLease::new(reservation, window.as_mut_slice().as_ptr() as usize)
                    .expect("the owner's window is a valid reservation"),
            )
            .expect("the provider imports the owner's window");
    }
    let (trace, mut resources) = trace_for(
        fixture,
        case,
        texture_view(case, TextureSource::BorrowedNoCopy(lease)),
    );
    resources
        .insert_lease(reservation)
        .expect("the window's reservation covers it");
    let frame = submit_frame(fixture, &trace, resources);
    provider
        .release_borrowed_lease(lease)
        .expect("no retain is outstanding after the submission");
    frame
}

/// The gathered sibling lands the host gather's frame, per byte, out of the
/// owner's own mapping.
///
/// This is the arm's whole claim (`research/docs/23` §111, E-TX12): the reviewed
/// pair's second fragment module reads the destination grid's texel with the
/// index the rail's own integer definition states, on the device, without a host
/// copy of the window and without a sampler deciding anything. The expectation
/// is the fixture's own table (the two small cases) or its definitional oracle
/// (the census scale), and the *sampling* sibling's frame for the same bytes is
/// the second reading — the two arms must agree byte for byte, or one of them is
/// not the destination grid.
#[test]
fn the_gathered_sibling_lands_the_host_gather_frame_from_the_owners_window() {
    let Some(fixture) = fixture() else {
        return;
    };
    for case in CASES.iter().filter(|case| case.arm != Arm::Translated) {
        let bytes = texels(case);
        let expected = expected_frame(case);
        let sampled_case = Case {
            arm: Arm::Sampled,
            ..*case
        };
        let host = owned_frame(&fixture, &sampled_case);
        assert_eq!(
            host, expected,
            "{}: the host gather is the destination grid the fixture states",
            case.name
        );
        let window = window_frame(&fixture, case, REVIEWED_LEASE, bytes.clone());
        assert_eq!(
            window, expected,
            "{}: the owner's window is read at the destination grid's index",
            case.name
        );
        assert_eq!(
            window, host,
            "{}: the two arms land the same frame byte for byte",
            case.name
        );
        eprintln!(
            "no-copy gathered extent: {} landed {} bytes, equal to the host gather",
            case.name,
            window.len()
        );
    }
}

/// A translated module reads the owner's window at the source's own extent.
///
/// The census's shapes arrive through this arm (the fork registers the guest's
/// translated stages), so the bit's statement has to be true for it as well: the
/// module states two absolute sample coordinates, the rail binds the window at
/// the source's own extent, and the frame is the module's own reading — not the
/// destination grid's.
#[test]
fn a_translated_module_reads_the_owners_window_at_its_own_extent() {
    let Some(fixture) = fixture() else {
        return;
    };
    let case = CASES
        .iter()
        .find(|case| case.arm == Arm::Translated)
        .expect("the translated case is in the table");
    let frame = window_frame(&fixture, case, TRANSLATED_LEASE, texels(case));
    let expected = expected_frame(case);
    assert_eq!(
        frame, expected,
        "the translated module's own samples decide the frame"
    );
    let mut gathered = expected;
    // The destination grid would answer the second sample with source column 2
    // (`32`), which is what separates the two rules.
    gathered[0] = 0x50;
    gathered[1] = 0x20;
    assert_ne!(
        frame, gathered,
        "the frame is not a destination-grid gather"
    );
    eprintln!("no-copy translated extent: landed {} bytes", frame.len());
}

/// The frame is a function of the owner's bytes: a source texel the grid reads
/// moves it, and one no destination pixel reads leaves it alone.
#[test]
fn a_source_texel_the_grid_reads_moves_the_owners_window_frame() {
    let Some(fixture) = fixture() else {
        return;
    };
    let case = &CASES[0];
    let baseline = window_frame(&fixture, case, REVIEWED_LEASE, texels(case));
    let texel = |x: u32, y: u32| ((y * case.source[0] + x) * 4) as usize;
    // Column 1 is read by no destination pixel of the `6x4 into 4x4` grid (its
    // columns are 0, 2, 3 and 5), so replacing it must leave the frame alone.
    let mut unread = texels(case);
    unread[texel(1, 0)] = 0xee;
    assert_eq!(
        window_frame(&fixture, case, REVIEWED_LEASE, unread),
        baseline,
        "a source texel the grid does not name cannot move the frame"
    );
    // Column 2 is read by destination column 1, so replacing one texel of it
    // must move exactly the destination pixel that names that texel.
    let mut read = texels(case);
    read[texel(2, 1)] = 0xee;
    let moved = window_frame(&fixture, case, REVIEWED_LEASE, read);
    assert_ne!(moved, baseline, "a texel the grid names moves the frame");
    for row in 0..case.destination[1] as usize {
        for column in 0..case.destination[0] as usize {
            let at = (row * case.destination[0] as usize + column) * 4;
            let expected = if (column, row) == (1, 1) {
                0xee
            } else {
                baseline[at]
            };
            assert_eq!(
                moved[at], expected,
                "destination pixel ({column}, {row}) reads the source texel the table names"
            );
            assert_eq!(
                moved[at + 1..at + 4],
                baseline[at + 1..at + 4],
                "the other channels of that pixel are the source texel's own row, blue and alpha"
            );
        }
    }
    eprintln!("no-copy source bytes: unread texel leaves the frame, read texel moves it");
}

/// The arm is not a licence to read any source through the gathered module: a
/// registration that declares the gathered sibling over the trace's own bytes
/// (or an equal extent) is refused by name, so the index arithmetic is only ever
/// executed for the shape it was reviewed for.
#[test]
fn the_gathered_sibling_refuses_a_source_that_is_not_the_owners_window() {
    let Some(fixture) = fixture() else {
        return;
    };
    let case = &CASES[0];
    let (trace, resources) = trace_for(
        &fixture,
        case,
        texture_view(case, TextureSource::OwnedBytes(texels(case))),
    );
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the snapshot's admission holds the format and the count, not the arm");
    let refused = match fixture.provider.submit(admitted) {
        Err(error) => error,
        Ok(_) => panic!("the gathered sibling states one arm and only one"),
    };
    eprintln!("gathered sibling over the trace's own bytes: {refused:?}");
    assert_eq!(
        refused.slug,
        "render_texture_gathered_fetch_arm_unsupported"
    );
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
    assert_eq!(
        refused.fields.get("source"),
        Some(&FieldValue::Text("owned_bytes".to_owned()))
    );
    assert_eq!(refused.fields.get("width"), Some(&FieldValue::Unsigned(6)));
    assert_eq!(refused.fields.get("height"), Some(&FieldValue::Unsigned(4)));
}

/// The *sampling* sibling keeps the owner's window of another extent refused by
/// name, with the fields the pre-E-TX12 window published: the pair's second
/// module is the statement of the arm, not a replacement for it.
#[test]
fn the_sampling_sibling_keeps_the_owners_window_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    let case = Case {
        arm: Arm::Sampled,
        ..CASES[0]
    };
    let (trace, resources) = trace_for(
        &fixture,
        &case,
        texture_view(&case, TextureSource::BorrowedNoCopy(REVIEWED_LEASE)),
    );
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the snapshot's admission holds the format and the count, not the extent");
    let refused = match fixture.provider.submit(admitted) {
        Err(error) => error,
        Ok(_) => panic!("the sampling sibling cannot read the owner's no-copy window"),
    };
    eprintln!("sampling sibling over the owner's window: {refused:?}");
    assert_eq!(refused.slug, "render_texture_extent_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("source"),
        Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
    );
    assert_eq!(
        refused.fields.get("render_width"),
        Some(&FieldValue::Unsigned(4))
    );
}
