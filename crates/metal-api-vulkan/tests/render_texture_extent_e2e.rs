//! The render sampler's destination grid (`research/docs/23` §111, E-TX5).
//!
//! The census v18's `texture_extent` bucket is this file's shape: 640 refusals
//! (7.8% of the first-failure buckets, the fifth-largest name) where a draw
//! samples a texture whose extent is not the pass's own. The two biggest are
//! `64x64` in an `80x64` pass and `160x64` in a `186x100` pass, and the fork's
//! class gate answers the whole family by name today because the provider did
//! (`render_texture_extent_unsupported`).
//!
//! The widened window has two arms, one per kind of module (`research/docs/23`
//! §111):
//!
//! * the *reviewed* pair leaves its sample coordinate implicit — the fragment's
//!   own centre — so the rail resolves that centre's texel itself, gathers the
//!   source into a render-area-sized surface, and the module reads that surface
//!   identity-wise. No sampler decides a texel, so no driver's filtering
//!   precision and no texel-boundary rounding enters the bytes;
//! * a *translated* module — the arm the census's shapes actually arrive
//!   through, because the fork registers the guest's translated stages —
//!   states its own sample coordinates, so the rail binds the source at its own
//!   extent and the module reads it exactly as the engine's copy of that module
//!   does.
//!
//! The readings are:
//!
//! * two extents the milestone never admitted — a `6x4` source in a `4x4` pass
//!   and a `2x8` source in a `4x4` pass — **enter the provider** and land the
//!   frame the fixture's own hand-written grid tables name;
//! * the census's own scale (`32x32` in a `40x32` pass, the `0.8` ratio of
//!   `64x64` in `80x64`) lands the frame the fixture's definitional oracle
//!   names, including the destination columns whose centres sit exactly on a
//!   source boundary;
//! * the census's own arm — a **translated** module over a `6x4` source in a
//!   `4x4` pass — lands the frame its two absolute samples read out of the
//!   source's own extent, which is not the destination grid's answer;
//! * the same frames come out of the **object rail** byte for byte;
//! * the one source the rail cannot gather — the owner's no-copy window — is
//!   still refused **by name**, with the fields the pre-widening refusal
//!   published.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LeaseId,
    LoadOp, OperationId, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SamplerAddressMode, SamplerFilter,
    SamplerPolicy, SemanticDigest, StoreOp, TextureAccess, TextureBindingContract, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage,
    VulkanComputeProvider, VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed sampling pair (`research/docs/23` §3.3, v70): the full-screen
/// triangle with its `Location 0` uv varying, and the fragment module that
/// samples `DescriptorSet 0 / Binding 0` there with the pair's one state
/// (nearest, clamp-to-edge).
const SAMPLED_QUAD_VERT_SPV: &[u8] = include_bytes!("../src/render_spv/sampled_quad.vert.spv");
const SAMPLED_UNORM8_FRAG_SPV: &[u8] =
    include_bytes!("../src/render_spv/solid_unorm8_sampled.frag.spv");

/// The translated arm's pair: the milestone vertex AIR and the reviewed E-RS1
/// sampling AIR, whose fragment stage reads two absolute coordinates with
/// nearest + clamp-to-edge. The fork registers exactly this kind of pair — the
/// guest's own stages, translated — so this is the arm the census's
/// `texture_extent` shapes reach the provider through.
const TRANSLATED_VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const TRANSLATED_VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const TRANSLATED_FRAGMENT_AIR: &str =
    include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");
const TRANSLATED_FRAGMENT_ENTRY: &str = "render_sample_texture_2d";

/// The declaring compute pass's kernel, the trace's own declaration of an
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(981);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(982);
const SCRATCH_VIEW: ViewId = ViewId::new(983);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(984);
const TEXTURE_VIEW: ViewId = ViewId::new(985);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(986);

const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];

const NEAREST_CLAMP: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest")
}

/// Which module reads the case's source. The two arms answer the extent
/// question differently, and both are measured.
#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// The rail's own reviewed pair: samples at the fragment's centre, so the
    /// rail resolves the destination grid itself.
    Reviewed,
    /// The guest's translated stages, the census's arm: the module states its
    /// own coordinates, so the source is bound at its own extent.
    Translated,
}

/// One fixture: the sampled texture's own extent and the render area's, in
/// texels. The two small reviewed cases carry hand-written grids; the
/// census-scale reviewed case carries the fixture's definitional oracle; the
/// translated case carries the module's own two samples.
struct Case {
    name: &'static str,
    arm: Arm,
    source: [u32; 2],
    destination: [u32; 2],
    /// The reading for the two small cases: the source texel `(x, y)` every
    /// destination pixel reads, in the destination's own order, written out by
    /// hand so the expectation is not a restatement of any formula.
    table: Option<[[(u8, u8); 4]; 4]>,
}

const CASES: [Case; 4] = [
    Case {
        name: "six by four into four by four",
        arm: Arm::Reviewed,
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
        name: "two by eight into four by four",
        arm: Arm::Reviewed,
        source: [2, 8],
        destination: [4, 4],
        table: Some([
            [(0, 1), (0, 1), (1, 1), (1, 1)],
            [(0, 3), (0, 3), (1, 3), (1, 3)],
            [(0, 5), (0, 5), (1, 5), (1, 5)],
            [(0, 7), (0, 7), (1, 7), (1, 7)],
        ]),
    },
    Case {
        name: "the census scale: thirty-two by thirty-two into forty by thirty-two",
        arm: Arm::Reviewed,
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

/// The source texels, row-major: every texel names its own position, so a frame
/// built from a grid is a byte-for-byte statement of which source texel every
/// destination pixel read. No texel can be mistaken for the clear sentinel, and
/// no two texels are equal.
///
/// The translated arm spaces its red channel by sixteen instead, because its
/// fixture samples two absolute coordinates and the *channels* of its output
/// are what separate the two rules: the second sample reads a source column
/// this arm keeps (1) and the destination grid would move (2), and 16 vs 32 is
/// a reading either rule can be checked against by eye.
fn texels(case: &Case) -> Vec<u8> {
    let red = |x: u32| match case.arm {
        Arm::Reviewed => x as u8,
        Arm::Translated => (16 * x) as u8,
    };
    (0..case.source[1])
        .flat_map(|y| (0..case.source[0]).flat_map(move |x| [red(x), y as u8, 0x80, 0xff]))
        .collect()
}

/// The translated fixture's own reading (`research/docs/23` §111, E-TX5).
///
/// Its fragment stage samples `(1.375, 0.125)` and `(0.3125, 0.125)` with
/// nearest + clamp-to-edge. Over a `6x4` source whose column `x` holds `16 * x`
/// in red, the first sample clamps to column 5 (`80`) and the second stands in
/// column 1 (`16`), so every fragment lands `50 10 00 ff`. A rail that gathered
/// the source into the destination grid would answer the second sample with the
/// *gathered* column 1, which is source column 2 (`32`) — `50 20 00 ff` — so
/// this frame separates the two arms' rules instead of restating either.
const TRANSLATED_FRAME: [u8; 4] = [0x50, 0x10, 0x00, 0xff];

/// The fixture's own oracle for one axis of the destination grid — the
/// definition, not the rail's arithmetic.
///
/// Source texel `i` spans `[i, i + 1)` in texel units, so the texel a
/// destination pixel's centre, `(2c + 1) / (2 * destination_width)`, falls in is
/// the *count of source boundaries not past it*. Counting by
/// cross-multiplying the two fractions keeps this derivation independent of the
/// rail's own division (and of any float), which is what makes the census-scale
/// reading falsifiable rather than circular.
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

/// The frame one case's pass lands: the source texel every destination pixel
/// reads, from the case's own table when it has one and from the fixture's
/// oracle otherwise.
fn expected_frame(case: &Case) -> Vec<u8> {
    let mut frame = Vec::with_capacity((case.destination[0] * case.destination[1] * 4) as usize);
    match (case.arm, case.table) {
        (Arm::Translated, _) => {
            // The module samples the same two absolute coordinates for every
            // fragment, so its frame is uniform.
            for _ in 0..case.destination[0] * case.destination[1] {
                frame.extend_from_slice(&TRANSLATED_FRAME);
            }
        }
        (Arm::Reviewed, Some(grid)) => {
            for row in grid {
                for (x, y) in row {
                    frame.extend_from_slice(&[x, y, 0x80, 0xff]);
                }
            }
        }
        (Arm::Reviewed, None) => {
            for row in 0..case.destination[1] {
                let y = oracle_axis(row, case.source[1], case.destination[1]) as u8;
                for column in 0..case.destination[0] {
                    let x = oracle_axis(column, case.source[0], case.destination[0]) as u8;
                    frame.extend_from_slice(&[x, y, 0x80, 0xff]);
                }
            }
        }
    }
    frame
}

/// The context the readings run in: one Lavapipe device, the reviewed sampling
/// pair registered once, and the declaring kernel the attachment's own
/// declaration runs.
struct Fixture {
    provider: Arc<VulkanComputeProvider>,
    render: CompiledComputePipeline,
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
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    let compute = provider
        .compile_pipeline(&function, digest(b"render_texture_extent_compute"))
        .expect("the compute pipeline registers");
    let render = provider
        .register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: "vertex_main".to_owned(),
                fragment_entry: "fragment_main".to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: vec![TextureBindingContract::sampled(
                    0,
                    TextureFormat::Rgba8Unorm,
                    NEAREST_CLAMP,
                )],
            },
            vertex_spirv: SAMPLED_QUAD_VERT_SPV.to_vec(),
            fragment_spirv: SAMPLED_UNORM8_FRAG_SPV.to_vec(),
            logical_digest: digest(b"render_texture_extent_stages"),
        })
        .expect("the sampled pair registers");
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
            logical_digest: digest(b"render_texture_extent_translated"),
        })
        .expect("the translated sampling pair registers");
    Some(Fixture {
        provider,
        render,
        translated,
        compute,
    })
}

/// The sampled texture one case declares: its own extent, its own bytes.
fn texture_view(case: &Case) -> TextureView {
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
        source: TextureSource::OwnedBytes(texels(case)),
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
        pipeline: match case.arm {
            Arm::Reviewed => fixture.render.pipeline_id,
            Arm::Translated => fixture.translated.pipeline_id,
        },
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

/// The trace one case submits: the declaring compute pass (which is what makes
/// the attachment view a view of this trace) and the sampling pass.
fn trace_for(
    fixture: &Fixture,
    case: &Case,
    textures: Vec<TextureView>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(case.destination[0]) * u64::from(case.destination[1]) * 4;
    let sampled = match case.arm {
        Arm::Reviewed => &fixture.render,
        Arm::Translated => &fixture.translated,
    };
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: fixture.provider.device_epoch(),
        operation_id: OperationId::new(51),
        pipelines: vec![fixture.compute.clone(), sampled.clone()],
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
            TracePass::Render(render_pass(fixture, case, textures)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    for (allocation, size) in [
        (ATTACHMENT_ALLOCATION, attachment_bytes),
        (SCRATCH_ALLOCATION, 8),
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

/// The trace rail's frame for one case.
fn trace_frame(fixture: &Fixture, case: &Case) -> Vec<u8> {
    let (trace, resources) = trace_for(fixture, case, vec![texture_view(case)]);
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("a sampled source of another extent is admitted");
    let submitted = fixture
        .provider
        .submit(admitted)
        .expect("the submission completes");
    submitted
        .validate_for_trace(&trace)
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

/// The object rail's frame for the same case: one texture object of the source's
/// own extent, one attachment of the render area's, and the same sampling
/// pipeline wrapped for the object API.
fn object_frame(fixture: &Fixture, case: &Case) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(&fixture.provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(match case.arm {
            Arm::Reviewed => &fixture.render,
            Arm::Translated => &fixture.translated,
        })
        .expect("the registration wraps for the object API");
    let attachment_bytes = u64::from(case.destination[0]) * u64::from(case.destination[1]) * 4;
    let attachment = device
        .new_buffer_with_bytes(vec![0x00; attachment_bytes as usize])
        .expect("the attachment's landing buffer is declared");
    let attachment_view = attachment
        .view(0, attachment_bytes as usize)
        .expect("the attachment view is declared");
    let texture = device
        .new_texture_with_bytes(
            TextureFormat::Rgba8Unorm,
            u64::from(case.source[0]),
            u64::from(case.source[1]),
            texels(case),
        )
        .expect("the sampled texture is declared at its own extent");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"render_texture_extent_object_declaring"),
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
            .expect("the attachment's bytes are the pass's own declaration");
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
            .expect("the sampling pipeline is bound");
        encoder
            .set_fragment_texture(0, &texture)
            .expect("the texture is bound at its own extent");
        encoder
            .draw_render_pass(
                &attachment_view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(case.destination[0]),
                u64::from(case.destination[1]),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the sampling pass records");
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

/// The two new extents, the census's own scale, and the two rails: every case's
/// trace frame is the fixture's expectation and every case's object frame is
/// the trace's, byte for byte.
#[test]
fn a_sampled_texture_of_another_extent_enters_the_provider() {
    let Some(fixture) = fixture() else {
        return;
    };
    for case in &CASES {
        let expected = expected_frame(case);
        let trace = trace_frame(&fixture, case);
        eprintln!(
            "{}: {} texels read from {}x{} into {}x{}, frame {}",
            case.name,
            case.destination[0] * case.destination[1],
            case.source[0],
            case.source[1],
            case.destination[0],
            case.destination[1],
            hex(&trace)
        );
        assert_eq!(
            trace, expected,
            "{}: the gathered frame has to be the destination grid's own reading",
            case.name
        );
        assert!(
            !trace.chunks_exact(4).any(|texel| texel == CLEAR_SENTINEL),
            "{}: a surviving clear sentinel means the sampled pass did not cover every texel",
            case.name
        );
        let object = object_frame(&fixture, case);
        assert_eq!(
            object, trace,
            "{}: the object rail lands the trace rail's frame, byte for byte",
            case.name
        );
    }
}

/// The one source the widened window cannot gather keeps its name: the owner's
/// no-copy window is read by the *device* from the owner's mapping, so a source
/// of another extent would need a host copy the arm does not carry. The refusal
/// arrives before any device object exists — and before the lease channel is
/// even asked — with the fields the pre-widening refusal published, so a capture
/// can still read which shape stood on the boundary.
#[test]
fn a_no_copy_sampled_texture_of_another_extent_is_refused_by_name() {
    let Some(fixture) = fixture() else {
        return;
    };
    let case = &CASES[0];
    let mut view = texture_view(case);
    view.source = TextureSource::BorrowedNoCopy(LeaseId::new(9));
    let (trace, resources) = trace_for(&fixture, case, vec![view]);
    let admitted = fixture
        .provider
        .capabilities()
        .validate_trace(trace.clone(), resources)
        .expect("the snapshot's admission holds the format and the count, not the extent");
    let refused = match fixture.provider.submit(admitted) {
        Err(error) => error,
        Ok(_) => panic!("the no-copy window of another extent cannot be gathered"),
    };
    eprintln!("no-copy sampled texture: {refused:?}");
    assert_eq!(refused.slug, "render_texture_extent_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("binding"),
        Some(&FieldValue::Unsigned(0))
    );
    assert_eq!(refused.fields.get("width"), Some(&FieldValue::Unsigned(6)));
    assert_eq!(refused.fields.get("height"), Some(&FieldValue::Unsigned(4)));
    assert_eq!(
        refused.fields.get("render_width"),
        Some(&FieldValue::Unsigned(4))
    );
    assert_eq!(
        refused.fields.get("render_height"),
        Some(&FieldValue::Unsigned(4))
    );
    assert_eq!(
        refused.fields.get("source"),
        Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
    );
    let detail = refused
        .detail
        .as_deref()
        .expect("the refusal carries its own sentence");
    assert!(
        detail.contains("no-copy window"),
        "the refusal names the arm it cannot read: {detail}"
    );
}
