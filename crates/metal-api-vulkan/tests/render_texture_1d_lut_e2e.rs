//! The render sampler's one-dimensional LUT arm (2026-09-19, census b10's
//! `texture_shape` bucket): a fragment stage whose `[[texture(0)]]` is a
//! **one-dimensional array** LUT — `texture1d_array<float, sample>`, bound as
//! an `MTLTextureType1DArray` view — sampled out of its single row.
//!
//! The census's own two shapes are the widening's whole statement: a `16384x1`
//! `R32_SFLOAT` colour-transfer table and a `1024x1` `R16_SFLOAT` one, both
//! single-slice arrays whose texels the guest reads as floats
//! (`evidence/gate3-census-b10-2026-09-19/`). This test is the Vulkan rail's
//! executable half of that widening, and it is deliberately sharper than "the
//! frame changed":
//!
//! * the fragment stage samples four texel centres of its own eight-texel row,
//!   taking one whole component of each, so a flipped, transposed or
//!   row-shifted read lands another texel's number;
//! * two of the LUT's values are outside the 8-bit range — `2.5` clamps to
//!   `0xff` while its own leading byte is `0x00`, and `-0.5` clamps to `0x00`
//!   while its leading byte is `0xbf` — so a rail that read the texel's first
//!   byte as a normalised component, or quantised the LUT on the way in, lands
//!   a different frame;
//! * the same fixture runs over **both** lanes the census states (`r32_float`
//!   and `r16_float`), so the width of the texel is the bind's own fact and not
//!   a second code path;
//! * another LUT moves a sampled component, so the reading measures the upload
//!   rather than the run;
//! * the declaration's own array axis and the view's height are refusals by
//!   name beside the admitted shape: a plain `d1` declaration for the arrayed
//!   module, and a one-dimensional view with more than one row.
//!
//! The frame the rail lands is compared against the *engine's* own frame for
//! the same shape by the reims-vgpu rail (its `provider_render_rail` test,
//! `the_one_dimensional_lut_is_read_from_the_frame_and_keeps_the_old_window_by_name`),
//! which is where "the two rails agree byte for byte" is stated for this arm:
//! the engine is the rail that ran these draws before the widening.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderErrorClass, RenderAttachment, RenderPassDescriptor,
    RenderPipelineContract, ResourceTableSnapshot, SamplerAddressMode, SamplerFilter,
    SamplerPolicy, SemanticDigest, StoreOp, TextureAccess, TextureBindingContract, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{ComputeExecutor, Device};
use metal_api_vulkan::{
    RenderStage, TranslatedRenderPipelineRequest, TranslatedRenderStage, VulkanComputeProvider,
    VulkanExecutor,
};
use std::sync::Arc;

/// The reviewed milestone vertex stage, written as the AIR the translator
/// consumes: a full-screen triangle whose `vertex_id` positions cover the whole
/// attachment, and which forwards no varying at all — the fragment fixture
/// samples at constant texel centres, so it needs none.
const VERTEX_AIR: &str = include_str!("fixtures/render_offscreen_2x2.vert.ll");
const VERTEX_ENTRY: &str = "render_fullscreen_triangle";
/// The declaring compute pass's kernel, the trace's own declaration of the
/// attachment's bytes (`research/docs/23` §3.6).
const COPY_WORD_AIR: &str =
    include_str!("../../../examples/metal-smoke/shaders/kernel_copy_word.ll");

/// The one-dimensional LUT fixture (`research/docs/23` §119): four texel
/// centres of the module's own single row, one component each.
const FRAGMENT_ENTRY: &str = "render_sample_texture_1d_array_lut";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_1d_array_lut.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(960);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(961);
const SCRATCH_VIEW: ViewId = ViewId::new(962);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(963);
const TEXTURE_VIEW: ViewId = ViewId::new(964);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(965);

/// 4x4, the extent of the attachment: the render area the module's own
/// constant coordinates are independent of.
const EXTENT: u32 = 4;
/// 8, the LUT's own row — the census's `16384x1`/`1024x1` shapes one scale
/// down, on the same texel-centre grid.
const LUT_WIDTH: u32 = 8;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The sampler state the fragment fixture's AIR carries (nearest +
/// clamp-to-edge), the state the registration has to repeat.
const MODULE_SAMPLER: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// Which values the fixture's eight-texel row carries, and what the attachment
/// lands for the four the fragment stage reaches.
///
/// The four reached texels are 0, 2, 4 and 6; the values between them are fill,
/// and the two out-of-range members are the falsifiers the header states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    /// red 0.25, green 0.75, blue 2.5, alpha 0.6.
    Primary,
    /// The same row one step elsewhere: green moves from 0.75 to 0.2, which is
    /// the component the moved-LUT reading watches.
    Moved,
}

impl Pattern {
    /// The eight values the LUT's row carries, in texel order.
    fn lut(self) -> [f32; 8] {
        match self {
            Self::Primary => [0.25, 0.5, 0.75, 1.0, 2.5, -0.5, 0.6, 0.0],
            Self::Moved => [0.25, 0.5, 0.2, 1.0, 2.5, -0.5, 0.6, 0.0],
        }
    }

    /// The four values the fragment stage's four samples reach, in the order
    /// they reach it: red, green, blue, alpha.
    fn sampled(self) -> [f32; 4] {
        let lut = self.lut();
        [lut[0], lut[2], lut[4], lut[6]]
    }

    /// The frame the attachment lands: the API's own float-to-unorm conversion
    /// of each sampled value — `round(clamp(value, 0, 1) * 255)`, ties to even.
    /// Written out rather than computed so the fixture's expectation is a
    /// statement of the rule and not a second implementation of it.
    fn expected(self) -> [u8; 4] {
        match self {
            // 0.25 -> 64, 0.75 -> 191, 2.5 -> clamped 255, 0.6 -> 153.
            Self::Primary => [0x40, 0xbf, 0xff, 0x99],
            // 0.2 -> 51.
            Self::Moved => [0x40, 0x33, 0xff, 0x99],
        }
    }

    /// The LUT's tightly packed row at each lane's own texel width.
    fn bytes(self, format: TextureFormat) -> Vec<u8> {
        match format {
            TextureFormat::R32Float => self
                .lut()
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
            TextureFormat::R16Float => self
                .lut()
                .iter()
                .flat_map(|value| half_bits(*value).to_le_bytes())
                .collect(),
            other => panic!("{other:?} is not a one-dimensional float lane"),
        }
    }
}

/// The IEEE 754 binary16 encoding of one of the fixture's values, by the same
/// arithmetic the frame expectation assumes, with the *conversion* round trip
/// asserted: a half carries fewer significant bits than an `f32`, so a value
/// like `0.6` rounds to the nearest half (`0.60009766`), and what has to agree
/// with the written-out expectation is the 8-bit byte that value converts to —
/// not the value itself.
fn half_bits(value: f32) -> u16 {
    let half = half_from_f32(value);
    assert_eq!(
        to_unorm8(f32_from_half(half)),
        to_unorm8(value),
        "the fixture's values have to convert to the byte their own expectation states"
    );
    half
}

/// The API's own float-to-unorm conversion of a sampled value: the attachment
/// stores `round(clamp(value, 0, 1) * 255)`, and none of the fixture's values
/// lands on a tie.
fn to_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The binary16 encoding of `value` (round to nearest, ties to even).
fn half_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if bits & 0x7fff_ffff == 0 {
        // The fixture's fill value is a zero, which is its own encoding.
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    assert!(exponent > 0, "the fixture encodes no subnormal halves");
    assert!(exponent < 0x1f, "the fixture's values fit in binary16");
    let rounded = mantissa + 0x0fff + u32::from(mantissa & 0x1fff > 0x1000);
    let (exponent, mantissa) = if rounded >= 0x80_0000 {
        (exponent + 1, 0)
    } else {
        (exponent, rounded)
    };
    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

/// The `f32` a binary16 encoding names, by the same arithmetic
/// [`half_from_f32`] inverts.
fn f32_from_half(half: u16) -> f32 {
    let sign = if half & 0x8000 != 0 { -1.0_f32 } else { 1.0 };
    let exponent = ((half >> 10) & 0x1f) as i32;
    let mantissa = f32::from(half & 0x3ff);
    if exponent == 0 {
        return sign * mantissa * 2.0_f32.powi(-24);
    }
    assert!(exponent < 0x1f, "the fixture encodes no infinities or NaNs");
    sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(case: &[u8]) -> SemanticDigest {
    SemanticDigest::new("metal-smoke-fixture-v1", case.to_vec()).expect("digest")
}

fn executor_and_provider() -> Option<(Arc<VulkanExecutor>, Arc<VulkanComputeProvider>)> {
    let executor = match VulkanExecutor::new() {
        Ok(executor) => executor,
        Err(error) => {
            eprintln!("SKIP: no Vulkan device: {error}");
            return None;
        }
    };
    let provider = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(&executor)).expect("provider context"),
    );
    Some((executor, provider))
}

/// Translate one fragment fixture the way a host feeding guest AIR would.
fn translated_pair(
    executor: &Arc<VulkanExecutor>,
    fragment_air: &str,
    entry: &str,
) -> (TranslatedRenderStage, TranslatedRenderStage) {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let library = device
        .new_library_with_air(VERTEX_AIR)
        .expect("the vertex fixture loads");
    let function = library
        .function(VERTEX_ENTRY)
        .expect("the vertex entry exists");
    let vertex = TranslatedRenderStage::translate(RenderStage::Vertex, &function)
        .expect("the vertex stage translates");
    let library = device
        .new_library_with_air(fragment_air)
        .expect("the fragment fixture loads");
    let function = library.function(entry).expect("the fragment entry exists");
    let fragment = TranslatedRenderStage::translate(RenderStage::Fragment, &function)
        .expect("the fragment stage translates");
    eprintln!(
        "translated {entry}: {} bytes, reflection entry {:?}, shapes {:?}",
        fragment.spirv().len(),
        fragment.reflection().entry_point,
        fragment
            .reflection()
            .bindings
            .iter()
            .filter_map(|binding| binding
                .texture_shape
                .as_ref()
                .map(|shape| (shape.dimension, shape.arrayed)))
            .collect::<Vec<_>>(),
    );
    (vertex, fragment)
}

fn compile_declaring_kernel(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
) -> CompiledComputePipeline {
    let device = Device::new(Arc::clone(executor) as Arc<dyn ComputeExecutor>);
    let function = device
        .new_library_with_air(COPY_WORD_AIR)
        .expect("the fixture library loads")
        .function("copy_word")
        .expect("the fixture entry exists");
    provider
        .compile_pipeline(&function, digest(b"one-dim-lut-compute"))
        .expect("the compute pipeline registers")
}

/// The contract a one-dimensional sampling registration states: the fixture's
/// own two entries, one `rgba8_unorm` attachment, no vertex stream, and one
/// sampled texture at `[[texture(0)]]` whose format is the caller's and whose
/// type is the module's own array axis.
fn contract(format: TextureFormat, texture_type: TextureType) -> RenderPipelineContract {
    let mut sampled = TextureBindingContract::sampled(0, format, MODULE_SAMPLER);
    sampled.texture_type = texture_type;
    RenderPipelineContract {
        stage_buffers: Vec::new(),
        vertex_entry: VERTEX_ENTRY.to_owned(),
        fragment_entry: FRAGMENT_ENTRY.to_owned(),
        color_formats: vec![AttachmentFormat::Rgba8Unorm],
        vertex_layout: VertexLayout::None,
        textures: vec![sampled],
    }
}

/// One `D1Array` view over the LUT's own single row: one slice, one sample,
/// one descriptor, and the bytes the pattern states at the lane's width.
fn lut_view(format: TextureFormat, pattern: Pattern) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D1Array,
        format,
        width: u64::from(LUT_WIDTH),
        height: 1,
        depth: 1,
        array_length: 1,
        sample_count: 1,
        access: TextureAccess::Sampled,
        source: TextureSource::OwnedBytes(pattern.bytes(format)),
    }
}

fn render_pass(pipeline: PipelineId, textures: Vec<TextureView>) -> RenderPassDescriptor {
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
        pipeline,
        color_attachments: vec![RenderAttachment {
            view_id: ATTACHMENT_VIEW,
            allocation_id: ATTACHMENT_ALLOCATION,
            format: AttachmentFormat::Rgba8Unorm,
            width: u64::from(EXTENT),
            height: u64::from(EXTENT),
            load: LoadOp::Clear(ClearColor::new(CLEAR_SENTINEL)),
            store: StoreOp::Store,
        }],
        viewport: [0, 0, EXTENT, EXTENT],
        scissor: None,
        vertices: 3,
        vertex_buffers: Vec::new(),
        indices: None,
        instance_count: 1,
        textures,
        present: None,
    }
}

/// The trace the declaring compute pass and the sampling pass share
/// (`research/docs/23` §3.6): the compute pass states the attachment's own
/// bytes, so the readback is the pass's declaration rather than a driver's
/// initial contents.
fn trace_for(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    textures: Vec<TextureView>,
) -> (ComputeTrace, ResourceTableSnapshot) {
    let attachment_bytes = u64::from(EXTENT) * u64::from(EXTENT) * 4;
    let trace = ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch: provider.device_epoch(),
        operation_id: OperationId::new(61),
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
                        length: attachment_bytes,
                        access: BufferAccess::Read,
                        attribute_stride: None,
                        source: BufferSource::OwnedBytes(
                            ATTACHMENT_WORD.repeat(attachment_bytes as usize / 4),
                        ),
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
            TracePass::Render(render_pass(render.pipeline_id, textures)),
        ],
        completion_policy: CompletionPolicy::HostReadback,
        heap: None,
        indirect: None,
    };
    let mut resources = ResourceTableSnapshot::new();
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: ATTACHMENT_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: attachment_bytes,
        })
        .expect("attachment allocation");
    resources
        .insert_allocation(AllocationRecord {
            allocation_id: SCRATCH_ALLOCATION,
            owner_epoch: provider.device_epoch(),
            size: 8,
        })
        .expect("scratch allocation");
    (trace, resources)
}

/// The frame the trace rail lands for one LUT, or the refusal that stopped it
/// before the device ever ran.
fn trace_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    texture: TextureView,
) -> Result<Vec<u8>, metal_api_core::provider::ProviderError> {
    let (trace, resources) = trace_for(provider, compute, render, vec![texture]);
    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources)?;
    let submitted = provider.submit(admitted)?;
    submitted
        .validate_for_trace(&trace)
        .expect("the writebacks cover the trace");
    assert!(matches!(
        submitted.completion,
        CompletionDisposition::CompletedVisible { .. }
    ));
    Ok(submitted
        .writebacks
        .into_iter()
        .find(|writeback| writeback.view_id == ATTACHMENT_VIEW)
        .map(|writeback| writeback.bytes)
        .expect("the attachment has a writeback"))
}

/// The frame as the fixture's own definition: every fragment samples the same
/// four texels, so the whole attachment carries one colour.
fn uniform_texel(bytes: &[u8]) -> [u8; 4] {
    assert_eq!(
        bytes.len() as u64,
        u64::from(EXTENT) * u64::from(EXTENT) * 4
    );
    let texel = [bytes[0], bytes[1], bytes[2], bytes[3]];
    for chunk in bytes.chunks_exact(4) {
        assert_eq!(
            chunk,
            texel,
            "every fragment samples the same four texels, so every texel has to carry the same \
             colour: {}",
            hex(bytes)
        );
    }
    texel
}

fn register(
    provider: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    format: TextureFormat,
    texture_type: TextureType,
    what: &str,
) -> Result<CompiledComputePipeline, metal_api_core::provider::ProviderError> {
    let (vertex, fragment) = translated_pair(executor, FRAGMENT_AIR, FRAGMENT_ENTRY);
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: contract(format, texture_type),
        vertex,
        fragment,
        logical_digest: digest(what.as_bytes()),
    })
}

/// Reading 1 (2026-09-19, census b10's `texture_shape` bucket): the
/// one-dimensional bind enters the provider, the frame is the LUT's own floats'
/// conversion, and both lanes the census states run the same fixture.
#[test]
fn the_one_dimensional_lut_executes_and_the_frame_is_its_own_floats() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The frame the consumer's own class gate reads must carry both lanes and a
    // window at least as wide as this fixture's row: the widening is answered
    // from *these* fields, not from a second table.
    let capabilities = provider.capabilities();
    assert!(
        capabilities
            .supported_render_texture_formats
            .contains(&TextureFormat::R32Float)
            && capabilities
                .supported_render_texture_formats
                .contains(&TextureFormat::R16Float),
        "the provider's capability frame states both one-dimensional float lanes: {:?}",
        capabilities.supported_render_texture_formats
    );
    assert!(
        capabilities.max_render_texture_dimension_1d >= u64::from(LUT_WIDTH),
        "the provider's one-dimensional window covers the fixture's row: {}",
        capabilities.max_render_texture_dimension_1d
    );

    let compute = compile_declaring_kernel(&provider, &executor);
    // The written-out expectation is the API's own conversion of the four
    // sampled values, asserted rather than assumed.
    for pattern in [Pattern::Primary, Pattern::Moved] {
        assert_eq!(
            pattern.sampled().map(to_unorm8),
            pattern.expected(),
            "{pattern:?}: the expectation is the sampled values' own conversion"
        );
    }
    for format in [TextureFormat::R32Float, TextureFormat::R16Float] {
        let pipeline = register(
            &provider,
            &executor,
            format,
            TextureType::D1Array,
            &format!("one-dim {format:?}"),
        )
        .expect("the one-dimensional declaration registers");
        let frame = trace_readback(
            &provider,
            &compute,
            &pipeline,
            lut_view(format, Pattern::Primary),
        )
        .expect("the one-dimensional bind executes");
        let expected = Pattern::Primary.expected();
        eprintln!(
            "{format:?}: frame {} (expected {})",
            hex(&frame),
            hex(&expected)
        );
        assert_eq!(
            uniform_texel(&frame),
            expected,
            "{format:?}: {}",
            hex(&frame)
        );
        // The falsifier: 2.5's own leading byte is `0x00`, so a rail that read
        // a texel's first byte as a normalised component would land it where
        // the frame carries the clamp's `0xff`.
        let uploaded = Pattern::Primary.bytes(format);
        let texel_bytes = match format {
            TextureFormat::R32Float => 4,
            _ => 2,
        };
        assert_eq!(
            uploaded[texel_bytes * 4],
            0x00,
            "the out-of-range high value's own first byte is the falsifier"
        );
        assert_ne!(
            uniform_texel(&frame)[2],
            uploaded[texel_bytes * 4],
            "the blue lane is the float's converted value, not the texel's first byte"
        );
        assert_eq!(
            Pattern::Primary.sampled(),
            [0.25, 0.75, 2.5, 0.6],
            "the four sampled values are the ones the expectation converts"
        );
    }
}

/// Reading 2 (2026-09-19, census b10's `texture_shape` bucket): another LUT
/// moves a sampled component, so the reading is the upload and not the run.
#[test]
fn another_one_dimensional_lut_moves_the_frame() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &provider,
        &executor,
        TextureFormat::R32Float,
        TextureType::D1Array,
        "one-dim moved",
    )
    .expect("the one-dimensional declaration registers");

    let primary = trace_readback(
        &provider,
        &compute,
        &pipeline,
        lut_view(TextureFormat::R32Float, Pattern::Primary),
    )
    .expect("the primary LUT executes");
    let moved = trace_readback(
        &provider,
        &compute,
        &pipeline,
        lut_view(TextureFormat::R32Float, Pattern::Moved),
    )
    .expect("the moved LUT executes");

    let expected_moved = Pattern::Moved.expected();
    eprintln!(
        "moved frames: primary {} moved {} (expected {})",
        hex(&primary),
        hex(&moved),
        hex(&expected_moved)
    );
    assert_eq!(
        uniform_texel(&moved),
        expected_moved,
        "moved frame: {}",
        hex(&moved)
    );
    assert_ne!(
        uniform_texel(&primary),
        expected_moved,
        "the moved LUT has to move the frame: {} vs {}",
        hex(&primary),
        hex(&moved)
    );
    assert_eq!(uniform_texel(&primary)[0], 0x40, "red does not move");
    assert_ne!(uniform_texel(&primary)[1], 0x33, "green does");
}

/// Reading 3 (2026-09-19, census b10's `texture_shape` bucket): the shapes
/// beside the admitted one keep their refusals by name, and the contract's own
/// structural rule answers a one-dimensional declaration that states more than
/// one row.
#[test]
fn the_shapes_beside_the_one_dimensional_lut_keep_their_names() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };

    // A plain `d1` declaration for the arrayed module: the view the rail
    // creates is the module's own array axis, so the two have to agree.
    let refused = register(
        &provider,
        &executor,
        TextureFormat::R32Float,
        TextureType::D1,
        "plain d1 for an arrayed module",
    )
    .expect_err("the declaration has to restate the module's own array axis");
    eprintln!("plain d1 refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_shape_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(refused.fields.get("arrayed"), Some(&FieldValue::Bool(true)));

    // A two-dimensional declaration for the one-dimensional module: the same
    // door, the other direction.
    let refused = register(
        &provider,
        &executor,
        TextureFormat::R32Float,
        TextureType::D2,
        "d2 for a one-dimensional module",
    )
    .expect_err("a 2D declaration does not cover a one-dimensional module");
    eprintln!("d2 refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_shape_unsupported");

    // The structural rule: a one-dimensional view with more than one row is not
    // a shape a rail could execute, and the contract answers it before any
    // capability is asked — the view's own byte extent would be a grid of rows
    // while the image it names has one.
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &provider,
        &executor,
        TextureFormat::R32Float,
        TextureType::D1Array,
        "one-dim rows",
    )
    .expect("the one-dimensional declaration registers");
    let mut two_rows = lut_view(TextureFormat::R32Float, Pattern::Primary);
    two_rows.height = 2;
    two_rows.source = TextureSource::OwnedBytes(vec![0x00; LUT_WIDTH as usize * 2 * 4]);
    let (trace, resources) = trace_for(&provider, &compute, &pipeline, vec![two_rows]);
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a one-dimensional texture has one row");
    eprintln!("two rows refused: {refused:?}");
    assert_eq!(refused.slug, "texture_shape_mismatch");
}
