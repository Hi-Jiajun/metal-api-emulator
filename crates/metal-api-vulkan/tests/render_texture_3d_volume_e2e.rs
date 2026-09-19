//! The render sampler's three-dimensional volume arm (2026-09-20): a fragment
//! stage whose `[[texture(0)]]` is a **volume** — `texture3d<float, sample>`,
//! bound as an `MTLTextureType3D` view — sampled at four texel centres of its
//! own `4 x 4 x 2` grid.
//!
//! The arm exists because the census's remaining sampled draws declare it: the
//! wallpaper/layer low-pass-filter family binds three `D3` textures per draw
//! and samples them through their own `float3` coordinates, so the canonical
//! provider has to be able to create, upload and bind a volume before any of
//! those draws can leave the engine. This test is the Vulkan rail's executable
//! half of that widening, and it is deliberately sharper than "the frame
//! changed":
//!
//! * two of the four sampled texels live in the volume's *second* slice, so a
//!   rail that uploaded one slice, or that read the third coordinate as an
//!   array layer of a different image, lands the fill value in those lanes;
//! * the four positions differ in x and y as well, so a transposed or
//!   row-shifted read lands another texel's number;
//! * one value is outside the 8-bit range — `2.5` clamps to `0xff` while its
//!   own leading byte is `0x00` — so a rail that read a texel's first byte as a
//!   normalised component lands a different frame; the three in-range values
//!   have the same property (`0x40`, `0xbf` and `0xdf` against a leading
//!   `0x00`);
//! * another volume moves a sampled component, so the reading measures the
//!   upload rather than the run;
//! * the trace rail and the object rail land that frame byte for byte, which is
//!   where "the two lanes agree" is stated for this arm;
//! * the shapes beside the admitted one keep their refusals by name: a `D2`
//!   declaration for the volume module, a `D3` declaration for a
//!   two-dimensional module, and a volume wider than the contract's review
//!   ceiling.

use metal_api_core::provider::{
    AllocationId, AllocationRecord, AttachmentFormat, BufferAccess, BufferSource, BufferView,
    ClearColor, CompiledComputePipeline, CompletionDisposition, CompletionPolicy, ComputePass,
    ComputeProvider, ComputeTrace, Dispatch, DispatchKind, DispatchType, FieldValue, LoadOp,
    OperationId, PipelineId, ProviderError, ProviderErrorClass, RenderAttachment,
    RenderPassDescriptor, RenderPipelineContract, ResourceTableSnapshot, SamplerAddressMode,
    SamplerFilter, SamplerPolicy, SemanticDigest, StoreOp, TextureAccess, TextureBindingContract,
    TextureFormat, TextureSource, TextureType, TextureView, TracePass, VertexLayout, ViewId,
    PROVIDER_SCHEMA_VERSION,
};
use metal_api_core::{provider_api as objects, ComputeExecutor, Device};
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
/// The volume fixture: four texel centres of the module's own `4 x 4 x 2` grid.
const FRAGMENT_ENTRY: &str = "render_sample_texture_3d_volume";
const FRAGMENT_AIR: &str = include_str!("fixtures/render_sample_texture_3d_volume.frag.ll");
/// The two-dimensional sibling the shape refusals are measured against: the
/// reviewed sampler's nearest/clamp module.
const TWO_DIM_ENTRY: &str = "render_sample_texture_2d";
const TWO_DIM_AIR: &str = include_str!("fixtures/render_sample_texture_2d_nearest_clamp.frag.ll");

const ATTACHMENT_VIEW: ViewId = ViewId::new(970);
const ATTACHMENT_ALLOCATION: AllocationId = AllocationId::new(971);
const SCRATCH_VIEW: ViewId = ViewId::new(972);
const SCRATCH_ALLOCATION: AllocationId = AllocationId::new(973);
const TEXTURE_VIEW: ViewId = ViewId::new(974);
const TEXTURE_ALLOCATION: AllocationId = AllocationId::new(975);

/// 4x4, the extent of the attachment: the render area whose fragments all
/// sample the same four volume texels.
const EXTENT: u32 = 4;
/// The volume's own three extents: the census's `D3` sources one scale down.
const VOLUME_WIDTH: u64 = 4;
const VOLUME_HEIGHT: u64 = 4;
const VOLUME_DEPTH: u64 = 2;
const CLEAR_SENTINEL: [u8; 4] = [0xfe, 0xfe, 0xfe, 0xfe];
const ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

/// The sampler state the fragment fixture's AIR carries (linear +
/// clamp-to-edge), the state the registration has to repeat.
const MODULE_SAMPLER: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Linear,
    address: SamplerAddressMode::ClampToEdge,
};

/// The state the two-dimensional sibling fixture's own AIR carries (nearest +
/// clamp-to-edge): the declaration has to repeat *that* module's state, or the
/// refusal under test would be the sampler's rather than the shape's.
const TWO_DIM_SAMPLER: SamplerPolicy = SamplerPolicy {
    filter: SamplerFilter::Nearest,
    address: SamplerAddressMode::ClampToEdge,
};

/// The texel every unsampled position of the volume carries: `2^-10`, whose
/// 8-bit conversion is `0x00` — distinct from every sampled expectation, so a
/// rail that reads the wrong texel lands a different frame.
const FILL: f32 = 0.000_976_562_5;
/// The fill value's own bytes, asserted by the construction below.
const FILL_UNORM: u8 = 0x00;

/// The texel every unsampled position of the **eight-bit** volume carries, in
/// the lane's own `blue, green, red, alpha` memory order.
///
/// A `B8G8R8A8_UNORM` image's *red* channel is the third byte, and the
/// fixture's samples take `.x` — the red channel — so the byte the frame lands
/// is that third one: `0x00` here, which is none of the four bytes the four
/// sampled texels carry. A rail that uploaded the source at another texel width
/// (one row per four bytes rather than per sixteen, say) lands this fill where
/// the frame carries a texel, and a rail that named the image `R8G8B8A8_UNORM`
/// — the other four-byte order — hands the module the *first* byte, `0x11`,
/// which no lane of the expectation holds either.
const EIGHT_BIT_FILL: [u8; 4] = [0x11, 0x22, 0x00, 0x33];

/// Which values the volume's two slices carry, and what the attachment lands
/// for the four texels the fragment stage reaches.
///
/// The four reached texels are slice 0's `(0, 0)` and `(2, 0)` and slice 1's
/// `(0, 2)` and `(2, 2)`; every other position is [`FILL`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    /// red 0.25, green 0.75, blue 2.5 (clamped), alpha 0.875.
    Primary,
    /// The same volume one step elsewhere: alpha moves from 0.875 to 0.125,
    /// which is the component the moved-volume reading watches.
    Moved,
}

impl Pattern {
    /// The `width x height x depth` texels of the volume, slice by slice and
    /// row by row inside each slice.
    fn volume(self) -> [f32; (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH) as usize] {
        let mut texels = [FILL; (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH) as usize];
        let position = |slice: u64, row: u64, column: u64| -> usize {
            (slice * VOLUME_HEIGHT * VOLUME_WIDTH + row * VOLUME_WIDTH + column) as usize
        };
        texels[position(0, 0, 0)] = 0.25;
        texels[position(0, 2, 0)] = 2.5;
        texels[position(1, 0, 2)] = 0.75;
        texels[position(1, 2, 2)] = match self {
            Self::Primary => 0.875,
            Self::Moved => 0.125,
        };
        texels
    }

    /// The four values the fragment stage's four samples reach, in the order
    /// they reach it: red, green, blue, alpha.
    fn sampled(self) -> [f32; 4] {
        let volume = self.volume();
        let position = |slice: u64, row: u64, column: u64| -> usize {
            (slice * VOLUME_HEIGHT * VOLUME_WIDTH + row * VOLUME_WIDTH + column) as usize
        };
        [
            volume[position(0, 0, 0)],
            volume[position(1, 0, 2)],
            volume[position(0, 2, 0)],
            volume[position(1, 2, 2)],
        ]
    }

    /// The frame the attachment lands: the API's own float-to-unorm conversion
    /// of each sampled value — `round(clamp(value, 0, 1) * 255)`, ties to even.
    /// Written out rather than computed so the fixture's expectation is a
    /// statement of the rule and not a second implementation of it.
    fn expected(self) -> [u8; 4] {
        match self {
            // 0.25 -> 64, 0.75 -> 191, 2.5 -> clamped 255, 0.875 -> 223.
            Self::Primary => [0x40, 0xbf, 0xff, 0xdf],
            // 0.125 -> 32.
            Self::Moved => [0x40, 0xbf, 0xff, 0x20],
        }
    }

    /// The volume's tightly packed slice-major bytes at the lane's own texel
    /// width.
    fn bytes(self, format: TextureFormat) -> Vec<u8> {
        match format {
            TextureFormat::R32Float => self
                .volume()
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
            TextureFormat::Bgra8Unorm => self.eight_bit_volume().to_vec(),
            other => panic!("the volume fixtures state two lanes, not {other:?}"),
        }
    }

    /// The volume's texels in the **eight-bit** lane's own byte order,
    /// `blue, green, red, alpha` per texel and slice by slice inside it.
    ///
    /// The fixture reads each sample's `.x` component, and Vulkan fills that
    /// from a `B8G8R8A8_UNORM` view's **red** channel — the third byte of the
    /// lane's own memory order — so each sampled texel carries the byte the
    /// frame has to land in its `red` position and two decoys (`0x11` in the
    /// blue byte, `0x22` in the green one) that no lane of the expectation
    /// holds. `0x11` is exactly what a view of the *other* four-byte order
    /// would hand the module, which is the mix-up the decoys are there for.
    fn eight_bit_volume(self) -> [u8; (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH) as usize * 4] {
        let mut texels = [EIGHT_BIT_FILL; (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH) as usize];
        let position = |slice: u64, row: u64, column: u64| -> usize {
            (slice * VOLUME_HEIGHT * VOLUME_WIDTH + row * VOLUME_WIDTH + column) as usize
        };
        // The same four positions the float lane samples, in the same order:
        // the two slices are what tells a volume walk from a one-slice upload,
        // and the four columns and rows are what tell a row pitch from a slice
        // pitch at the eight-bit lane's own four-byte texel.
        texels[position(0, 0, 0)][2] = 0x40;
        texels[position(0, 2, 0)][2] = 0xff;
        texels[position(1, 0, 2)][2] = 0xbf;
        texels[position(1, 2, 2)][2] = match self {
            Self::Primary => 0xdf,
            Self::Moved => 0x20,
        };
        let mut bytes = Vec::with_capacity(texels.len() * 4);
        for texel in texels {
            bytes.extend_from_slice(&texel);
        }
        bytes
            .try_into()
            .expect("the volume's own tightly packed byte extent")
    }
}

/// The API's own float-to-unorm conversion of a sampled value: the attachment
/// stores `round(clamp(value, 0, 1) * 255)`, and none of the fixture's values
/// lands on a tie.
fn to_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
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
        .compile_pipeline(&function, digest(b"three-dim-volume-compute"))
        .expect("the compute pipeline registers")
}

/// The contract a volume sampling registration states: the fixture's own two
/// entries, one `rgba8_unorm` attachment, no vertex stream, and one sampled
/// texture at `[[texture(0)]]` whose format is the caller's and whose type is
/// the shape the module declared.
fn contract(
    format: TextureFormat,
    texture_type: TextureType,
    sampler: SamplerPolicy,
) -> RenderPipelineContract {
    let mut sampled = TextureBindingContract::sampled(0, format, sampler);
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

/// One `D3` view over the volume's own `4 x 4 x 2` grid: one sample, one
/// layer, one descriptor, and the bytes the pattern states.
fn volume_view(format: TextureFormat, pattern: Pattern) -> TextureView {
    TextureView {
        view_id: TEXTURE_VIEW,
        metal_binding: 0,
        allocation_id: TEXTURE_ALLOCATION,
        texture_type: TextureType::D3,
        format,
        width: VOLUME_WIDTH,
        height: VOLUME_HEIGHT,
        depth: VOLUME_DEPTH,
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
        operation_id: OperationId::new(71),
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

/// The frame the trace rail lands for one volume, or the refusal that stopped
/// it before the device ever ran.
fn trace_readback(
    provider: &VulkanComputeProvider,
    compute: &CompiledComputePipeline,
    render: &CompiledComputePipeline,
    texture: TextureView,
) -> Result<Vec<u8>, ProviderError> {
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

/// The frame the object rail lands for the same registration and volume: the
/// object API's own command buffer over the very provider the trace rail used.
fn object_readback(
    provider: &Arc<VulkanComputeProvider>,
    render: &CompiledComputePipeline,
    format: TextureFormat,
    pattern: Pattern,
) -> Vec<u8> {
    use metal_api_core::provider::{PipelineCompileRequest, ShaderSource};
    use metal_api_core::provider_api::RenderAttachmentLoad;
    use metal_api_core::Size;

    let handle: Arc<VulkanComputeProvider> = Arc::clone(provider);
    let device = objects::Device::new(handle);
    let pipeline = device
        .render_pipeline(render)
        .expect("the registration wraps for the object API");
    let attachment = device
        .new_buffer_with_bytes(vec![0xfe; 64])
        .expect("the attachment buffer is declared");
    let view = attachment
        .view(0, (u64::from(EXTENT) * u64::from(EXTENT) * 4) as usize)
        .expect("the attachment view is declared");
    let scratch = device
        .new_buffer_with_bytes(vec![0xab; 4])
        .expect("the scratch buffer is declared");
    let scratch_view = scratch.view(0, 4).expect("the scratch view is declared");
    let volume = device
        .new_volume_texture_with_bytes(
            format,
            VOLUME_WIDTH,
            VOLUME_HEIGHT,
            VOLUME_DEPTH,
            pattern.bytes(format),
        )
        .expect("the sampled volume is declared");
    let command = device.new_command_queue().command_buffer();
    {
        let declaring = device
            .compile_pipeline(PipelineCompileRequest {
                entry_name: "copy_word".to_owned(),
                logical_digest: digest(b"three-dim-volume-object-declaring"),
                source: ShaderSource::SanitizedLl(COPY_WORD_AIR.to_owned()),
            })
            .expect("the declaring kernel registers");
        let mut encoder = command.compute_command_encoder().expect("compute encoder");
        encoder
            .set_compute_pipeline_state(&declaring)
            .expect("compute pipeline state");
        encoder
            .set_buffer(0, &view)
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
        let mut render_encoder = command
            .render_command_encoder()
            .expect("the render encoder opens");
        render_encoder
            .set_render_pipeline_state(&pipeline)
            .expect("the sampling pipeline is bound");
        render_encoder
            .set_fragment_texture(0, &volume)
            .expect("the sampled volume is bound at its own index");
        render_encoder
            .draw_render_pass(
                &view,
                AttachmentFormat::Rgba8Unorm,
                u64::from(EXTENT),
                u64::from(EXTENT),
                RenderAttachmentLoad::Clear(CLEAR_SENTINEL),
                None,
            )
            .expect("the full-screen triangle is drawn");
        render_encoder
            .end_encoding()
            .expect("the render encoder closes");
    }
    command.commit().expect("the object command commits");
    command
        .wait_until_completed()
        .expect("the object command completes");
    attachment
        .read()
        .expect("the attachment bytes are readable")
}

/// One texel of the readback: the fixture reads the same four texels for every
/// fragment, so the whole attachment carries one colour.
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
    fragment_air: &str,
    fragment_entry: &str,
    format: TextureFormat,
    texture_type: TextureType,
    sampler: SamplerPolicy,
) -> Result<CompiledComputePipeline, ProviderError> {
    let (vertex, fragment) = translated_pair(executor, fragment_air, fragment_entry);
    let mut declaring = contract(format, texture_type, sampler);
    declaring.fragment_entry = fragment_entry.to_owned();
    // One logical pipeline per shape under test: the entry, the type and the
    // sampler are what tell the registrations apart, so they are the digest.
    let logical = format!("{fragment_entry}:{texture_type:?}:{sampler:?}");
    provider.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract: declaring,
        vertex,
        fragment,
        logical_digest: digest(logical.as_bytes()),
    })
}

/// Reading 1 (2026-09-20): the volume bind enters the provider, the frame is
/// the volume's own floats' conversion, and the capability frame the consumer's
/// class gate reads states the volume window.
#[test]
fn the_three_dimensional_volume_executes_and_the_frame_is_its_own_texels() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The frame the consumer's own class gate reads must carry the lane and a
    // window at least as wide as this fixture's volume: the widening is
    // answered from *these* fields, not from a second table.
    let capabilities = provider.capabilities();
    assert!(
        capabilities
            .supported_render_texture_formats
            .contains(&TextureFormat::R32Float),
        "the provider's capability frame states the volume's float lane: {:?}",
        capabilities.supported_render_texture_formats
    );
    assert!(
        capabilities.max_render_texture_dimension_3d >= VOLUME_WIDTH,
        "the provider's volume window covers the fixture's extents: {}",
        capabilities.max_render_texture_dimension_3d
    );
    assert_eq!(
        to_unorm8(FILL),
        FILL_UNORM,
        "the fill value's own conversion is the byte the falsifier assumes"
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
    assert_eq!(
        Pattern::Primary.sampled(),
        [0.25, 0.75, 2.5, 0.875],
        "the four sampled values are the ones the expectation converts"
    );

    let pipeline = register(
        &provider,
        &executor,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        TextureFormat::R32Float,
        TextureType::D3,
        MODULE_SAMPLER,
    )
    .expect("the volume declaration registers");
    let frame = trace_readback(
        &provider,
        &compute,
        &pipeline,
        volume_view(TextureFormat::R32Float, Pattern::Primary),
    )
    .expect("the volume bind executes");
    let expected = Pattern::Primary.expected();
    eprintln!(
        "trace volume frame {} (expected {})",
        hex(&frame),
        hex(&expected)
    );
    assert_eq!(uniform_texel(&frame), expected, "{}", hex(&frame));

    // The falsifier: 2.5's own leading byte is `0x00`, so a rail that read a
    // texel's first byte as a normalised component would land it where the
    // frame carries the clamp's `0xff` — and the same holds for the three
    // in-range values beside it.
    let uploaded = Pattern::Primary.bytes(TextureFormat::R32Float);
    let sampled_bytes = Pattern::Primary
        .sampled()
        .map(|value| value.to_bits().to_le_bytes()[0]);
    for (lane, (byte, expected_byte)) in sampled_bytes.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            *byte, 0x00,
            "lane {lane}: the texel's own first byte is the falsifier"
        );
        assert_ne!(
            *expected_byte, *byte,
            "lane {lane}: the frame has to carry the float's conversion, not the texel's byte"
        );
    }
    assert_eq!(
        uploaded.len(),
        4 * 4 * 2 * 4,
        "the volume's own byte extent"
    );
}

/// Reading 4 (2026-09-20, census v48's volume lane gate): the volume's
/// **eight-bit** lane — the census's own `B8G8R8A8_UNORM` volumes — executes,
/// and what admits it is the *device's* own answer rather than the surface
/// format list beside it.
///
/// The reading is the float lane's own frame over the other lane's texels: the
/// four sampled positions carry the bytes `0x40`, `0xbf`, `0xff` and `0xdf` in
/// their blue channel (the component a `Bgra8Unorm` view hands the fixture's
/// `.x` sample), with three distinct decoys beside each one and a fill texel
/// the frame may not contain. So a rail that walked the source at another texel
/// width, that uploaded one slice, that read the rows in the wrong order, or
/// that handed the module a reordered byte lands a frame this expectation does
/// not have — and the two lanes (trace and objects) still have to land it byte
/// for byte.
///
/// The second half is the compat rule the section states: the same trace
/// against the same device's snapshot with the lane list cleared is refused by
/// name, which is exactly what a consumer reading a frame written before the
/// section does.
#[test]
fn the_eight_bit_volume_lane_executes_and_the_empty_list_refuses_it() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    // The device's own answer, archived with the frame: a device that refused
    // this lane reports it here instead of at the first `vkCreateImage` of the
    // pass's volume.
    let capabilities = provider.capabilities();
    eprintln!(
        "volume lanes {:?} with window {} (the surface list carries {})",
        capabilities.supported_render_texture_volume_formats,
        capabilities.max_render_texture_dimension_3d,
        match capabilities
            .supported_render_texture_formats
            .contains(&TextureFormat::Bgra8Unorm)
        {
            true => "bgra8_unorm",
            false => "no bgra8_unorm",
        }
    );
    assert!(
        capabilities
            .supported_render_texture_volume_formats
            .contains(&TextureFormat::Bgra8Unorm),
        "the device answers for the census's own volume lane: {:?}",
        capabilities.supported_render_texture_volume_formats
    );
    assert!(
        capabilities.max_render_texture_dimension_3d >= VOLUME_WIDTH,
        "the device's volume window covers the fixture's extents: {}",
        capabilities.max_render_texture_dimension_3d
    );

    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &provider,
        &executor,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        TextureFormat::Bgra8Unorm,
        TextureType::D3,
        MODULE_SAMPLER,
    )
    .expect("the eight-bit volume declaration registers");

    let primary = trace_readback(
        &provider,
        &compute,
        &pipeline,
        volume_view(TextureFormat::Bgra8Unorm, Pattern::Primary),
    )
    .expect("the eight-bit volume executes");
    let objects_primary = object_readback(
        &provider,
        &pipeline,
        TextureFormat::Bgra8Unorm,
        Pattern::Primary,
    );
    eprintln!(
        "eight-bit frames: trace {} objects {} (expected {})",
        hex(&primary),
        hex(&objects_primary),
        hex(&Pattern::Primary.expected())
    );
    assert_eq!(
        uniform_texel(&primary),
        Pattern::Primary.expected(),
        "trace frame: {}",
        hex(&primary)
    );
    assert_eq!(
        uniform_texel(&objects_primary),
        Pattern::Primary.expected(),
        "object frame: {}",
        hex(&objects_primary)
    );
    assert_eq!(
        primary, objects_primary,
        "the two lanes land the eight-bit volume's frame"
    );

    let moved = trace_readback(
        &provider,
        &compute,
        &pipeline,
        volume_view(TextureFormat::Bgra8Unorm, Pattern::Moved),
    )
    .expect("the moved eight-bit volume executes");
    eprintln!(
        "eight-bit moved frame {} (expected {})",
        hex(&moved),
        hex(&Pattern::Moved.expected())
    );
    assert_eq!(
        uniform_texel(&moved),
        Pattern::Moved.expected(),
        "the moved volume changes the lane the reading watches: {}",
        hex(&moved)
    );

    // The lane's own texel width is the arm's whole question at this lane: the
    // source is thirty-two **four-byte** texels, and the bytes the frame lands
    // are the third byte of each sampled texel rather than the fill's `0x11`
    // two bytes to the left of it, which is the byte a view of the other
    // four-byte order would have handed the module.
    let uploaded = Pattern::Primary.bytes(TextureFormat::Bgra8Unorm);
    assert_eq!(uploaded.len(), 4 * 4 * 2 * 4, "the volume's byte extent");
    for (lane, expected_byte) in Pattern::Primary.expected().iter().enumerate() {
        assert_ne!(
            *expected_byte, EIGHT_BIT_FILL[2],
            "lane {lane}: the frame carries a sampled texel, not the fill"
        );
        assert_ne!(
            *expected_byte, EIGHT_BIT_FILL[0],
            "lane {lane}: the frame carries the lane's red byte, not the blue one the other \
             four-byte order would read"
        );
    }
    assert!(
        !uploaded
            .chunks_exact(4)
            .filter(|texel| [texel[0], texel[1], texel[3]]
                == [EIGHT_BIT_FILL[0], EIGHT_BIT_FILL[1], EIGHT_BIT_FILL[3]])
            .all(|texel| texel[2] == EIGHT_BIT_FILL[2]),
        "the source's sampled texels carry a byte the fill does not"
    );

    // The compat rule, measured on the same device: the lane list is what
    // admits the shape, and a snapshot that declares the window but no lane
    // keeps the pre-increment reading, where `r32_float` was the arm's only
    // lane — the refusal by name a frame written before the section produces.
    let (trace, resources) = trace_for(
        &provider,
        &compute,
        &pipeline,
        vec![volume_view(TextureFormat::Bgra8Unorm, Pattern::Primary)],
    );
    let mut pre_increment = provider.capabilities();
    pre_increment
        .supported_render_texture_volume_formats
        .clear();
    assert!(!pre_increment.declares_render_texture_volume_formats());
    let refusal = match pre_increment.validate_trace(trace, resources) {
        Ok(_) => panic!("the pre-increment reading refuses the eight-bit lane"),
        Err(error) => error,
    };
    assert_eq!(refusal.slug, "render_texture_volume_format_unsupported");
    eprintln!(
        "pre-increment refusal: {} {:?}",
        refusal.slug,
        refusal.fields.get("format")
    );
}

/// Reading 2 (2026-09-20): both lanes land one frame, and another volume moves
/// every sampled channel the reading watches — so the measurement is the
/// upload and not the run.
#[test]
fn both_lanes_land_the_volume_frame_and_another_volume_moves_it() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &provider,
        &executor,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        TextureFormat::R32Float,
        TextureType::D3,
        MODULE_SAMPLER,
    )
    .expect("the volume declaration registers");

    let primary = trace_readback(
        &provider,
        &compute,
        &pipeline,
        volume_view(TextureFormat::R32Float, Pattern::Primary),
    )
    .expect("the primary volume executes");
    let objects_primary = object_readback(
        &provider,
        &pipeline,
        TextureFormat::R32Float,
        Pattern::Primary,
    );
    eprintln!(
        "primary frames: trace {} objects {} (expected {})",
        hex(&primary),
        hex(&objects_primary),
        hex(&Pattern::Primary.expected())
    );
    assert_eq!(
        uniform_texel(&primary),
        Pattern::Primary.expected(),
        "trace frame: {}",
        hex(&primary)
    );
    assert_eq!(
        uniform_texel(&objects_primary),
        Pattern::Primary.expected(),
        "object frame: {}",
        hex(&objects_primary)
    );
    assert_eq!(primary, objects_primary, "the two lanes land one frame");

    let moved = trace_readback(
        &provider,
        &compute,
        &pipeline,
        volume_view(TextureFormat::R32Float, Pattern::Moved),
    )
    .expect("the moved volume executes");
    let objects_moved = object_readback(
        &provider,
        &pipeline,
        TextureFormat::R32Float,
        Pattern::Moved,
    );
    eprintln!(
        "moved frames: trace {} objects {} (expected {})",
        hex(&moved),
        hex(&objects_moved),
        hex(&Pattern::Moved.expected())
    );
    assert_eq!(
        uniform_texel(&moved),
        Pattern::Moved.expected(),
        "moved trace frame: {}",
        hex(&moved)
    );
    assert_eq!(
        uniform_texel(&objects_moved),
        Pattern::Moved.expected(),
        "moved object frame: {}",
        hex(&objects_moved)
    );
    assert_eq!(moved, objects_moved, "the two lanes land one frame");
    assert_ne!(
        uniform_texel(&primary),
        Pattern::Moved.expected(),
        "the moved volume has to move the frame: {} vs {}",
        hex(&primary),
        hex(&moved)
    );
    // The first three channels are the ones the moved volume leaves alone: only
    // the alpha texel's own value changes.
    assert_eq!(uniform_texel(&primary)[..3], Pattern::Moved.expected()[..3]);
    assert_ne!(uniform_texel(&primary)[3], Pattern::Moved.expected()[3]);
}

/// Reading 3 (2026-09-20): the shapes beside the admitted one keep their
/// refusals by name — the declaration has to restate the module's own axis in
/// both directions, and the volume's extents stay inside the reviewed window.
#[test]
fn the_shapes_beside_the_three_dimensional_volume_keep_their_names() {
    let Some((executor, provider)) = executor_and_provider() else {
        return;
    };

    // A two-dimensional declaration for the volume module: the view the rail
    // creates would be `TYPE_2D` while the module's own image is `Dim 3D`.
    let refused = register(
        &provider,
        &executor,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        TextureFormat::R32Float,
        TextureType::D2,
        MODULE_SAMPLER,
    )
    .expect_err("a 2D declaration does not cover a volume module");
    eprintln!("d2 for volume refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_shape_unsupported");
    assert_eq!(refused.class, ProviderErrorClass::Capability);
    assert_eq!(
        refused.fields.get("texture_type"),
        Some(&FieldValue::Text("D2".to_owned()))
    );

    // The other direction: a three-dimensional declaration for the reviewed
    // two-dimensional module.
    let refused = register(
        &provider,
        &executor,
        TWO_DIM_AIR,
        TWO_DIM_ENTRY,
        TextureFormat::R32Float,
        TextureType::D3,
        TWO_DIM_SAMPLER,
    )
    .expect_err("a volume declaration does not cover a two-dimensional module");
    eprintln!("d3 for 2D module refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_shape_unsupported");

    // The reviewed window: a volume wider than the contract's ceiling is
    // refused by the rail's own gate with the axis it read, before any device
    // object exists.
    let compute = compile_declaring_kernel(&provider, &executor);
    let pipeline = register(
        &provider,
        &executor,
        FRAGMENT_AIR,
        FRAGMENT_ENTRY,
        TextureFormat::R32Float,
        TextureType::D3,
        MODULE_SAMPLER,
    )
    .expect("the volume declaration registers");
    let mut oversized = volume_view(TextureFormat::R32Float, Pattern::Primary);
    oversized.width = 4_096;
    // The structural length rule runs before any capability gate, so the
    // oversized declaration carries its own byte extent — the refusal under
    // test is the window's, not the source's.
    oversized.source = TextureSource::OwnedBytes(vec![
        0u8;
        (4_096 * VOLUME_HEIGHT * VOLUME_DEPTH * 4)
            as usize
    ]);
    let (trace, resources) = trace_for(&provider, &compute, &pipeline, vec![oversized]);
    let refused = provider
        .capabilities()
        .validate_trace(trace, resources)
        .expect_err("a volume above the reviewed ceiling is refused");
    eprintln!("oversized volume refused: {refused:?}");
    assert_eq!(refused.slug, "render_texture_dimension_3d_limit");
    assert_eq!(
        refused.fields.get("axis"),
        Some(&FieldValue::Text("width".to_owned()))
    );
}
