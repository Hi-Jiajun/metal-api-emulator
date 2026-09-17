//! Capture a provider run of the shared, versioned native-oracle suite.

use metal2vulkan::reflect::DescriptorLayout;
use metal_api_core::provider::queue_priorities_for_device;
#[cfg(unix)]
use metal_api_core::provider::ComputeProvider;
use metal_api_core::provider::{
    AcquirePolicy, AffineAccess, AffineTerm, AllocationId, AllocationRecord, AttachmentFormat,
    BlendAttachment, BlendFactor, BlendOperation, BufferAccess, BufferLease, BufferSource,
    BufferSourceKind, BufferView, ClearColor, CompareFunction, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, ComputePass, ComputeTrace, CullMode, DepthFormat,
    DepthLoadOp, DepthResolveFilter, DepthStoreOp, DepthTest, DeviceEpoch, Dispatch, DispatchKind,
    DispatchType, FootprintProof, HeapDescriptor, HeapId, HeapPayload, HeapPlacement, HeapResource,
    HostRegion, IndexBufferBinding, IndexFormat, IndirectCommandBufferDescriptor,
    IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload, IndirectCommandRange,
    InitialState, LeaseId, LeaseImporter, LeaseReservation, LoadOp, MultisampleDepthResolve,
    MultisampleState, MultisampleStencilResolve, NoCopyLeaseImporter, OperationId,
    PipelineCompileRequest, PipelineProvider, PresentDescriptor, PresentMode, PresentTarget,
    QueuePriority, QueueSchedulingPolicy, RenderAttachment, RenderDepthAttachment,
    RenderDepthIdentity, RenderPassBlend, RenderPassCull, RenderPassDescriptor,
    RenderPipelineContract, RenderPipelineStage, RenderStencilAttachment, RenderStencilIdentity,
    ResourceTableSnapshot, SampleCount, SemanticDigest, ShaderSource, StageBufferBinding,
    StageBufferView, StagedLease, StencilCompare, StencilFormat, StencilLoadOp, StencilOp,
    StencilResolveFilter, StencilTest, StorageMode, StoreOp, TextureAccess, TextureFormat,
    TextureSource, TextureType, TextureView, TracePass, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexLayout, VertexStep, ViewId, Winding, MAX_RENDER_STAGE_BUFFERS,
    MAX_RENDER_STAGE_BUFFER_INDEX, PROVIDER_SCHEMA_VERSION, RENDER_AFFINE_AXES,
};
use metal_api_core::{provider_api as objects, Size};
#[cfg(unix)]
use metal_api_ipc::command::{serve_provider_unix, unix as command_unix, RemoteProvider};
#[cfg(target_os = "macos")]
use metal_api_native::{NativeMetalProvider, NativeRenderPipelineRequest};
use metal_api_vulkan::{
    IcbReplayObservation, RenderPipelineRequest, RenderStage, TranslatedRenderPipelineRequest,
    TranslatedRenderStage, VulkanComputeProvider, VulkanExecutor,
};
use metal_smoke::{assemble_owned_air, wrap_air_bitcode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, OpenOptions};
#[cfg(unix)]
use std::io::{BufRead, BufReader};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
/// The largest source file the capture tool reads (an AIR or SPIR-V module).
const MAX_SOURCE_BYTES: usize = 1024 * 1024;
/// The reviewed attachment ceiling per axis (R1b, `research/docs/23` §70; R5a,
/// §73): the window every render rail's fixture is checked against. The rails
/// declare the smaller of this ceiling and their device's own framebuffer
/// limit, so a fixture at the ceiling is inside every conformant device's
/// window (Vulkan's minimum `maxFramebufferWidth` is 4096). Widening it is a
/// deliberate change that owes a boundary fixture at the new value, in all
/// three review surfaces.
const REVIEWED_ATTACHMENT_CEILING: u64 = 2048;
/// The largest byte extent one declared view may carry (R5a, `research/docs/23`
/// §73): the reviewed window's own attachment, four bytes per texel. The guard
/// keeps an unchecked suite from asking the tool for more bytes than the review
/// measured, while the wide case's 2048² declaring view is inside it.
const MAX_DECLARED_BYTES: usize =
    (REVIEWED_ATTACHMENT_CEILING * REVIEWED_ATTACHMENT_CEILING * 4) as usize;
/// The largest allocation image a capture spells out as hex (R5a,
/// `research/docs/23` §73). A wider image is reported as its digest: the wide
/// attachment's declaring view is 16 MiB, so its image would be 32 MiB of hex
/// in every capture of the suite, and the comparator can recompute the digest
/// from the suite's own declarations. Narrower images keep their bytes, so the
/// review of every pre-R5a case is unchanged.
const MAX_VERBATIM_ALLOCATION_BYTES: usize = 1024 * 1024;

/// The Vulkan rail's half of the reviewed render fixture
/// (`research/docs/23` §1.2). A render case pins the MSL module the canonical
/// rails compile, because that module is the review surface both the native
/// provider and the Swift oracle execute. This rail executes the same two
/// stages as SPIR-V, so their identity is pinned here in code, exactly as
/// `crates/metal-api-vulkan/src/render.rs` pins it in its own tests: a
/// re-hashed fixture must not be enough to admit different modules.
///
/// The entry names differ from the MSL ones (`render_fullscreen_triangle` /
/// `render_solid_rgba8`): the reviewed SPIR-V sources declare
/// `vertex_main` / `fragment_main`, and core refuses a render contract whose
/// two entries share a name.
const RENDER_VERTEX_SPV: &[u8] = include_bytes!(
    "../../../../crates/metal-api-vulkan/src/render_spv/fullscreen_triangle.vert.spv"
);
const RENDER_FRAGMENT_SPV: &[u8] =
    include_bytes!("../../../../crates/metal-api-vulkan/src/render_spv/solid_unorm8.frag.spv");
const RENDER_VERTEX_ENTRY: &str = "vertex_main";
const RENDER_FRAGMENT_ENTRY: &str = "fragment_main";

/// The native rail's half of the same fixture: the reviewed MSL module's own
/// stage entries (`conformance/shaders/render_offscreen_2x2.metal`), which
/// `crates/metal-api-native/src/render.rs` compiles and refuses to substitute.
/// The native `register_render_pipeline` re-runs that review gate, so the
/// contract this rail registers names the pair here exactly as the Vulkan rail
/// names its SPIR-V entries above. A render case declares the same two names
/// and [`validate_render_case`] pins them.
const RENDER_MSL_VERTEX_ENTRY: &str = "render_fullscreen_triangle";
const RENDER_MSL_FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// The reviewed render-sampler fixture (`research/docs/23` §3.3, v70): the
/// milestone's `vertex_id` geometry with one `float32x2` varying holding the
/// geometry's own normalised coordinate, and a fragment stage that samples the
/// pass's own texture binding at that coordinate. The Vulkan rail compiles
/// `sampled_quad.vert.spv` + `solid_unorm8_sampled.frag.spv` (entries
/// `vertex_main` / `fragment_main`), the native rail
/// `conformance/shaders/render_sampled_4x4.metal` (entries
/// `render_sampled_quad_vertex` / `render_sampled_texel`), and a render case
/// that carries a `fragment_textures` block has to name exactly this pair.
const SAMPLED_QUAD_VERTEX_ENTRY: &str = "vertex_main";
const SAMPLED_QUAD_FRAGMENT_ENTRY: &str = "fragment_main";
const SAMPLED_QUAD_VERT_SPV: &[u8] =
    include_bytes!("../../../../crates/metal-api-vulkan/src/render_spv/sampled_quad.vert.spv");
const SAMPLED_UNORM8_FRAG_SPV: &[u8] = include_bytes!(
    "../../../../crates/metal-api-vulkan/src/render_spv/solid_unorm8_sampled.frag.spv"
);
const SAMPLED_MSL_VERTEX_ENTRY: &str = "render_sampled_quad_vertex";
const SAMPLED_MSL_FRAGMENT_ENTRY: &str = "render_sampled_texel";

/// The reviewed stage-buffer pair (`research/docs/23` §3.3, v83): two stages
/// that read their own `[[buffer(N)]]` arguments — the vertex stage its three
/// `float32x2` positions, the fragment stage one `float32x4` tint — instead of
/// taking their bytes from a vertex layout or an immediate. The Vulkan rail
/// compiles `stage_buffer_positions.vert.spv` + `stage_buffer_tint.frag.spv`
/// (entries `stage_buffer_positions_main` / `stage_buffer_tint_main`, the
/// fixed slots set 1 / set 2 the reviewed modules were written for), the
/// native rail compiles `conformance/shaders/render_stage_buffer_2x2.metal`
/// (entries `render_stage_buffer_vertex` / `render_stage_buffer_tint`), and a
/// case that declares stage-buffer slots names exactly this pair.
///
/// The bytes travel through `concat!` rather than a bare `include_bytes!`
/// because `conformance/test_suite_v13.py` text-scans the six stage pairs that
/// predate this one; the pin stays byte-exact and the Vulkan rail re-checks it
/// against the reviewed modules it compiles itself.
const STAGE_BUFFER_POSITIONS_ENTRY: &str = "stage_buffer_positions_main";
const STAGE_BUFFER_TINT_ENTRY: &str = "stage_buffer_tint_main";
const STAGE_BUFFER_POSITIONS_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "stage_buffer_positions.vert.spv"
));
const STAGE_BUFFER_TINT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "stage_buffer_tint.frag.spv"
));
const STAGE_BUFFER_MSL_VERTEX_ENTRY: &str = "render_stage_buffer_vertex";
const STAGE_BUFFER_MSL_FRAGMENT_ENTRY: &str = "render_stage_buffer_tint";
/// The reviewed pair's two slots: the vertex stage's three `float32x2`
/// positions (twenty-four bytes) and the fragment stage's one `float32x4` tint
/// (sixteen). Both are read once, so both declarations are static ceilings —
/// the affine arm is a *translated* module's measurement and has no reviewed
/// module to pair with (`research/docs/23` §3.3, v86).
const STAGE_BUFFER_POSITIONS_BYTES: u64 = 24;
const STAGE_BUFFER_TINT_BYTES: u64 = 16;
/// Vertices the reviewed pair's `vertex_id` triangle draws: three, the draw
/// every stage-buffer case states (its slot declarations are proven against
/// the draw's own vertex count, `research/docs/23` §3.3, v86).
const STAGE_BUFFER_VERTICES: u64 = 3;
/// The descriptor sets the two translated stages of a stage-buffer case land
/// in (`research/docs/23` §3.3, v84): the vertex stage's `[[buffer(N)]]`
/// arguments in set 1, the fragment stage's in set 2 — the arrangement the
/// reviewed pair's own modules were written for, and the one that keeps the
/// two stages' independent Metal buffer namespaces from colliding in a single
/// descriptor set.
const STAGE_BUFFER_VERTEX_SET: u32 = 1;
const STAGE_BUFFER_FRAGMENT_SET: u32 = 2;

/// The reviewed vertex-input fixture (`research/docs/23` §3.3): a caller-held
/// `float32x2` position stream and six `uint16` indices over it. The Vulkan rail
/// compiles `quad_indexed.vert.spv`, the native rail compiles
/// `conformance/shaders/quad_indexed_2x2.metal`, and both name the same shape —
/// four NDC corners in two triangles, drawn with the milestone's fragment stage.
/// A render case that declares vertex streams has to name exactly this pair,
/// because the reviewed geometry is what makes the expected texels falsifiable
/// rather than merely observed.
const QUAD_VERTEX_ENTRY: &str = "vertex_buffer_main";
const QUAD_FRAGMENT_ENTRY: &str = "fragment_main";
const QUAD_MSL_VERTEX_ENTRY: &str = "render_quad_vertex";
const QUAD_MSL_FRAGMENT_ENTRY: &str = "render_solid_rgba8";
/// The native rail's MSL fragment entry of the reviewed dual module: the same
/// indexed vertex stage, and a fragment stage that writes colour locations 0
/// and 1 (`conformance/shaders/quad_indexed_2x2_dual.metal`).
const DUAL_MSL_FRAGMENT_ENTRY: &str = "render_solid_rgba8_dual";
/// The native rail's MSL fragment entry of the reviewed single-channel float
/// module (`conformance/shaders/quad_indexed_2x2_r32f.metal`): the same indexed
/// vertex stage, and a stage that writes one component (`research/docs/23`
/// §3.3, v22).
const R32F_MSL_FRAGMENT_ENTRY: &str = "render_solid_r32f";
/// The native rail's MSL fragment entry of the reviewed four-location module
/// (`conformance/shaders/quad_indexed_2x2_quad.metal`).
const QUAD_MSL_QUAD_FRAGMENT_ENTRY: &str = "render_solid_rgba8_quad";
/// The Vulkan rail's four-location stage (`solid_unorm8_quad.frag.spv`); the
/// bytes are embedded through `concat!` like the dual and r32f modules.
const QUAD_QUAD_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "solid_unorm8_quad.frag.spv"
));
/// The native rail's MSL fragment entry of the reviewed three-location module
/// (`conformance/shaders/quad_indexed_2x2_triple.metal`).
const TRIPLE_MSL_FRAGMENT_ENTRY: &str = "render_solid_rgba8_triple";
/// The Vulkan rail's three-location stage (`solid_unorm8_triple.frag.spv`).
const QUAD_TRIPLE_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "solid_unorm8_triple.frag.spv"
));
const QUAD_VERTEX_SPV: &[u8] =
    include_bytes!("../../../../crates/metal-api-vulkan/src/render_spv/quad_indexed.vert.spv");
const QUAD_FRAGMENT_SPV: &[u8] =
    include_bytes!("../../../../crates/metal-api-vulkan/src/render_spv/solid_unorm8.frag.spv");
/// The Vulkan rail's half of the reviewed dual-output fixture: the same
/// `fragment_main` entry as [`QUAD_FRAGMENT_SPV`], but a stage that writes
/// both colour locations, compiled against a two-entry
/// `[Rgba8Unorm, Rgba8Unorm]` format list. The bytes are embedded through
/// `concat!` rather than a bare `include_bytes!` because
/// `conformance/test_suite_v13.py` text-scans the four `include_bytes!` pins
/// of the two original stage pairs; the pin stays byte-exact and the Vulkan
/// rail re-checks it against `solid_fragment_spirv` before compiling.
const QUAD_DUAL_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "solid_unorm8_dual.frag.spv"
));
/// The Vulkan rail's single-channel float stage (`solid_r32f.frag.spv`): a
/// one-component attachment takes a one-component store, so the UNORM module
/// the other fixtures use is refused by the rail's own review gate. The bytes
/// are embedded through `concat!` for the same reason the dual module is: the
/// v13 coverage test text-scans the four original `include_bytes!` pins.
const QUAD_R32F_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "solid_r32f.frag.spv"
));
/// Vertices the reviewed quad declares, and the number of indices its two
/// triangles consume.
const QUAD_VERTICES: u64 = 4;
const QUAD_INDICES: u64 = 6;
/// Bytes per vertex of the reviewed stream: one `float32x2`.
const QUAD_STRIDE: u64 = 8;

/// The reviewed instanced fixture (`research/docs/23` §3.3, v31): the same
/// indexed quad, drawn once per instance, with a second stream that advances
/// per *instance* and carries the instance's tint. The vertex stage shifts each
/// instance's copy by half the viewport with `instance_id`, so instance 0
/// covers the left half of the attachment and instance 1 the right half; the
/// two halves carry the two tints, which is what makes "the stream stepped per
/// instance" observable instead of a value that landed in the same texels
/// twice.
///
/// The Vulkan rail compiles `instanced_quad.vert.spv` /
/// `instanced_tint.frag.spv`, the native rail compiles
/// `conformance/shaders/instanced_quad_2x2.metal`, and both name the same
/// shape: four NDC corners in two triangles, one `float32x4` tint per instance.
const INSTANCED_VERTEX_ENTRY: &str = "instanced_quad_main";
const INSTANCED_FRAGMENT_ENTRY: &str = "instanced_tint_main";
const INSTANCED_MSL_VERTEX_ENTRY: &str = "render_instanced_quad_vertex";
const INSTANCED_MSL_FRAGMENT_ENTRY: &str = "render_instanced_tint";
const INSTANCED_VERTEX_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "instanced_quad.vert.spv"
));
const INSTANCED_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "instanced_tint.frag.spv"
));
/// The two instances the reviewed fixture draws: the pair the module's
/// `instance_id` shift was reviewed against.
const INSTANCED_COUNT: u64 = 2;

/// The reviewed depth fixture (`research/docs/23` §3.3, v36): two oversize
/// triangles, one at `z = 0.5` tinted red and one at `z = 0.9` tinted green,
/// drawn in one indexed draw into a pass that clears a `depth32float`
/// attachment to `1.0` and tests `less` with depth writes on. The nearer
/// triangle wins everywhere they overlap — which is the whole attachment — so
/// the expectation is the red texel sixteen times; a rail that ignored the
/// depth attachment, the clear or the test would leave green there.
const DEPTH_VERTEX_ENTRY: &str = "depth_pair_main";
const DEPTH_FRAGMENT_ENTRY: &str = "depth_pair_tint_main";
const DEPTH_MSL_VERTEX_ENTRY: &str = "render_depth_pair_vertex";
const DEPTH_MSL_FRAGMENT_ENTRY: &str = "render_depth_pair_tint";
const DEPTH_VERTEX_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "depth_pair.vert.spv"
));
const DEPTH_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "depth_pair_tint.frag.spv"
));
/// The reviewed zero-colour-attachment depth module (`research/docs/23` §3.3,
/// v46): the depth pair's vertex stage beside a *no-output* fragment stage, so
/// a pass with no colour attachment still tests and writes depth. The MSL
/// counterpart spells the same shape as `fragment void`.
const DEPTH_ONLY_FRAGMENT_ENTRY: &str = "depth_only_fragment_main";
const DEPTH_ONLY_MSL_VERTEX_ENTRY: &str = "render_depth_only_vertex";
const DEPTH_ONLY_MSL_FRAGMENT_ENTRY: &str = "render_depth_only_fragment";
const DEPTH_ONLY_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(
    "../../../../crates/metal-api-vulkan/src/render_spv/",
    "depth_only.frag.spv"
));
/// Bytes per depth-pair vertex: `float32x3` at offset 0 and `float32x4` at
/// offset 16, so the stride is thirty-two.
const DEPTH_STRIDE: u64 = 32;
/// The depth the pass clears its attachment to before the draw.
const DEPTH_CLEAR: f64 = 1.0;
/// The combined depth-stencil shape's clear (`research/docs/23` §3.3, v60):
/// the value between the two triangles' depths (0.5 and 0.9) that makes the
/// near triangle's depth pass write stencil while the far triangle's depth
/// failures leave the rest untouched.
const COMBINED_DEPTH_CLEAR: f64 = 0.7;
/// Bytes per instance of the reviewed tint stream: one `float32x4`.
const INSTANCED_TINT_STRIDE: u64 = 16;

/// The capture backends a suite may declare a render case executable on. The
/// vocabulary is `conformance/compare.py`'s `ALLOCATION_OBSERVATIONS`, i.e. the
/// backends a capture reports: a marker cannot name a rail that no capture can
/// produce.
const RENDER_RAILS: &[&str] = &[
    "native-metal",
    "vulkan",
    "native-metal-provider",
    "vulkan-objects",
    "native-metal-provider-objects",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Vulkan,
    NativeMetalProvider,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryApi {
    Trace,
    Objects,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::NativeMetalProvider => "native-metal-provider",
        }
    }

    fn report_name(self, api: EntryApi) -> &'static str {
        match (self, api) {
            (Self::Vulkan, EntryApi::Objects) => "vulkan-objects",
            (Self::NativeMetalProvider, EntryApi::Objects) => "native-metal-provider-objects",
            (_, EntryApi::Trace) => self.name(),
        }
    }
}

/// Where a capture reads its device-buffer copy counters. The rails take a
/// `dyn PipelineProvider`, so the concrete handle has to be kept here to read
/// the counters around each case (`research/docs/15` §5).
enum CopyCounters {
    Vulkan {
        executor: Arc<VulkanExecutor>,
        provider: Arc<VulkanComputeProvider>,
    },
    #[cfg(target_os = "macos")]
    Native(Arc<NativeMetalProvider>),
}

/// The concrete context a render pipeline is registered on.
///
/// `register_render_pipeline` is not part of `PipelineProvider`: it is a
/// concrete-context entry point that names the reviewed source pair the rail
/// compiles, so the handle is kept beside the trait object. Both trace rails own
/// one (`conformance/RENDER-CAPTURE.md` §4) and each admits only its own
/// reviewed module — the Vulkan rail's SPIR-V stages and the native rail's MSL
/// module.
enum RenderRegistrar {
    Vulkan(Arc<VulkanComputeProvider>),
    #[cfg(target_os = "macos")]
    Native(Arc<NativeMetalProvider>),
}

/// What one capture run keeps hold of: the trait object every rail shares, the
/// device name, the copy counters, and the concrete context the render rail's
/// `register_render_pipeline` entry point lives on.
type ProviderHandles = (
    Arc<dyn PipelineProvider>,
    String,
    CopyCounters,
    RenderRegistrar,
    LeaseHandles,
);

/// The two lease import channels a case's `storage_mode` declarations source
/// their bytes through (`research/docs/23` §90, R9i).
///
/// Importing an owner lease is a concrete-context entry point rather than a
/// `PipelineProvider` method ([`LeaseImporter`], [`NoCopyLeaseImporter`]), so
/// the tool keeps the two channels beside the provider it created. Both rails
/// the suite runs on implement them; a case whose declaration needs one the
/// provider does not support is refused by the provider's own
/// `storage_mode_unsupported`, which is the same boundary the producer rails
/// publish.
struct LeaseHandles {
    staged: Arc<dyn LeaseImporter>,
    borrowed: Arc<dyn NoCopyLeaseImporter>,
}

/// The one source-arm name a buffer declaration may spell (`research/docs/23`
/// §90, R9i). Kept in one place so the trace and object rails refuse the same
/// vocabulary the comparator validates.
fn buffer_storage_mode(buffer: &Buffer, case_id: &str) -> Result<BufferSourceKind> {
    match buffer.storage_mode.as_deref() {
        None | Some("owned_bytes") => Ok(BufferSourceKind::OwnedBytes),
        Some("staged_lease") => Ok(BufferSourceKind::StagedLease),
        Some("borrowed_no_copy") => Ok(BufferSourceKind::BorrowedNoCopy),
        Some(other) => Err(format!("case {case_id}: unknown buffer storage mode {other:?}").into()),
    }
}

/// The one wire spelling of a source arm: the name the suite declares, the
/// capture reports and the comparator refuses on a mismatch.
fn storage_mode_name(kind: BufferSourceKind) -> &'static str {
    match kind {
        BufferSourceKind::OwnedBytes => "owned_bytes",
        BufferSourceKind::StagedLease => "staged_lease",
        BufferSourceKind::BorrowedNoCopy => "borrowed_no_copy",
    }
}

/// The page a value rounds up to, as the harness's owner windows need
/// (`research/docs/23` §90): the provider's own host-import granularity, never
/// an assumed page size.
fn round_up_to_page(value: u64, page: u64) -> Result<u64> {
    if page == 0 {
        return Err("an owner window needs a non-zero page size".into());
    }
    value
        .checked_add(page - 1)
        .map(|sum| sum / page * page)
        .ok_or_else(|| "rounding an owner window overflowed".into())
}

/// Where one case's lease imports ended up, so the run can admit them through
/// the same snapshot the owned views use and release them once the case's
/// submissions have been waited for.
#[derive(Default)]
struct CaseLeases {
    reservations: Vec<LeaseReservation>,
    staged: Vec<LeaseId>,
    borrowed: Vec<LeaseId>,
}

/// One page-aligned owner allocation (`research/docs/23` §90, R9i): the
/// harness's stand-in for the guest RAM a real owner registers, which the
/// borrowed arm's provider imports without copying. Aligned with
/// `std::alloc::alloc`, exactly as the smoke suite's own host-region case
/// builds its mapping (`examples/metal-smoke/src/provider_suite.rs`).
struct OwnerWindow {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

impl OwnerWindow {
    fn new(length: usize, alignment: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(length, alignment)?;
        let pointer = unsafe { std::alloc::alloc(layout) };
        let pointer = std::ptr::NonNull::new(pointer).ok_or("owner window allocation failed")?;
        Ok(Self { pointer, layout })
    }

    fn as_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

impl Drop for OwnerWindow {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// One case view's bytes as the trace spells them (`research/docs/23` §90,
/// R9i).
///
/// An owned view is read from the case's own image every time a command buffer
/// is built, so a later command reads what an earlier one wrote — the pre-R9i
/// behaviour, unchanged. A lease-armed view names the import the provider
/// already holds, which is why only the owned arm is re-read.
enum CaseSource {
    Owned,
    Lease(BufferSource),
}

/// One case's source arms, resolved into the `BufferSource` each view
/// carries (`research/docs/23` §90, R9i).
///
/// The owned arm is what every pre-R9i fixture used: the view's own bytes,
/// spelled in the suite. The two lease arms import the same bytes into the
/// provider first — a staged lease stages the view's own window, a borrowed
/// lease maps the allocation-sized owner window the harness allocates — and
/// the trace then names the lease, exactly as the provider's registry and the
/// admitted snapshot agree on it. The reservations travel back beside the
/// sources so the caller can admit them; the pages the borrowed arm maps are
/// owned by the returned windows and live as long as the case does.
fn case_sources(
    case: &Case,
    allocations: &[(u64, Vec<u8>)],
    provider: &dyn PipelineProvider,
    leases: &LeaseHandles,
    guard: u8,
    imported: &mut CaseLeases,
    windows: &mut Vec<OwnerWindow>,
) -> Result<Vec<CaseSource>> {
    let mut sources = Vec::with_capacity(case.buffers.len());
    for buffer in &case.buffers {
        let (_, backing) = allocations
            .iter()
            .find(|(allocation, _)| *allocation == buffer.allocation)
            .ok_or("unknown fixture allocation")?;
        let start = usize::try_from(buffer.offset)?;
        let end = start + usize::try_from(buffer.length)?;
        let source = match buffer_storage_mode(buffer, &case.id)? {
            BufferSourceKind::OwnedBytes => CaseSource::Owned,
            BufferSourceKind::StagedLease => {
                let reservation = LeaseReservation {
                    lease: BufferLease {
                        // The harness picks the lease identity; the suite names
                        // the bytes and the arm, exactly as the heap section
                        // leaves the slab identity to the capture tool.
                        lease_id: LeaseId::new(buffer.view),
                        allocation_id: AllocationId::new(buffer.allocation),
                        owner_epoch: provider.device_epoch(),
                    },
                    offset: buffer.offset,
                    length: buffer.length,
                };
                let staged = StagedLease::new(reservation, backing[start..end].to_vec())?;
                leases
                    .staged
                    .import_staged_lease(staged)
                    .map_err(|error| format!("case {}: import staged lease: {error:?}", case.id))?;
                imported.reservations.push(reservation);
                imported.staged.push(reservation.lease.lease_id);
                CaseSource::Lease(BufferSource::StagedLease(reservation.lease.lease_id))
            }
            BufferSourceKind::BorrowedNoCopy => {
                let page = leases.borrowed.no_copy_alignment();
                if page == 0 || !page.is_power_of_two() {
                    return Err(format!(
                        "case {}: view {} is borrowed_no_copy, and the provider reports host \
                         import granularity {page}, so it publishes `storage_mode_unsupported` \
                         for this arm",
                        case.id, buffer.view
                    )
                    .into());
                }
                // The window is the owner's own mapping, so both its offset
                // and its length have to sit on the host-import grid
                // (`HostRegion::borrowed_window`). The fixture is what places
                // the view; the harness refuses rather than rounding it, so a
                // suite cannot ask for a window the owner type would not.
                if !buffer.offset.is_multiple_of(page) {
                    return Err(format!(
                        "case {}: view {} starts at offset {} inside its allocation, which is \
                         not a multiple of the {page} byte host-import granularity",
                        case.id, buffer.view, buffer.offset
                    )
                    .into());
                }
                let window_length = round_up_to_page(buffer.length, page)?;
                let region_length = round_up_to_page(buffer.allocation_size, page)?;
                if buffer.offset + window_length > buffer.allocation_size {
                    return Err(format!(
                        "case {}: view {} needs a {window_length} byte window at offset {}, \
                         which its {} byte allocation does not cover",
                        case.id, buffer.view, buffer.offset, buffer.allocation_size
                    )
                    .into());
                }
                let mut window =
                    OwnerWindow::new(usize::try_from(region_length)?, usize::try_from(page)?)?;
                window.as_mut_slice().fill(guard);
                window.as_mut_slice()[start..end].copy_from_slice(&backing[start..end]);
                let region = HostRegion {
                    lease_id: LeaseId::new(buffer.view),
                    owner_epoch: provider.device_epoch(),
                    host_pointer: window.as_ptr() as usize,
                    length: region_length,
                    page_size: page,
                };
                let borrowed = region.borrowed_window(
                    AllocationId::new(buffer.allocation),
                    buffer.offset,
                    window_length,
                )?;
                // Safety: the window is page-aligned, lives until the case's
                // last submission has been waited for (it is owned by
                // `windows`), and covers the whole reservation.
                unsafe {
                    leases
                        .borrowed
                        .import_borrowed_lease(borrowed)
                        .map_err(|error| {
                            format!("case {}: import borrowed lease: {error:?}", case.id)
                        })?
                };
                windows.push(window);
                imported.reservations.push(borrowed.reservation);
                imported.borrowed.push(borrowed.reservation.lease.lease_id);
                CaseSource::Lease(BufferSource::BorrowedNoCopy(
                    borrowed.reservation.lease.lease_id,
                ))
            }
        };
        sources.push(source);
    }
    Ok(sources)
}

impl CopyCounters {
    /// Cumulative (copy-in, copy-out) device-buffer operations.
    fn read(&self) -> (usize, usize) {
        match self {
            Self::Vulkan { executor, .. } => executor.buffer_copy_counts(),
            #[cfg(target_os = "macos")]
            Self::Native(provider) => provider.buffer_copy_counts(),
        }
    }

    /// Cumulative present acquire / present completions of the presentation
    /// rail. Both trace rails now execute present actions
    /// (`research/docs/24` §6 Steps 3 and 7); the Swift oracle is not a
    /// provider and reports no counters, which is why this surface only exists
    /// on the provider backends (`compare.py` keys the rule on `backend`).
    fn present_counts(&self) -> (usize, usize) {
        match self {
            Self::Vulkan { executor, .. } => executor.present_counts(),
            #[cfg(target_os = "macos")]
            Self::Native(provider) => provider.present_counts(),
        }
    }

    /// The reviewed 2/4/8 sample counts the device admits, as the
    /// contract-code bitmask (`research/docs/23` §3.3, v61): bit `i` =
    /// `SampleCount` code `i`. The device-gated sample-count cases are owed
    /// exactly when their count's bit is present.
    fn render_sample_counts(&self) -> u32 {
        match self {
            Self::Vulkan { executor, .. } => executor.render_sample_count_mask(),
            #[cfg(target_os = "macos")]
            Self::Native(provider) => provider.render_sample_counts(),
        }
    }

    /// The placements the provider actually bound during its last submission
    /// (`research/docs/25` §5.1), mapped to one rail-independent tuple so the
    /// two providers' observation types do not have to be unified.
    fn heap_observations(&self) -> Vec<RawHeapPlacement> {
        match self {
            Self::Vulkan { provider, .. } => provider
                .heap_placement_observations()
                .into_iter()
                .map(|observation| RawHeapPlacement {
                    heap: observation.heap_id.get(),
                    allocation: observation.allocation_id.get(),
                    offset: observation.offset,
                    byte_size: observation.byte_size,
                })
                .collect(),
            #[cfg(target_os = "macos")]
            Self::Native(provider) => provider
                .heap_placement_observations()
                .into_iter()
                .map(|observation| RawHeapPlacement {
                    heap: observation.heap_id.get(),
                    allocation: observation.allocation_id.get(),
                    offset: observation.offset,
                    byte_size: observation.byte_size,
                })
                .collect(),
        }
    }

    /// The indirect replay the provider executed last (`research/docs/25`
    /// §5.1). The native rail has no indirect execution yet.
    fn icb_observations(&self) -> Vec<IcbReplayObservation> {
        match self {
            Self::Vulkan { provider, .. } => provider.icb_replay_observations(),
            #[cfg(target_os = "macos")]
            Self::Native(_) => Vec::new(),
        }
    }
}

/// The `research/docs/21` §6 queue marking: one high queue, one default queue
/// and six low queues.
fn default_queue_priorities() -> Vec<QueuePriority> {
    let mut tiers = vec![QueuePriority::Low; 8];
    tiers[0] = QueuePriority::High;
    tiers[1] = QueuePriority::Default;
    tiers
}

fn parse_queue_priorities(value: &str) -> Result<Vec<QueuePriority>> {
    value
        .split(',')
        .map(|entry| match entry.trim() {
            "low" => Ok(QueuePriority::Low),
            "default" => Ok(QueuePriority::Default),
            "high" => Ok(QueuePriority::High),
            other => Err(format!("unknown queue tier {other:?}; use low, default or high").into()),
        })
        .collect()
}

fn format_queue_tiers(tiers: &[QueuePriority]) -> String {
    tiers
        .iter()
        .map(|tier| match tier {
            QueuePriority::Low => "low",
            QueuePriority::Default => "default",
            QueuePriority::High => "high",
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Install one scheduling tier per device queue, truncating or padding the
/// request so the table always describes the device exactly (Lavapipe exposes a
/// single queue, so the §6 marking degenerates to one high queue there). The
/// expansion is the core `queue_priorities_for_device`, i.e. the same rule a
/// provider applies to a marking that arrives over the command channel.
fn install_queue_priorities(executor: &VulkanExecutor, requested: &[QueuePriority]) -> Result<()> {
    let queues = executor.queue_count();
    let tiers = queue_priorities_for_device(queues, requested);
    executor
        .set_queue_priorities(&tiers)
        .map_err(|error| format!("install queue priorities: {error:?}"))?;
    println!(
        "queue_priority_probe device={} queues={} requested={} installed={} truncated={} padded={}",
        executor.device_name(),
        queues,
        format_queue_tiers(requested),
        format_queue_tiers(&tiers),
        requested.len().saturating_sub(queues),
        queues.saturating_sub(requested.len()),
    );
    Ok(())
}

/// `research/docs/21` §6 on a real device: mark the queues, submit `submissions`
/// mutually independent command buffers through the queue-selecting submit path
/// and report what both observation surfaces see.
///
/// Every command buffer is committed and retired before the next one is
/// recorded, so the device queues are idle at each selection and the policy
/// window — not the in-flight load — decides the tier. Lavapipe exposes one
/// queue: the probe still runs (the single-queue degenerate path) and skips the
/// window contract, which a one-tier device cannot show.
///
/// The probe submits through the deferred object path, which reports one probe
/// call per commit and therefore carries the whole selection sequence. The
/// synchronous paths select through the same policy, but they report one queue
/// per submit rather than a sequence, so the window contract is asserted here.
fn run_queue_priority_probe(executor: &Arc<VulkanExecutor>, submissions: usize) -> Result<()> {
    let installed = executor.queue_priorities();
    if installed.len() != executor.queue_count() {
        return Err("the installed queue priority table does not describe the device".into());
    }
    let observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
        if let Ok(mut sequence) = sink.lock() {
            sequence.push(queue);
        }
    }));
    let provider: Arc<dyn PipelineProvider> = Arc::new(
        VulkanComputeProvider::with_executor(Arc::clone(executor))
            .map_err(|error| format!("create Vulkan provider: {error:?}"))?
            .with_async_execution(true),
    );
    drive_priority_probe(provider, submissions)?;
    executor.clear_enqueue_probe_for_test();
    let sequence = observed
        .lock()
        .map_err(|_| "the enqueue probe sequence is poisoned")?
        .clone();
    report_queue_priority_probe(executor, &installed, &sequence, submissions)
}

/// Submit `submissions` mutually independent command buffers through the
/// queue-selecting object path and assert the landing bytes.
///
/// The provider is a parameter so the same scenario drives a provider in this
/// process and a provider in another one (`--command-socket`); only the local
/// run can read the selection sequence back, because the enqueue probe is
/// host state inside the provider process.
fn drive_priority_probe(provider: Arc<dyn PipelineProvider>, submissions: usize) -> Result<()> {
    let device = objects::Device::new(provider);
    let pipeline = device.compile_pipeline(PipelineCompileRequest {
        entry_name: "read_texture_2d".to_owned(),
        logical_digest: SemanticDigest::new(
            "metal-smoke-fixture-v1",
            b"queue-priority-probe".to_vec(),
        )?,
        source: ShaderSource::SanitizedLl(
            include_str!("../../shaders/kernel_read_texture_2d.ll").to_owned(),
        ),
    })?;
    let mut texels = Vec::with_capacity(64);
    for value in 0..16_u32 {
        texels.extend_from_slice(&value.to_le_bytes());
    }
    let texture = device.new_texture_with_bytes(TextureFormat::R32Uint, 4, 4, texels)?;
    let output = device.new_buffer_with_bytes(vec![0_u8; 64])?;
    let queue = device.new_command_queue();
    for _ in 0..submissions {
        let command = queue.command_buffer();
        {
            let mut encoder = command.compute_command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline)?;
            encoder.set_texture(0, &texture)?;
            encoder.set_buffer(0, &output.view(0, 64)?)?;
            encoder.dispatch_threads(Size::new(1, 1, 1)?, Size::new(1, 1, 1)?)?;
            encoder.end_encoding()?;
        }
        command.commit()?;
        if command.status()? != metal_api_core::CommandBufferStatus::Committed {
            return Err("async commit did not leave the probe command pending".into());
        }
        command.wait_until_completed()?;
    }
    if output.read()?[..4] != 0_u32.to_le_bytes() {
        return Err("the queue priority probe landed unexpected bytes".into());
    }
    Ok(())
}

/// Cross-check both observation surfaces and the §6 assertions, printing one
/// machine-greppable PASS or SKIP line per contract.
fn report_queue_priority_probe(
    executor: &VulkanExecutor,
    installed: &[QueuePriority],
    sequence: &[usize],
    submissions: usize,
) -> Result<()> {
    let queues = installed.len();
    if sequence.len() != submissions {
        return Err(format!(
            "the enqueue probe observed {} of {submissions} submissions",
            sequence.len()
        )
        .into());
    }
    if let Some(index) = sequence.iter().find(|index| **index >= queues) {
        return Err(format!("the enqueue probe observed queue {index} outside the device").into());
    }
    let counts = executor.queue_submission_counts();
    if counts.len() != queues || counts.iter().sum::<usize>() != submissions {
        return Err(format!(
            "queue_submission_counts() reports {counts:?} for {submissions} submissions"
        )
        .into());
    }
    for (index, count) in counts.iter().enumerate() {
        let observed = sequence.iter().filter(|picked| **picked == index).count();
        if *count != observed {
            return Err(format!(
                "queue {index}: queue_submission_counts={count} enqueue_probe={observed}"
            )
            .into());
        }
    }
    // The production observation surfaces agree with the probe queue by queue:
    // selections are counted at enqueue time and retirements at completion
    // time, so the scheduler's allocation is queryable without the test-only
    // probe (`research/docs/21` §6 observation).
    let enqueues = executor.queue_enqueue_counts();
    let completions = executor.queue_completion_counts();
    if enqueues.len() != queues || enqueues.iter().sum::<usize>() != submissions {
        return Err(format!(
            "queue_enqueue_counts() reports {enqueues:?} for {submissions} submissions"
        )
        .into());
    }
    if completions.len() != queues || completions.iter().sum::<usize>() != submissions {
        return Err(format!(
            "queue_completion_counts() reports {completions:?} for {submissions} submissions"
        )
        .into());
    }
    for (index, enqueue_count) in enqueues.iter().enumerate() {
        let observed = sequence.iter().filter(|picked| **picked == index).count();
        if *enqueue_count != observed {
            return Err(format!(
                "queue {index}: queue_enqueue_counts={enqueue_count} enqueue_probe={observed}"
            )
            .into());
        }
        if completions[index] != counts[index] {
            return Err(format!(
                "queue {index}: queue_completion_counts={} queue_submission_counts={}",
                completions[index], counts[index]
            )
            .into());
        }
    }

    let policy = QueueSchedulingPolicy::default();
    let tier_of = |index: usize| installed[index];
    let share = |tier: QueuePriority| {
        sequence
            .iter()
            .filter(|index| tier_of(**index) == tier)
            .count()
    };
    println!(
        "queue_priority_probe sequence={}",
        sequence
            .iter()
            .map(|index| match tier_of(*index) {
                QueuePriority::Low => 'L',
                QueuePriority::Default => 'D',
                QueuePriority::High => 'H',
            })
            .collect::<String>()
    );
    println!(
        "queue_priority_probe index_counts={}",
        counts
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "queue_priority_probe tier_counts=high={} default={} low={}",
        share(QueuePriority::High),
        share(QueuePriority::Default),
        share(QueuePriority::Low)
    );
    println!(
        "queue_priority_probe enqueue_counts={}",
        enqueues
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "queue_priority_probe completion_counts={}",
        completions
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "PASS queue_priority_probe submissions={submissions} queues={queues} \
         probe_matches_counts=exact enqueue_matches=exact completion_matches=exact \
         writeback=exact"
    );

    let window = usize::try_from(policy.window())?;
    let present = installed.iter().collect::<BTreeSet<_>>().len();
    if queues < window || present < 3 {
        println!(
            "SKIP queue_priority_probe_window reason=insufficient_queues queues={queues} \
             tiers_present={present} window={window}"
        );
        return Ok(());
    }
    let windows = submissions / window;
    if windows == 0 {
        return Err(
            format!("{submissions} submissions do not fill one {window}-slot window").into(),
        );
    }
    let tiers: Vec<QueuePriority> = sequence.iter().map(|index| tier_of(*index)).collect();
    let mut streak = 0_usize;
    let mut longest = 0_usize;
    for tier in &tiers {
        streak = if *tier == QueuePriority::High {
            streak + 1
        } else {
            0
        };
        longest = longest.max(streak);
    }
    let limit = policy.high_priority_streak_limit() as usize;
    if longest > limit {
        return Err(format!("the high tier ran {longest} times in a row, limit {limit}").into());
    }
    let mut low_per_window_min = usize::MAX;
    for window_tiers in tiers.chunks(window).take(windows) {
        let lows = window_tiers
            .iter()
            .filter(|tier| **tier == QueuePriority::Low)
            .count();
        low_per_window_min = low_per_window_min.min(lows);
    }
    if low_per_window_min == 0 {
        return Err("a window starved the low tier".into());
    }
    let expected_high = policy.high_weight() as usize * windows;
    let expected_default = policy.medium_weight() as usize * windows;
    let high = share(QueuePriority::High);
    let default = share(QueuePriority::Default);
    if high != expected_high || default != expected_default {
        return Err(format!(
            "window shares high={high} default={default} low={} do not match {expected_high}:{expected_default}",
            share(QueuePriority::Low)
        )
        .into());
    }
    println!(
        "PASS queue_priority_probe_window windows={windows} high={high} default={default} \
         low={} max_high_streak={longest} limit={limit} low_per_window_min={low_per_window_min}",
        share(QueuePriority::Low)
    );
    Ok(())
}

/// Log one selection sequence as the tier letter of each selected queue.
#[cfg(unix)]
fn format_queue_sequence(installed: &[QueuePriority], sequence: &[usize]) -> String {
    sequence
        .iter()
        .map(|index| match installed[*index] {
            QueuePriority::Low => 'L',
            QueuePriority::Default => 'D',
            QueuePriority::High => 'H',
        })
        .collect()
}

/// `--queue-priority-probe --command-socket <path>`: the probe scenario with
/// the provider in another process.
///
/// The owner sets the marking, drives the same submissions the in-process probe
/// drives, and compares the provider's own report of the tiers it read against
/// the table its marking installed. The enqueue probe is host state inside the
/// provider process, so the provider prints its selection sequence itself; the
/// owner asserts the two sides agree before this returns.
#[cfg(unix)]
fn run_remote_queue_priority_probe(
    command_path: &Path,
    requested: &[QueuePriority],
    submissions: usize,
) -> Result<()> {
    use metal_api_ipc::command::unix::UnixListenerCommandTransport;

    let listener = UnixListenerCommandTransport::bind(command_path)?;
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--queue-priority-child")
        .arg(command_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("the queue priority child stdout was not piped")?;
    let mut lines = BufReader::new(stdout).lines();

    let transport = match listener.accept() {
        Ok(transport) => transport,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("accept the queue priority child: {error}").into());
        }
    };
    let remote = RemoteProvider::connect(transport)?;
    // The response is the provider's own expansion of the marking, so this is
    // the table the scheduler in the child process reads.
    let installed = remote
        .set_queue_priorities(requested)
        .map_err(|error| format!("install remote queue priorities: {error:?}"))?;
    println!(
        "queue_priority_probe owner=remote child=command-socket sent={} installed={} queues={}",
        format_queue_tiers(requested),
        format_queue_tiers(&installed),
        installed.len()
    );
    // Dropping the device closes the command channel, which is how the child
    // learns that the session is over.
    drive_priority_probe(Arc::new(remote) as Arc<dyn PipelineProvider>, submissions)?;

    let mut child_lines = Vec::new();
    for line in &mut lines {
        child_lines.push(line?);
    }
    let status = child.wait()?;
    let _ = std::fs::remove_file(command_path);
    for line in &child_lines {
        println!("child: {line}");
    }
    if !status.success() {
        return Err(format!("the queue priority child exited with {status}").into());
    }

    // Owner-side assertion of the provider-side read: the child reports the
    // table it installed, and it has to be the table this marking installed.
    let report = child_lines
        .iter()
        .find(|line| line.starts_with("queue_priority_child read="))
        .ok_or("the queue priority child did not report the tiers it read")?;
    let field = |name: &str| {
        report
            .split_whitespace()
            .find_map(|part| part.strip_prefix(name))
    };
    let read = field("read=").ok_or("the child report has no read= field")?;
    let queues = field("queues=").ok_or("the child report has no queues= field")?;
    if read != format_queue_tiers(&installed) || queues != installed.len().to_string() {
        return Err(format!(
            "the provider read {read} on {queues} queues, the owner installed {} on {}",
            format_queue_tiers(&installed),
            installed.len()
        )
        .into());
    }
    if !child_lines
        .iter()
        .any(|line| line.starts_with("PASS queue_priority_child"))
    {
        return Err("the queue priority child did not report a passing observation".into());
    }
    println!(
        "PASS queue_priority_probe_remote sent={} read={} queues={} submissions={submissions} \
         probe_matches_counts=exact writeback=exact",
        format_queue_tiers(requested),
        read,
        installed.len()
    );
    Ok(())
}

/// `--queue-priority-probe --command-socket <path>` on a platform without Unix
/// domain sockets.
#[cfg(not(unix))]
fn run_remote_queue_priority_probe(
    _command_path: &Path,
    _requested: &[QueuePriority],
    _submissions: usize,
) -> Result<()> {
    Err("--queue-priority-probe --command-socket requires Unix domain sockets".into())
}

/// Entry point for `--queue-priority-child`.
fn run_queue_priority_child_mode(path: std::ffi::OsString) -> Result<()> {
    #[cfg(unix)]
    {
        run_queue_priority_child(&path)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("--queue-priority-child requires Unix domain sockets".into())
    }
}

/// `--queue-priority-child <command-socket>`: the provider half of the remote
/// probe.
///
/// It owns a Vulkan device, serves the command channel, and once the owner
/// closes it reports the queue table it read, the selection sequence it
/// observed and the two provider-side observation surfaces it cross-checked.
/// This line is the provider-side half of the cross-process evidence.
#[cfg(unix)]
fn run_queue_priority_child(command_path: &std::ffi::OsStr) -> Result<()> {
    let executor = Arc::new(
        VulkanExecutor::new().map_err(|error| format!("create Vulkan executor: {error:?}"))?,
    );
    let observed = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    executor.set_enqueue_probe_for_test(Arc::new(move |queue| {
        if let Ok(mut sequence) = sink.lock() {
            sequence.push(queue);
        }
    }));
    let provider = VulkanComputeProvider::with_executor(Arc::clone(&executor))
        .map_err(|error| format!("create Vulkan provider: {error:?}"))?
        .with_async_execution(true);
    let mut transport = command_unix::connect(command_path)?;
    serve_provider_unix(&provider, &mut transport)?;
    drop(transport);
    drop(provider);
    executor.clear_enqueue_probe_for_test();

    let installed = executor.queue_priorities();
    let counts = executor.queue_submission_counts();
    let enqueues = executor.queue_enqueue_counts();
    let completions = executor.queue_completion_counts();
    let sequence = observed
        .lock()
        .map_err(|_| "the enqueue probe sequence is poisoned")?
        .clone();
    if sequence.iter().any(|index| *index >= installed.len()) {
        return Err("the enqueue probe observed a queue outside the device".into());
    }
    let tiers: Vec<QueuePriority> = sequence.iter().map(|index| installed[*index]).collect();
    let share = |tier: QueuePriority| tiers.iter().filter(|seen| **seen == tier).count();
    println!(
        "queue_priority_child read={} queues={} submissions={} sequence={} index_counts={} \
         enqueue_counts={} completion_counts={} tier_counts=high={} default={} low={}",
        format_queue_tiers(&installed),
        installed.len(),
        sequence.len(),
        format_queue_sequence(&installed, &sequence),
        counts
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(","),
        enqueues
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(","),
        completions
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(","),
        share(QueuePriority::High),
        share(QueuePriority::Default),
        share(QueuePriority::Low)
    );

    // The two provider-side observation surfaces have to agree, exactly as the
    // in-process probe requires.
    if counts.len() != installed.len() || counts.iter().sum::<usize>() != sequence.len() {
        return Err(format!(
            "queue_submission_counts() reports {counts:?} for {} submissions",
            sequence.len()
        )
        .into());
    }
    for (index, count) in counts.iter().enumerate() {
        let seen = sequence.iter().filter(|picked| **picked == index).count();
        if *count != seen {
            return Err(format!(
                "queue {index}: queue_submission_counts={count} enqueue_probe={seen}"
            )
            .into());
        }
    }
    if enqueues.len() != installed.len() || enqueues.iter().sum::<usize>() != sequence.len() {
        return Err(format!(
            "queue_enqueue_counts() reports {enqueues:?} for {} submissions",
            sequence.len()
        )
        .into());
    }
    if completions.len() != installed.len() || completions.iter().sum::<usize>() != sequence.len() {
        return Err(format!(
            "queue_completion_counts() reports {completions:?} for {} submissions",
            sequence.len()
        )
        .into());
    }
    for (index, count) in counts.iter().enumerate() {
        let seen = sequence.iter().filter(|picked| **picked == index).count();
        if enqueues[index] != seen {
            return Err(format!(
                "queue {index}: queue_enqueue_counts={} enqueue_probe={seen}",
                enqueues[index]
            )
            .into());
        }
        if completions[index] != *count {
            return Err(format!(
                "queue {index}: queue_completion_counts={} queue_submission_counts={count}",
                completions[index]
            )
            .into());
        }
    }
    println!(
        "PASS queue_priority_child queues={} submissions={} read={} probe_matches_counts=exact \
         enqueue_matches=exact completion_matches=exact",
        installed.len(),
        sequence.len(),
        format_queue_tiers(&installed)
    );
    Ok(())
}

fn create_provider(
    backend: Backend,
    async_execution: bool,
    queue_priorities: Option<&[QueuePriority]>,
) -> Result<ProviderHandles> {
    match backend {
        Backend::Vulkan => {
            let executor = VulkanExecutor::new()
                .map_err(|error| format!("create Vulkan executor: {error:?}"))?;
            if let Some(requested) = queue_priorities {
                install_queue_priorities(&executor, requested)?;
            }
            let provider = Arc::new(
                VulkanComputeProvider::with_executor(Arc::clone(&executor))
                    .map_err(|error| format!("create Vulkan provider: {error:?}"))?
                    .with_async_execution(async_execution),
            );
            let name = provider.device_name().to_owned();
            // The render rail is a concrete-context entry point
            // (`register_render_pipeline` is not part of `PipelineProvider`), so
            // the handle is kept beside the trait object.
            let leases = LeaseHandles {
                staged: Arc::clone(&provider) as Arc<dyn LeaseImporter>,
                borrowed: Arc::clone(&provider) as Arc<dyn NoCopyLeaseImporter>,
            };
            Ok((
                Arc::clone(&provider) as Arc<dyn PipelineProvider>,
                name,
                CopyCounters::Vulkan {
                    executor,
                    provider: Arc::clone(&provider),
                },
                RenderRegistrar::Vulkan(provider),
                leases,
            ))
        }
        Backend::NativeMetalProvider => {
            #[cfg(target_os = "macos")]
            {
                let provider = Arc::new(
                    NativeMetalProvider::new()
                        .map_err(|error| format!("create native Metal provider: {error:?}"))?
                        .with_async_execution(async_execution),
                );
                let name = provider.device_name().to_owned();
                let leases = LeaseHandles {
                    staged: Arc::clone(&provider) as Arc<dyn LeaseImporter>,
                    borrowed: Arc::clone(&provider) as Arc<dyn NoCopyLeaseImporter>,
                };
                Ok((
                    Arc::clone(&provider) as Arc<dyn PipelineProvider>,
                    name,
                    CopyCounters::Native(Arc::clone(&provider)),
                    RenderRegistrar::Native(provider),
                    leases,
                ))
            }
            #[cfg(not(target_os = "macos"))]
            Err("native-metal-provider requires macOS".into())
        }
    }
}

/// Register the reviewed render pipeline on one trace rail's concrete context.
///
/// The render rail is a concrete-context entry point (`register_render_pipeline`
/// is not part of `PipelineProvider`), and each rail is the only owner of its
/// reviewed source pair: the Vulkan rail takes the SPIR-V stages pinned above,
/// the native rail the reviewed MSL module
/// (`crates/metal-api-native/src/render.rs::REVIEWED_SOURCE`). Both hand back
/// the trace-table entry a render pass has to name, under the same
/// caller-issued fixture identity both rails' `compile` siblings take, so the
/// digest names the suite and the pipeline rather than a rail.
fn register_render_pipeline(
    registrar: &RenderRegistrar,
    identity: &str,
    case: &RenderCase,
    geometry: RenderGeometry,
    formats: &[AttachmentFormat],
) -> Result<CompiledComputePipeline> {
    let attachment_count = formats.len();
    let logical_digest = SemanticDigest::new(
        "suite-sha256-entry-v1",
        format!("{identity}:offscreen_render_pipeline:{attachment_count}").into_bytes(),
    )?;
    // The pipeline-level half of a stage-buffer case's declaration
    // (`research/docs/23` §3.3, v83-v86). Every other geometry declares no
    // slots, so its contract's list stays empty and its bytes are unchanged.
    let stage_buffers = if geometry == RenderGeometry::StageBuffers {
        stage_buffer_declarations(case)?
    } else {
        Vec::new()
    };
    // The registration names the reviewed pair for the case's geometry: the
    // Vulkan rail compiles the vertex stage's SPIR-V, the native rail its MSL
    // sibling, and both re-run their own review gate over the pair
    // (`research/docs/23` §3.3).
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let (vulkan_entries, msl_entries, vulkan_stages, layout) = match geometry {
        RenderGeometry::Milestone => (
            (RENDER_VERTEX_ENTRY, RENDER_FRAGMENT_ENTRY),
            (RENDER_MSL_VERTEX_ENTRY, RENDER_MSL_FRAGMENT_ENTRY),
            (RENDER_VERTEX_SPV, RENDER_FRAGMENT_SPV),
            VertexLayout::None,
        ),
        RenderGeometry::IndexedQuad => match (attachment_count, formats) {
            (1, [AttachmentFormat::R32Float]) => (
                (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
                (QUAD_MSL_VERTEX_ENTRY, R32F_MSL_FRAGMENT_ENTRY),
                (QUAD_VERTEX_SPV, QUAD_R32F_FRAGMENT_SPV),
                reviewed_quad_layout(),
            ),
            (1, _) => (
                (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
                (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_FRAGMENT_ENTRY),
                (QUAD_VERTEX_SPV, QUAD_FRAGMENT_SPV),
                reviewed_quad_layout(),
            ),
            (2, _) => (
                (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
                (QUAD_MSL_VERTEX_ENTRY, DUAL_MSL_FRAGMENT_ENTRY),
                (QUAD_VERTEX_SPV, QUAD_DUAL_FRAGMENT_SPV),
                reviewed_quad_layout(),
            ),
            (3, _) => (
                (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
                (QUAD_MSL_VERTEX_ENTRY, TRIPLE_MSL_FRAGMENT_ENTRY),
                (QUAD_VERTEX_SPV, QUAD_TRIPLE_FRAGMENT_SPV),
                reviewed_quad_layout(),
            ),
            (4, _) => (
                (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
                (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_QUAD_FRAGMENT_ENTRY),
                (QUAD_VERTEX_SPV, QUAD_QUAD_FRAGMENT_SPV),
                reviewed_quad_layout(),
            ),
            _ => return Err("the reviewed MRT shapes are one, two and four attachments".into()),
        },
        // The instanced fixture owns a module pair of its own
        // (`research/docs/23` §3.3, v31): the vertex stage reads the
        // per-instance tint and shifts each instance's copy with
        // `instance_id`, and the fragment stage stores the forwarded tint. The
        // solid modules cannot stand in for either half.
        RenderGeometry::InstancedPair => (
            (INSTANCED_VERTEX_ENTRY, INSTANCED_FRAGMENT_ENTRY),
            (INSTANCED_MSL_VERTEX_ENTRY, INSTANCED_MSL_FRAGMENT_ENTRY),
            (INSTANCED_VERTEX_SPV, INSTANCED_FRAGMENT_SPV),
            reviewed_instanced_layout(),
        ),
        // The base-vertex shape is the reviewed quad's module and layout: the
        // offset is draw state, not pipeline state, so the pair does not change
        // (`research/docs/23` §3.3, v34).
        RenderGeometry::BaseVertexQuad => (
            (QUAD_VERTEX_ENTRY, QUAD_FRAGMENT_ENTRY),
            (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_FRAGMENT_ENTRY),
            (QUAD_VERTEX_SPV, QUAD_FRAGMENT_SPV),
            reviewed_quad_layout(),
        ),
        // The depth pair carries its own reviewed module
        // (`research/docs/23` §3.3, v36): the position's z decides which
        // triangle survives, and the tint varying is what makes the surviving
        // one visible, so neither the solid nor the instanced module can stand
        // in for it.
        // The zero-colour-attachment depth pass (`research/docs/23` §3.3, v46)
        // is the depth pair's shape with no colour target at all: the same
        // vertex stage, paired with the reviewed fragment stage that declares
        // no output.
        RenderGeometry::DepthPair if formats.is_empty() => (
            (DEPTH_VERTEX_ENTRY, DEPTH_ONLY_FRAGMENT_ENTRY),
            (DEPTH_ONLY_MSL_VERTEX_ENTRY, DEPTH_ONLY_MSL_FRAGMENT_ENTRY),
            (DEPTH_VERTEX_SPV, DEPTH_ONLY_FRAGMENT_SPV),
            reviewed_depth_layout(),
        ),
        RenderGeometry::DepthPair => (
            (DEPTH_VERTEX_ENTRY, DEPTH_FRAGMENT_ENTRY),
            (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
            (DEPTH_VERTEX_SPV, DEPTH_FRAGMENT_SPV),
            reviewed_depth_layout(),
        ),
        // The blend shape compiles the same reviewed pair module: the tint's
        // alpha is what the blend state scales, and the oversize triangle
        // covers the attachment so the stored texels are the blend's own
        // result (`research/docs/23` §3.3, v40).
        RenderGeometry::BlendTriangle => (
            (DEPTH_VERTEX_ENTRY, DEPTH_FRAGMENT_ENTRY),
            (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
            (DEPTH_VERTEX_SPV, DEPTH_FRAGMENT_SPV),
            reviewed_depth_layout(),
        ),
        // The cull shape compiles the same reviewed pair module: the positions
        // and tints are the reviewed ones, and the culling state is what picks
        // which triangle survives (`research/docs/23` §3.3, v39).
        RenderGeometry::CullPair => (
            (DEPTH_VERTEX_ENTRY, DEPTH_FRAGMENT_ENTRY),
            (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
            (DEPTH_VERTEX_SPV, DEPTH_FRAGMENT_SPV),
            reviewed_depth_layout(),
        ),
        // The render sampler owns a module pair of its own
        // (`research/docs/23` §3.3, v70): the vertex stage carries the
        // geometry's normalised coordinate and the fragment stage samples the
        // pass's own texture binding, so neither the solid stage nor the
        // milestone vertex stage can stand in for either half.
        RenderGeometry::SampledTexture => (
            (SAMPLED_QUAD_VERTEX_ENTRY, SAMPLED_QUAD_FRAGMENT_ENTRY),
            (SAMPLED_MSL_VERTEX_ENTRY, SAMPLED_MSL_FRAGMENT_ENTRY),
            (SAMPLED_QUAD_VERT_SPV, SAMPLED_UNORM8_FRAG_SPV),
            VertexLayout::None,
        ),
        // The reviewed stage-buffer pair (`research/docs/23` §3.3, v83): the
        // vertex stage's three positions and the fragment stage's one tint,
        // each read from the slot the case declares. The modules fix their own
        // descriptor slots (the vertex stage's bindings in set 1, the fragment
        // stage's in set 2), so the pair carries no vertex layout at all.
        RenderGeometry::StageBuffers => (
            (STAGE_BUFFER_POSITIONS_ENTRY, STAGE_BUFFER_TINT_ENTRY),
            (
                STAGE_BUFFER_MSL_VERTEX_ENTRY,
                STAGE_BUFFER_MSL_FRAGMENT_ENTRY,
            ),
            (STAGE_BUFFER_POSITIONS_SPV, STAGE_BUFFER_TINT_SPV),
            VertexLayout::None,
        ),
    };
    let registered = match registrar {
        RenderRegistrar::Vulkan(vulkan) => vulkan.register_render_pipeline(RenderPipelineRequest {
            contract: RenderPipelineContract {
                stage_buffers: stage_buffers.clone(),
                vertex_entry: vulkan_entries.0.to_owned(),
                fragment_entry: vulkan_entries.1.to_owned(),
                color_formats: formats.to_vec(),
                vertex_layout: layout.clone(),
            },
            vertex_spirv: vulkan_stages.0.to_vec(),
            fragment_spirv: vulkan_stages.1.to_vec(),
            logical_digest,
        }),
        #[cfg(target_os = "macos")]
        RenderRegistrar::Native(native) => {
            // The native rail compiles the MSL module itself, so its contract
            // names that module's own stage entries. The registration re-runs
            // the review gate and refuses any other pair with
            // `native_render_source_not_reviewed`, exactly as an unreviewed MSL
            // fixture is refused.
            native.register_render_pipeline(NativeRenderPipelineRequest {
                contract: RenderPipelineContract {
                    stage_buffers: stage_buffers.clone(),
                    vertex_entry: msl_entries.0.to_owned(),
                    fragment_entry: msl_entries.1.to_owned(),
                    color_formats: formats.to_vec(),
                    vertex_layout: layout.clone(),
                },
                logical_digest,
            })
        }
    }
    .map_err(|error| format!("register render pipeline: {error:?}"))?;
    Ok(registered)
}

/// Register one render case's pipeline, whichever arm its stages belong to
/// (`research/docs/23` §3.3, v83-v86).
///
/// A reviewed case hands its geometry to [`register_render_pipeline`], which
/// mints the rail's own reviewed pair. A case that carries `translated_stages`
/// has no reviewed pair: the Vulkan rail translates its two AIR modules itself
/// and registers the result, so the descriptor slots come from each module's
/// own reflection and the contract's declarations are what the rail pairs that
/// reflection with.
fn register_render_case_pipeline(
    registrar: &RenderRegistrar,
    counters: &CopyCounters,
    identity: &str,
    case: &RenderCase,
    directory: &Path,
    geometry: RenderGeometry,
    formats: &[AttachmentFormat],
) -> Result<CompiledComputePipeline> {
    if case.translated_stages.is_none() {
        return register_render_pipeline(registrar, identity, case, geometry, formats);
    }
    match (registrar, counters) {
        (RenderRegistrar::Vulkan(vulkan), CopyCounters::Vulkan { executor, .. }) => {
            register_translated_stage_buffer_pipeline(
                vulkan, executor, identity, case, directory, formats,
            )
        }
        // Only the Vulkan trace rail translates AIR, and `validate_render_case`
        // marks a translated case for that rail alone, so this arm is the
        // native build's honest answer rather than a path a suite can reach.
        #[cfg(target_os = "macos")]
        _ => Err(format!(
            "render case {}: a translated stage-buffer case runs on the Vulkan trace rail, and \
             this rail translates no AIR to execute",
            case.id
        )
        .into()),
    }
}

/// Translate and register the two AIR stages of a stage-buffer case on the
/// Vulkan trace rail (`research/docs/23` §3.3, v84/v86).
///
/// Each stage translates into its own descriptor set — the vertex stage's
/// `[[buffer(N)]]` arguments in set 1, the fragment stage's in set 2 — the
/// arrangement the rail's reviewed stage-buffer pair binds. The two stages'
/// Metal buffer namespaces are independent, so one set cannot carry both
/// without one stage's slot overwriting the other's, and the rail refuses that
/// collision by name.
///
/// The rail's own registration gate pairs each reflection with the contract's
/// declarations field by field, so a module that writes a slot the case
/// declares read-only, or reaches past a static ceiling, is refused by name
/// rather than executed with the disagreement dropped.
fn register_translated_stage_buffer_pipeline(
    vulkan: &VulkanComputeProvider,
    executor: &Arc<VulkanExecutor>,
    identity: &str,
    case: &RenderCase,
    directory: &Path,
    formats: &[AttachmentFormat],
) -> Result<CompiledComputePipeline> {
    let where_ = format!("render case {}", case.id);
    let translated = case
        .translated_stages
        .as_ref()
        .ok_or("a translated stage-buffer case carries its two AIR modules")?;
    let device = metal_api_core::Device::new(
        Arc::clone(executor) as Arc<dyn metal_api_core::ComputeExecutor>
    );
    let policy = executor.spirv_feature_policy();
    let mut stages = Vec::with_capacity(2);
    for (stage, source, entry, set) in [
        (
            RenderStage::Vertex,
            &translated.vertex,
            case.vertex_entry.as_str(),
            STAGE_BUFFER_VERTEX_SET,
        ),
        (
            RenderStage::Fragment,
            &translated.fragment,
            case.fragment_entry.as_str(),
            STAGE_BUFFER_FRAGMENT_SET,
        ),
    ] {
        let air = String::from_utf8(verified_source(directory, source)?)?;
        let library = device
            .new_library_with_air(air.as_str())
            .map_err(|error| format!("{where_}: load translated stage AIR: {error:?}"))?;
        let function = library.function(entry).map_err(|error| {
            format!("{where_}: unknown translated stage entry {entry}: {error:?}")
        })?;
        let translated_stage = TranslatedRenderStage::translate_with_policy_and_layout(
            stage,
            &function,
            policy,
            DescriptorLayout {
                set,
                ..DescriptorLayout::default()
            },
        )
        .map_err(|error| format!("{where_}: translate {entry}: {error:?}"))?;
        stages.push(translated_stage);
    }
    let [vertex, fragment]: [TranslatedRenderStage; 2] = stages
        .try_into()
        .map_err(|_| format!("{where_}: two translated stages"))?;
    let contract = RenderPipelineContract {
        vertex_entry: case.vertex_entry.clone(),
        fragment_entry: case.fragment_entry.clone(),
        color_formats: formats.to_vec(),
        vertex_layout: VertexLayout::None,
        stage_buffers: stage_buffer_declarations(case)?,
    };
    let logical_digest = SemanticDigest::new(
        "suite-sha256-entry-v1",
        format!(
            "{identity}:translated_render_pipeline:{}:{}:{}",
            case.vertex_entry,
            case.fragment_entry,
            formats.len()
        )
        .into_bytes(),
    )?;
    let registered = vulkan.register_translated_render_pipeline(TranslatedRenderPipelineRequest {
        contract,
        vertex,
        fragment,
        logical_digest,
    });
    registered
        .map_err(|error| format!("{where_}: register translated render pipeline: {error:?}").into())
}

/// Retire the registration [`register_render_pipeline`] minted, on the context
/// that owns it.
fn release_render_pipeline(
    registrar: &RenderRegistrar,
    pipeline: &CompiledComputePipeline,
) -> Result<()> {
    match registrar {
        RenderRegistrar::Vulkan(vulkan) => {
            vulkan
                .release_render_pipeline(pipeline)
                .map_err(|error| format!("release render pipeline: {error:?}"))?;
            Ok(())
        }
        #[cfg(target_os = "macos")]
        RenderRegistrar::Native(native) => {
            native
                .release_render_pipeline(pipeline)
                .map_err(|error| format!("release render pipeline: {error:?}"))?;
            Ok(())
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    schema_version: u32,
    suite: String,
    guard_byte: u8,
    cases: Vec<Case>,
    /// Offscreen render cases (`research/docs/23` §1.2). A render case is not a
    /// compute case: its observable is one colour attachment's texels, so it
    /// lives in its own array and names the compute case whose pass declares
    /// the attachment view.
    #[serde(default)]
    render_cases: Vec<RenderCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    entry: String,
    grid: [u64; 3],
    local: [u64; 3],
    #[serde(default)]
    air_encoding: AirEncoding,
    air: Source,
    metal: Source,
    buffers: Vec<Buffer>,
    /// Sampled textures bound by every pass of the case (v11 and later).
    /// `research/docs/18` step 1.
    #[serde(default)]
    textures: Vec<Texture>,
    expected_writebacks: Vec<ExpectedWriteback>,
    dispatches: Option<Vec<CaseDispatch>>,
    programs: Option<Vec<CaseProgram>>,
    command_buffers: Option<Vec<Vec<usize>>>,
    /// Optional heap section (`research/docs/25` §4.2). A case that carries it
    /// also carries `capture_rails`, because only a rail that declares heap
    /// support can report the placement observation; the capture tool skips a
    /// heap case on any rail its marker does not name.
    #[serde(default)]
    heap: Option<HeapCase>,
    #[serde(default)]
    capture_rails: Option<Vec<String>>,
    /// Optional indirect-command section (`research/docs/25` §4.3). A compute
    /// case that carries it replays its single dispatch from one encoded
    /// command instead of `vkCmdDispatch`.
    #[serde(default)]
    icb: Option<IcbCase>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeapCase {
    size: u64,
    storage_mode: String,
    allows_aliasing: bool,
    placements: Vec<HeapPlacementCase>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeapPlacementCase {
    allocation: u64,
    offset: u64,
    byte_size: u64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Texture {
    binding: u32,
    allocation: u64,
    view: u64,
    width: u64,
    height: u64,
    format: String,
    access: String,
    initial_hex: String,
}

/// One offscreen render case (`research/docs/23` §1.2, §5.1).
///
/// The shape is a whitelist rather than a per-case table: the first render
/// increment has exactly one render shape, so the shape *is* the review and a
/// fixture cannot widen it by renaming a case.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderCase {
    id: String,
    /// The compute case of this suite whose pass declares the attachment view.
    /// The render trace replays that pass, so the target resolves against a
    /// resource table the trace itself carries (`research/docs/23` §3.6).
    declaring_case: String,
    vertex_entry: String,
    fragment_entry: String,
    /// The reviewed MSL module the case's stages were written as, or `None`
    /// for a *translated* case (`research/docs/23` §3.3, v84/v86): a case whose
    /// stages are translated from AIR has no canonical MSL sibling yet, so it
    /// pins its two AIR modules through `translated_stages` instead. Every
    /// reviewed case still pin its MSL module, and
    /// [`reviewed_stage_buffer_geometry`] refuses the two spellings together.
    #[serde(default)]
    metal: Option<Source>,
    vertices: u64,
    viewport: [u64; 4],
    /// The scissor rectangle the pass clips to, in `[x, y, width, height]`, or
    /// absent for "the whole viewport" (`research/docs/23` §3.3, v29).
    #[serde(default)]
    scissor: Option<[u64; 4]>,
    /// The first render increments' single attachment, or `None` for an MRT
    /// case that declares `attachments` instead. Exactly one of the two forms
    /// is present, and [`validate_render_case`] pins that before any rail runs.
    #[serde(default)]
    attachment: Option<RenderAttachmentDefinition>,
    /// The MRT case's attachment list, in location order; mutually exclusive
    /// with `attachment`. Every entry carries its own `expected_hex`.
    #[serde(default)]
    attachments: Option<Vec<RenderAttachmentDefinition>>,
    /// The single-attachment case's expectation. An MRT case leaves this
    /// absent and spells the expectation on each attachment entry instead.
    #[serde(default)]
    expected_hex: Option<String>,
    /// The rule the single attachment's own bytes follow, when the case cannot
    /// spell them (R5a, `research/docs/23` §73). It is the reviewed sampling
    /// shape's identity rule restated: the fragment stage copies the bound
    /// texture texel for texel, so the expectation is the same rule the texture
    /// carries, and the case has to name it here rather than leave the
    /// attachment's expectation implicit.
    #[serde(default)]
    expected_rule: Option<String>,
    /// The rectangles of a rule-expected attachment the capture reports
    /// verbatim (R5a, `research/docs/23` §73), in the order the comparator
    /// checks them. Required exactly when `expected_rule` is present: the
    /// digest covers the whole plane, and these windows are where the bytes
    /// themselves stay checkable.
    #[serde(default)]
    readback_windows: Option<Vec<ReadbackWindowDefinition>>,
    /// The capture backends this case is executable on. A rail in this list has
    /// to report the case; a rail outside it has to omit it.
    capture_rails: Vec<String>,
    /// Optional present action (`research/docs/24` §6 Step 3). When present the
    /// render case hands its own attachment on as a present target; the
    /// expected `acquire`/`present` counts are the fixture's own assertion,
    /// checked against the provider's counters when the case runs.
    #[serde(default)]
    present: Option<PresentDefinition>,
    /// Optional indirect-command section (`research/docs/25` §4.3). A case
    /// that carries it replays its draw from one encoded command instead of a
    /// direct draw; only the rails its marker names execute it.
    #[serde(default)]
    icb: Option<IcbCase>,
    /// Optional vertex layout (`research/docs/23` §3.3): the streams the
    /// pipeline's vertex input state describes. Absent means the milestone's
    /// `vertex_id` triangle, which binds no stream at all.
    #[serde(default)]
    vertex_layout: Option<VertexLayoutDefinition>,
    /// The caller-held vertex streams the pass binds, in binding order. Each
    /// one declares its own bytes, so a render-only trace needs no compute pass
    /// to carry them (`research/docs/23` §3.6).
    #[serde(default)]
    vertex_buffers: Vec<RenderInputDefinition>,
    /// The stage-buffer slots this case's pipeline declares and its pass binds
    /// (`research/docs/23` §3.3, v83-v86): one entry per slot, carrying the
    /// pipeline's declaration beside the pass's view. Empty for every pre-v83
    /// case, which declares and binds none.
    #[serde(default)]
    stage_buffers: Vec<StageBufferDefinition>,
    /// The two AIR stages a *translated* pipeline is registered from
    /// (`research/docs/23` §3.3, v84): present only for a case whose stages are
    /// not one of the reviewed pairs. Absent for every case the reviewed
    /// registry serves, which is every pre-R9f render case.
    #[serde(default)]
    translated_stages: Option<TranslatedStagesDefinition>,
    /// The index buffer the draw runs through, when the case is indexed.
    #[serde(default)]
    indices: Option<IndexInputDefinition>,
    /// Instances the draw runs (`research/docs/23` §3.3, v31). Absent means the
    /// single instance every pre-v31 case draws; the reviewed instanced fixture
    /// declares exactly two.
    #[serde(default = "default_instance_count")]
    instance_count: u64,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34). Absent means `0`, the shape every pre-v34 case draws; the reviewed
    /// base-vertex fixture declares exactly `1` over its five-vertex stream.
    #[serde(default)]
    base_vertex: u64,
    /// The depth attachment the pass opens, or absent for a pass with no depth
    /// surface (`research/docs/23` §3.3, v36). The attachment is rail-owned:
    /// the case states its shape, not a resource identity.
    #[serde(default)]
    depth: Option<DepthAttachmentDefinition>,
    /// The pass's depth state, or absent for "the attachment exists and
    /// nothing tests it" (`research/docs/23` §3.3, v36).
    #[serde(default)]
    depth_test: Option<DepthTestDefinition>,
    /// The rail-owned stencil attachment the pass opens, or absent for a pass
    /// with no stencil surface (`research/docs/23` §3.3, v47).
    #[serde(default)]
    stencil: Option<StencilAttachmentDefinition>,
    /// The pass's stencil state, or absent for "the attachment exists and
    /// nothing tests it".
    #[serde(default)]
    stencil_test: Option<StencilTestDefinition>,
    /// The pass-wide multisample state, or absent for the single-sample raster
    /// every pre-v51 case runs (`research/docs/23` §3.3, v51). The reviewed
    /// fixture states the four-sample raster and observes the resolve of the
    /// fragment output and the load's own colour in the attachment view.
    #[serde(default)]
    multisample: Option<MultisampleDefinition>,
    /// The sampled textures the fragment stage reads, in binding order
    /// (`research/docs/23` §3.3, v70), or absent for the pre-v70 pass that
    /// samples nothing. The reviewed sampling shape is exactly one
    /// `rgba8_unorm` surface whose extent equals the render area's own, so the
    /// expectation is its own uploaded texels.
    #[serde(default)]
    fragment_textures: Option<Vec<FragmentTextureDefinition>>,
    /// The depth resolve a stored multisampled depth surface states
    /// (`research/docs/23` §3.3, v57), or absent for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose depth attachment
    /// is stored; the reviewed fixture states the `Sample0` filter the
    /// Lavapipe device reports.
    #[serde(default)]
    depth_resolve: Option<DepthResolveDefinition>,
    /// The stencil resolve a stored multisampled stencil surface states
    /// (`research/docs/23` §3.3, v60), or absent for a pass that resolves
    /// nothing. Only legal beside a multisample raster whose stencil
    /// attachment is stored; the `depth_resolved_sample` filter additionally
    /// requires the depth resolve the selected sample comes from.
    #[serde(default)]
    stencil_resolve: Option<StencilResolveDefinition>,
    /// The device gate one depth-resolve case may state (`research/docs/23`
    /// §3.3, v57d): the case appears in a capture if and only if the device
    /// capability mask carries the named filter's bit. The marker still
    /// decides which rails own the case; the gate is the device-side half of
    /// the same question, so a rail whose device lacks the filter skips the
    /// case instead of running (and refusing) it.
    #[serde(default)]
    requires_depth_resolve_filter: Option<String>,
    /// The device gate one stencil-resolve case may state
    /// (`research/docs/23` §3.3, v60): the case appears in a capture if and
    /// only if the device capability mask carries the named filter's bit. The
    /// marker still decides which rails own the case; the gate is the
    /// device-side half of the same question, so a rail whose device lacks
    /// the filter skips the case instead of running (and refusing) it.
    #[serde(default)]
    requires_stencil_resolve_filter: Option<String>,
    /// The device gate one sample-count case may state (`research/docs/23`
    /// §3.3, v61): the case appears in a capture if and only if the device
    /// capability snapshot's sample ceiling is at least the count the case
    /// states. The marker still decides which rails own the case; the gate is
    /// the device-side half of the same question, so a rail whose device lacks
    /// the count skips the case instead of running (and refusing) it.
    #[serde(default)]
    requires_sample_count: Option<u64>,
    /// The culling state the pass draws with (`research/docs/23` §3.3, v39),
    /// or absent for "keep every triangle".
    #[serde(default)]
    cull: Option<CullDefinition>,
    /// The blend state the pass draws with (`research/docs/23` §3.3, v40), one
    /// entry per colour attachment in location order, or absent for "write the
    /// fragment output".
    #[serde(default)]
    blend: Option<Vec<BlendAttachmentDefinition>>,
    /// The coverage claim (`research/docs/23` §3.3, v38): `"partial"` says the
    /// draw covers only part of the attachment, so the expectation mixes the
    /// fragment output with the colour the pass started from. Absent means the
    /// milestone's stricter rule: every texel is the output.
    #[serde(default)]
    coverage: Option<String>,
    /// Texels the case does not claim, in row-major order
    /// (`research/docs/23` §3.3, v33). Only a `dontcare` load may leave bytes
    /// unclaimed — the undefined pre-pass contents are exactly what makes them
    /// unobservable — and the list has to leave at least one texel observed.
    #[serde(default)]
    wildcard_texels: Option<Vec<u64>>,
    /// Texels the case does not pin, each with the *closed* set of byte values
    /// its bytes may carry (`research/docs/23` §3.3, v67). Beside a
    /// multisample raster, such a texel's bytes are a resolve whose covered
    /// samples the draw wrote and whose other samples kept the pass's
    /// reference colour, so the set is exactly the mixtures those two colours
    /// produce. The channel is the single-attachment shape's, the load has to
    /// be `dontcare`, and it cannot be stated beside the free list: a byte
    /// either has a claim or it has none.
    #[serde(default)]
    wildcard_allowed_texels: Option<Vec<WildcardAllowedTexel>>,
}

/// The allowed values of one constrained wildcard texel
/// (`research/docs/23` §3.3, v67): the row-major texel index and the four-byte
/// values its bytes may take, each spelled as lowercase hex.
#[derive(Clone, Deserialize)]
struct WildcardAllowedTexel {
    index: u64,
    allowed: Vec<String>,
}

/// The instance count a case draws when it says nothing: the single instance
/// every pre-v31 fixture means.
fn default_instance_count() -> u64 {
    1
}

/// The state families a render case declares, in the reviewer's terms
/// (`research/docs/23` §3.3, v54 review H1/M1).
///
/// One family is one thing an object-API recording entry can carry: a
/// pass-wide raster, a depth surface, a depth resolve, a stencil surface, a
/// blend state, a culling state or a vertex offset. The counts
/// (`instance_count`, the index count) are deliberately not families — every
/// indexed entry carries them — and neither is the indirect payload, which has
/// an entry of its own.
fn object_state_families(case: &RenderCase) -> Vec<&'static str> {
    let mut families = Vec::new();
    if case.multisample.is_some() {
        families.push("multisample");
    }
    if case.depth.is_some() {
        families.push("depth");
    }
    if case.depth_resolve.is_some() {
        families.push("depth_resolve");
    }
    if case.stencil.is_some() {
        families.push("stencil");
    }
    if case.stencil_resolve.is_some() {
        families.push("stencil_resolve");
    }
    if case.blend.is_some() {
        families.push("blend");
    }
    if case.cull.is_some() {
        families.push("cull");
    }
    if case.base_vertex != 0 {
        families.push("base_vertex");
    }
    families
}

/// The family sets the reviewed object-API entries carry, spelled once so the
/// admission table and its test cannot drift (the v54 review's N3).
///
/// Each reviewed entry carries one family, and the combined entries carry a
/// raster plus one surface (`multisample+depth` since v54,
/// `multisample+stencil` since v56), plus the stored surface's resolve
/// (`multisample+depth+depth_resolve` since v58) — or the two faces of the
/// combined depth-stencil surface one entry carries together
/// (`multisample+depth+stencil` since v68), kept by neither face — or the same
/// two faces kept through their own resolves
/// (`multisample+depth+depth_resolve+stencil+stencil_resolve` since v70).
/// Every other combination would be recorded through an entry that silently
/// drops the rest — the failure mode the v54 review found — so the object rail
/// refuses it by name instead.
const REVIEWED_FAMILY_SETS: &[&[&str]] = &[
    &[],
    &["multisample"],
    &["depth"],
    &["stencil"],
    &["blend"],
    &["cull"],
    &["base_vertex"],
    &["multisample", "depth"],
    &["multisample", "depth", "depth_resolve"],
    &["multisample", "stencil"],
    &["multisample", "depth", "stencil"],
    &[
        "multisample",
        "depth",
        "depth_resolve",
        "stencil",
        "stencil_resolve",
    ],
];

/// Whether one object-API recording entry carries every family a case declares.
fn object_entry_admits(families: &[&str]) -> bool {
    REVIEWED_FAMILY_SETS.contains(&families)
}

/// Whether an entry that carries no state family at all — an indirect replay or
/// the `vertex_id` milestone — may record this case
/// (`research/docs/23` §3.3, v54 review N1).
fn object_entry_carries_no_state(families: &[&str]) -> bool {
    families.is_empty()
}

/// The pass-wide multisample state a case states, in the contract's own shape
/// (`research/docs/23` §3.3, v51/v52/v61).
///
/// `validate_render_case` already refused every count but the reviewed 2/4/8
/// family before this runs, so the mapping is total over the shapes that can
/// reach either rail; the refusal below keeps the helper total for a directly
/// constructed case.
fn case_multisample(case: &RenderCase) -> Result<Option<MultisampleState>> {
    match &case.multisample {
        Some(definition) => Ok(Some(MultisampleState {
            sample_count: match definition.sample_count {
                2 => SampleCount::Two,
                4 => SampleCount::Four,
                8 => SampleCount::Eight,
                other => {
                    return Err(format!(
                        "render case {}: unsupported multisample count {other}",
                        case.id
                    )
                    .into())
                }
            },
        })),
        None => Ok(None),
    }
}

/// The depth resolve one render case states, in the contract's own shape
/// (`research/docs/23` §3.3, v57).
///
/// `validate_render_case` already refused every filter outside the closed
/// family before this runs, so the mapping is total over the shapes that can
/// reach either rail; the refusal below keeps the helper total for a directly
/// constructed case.
fn case_depth_resolve(case: &RenderCase) -> Result<Option<MultisampleDepthResolve>> {
    match &case.depth_resolve {
        Some(definition) => Ok(Some(MultisampleDepthResolve {
            filter: match definition.filter.as_str() {
                "sample0" => DepthResolveFilter::Sample0,
                "min" => DepthResolveFilter::Min,
                "max" => DepthResolveFilter::Max,
                other => {
                    return Err(format!(
                        "render case {}: unsupported depth resolve filter {other:?}",
                        case.id
                    )
                    .into())
                }
            },
        })),
        None => Ok(None),
    }
}

/// The stencil resolve of a reviewed case, or `None` for every pass that
/// resolves nothing (`research/docs/23` §3.3, v60).
///
/// `validate_render_case` already refused every filter outside the closed
/// family before this runs, so the mapping is total over the shapes that can
/// reach either rail; the refusal below keeps the helper total for a directly
/// constructed case.
fn case_stencil_resolve(case: &RenderCase) -> Result<Option<MultisampleStencilResolve>> {
    match &case.stencil_resolve {
        Some(definition) => Ok(Some(MultisampleStencilResolve {
            filter: match definition.filter.as_str() {
                "sample0" => StencilResolveFilter::Sample0,
                "depth_resolved_sample" => StencilResolveFilter::DepthResolvedSample,
                other => {
                    return Err(format!(
                        "render case {}: unsupported stencil resolve filter {other:?}",
                        case.id
                    )
                    .into())
                }
            },
        })),
        None => Ok(None),
    }
}

/// The resolve of one texel's samples (`research/docs/23` §3.3, v51).
///
/// `covered` of `samples` samples carry the fragment output and the rest the
/// colour the pass started from, so each resolved channel is their arithmetic
/// mean. A mean that is not exactly representable is refused — `None` — rather
/// than rounded, because that is what keeps the expectation independent of a
/// driver's rounding rule: the fixture has to choose colours whose mixes divide
/// exactly, and the reviewed one does.
fn resolve_texel(fragment: &[u8; 4], clear: &[u8], covered: u32, samples: u32) -> Option<[u8; 4]> {
    let mut texel = [0_u8; 4];
    for channel in 0..4 {
        let sum = u32::from(fragment[channel])
            .checked_mul(covered)?
            .checked_add(u32::from(clear[channel]).checked_mul(samples - covered)?)?;
        if sum % samples != 0 {
            return None;
        }
        texel[channel] = u8::try_from(sum / samples).ok()?;
    }
    Some(texel)
}

/// One colour attachment's blend state (`research/docs/23` §3.3, v40).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlendAttachmentDefinition {
    source_rgb: String,
    destination_rgb: String,
    source_alpha: String,
    destination_alpha: String,
    operation: String,
}

/// The culling state a render case draws with (`research/docs/23` §3.3, v39).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CullDefinition {
    mode: String,
    winding: String,
}

/// The depth attachment a render case opens (`research/docs/23` §3.3, v36).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DepthAttachmentDefinition {
    format: String,
    width: u64,
    height: u64,
    load: String,
    #[serde(default)]
    clear_depth: Option<f64>,
    /// The store action the pass states (`research/docs/23` §3.3, v43), or
    /// absent for the pre-v43 shape: the rail-owned surface disappears with the
    /// pass and nothing observes its texels.
    #[serde(default)]
    store: Option<String>,
    /// The allocation the stored texels land in. Present exactly when `store`
    /// is, together with `view` and `expected_hex`.
    #[serde(default)]
    allocation: Option<u64>,
    /// The view inside that allocation the landing covers.
    #[serde(default)]
    view: Option<u64>,
    /// The depth texels the readback has to carry, as lowercase hex. The
    /// fixture states them because the comparison is byte-exact: a rail that
    /// skipped the store, the readback or the draw's depth write lands other
    /// bytes and fails here.
    #[serde(default)]
    expected_hex: Option<String>,
}

/// The depth state a render case's draw tests with (`research/docs/23` §3.3,
/// v36).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DepthTestDefinition {
    compare: String,
    write: bool,
}

/// The rail-owned stencil attachment a render case declares
/// (`research/docs/23` §3.3, v47).
///
/// The depth sibling's shape one byte wide: the format spelling, the extent and
/// the load operation with the value a clear starts from. The surface has no
/// trace identity yet — nothing reads it back — so the case names no view for
/// it.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StencilAttachmentDefinition {
    format: String,
    width: u64,
    height: u64,
    load: String,
    /// The value a `"clear"` load starts from; absent for any other load.
    #[serde(default)]
    clear_value: Option<u8>,
    /// The store action the pass states (`research/docs/23` §3.3, v49), or
    /// absent for the rail-owned shape: the surface disappears with the pass
    /// and nothing observes its texels.
    #[serde(default)]
    store: Option<String>,
    /// The allocation the stored stencil texels land in. Present exactly when
    /// `store` is, together with `view` and `expected_hex`.
    #[serde(default)]
    allocation: Option<u64>,
    /// The view inside that allocation the landing covers.
    #[serde(default)]
    view: Option<u64>,
    /// The stencil texels the readback has to carry, as lowercase hex — one
    /// byte per texel. The fixture states them because the comparison is
    /// byte-exact: a rail that skipped the store, the readback or the draw's
    /// stencil write lands other bytes and fails here.
    #[serde(default)]
    expected_hex: Option<String>,
}

/// The stencil state a render case's draw tests and writes with
/// (`research/docs/23` §3.3, v47).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StencilTestDefinition {
    compare: String,
    fail_op: String,
    depth_fail_op: String,
    pass_op: String,
    read_mask: u8,
    write_mask: u8,
    reference: u8,
}

/// The pass-wide multisample state (`research/docs/23` §3.3, v51).
///
/// The reviewed shape is the four-sample raster both rails spell
/// `SampleCount4`/`TYPE_4`, whose resolve lands in the attachment view the case
/// declares. The state is the pass's own, so it carries no attachment identity
/// and no per-attachment fields.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultisampleDefinition {
    /// Samples per texel. The reviewed fixture states `4`.
    sample_count: u64,
}

/// The depth resolve one render case states (`research/docs/23` §3.3, v57).
///
/// The pass-level filter the two APIs spell differently: the case's own
/// spelling is the closed family's wire name (`"sample0"`/`"min"`/`"max"`),
/// and the rails map it onto their own constants. The reviewed fixture states
/// `"sample0"`, the one filter the Lavapipe device reports.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DepthResolveDefinition {
    filter: String,
}

/// The stencil resolve filter a case states (`research/docs/23` §3.3, v60):
/// one of the two closed names, which the validator holds to the family and
/// the rails map onto their own constants.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StencilResolveDefinition {
    filter: String,
}

/// One vertex layout: the reviewed stream list, in binding order.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct VertexLayoutDefinition {
    buffers: Vec<VertexStreamDefinition>,
}

/// One stream of a vertex layout: its stride and the attributes read out of it.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct VertexStreamDefinition {
    stride: u64,
    /// How often the stream advances: `"per_vertex"` (the default every
    /// pre-v31 suite leaves implicit) or `"per_instance"` for the reviewed
    /// instanced fixture (`research/docs/23` §3.3, v31).
    #[serde(default = "default_vertex_step")]
    step: String,
    attributes: Vec<VertexAttributeDefinition>,
}

/// The step a stream declares when the case says nothing about it: the
/// per-vertex advance every pre-v31 fixture means.
fn default_vertex_step() -> String {
    "per_vertex".to_owned()
}

/// One attribute of a stream, in the contract's own terms.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct VertexAttributeDefinition {
    location: u32,
    offset: u64,
    format: String,
}

/// One render input view: identity, range and bytes, exactly the fields a
/// `BufferView` needs (`research/docs/23` §3.6).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderInputDefinition {
    allocation: u64,
    view: u64,
    offset: u64,
    length: u64,
    initial_hex: String,
}

/// One stage-buffer slot a render case declares (`research/docs/23` §3.3,
/// v83-v86, R9f/R9i): the pipeline's own declaration of the slot
/// ([`StageBufferBinding`]: the stage, the index inside that stage's Metal
/// buffer namespace, the access and the byte footprint the module's reflection
/// has to agree with) beside the pass's view of it ([`StageBufferView`]: the
/// identity, the range and the bytes, or the lease they come from).
///
/// One entry carries both halves on purpose: the pair rules hold the two to
/// each other field by field (`RenderPipelineContract::validate_against`), so a
/// fixture that spelled them twice could state a pass the pipeline never
/// declares. The `access` field is the one both halves carry, and the
/// `footprint` is the pipeline's half alone — the pass view states the bytes,
/// not the reach.
///
/// A *writable* entry is a landing: the pass's bytes leave through the same
/// byte-keyed writeback channel a stored attachment's texels use, so the entry
/// carries the `expected_hex` the readback has to show, and the view has to be
/// one the declaring compute pass also declares (a write lands in the trace's
/// own view pool).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageBufferDefinition {
    stage: String,
    index: u32,
    access: String,
    footprint: FootprintDefinition,
    allocation: u64,
    view: u64,
    offset: u64,
    length: u64,
    /// The owner window's byte length, when the view's bytes are imported
    /// through a lease (`research/docs/23` §90, R9i). Required exactly when
    /// `storage_mode` names a lease arm: the borrowed arm maps the allocation
    /// sized owner window, so the fixture has to state how long the owner's
    /// registration is.
    #[serde(default)]
    allocation_size: Option<u64>,
    initial_hex: String,
    /// Where the view's bytes come from: `owned_bytes` (the default the
    /// pre-R9i fixtures meant), `staged_lease` or `borrowed_no_copy`. The bytes
    /// themselves stay in `initial_hex` for every arm — they are the owner's
    /// own window — exactly as a compute case's `storage_mode` declares them.
    #[serde(default)]
    storage_mode: Option<String>,
    /// The bytes a writable slot's writeback has to carry. Required exactly
    /// when the access is writable; refused beside a read-only slot, which has
    /// no landing.
    #[serde(default)]
    expected_hex: Option<String>,
}

/// One stage-buffer slot's byte footprint (`research/docs/23` §3.3, v86), the
/// declaration half of the pair above.
///
/// Two arms exist and no third: a static ceiling in bytes (the shape the
/// reviewed stage-buffer pair declares), or the affine access set a translated
/// module's reflection states — one entry per `constant + stride * index`
/// access, each naming the axis of the draw's invocation indices it strides
/// over ([`RENDER_AFFINE_AXES`]: `0` is the vertex index, `1` the instance
/// index). `unbounded` is deliberately absent: the render contract refuses it
/// by name rather than executing a reach nobody can prove.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
enum FootprintDefinition {
    Static {
        max_bytes: u64,
    },
    Affine {
        accesses: Vec<AffineAccessDefinition>,
    },
}

/// One affine access (`metal_api_core::provider::AffineAccess`): the base
/// offset, the byte width of the access, and the terms that stride it.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffineAccessDefinition {
    base_offset: u64,
    access_size: u64,
    terms: Vec<AffineTermDefinition>,
}

/// One affine term (`metal_api_core::provider::AffineTerm`): the invocation
/// axis and the byte stride the access advances by.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffineTermDefinition {
    axis: usize,
    stride: u64,
}

/// The two AIR stages a case declares when its pipeline is a *translation*
/// rather than one of the reviewed pairs (`research/docs/23` §3.3, v84/v86).
///
/// The Vulkan rail translates each stage from the AIR here and registers the
/// pair through `register_translated_render_pipeline`, so the descriptor slots
/// come from the module's own reflection. The native rails compile no AIR, so a
/// case that carries this section is executable on the Vulkan trace rail alone
/// and its marker has to say so; `validate_render_case` refuses the alternative.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TranslatedStagesDefinition {
    vertex: Source,
    fragment: Source,
}

/// One index buffer view: the same fields plus the width of the indices.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexInputDefinition {
    allocation: u64,
    view: u64,
    offset: u64,
    length: u64,
    initial_hex: String,
    format: String,
}

/// The colour attachment a render case draws into. The fields mirror
/// `metal_api_core::provider::RenderAttachment`: identity, format, extent and
/// the load/store pair (`clear_hex` in memory order for a clear,
/// `initial_hex` for the previous contents).
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderAttachmentDefinition {
    allocation: u64,
    view: u64,
    format: String,
    width: u64,
    height: u64,
    load: String,
    /// The attachment's store operation: `"store"` keeps the pass's writes on
    /// the observable surface, `"dontcare"` discards them (`docs/23` §3.6,
    /// v19). Defaults to `"store"` so the v13-v18 fixtures that predate the
    /// field deserialize unchanged.
    #[serde(default = "default_attachment_store")]
    store: String,
    clear_hex: Option<String>,
    initial_hex: Option<String>,
    /// The MRT case's per-attachment expectation; absent for the
    /// single-attachment form, whose expectation is case-level.
    #[serde(default)]
    expected_hex: Option<String>,
}

fn default_attachment_store() -> String {
    "store".to_owned()
}

/// One rectangle of a rule-expected attachment the suite asks the capture to
/// report verbatim (R5a, `research/docs/23` §73).
///
/// The rule covers the whole plane; the windows are where the comparator checks
/// the bytes itself, because a digest alone would leave the per-texel rule
/// unobservable in the capture. The rectangle is in texels, its origin inside
/// the plane, and the check refuses a window that leaves it.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadbackWindowDefinition {
    x: u64,
    y: u64,
    width: u64,
    height: u64,
}

impl ReadbackWindowDefinition {
    /// The window's own bytes, tightly packed rows, out of one row-major plane.
    fn observe(&self, plane: &[u8], plane_width: u64) -> Result<ObservedWindow> {
        let row_bytes = usize::try_from(self.width)? * 4;
        let mut bytes = Vec::with_capacity(row_bytes * usize::try_from(self.height)?);
        for row in 0..self.height {
            let start = usize::try_from((self.y + row) * plane_width + self.x)? * 4;
            let end = start + row_bytes;
            bytes.extend_from_slice(
                plane
                    .get(start..end)
                    .ok_or("a readback window leaves the attachment plane")?,
            );
        }
        Ok(ObservedWindow {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            bytes_hex: hex(&bytes),
        })
    }
}

/// The suite-side shape of a render case's present action (the capture suite's
/// present input contract, shared with the compare rail). `initial_hex`
/// defaults to absent, i.e. `InitialState::Undefined`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PresentDefinition {
    mode: String,
    image_count: u32,
    acquire: u64,
    present: u64,
    #[serde(default)]
    initial_hex: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
enum AirEncoding {
    #[default]
    Text,
    Raw,
    Wrapped,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaseDispatch {
    grid: [u64; 3],
    local: [u64; 3],
    bindings: Option<Vec<u64>>,
    program: Option<usize>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaseProgram {
    entry: String,
    air: Source,
    metal: Source,
    buffer_slots: Option<Vec<BufferSlot>>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct BufferSlot {
    binding: u32,
    access: String,
    length: u64,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Source {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Buffer {
    binding: u32,
    allocation: u64,
    view: u64,
    offset: u64,
    length: u64,
    allocation_size: u64,
    access: String,
    /// The view's initial bytes, spelled as hex: the form every pre-R5a buffer
    /// carries. Exactly one of this field and [`Buffer::initial_repeat_hex`] is
    /// present.
    #[serde(default)]
    initial_hex: Option<String>,
    /// The view's initial bytes as one short pattern repeated to `length` (R5a,
    /// `research/docs/23` §73): the wide attachment's declaring view is 16 MiB,
    /// so spelling its `cd` fill out would put 32 MiB of hex in the fixture.
    /// The pattern's length has to divide the view length.
    #[serde(default)]
    initial_repeat_hex: Option<String>,
    /// The source arm the provider rails have to source this view's bytes
    /// through (`research/docs/23` §90, R9i): `owned_bytes` (the default),
    /// `staged_lease` or `borrowed_no_copy`. The bytes themselves stay in
    /// `initial_hex`/`initial_repeat_hex` for every arm — they are the owner's
    /// own window — so the oracle and the expectation read exactly the fields
    /// the pre-R9i fixtures carried, and the lease arms change only where the
    /// bytes come *from* on a provider rail.
    #[serde(default)]
    storage_mode: Option<String>,
}

impl Buffer {
    /// The bytes this declaration pre-seeds its view with.
    fn initial_bytes(&self) -> Result<Vec<u8>> {
        match (&self.initial_hex, &self.initial_repeat_hex) {
            (Some(hex_), None) => {
                let bytes = unhex(hex_)?;
                if bytes.len() as u64 != self.length {
                    return Err("initial data length differs from declared view length".into());
                }
                Ok(bytes)
            }
            (None, Some(pattern)) => {
                let unit = unhex(pattern)?;
                if unit.is_empty() {
                    return Err("an initial repeat pattern is at least one byte".into());
                }
                if !self.length.is_multiple_of(unit.len() as u64) {
                    return Err(format!(
                        "the initial repeat pattern of {} bytes does not divide the {} byte view",
                        unit.len(),
                        self.length
                    )
                    .into());
                }
                let mut bytes = Vec::with_capacity(usize::try_from(self.length)?);
                for _ in 0..self.length / unit.len() as u64 {
                    bytes.extend_from_slice(&unit);
                }
                Ok(bytes)
            }
            (Some(_), Some(_)) => {
                Err("a view carries either initial_hex or initial_repeat_hex, not both".into())
            }
            (None, None) => Err("a view needs its initial bytes: initial_hex or \
                                 initial_repeat_hex"
                .into()),
        }
    }
}

/// The reviewed per-texel rule of the wide attachment (R5a, `research/docs/23`
/// §73): texel `(x, y)` carries `x` and `y` as two little-endian `u16`s, i.e.
/// `[x & 0xff, (x >> 8) & 0xff, y & 0xff, (y >> 8) & 0xff]`.
///
/// The rule is the expectation form a 2048×2048 attachment needs: the whole
/// plane is four million texels, so the fixture states the function instead of
/// the bytes. It is injective over any window up to 65536 texels per axis, so a
/// flipped, transposed or row-shifted readback differs from it, and the check is
/// still byte for byte — the capture reports the plane's digest and the
/// declared windows' own bytes, and the comparator recomputes both from this
/// rule.
const XY_U16LE_V1: &str = "xy_u16le_v1";

/// The four bytes the reviewed rule stores at texel `(x, y)`.
fn rule_texel(rule: &str, x: u64, y: u64) -> Result<[u8; 4]> {
    if rule != XY_U16LE_V1 {
        return Err(format!("unknown texel rule {rule:?}").into());
    }
    let x = u16::try_from(x).map_err(|_| -> Box<dyn Error> {
        "the reviewed rule addresses texels up to 65536 per axis".into()
    })?;
    let y = u16::try_from(y).map_err(|_| -> Box<dyn Error> {
        "the reviewed rule addresses texels up to 65536 per axis".into()
    })?;
    let [x0, x1] = x.to_le_bytes();
    let [y0, y1] = y.to_le_bytes();
    Ok([x0, x1, y0, y1])
}

/// The whole plane the reviewed rule describes, row-major.
fn rule_bytes(rule: &str, width: u64, height: u64) -> Result<Vec<u8>> {
    let length = width
        .checked_mul(height)
        .and_then(|texels| texels.checked_mul(4))
        .ok_or("rule extent overflows")?;
    let mut bytes = Vec::with_capacity(usize::try_from(length)?);
    for y in 0..height {
        for x in 0..width {
            bytes.extend_from_slice(&rule_texel(rule, x, y)?);
        }
    }
    Ok(bytes)
}

/// The lowercase SHA-256 of one byte plane.
fn plane_sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The smallest extent a rule expectation is admissible at (R5a,
/// `research/docs/23` §73): the rule form exists for the megapixel-class
/// attachments whose hex spelling the fixture cannot carry, so a small case
/// keeps stating its texels and the pairwise-distinctness scan stays available.
const RULE_MIN_DIMENSION: u64 = 1024;

/// The largest extent the reviewed rule addresses: both coordinates travel as
/// little-endian `u16`s, so a wider axis could not be spelled injectively.
const RULE_ADDRESS_CEILING: u64 = 65_536;

/// How many readback windows one rule-expected case may declare.
const MAX_READBACK_WINDOWS: usize = 8;

/// The largest readback window per axis: the capture reports these bytes as
/// hex, so the reviewed shape keeps each window small (64×64 = 16 KiB).
const MAX_READBACK_WINDOW_DIMENSION: u64 = 64;

/// Whether a four-byte colour is one of the reviewed rule's texels over a
/// window of this extent.
///
/// The closed form of the distinctness rule: the first two bytes are a
/// little-endian `x` and the last two a little-endian `y`, so the rule reaches
/// the colour exactly when both addresses are inside the extent. A clear
/// colour the rule reaches would let a rail that ignored the draw land bytes
/// the expectation claims, which is why the check refuses it instead of
/// scanning a four-million-texel plane.
fn rule_reaches_colour(rule: &str, colour: &[u8], width: u64, height: u64) -> Result<bool> {
    if rule != XY_U16LE_V1 {
        return Err(format!("unknown texel rule {rule:?}").into());
    }
    if colour.len() != 4 {
        return Err("a clear colour is four bytes".into());
    }
    let x = u64::from(u16::from_le_bytes([colour[0], colour[1]]));
    let y = u64::from(u16::from_le_bytes([colour[2], colour[3]]));
    Ok(x < width && y < height)
}

/// One writeback a *suite* expects from a compute case: the bytes the view has
/// to land. The capture's own report shape below is not this one — its
/// rule-expected form carries a digest instead of bytes — so the two are
/// separate types and a report cannot be read back as a request.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedWriteback {
    allocation: u64,
    view: u64,
    offset: u64,
    bytes_hex: String,
}

/// One landed writeback: the bytes a view landed, or — for a case that declares
/// a rule expectation (R5a, `research/docs/23` §73) — the digest of the whole
/// plane plus the declared readback windows' own bytes.
///
/// The two forms are mutually exclusive: a rule-expected attachment is four
/// million texels, so the report carries the plane's SHA-256 (the comparator
/// recomputes it from the rule) and the small windows the suite names, while
/// every pre-R5a case keeps reporting its bytes.
#[derive(Serialize)]
struct Writeback {
    allocation: u64,
    view: u64,
    offset: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_windows: Option<Vec<ObservedWindow>>,
}

/// One declared readback window's actual bytes (`research/docs/23` §73).
///
/// The suite states the rectangle; the capture reports the bytes it read there,
/// so the comparator checks those texels against the rule itself rather than
/// trusting the plane digest alone.
#[derive(Serialize)]
struct ObservedWindow {
    x: u64,
    y: u64,
    width: u64,
    height: u64,
    bytes_hex: String,
}

/// One landed allocation image: the whole image's bytes, or its digest when the
/// case declares a rule expectation.
#[derive(Serialize)]
struct Allocation {
    allocation: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_length: Option<u64>,
}

impl Writeback {
    /// How many bytes this report covers, in either form: the digest form
    /// states the length, the byte form carries it as hex.
    fn observed_bytes(&self) -> usize {
        match (self.bytes_length, self.bytes_hex.as_deref()) {
            (Some(length), _) => usize::try_from(length).unwrap_or(usize::MAX),
            (None, Some(hex_)) => hex_.len() / 2,
            (None, None) => 0,
        }
    }

    /// The pre-R5a form: the bytes the view landed, verbatim.
    fn encoded(allocation: u64, view: u64, offset: u64, bytes: &[u8]) -> Self {
        Self {
            allocation,
            view,
            offset,
            bytes_hex: Some(hex(bytes)),
            bytes_sha256: None,
            bytes_length: None,
            observed_windows: None,
        }
    }

    /// The rule form: the plane's digest plus the declared windows' bytes.
    fn ruled(
        allocation: u64,
        view: u64,
        offset: u64,
        plane: &[u8],
        width: u64,
        windows: &[ReadbackWindowDefinition],
    ) -> Result<Self> {
        Ok(Self {
            allocation,
            view,
            offset,
            bytes_hex: None,
            bytes_sha256: Some(plane_sha256(plane)),
            bytes_length: Some(plane.len() as u64),
            observed_windows: Some(
                windows
                    .iter()
                    .map(|window| window.observe(plane, width))
                    .collect::<Result<Vec<_>>>()?,
            ),
        })
    }
}

impl Allocation {
    /// The form one allocation image is reported in: its bytes, or — when the
    /// image is wider than the verbatim cap — its digest (R5a,
    /// `research/docs/23` §73).
    fn observed(allocation: u64, bytes: &[u8]) -> Self {
        if bytes.len() > MAX_VERBATIM_ALLOCATION_BYTES {
            Self::digested(allocation, bytes)
        } else {
            Self::encoded(allocation, bytes)
        }
    }

    /// The pre-R5a form: the whole image, verbatim.
    fn encoded(allocation: u64, bytes: &[u8]) -> Self {
        Self {
            allocation,
            bytes_hex: Some(hex(bytes)),
            bytes_sha256: None,
            bytes_length: None,
        }
    }

    /// The wide form: the whole image's digest.
    fn digested(allocation: u64, bytes: &[u8]) -> Self {
        Self {
            allocation,
            bytes_hex: None,
            bytes_sha256: Some(plane_sha256(bytes)),
            bytes_length: Some(bytes.len() as u64),
        }
    }
}

/// Report one stored colour attachment's landing.
///
/// Every pre-R5a case reports the bytes themselves. A rule-expected case (R5a,
/// `research/docs/23` §73) reports the plane's digest plus the declared
/// readback windows' bytes instead, because the plane is four million texels:
/// the comparator recomputes the digest from the rule and checks every window
/// texel against it. The caller has already refused every shape where the two
/// planes could differ, so the writeback and the allocation image are the same
/// bytes here.
fn attachment_observation(
    rule: Option<&str>,
    windows: &[ReadbackWindowDefinition],
    // The attachment's own identity and row pitch: `(allocation, view, offset,
    // width)`, in the order the report states the first three.
    attachment: (u64, u64, u64, u64),
    write: &[u8],
    image: &[u8],
) -> Result<(Writeback, Allocation)> {
    let (allocation, view, offset, width) = attachment;
    match rule {
        None => Ok((
            Writeback::encoded(allocation, view, offset, write),
            Allocation::observed(allocation, image),
        )),
        Some(_) => {
            if write.len() != image.len() {
                return Err("a rule-expected attachment reports one plane".into());
            }
            Ok((
                Writeback::ruled(allocation, view, offset, write, width, windows)?,
                Allocation::digested(allocation, image),
            ))
        }
    }
}

/// Device-buffer copy counters for one command buffer. A case that splits its
/// dispatch sequence across several command buffers submits once per group, so
/// the accumulated case counters cannot show which boundary copied what
/// (`research/docs/15` §5b).
#[derive(Serialize)]
struct GroupCounts {
    copy_in: u32,
    copy_out: u32,
}

/// Present acquire / present completions for one render case, reported when the
/// case carries a present action (`research/docs/24` §5.3).
#[derive(Serialize)]
struct PresentCounts {
    acquire: u32,
    present: u32,
}

/// The indirect replay a render case declares (`research/docs/25` §4.3): one
/// command kind, the buffer's command cap and kind whitelist, the replayed
/// range and the command's own parameters. The capture builds the trace
/// payload from it; the report's segment comes from the provider's observation
/// of what it replayed, not from this request.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct IcbCase {
    kind: String,
    max_commands: u32,
    kinds: Vec<String>,
    range: IcbRangeCase,
    command: IcbCommandCase,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct IcbRangeCase {
    start: u32,
    count: u32,
}

#[derive(Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct IcbCommandCase {
    #[serde(default)]
    vertex_count: Option<u32>,
    #[serde(default)]
    index_count: Option<u32>,
    #[serde(default)]
    instance_count: Option<u32>,
    #[serde(default)]
    x: Option<u32>,
    #[serde(default)]
    y: Option<u32>,
    #[serde(default)]
    z: Option<u32>,
}

/// The heap placement observation for one compute case, reported when the case
/// carries a heap section (`research/docs/25` §5.1). The bytes stay with the
/// ordinary writeback comparison; this segment is what proves the placements
/// landed in one slab at the declared offsets.
#[derive(Serialize)]
struct HeapSegment {
    heap: u64,
    same_slab: bool,
    placements: Vec<HeapPlacementReport>,
}

#[derive(Serialize)]
struct HeapPlacementReport {
    allocation: u64,
    offset: u64,
    byte_size: u64,
}

/// The provider's own record of one indirect replay (`research/docs/25` §5.1).
#[derive(Serialize)]
struct IcbSegment {
    kind: &'static str,
    start: u32,
    count: u32,
    commands: u32,
}

/// One provider-agnostic placement record, mapped from either rail's
/// observation type before it becomes a [`HeapSegment`].
struct RawHeapPlacement {
    heap: u64,
    allocation: u64,
    offset: u64,
    byte_size: u64,
}

#[derive(Serialize)]
struct CaseResult {
    id: String,
    completion: &'static str,
    writebacks: Vec<Writeback>,
    allocations: Vec<Allocation>,
    /// Device-buffer copy-in / copy-out operations for this case, summed over
    /// its submissions. One of each per touched allocation, not per view
    /// (`research/docs/15` §3.3). Absent from the Swift reference oracle,
    /// which is not a provider.
    copy_in: Option<u32>,
    copy_out: Option<u32>,
    /// Per-command-buffer counters, one entry per committed command buffer and
    /// in commit order. Recorded only for cases that split their sequence, so
    /// the flat totals above stay the sum of the groups
    /// (`research/docs/15` §5b).
    #[serde(skip_serializing_if = "Option::is_none")]
    group_counts: Option<Vec<GroupCounts>>,
    /// The present action's `acquire`/`present` completions. Absent from cases
    /// that carry no present, and from the Swift reference oracle which is not
    /// a provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    present: Option<PresentCounts>,
    /// The heap placement observation. Absent from cases that carry no heap
    /// section.
    #[serde(skip_serializing_if = "Option::is_none")]
    heap: Option<HeapSegment>,
    /// The indirect replay observation. Absent from cases that carry no
    /// indirect command.
    #[serde(skip_serializing_if = "Option::is_none")]
    icb: Option<IcbSegment>,
    /// The source arm each of this case's views ran with
    /// (`research/docs/23` §90, R9i). Absent from cases whose suite declares no
    /// lease arm, and from the Swift reference oracle, which is not a provider:
    /// it places the same bytes in its own Metal buffer, so there is no arm for
    /// it to execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_modes: Option<Vec<StorageModeObservation>>,
}

/// One view's source arm as the run used it (`research/docs/23` §90, R9i).
#[derive(Serialize)]
struct StorageModeObservation {
    view: u64,
    mode: &'static str,
}

#[derive(Serialize)]
struct Capture {
    schema_version: u32,
    suite: String,
    suite_sha256: String,
    backend: &'static str,
    allocation_observation: &'static str,
    /// The device's depth resolve capability mask (`research/docs/23` §3.3,
    /// v57d): the same bitmask the provider snapshots, bit `i` = filter code
    /// `i`. The comparator reads the device-gated cases' presence against it.
    depth_resolve_modes: u32,
    /// The device's stencil resolve capability mask (`research/docs/23` §3.3,
    /// v60): the same bitmask the provider snapshots, bit `i` = filter code
    /// `i`. The comparator reads the device-gated stencil cases' presence
    /// against it.
    stencil_resolve_modes: u32,
    /// The device's reviewed sample-count mask (`research/docs/23` §3.3,
    /// v61): bit `i` = `SampleCount` code `i`. The comparator reads the
    /// device-gated sample-count cases' presence against it.
    render_sample_counts: u32,
    device: String,
    platform: String,
    results: Vec<CaseResult>,
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let mut suite_path = None;
    let mut output_path = None;
    let mut backend = None;
    let mut api = None;
    let mut async_execution = false;
    let mut queue_priorities = None;
    let mut probe = false;
    let mut probe_submissions = None;
    let mut command_socket = None;
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "usage: provider-capture --suite conformance/suite.json [--output capture.json] \
                 [--backend vulkan|native-metal-provider] [--api trace|objects] [--async] \
                 [--queue-priorities low,default,high,...]\n\
                 usage: provider-capture --queue-priority-probe \
                 [--queue-priorities low,default,high,...] [--queue-priority-submissions N] \
                 [--command-socket <path>]\n\
                 usage: provider-capture --queue-priority-child <command-socket>"
            );
            return Ok(());
        }
        if flag == "--queue-priority-child" {
            let path = args
                .next()
                .ok_or("--queue-priority-child requires a command socket")?;
            return run_queue_priority_child_mode(path);
        }
        if flag == "--async" {
            async_execution = true;
            continue;
        }
        if flag == "--queue-priority-probe" {
            probe = true;
            continue;
        }
        if flag == "--command-socket" && command_socket.is_none() {
            command_socket = Some(PathBuf::from(
                args.next().ok_or("--command-socket requires a path")?,
            ));
            continue;
        }
        if flag == "--queue-priorities" && queue_priorities.is_none() {
            let value = args.next().ok_or("missing argument value")?;
            queue_priorities = Some(parse_queue_priorities(
                value
                    .to_str()
                    .ok_or("--queue-priorities must be valid UTF-8")?,
            )?);
            continue;
        }
        if flag == "--queue-priority-submissions" && probe_submissions.is_none() {
            let value = args.next().ok_or("missing argument value")?;
            probe_submissions = Some(
                value
                    .to_str()
                    .ok_or("--queue-priority-submissions must be valid UTF-8")?
                    .parse()
                    .map_err(|error| format!("--queue-priority-submissions: {error}"))?,
            );
            continue;
        }
        if flag == "--backend" && backend.is_none() {
            backend = Some(
                match args.next().as_deref().and_then(|value| value.to_str()) {
                    Some("vulkan") => Backend::Vulkan,
                    Some("native-metal-provider") => Backend::NativeMetalProvider,
                    _ => return Err("--backend requires vulkan or native-metal-provider".into()),
                },
            );
            continue;
        }
        if flag == "--api" && api.is_none() {
            api = Some(
                match args.next().as_deref().and_then(|value| value.to_str()) {
                    Some("trace") => EntryApi::Trace,
                    Some("objects") => EntryApi::Objects,
                    _ => return Err("--api requires trace or objects".into()),
                },
            );
            continue;
        }
        let destination = if flag == "--suite" && suite_path.is_none() {
            &mut suite_path
        } else if flag == "--output" && output_path.is_none() {
            &mut output_path
        } else {
            return Err("unknown or duplicate argument; use --help".into());
        };
        *destination = Some(PathBuf::from(args.next().ok_or("missing argument value")?));
    }
    let backend = backend.unwrap_or(Backend::Vulkan);
    let api = api.unwrap_or(EntryApi::Trace);
    if async_execution && api != EntryApi::Objects {
        return Err("--async requires --api objects".into());
    }
    if probe {
        if suite_path.is_some() || output_path.is_some() || async_execution {
            return Err(
                "--queue-priority-probe runs its own scenario: it takes neither --suite, \
                 --output nor --async"
                    .into(),
            );
        }
        if backend == Backend::NativeMetalProvider {
            return Err("--queue-priority-probe requires the Vulkan backend".into());
        }
        if probe_submissions.is_some_and(|submissions| submissions == 0) {
            return Err("--queue-priority-submissions must be greater than zero".into());
        }
        let requested = queue_priorities.unwrap_or_else(default_queue_priorities);
        if requested.is_empty() {
            return Err("--queue-priorities must name at least one queue".into());
        }
        if let Some(path) = command_socket {
            // The provider lives in another process, so the marking and the
            // scenario both travel over the command channel.
            return run_remote_queue_priority_probe(
                &path,
                &requested,
                probe_submissions.unwrap_or(70),
            );
        }
        let executor =
            VulkanExecutor::new().map_err(|error| format!("create Vulkan executor: {error:?}"))?;
        install_queue_priorities(&executor, &requested)?;
        return run_queue_priority_probe(&executor, probe_submissions.unwrap_or(70));
    }
    if probe_submissions.is_some() {
        return Err("--queue-priority-submissions requires --queue-priority-probe".into());
    }
    if command_socket.is_some() {
        return Err("--command-socket requires --queue-priority-probe".into());
    }
    let suite_path = suite_path.ok_or("--suite is required")?;
    if output_path.as_ref().is_some_and(|path| path.exists()) {
        return Err("refusing to overwrite an existing capture".into());
    }
    // Validate every source and case before creating either provider device.
    // The reviewed suite has outgrown the 64 KiB bound the first suites fit in
    // (v67's constrained wildcard fixture pushed `suite-v28.json` past it), so
    // the ceiling is a quarter of the 1 MiB every other reviewed document gets;
    // the cap still refuses a file that is not a suite at all.
    let raw = read_bounded(&suite_path, 256 * 1024)?;
    let suite: Suite = serde_json::from_slice(&raw)?;
    validate_suite(&suite)?;
    let directory = suite_path.parent().unwrap_or(Path::new("."));
    let mut sources = BTreeMap::new();
    for case in &suite.cases {
        for program in case_programs(case) {
            let air = verified_source(directory, &program.air)?;
            let metal = verified_source(directory, &program.metal)?;
            let source = match backend {
                Backend::Vulkan => {
                    let air = String::from_utf8(air)?;
                    match case.air_encoding {
                        AirEncoding::Text => ShaderSource::SanitizedLl(air),
                        AirEncoding::Raw => ShaderSource::BinaryAir(assemble_owned_air(&air)?),
                        AirEncoding::Wrapped => {
                            ShaderSource::BinaryAir(wrap_air_bitcode(&assemble_owned_air(&air)?)?)
                        }
                    }
                }
                Backend::NativeMetalProvider => {
                    ShaderSource::MetalSource(String::from_utf8(metal)?)
                }
            };
            sources.insert((program.entry, case.air_encoding), source);
        }
    }
    // A reviewed render case pins the one reviewed MSL module; each trace rail
    // executes its own reviewed source pair — the Vulkan rail the matching
    // SPIR-V stages pinned in code above, the native rail the MSL module
    // itself — so only the module's declared identity is verified here. A
    // translated stage-buffer case pins its two AIR modules instead
    // (`research/docs/23` §3.3, v84), and both pins are verified before any
    // provider exists.
    for case in &suite.render_cases {
        if let Some(metal) = &case.metal {
            verified_source(directory, metal)?;
        }
        if let Some(translated) = &case.translated_stages {
            verified_source(directory, &translated.vertex)?;
            verified_source(directory, &translated.fragment)?;
        }
    }
    // Every rail owns a render execution path now: both trace rails
    // (`conformance/RENDER-CAPTURE.md` §4) and both object rails — the Vulkan
    // object rail landed it in `research/docs/24` §6 Step 5, and the native
    // object rail now runs the same reviewed pass through the native provider's
    // render entry point. A rail a render case's marker names therefore always
    // reports the case rather than omitting it.
    let identity = hex(&Sha256::digest(&raw));
    let (provider, device_name, counters, render_registrar, leases) =
        create_provider(backend, async_execution, queue_priorities.as_deref())?;
    let object_device =
        (api == EntryApi::Objects).then(|| objects::Device::new(Arc::clone(&provider)));
    let mut results = Vec::new();
    let mut pipelines = BTreeMap::new();
    let mut object_pipelines = BTreeMap::new();
    for ((entry, encoding), source) in sources {
        let request = PipelineCompileRequest {
            entry_name: entry.clone(),
            logical_digest: SemanticDigest::new(
                "suite-sha256-entry-v1",
                format!("{identity}:{entry}").into_bytes(),
            )?,
            source,
        };
        let pipeline = if let Some(device) = &object_device {
            let pipeline = device.compile_pipeline(request)?;
            let metadata = pipeline.metadata().clone();
            object_pipelines.insert((entry.clone(), encoding), pipeline);
            metadata
        } else {
            provider
                .compile(request)
                .map_err(|error| format!("compile {entry}: {error:?}"))?
        };
        if backend == Backend::Vulkan && (entry == "transform_3d" || entry == "mix_3d") {
            verify_transform_contract(&pipeline)?;
        }
        if backend == Backend::Vulkan && entry == "remap_3d" {
            let bindings = &pipeline.contract.buffer_bindings;
            if bindings
                .iter()
                .map(|b| (b.metal_binding, b.access))
                .collect::<Vec<_>>()
                != [
                    (1, BufferAccess::Read),
                    (3, BufferAccess::Read),
                    (7, BufferAccess::Write),
                ]
            {
                return Err("remap sparse layout/access reflection mismatch".into());
            }
            verify_xyz_access(&bindings[1].footprint)?;
            verify_xyz_access(&bindings[2].footprint)?;
            if bindings[0].footprint != (FootprintProof::Static { max_bytes: 4 }) {
                return Err("remap scalar bias reach mismatch".into());
            }
        }
        if backend == Backend::Vulkan && entry == "copy_3d" {
            verify_copy_contract(&pipeline)?;
        }
        eprintln!("{} artifact registered: entry={entry}", backend.name());
        pipelines.insert((entry, encoding), pipeline);
    }
    for (index, case) in suite.cases.iter().enumerate() {
        let programs = case_programs(case)
            .iter()
            .map(|program| pipelines[&(program.entry.clone(), case.air_encoding)].clone())
            .collect::<Vec<_>>();
        for (source, compiled) in case_programs(case).iter().zip(&programs) {
            if let Some(slots) = &source.buffer_slots {
                let expected = slots
                    .iter()
                    .map(|slot| {
                        (
                            slot.binding,
                            match slot.access.as_str() {
                                "read" => BufferAccess::Read,
                                "write" => BufferAccess::Write,
                                _ => BufferAccess::ReadWrite,
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                if compiled
                    .contract
                    .buffer_bindings
                    .iter()
                    .map(|b| (b.metal_binding, b.access))
                    .collect::<Vec<_>>()
                    != expected
                {
                    return Err(
                        "compiled reflection differs from per-program fixture layout".into(),
                    );
                }
            }
            // The texture face is compared the same way (`research/docs/26`
            // §21.3, step 1): the contract the harness registered must name the
            // case's own texture bindings, or the pairing the provider enforces
            // would be against a declaration the fixture never stated.
            let mut expected_textures = Vec::with_capacity(case.textures.len());
            for texture in &case.textures {
                let format = match texture.format.as_str() {
                    "r32_uint" => TextureFormat::R32Uint,
                    _ => return Err("unsupported texture format".into()),
                };
                let access = match texture.access.as_str() {
                    "sampled" => TextureAccess::Sampled,
                    "storage" => TextureAccess::Storage,
                    _ => return Err("unsupported texture access".into()),
                };
                expected_textures.push((texture.binding, format, access));
            }
            if compiled
                .contract
                .texture_bindings
                .iter()
                .map(|binding| (binding.metal_binding, binding.format, binding.access))
                .collect::<Vec<_>>()
                != expected_textures
            {
                return Err("compiled texture reflection differs from the fixture layout".into());
            }
        }
        let before = counters.read();
        // A heap case is owed only by the rails its marker names
        // (`research/docs/25` §5.2): a rail that declares no heap support must
        // omit the case rather than report a placement it never bound.
        if let Some(rails) = &case.capture_rails {
            if !rails.iter().any(|rail| rail == backend.report_name(api)) {
                continue;
            }
        }
        let (mut result, heap_remap) = if let Some(device) = &object_device {
            let programs = case_programs(case)
                .iter()
                .map(|program| {
                    object_pipelines[&(program.entry.clone(), case.air_encoding)].clone()
                })
                .collect::<Vec<_>>();
            let (result, allocation_ids) = run_object_case(
                device,
                &programs,
                case,
                suite.guard_byte,
                async_execution,
                &mut || counters.read(),
            )?;
            (result, Some(allocation_ids))
        } else {
            (
                run_case(
                    provider.as_ref(),
                    &programs,
                    case,
                    index as u64 + 1,
                    suite.guard_byte,
                    &mut || counters.read(),
                    &leases,
                )?,
                None,
            )
        };
        let after = counters.read();
        result.copy_in = Some(u32::try_from(after.0 - before.0)?);
        result.copy_out = Some(u32::try_from(after.1 - before.1)?);
        if case.heap.is_some() {
            result.heap = Some(heap_segment(
                counters.heap_observations(),
                heap_remap.as_ref(),
            )?);
        }
        if case.icb.is_some() {
            result.icb = Some(icb_segment(counters.icb_observations())?);
        }
        results.push(result);
    }
    // Render cases run after the compute cases on every rail, since every rail
    // now owns a render execution path: the Vulkan trace rail, the native
    // provider's trace rail (`conformance/RENDER-CAPTURE.md` §4), and both
    // object rails (`research/docs/24` §6 Step 5 plus the native object rail's
    // render entry point added here).
    // The render pipeline registration is keyed by the attachment count: the
    // dual-output fixture registers the reviewed dual fragment module, while a
    // single-attachment case keeps the pre-MRT pair. Every committed suite
    // carries one render shape today, but the cache refuses to reuse a
    // registration minted for another count rather than guessing.
    // The cache is keyed by (formats, geometry): one attachment list can now
    // carry two different reviewed module pairs — the indexed quad and the
    // instanced pair — so a registration minted for one geometry must not be
    // reused for the other (`research/docs/23` §3.3, v31).
    let mut render_pipeline: Option<(
        Vec<AttachmentFormat>,
        RenderGeometry,
        CompiledComputePipeline,
    )> = None;
    let mut object_render_pipeline: Option<(
        Vec<AttachmentFormat>,
        RenderGeometry,
        objects::RenderPipeline,
    )> = None;
    for (offset, case) in suite.render_cases.iter().enumerate() {
        // A render case is owed only by the rails its marker names: an indirect
        // case names the rails that declare indirect support, and a rail that
        // is not named must omit the case rather than run the direct shape.
        if !case
            .capture_rails
            .iter()
            .any(|rail| rail == backend.report_name(api))
        {
            // A rail the marker does not name omits the case. A stage-buffer
            // case says why out loud, because the omission is a boundary of
            // the shape rather than a device's answer: the object API has no
            // stage-buffer entry point yet (`research/docs/23` §3.3, v83), and
            // the native rails carry no translator for it.
            if !case.stage_buffers.is_empty() {
                println!(
                    "render case skipped: {} ({} binds no stage buffers; the case is marked for \
                     {})",
                    case.id,
                    backend.report_name(api),
                    case.capture_rails.join(", ")
                );
            }
            continue;
        }
        // A rail the marker *does* name has to execute the shape: executing a
        // stage-buffer case through the object API would drop the bindings,
        // which is why the refusal is by name rather than a silent run.
        if !case.stage_buffers.is_empty() && object_device.is_some() {
            return Err(format!(
                "render case {}: the object API binds no stage buffers yet, so this case runs \
                 on the trace rail alone",
                case.id
            )
            .into());
        }
        // The device gate (`research/docs/23` §3.3, v57d): a case that
        // requires a depth resolve filter appears in the capture if and only
        // if the device's capability mask carries that filter's bit. The skip
        // is the case-level gate — the marker already decided this rail owns
        // the case, and the device now decides whether it can run it. A rail
        // whose device lacks the bit omits the result; admission would refuse
        // the resolve otherwise, and the comparator pins the presence-iff-bit
        // rule against this observation.
        if let Some(filter) = &case.requires_depth_resolve_filter {
            let bit = 1u32
                << u32::from(match filter.as_str() {
                    "min" => DepthResolveFilter::Min.code(),
                    "max" => DepthResolveFilter::Max.code(),
                    _ => unreachable!("validate_render_case held the gate to min/max"),
                });
            if provider.capabilities().depth_resolve_modes & bit == 0 {
                println!("render case skipped: {} (device lacks {})", case.id, filter);
                continue;
            }
        }
        // The stencil-resolve device gate (`research/docs/23` §3.3, v60): the
        // depth gate's sibling — a case that requires the
        // depth-resolved-sample filter appears if and only if the device's
        // stencil mask carries its bit.
        if let Some(filter) = &case.requires_stencil_resolve_filter {
            let bit = 1u32
                << u32::from(match filter.as_str() {
                    "depth_resolved_sample" => StencilResolveFilter::DepthResolvedSample.code(),
                    _ => {
                        unreachable!("validate_render_case held the gate to depth_resolved_sample")
                    }
                });
            if provider.capabilities().stencil_resolve_modes & bit == 0 {
                println!("render case skipped: {} (device lacks {})", case.id, filter);
                continue;
            }
        }
        // The sample-count device gate (`research/docs/23` §3.3, v61): the
        // depth and stencil gates' sibling — a case that requires a sample
        // count appears if and only if the device's sample-count mask carries
        // that count's bit. The mask is the per-count answer the snapshot's
        // ceiling cannot give: Lavapipe admits 4x and 8x but not 2x.
        if let Some(count) = case.requires_sample_count {
            let bit = 1u32
                << u32::from(match count {
                    2 => SampleCount::Two.code(),
                    8 => SampleCount::Eight.code(),
                    _ => unreachable!("validate_render_case held the gate to 2x or 8x"),
                });
            if counters.render_sample_counts() & bit == 0 {
                println!(
                    "render case skipped: {} (device lacks the {}-sample raster)",
                    case.id, count
                );
                continue;
            }
        }
        let declaring = suite
            .cases
            .iter()
            .find(|declared| declared.id == case.declaring_case)
            .ok_or("render case declaring pass is not a case of this suite")?;
        let before = counters.read();
        let (acquires_before, presents_before) = counters.present_counts();
        let geometry = render_geometry(case, &format!("render case {}", case.id))?;
        let attachment_formats = render_case_attachments(case)
            .iter()
            .map(|attachment| attachment_format(&attachment.format))
            .collect::<Result<Vec<_>>>()?;
        let mut result = if let Some(device) = &object_device {
            let object_programs = case_programs(declaring)
                .iter()
                .map(|program| {
                    object_pipelines[&(program.entry.clone(), declaring.air_encoding)].clone()
                })
                .collect::<Vec<_>>();
            let object_pipeline = match &object_render_pipeline {
                Some((formats, cached_geometry, pipeline))
                    if *formats == attachment_formats && *cached_geometry == geometry =>
                {
                    pipeline.clone()
                }
                _ => {
                    let registered = register_render_case_pipeline(
                        &render_registrar,
                        &counters,
                        &identity,
                        case,
                        directory,
                        geometry,
                        &attachment_formats,
                    )?;
                    let wrapped = device.render_pipeline(&registered)?;
                    render_pipeline = Some((attachment_formats.clone(), geometry, registered));
                    object_render_pipeline =
                        Some((attachment_formats.clone(), geometry, wrapped.clone()));
                    wrapped
                }
            };
            run_object_render_case(
                device,
                &object_programs,
                declaring,
                case,
                &object_pipeline,
                suite.guard_byte,
                async_execution,
            )?
        } else {
            let programs = case_programs(declaring)
                .iter()
                .map(|program| pipelines[&(program.entry.clone(), declaring.air_encoding)].clone())
                .collect::<Vec<_>>();
            let pipeline = match &render_pipeline {
                Some((formats, cached_geometry, pipeline))
                    if *formats == attachment_formats && *cached_geometry == geometry =>
                {
                    pipeline.clone()
                }
                _ => {
                    let registered = register_render_case_pipeline(
                        &render_registrar,
                        &counters,
                        &identity,
                        case,
                        directory,
                        geometry,
                        &attachment_formats,
                    )?;
                    // A translated case registers its own two stages, so two of
                    // them never share one cached registration: only the
                    // reviewed arm's registration is reused across cases.
                    if case.translated_stages.is_none() {
                        render_pipeline =
                            Some((attachment_formats.clone(), geometry, registered.clone()));
                    }
                    registered
                }
            };
            run_render_case(
                provider.as_ref(),
                &programs,
                declaring,
                case,
                &pipeline,
                1000 + offset as u64,
                suite.guard_byte,
                &leases,
            )?
        };
        let after = counters.read();
        let (acquires_after, presents_after) = counters.present_counts();
        result.copy_in = Some(u32::try_from(after.0 - before.0)?);
        result.copy_out = Some(u32::try_from(after.1 - before.1)?);
        if case.present.is_some() {
            result.present = Some(PresentCounts {
                acquire: u32::try_from(acquires_after - acquires_before)?,
                present: u32::try_from(presents_after - presents_before)?,
            });
        }
        if case.icb.is_some() {
            result.icb = Some(icb_segment(counters.icb_observations())?);
        }
        results.push(result);
    }
    if api == EntryApi::Trace {
        for pipeline in pipelines.values() {
            provider
                .release_pipeline(pipeline)
                .map_err(|error| format!("release pipeline: {error:?}"))?;
        }
    }
    if let Some((_, _, pipeline)) = render_pipeline.as_ref() {
        release_render_pipeline(&render_registrar, pipeline)?;
    }
    let capture = Capture {
        schema_version: 1,
        suite: suite.suite,
        suite_sha256: identity,
        backend: backend.report_name(api),
        allocation_observation: "host-writeback-landing",
        depth_resolve_modes: provider.capabilities().depth_resolve_modes,
        stencil_resolve_modes: provider.capabilities().stencil_resolve_modes,
        render_sample_counts: counters.render_sample_counts(),
        device: device_name,
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        results,
    };
    let mut bytes = serde_json::to_vec_pretty(&capture)?;
    bytes.push(b'\n');
    if let Some(path) = output_path {
        // create_new also closes the race after the initial existence check.
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&bytes)?;
    } else {
        std::io::stdout().lock().write_all(&bytes)?;
    }
    Ok(())
}

fn validate_suite(suite: &Suite) -> Result<()> {
    let case_ids: &[&str] = match (suite.schema_version, suite.suite.as_str()) {
        (1, "compute-buffer-v1") => &["copy_word", "indexed_boundary"],
        (1, "compute-buffer-v2") => &[
            "copy_seed_a",
            "copy_seed_b",
            "indexed_tail",
            "indexed_full",
            "indexed_small_grid",
            "indexed_unit",
            "transform_tail",
            "transform_small_grid",
        ],
        (1, "compute-buffer-v3") => &[
            "transform_twice",
            "transform_three_times",
            "transform_eight_times",
        ],
        (1, "compute-buffer-v4") => &[
            "transform_pingpong_two",
            "transform_pingpong_three",
            "transform_pingpong_eight",
            "copy_pingpong",
        ],
        (1, "compute-buffer-v5") => &[
            "pipeline_chain_two",
            "pipeline_chain_three",
            "pipeline_chain_eight",
        ],
        (1, "compute-buffer-v6") => &[
            "layout_chain_two",
            "layout_chain_three",
            "layout_chain_eight",
        ],
        (1, "compute-buffer-v7") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
        ],
        (1, "compute-buffer-v8") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
        ],
        (1, "compute-buffer-v9") => &[
            "subset_chain_two",
            "subset_chain_four",
            "subset_chain_eight",
        ],
        (1, "compute-buffer-v10") => &["alias_disjoint_pair", "alias_disjoint_pair_reversed"],
        (1, "compute-buffer-v11") => &["sampled_texture_first_texel"],
        (1, "compute-buffer-v12") => &["texture_cell_local_4x4", "texture_cell_local_1x1"],
        (1, "compute-buffer-v13") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v14") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v15") => &["heap_placement_copy_word", "icb_dispatch_copy_word"],
        (1, "compute-buffer-v16") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v17") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v18") => &["render_declaring_two_attachments"],
        (1, "compute-buffer-v19") => &["render_declaring_store_and_discard"],
        (1, "compute-buffer-v20") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v21") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v22") => &["render_declaring_copy_word"],
        (1, "compute-buffer-v23") => &["render_declaring_four_attachments"],
        (1, "compute-buffer-v24") => &["render_declaring_three_attachments"],
        (1, "compute-buffer-v25") => &["render_declaring_two_attachments"],
        (1, "compute-buffer-v26") => &["render_declaring_quad_extent"],
        (1, "compute-buffer-v27") => &["render_declaring_two_attachments"],
        (1, "compute-buffer-v28") => &[
            "render_declaring_quad_extent",
            "render_declaring_multisample_seed",
            "render_declaring_depth_store",
            "render_declaring_depth_resolve",
            "render_declaring_stencil_store",
            "render_declaring_stencil_resolve",
            "render_declaring_attachment_16x16",
            "render_declaring_attachment_64x64",
            "render_declaring_attachment_2048x2048",
        ],
        // The compute texture face's content pair (`research/docs/26` §21.3):
        // one kernel, two texture payloads, two different writebacks, so a
        // readback that ignores the texture's own bytes fails a real capture
        // instead of agreeing with a constant expectation.
        (1, "compute-buffer-v29") => &[
            "sampled_cell_ascending_content",
            "sampled_cell_descending_content",
        ],
        // The lease face (`research/docs/23` §90, R9i): the two source arms
        // the providers execute beyond owned bytes, one case each.
        (1, "compute-buffer-v30") => &["staged_lease_copy_word", "borrowed_lease_copy_word"],
        // The stage-buffer face (`research/docs/23` §3.3, v83-v86): the two
        // declaring passes the stage-buffer render cases resolve their views
        // against — the writable sink's own pool view, and the borrowed tint's
        // owner registration.
        (1, "compute-buffer-v31") => &[
            "render_declaring_stage_buffer_sink",
            "render_declaring_stage_buffer_lease",
        ],
        _ => return Err("unsupported suite identity/version".into()),
    };
    if suite.cases.len() != case_ids.len()
        || suite
            .cases
            .iter()
            .any(|case| !case_ids.contains(&case.id.as_str()))
    {
        return Err("incorrect case set for suite".into());
    }
    let encodings = suite
        .cases
        .iter()
        .map(|case| case.air_encoding)
        .collect::<BTreeSet<_>>();
    if suite.suite == "compute-buffer-v8" {
        if !encodings.contains(&AirEncoding::Raw) || !encodings.contains(&AirEncoding::Wrapped) {
            return Err("v8 suite must cover raw and wrapped binary AIR".into());
        }
    } else if encodings
        .iter()
        .any(|encoding| *encoding != AirEncoding::Text)
    {
        return Err("binary AIR encodings are only qualified by the v8 suite".into());
    }
    let mut ids = BTreeSet::new();
    // The view a render case draws into is the attachment's own allocation: the
    // render rail reads the image back directly, so the guard-byte discipline
    // that makes a compute case's offset mistake observable in the allocation
    // image does not apply to it (`research/docs/23` §5.2). A stored depth
    // attachment is the same shape from v43 on: the render pass writes its
    // whole view, so that view needs no guard bytes either.
    let attachment_views = suite
        .render_cases
        .iter()
        .flat_map(|case| {
            render_case_attachments(case)
                .into_iter()
                .map(|attachment| (attachment.allocation, attachment.view))
                .chain(case.depth.as_ref().and_then(depth_attachment_identity))
                .chain(case.stencil.as_ref().and_then(stencil_attachment_identity))
        })
        .collect::<BTreeSet<_>>();
    for case in &suite.cases {
        validate_case_programs(case)?;
        validate_case_dispatches(case)?;
        validate_case_command_buffers(&suite.suite, case)?;
        if !ids.insert(&case.id) {
            return Err("duplicate case identity".into());
        }
        let (entry, grid, local, buffers) = case_shape(&case.id)?;
        if case.entry != entry
            || case.grid != grid
            || case.local != local
            || case.buffers.len() != buffers.len()
        {
            return Err(
                format!("case {} is outside the qualified dispatch subset", case.id).into(),
            );
        }
        let mut allocations = BTreeMap::<u64, Vec<(u64, u64)>>::new();
        let mut views = BTreeSet::new();
        for (index, buffer) in case.buffers.iter().enumerate() {
            let end = buffer
                .offset
                .checked_add(buffer.length)
                .ok_or("view range overflow")?;
            let attachment_target = attachment_views.contains(&(buffer.allocation, buffer.view));
            if buffer.binding != buffers[index].0
                || buffer.access != buffers[index].1
                || buffer.length != buffers[index].2
                || buffer.allocation_size > MAX_DECLARED_BYTES as u64
                || end > buffer.allocation_size
                || (!attachment_target && (buffer.offset < 4 || buffer.allocation_size - end < 4))
                || !buffer.offset.is_multiple_of(4)
                || buffer.allocation == 0
                || buffer.view == 0
                || !views.insert(buffer.view)
            {
                return Err(format!("invalid buffer declaration in {}", case.id).into());
            }
            // Several buffers may name one allocation while their byte ranges
            // stay disjoint: that is the v10 ranged-alias shape. Overlapping
            // ranges would make the observed allocation image depend on write
            // order, so they are refused here exactly as provider admission
            // refuses them.
            let ranges = allocations.entry(buffer.allocation).or_default();
            if ranges
                .iter()
                .any(|(start, other_end)| buffer.offset < *other_end && *start < end)
            {
                return Err(format!(
                    "overlapping views of allocation {} in {}",
                    buffer.allocation, case.id
                )
                .into());
            }
            // The view's own initial bytes: the hex form every pre-R5a
            // declaration carries, or the repeated pattern the wide case uses
            // (R5a, `research/docs/23` §73). Both resolve to exactly `length`
            // bytes, which is what this check is about.
            let initial = buffer.initial_bytes()?;
            if initial.len() as u64 != buffer.length {
                return Err("initial data length differs from declared view length".into());
            }
            ranges.push((buffer.offset, end));
        }
        let mut writable: Vec<_> = case
            .buffers
            .iter()
            .filter(|b| ever_writable(case).contains(&b.view))
            .collect();
        writable.sort_by_key(|buffer| (buffer.allocation, buffer.view));
        if case.expected_writebacks.len() != writable.len() {
            return Err("expected result does not cover writable views".into());
        }
        for (expected, buffer) in case.expected_writebacks.iter().zip(writable) {
            if expected.allocation != buffer.allocation
                || expected.view != buffer.view
                || expected.offset != buffer.offset
                || unhex(&expected.bytes_hex)?.len() as u64 != buffer.length
            {
                return Err("expected result identity/range mismatch".into());
            }
        }
        // v11 texture section: every texture carries its full tightly packed
        // image, and each one names a distinct allocation the provider can
        // resolve.
        let mut texture_bindings = BTreeSet::new();
        let mut texture_allocations = BTreeSet::new();
        for texture in &case.textures {
            if !texture_bindings.insert(texture.binding) {
                return Err(format!("duplicate texture binding in {}", case.id).into());
            }
            if !texture_allocations.insert(texture.allocation) {
                return Err(format!("duplicate texture allocation in {}", case.id).into());
            }
            let expected = texture
                .width
                .checked_mul(texture.height)
                .and_then(|extent| extent.checked_mul(4))
                .ok_or("texture extent overflows")?;
            if unhex(&texture.initial_hex)?.len() as u64 != expected {
                return Err(format!(
                    "texture initial data length differs from declared extent in {}",
                    case.id
                )
                .into());
            }
        }
    }
    let mut render_ids = BTreeSet::new();
    for case in &suite.render_cases {
        validate_render_case(suite, case)?;
        if !render_ids.insert(&case.id) || ids.contains(&case.id) {
            return Err("duplicate render case identity".into());
        }
    }
    Ok(())
}

/// Validate one render case against the declaring case it draws into.
///
/// The rules mirror `conformance/compare.py`'s render plan and the Swift
/// oracle's `validateRenderCase`, including the two falsifiability rules: every
/// texel of the expectation has to be the fragment output, and the expectation
/// has to differ from the value the pass started from, so "the pass never ran"
/// cannot satisfy it.
/// The reviewed geometry a render case draws (`research/docs/23` §3.3).
///
/// Two shapes exist and no third: the milestone's `vertex_id` triangle, and the
/// reviewed indexed quad whose vertex stream is a `float32x2` position at stride
/// eight plus six `uint16` indices. A case cannot describe a geometry the two
/// rails have not been reviewed against — that is what keeps the expected texels
/// falsifiable instead of merely observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RenderGeometry {
    Milestone,
    IndexedQuad,
    /// The reviewed instanced pair (`research/docs/23` §3.3, v31): the same
    /// indexed quad, a second per-instance tint stream, and two instances.
    InstancedPair,
    /// The reviewed base-vertex shape (`research/docs/23` §3.3, v34): the same
    /// reviewed layout and module over a five-vertex stream whose first vertex
    /// is a degenerate centre, drawn with `base_vertex: 1` so the four reviewed
    /// corners are the ones the indices reach.
    BaseVertexQuad,
    /// The reviewed depth pair (`research/docs/23` §3.3, v36): two oversize
    /// triangles at different z, each carrying its own tint, drawn with a
    /// depth attachment and a `less` test.
    DepthPair,
    /// The reviewed cull pair (`research/docs/23` §3.3, v39): the same pair
    /// module over two oversize triangles at one z whose vertex orders are
    /// opposite, so the pass's cull state decides which tint survives.
    CullPair,
    /// The reviewed blend triangle (`research/docs/23` §3.3, v40): the pair
    /// module over one oversize triangle whose tint carries an alpha below one,
    /// so the pass's blend state is what the attachment's bytes measure.
    BlendTriangle,
    /// The reviewed render sampler (`research/docs/23` §3.3, v70): the
    /// milestone's `vertex_id` geometry whose fragment stage samples one
    /// `rgba8_unorm` texture of the render area's own extent, so the attachment
    /// reads back exactly the uploaded texels.
    SampledTexture,
    /// The stage-buffer shape (`research/docs/23` §3.3, v83-v86): the pass
    /// binds one or more `[[buffer(N)]]` slots and the pipeline declares what
    /// each stage reaches there. One reviewed pair serves the *reviewed* arm
    /// (the two stages read their own slots with static footprints), while a
    /// case that carries `translated_stages` registers its stages from AIR and
    /// executes the writable and affine faces the reviewed modules cannot
    /// state (R9f).
    StageBuffers,
}

/// One sampled texture a render case binds (`research/docs/23` §3.3, v70): the
/// view its texel bytes travel under, the extent the pass has to share with it,
/// and the bytes themselves — as hex for every pre-R5a case, or as a reviewed
/// per-texel rule for the wide one (R5a, §73), where spelling four million
/// texels out is what the rule replaces.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FragmentTextureDefinition {
    allocation: u64,
    view: u64,
    format: String,
    width: u64,
    height: u64,
    #[serde(default)]
    initial_hex: Option<String>,
    #[serde(default)]
    texel_rule: Option<String>,
}

impl FragmentTextureDefinition {
    /// The texels this texture is uploaded with.
    fn texels(&self) -> Result<Vec<u8>> {
        match (&self.initial_hex, &self.texel_rule) {
            (Some(hex_), None) => {
                let bytes = unhex(hex_)?;
                let expected = self
                    .width
                    .checked_mul(self.height)
                    .and_then(|texels| texels.checked_mul(4))
                    .ok_or("texture extent overflows")?;
                if bytes.len() as u64 != expected {
                    return Err("the uploaded texels do not match the extent".into());
                }
                Ok(bytes)
            }
            (None, Some(rule)) => rule_bytes(rule, self.width, self.height),
            (Some(_), Some(_)) => Err("a texture carries either initial_hex or texel_rule, \
                                       not both"
                .into()),
            (None, None) => Err("a texture needs its texels: initial_hex or texel_rule".into()),
        }
    }

    /// The rule the texture's texels follow, when it declares one.
    fn rule(&self) -> Option<&str> {
        self.texel_rule.as_deref()
    }
}

/// The colour attachments a render case declares, in location order: the
/// single form wrapped in a one-entry list, or the MRT list. Both forms at
/// once, and neither form, are refused by [`validate_render_case`] before
/// this runs.
fn render_case_attachments(case: &RenderCase) -> Vec<&RenderAttachmentDefinition> {
    if let Some(attachments) = &case.attachments {
        return attachments.iter().collect();
    }
    case.attachment.iter().collect()
}

/// The identity a render case's stored depth attachment lands in
/// (`research/docs/23` §3.3, v43), or `None` for the pre-v43 shape: a depth
/// attachment with no store action is rail-owned and names nowhere its texels
/// would land.
fn depth_attachment_identity(depth: &DepthAttachmentDefinition) -> Option<(u64, u64)> {
    match (depth.store.as_deref(), depth.allocation, depth.view) {
        (Some("store"), Some(allocation), Some(view)) => Some((allocation, view)),
        _ => None,
    }
}

/// The identity a render case's stored stencil attachment lands in
/// (`research/docs/23` §3.3, v49), or `None` for the rail-owned shape.
fn stencil_attachment_identity(stencil: &StencilAttachmentDefinition) -> Option<(u64, u64)> {
    match (stencil.store.as_deref(), stencil.allocation, stencil.view) {
        (Some("store"), Some(allocation), Some(view)) => Some((allocation, view)),
        _ => None,
    }
}

/// The (attachment, expected texel hex) pairs a render case declares, in
/// location order. The single form wraps its one attachment with the
/// case-level `expected_hex`; the MRT form spells the expectation on each
/// attachment entry; a discarded attachment carries none at all, which is why
/// the expectation is optional and [`validate_render_case`] is the single
/// place that pins which attachment may leave it out (`docs/23` §3.6, v19).
/// The attachment format a render case declares (`research/docs/23` §3.3, v21).
///
/// Two 8-bit UNORM layouts are admitted: the reviewed fragment stage stores the
/// same colour either way, and the attachment's own layout decides which channel
/// lands in which byte, so the fixture's expected texels pin the layout.
fn attachment_format(name: &str) -> Result<AttachmentFormat> {
    match name {
        "rgba8_unorm" => Ok(AttachmentFormat::Rgba8Unorm),
        "bgra8_unorm" => Ok(AttachmentFormat::Bgra8Unorm),
        "r32float" => Ok(AttachmentFormat::R32Float),
        other => Err(format!("unsupported attachment format {other:?}").into()),
    }
}

fn render_attachment_shapes(
    case: &RenderCase,
) -> Result<Vec<(&RenderAttachmentDefinition, Option<String>)>> {
    if let Some(attachment) = &case.attachment {
        Ok(vec![(attachment, case.expected_hex.clone())])
    } else if let Some(attachments) = &case.attachments {
        attachments
            .iter()
            .map(|attachment| Ok((attachment, attachment.expected_hex.clone())))
            .collect()
    } else if case
        .depth
        .as_ref()
        .is_some_and(|depth| depth.store.as_deref() == Some("store"))
    {
        // The zero-colour-attachment depth pass (`research/docs/23` §3.3,
        // v46): the case declares no colour attachment at all, and its stored
        // depth attachment is the whole landing.
        Ok(Vec::new())
    } else {
        Err(format!("render case {}: no attachment declared", case.id).into())
    }
}

/// The reviewed quad layout (`research/docs/23` §3.3): one `float32x2` position
/// stream at stride eight.
fn reviewed_quad_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: QUAD_STRIDE,
        step: VertexStep::PerVertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: VertexFormat::Float32x2,
        }],
    }])
}

/// The reviewed instanced layout (`research/docs/23` §3.3, v31): the same
/// `float32x2` position stream, plus a `float32x4` tint that advances once per
/// *instance*.
fn reviewed_instanced_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![
        VertexBufferLayout {
            stride: QUAD_STRIDE,
            step: VertexStep::PerVertex,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        },
        VertexBufferLayout {
            stride: INSTANCED_TINT_STRIDE,
            step: VertexStep::PerInstance,
            attributes: vec![VertexAttribute {
                location: 1,
                offset: 0,
                format: VertexFormat::Float32x4,
            }],
        },
    ])
}

/// The reviewed corner bytes of the instanced fixture's position stream: the
/// same four NDC corners the indexed quad carries, in the same order.
const INSTANCED_POSITION_HEX: &str =
    "000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f";
/// The first reviewed instance tint, `(1, 0, 0, 1)` as `float32x4` bytes.
const INSTANCED_TINT_RED_HEX: &str = "0000803f00000000000000000000803f";
/// The second reviewed instance tint, `(0, 1, 0, 1)` as `float32x4` bytes.
const INSTANCED_TINT_GREEN_HEX: &str = "000000000000803f000000000000803f";
/// The UNORM8 bytes those two tints store in an `rgba8_unorm` attachment. Each
/// component is exactly `0.0` or `1.0`, so the byte mapping is exact and not a
/// rounding question; a fixture that disagreed with the tints would fail the
/// rails' own captures rather than pass here.
const INSTANCED_TINT_RED_BYTES: [u8; 4] = [0xff, 0x00, 0x00, 0xff];
const INSTANCED_TINT_GREEN_BYTES: [u8; 4] = [0x00, 0xff, 0x00, 0xff];

/// The reviewed base-vertex fixture's stream (`research/docs/23` §3.3, v34):
/// a degenerate centre vertex followed by the reviewed quad corners, so the
/// same indices draw the reviewed quad only when the draw adds a base vertex
/// of one. The exact bytes are pinned for the same reason the quad's are: the
/// shape is the review.
const BASE_VERTEX_POSITION_HEX: &str = "0000000000000000\
     000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f";
/// The offset the reviewed fixture draws with: one degenerate vertex in front
/// of the reviewed corners.
const BASE_VERTEX_OFFSET: u64 = 1;

/// Pin the reviewed base-vertex shape (`research/docs/23` §3.3, v34).
///
/// The layout and the index buffer are the reviewed quad's; what the pair adds
/// is the five-vertex stream (`BASE_VERTEX_POSITION_HEX`) and `base_vertex: 1`.
/// A rail that ignored the offset would read the first four vertices — the
/// degenerate centre and three corners — and leave part of the attachment at
/// the clear colour, which is what makes the fixture falsifiable.
fn reviewed_base_vertex_geometry(
    case: &RenderCase,
    layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the reviewed base-vertex shape binds one stream").into());
    }
    let stream = &layout.buffers[0];
    if stream.stride != QUAD_STRIDE || stream.step != "per_vertex" || stream.attributes.len() != 1 {
        return Err(format!(
            "{where_}: the reviewed base-vertex stream is one float32x2 at stride {QUAD_STRIDE}"
        )
        .into());
    }
    let attribute = &stream.attributes[0];
    if attribute.location != 0 || attribute.offset != 0 || attribute.format != "float32x2" {
        return Err(format!(
            "{where_}: the reviewed base-vertex attribute is location 0, offset 0, float32x2"
        )
        .into());
    }
    if case.base_vertex != BASE_VERTEX_OFFSET {
        return Err(format!(
            "{where_}: the reviewed base-vertex draw offsets its indices by {BASE_VERTEX_OFFSET}"
        )
        .into());
    }
    if case.instance_count != 1 {
        return Err(format!("{where_}: the reviewed base-vertex shape draws one instance").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let required = QUAD_STRIDE * (QUAD_VERTICES + BASE_VERTEX_OFFSET);
    if buffer.length != required || buffer.initial_hex != BASE_VERTEX_POSITION_HEX {
        return Err(format!(
            "{where_}: the reviewed base-vertex stream is the degenerate centre plus the quad"
        )
        .into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed base-vertex shape is indexed").into());
    };
    if indices.allocation == 0 || indices.view == 0 {
        return Err(format!("{where_}: zero index buffer identity").into());
    }
    let width = match indices.format.as_str() {
        "uint16" => 2_u64,
        "uint32" => 4,
        other => return Err(format!("{where_}: unsupported index format {other:?}").into()),
    };
    let required = QUAD_INDICES * width;
    if indices.length != required {
        return Err(format!(
            "{where_}: the reviewed base-vertex index buffer is {QUAD_INDICES} indices wide"
        )
        .into());
    }
    let bytes = unhex(&indices.initial_hex)?;
    if bytes.len() != usize::try_from(indices.length)? {
        return Err(format!("{where_}: the index bytes do not match their length").into());
    }
    // The reviewed index list is the quad's over the reviewed four corners: the
    // draw reaches vertices `base_vertex + index`, and the rails prove that
    // span against the stream's own footprint before a device object exists.
    let reviewed = "000001000200010003000200";
    if indices.initial_hex != reviewed {
        return Err(
            format!("{where_}: the reviewed base-vertex indices are the reviewed quad's").into(),
        );
    }
    let _ = width;
    Ok(RenderGeometry::BaseVertexQuad)
}

/// The reviewed depth layout (`research/docs/23` §3.3, v36): one stream whose
/// vertices carry a `float32x3` position (offset 0) and a `float32x4` tint
/// (offset 16) at stride thirty-two.
fn reviewed_depth_layout() -> VertexLayout {
    VertexLayout::Buffers(vec![VertexBufferLayout {
        stride: DEPTH_STRIDE,
        step: VertexStep::PerVertex,
        attributes: vec![
            VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x3,
            },
            VertexAttribute {
                location: 1,
                offset: 16,
                format: VertexFormat::Float32x4,
            },
        ],
    }])
}

/// The reviewed depth stream: two oversize triangles, the first at `z = 0.5`
/// with the red tint and the second at `z = 0.9` with the green one. The bytes
/// are pinned because the shape *is* the review: the two triangles cover the
/// same texels, and only their depth differs.
const DEPTH_PAIR_POSITIONS_HEX: [&str; 6] = [
    // The oversize triangle (-1, -1), (3, -1), (-1, 3) at z = 0.5.
    "000080bf000080bf0000003f",
    "00004040000080bf0000003f",
    "000080bf000040400000003f",
    // The same triangle at z = 0.9 (0x3f666666).
    "000080bf000080bf6666663f",
    "00004040000080bf6666663f",
    "000080bf000040406666663f",
];
/// The two reviewed tints, as the `float32x4` bytes the stream carries:
/// `(1, 0, 0, 1)` for the near triangle and `(0, 1, 0, 1)` for the far one.
const DEPTH_PAIR_RED_HEX: &str = "0000803f00000000000000000000803f";
const DEPTH_PAIR_GREEN_HEX: &str = "000000000000803f000000000000803f";

/// The reviewed device-gated depth-resolve stream (`research/docs/23` §3.3,
/// v57d): the same `depth32float` pair module over a *different* near
/// triangle, so the Min and Max reductions of the stored four-sample texels
/// disagree. The near triangle covers `x <= 0.25` NDC — the v51 edge shape,
/// which leaves the third texel column half covered — and the far triangle
/// covers the whole viewport; both carry the red tint, so the colour
/// observation stays uniform and the two landings differ only in the depth
/// resolve's own reduction.
const DEPTH_RESOLVE_EDGE_POSITIONS_HEX: [&str; 6] = [
    // The half-plane triangle (0.25, -1), (0.25, 3), (-3, -1) at z = 0.5:
    // its vertical right edge sits at x = 0.25 NDC, the v51 edge fixture's own
    // boundary, and its hypotenuse stays left of the viewport.
    "0000803e000080bf0000003f",
    "0000803e000040400000003f",
    "000040c0000080bf0000003f",
    // The oversize full-screen triangle at z = 0.9.
    "000080bf000080bf6666663f",
    "00004040000080bf6666663f",
    "000080bf000040406666663f",
];

/// The reviewed stream, reassembled vertex by vertex: each vertex is its
/// `float32x3` position at offset 0, the four padding bytes that align the tint
/// to offset 16, and the triangle's `float32x4` tint there — the exact offsets
/// the reviewed layout declares.
fn reviewed_depth_stream_hex() -> String {
    let mut expected = String::new();
    for (vertex, position) in DEPTH_PAIR_POSITIONS_HEX.iter().enumerate() {
        expected.push_str(position);
        expected.push_str("00000000");
        expected.push_str(if vertex < 3 {
            DEPTH_PAIR_RED_HEX
        } else {
            DEPTH_PAIR_GREEN_HEX
        });
    }
    expected
}

/// The device-gated pair stream, reassembled the same way: the near
/// half-plane triangle and the far full-screen one both carry the red tint,
/// because the gate's whole point is that the two filters disagree about the
/// *depth* reduction while every colour texel stays the one uniform output.
fn reviewed_depth_resolve_edge_stream_hex() -> String {
    let mut expected = String::new();
    for position in DEPTH_RESOLVE_EDGE_POSITIONS_HEX {
        expected.push_str(position);
        expected.push_str("00000000");
        expected.push_str(DEPTH_PAIR_RED_HEX);
    }
    expected
}

/// The reviewed rail-owned combined depth-stencil pair's stream
/// (`research/docs/23` §3.3, v66): three copies of the v51 edge triangle —
/// the half-plane one whose vertical edge sits at `x = 0.25` NDC — at
/// `z = 0.5` (red), `z = 0.9` (green) and `z = 0.4` (blue). The one shared
/// coverage is what makes the two faces' tests decide per sample: the first
/// triangle's depth pass writes depth, the second triangle's depth failure
/// writes stencil, and the third is tested against it.
///
/// The first triangle's tint is the reviewed red, whose channels are all zero
/// or one: the colour observation's every byte is then either the tint, the
/// clear, or the exact two-of-four mean of the two — no channel sits on a
/// rounding tie, which is what keeps the expectation independent of a driver's
/// UNORM conversion rule.
const COMBINED_PAIR_POSITIONS_HEX: [&str; 9] = [
    // The half-plane triangle (0.25, -1), (0.25, 3), (-3, -1) at z = 0.5.
    "0000803e000080bf0000003f",
    "0000803e000040400000003f",
    "000040c0000080bf0000003f",
    // The same triangle at z = 0.9 (0x3f666666).
    "0000803e000080bf6666663f",
    "0000803e000040406666663f",
    "000040c0000080bf6666663f",
    // The same triangle at z = 0.4 (0x3ecccccd).
    "0000803e000080bfcdcccc3e",
    "0000803e00004040cdcccc3e",
    "000040c0000080bfcdcccc3e",
];
/// The third reviewed tint, as the `float32x4` bytes the stream carries:
/// `(0, 0, 1, 1)` for the triangle that tests the stencil the second one
/// wrote. The first two reuse the reviewed depth pair's own red and green.
const COMBINED_PAIR_BLUE_HEX: &str = "00000000000000000000803f0000803f";
/// The reviewed combined pair's index buffer: the nine vertices in triangle
/// order.
const COMBINED_PAIR_INDICES_HEX: &str = "000001000200030004000500060007000800";

/// The reviewed combined pair's stream, reassembled vertex by vertex: each
/// vertex is its `float32x3` position at offset 0, the four padding bytes that
/// align the tint to offset 16, and its triangle's `float32x4` tint there.
fn reviewed_combined_pair_stream_hex() -> String {
    let mut expected = String::new();
    for (vertex, position) in COMBINED_PAIR_POSITIONS_HEX.iter().enumerate() {
        expected.push_str(position);
        expected.push_str("00000000");
        expected.push_str(match vertex / 3 {
            0 => DEPTH_PAIR_RED_HEX,
            1 => DEPTH_PAIR_GREEN_HEX,
            _ => COMBINED_PAIR_BLUE_HEX,
        });
    }
    expected
}

/// The reviewed blend fixture (`research/docs/23` §3.3, v40): one oversize
/// triangle whose tint is `(64/255, 128/255, 192/255, 128/255)`. With the
/// reviewed blend state — source alpha against one-minus-source-alpha — over a
/// cleared-to-zero attachment, the stored texel is
/// `(32, 64, 96, 64)`, and none of those four values sits on a half-integer
/// UNORM tie (`research/docs/23` §3.5): the four products are 32.125, 64.25,
/// 96.376 and 64.25 before rounding.
const BLEND_PAIR_TINT_HEX: &str = "8180803e8180003fc1c0403f8180003f";

/// The reviewed cull stream (`research/docs/23` §3.3, v39): the same oversize
/// triangle twice, at one depth. The first copy's vertex order is clockwise in
/// framebuffer coordinates and the second copy's is its reverse, so a pass that
/// culls back faces with a counter-clockwise front keeps exactly the second
/// one. Each copy carries its own tint, which is what makes the surviving one
/// visible.
const CULL_PAIR_POSITIONS_HEX: [&str; 6] = [
    // (-1, -1, 0.5), (3, -1, 0.5), (-1, 3, 0.5)
    "000080bf000080bf0000003f",
    "00004040000080bf0000003f",
    "000080bf000040400000003f",
    // The same three corners in reverse order: (3, -1), (-1, 3), (-1, -1)
    "00004040000080bf0000003f",
    "000080bf000040400000003f",
    "000080bf000080bf0000003f",
];

fn reviewed_cull_stream_hex() -> String {
    let mut expected = String::new();
    for (vertex, position) in CULL_PAIR_POSITIONS_HEX.iter().enumerate() {
        expected.push_str(position);
        expected.push_str("00000000");
        expected.push_str(if vertex < 3 {
            DEPTH_PAIR_RED_HEX
        } else {
            DEPTH_PAIR_GREEN_HEX
        });
    }
    expected
}

/// Pin the reviewed cull shape (`research/docs/23` §3.3, v39).
///
/// The layout, the two opposite-order triangles, the culling state and the
/// expectation are the whole review surface: the pass has to cull back faces
/// with a counter-clockwise front face, the stream has to be the two reviewed
/// copies of the oversize triangle, and the expectation is the tint of the
/// copy that survives — the green one, whose reversed order is the
/// counter-clockwise (front) one under that state. The reflected derivation
/// matters: the first copy's order is clockwise in the framebuffer, which is
/// what the local control run reads back (the first copy's red never lands).
fn reviewed_cull_geometry(
    case: &RenderCase,
    layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    let stream = &layout.buffers[0];
    if layout.buffers.len() != 1
        || stream.stride != DEPTH_STRIDE
        || stream.step != "per_vertex"
        || stream.attributes.len() != 2
    {
        return Err(format!(
            "{where_}: the reviewed cull stream is one stride-{DEPTH_STRIDE} stream with two attributes"
        )
        .into());
    }
    let position = &stream.attributes[0];
    if position.location != 0 || position.offset != 0 || position.format != "float32x3" {
        return Err(format!(
            "{where_}: the reviewed cull position is location 0, offset 0, float32x3"
        )
        .into());
    }
    let tint = &stream.attributes[1];
    if tint.location != 1 || tint.offset != 16 || tint.format != "float32x4" {
        return Err(format!(
            "{where_}: the reviewed cull tint is location 1, offset 16, float32x4"
        )
        .into());
    }
    let Some(cull) = &case.cull else {
        return Err(format!("{where_}: the reviewed cull shape carries a culling state").into());
    };
    if cull.mode != "back" || cull.winding != "counter_clockwise" {
        return Err(format!(
            "{where_}: the reviewed cull state is a back-face cull with a counter-clockwise front"
        )
        .into());
    }
    if case.depth.is_some() {
        return Err(
            format!("{where_}: the reviewed cull shape carries no depth attachment").into(),
        );
    }
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the reviewed cull shape binds one stream").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    if buffer.length != DEPTH_STRIDE * 6 {
        return Err(format!(
            "{where_}: the reviewed cull stream is six stride-{DEPTH_STRIDE} vertices"
        )
        .into());
    }
    if buffer.initial_hex != reviewed_cull_stream_hex() {
        return Err(format!(
            "{where_}: the reviewed cull stream is the two opposite-order triangles"
        )
        .into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed cull shape is indexed").into());
    };
    if indices.initial_hex != "000001000200030004000500" {
        return Err(
            format!("{where_}: the reviewed cull indices are the two reviewed triangles").into(),
        );
    }
    Ok(RenderGeometry::CullPair)
}

/// Pin the reviewed blend shape (`research/docs/23` §3.3, v40).
///
/// The layout and module are the reviewed pair's; the shape adds one oversize
/// triangle whose tint is the reviewed four bytes, the reviewed blend state, a
/// cleared-to-zero attachment and the expectation those two imply. The state is
/// what the bytes measure: a rail that ignored it would store the tint itself
/// (`4080c080`), and one that swapped the factors would store the clear colour.
fn reviewed_blend_geometry(
    case: &RenderCase,
    layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    let stride = DEPTH_STRIDE;
    let stream = &layout.buffers[0];
    if layout.buffers.len() != 1
        || stream.stride != stride
        || stream.step != "per_vertex"
        || stream.attributes.len() != 2
    {
        return Err(format!(
            "{where_}: the reviewed blend stream is one stride-{stride} stream with two attributes"
        )
        .into());
    }
    let position = &stream.attributes[0];
    if position.location != 0 || position.offset != 0 || position.format != "float32x3" {
        return Err(format!(
            "{where_}: the reviewed blend position is location 0, offset 0, float32x3"
        )
        .into());
    }
    let tint = &stream.attributes[1];
    if tint.location != 1 || tint.offset != 16 || tint.format != "float32x4" {
        return Err(format!(
            "{where_}: the reviewed blend tint is location 1, offset 16, float32x4"
        )
        .into());
    }
    if case.depth.is_some() || case.cull.is_some() {
        return Err(format!(
            "{where_}: the reviewed blend shape carries neither a depth attachment nor a culling state"
        )
        .into());
    }
    let Some(blend) = &case.blend else {
        return Err(format!("{where_}: the reviewed blend shape carries a blend state").into());
    };
    let [attachment] = blend.as_slice() else {
        return Err(format!("{where_}: the reviewed blend shape states one attachment").into());
    };
    if (
        attachment.source_rgb.as_str(),
        attachment.destination_rgb.as_str(),
        attachment.source_alpha.as_str(),
        attachment.destination_alpha.as_str(),
        attachment.operation.as_str(),
    ) != (
        "source_alpha",
        "one_minus_source_alpha",
        "source_alpha",
        "one_minus_source_alpha",
        "add",
    ) {
        return Err(format!(
            "{where_}: the reviewed blend state is source alpha against one-minus-source-alpha with an add"
        )
        .into());
    }
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the reviewed blend shape binds one stream").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    if buffer.length != stride * 3 {
        return Err(format!(
            "{where_}: the reviewed blend stream is three stride-{stride} vertices"
        )
        .into());
    }
    // One oversize triangle: the reviewed position bytes followed by the
    // reviewed tint, with the alignment padding the layout declares.
    let mut expected = String::new();
    for position in &CULL_PAIR_POSITIONS_HEX[..3] {
        expected.push_str(position);
        expected.push_str("00000000");
        expected.push_str(BLEND_PAIR_TINT_HEX);
    }
    if buffer.initial_hex != expected {
        return Err(format!("{where_}: the reviewed blend stream is the reviewed triangle").into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed blend shape is indexed").into());
    };
    if indices.initial_hex != "000001000200" {
        return Err(
            format!("{where_}: the reviewed blend indices are the reviewed triangle").into(),
        );
    }
    Ok(RenderGeometry::BlendTriangle)
}

/// Pin the reviewed depth shape (`research/docs/23` §3.3, v36).
///
/// The layout, the two triangles, the depth attachment's shape and the depth
/// state are the whole review surface: the attachment has to be the reviewed
/// `depth32float` extent cleared to one, the test has to be `less` with writes
/// on, and the stream has to be the two oversize triangles at the reviewed
/// depths with the reviewed tints. Anything else describes a shape no rail has
/// been reviewed against.
fn reviewed_depth_geometry(
    case: &RenderCase,
    layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    // The stencil sibling masks with a stencil attachment instead of a depth
    // one (`research/docs/23` §3.3, v47): the same pair stream, one rail-owned
    // `stencil8` surface, and exactly the reviewed state. The two surfaces
    // combine through the combined depth-stencil surface — one attachment both
    // faces share — in either of the two reviewed shapes: the v60 resolve
    // shape, which states the stencil resolve its stored faces land through,
    // or the v66 rail-owned write-then-test pair. A case that declares both
    // with a lopsided store decision is refused rather than classified as
    // either (`research/docs/23` §3.3, v60/v66).
    if case.stencil.is_some() {
        if case.stencil_test.is_none() {
            return Err(
                format!("{where_}: the reviewed stencil shape carries a stencil test").into(),
            );
        }
    } else if case.stencil_test.is_some() {
        return Err(format!("{where_}: a stencil test needs a stencil attachment").into());
    }
    if let Some(stencil) = &case.stencil {
        if stencil.format != "stencil8" || stencil.load != "clear" || stencil.clear_value != Some(0)
        {
            return Err(format!(
                "{where_}: the reviewed stencil attachment is a stencil8 surface cleared to zero"
            )
            .into());
        }
        if stencil.width != 4 || stencil.height != 4 {
            return Err(format!("{where_}: the reviewed stencil attachment is 4x4 texels").into());
        }
        let Some(test) = &case.stencil_test else {
            unreachable!("the presence rule above proved the state is there");
        };
        // The v66 rail-owned pair's write-then-test state (`research/docs/23`
        // §3.3, v66) is its own reviewed shape: the equal-zero test keeps the
        // value on pass and increments-wraps it on depth failure, so the
        // second triangle's depth failure is what the third triangle then
        // fails against. It is classified before the single-surface states,
        // and the rest of its review (stream, indices, both faces' clears and
        // the shared coverage) lives in `reviewed_combined_pair_geometry`.
        if case.depth.is_some() && case.stencil_resolve.is_none() {
            let reviewed_state = test.compare == "equal"
                && test.reference == 0
                && test.read_mask == 0xff
                && test.write_mask == 0xff
                && test.fail_op == "keep"
                && test.depth_fail_op == "increment_wrap"
                && test.pass_op == "keep";
            if !reviewed_state {
                return Err(format!(
                    "{where_}: the rail-owned combined pair's stencil state is the equal-zero \
                     test that keeps on pass and increments-wraps on depth failure, with both \
                     masks wide open"
                )
                .into());
            }
            return reviewed_combined_pair_geometry(case, where_);
        }
        let reviewed_state = (test.compare == "equal"
            || (test.compare == "always"
                && case.depth.is_some()
                && case.stencil_resolve.is_some()))
            && test.reference == 0
            && test.read_mask == 0xff
            && test.write_mask == 0xff
            && test.fail_op == "keep"
            && test.depth_fail_op == "keep"
            && test.pass_op == "increment_wrap";
        if !reviewed_state {
            return Err(format!(
                "{where_}: the reviewed stencil state is the equal-zero test or the combined \
                 shape's always test, both keeping the stored value on failure and incrementing \
                 it with wraparound on pass"
            )
            .into());
        }
        // The stencil readback pair (`research/docs/23` §3.3, v49): a pass that
        // keeps its stencil surface states its store action, where the texels
        // land and what the readback has to contain, and states all three
        // together. The expectation must be the surface's own one-byte-per-texel
        // extent and must differ from the clear value, or "the store ran" and
        // "the surface was never written" would read back the same bytes.
        let stencil_extent = stencil
            .width
            .checked_mul(stencil.height)
            .ok_or("stencil extent overflows")?;
        match (&stencil.store, &stencil.expected_hex) {
            (None, None) => {
                if stencil.allocation.is_some() || stencil.view.is_some() {
                    return Err(format!(
                        "{where_}: a discarded stencil attachment carries no identity or expectation"
                    )
                    .into());
                }
            }
            (Some(store), Some(expected_hex)) => {
                if store != "store" {
                    return Err(format!(
                        "{where_}: the only stencil store action is \"store\", got {store:?}"
                    )
                    .into());
                }
                let (Some(allocation), Some(view)) = (stencil.allocation, stencil.view) else {
                    return Err(format!(
                        "{where_}: a stored stencil attachment needs its allocation and view"
                    )
                    .into());
                };
                if allocation == 0 || view == 0 {
                    return Err(format!("{where_}: zero stencil identity").into());
                }
                let texels = unhex(expected_hex)?;
                if texels.len() as u64 != stencil_extent {
                    return Err(format!(
                        "{where_}: the expected stencil texels do not match the stencil extent"
                    )
                    .into());
                }
                let clear = stencil.clear_value.unwrap_or(0);
                if texels.iter().all(|texel| *texel == clear) {
                    return Err(format!(
                        "{where_}: the expected stencil texels equal the clear value"
                    )
                    .into());
                }
                if let Some(attachment) = &case.attachment {
                    if attachment.allocation == allocation && attachment.view == view {
                        return Err(format!(
                            "{where_}: the stencil identity has to differ from the colour attachment"
                        )
                        .into());
                    }
                }
            }
            _ => {
                return Err(format!(
                    "{where_}: the stencil store action, its identity and its expectation travel \
                     together"
                )
                .into())
            }
        }
    }
    let stream = &layout.buffers[0];
    if layout.buffers.len() != 1
        || stream.stride != DEPTH_STRIDE
        || stream.step != "per_vertex"
        || stream.attributes.len() != 2
    {
        return Err(format!(
            "{where_}: the reviewed depth stream is one stride-{DEPTH_STRIDE} stream with two attributes"
        )
        .into());
    }
    let position = &stream.attributes[0];
    if position.location != 0 || position.offset != 0 || position.format != "float32x3" {
        return Err(format!(
            "{where_}: the reviewed depth position is location 0, offset 0, float32x3"
        )
        .into());
    }
    let tint = &stream.attributes[1];
    if tint.location != 1 || tint.offset != 16 || tint.format != "float32x4" {
        return Err(format!(
            "{where_}: the reviewed depth tint is location 1, offset 16, float32x4"
        )
        .into());
    }
    // From here on the review is the *depth* half of the shape: the stencil
    // sibling's own rules are stated above, and the stream and index checks
    // below are shared by both.
    let Some(depth) = &case.depth else {
        return reviewed_stencil_geometry_stream(case, layout, where_);
    };
    if depth.format != "depth32float" || depth.load != "clear" {
        return Err(
            format!("{where_}: the reviewed depth attachment is a cleared depth32float").into(),
        );
    }
    // The combined shape clears to the value between the two triangles'
    // depths (0.7), which is what lets the near triangle's depth pass write
    // stencil while the far triangle's depth failures leave the rest at zero
    // (`research/docs/23` §3.3, v60); the single-surface depth pair keeps the
    // reviewed clear of one.
    let reviewed_clear = if case.stencil.is_some() {
        COMBINED_DEPTH_CLEAR
    } else {
        DEPTH_CLEAR
    };
    if depth.clear_depth != Some(reviewed_clear) {
        return Err(format!("{where_}: the reviewed depth clear is {reviewed_clear}").into());
    }
    // The depth readback pair (`research/docs/23` §3.3, v43): a pass that keeps
    // its depth surface states its store action, where the texels land and what
    // the readback has to contain, and states all three together. The
    // expectation is what makes the channel falsifiable: it must be the depth
    // extent's own bytes and must differ from the clear value, or "the store
    // ran" and "the surface was never written" would read back the same bytes.
    let depth_extent = depth
        .width
        .checked_mul(depth.height)
        .and_then(|texels| texels.checked_mul(4))
        .ok_or("depth extent overflows")?;
    match (&depth.store, &depth.expected_hex) {
        (None, None) => {
            if depth.allocation.is_some() || depth.view.is_some() {
                return Err(format!(
                    "{where_}: a discarded depth attachment carries no identity or expectation"
                )
                .into());
            }
        }
        (Some(store), Some(expected_hex)) => {
            if store != "store" {
                return Err(format!(
                    "{where_}: the only depth store action is \"store\", got {store:?}"
                )
                .into());
            }
            let (Some(allocation), Some(view)) = (depth.allocation, depth.view) else {
                return Err(format!(
                    "{where_}: a stored depth attachment needs its allocation and view"
                )
                .into());
            };
            if allocation == 0 || view == 0 {
                return Err(format!("{where_}: zero depth identity").into());
            }
            let texels = unhex(expected_hex)?;
            if texels.len() as u64 != depth_extent {
                return Err(format!(
                    "{where_}: the expected depth texels do not match the depth extent"
                )
                .into());
            }
            let clear_texel = (DEPTH_CLEAR as f32).to_le_bytes();
            if texels
                .chunks_exact(4)
                .all(|texel| texel == clear_texel.as_slice())
            {
                return Err(
                    format!("{where_}: the expected depth texels equal the clear depth").into(),
                );
            }
            // The depth surface is its own raster, so its landing cannot be the
            // colour attachment's view: one view carrying two different
            // attachments' bytes could not be compared at all.
            if let Some(attachment) = &case.attachment {
                if attachment.allocation == allocation && attachment.view == view {
                    return Err(format!(
                        "{where_}: the depth identity has to differ from the colour attachment"
                    )
                    .into());
                }
            }
        }
        _ => {
            return Err(format!(
                "{where_}: the depth store action, its identity and its expectation travel together"
            )
            .into())
        }
    }
    let Some(test) = &case.depth_test else {
        return Err(format!("{where_}: the reviewed depth shape carries a depth test").into());
    };
    if test.compare != "less" || !test.write {
        return Err(
            format!("{where_}: the reviewed depth state is a less test with writes on").into(),
        );
    }
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the reviewed depth shape binds one stream").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let required = DEPTH_STRIDE * 6;
    if buffer.length != required {
        return Err(format!(
            "{where_}: the reviewed depth stream is six stride-{DEPTH_STRIDE} vertices"
        )
        .into());
    }
    // The position and tint bytes share each vertex, so the expectation is
    // reassembled from the reviewed pieces rather than read off one constant.
    // The device-gated pair (`research/docs/23` §3.3, v57d) is the second
    // reviewed stream: it is admitted only beside a Min or Max resolve, because
    // the edge shape exists to make those two reductions disagree — a
    // full-coverage pair cannot, and Sample0 leaves the review surface as the
    // pre-v57d fixture. The v60 stencil-resolve pair reuses the same edge
    // geometry: its mixed column is what makes the two stencil filters
    // disagree, so a case that states a stencil resolve also admits the edge
    // stream (`research/docs/23` §3.3, v60).
    if buffer.initial_hex == reviewed_depth_resolve_edge_stream_hex() {
        match case
            .depth_resolve
            .as_ref()
            .map(|resolve| resolve.filter.as_str())
        {
            Some("min" | "max") => {}
            _ if case.stencil_resolve.is_some() => {}
            _ => {
                return Err(format!(
                    "{where_}: the edge depth stream is the min/max device-gated shape"
                )
                .into())
            }
        }
    } else if buffer.initial_hex != reviewed_depth_stream_hex() {
        return Err(
            format!("{where_}: the reviewed depth stream is the two reviewed triangles").into(),
        );
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed depth shape is indexed").into());
    };
    if indices.initial_hex != "000001000200030004000500" {
        return Err(
            format!("{where_}: the reviewed depth indices are the two reviewed triangles").into(),
        );
    }
    Ok(RenderGeometry::DepthPair)
}

/// Pin the v66 rail-owned combined depth-stencil pair (`research/docs/23`
/// §3.3, v66).
///
/// The whole review surface is the three oversize triangles with their one
/// coverage, the two rail-owned faces behind the one combined attachment and
/// the pair's own clears: the depth face opens from one, the stencil face from
/// zero, neither states a store action, an identity or an expectation — the
/// surfaces leave with the pass — and the two tests are the reviewed `less`
/// test with writes on and the write-then-test stencil state the caller already
/// pinned. Anything else describes a shape no rail has been reviewed against.
fn reviewed_combined_pair_geometry(case: &RenderCase, where_: &str) -> Result<RenderGeometry> {
    let Some(depth) = &case.depth else {
        return Err(format!("{where_}: the combined pair opens its depth face").into());
    };
    let Some(stencil) = &case.stencil else {
        return Err(format!("{where_}: the combined pair opens its stencil face").into());
    };
    if depth.format != "depth32float"
        || depth.load != "clear"
        || depth.clear_depth != Some(DEPTH_CLEAR)
    {
        return Err(format!(
            "{where_}: the combined pair's depth face is a depth32float surface cleared to one"
        )
        .into());
    }
    if stencil.format != "stencil8" || stencil.load != "clear" || stencil.clear_value != Some(0) {
        return Err(format!(
            "{where_}: the combined pair's stencil face is a stencil8 surface cleared to zero"
        )
        .into());
    }
    if depth.width != 4 || depth.height != 4 || stencil.width != 4 || stencil.height != 4 {
        return Err(format!("{where_}: the combined pair's faces are 4x4 texels").into());
    }
    // Both faces share one surface and both are rail-owned: neither may name a
    // store action, an identity or a landing (`research/docs/23` §3.3, v66).
    if depth.store.is_some()
        || depth.allocation.is_some()
        || depth.view.is_some()
        || depth.expected_hex.is_some()
        || stencil.store.is_some()
        || stencil.allocation.is_some()
        || stencil.view.is_some()
        || stencil.expected_hex.is_some()
    {
        return Err(format!(
            "{where_}: the combined pair's faces are rail-owned and carry no store action, \
             identity or expectation"
        )
        .into());
    }
    if case
        .depth_test
        .as_ref()
        .is_none_or(|test| test.compare != "less" || !test.write)
    {
        return Err(format!(
            "{where_}: the combined pair's depth state is a less test with writes on"
        )
        .into());
    }
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the combined pair binds one stream").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    if buffer.length != DEPTH_STRIDE * 9 {
        return Err(format!(
            "{where_}: the reviewed combined pair stream is nine stride-{DEPTH_STRIDE} vertices"
        )
        .into());
    }
    if buffer.initial_hex != reviewed_combined_pair_stream_hex() {
        return Err(format!(
            "{where_}: the reviewed combined pair stream is the three reviewed triangles"
        )
        .into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed combined pair is indexed").into());
    };
    if indices.initial_hex != COMBINED_PAIR_INDICES_HEX {
        return Err(format!(
            "{where_}: the reviewed combined pair indices are the three reviewed triangles"
        )
        .into());
    }
    Ok(RenderGeometry::DepthPair)
}

/// The tail of the reviewed pair review for the stencil sibling
/// (`research/docs/23` §3.3, v47): the stream and attribute checks already ran
/// in [`reviewed_depth_geometry`], so what is left is the caller-held stream's
/// own bytes and the two-triangle index shape both pair fixtures pin.
fn reviewed_stencil_geometry_stream(
    case: &RenderCase,
    _layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    if case.vertex_buffers.len() != 1 {
        return Err(format!("{where_}: the reviewed stencil shape binds one stream").into());
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let required = DEPTH_STRIDE * 6;
    if buffer.length != required {
        return Err(format!(
            "{where_}: the reviewed stencil stream is six stride-{DEPTH_STRIDE} vertices"
        )
        .into());
    }
    if buffer.initial_hex != reviewed_depth_stream_hex() {
        return Err(
            format!("{where_}: the reviewed stencil stream is the two reviewed triangles").into(),
        );
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed stencil shape is indexed").into());
    };
    if indices.initial_hex != "000001000200030004000500" {
        return Err(format!(
            "{where_}: the reviewed stencil indices are the two reviewed triangles"
        )
        .into());
    }
    Ok(RenderGeometry::DepthPair)
}

/// Pin the reviewed instanced shape (`research/docs/23` §3.3, v31).
///
/// The pair is the whole review surface: the position stream has to be the
/// reviewed quad corners, the second binding has to be a per-instance
/// `float32x4` tint stream carrying exactly the red and green records, the draw
/// has to run exactly two instances, and the index buffer has to select the
/// reviewed quad. Anything else describes a shape no rail has been reviewed
/// against, so it is refused before a device object exists.
fn reviewed_instanced_geometry(
    case: &RenderCase,
    layout: &VertexLayoutDefinition,
    where_: &str,
) -> Result<RenderGeometry> {
    if case.vertex_buffers.len() != 2 {
        return Err(format!(
            "{where_}: the reviewed instanced shape binds two streams and two bindings"
        )
        .into());
    }
    let position = &layout.buffers[0];
    if position.stride != QUAD_STRIDE
        || position.step != "per_vertex"
        || position.attributes.len() != 1
    {
        return Err(format!(
            "{where_}: the reviewed instanced position stream is one float32x2 at stride {QUAD_STRIDE}"
        )
        .into());
    }
    let attribute = &position.attributes[0];
    if attribute.location != 0 || attribute.offset != 0 || attribute.format != "float32x2" {
        return Err(format!(
            "{where_}: the reviewed instanced position attribute is location 0, offset 0, float32x2"
        )
        .into());
    }
    let tint = &layout.buffers[1];
    if tint.stride != INSTANCED_TINT_STRIDE
        || tint.step != "per_instance"
        || tint.attributes.len() != 1
    {
        return Err(format!(
            "{where_}: the reviewed instanced tint stream is one float32x4 at stride \
             {INSTANCED_TINT_STRIDE}, stepped per instance"
        )
        .into());
    }
    let attribute = &tint.attributes[0];
    if attribute.location != 1 || attribute.offset != 0 || attribute.format != "float32x4" {
        return Err(format!(
            "{where_}: the reviewed instanced tint attribute is location 1, offset 0, float32x4"
        )
        .into());
    }
    if case.instance_count != INSTANCED_COUNT {
        return Err(format!(
            "{where_}: the reviewed instanced draw runs exactly {INSTANCED_COUNT} instances"
        )
        .into());
    }
    let positions = &case.vertex_buffers[0];
    if positions.allocation == 0 || positions.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let required = QUAD_STRIDE * QUAD_VERTICES;
    if positions.length != required || positions.initial_hex != INSTANCED_POSITION_HEX {
        return Err(format!(
            "{where_}: the reviewed instanced position stream is the reviewed quad corners"
        )
        .into());
    }
    let tints = &case.vertex_buffers[1];
    if tints.allocation == 0 || tints.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let expected_tints = format!("{INSTANCED_TINT_RED_HEX}{INSTANCED_TINT_GREEN_HEX}");
    if tints.length != INSTANCED_TINT_STRIDE * INSTANCED_COUNT
        || tints.initial_hex != expected_tints
    {
        return Err(format!(
            "{where_}: the reviewed instanced tint stream is the reviewed red and green records"
        )
        .into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed instanced shape is indexed").into());
    };
    if indices.allocation == 0 || indices.view == 0 {
        return Err(format!("{where_}: zero index buffer identity").into());
    }
    let width = match indices.format.as_str() {
        "uint16" => 2_u64,
        "uint32" => 4,
        other => return Err(format!("{where_}: unsupported index format {other:?}").into()),
    };
    let required = QUAD_INDICES * width;
    if indices.length != required {
        return Err(format!(
            "{where_}: the reviewed instanced index buffer is {QUAD_INDICES} indices wide"
        )
        .into());
    }
    let bytes = unhex(&indices.initial_hex)?;
    let width = usize::try_from(width)?;
    if bytes.len() != usize::try_from(indices.length)? {
        return Err(format!("{where_}: the index bytes do not match their length").into());
    }
    for (position, chunk) in bytes.chunks_exact(width).enumerate() {
        let index = match width {
            2 => u64::from(u16::from_le_bytes([chunk[0], chunk[1]])),
            _ => u64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        };
        if index >= QUAD_VERTICES {
            return Err(format!(
                "{where_}: index {position} names vertex {index}, outside the reviewed quad"
            )
            .into());
        }
    }
    Ok(RenderGeometry::InstancedPair)
}

/// The reviewed render-sampler shape (`research/docs/23` §3.3, v70).
///
/// The case's own claim is the identity: exactly one `rgba8_unorm` texture
/// whose extent equals the stored attachment's, and an expectation that *is*
/// that texture's uploaded texels, byte for byte. The texels have to be
/// pairwise distinct and differ from the clear colour, so a rail that ignores
/// the binding (clear), filters it (a neighbour's texel), or flips/transposes
/// the uv (another row or column) cannot pass.
fn reviewed_sampled_geometry(
    case: &RenderCase,
    textures: &[FragmentTextureDefinition],
    where_: &str,
) -> Result<RenderGeometry> {
    if textures.len() != 1 {
        return Err(format!(
            "{where_}: the reviewed render-sampler shape binds exactly one texture"
        )
        .into());
    }
    if case.vertex_layout.is_some() || !case.vertex_buffers.is_empty() || case.indices.is_some() {
        return Err(format!(
            "{where_}: the reviewed render-sampler shape is the vertex_id geometry and binds no \
             vertex stream or index buffer"
        )
        .into());
    }
    if case.attachments.is_some() {
        return Err(
            format!("{where_}: the reviewed render-sampler shape stores one attachment").into(),
        );
    }
    let attachment = case.attachment.as_ref().ok_or_else(|| {
        format!("{where_}: the reviewed render-sampler shape needs its stored attachment")
    })?;
    if attachment.format != "rgba8_unorm" || attachment.store != "store" {
        return Err(format!(
            "{where_}: the reviewed render-sampler shape stores one rgba8_unorm attachment"
        )
        .into());
    }
    let texture = &textures[0];
    if texture.format != "rgba8_unorm" {
        return Err(format!(
            "{where_}.fragment_textures[0]: the reviewed sampling stage reads one rgba8_unorm \
             surface"
        )
        .into());
    }
    if texture.width != attachment.width || texture.height != attachment.height {
        return Err(format!(
            "{where_}.fragment_textures[0]: the sampled texture has to share the attachment's \
             extent, so every fragment stands on a texel centre"
        )
        .into());
    }
    // The texels the texture is uploaded with: the hex spelling for every
    // pre-R5a case, the reviewed per-texel rule for the wide one (R5a,
    // `research/docs/23` §73). Both forms already resolve to the extent's own
    // byte count, so a short or long spelling cannot reach the checks below.
    let texels = texture.texels()?;
    let clear = unhex(attachment.clear_hex.as_deref().unwrap_or_default())?;
    if attachment.load != "clear" || clear.len() != 4 {
        return Err(format!(
            "{where_}.attachment: the reviewed render-sampler shape clears its attachment, so a \
             rail that ignores the texture is observable"
        )
        .into());
    }
    // The rule form: the fixture cannot spell four million texels, so it states
    // the function instead. The rule is the case's own expectation, it stays
    // injective over the extent it addresses (`x` and `y` as two little-endian
    // `u16`s), and the clear colour stays outside its reach — the closed-form
    // sibling of the pairwise-distinctness scan below, which is exactly what
    // the rule form cannot run on a 2048² plane.
    if let Some(rule) = texture.rule() {
        if texture.width < RULE_MIN_DIMENSION || texture.height < RULE_MIN_DIMENSION {
            return Err(format!(
                "{where_}.fragment_textures[0]: a texel rule is the megapixel form; {}×{} \
                     states its texels instead",
                texture.width, texture.height
            )
            .into());
        }
        if texture.width > RULE_ADDRESS_CEILING || texture.height > RULE_ADDRESS_CEILING {
            return Err(format!(
                "{where_}.fragment_textures[0]: the reviewed rule addresses at most \
                     {RULE_ADDRESS_CEILING} texels per axis"
            )
            .into());
        }
        if case.expected_rule.as_deref() != Some(rule) {
            return Err(format!(
                "{where_}: the expectation has to be the texture's own rule: the sampling \
                     stage's sample at a texel centre is an identity copy"
            )
            .into());
        }
        let windows = case.readback_windows.as_deref().ok_or_else(|| {
            format!("{where_}: a rule-expected attachment needs its readback windows")
        })?;
        if windows.is_empty() || windows.len() > MAX_READBACK_WINDOWS {
            return Err(format!(
                "{where_}.readback_windows: one to {MAX_READBACK_WINDOWS} windows"
            )
            .into());
        }
        for (index, window) in windows.iter().enumerate() {
            let window_where = format!("{where_}.readback_windows[{index}]");
            if window.width == 0 || window.height == 0 {
                return Err(format!("{window_where}: an empty readback window").into());
            }
            if window.width > MAX_READBACK_WINDOW_DIMENSION
                || window.height > MAX_READBACK_WINDOW_DIMENSION
            {
                return Err(format!(
                    "{window_where}: a readback window is at most \
                         {MAX_READBACK_WINDOW_DIMENSION} texels per axis"
                )
                .into());
            }
            let right = window.x.checked_add(window.width);
            let bottom = window.y.checked_add(window.height);
            if right.is_none_or(|end| end > texture.width)
                || bottom.is_none_or(|end| end > texture.height)
            {
                return Err(format!(
                    "{window_where}: the readback window leaves the attachment plane"
                )
                .into());
            }
        }
        if rule_reaches_colour(rule, &clear, texture.width, texture.height)? {
            return Err(format!(
                "{where_}.attachment: a rule texel equals the clear colour, so a rail that \
                     ignored the draw could pass"
            )
            .into());
        }
        return Ok(RenderGeometry::SampledTexture);
    }
    if case.expected_rule.is_some() || case.readback_windows.is_some() {
        return Err(format!(
            "{where_}: a rule expectation and its readback windows travel with the texture's own \
             texel rule"
        )
        .into());
    }
    let expected = unhex(case.expected_hex.as_deref().ok_or_else(|| {
        format!("{where_}: the reviewed render-sampler shape needs its expectation")
    })?)?;
    if expected != texels {
        return Err(format!(
            "{where_}: the expectation has to be the uploaded texels: the sampling stage's \
             sample at a texel centre is an identity copy"
        )
        .into());
    }
    let unique = texels
        .chunks_exact(4)
        .map(|texel| texel.to_vec())
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != texels.len() / 4 {
        return Err(format!(
            "{where_}.fragment_textures[0]: the uploaded texels have to be pairwise distinct, or \
             a repeated read could pass"
        )
        .into());
    }
    if unique.contains(&clear) {
        return Err(format!(
            "{where_}.fragment_textures[0]: an uploaded texel equals the clear colour, so \
             ignoring the texture could pass"
        )
        .into());
    }
    Ok(RenderGeometry::SampledTexture)
}

/// Classify a stage-buffer case and pin the declarations the reviewed arms
/// admit (`research/docs/23` §3.3, v83-v86).
///
/// Two arms exist and no third. A case without `translated_stages` names the
/// reviewed pair and declares exactly its two slots — the vertex stage's
/// twenty-four byte positions and the fragment stage's sixteen byte tint, both
/// read once under a static ceiling — while a case that carries
/// `translated_stages` hands its stages to the Vulkan rail's translator and is
/// executable there alone, so its marker has to say so and it pins no MSL
/// sibling (there is no reviewed module to name).
///
/// The geometry itself is the `vertex_id` triangle in both arms: the stages
/// read their bytes from their own buffers, so no vertex layout, stream or
/// index buffer takes part, and the draw's vertex count is what the
/// declaration's footprint proof is evaluated against.
fn reviewed_stage_buffer_geometry(case: &RenderCase, where_: &str) -> Result<RenderGeometry> {
    if case.vertex_layout.is_some() || !case.vertex_buffers.is_empty() || case.indices.is_some() {
        return Err(format!(
            "{where_}: a stage-buffer case binds no vertex stream, layout or index buffer"
        )
        .into());
    }
    if case.vertices != STAGE_BUFFER_VERTICES {
        return Err(format!(
            "{where_}: the reviewed stage-buffer geometry draws {STAGE_BUFFER_VERTICES} vertices"
        )
        .into());
    }
    // The rest of the reviewed state families describe other geometries: a
    // depth, stencil, cull, blend, multisample or sampled-texture shape beside
    // a stage-buffer declaration would be a second reviewed shape inside one
    // case, which is exactly what the whitelist refuses.
    if case.fragment_textures.is_some()
        || case.depth.is_some()
        || case.stencil.is_some()
        || case.cull.is_some()
        || case.blend.is_some()
        || case.multisample.is_some()
        || case.depth_resolve.is_some()
        || case.stencil_resolve.is_some()
        || case.present.is_some()
        || case.icb.is_some()
    {
        return Err(format!(
            "{where_}: a stage-buffer case carries no texture, depth, stencil, cull, blend, \
             multisample, resolve, present or indirect section"
        )
        .into());
    }
    if case.stage_buffers.is_empty() {
        return Err(format!(
            "{where_}: a translated stage-buffer case declares the slots its stages read"
        )
        .into());
    }
    if case.stage_buffers.len() > MAX_RENDER_STAGE_BUFFERS {
        return Err(format!(
            "{where_}: the reviewed stage-buffer ceiling is {MAX_RENDER_STAGE_BUFFERS} slots"
        )
        .into());
    }
    // One rail executes the shape today, and the marker has to say so. The
    // native rails compile no AIR and their render stage-buffer capability is
    // off at this commit (`crates/metal-api-native/src/render.rs` publishes
    // `supports_render_stage_buffers = false`), while the object API has no
    // stage-buffer entry point at all (`crates/metal-api-core/src/provider_api.rs`):
    // a case that named either would claim a capture that rail cannot report,
    // so the marker is the Vulkan trace rail alone.
    if case.capture_rails.as_slice() != ["vulkan"] {
        return Err(format!(
            "{where_}: a stage-buffer case runs on the Vulkan trace rail alone, so its \
             capture_rails has to be [\"vulkan\"]"
        )
        .into());
    }
    if let Some(translated) = &case.translated_stages {
        if case.metal.is_some() {
            return Err(format!(
                "{where_}: a translated stage-buffer case has no MSL sibling to pin, so it \
                 carries no metal source"
            )
            .into());
        }
        if translated.vertex.path == translated.fragment.path {
            return Err(
                format!("{where_}: the two translated stages name their own AIR modules").into(),
            );
        }
    }
    let mut slots = Vec::with_capacity(case.stage_buffers.len());
    for definition in &case.stage_buffers {
        let stage = stage_buffer_stage(&definition.stage, where_)?;
        let access = stage_buffer_access(&definition.access, where_)?;
        if definition.index >= MAX_RENDER_STAGE_BUFFER_INDEX {
            return Err(format!(
                "{where_}: stage buffer index {} is at or above the reviewed ceiling \
                 {MAX_RENDER_STAGE_BUFFER_INDEX}",
                definition.index
            )
            .into());
        }
        if definition.allocation == 0 || definition.view == 0 {
            return Err(format!("{where_}: zero stage-buffer view identity").into());
        }
        // The contract's list is canonical — vertex bindings before fragment
        // bindings, ascending inside each stage — and a slot cannot be named
        // twice (`validate_stage_buffer_bindings`). The fixture spells the same
        // order so the declaration it feeds the rail is the one it states.
        let slot = (stage.code(), definition.index);
        if slots.last().is_some_and(|previous| *previous >= slot) {
            return Err(format!(
                "{where_}: stage buffers are declared once each, vertex bindings before \
                 fragment bindings and ascending inside each stage (at {}/{})",
                definition.stage, definition.index
            )
            .into());
        }
        slots.push(slot);
        let bytes = unhex(&definition.initial_hex)?;
        if bytes.len() != usize::try_from(definition.length)? {
            return Err(format!(
                "{where_}: stage buffer {}/{} declares {} bytes of view but {} bytes of data",
                definition.stage,
                definition.index,
                definition.length,
                bytes.len()
            )
            .into());
        }
        let mode = stage_buffer_storage_mode(definition, where_)?;
        let landed = access.is_writable();
        match (&definition.expected_hex, landed) {
            (Some(hex_), true) => {
                if unhex(hex_)?.len() != usize::try_from(definition.length)? {
                    return Err(format!(
                        "{where_}: stage buffer {}/{} has to land exactly the bytes its view \
                         covers",
                        definition.stage, definition.index
                    )
                    .into());
                }
            }
            (Some(_), false) => {
                return Err(format!(
                    "{where_}: a read-only stage buffer carries no expectation, and {}/{} \
                     states one",
                    definition.stage, definition.index
                )
                .into());
            }
            (None, true) => {
                return Err(format!(
                    "{where_}: writable stage buffer {}/{} has to state the bytes its writeback \
                     lands",
                    definition.stage, definition.index
                )
                .into());
            }
            (None, false) => {}
        }
        if mode != BufferSourceKind::OwnedBytes && definition.allocation_size.is_none() {
            return Err(format!(
                "{where_}: the lease-armed stage buffer {}/{} states the allocation size its \
                 owner window covers",
                definition.stage, definition.index
            )
            .into());
        }
        if mode == BufferSourceKind::OwnedBytes && definition.allocation_size.is_some() {
            return Err(format!(
                "{where_}: an owned stage buffer maps no owner window, so {}/{} states no \
                 allocation size",
                definition.stage, definition.index
            )
            .into());
        }
        match &definition.footprint {
            FootprintDefinition::Static { max_bytes } => {
                if *max_bytes == 0 {
                    return Err(format!(
                        "{where_}: a static stage-buffer ceiling is at least one byte"
                    )
                    .into());
                }
            }
            FootprintDefinition::Affine { accesses } => {
                if accesses.is_empty() {
                    return Err(format!(
                        "{where_}: an affine stage-buffer declaration names at least one access"
                    )
                    .into());
                }
                for access_ in accesses {
                    if access_.access_size == 0 {
                        return Err(format!(
                            "{where_}: an affine access is at least one byte wide"
                        )
                        .into());
                    }
                    for term in &access_.terms {
                        if term.axis >= RENDER_AFFINE_AXES {
                            return Err(format!(
                                "{where_}: affine axis {} is above the two the draw states \
                                 (vertex, instance)",
                                term.axis
                            )
                            .into());
                        }
                        if term.stride == 0 {
                            return Err(format!(
                                "{where_}: an affine term needs a non-zero stride"
                            )
                            .into());
                        }
                    }
                }
            }
        }
        if case.translated_stages.is_none() {
            if access != BufferAccess::Read {
                return Err(format!(
                    "{where_}: this rail's reviewed stage-buffer pair reads its slots, so a \
                     writable declaration has no writer behind it; the writable face is a \
                     translated case's"
                )
                .into());
            }
            let positions_slot = matches!(
                (stage, definition.index, &definition.footprint),
                (RenderPipelineStage::Vertex, 0, FootprintDefinition::Static { max_bytes })
                    if *max_bytes == STAGE_BUFFER_POSITIONS_BYTES
            );
            let tint_slot = matches!(
                (stage, definition.index, &definition.footprint),
                (RenderPipelineStage::Fragment, 0, FootprintDefinition::Static { max_bytes })
                    if *max_bytes == STAGE_BUFFER_TINT_BYTES
            );
            if !positions_slot && !tint_slot {
                return Err(format!(
                    "{where_}: the reviewed stage-buffer pair declares vertex slot 0 \
                     ({STAGE_BUFFER_POSITIONS_BYTES} bytes) and fragment slot 0 \
                     ({STAGE_BUFFER_TINT_BYTES} bytes), both read once with a static ceiling"
                )
                .into());
            }
        }
    }
    if case.translated_stages.is_none() {
        let reviewed = [
            (RenderPipelineStage::Vertex.code(), 0),
            (RenderPipelineStage::Fragment.code(), 0),
        ];
        if slots.len() != reviewed.len() || !reviewed.iter().all(|slot| slots.contains(slot)) {
            return Err(format!(
                "{where_}: the reviewed stage-buffer pair reads both of its slots, so the case \
                 declares vertex slot 0 and fragment slot 0"
            )
            .into());
        }
    }
    Ok(RenderGeometry::StageBuffers)
}

/// One stage-buffer declaration's stage, in the contract's own vocabulary.
fn stage_buffer_stage(spelling: &str, where_: &str) -> Result<RenderPipelineStage> {
    match spelling {
        "vertex" => Ok(RenderPipelineStage::Vertex),
        "fragment" => Ok(RenderPipelineStage::Fragment),
        other => Err(format!("{where_}: unknown stage-buffer stage {other:?}").into()),
    }
}

/// One stage-buffer declaration's access, in the contract's own vocabulary.
fn stage_buffer_access(spelling: &str, where_: &str) -> Result<BufferAccess> {
    match spelling {
        "read" => Ok(BufferAccess::Read),
        "write" => Ok(BufferAccess::Write),
        "read_write" => Ok(BufferAccess::ReadWrite),
        other => Err(format!("{where_}: unknown stage-buffer access {other:?}").into()),
    }
}

/// The source arm one stage-buffer declaration's bytes come from
/// (`research/docs/23` §90, R9i): the same three names a compute case's
/// `storage_mode` spells.
fn stage_buffer_storage_mode(
    definition: &StageBufferDefinition,
    where_: &str,
) -> Result<BufferSourceKind> {
    match definition.storage_mode.as_deref() {
        None | Some("owned_bytes") => Ok(BufferSourceKind::OwnedBytes),
        Some("staged_lease") => Ok(BufferSourceKind::StagedLease),
        Some("borrowed_no_copy") => Ok(BufferSourceKind::BorrowedNoCopy),
        Some(other) => Err(format!(
            "{where_}: stage buffer {}/{} names unknown storage mode {other:?}",
            definition.stage, definition.index
        )
        .into()),
    }
}

/// The source arm every lease-armed stage buffer of one render case ran with
/// (`research/docs/23` §90, R9i), or `None` for a case that declares no lease
/// arm: a capture of an owned-only fixture keeps the result shape it always
/// had, while a lease case states which arm each view ran with — the
/// observation the comparator checks instead of trusting the trace's own
/// naming.
fn stage_buffer_storage_modes(case: &RenderCase) -> Result<Option<Vec<StorageModeObservation>>> {
    let where_ = format!("render case {}", case.id);
    let declared = case
        .stage_buffers
        .iter()
        .map(|definition| {
            Ok((
                definition.view,
                stage_buffer_storage_mode(definition, &where_)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(declared
        .iter()
        .any(|(_, mode)| *mode != BufferSourceKind::OwnedBytes)
        .then(|| {
            declared
                .iter()
                .map(|(view, mode)| StorageModeObservation {
                    view: *view,
                    mode: storage_mode_name(*mode),
                })
                .collect()
        }))
}

/// The pipeline-level declarations one render case's stage-buffer slots carry
/// (`research/docs/23` §3.3, v83-v86): the exact list the contract hands the
/// rail, built from the same entries the pass's views come from. Both halves of
/// one entry travel together, so the declaration cannot drift from the view the
/// rail binds.
fn stage_buffer_declarations(case: &RenderCase) -> Result<Vec<StageBufferBinding>> {
    let where_ = format!("render case {}", case.id);
    case.stage_buffers
        .iter()
        .map(|definition| {
            Ok(StageBufferBinding {
                stage: stage_buffer_stage(&definition.stage, &where_)?,
                index: definition.index,
                access: stage_buffer_access(&definition.access, &where_)?,
                footprint: match &definition.footprint {
                    FootprintDefinition::Static { max_bytes } => FootprintProof::Static {
                        max_bytes: *max_bytes,
                    },
                    FootprintDefinition::Affine { accesses } => FootprintProof::Affine {
                        accesses: accesses
                            .iter()
                            .map(|access| -> Result<AffineAccess> {
                                Ok(AffineAccess {
                                    base_offset: access.base_offset,
                                    access_size: access.access_size,
                                    terms: access
                                        .terms
                                        .iter()
                                        .map(|term| {
                                            Ok(AffineTerm {
                                                axis: u8::try_from(term.axis).map_err(
                                                    |_| -> Box<dyn Error> {
                                                        format!(
                                                            "{where_}: affine axis {} has no \
                                                             wire spelling",
                                                            term.axis
                                                        )
                                                        .into()
                                                    },
                                                )?,
                                                stride: term.stride,
                                            })
                                        })
                                        .collect::<Result<Vec<_>>>()?,
                                })
                            })
                            .collect::<Result<Vec<_>>>()?,
                    },
                },
            })
        })
        .collect()
}

/// The pass-level views one render case's stage-buffer slots bind
/// (`research/docs/23` §3.3, v83): the same entries the declarations are built
/// from, resolved to the bytes their own source arm names.
///
/// An owned view reads the case's own bytes every time a trace is built, while
/// a lease-armed view names the import the provider already holds — exactly the
/// rule a compute case's `case_sources` follows (`research/docs/23` §90, R9i).
fn stage_buffer_views(
    case: &RenderCase,
    leases: &LeaseHandles,
    provider: &dyn PipelineProvider,
    guard: u8,
    imported: &mut CaseLeases,
    windows: &mut Vec<OwnerWindow>,
) -> Result<Vec<StageBufferView>> {
    let where_ = format!("render case {}", case.id);
    let mut views = Vec::with_capacity(case.stage_buffers.len());
    for definition in &case.stage_buffers {
        let stage = stage_buffer_stage(&definition.stage, &where_)?;
        let access = stage_buffer_access(&definition.access, &where_)?;
        let bytes = unhex(&definition.initial_hex)?;
        let source = match stage_buffer_storage_mode(definition, &where_)? {
            BufferSourceKind::OwnedBytes => BufferSource::OwnedBytes(bytes),
            BufferSourceKind::StagedLease => {
                let reservation = LeaseReservation {
                    lease: BufferLease {
                        lease_id: LeaseId::new(definition.view),
                        allocation_id: AllocationId::new(definition.allocation),
                        owner_epoch: provider.device_epoch(),
                    },
                    offset: definition.offset,
                    length: definition.length,
                };
                let staged = StagedLease::new(reservation, bytes)?;
                leases
                    .staged
                    .import_staged_lease(staged)
                    .map_err(|error| format!("{where_}: import staged lease: {error:?}"))?;
                imported.reservations.push(reservation);
                imported.staged.push(reservation.lease.lease_id);
                BufferSource::StagedLease(reservation.lease.lease_id)
            }
            BufferSourceKind::BorrowedNoCopy => {
                // The window is the owner's own mapping, so the same alignment
                // and coverage rules a compute case's borrowed view follows
                // apply here (`research/docs/23` §90): the view's offset and its
                // rounded window have to sit on the host-import grid, and the
                // allocation-sized registration has to cover them both.
                let page = leases.borrowed.no_copy_alignment();
                if page == 0 || !page.is_power_of_two() {
                    return Err(format!(
                        "{where_}: view {} is borrowed_no_copy, and the provider reports host \
                         import granularity {page}, so it publishes `storage_mode_unsupported` \
                         for this arm",
                        definition.view
                    )
                    .into());
                }
                let allocation_size = definition
                    .allocation_size
                    .ok_or("a borrowed stage buffer states its allocation size")?;
                if !definition.offset.is_multiple_of(page) {
                    return Err(format!(
                        "{where_}: view {} starts at offset {} inside its allocation, which is \
                         not a multiple of the {page} byte host-import granularity",
                        definition.view, definition.offset
                    )
                    .into());
                }
                let window_length = round_up_to_page(definition.length, page)?;
                let region_length = round_up_to_page(allocation_size, page)?;
                if definition.offset + window_length > allocation_size {
                    return Err(format!(
                        "{where_}: view {} needs a {window_length} byte window at offset {}, \
                         which its {allocation_size} byte allocation does not cover",
                        definition.view, definition.offset
                    )
                    .into());
                }
                let mut window =
                    OwnerWindow::new(usize::try_from(region_length)?, usize::try_from(page)?)?;
                window.as_mut_slice().fill(guard);
                let start = usize::try_from(definition.offset)?;
                window.as_mut_slice()[start..start + bytes.len()].copy_from_slice(&bytes);
                let region = HostRegion {
                    lease_id: LeaseId::new(definition.view),
                    owner_epoch: provider.device_epoch(),
                    host_pointer: window.as_ptr() as usize,
                    length: region_length,
                    page_size: page,
                };
                let borrowed = region.borrowed_window(
                    AllocationId::new(definition.allocation),
                    definition.offset,
                    window_length,
                )?;
                // Safety: the window is page-aligned, lives until the case's
                // last submission has been waited for (it is owned by
                // `windows`), and covers the whole reservation.
                unsafe {
                    leases
                        .borrowed
                        .import_borrowed_lease(borrowed)
                        .map_err(|error| format!("{where_}: import borrowed lease: {error:?}"))?
                };
                windows.push(window);
                imported.reservations.push(borrowed.reservation);
                imported.borrowed.push(borrowed.reservation.lease.lease_id);
                BufferSource::BorrowedNoCopy(borrowed.reservation.lease.lease_id)
            }
        };
        views.push(StageBufferView {
            stage,
            view: BufferView {
                view_id: ViewId::new(definition.view),
                metal_binding: definition.index,
                allocation_id: AllocationId::new(definition.allocation),
                offset: definition.offset,
                length: definition.length,
                access,
                attribute_stride: None,
                source,
            },
        });
    }
    Ok(views)
}

/// Classify a render case's geometry and pin the reviewed shape.
///
/// The checks are deliberately exact: the vertex-input case names one stream,
/// one attribute and one index buffer, and its byte ranges have to cover the
/// draw the fixture claims, with every index naming one of the four reviewed
/// vertices. Anything else is a case the reviewers have not seen.
fn render_geometry(case: &RenderCase, where_: &str) -> Result<RenderGeometry> {
    // The stage-buffer shape is classified first (`research/docs/23` §3.3,
    // v83-v86): its stages read their bytes from `[[buffer(N)]]` arguments
    // rather than from a vertex layout or the vertex index, so a case that
    // declares one is that shape and nothing else may be added to it.
    if !case.stage_buffers.is_empty() || case.translated_stages.is_some() {
        return reviewed_stage_buffer_geometry(case, where_);
    }
    // The render sampler is classified first: its shape is the milestone's
    // vertex_id geometry plus one texture binding, so a case that carries the
    // binding is the sampled shape and nothing else may be added to it
    // (`research/docs/23` §3.3, v70).
    if let Some(textures) = &case.fragment_textures {
        return reviewed_sampled_geometry(case, textures, where_);
    }
    let Some(layout) = &case.vertex_layout else {
        if !case.vertex_buffers.is_empty() || case.indices.is_some() {
            return Err(format!(
                "{where_}: vertex buffers without a vertex layout describe no stream"
            )
            .into());
        }
        return Ok(RenderGeometry::Milestone);
    };
    // The instanced shape is the one two-stream layout this suite admits: the
    // reviewed position stream at binding 0, the per-instance tint at binding
    // 1, and exactly the two instances the reviewed module's `instance_id`
    // shift was written for (`research/docs/23` §3.3, v31).
    // The two reviewed pair shapes are mutually exclusive, and the culling one
    // is checked first so a case that declares both is classified as the cull
    // shape and refused by its own rule (`research/docs/23` §3.3, v39); the
    // depth shape is the one that opens a depth attachment.
    if case.blend.is_some() {
        return reviewed_blend_geometry(case, layout, where_);
    }
    if case.cull.is_some() {
        return reviewed_cull_geometry(case, layout, where_);
    }
    // The depth and stencil shapes are the same pair geometry with one
    // depth-stencil surface behind it (`research/docs/23` §3.3, v36/v47): the
    // geometry classifier routes both to the same review, and that review
    // requires exactly one of the two.
    if case.depth.is_some() || case.stencil.is_some() {
        return reviewed_depth_geometry(case, layout, where_);
    }
    if layout.buffers.len() == 2 {
        return reviewed_instanced_geometry(case, layout, where_);
    }
    if case.base_vertex != 0 {
        return reviewed_base_vertex_geometry(case, layout, where_);
    }
    if layout.buffers.len() != 1 || case.vertex_buffers.len() != 1 {
        return Err(format!(
            "{where_}: the reviewed vertex-input shape is one stream and one binding"
        )
        .into());
    }
    let stream = &layout.buffers[0];
    if stream.stride != QUAD_STRIDE || stream.attributes.len() != 1 {
        return Err(format!(
            "{where_}: the reviewed stream is one float32x2 position at stride {QUAD_STRIDE}"
        )
        .into());
    }
    let attribute = &stream.attributes[0];
    if attribute.location != 0 || attribute.offset != 0 || attribute.format != "float32x2" {
        return Err(
            format!("{where_}: the reviewed attribute is location 0, offset 0, float32x2").into(),
        );
    }
    let buffer = &case.vertex_buffers[0];
    if buffer.allocation == 0 || buffer.view == 0 {
        return Err(format!("{where_}: zero vertex stream identity").into());
    }
    let required = QUAD_STRIDE
        .checked_mul(QUAD_VERTICES)
        .ok_or("vertex stream footprint overflows")?;
    if buffer.length < required {
        return Err(format!(
            "{where_}: the vertex stream declares {} bytes, fewer than the {required} the reviewed quad reads",
            buffer.length
        )
        .into());
    }
    let bytes = unhex(&buffer.initial_hex)?;
    if bytes.len() != usize::try_from(buffer.length)? {
        return Err(format!("{where_}: the vertex stream bytes do not match its length").into());
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed vertex-input shape is indexed").into());
    };
    if indices.allocation == 0 || indices.view == 0 {
        return Err(format!("{where_}: zero index buffer identity").into());
    }
    let width = match indices.format.as_str() {
        "uint16" => 2_u64,
        "uint32" => 4,
        other => return Err(format!("{where_}: unsupported index format {other:?}").into()),
    };
    let required = QUAD_INDICES
        .checked_mul(width)
        .ok_or("index footprint overflows")?;
    if indices.length < required {
        return Err(format!(
            "{where_}: the index buffer declares {} bytes, fewer than the {required} the reviewed quad reads",
            indices.length
        )
        .into());
    }
    let bytes = unhex(&indices.initial_hex)?;
    if bytes.len() != usize::try_from(indices.length)? {
        return Err(format!("{where_}: the index bytes do not match their length").into());
    }
    let width = usize::try_from(width)?;
    for (position, chunk) in bytes.chunks_exact(width).enumerate() {
        let index = match width {
            2 => u64::from(u16::from_le_bytes([chunk[0], chunk[1]])),
            _ => u64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        };
        if index >= QUAD_VERTICES {
            return Err(format!(
                "{where_}: index {position} names vertex {index}, outside the reviewed quad"
            )
            .into());
        }
    }
    Ok(RenderGeometry::IndexedQuad)
}

/// Turn a render case's declared streams into the pass's own bindings.
///
/// Each stream carries its bytes, so the trace needs no compute binding for
/// them (`research/docs/23` §3.6). The reviewed-shape validation runs before
/// this point, so the translation cannot meet a shape the rails have not been
/// reviewed against.
fn render_inputs(
    case: &RenderCase,
    where_: &str,
) -> Result<(Vec<BufferView>, Option<IndexBufferBinding>)> {
    // The render sampler binds no stream either: its geometry is the
    // milestone's `vertex_id` triangle (`research/docs/23` §3.3, v70), and the
    // stage-buffer shape is the same triangle whose vertices come from the
    // vertex stage's own `[[buffer(0)]]` argument rather than a stream
    // (`research/docs/23` §3.3, v83).
    if matches!(
        render_geometry(case, where_)?,
        RenderGeometry::Milestone | RenderGeometry::SampledTexture | RenderGeometry::StageBuffers
    ) {
        return Ok((Vec::new(), None));
    }
    // One pass view per bound stream, in binding order: the pass is positional
    // exactly like the pipeline layout, so the entry's position is the
    // `metal_binding` its view carries
    // (`research/docs/23` §3.6, v31 for the two-stream shape).
    let mut vertex_buffers = Vec::with_capacity(case.vertex_buffers.len());
    for (binding, definition) in case.vertex_buffers.iter().enumerate() {
        vertex_buffers.push(BufferView {
            view_id: ViewId::new(definition.view),
            metal_binding: u32::try_from(binding)?,
            allocation_id: AllocationId::new(definition.allocation),
            offset: definition.offset,
            length: definition.length,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(unhex(&definition.initial_hex)?),
        });
    }
    let Some(indices) = &case.indices else {
        return Err(format!("{where_}: the reviewed vertex-input shape is indexed").into());
    };
    let format = match indices.format.as_str() {
        "uint16" => IndexFormat::Uint16,
        "uint32" => IndexFormat::Uint32,
        other => return Err(format!("{where_}: unsupported index format {other:?}").into()),
    };
    Ok((
        vertex_buffers,
        Some(IndexBufferBinding {
            view: BufferView {
                view_id: ViewId::new(indices.view),
                metal_binding: 0,
                allocation_id: AllocationId::new(indices.allocation),
                offset: indices.offset,
                length: indices.length,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(unhex(&indices.initial_hex)?),
            },
            format,
        }),
    ))
}

fn validate_render_case(suite: &Suite, case: &RenderCase) -> Result<()> {
    let where_ = format!("render case {}", case.id);
    // One of two mutually exclusive attachment shapes: the single `attachment`
    // object plus the case-level `expected_hex`, or the MRT `attachments` list
    // whose entries each carry their own expectation.
    let single = case.attachment.is_some() || case.expected_hex.is_some();
    let multiple = case.attachments.is_some();
    // A case may declare no colour attachment at all when its stored depth
    // attachment is the whole landing (`research/docs/23` §3.3, v46): the
    // depth-only shape proper, where the rasterizer writes depth into a surface
    // no colour target exists beside.
    let no_colour = case.attachment.is_none()
        && case.attachments.is_none()
        && case.expected_hex.is_none()
        && case
            .depth
            .as_ref()
            .is_some_and(|depth| depth.store.as_deref() == Some("store"));
    if single == multiple && !no_colour {
        return Err(
            format!("{where_}: exactly one of attachment and attachments is required").into(),
        );
    }
    // A rule expectation is the reviewed sampling shape's (`research/docs/23`
    // §73): the bound texture's own texels *are* the expectation, so a case
    // that states a rule without binding the texture carrying it is refused
    // rather than run against a plane no reviewed draw can produce.
    if case.expected_rule.is_some() && case.fragment_textures.is_none() {
        return Err(
            format!("{where_}: a rule expectation is the reviewed sampling shape's").into(),
        );
    }
    let shapes = render_attachment_shapes(case)?;
    if multiple && !(2..=metal_api_core::provider::MAX_COLOR_ATTACHMENTS).contains(&shapes.len()) {
        return Err(format!(
            "{where_}: the reviewed MRT shapes are two to \
             {} attachments",
            metal_api_core::provider::MAX_COLOR_ATTACHMENTS
        )
        .into());
    }
    let geometry = render_geometry(case, &where_)?;
    match geometry {
        RenderGeometry::Milestone => {
            if case.vertices != 3 {
                return Err(format!("{where_}: expected the reviewed full-screen triangle").into());
            }
            if multiple {
                return Err(format!(
                    "{where_}: the milestone vertex_id shape renders one attachment"
                )
                .into());
            }
        }
        RenderGeometry::IndexedQuad => {
            if case.vertices != QUAD_INDICES {
                return Err(format!(
                    "{where_}: the reviewed indexed quad draws {QUAD_INDICES} indices"
                )
                .into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a vertex-input case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::BlendTriangle => {
            if case.vertices != 3 {
                return Err(
                    format!("{where_}: the reviewed blend triangle draws three indices").into(),
                );
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a blend case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::CullPair => {
            if case.vertices != 6 {
                return Err(format!("{where_}: the reviewed cull pair draws six indices").into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a cull case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::DepthPair => {
            // The reviewed depth, stencil and resolve pairs draw the two
            // triangles' six indices; the v66 rail-owned combined pair draws
            // its three triangles' nine (`research/docs/23` §3.3, v66).
            let combined_pair =
                case.depth.is_some() && case.stencil.is_some() && case.stencil_resolve.is_none();
            let reviewed_indices = if combined_pair { 9 } else { 6 };
            if case.vertices != reviewed_indices {
                return Err(format!(
                    "{where_}: the reviewed depth pair draws {reviewed_indices} indices"
                )
                .into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a depth case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::BaseVertexQuad => {
            if case.vertices != QUAD_INDICES {
                return Err(format!(
                    "{where_}: the reviewed base-vertex quad draws {QUAD_INDICES} indices"
                )
                .into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a base-vertex case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::InstancedPair => {
            if case.vertices != QUAD_INDICES {
                return Err(format!(
                    "{where_}: the reviewed instanced pair draws {QUAD_INDICES} indices per instance"
                )
                .into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: an instanced case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::SampledTexture => {
            if case.vertices != 3 {
                return Err(format!(
                    "{where_}: the reviewed render sampler draws the full-screen triangle"
                )
                .into());
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a sampled case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
        RenderGeometry::StageBuffers => {
            // The stage-buffer geometry and its declarations were pinned by
            // `reviewed_stage_buffer_geometry` above; what is left here is the
            // landing half. A writable slot is a landing, so the view has to be
            // one the declaring case also declares — the trace's own view pool
            // is where a writeback lands (`research/docs/23` §3.3, v86).
            let declaring = suite
                .cases
                .iter()
                .find(|candidate| candidate.id == case.declaring_case);
            for definition in &case.stage_buffers {
                if !stage_buffer_access(&definition.access, &where_)?.is_writable() {
                    // A read-only slot carries its own bytes, exactly as a
                    // vertex stream does, so it cannot name a view the
                    // declaring case already declares: one identity would then
                    // stand for two byte strings. A writable slot is the other
                    // arm — it has to name the declaring view, because that is
                    // the pool entry its writeback lands in.
                    if let Some(declaring) = declaring {
                        if declaring
                            .buffers
                            .iter()
                            .any(|buffer| buffer.view == definition.view)
                        {
                            return Err(format!(
                                "{where_}: the read-only stage buffer {} names view {}, which \
                                 the declaring case already declares",
                                definition.index, definition.view
                            )
                            .into());
                        }
                    }
                    continue;
                }
                let Some(declaring) = declaring else {
                    return Err(format!(
                        "{where_}: unknown declaring case {:?}",
                        case.declaring_case
                    )
                    .into());
                };
                let pool = declaring.buffers.iter().find(|buffer| {
                    buffer.allocation == definition.allocation && buffer.view == definition.view
                });
                let Some(pool) = pool else {
                    return Err(format!(
                        "{where_}: the writable stage buffer {}/{} (allocation {}, view {}) \
                         lands in the trace's own view pool, so the declaring case has to \
                         declare that view",
                        definition.stage, definition.index, definition.allocation, definition.view
                    )
                    .into());
                };
                if pool.access != "read" {
                    return Err(format!(
                        "{where_}: the declaring pass reads the writable stage buffer's view, \
                         and {} is declared {:?}",
                        definition.view, pool.access
                    )
                    .into());
                }
                if pool.offset != definition.offset || pool.length != definition.length {
                    return Err(format!(
                        "{where_}: the writable stage buffer's view {} and the declaring case's \
                         declaration of it cover different ranges",
                        definition.view
                    )
                    .into());
                }
                if pool.initial_bytes()? != unhex(&definition.initial_hex)? {
                    return Err(format!(
                        "{where_}: the writable stage buffer's view {} carries different bytes \
                         than the declaring case's declaration of it",
                        definition.view
                    )
                    .into());
                }
            }
            if case.present.is_some() || case.icb.is_some() {
                return Err(format!(
                    "{where_}: a stage-buffer case carries neither a present action nor an ICB"
                )
                .into());
            }
        }
    }
    if let Some(present) = &case.present {
        if present.mode != "fifo" {
            return Err(format!(
                "{where_}: unsupported present mode {:?}; the first increment admits fifo",
                present.mode
            )
            .into());
        }
        if present.image_count != 1 {
            return Err(format!("{where_}: the first present increment admits one image").into());
        }
        if present.acquire != 1 || present.present != 1 {
            return Err(format!(
                "{where_}: the first present increment counts exactly one acquire and one present"
            )
            .into());
        }
        if let Some(hex) = &present.initial_hex {
            if unhex(hex)?.len() != 4 {
                return Err(format!("{where_}: a present sentinel is four bytes").into());
            }
        }
    }
    let reviewed_entries = match geometry {
        RenderGeometry::Milestone => (RENDER_MSL_VERTEX_ENTRY, RENDER_MSL_FRAGMENT_ENTRY),
        // The instanced pair is its own reviewed module
        // (`research/docs/23` §3.3, v31): the solid fragment stages cannot
        // stand in for it, because the tint travels through a varying the
        // reviewed instanced vertex stage is the only one to produce.
        RenderGeometry::InstancedPair => (INSTANCED_MSL_VERTEX_ENTRY, INSTANCED_MSL_FRAGMENT_ENTRY),
        // The base-vertex shape compiles the reviewed quad's module pair: the
        // offset is draw state, so the entries and the layout do not change
        // (`research/docs/23` §3.3, v34).
        RenderGeometry::BaseVertexQuad => (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_FRAGMENT_ENTRY),
        // The zero-colour-attachment depth pass is the depth pair's shape with
        // no colour target beside it (`research/docs/23` §3.3, v46), so it
        // compiles the module whose fragment stage generates no output.
        RenderGeometry::DepthPair if no_colour => {
            (DEPTH_ONLY_MSL_VERTEX_ENTRY, DEPTH_ONLY_MSL_FRAGMENT_ENTRY)
        }
        RenderGeometry::DepthPair => (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
        RenderGeometry::CullPair => (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
        RenderGeometry::BlendTriangle => (DEPTH_MSL_VERTEX_ENTRY, DEPTH_MSL_FRAGMENT_ENTRY),
        // The render sampler's MSL module carries its own entry pair
        // (`research/docs/23` §3.3, v70).
        RenderGeometry::SampledTexture => (SAMPLED_MSL_VERTEX_ENTRY, SAMPLED_MSL_FRAGMENT_ENTRY),
        // A translated stage-buffer case names its own two AIR entries rather
        // than a reviewed MSL pair — the stage module above pinned that the
        // case carries `translated_stages` and no MSL pin — while the reviewed
        // arm names the pair's MSL siblings (`research/docs/23` §3.3, v83/v86).
        RenderGeometry::StageBuffers if case.translated_stages.is_some() => {
            (case.vertex_entry.as_str(), case.fragment_entry.as_str())
        }
        RenderGeometry::StageBuffers => (
            STAGE_BUFFER_MSL_VERTEX_ENTRY,
            STAGE_BUFFER_MSL_FRAGMENT_ENTRY,
        ),
        RenderGeometry::IndexedQuad => match shapes.len() {
            // A single `r32float` attachment takes the reviewed one-component
            // MSL stage; every other single-output shape takes the
            // four-component one (`research/docs/23` §3.3, v22).
            1 if shapes[0].0.format == "r32float" => {
                (QUAD_MSL_VERTEX_ENTRY, R32F_MSL_FRAGMENT_ENTRY)
            }
            1 => (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_FRAGMENT_ENTRY),
            2 => (QUAD_MSL_VERTEX_ENTRY, DUAL_MSL_FRAGMENT_ENTRY),
            3 => (QUAD_MSL_VERTEX_ENTRY, TRIPLE_MSL_FRAGMENT_ENTRY),
            4 => (QUAD_MSL_VERTEX_ENTRY, QUAD_MSL_QUAD_FRAGMENT_ENTRY),
            _ => {
                return Err(format!(
                    "{where_}: the reviewed MRT shapes are one to four attachments"
                )
                .into());
            }
        },
    };
    if case.vertex_entry.trim().is_empty()
        || case.fragment_entry.trim().is_empty()
        || case.vertex_entry == case.fragment_entry
    {
        return Err(
            format!("{where_}: a render case names two distinct, non-empty stage entries").into(),
        );
    }
    if (case.vertex_entry.as_str(), case.fragment_entry.as_str()) != reviewed_entries {
        return Err(format!(
            "{where_}: unreviewed render pipeline identity {:?}/{:?}",
            case.vertex_entry, case.fragment_entry
        )
        .into());
    }
    // One attachment at a time: the single shape is the v13-v17 branch, and
    // every MRT entry restates the same per-field rules with its own
    // expectation. The v19 increment adds the store operation: a `dontcare`
    // attachment still renders and still resolves against its declaring view,
    // but it carries no expectation and no observation, and at least one
    // attachment has to stay stored (`docs/23` §3.6). The v20 increment adds
    // the undefined load: a `dontcare` load carries no clear colour and no
    // initial bytes, and its expectation still has to differ from the bytes
    // the declaring case pins for the same view (`docs/23` §13).
    // The coverage claim (`research/docs/23` §3.3, v38): only the single
    // attachment shape may make it, and only the one spelling exists.
    if let Some(coverage) = &case.coverage {
        if coverage != "partial" {
            return Err(format!("{where_}: the only coverage claim is \"partial\"").into());
        }
        if !single {
            return Err(
                format!("{where_}: the coverage claim is the single-attachment shape").into(),
            );
        }
    }
    // The depth resolve (`research/docs/23` §3.3, v57) only means something
    // beside a multisample raster that keeps its depth surface: the resolve is
    // the reduction of the stored four-sample texels, so any other shape is
    // refused instead of silently ignored.
    if case.depth_resolve.is_some() {
        if case.multisample.is_none() {
            return Err(format!("{where_}: a depth resolve needs a multisample raster").into());
        }
        if case
            .depth
            .as_ref()
            .is_none_or(|depth| depth.store.as_deref() != Some("store"))
        {
            return Err(format!("{where_}: a depth resolve needs a stored depth surface").into());
        }
    }
    // The stencil resolve (`research/docs/23` §3.3, v60) is the depth
    // resolve's sibling one byte wide: it only means something beside a
    // multisample raster that keeps its stencil surface, and the
    // `depth_resolved_sample` filter names the sample the depth resolve
    // selected, so it is refused without one instead of silently degrading to
    // sample zero.
    if case.stencil_resolve.is_some() {
        if case.multisample.is_none() {
            return Err(format!("{where_}: a stencil resolve needs a multisample raster").into());
        }
        if case
            .stencil
            .as_ref()
            .is_none_or(|stencil| stencil.store.as_deref() != Some("store"))
        {
            return Err(
                format!("{where_}: a stencil resolve needs a stored stencil surface").into(),
            );
        }
    }
    if let Some(resolve) = &case.stencil_resolve {
        if resolve.filter == "depth_resolved_sample" && case.depth_resolve.is_none() {
            return Err(format!(
                "{where_}: the depth_resolved_sample stencil resolve names the sample the \
                 depth resolve selects, so the case has to state a depth resolve"
            )
            .into());
        }
    }
    // The device gate (`research/docs/23` §3.3, v57d): a case that requires a
    // depth resolve filter has to state the resolve whose filter it names —
    // the gate is the case's own admission condition, not a second spelling of
    // the filter that could drift away from the pass the rails execute. Only
    // the two filters a device may lack are gateable: Sample0 is the API's own
    // baseline, so nothing needs to be measured against it per device.
    if let Some(filter) = &case.requires_depth_resolve_filter {
        if !matches!(filter.as_str(), "min" | "max") {
            return Err(format!(
                "{where_}: the device gate names the min or max depth resolve filter"
            )
            .into());
        }
        if case
            .depth_resolve
            .as_ref()
            .is_none_or(|resolve| resolve.filter != *filter)
        {
            return Err(format!(
                "{where_}: the device gate has to name the resolve filter the case states"
            )
            .into());
        }
    }
    // The stencil-resolve device gate (`research/docs/23` §3.3, v60): a case
    // that requires a stencil resolve filter has to state the resolve whose
    // filter it names. Only the depth-resolved-sample filter is gateable:
    // sample0 is the API's own baseline, so nothing needs to be measured
    // against it per device.
    if let Some(filter) = &case.requires_stencil_resolve_filter {
        if filter != "depth_resolved_sample" {
            return Err(format!(
                "{where_}: the device gate names the depth_resolved_sample stencil resolve filter"
            )
            .into());
        }
        if case
            .stencil_resolve
            .as_ref()
            .is_none_or(|resolve| resolve.filter != *filter)
        {
            return Err(format!(
                "{where_}: the device gate has to name the resolve filter the case states"
            )
            .into());
        }
    }
    // The sample-count device gate (`research/docs/23` §3.3, v61): a case
    // that requires a sample count has to state the raster whose count it
    // names. Only the two counts a device may lack are gateable — 2x and 8x —
    // because 4x is the v51 baseline every multisampling device admits, so
    // nothing needs to be measured against it per device.
    if let Some(count) = case.requires_sample_count {
        if !matches!(count, 2 | 8) {
            return Err(format!(
                "{where_}: the device gate names the two- or eight-sample \
                                raster"
            )
            .into());
        }
        if case
            .multisample
            .as_ref()
            .is_none_or(|multisample| multisample.sample_count != count)
        {
            return Err(format!(
                "{where_}: the device gate has to name the sample count the case states"
            )
            .into());
        }
    }
    // The multisample raster (`research/docs/23` §3.3, v51): the first
    // increment reviews exactly one shape — one colour attachment opened from
    // a clear, four samples, no depth or stencil surface, no present action,
    // no ICB and no wildcard texels. The resolve expectation itself is the
    // comparator's rule (every texel is the fragment output, the clear colour,
    // or the arithmetic mean of the two); what this gate holds is that the case
    // states the one shape the rails execute, so no rail can silently run a
    // different raster than the expectation describes.
    if let Some(multisample) = &case.multisample {
        if !single {
            return Err(
                format!("{where_}: the multisample raster is the single-attachment shape").into(),
            );
        }
        if !matches!(multisample.sample_count, 2 | 4 | 8) {
            return Err(format!(
                "{where_}: the reviewed multisample rasters are two, four or eight \
                         samples"
            )
            .into());
        }
        let (attachment, _) = shapes.first().ok_or(format!(
            "{where_}: a multisample raster needs an attachment"
        ))?;
        // The attachment's load (`research/docs/23` §3.3, v51/v67/v82): the
        // reviewed pass opens it from a clear, from v67 on from `dontcare`
        // while every unclaimed texel states the closed set its resolve may
        // land in, or from v82 on from `load` whose declared window is the one
        // repeated texel the seam seeds every sample with. The free list stays
        // refused here.
        if !matches!(attachment.load.as_str(), "clear" | "dontcare" | "load") {
            return Err(format!(
                "{where_}: the reviewed multisample pass opens its attachment from a clear, a \
                 seeded load or a dontcare load with constrained wildcard texels"
            )
            .into());
        }
        // The two surfaces open together as the combined depth-stencil shape
        // (`research/docs/23` §3.3, v60/v66): one attachment both faces share,
        // in either of the two reviewed shapes — the v60 resolve shape, whose
        // two stored faces land through their resolves, or the v66 rail-owned
        // write-then-test pair. A lopsided pair is refused rather than
        // silently narrowed to one of the faces.
        let combined_pair = case.depth.is_some() && case.stencil.is_some();
        if combined_pair && case.stencil_resolve.is_none() {
            let depth_stored = case
                .depth
                .as_ref()
                .is_some_and(|depth| depth.store.is_some());
            let stencil_stored = case
                .stencil
                .as_ref()
                .is_some_and(|stencil| stencil.store.is_some());
            if depth_stored || stencil_stored {
                return Err(format!(
                    "{where_}: the combined depth-stencil pair keeps both faces or neither, \
                     and a kept face needs its resolve"
                )
                .into());
            }
        }
        // A present action beside the raster is admitted from v62 on: the
        // pass resolves into the attachment view and the present hands that
        // single-sample landing on, so the present section's own rules apply
        // unchanged. An indirect replay beside the raster is still the later
        // increment that reviews the two together (`research/docs/25` §5.2).
        if case.icb.is_some() {
            return Err(format!("{where_}: a multisample case carries no ICB").into());
        }
        if case.wildcard_texels.is_some() {
            return Err(
                format!("{where_}: the multisample raster claims every texel it resolves").into(),
            );
        }
        // No reviewed fixture covers a multisampled draw whose indices are
        // offset, and the trace rails' footprint proof is the only thing that
        // would notice a stream too short for it. Refusing the shape here keeps
        // the fixture gate as strict as the object rail, which has no entry
        // that carries both (`research/docs/23` §3.3, v54 review H1).
        if case.base_vertex != 0 {
            return Err(
                format!("{where_}: the reviewed multisample shapes carry no base vertex").into(),
            );
        }
        // The raster's expectation shape depends on what it opens
        // (`research/docs/23` §3.3, v51/v53/v55): a colour-only raster resolves
        // fragment output and clear into the partially covered texels and has
        // to claim that shape, while a raster that opens a rail-owned depth or
        // stencil surface is the depth pair's own shape — both primitives cover
        // every sample, the first one decides which survives per sample, and
        // every texel is one fragment output.
        if case.depth.is_some() {
            let (depth, test) = case_depth(case, &where_)?;
            let depth = depth.ok_or(format!("{where_}: a depth surface needs its attachment"))?;
            if depth.store.is_some() {
                // A stored multisampled depth surface is admitted from v57
                // on, through the resolve the case then has to state: its
                // texels are only observable as the resolve's reduction, so a
                // stored surface without one is refused, and a filter outside
                // the closed family is refused by name
                // (`research/docs/23` §3.3, v57).
                let Some(resolve) = &case.depth_resolve else {
                    return Err(format!(
                        "{where_}: a stored multisampled depth surface needs its depth resolve"
                    )
                    .into());
                };
                if !matches!(resolve.filter.as_str(), "sample0" | "min" | "max") {
                    return Err(format!(
                        "{where_}: unsupported depth resolve filter {:?}",
                        resolve.filter
                    )
                    .into());
                }
            } else if case.depth_resolve.is_some() {
                // The resolve is the stored surface's own tail: a depth
                // resolve beside a surface the pass discards is refused
                // instead of silently ignored (`research/docs/23` §3.3, v57).
                return Err(
                    format!("{where_}: a depth resolve needs a stored depth surface").into(),
                );
            }
            // The combined shape's stencil half states the same stored
            // surface rule its single-surface sibling does
            // (`research/docs/23` §3.3, v60).
            if let Some(stencil) = &case.stencil {
                if stencil.store.is_some() && case.stencil_resolve.is_none() {
                    return Err(format!(
                        "{where_}: a stored multisampled stencil surface needs its stencil \
                         resolve"
                    )
                    .into());
                }
            }
            if test.is_none() {
                return Err(
                    format!("{where_}: a multisampled depth surface needs its test").into(),
                );
            }
            if combined_pair && case.stencil_resolve.is_none() {
                // The v66 rail-owned pair's whole point is its partially
                // covered column: the two faces' tests decide per sample, so
                // the case has to claim the partial coverage its expectation
                // then shows (`research/docs/23` §3.3, v66).
                if case.coverage.as_deref() != Some("partial") {
                    return Err(format!(
                        "{where_}: the rail-owned combined pair claims the partial coverage it \
                         resolves"
                    )
                    .into());
                }
            } else if case.coverage.is_some() {
                return Err(format!(
                    "{where_}: a multisample pass with a depth surface claims no partial coverage"
                )
                .into());
            }
        } else if case.stencil.is_some() {
            let (stencil, test) = case_stencil(case, &where_)?;
            let stencil =
                stencil.ok_or(format!("{where_}: a stencil surface needs its attachment"))?;
            if stencil.store.is_some() {
                // A stored multisampled stencil surface is admitted from v60
                // on, through the resolve the case then has to state: its
                // texels are only observable as the resolve's reduction, so a
                // stored surface without one is refused, and a filter outside
                // the closed family is refused by name
                // (`research/docs/23` §3.3, v60).
                let Some(resolve) = &case.stencil_resolve else {
                    return Err(format!(
                        "{where_}: a stored multisampled stencil surface needs its stencil resolve"
                    )
                    .into());
                };
                if !matches!(resolve.filter.as_str(), "sample0" | "depth_resolved_sample") {
                    return Err(format!(
                        "{where_}: unsupported stencil resolve filter {:?}",
                        resolve.filter
                    )
                    .into());
                }
            } else if case.stencil_resolve.is_some() {
                // The resolve is the stored surface's own tail: a stencil
                // resolve beside a surface the pass discards is refused
                // instead of silently ignored (`research/docs/23` §3.3, v60).
                return Err(
                    format!("{where_}: a stencil resolve needs a stored stencil surface").into(),
                );
            }
            if test.is_none() {
                return Err(
                    format!("{where_}: a multisampled stencil surface needs its test").into(),
                );
            }
            if case.coverage.is_some() {
                return Err(format!(
                    "{where_}: a multisample pass with a stencil surface claims no partial \
                     coverage"
                )
                .into());
            }
        } else {
            // The colour-only raster admits both expectation shapes the v61
            // increment reviews: `coverage: partial` is the v51 edge fixture's
            // resolve rule, and an absent claim is the v61 full-coverage
            // fixtures' uniform rule — every texel is the fragment output.
            // The general gate above already held a present claim to
            // `"partial"`, so nothing further to refuse here.
        }
    }
    // The wildcard channel (`research/docs/23` §3.3, v33): a case may name the
    // texels it does not claim, and only a `dontcare` load has bytes that may
    // legitimately be unclaimed. The list is the single-attachment shape's, it
    // has to leave at least one texel observed, and every entry has to name a
    // texel of that attachment.
    if let Some(wildcards) = &case.wildcard_texels {
        let (attachment, _) = shapes
            .first()
            .ok_or(format!("{where_}: a wildcard list needs an attachment"))?;
        if !single {
            return Err(
                format!("{where_}: the wildcard channel is the single-attachment shape").into(),
            );
        }
        if attachment.load != "dontcare" {
            return Err(
                format!("{where_}: only a dontcare load may leave texels unclaimed").into(),
            );
        }
        if wildcards.is_empty() {
            return Err(format!("{where_}: a wildcard list has to name at least one texel").into());
        }
        let texel_count = attachment.width * attachment.height;
        let unique = wildcards.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != wildcards.len() {
            return Err(format!("{where_}: duplicate wildcard texel").into());
        }
        if unique.len() as u64 >= texel_count || unique.iter().any(|texel| *texel >= texel_count) {
            return Err(format!(
                "{where_}: a wildcard list has to leave at least one texel observed"
            )
            .into());
        }
    }
    // The constrained wildcard channel (`research/docs/23` §3.3, v67): the
    // same single-attachment `dontcare` shape as the free list, but every
    // named texel states the closed set of values its bytes may carry instead
    // of leaving them unclaimed. The two channels cannot be stated together —
    // a byte either has a claim or it has none — and every entry is held to
    // the four-byte lowercase spelling here; what the values *are* is the
    // case's own shape and is checked with the attachment's colours below.
    if case.wildcard_allowed_texels.is_some() && case.wildcard_texels.is_some() {
        return Err(format!("{where_}: the two wildcard channels are mutually exclusive").into());
    }
    if let Some(allowed) = &case.wildcard_allowed_texels {
        let (attachment, _) = shapes
            .first()
            .ok_or(format!("{where_}: an allowed set needs an attachment"))?;
        if !single {
            return Err(format!(
                "{where_}: the constrained wildcard channel is the single-attachment shape"
            )
            .into());
        }
        // A `dontcare` load's pre-pass contents are undefined, which is what a
        // constrained claim bounds; a cleared raster has no undefined content,
        // so the channel is only admitted there beside a multisample raster
        // that claims the partial coverage its allowed set resolves
        // (`research/docs/23` §3.3, v67/v69). A loaded multisample raster is the
        // same shape one route along: its declared window is the one repeated
        // texel the rail seeds every sample of the raster with, so the resolve
        // is a mix of two colours the fixture owns
        // (`research/docs/23` §82, v82). Beside a single-sample load nothing is
        // unclaimed, so the channel keeps its refusal there.
        match attachment.load.as_str() {
            "dontcare" => {}
            "clear" => {
                if case.multisample.is_none() {
                    return Err(format!(
                        "{where_}: only a dontcare load may leave texels unclaimed by an \
                         allowed set"
                    )
                    .into());
                }
                if case.coverage.as_deref() != Some("partial") {
                    return Err(format!(
                        "{where_}: a cleared multisample raster states the partial coverage \
                         its allowed set resolves"
                    )
                    .into());
                }
            }
            "load" => {
                if case.multisample.is_none() {
                    return Err(
                        format!("{where_}: a loaded attachment has no unclaimed texel").into(),
                    );
                }
                if case.coverage.as_deref() != Some("partial") {
                    return Err(format!(
                        "{where_}: a loaded multisample raster states the partial coverage its \
                         seed resolves"
                    )
                    .into());
                }
            }
            other => return Err(format!("{where_}: unknown attachment load op {other:?}").into()),
        }
        if allowed.is_empty() {
            return Err(format!("{where_}: an allowed set has to name at least one texel").into());
        }
        let texel_count = attachment.width * attachment.height;
        let mut seen = BTreeSet::new();
        for (position, entry) in allowed.iter().enumerate() {
            if entry.index >= texel_count {
                return Err(format!(
                    "{where_}: wildcard texel {} is outside the attachment",
                    entry.index
                )
                .into());
            }
            if !seen.insert(entry.index) {
                return Err(format!("{where_}: duplicate wildcard texel {}", entry.index).into());
            }
            if entry.allowed.is_empty() {
                return Err(format!("{where_}: allowed set {position} has to name a value").into());
            }
            let mut distinct = BTreeSet::new();
            for (value_position, value) in entry.allowed.iter().enumerate() {
                let bytes = unhex(value)?;
                if value.len() != 8 || value != &hex(&bytes) {
                    return Err(format!(
                        "{where_}: allowed set {position} value {value_position} has to be \
                         four lowercase bytes"
                    )
                    .into());
                }
                if !distinct.insert(bytes) {
                    return Err(format!(
                        "{where_}: allowed set {position} names the value {value} twice"
                    )
                    .into());
                }
            }
        }
    }
    // The values a constrained wildcard texel may take (`research/docs/23`
    // §3.3, v67) are the case's own arithmetic, not the fixture's choice: a
    // resolve of `samples` samples where the draw wrote `k` of them and the
    // rest kept the pass's reference colour can only produce the exact
    // k-of-`samples` mixes of the fragment output and that colour, so every
    // declared set has to be exactly those values — a value outside them would
    // accept a byte no reviewed rail may produce, and a missing one would
    // refuse a resolve that is correct. The single-sample shape's mixes are
    // the two colours themselves.
    if let Some(allowed) = &case.wildcard_allowed_texels {
        let fragment = unhex(shapes[0].1.as_deref().ok_or(format!(
            "{where_}: a constrained wildcard texel needs the fragment output"
        ))?)?;
        let reference = wildcard_reference(case, shapes[0].0)?;
        if reference.len() != 4 {
            return Err(format!("{where_}: a reference colour is four bytes").into());
        }
        if reference == fragment {
            return Err(format!(
                "{where_}: a constrained wildcard texel needs a reference colour other than the \
                 fragment output"
            )
            .into());
        }
        let samples = case
            .multisample
            .as_ref()
            .map_or(1_u64, |multisample| multisample.sample_count);
        let mut candidates = BTreeSet::new();
        for covered in 0..=samples {
            let mut mix = Vec::with_capacity(4);
            let mut exact = true;
            for channel in 0..4 {
                let total = u64::from(fragment[channel]) * covered
                    + u64::from(reference[channel]) * (samples - covered);
                if !total.is_multiple_of(samples) {
                    exact = false;
                    break;
                }
                mix.push(u8::try_from(total / samples)?);
            }
            if exact {
                candidates.insert(mix);
            }
        }
        if candidates.len() <= 1 {
            return Err(format!(
                "{where_}: the fragment output and the reference colour have to differ"
            )
            .into());
        }
        for entry in allowed {
            let declared = entry
                .allowed
                .iter()
                .map(|value| unhex(value))
                .collect::<Result<BTreeSet<_>>>()?;
            for value in &declared {
                if !candidates.contains(value) {
                    return Err(format!(
                        "{where_}: texel {} states the allowed value 0x{}, which is not an exact \
                         mix of the fragment output and the reference colour",
                        entry.index,
                        hex(value)
                    )
                    .into());
                }
            }
            if declared != candidates {
                return Err(format!(
                    "{where_}: texel {} does not state the exact mix set of the case's colours",
                    entry.index
                )
                .into());
            }
        }
    }
    let mut parsed = Vec::new();
    let mut stored = Vec::new();
    for (attachment, expected_hex) in &shapes {
        // Both 8-bit UNORM layouts are admitted from v21 on (`docs/23` §3.3):
        // the reviewed fragment stage stores the same colour either way, and
        // the attachment's own layout decides which channel lands in which
        // byte.
        if attachment.format != "rgba8_unorm"
            && attachment.format != "bgra8_unorm"
            && attachment.format != "r32float"
        {
            return Err(format!("{where_}: unsupported attachment format").into());
        }
        if attachment.width == 0
            || attachment.height == 0
            || attachment.width > REVIEWED_ATTACHMENT_CEILING
            || attachment.height > REVIEWED_ATTACHMENT_CEILING
        {
            return Err(format!(
                "{where_}: the attachment extent is one to {REVIEWED_ATTACHMENT_CEILING} \
                     texels per axis"
            )
            .into());
        }
        if attachment.allocation == 0 || attachment.view == 0 {
            return Err(format!("{where_}: zero attachment identity").into());
        }
        if case.viewport != [0, 0, attachment.width, attachment.height] {
            return Err(format!("{where_}: the viewport must cover the attachment").into());
        }
        let extent = usize::try_from(
            attachment
                .width
                .checked_mul(attachment.height)
                .and_then(|texels| texels.checked_mul(4))
                .ok_or("attachment extent overflows")?,
        )?;
        match attachment.store.as_str() {
            "store" => {
                // R5a (`research/docs/23` §73): a rule-expected attachment
                // states its expectation as the reviewed per-texel rule instead
                // of a hex string. The rule's own plane is computed here, so
                // every expectation check below reads exactly the texels the
                // capture's digest and windows cover.
                let texels = match case.expected_rule.as_deref() {
                    Some(rule) => rule_bytes(rule, attachment.width, attachment.height)?,
                    None => {
                        let expected_hex = expected_hex
                            .as_deref()
                            .ok_or(format!("{where_}: a stored attachment needs expected_hex"))?;
                        unhex(expected_hex)?
                    }
                };
                if texels.len() != extent {
                    return Err(format!(
                        "{where_}: expected texel bytes do not match the attachment"
                    )
                    .into());
                }
                // The uniform-expectation rule belongs to the clearing shape:
                // a `Load` case deliberately mixes the fragment output with
                // the bytes the load handed it (`research/docs/23` §3.3).
                let uniform_texel = texels.chunks_exact(4).all(|chunk| chunk == &texels[..4]);
                match attachment.load.as_str() {
                    "clear" => {
                        let texel = &texels[..4];
                        // The instanced fixture is reviewed against one 4×4
                        // `rgba8_unorm` attachment, whose two halves are two
                        // texel columns each (`research/docs/23` §3.3, v31):
                        // the module's half-width shift only splits the
                        // attachment evenly at this extent and in this channel
                        // order.
                        if geometry == RenderGeometry::InstancedPair {
                            if attachment.format != "rgba8_unorm" {
                                return Err(format!(
                                    "{where_}: the reviewed instanced attachment is rgba8_unorm"
                                )
                                .into());
                            }
                            if attachment.width != 4 || attachment.height != 4 {
                                return Err(format!(
                                    "{where_}: the reviewed instanced attachment is 4x4 texels"
                                )
                                .into());
                            }
                        }
                        // A scissored pass covers a known rectangle: inside it
                        // the texels are the fragment output and outside it the
                        // clear colour, which is exactly what the comparator
                        // checks. Without a scissor the milestone's stricter
                        // rule stays: every texel is the output
                        // (`research/docs/23` §3.3, v29).
                        if let Some(multisample) = &case.multisample {
                            // A raster that opens a rail-owned depth or stencil
                            // surface keeps the pair's own expectation shape
                            // (`research/docs/23` §3.3, v53/v55): both
                            // primitives cover every sample, the first one
                            // decides which survives per sample, and every
                            // texel is one fragment output — so the resolve
                            // rule below, which exists for a *partially
                            // covered* raster, is not the one that applies.
                            // The combined depth-stencil shape's far triangle
                            // fails the depth test on the samples the near
                            // triangle covered, so the mixed column carries
                            // the colour resolve's k-of-`sample_count` mix
                            // like a partially covered raster does
                            // (`research/docs/23` §3.3, v60). The v61
                            // full-coverage colour-only fixtures keep the
                            // uniform rule too: their expectation is the
                            // fragment output on every texel, and a mixed
                            // texel would claim a raster the fixture does not
                            // describe (`research/docs/23` §3.3, v61).
                            // The v66 rail-owned combined pair is the exception
                            // to the masked pair's uniform rule: its three
                            // triangles share one partial coverage, so the
                            // mixed column carries the same k-of-`sample_count`
                            // resolve a partially covered colour-only raster
                            // does (`research/docs/23` §3.3, v66).
                            let combined_pair = case.depth.is_some()
                                && case.stencil.is_some()
                                && case.stencil_resolve.is_none();
                            if ((case.depth.is_some() || case.stencil.is_some())
                                && case.stencil_resolve.is_none()
                                && !combined_pair)
                                || (case.depth.is_none()
                                    && case.stencil.is_none()
                                    && case.coverage.as_deref() != Some("partial"))
                            {
                                if !uniform_texel {
                                    return Err(format!(
                                        "{where_}: a masked multisample expectation has to be one \
                                         fragment output"
                                    )
                                    .into());
                                }
                            } else {
                                // The multisample resolve (`research/docs/23` §3.3,
                                // v51): every texel is the arithmetic mean of the
                                // samples a primitive covered, so the expectation
                                // has to be a k-of-`sample_count` mix of the
                                // fragment output and the clear colour — and at
                                // least one texel has to be a *partial* mix, which
                                // is the one byte pattern a single-sample raster
                                // cannot produce.
                                let clear_colour = unhex(attachment.clear_hex.as_deref().ok_or(
                                    format!("{where_}: a clear attachment needs clear_hex"),
                                )?)?;
                                if clear_colour.len() != 4 {
                                    return Err(
                                        format!("{where_}: a clear colour is four bytes").into()
                                    );
                                }
                                let samples = u32::try_from(multisample.sample_count)?;
                                let fragment = [texel[0], texel[1], texel[2], texel[3]];
                                // A texel the case leaves to its allowed set
                                // (`research/docs/23` §3.3, v69) states its claim
                                // as that closed set instead of one byte, so the
                                // exact resolve rule holds for every texel the
                                // case does pin.
                                let claimed = case
                                    .wildcard_allowed_texels
                                    .as_ref()
                                    .map(|entries| {
                                        entries
                                            .iter()
                                            .map(|entry| entry.index)
                                            .collect::<BTreeSet<_>>()
                                    })
                                    .unwrap_or_default();
                                let mut partial = 0_usize;
                                for (index, chunk) in texels.chunks_exact(4).enumerate() {
                                    if claimed.contains(&u64::try_from(index)?) {
                                        continue;
                                    }
                                    let mut covered = None;
                                    for count in 0..=samples {
                                        let Some(mixed) =
                                            resolve_texel(&fragment, &clear_colour, count, samples)
                                        else {
                                            continue;
                                        };
                                        if chunk == mixed {
                                            covered = Some(count);
                                            break;
                                        }
                                    }
                                    let Some(count) = covered else {
                                        return Err(format!(
                                            "{where_}: texel {index} is not the resolve of any \
                                         coverage of the {samples}-sample raster"
                                        )
                                        .into());
                                    };
                                    if count > 0 && count < samples {
                                        partial += 1;
                                    }
                                }
                                if !claimed.is_empty()
                                    && !(1..samples).any(|count| {
                                        resolve_texel(&fragment, &clear_colour, count, samples)
                                            .is_some()
                                    })
                                {
                                    // The allowed sets are held to the case's own
                                    // arithmetic where they are parsed; what stays
                                    // to check here is that they admit a partial mix
                                    // — the one shape a single-sample raster could
                                    // not produce (`research/docs/23` §3.3, v69).
                                    return Err(format!(
                                        "{where_}: a cleared multisample raster that leaves a texel \
                                         to its allowed set needs an exactly representable partial \
                                         mix of its colours"
                                    )
                                    .into());
                                }
                                // A case that leaves texels to their allowed set
                                // states the partial coverage through that set
                                // instead of a pinned byte, so the rule binds
                                // the texels it does pin (`research/docs/23`
                                // §3.3, v69).
                                if claimed.is_empty() && partial == 0 {
                                    return Err(format!(
                                        "{where_}: a multisample expectation needs at least one \
                                     partially covered texel"
                                    )
                                    .into());
                                }
                            }
                        } else if case.coverage.as_deref() == Some("partial") {
                            let clear_colour =
                                unhex(attachment.clear_hex.as_deref().ok_or(format!(
                                    "{where_}: a clear attachment needs clear_hex"
                                ))?)?;
                            if clear_colour.len() != 4 {
                                return Err(
                                    format!("{where_}: a clear colour is four bytes").into()
                                );
                            }
                            // The draw covers part of the attachment: every
                            // texel is the fragment output or the clear colour,
                            // and both have to appear
                            // (`research/docs/23` §3.3, v38).
                            let mut drawn = 0_usize;
                            let mut kept = 0_usize;
                            for chunk in texels.chunks_exact(4) {
                                if chunk == &texels[..4] {
                                    drawn += 1;
                                } else if chunk == clear_colour.as_slice() {
                                    kept += 1;
                                } else {
                                    return Err(format!(
                                        "{where_}: a partial coverage claim needs every texel \
                                         to be the fragment output or the clear colour"
                                    )
                                    .into());
                                }
                            }
                            if drawn == 0 || kept == 0 {
                                return Err(format!(
                                    "{where_}: a partial coverage claim needs both drawn and \
                                     clear texels"
                                )
                                .into());
                            }
                        } else if let Some([x, y, scissor_width, scissor_height]) = case.scissor {
                            let texel_bytes = [texel[0], texel[1], texel[2], texel[3]];
                            let clear_bytes =
                                unhex(attachment.clear_hex.as_deref().ok_or(format!(
                                    "{where_}: a clear attachment needs clear_hex"
                                ))?)?;
                            if clear_bytes.len() != 4 {
                                return Err(
                                    format!("{where_}: a clear colour is four bytes").into()
                                );
                            }
                            let mut covered = 0;
                            for (index, chunk) in texels.chunks_exact(4).enumerate() {
                                let column = index as u64 % attachment.width;
                                let row = index as u64 / attachment.width;
                                let inside = column >= x
                                    && column < x + scissor_width
                                    && row >= y
                                    && row < y + scissor_height;
                                let expected_chunk = if inside {
                                    &texel_bytes
                                } else {
                                    clear_bytes.as_slice().try_into().map_err(|_| {
                                        format!("{where_}: a clear colour is four bytes")
                                    })?
                                };
                                if chunk != expected_chunk {
                                    return Err(format!(
                                        "{where_}: texel {index} does not match the declared scissor"
                                    )
                                    .into());
                                }
                                covered += usize::from(inside);
                            }
                            let total = texels.chunks_exact(4).count();
                            if covered == 0 || covered == total {
                                return Err(format!(
                                    "{where_}: the scissor has to clip part of the attachment"
                                )
                                .into());
                            }
                        } else if geometry == RenderGeometry::InstancedPair {
                            // The instanced fixture covers each half with its
                            // own instance tint (`research/docs/23` §3.3,
                            // v31): the left half has to be the first tint and
                            // the right half the second, in the reviewed
                            // order. A uniform expectation could not show the
                            // per-instance stream stepped at all, and a swapped
                            // pair would read as "the rails agreed on the wrong
                            // halves".
                            for (index, chunk) in texels.chunks_exact(4).enumerate() {
                                let column = index as u64 % attachment.width;
                                let expected_chunk = if column < attachment.width / 2 {
                                    INSTANCED_TINT_RED_BYTES
                                } else {
                                    INSTANCED_TINT_GREEN_BYTES
                                };
                                if chunk != expected_chunk {
                                    return Err(format!(
                                        "{where_}: texel {index} has to carry the instance tint of its half"
                                    )
                                    .into());
                                }
                            }
                        } else if case.fragment_textures.is_some() {
                            // The render sampler's expectation is the uploaded
                            // texture itself (`research/docs/23` §3.3, v70), and
                            // `reviewed_sampled_geometry` already held it to
                            // that byte for byte — including that the texels
                            // are pairwise distinct and none is the clear
                            // colour, which is what rules out a uniform store.
                        } else if !uniform_texel {
                            return Err(format!(
                                "{where_}: every texel of a cleared attachment has to be the fragment output"
                            )
                            .into());
                        }
                        let clear = unhex(
                            attachment
                                .clear_hex
                                .as_deref()
                                .ok_or(format!("{where_}: a clear attachment needs clear_hex"))?,
                        )?;
                        if clear.len() != 4 {
                            return Err(format!("{where_}: a clear colour is four bytes").into());
                        }
                        if attachment.initial_hex.is_some() {
                            return Err(format!(
                                "{where_}: a cleared attachment carries no initial bytes"
                            )
                            .into());
                        }
                        if clear == texel {
                            return Err(format!(
                                "{where_}: the clear colour equals the expected texel"
                            )
                            .into());
                        }
                        if geometry == RenderGeometry::InstancedPair
                            && clear == INSTANCED_TINT_GREEN_BYTES
                        {
                            return Err(format!(
                                "{where_}: the clear colour equals the second instance tint"
                            )
                            .into());
                        }
                    }
                    "load" => {
                        let initial = unhex(attachment.initial_hex.as_deref().ok_or(format!(
                            "{where_}: a loaded attachment needs its previous texels"
                        ))?)?;
                        if attachment.clear_hex.is_some() {
                            return Err(format!(
                                "{where_}: a loaded attachment carries no clear colour"
                            )
                            .into());
                        }
                        if initial.len() != extent {
                            return Err(format!(
                                "{where_}: initial texels do not match the attachment"
                            )
                            .into());
                        }
                        if initial == texels {
                            return Err(format!(
                                "{where_}: the initial texels equal the expectation"
                            )
                            .into());
                        }
                        // A loading pass uploads the declaring view's own bytes
                        // (`research/docs/23` §3.3), so the case's `initial_hex`
                        // has to be exactly what that case declares: a trace
                        // whose declaration and expectation disagree would
                        // report bytes the rail never held.
                        let declared = suite
                            .cases
                            .iter()
                            .find(|declared| declared.id == case.declaring_case)
                            .and_then(|declared| {
                                declared.buffers.iter().find(|buffer| {
                                    buffer.allocation == attachment.allocation
                                        && buffer.view == attachment.view
                                })
                            })
                            .ok_or(format!(
                                "{where_}: the declaring case does not carry the attachment view"
                            ))?;
                        if declared.initial_bytes()? != initial {
                            return Err(format!(
                                "{where_}: the declared view's bytes are not the attachment's initial texels"
                            )
                            .into());
                        }
                        // The loaded multisampled raster
                        // (`research/docs/23` §82, v82): the transfer commands
                        // are single-sample at both ends, so the rail states
                        // the declared window as one clear its own seed pass
                        // writes into every sample and the measured pass opens
                        // the image with `LOAD`. A clear value is one colour for
                        // the whole attachment, so the window has to be one
                        // repeated texel, and every pinned texel is the exact
                        // k-of-`sample_count` resolve of that seed and the
                        // fragment output.
                        if let Some(multisample) = &case.multisample {
                            let seed = initial[..4].to_vec();
                            if initial
                                .chunks_exact(4)
                                .any(|texel| texel != seed.as_slice())
                            {
                                return Err(format!(
                                    "{where_}: a seeded multisample raster loads one repeated \
                                     texel; a per-texel seed is not a shape the reviewed rails \
                                     execute"
                                )
                                .into());
                            }
                            let samples = u32::try_from(multisample.sample_count)?;
                            let fragment = [texels[0], texels[1], texels[2], texels[3]];
                            let claimed = case
                                .wildcard_allowed_texels
                                .as_ref()
                                .map(|entries| {
                                    entries
                                        .iter()
                                        .map(|entry| entry.index)
                                        .collect::<BTreeSet<_>>()
                                })
                                .unwrap_or_default();
                            let mut partial = 0_usize;
                            for (index, chunk) in texels.chunks_exact(4).enumerate() {
                                if claimed.contains(&u64::try_from(index)?) {
                                    continue;
                                }
                                let mut covered = None;
                                for count in 0..=samples {
                                    let Some(mixed) =
                                        resolve_texel(&fragment, &seed, count, samples)
                                    else {
                                        continue;
                                    };
                                    if chunk == mixed {
                                        covered = Some(count);
                                        break;
                                    }
                                }
                                let Some(count) = covered else {
                                    return Err(format!(
                                        "{where_}: texel {index} is not the resolve of any \
                                         coverage of the {samples}-sample raster"
                                    )
                                    .into());
                                };
                                if count > 0 && count < samples {
                                    partial += 1;
                                }
                            }
                            if claimed.is_empty() && partial == 0 {
                                return Err(format!(
                                    "{where_}: a multisample expectation needs at least one \
                                     partially covered texel"
                                )
                                .into());
                            }
                        } else {
                            // Partial coverage, in both directions: every texel is
                            // either the byte the load handed it or the pass's
                            // fragment output, every drawn texel carries the *same*
                            // output, and both halves appear.
                            let mut drawn: Option<&[u8]> = None;
                            let mut drawn_count = 0_usize;
                            let mut kept_count = 0_usize;
                            for (position, texel) in texels.chunks_exact(4).enumerate() {
                                let previous = &initial[position * 4..position * 4 + 4];
                                if texel == previous {
                                    kept_count += 1;
                                    continue;
                                }
                                match drawn {
                                    None => drawn = Some(texel),
                                    Some(value) if value == texel => {}
                                    Some(_) => {
                                        return Err(format!(
                                        "{where_}: drawn texels disagree about the fragment output"
                                    )
                                        .into())
                                    }
                                }
                                drawn_count += 1;
                            }
                            if drawn_count == 0 || kept_count == 0 {
                                return Err(format!(
                                "{where_}: a loaded attachment needs at least one drawn and one kept texel, \
                                 got {drawn_count} drawn and {kept_count} kept"
                            )
                            .into());
                            }
                        }
                    }
                    "dontcare" => {
                        // Undefined pre-pass contents (`docs/23` §13, v20):
                        // the pass starts from nothing, so every texel it
                        // *claims* has to be the fragment output and neither a
                        // clear colour nor initial bytes may travel with the
                        // attachment. The unclaimed texels are the wildcard
                        // list's (`research/docs/23` §3.3, v33), and without
                        // one the stricter v20 rule and its exact message stay.
                        // The constrained channel (`research/docs/23` §3.3, v67)
                        // is the other way to leave a texel unclaimed: the texel's
                        // bytes then have to be one of the values it states, and
                        // that set is held to the case's own colours below.
                        if let Some(allowed) = &case.wildcard_allowed_texels {
                            let mut claimed = BTreeMap::new();
                            for entry in allowed {
                                let values = entry
                                    .allowed
                                    .iter()
                                    .map(|value| unhex(value))
                                    .collect::<Result<Vec<_>>>()?;
                                claimed.insert(entry.index, values);
                            }
                            for (position, chunk) in texels.chunks_exact(4).enumerate() {
                                let position = u64::try_from(position)?;
                                let Some(values) = claimed.get(&position) else {
                                    if chunk != &texels[..4] {
                                        return Err(format!(
                                    "{where_}: texel {position} of a dontcare load has to be \
                                     the fragment output"
                                )
                                        .into());
                                    }
                                    continue;
                                };
                                if !values.iter().any(|value| value.as_slice() == chunk) {
                                    return Err(format!(
                                "{where_}: texel {position} of a dontcare load is not one of \
                                 its declared values"
                            )
                                    .into());
                                }
                            }
                        }
                        match &case.wildcard_texels {
                            None => {
                                if case.wildcard_allowed_texels.is_none() && !uniform_texel {
                                    return Err(format!(
                                "{where_}: every texel of a dontcare load has to be the fragment output"
                            )
                                    .into());
                                }
                            }
                            Some(wildcards) => {
                                let wildcard = wildcards.iter().copied().collect::<BTreeSet<_>>();
                                let texel_count = u64::try_from(texels.chunks_exact(4).count())?;
                                for (position, chunk) in texels.chunks_exact(4).enumerate() {
                                    let position = u64::try_from(position)?;
                                    if wildcard.contains(&position) {
                                        continue;
                                    }
                                    if chunk != &texels[..4] {
                                        return Err(format!(
                                            "{where_}: texel {position} of a dontcare load has to be the fragment output"
                                        )
                                        .into());
                                    }
                                }
                                if wildcard.len() as u64 >= texel_count
                                    || wildcard.iter().any(|texel| *texel >= texel_count)
                                {
                                    return Err(format!(
                                        "{where_}: a wildcard list has to leave at least one texel observed"
                                    )
                                    .into());
                                }
                            }
                        }
                        // The reference colour of a constrained wildcard texel
                        // (`research/docs/23` §3.3, v67) is stated beside a
                        // `dontcare` load as well: it is the second colour the
                        // resolve's mixes are built from, not a colour the pass
                        // clears the attachment to.
                        if attachment.clear_hex.is_some() && case.wildcard_allowed_texels.is_none()
                        {
                            return Err(format!(
                                "{where_}: a dontcare load carries no clear colour"
                            )
                            .into());
                        }
                        if attachment.initial_hex.is_some() {
                            return Err(format!(
                                "{where_}: a dontcare load carries no initial bytes"
                            )
                            .into());
                        }
                        // The declaring case still pins the view's bytes, but
                        // a dontcare load never hands them to the pass, so
                        // they must differ from the expectation: that is what
                        // shows the undefined contents never entered the
                        // observation.
                        let declared = suite
                            .cases
                            .iter()
                            .find(|declared| declared.id == case.declaring_case)
                            .and_then(|declared| {
                                declared.buffers.iter().find(|buffer| {
                                    buffer.allocation == attachment.allocation
                                        && buffer.view == attachment.view
                                })
                            })
                            .ok_or(format!(
                                "{where_}: the declaring case does not carry the attachment view"
                            ))?;
                        if declared.initial_bytes()? == texels {
                            return Err(format!(
                                "{where_}: the declared view's bytes equal the expectation"
                            )
                            .into());
                        }
                    }
                    other => {
                        return Err(format!("{where_}: unknown attachment load op {other:?}").into())
                    }
                }
                stored.push(*attachment);
                parsed.push((*attachment, Some(texels)));
            }
            "dontcare" => {
                if expected_hex.is_some() {
                    return Err(format!(
                        "{where_}: a discarded attachment carries no expected_hex"
                    )
                    .into());
                }
                // The pass still performs the load, so the load's own shape
                // stays pinned even though there is no expectation to compare.
                match attachment.load.as_str() {
                    "clear" => {
                        let clear = unhex(
                            attachment
                                .clear_hex
                                .as_deref()
                                .ok_or(format!("{where_}: a clear attachment needs clear_hex"))?,
                        )?;
                        if clear.len() != 4 {
                            return Err(format!("{where_}: a clear colour is four bytes").into());
                        }
                        if attachment.initial_hex.is_some() {
                            return Err(format!(
                                "{where_}: a cleared attachment carries no initial bytes"
                            )
                            .into());
                        }
                    }
                    "load" => {
                        let initial = unhex(attachment.initial_hex.as_deref().ok_or(format!(
                            "{where_}: a loaded attachment needs its previous texels"
                        ))?)?;
                        if attachment.clear_hex.is_some() {
                            return Err(format!(
                                "{where_}: a loaded attachment carries no clear colour"
                            )
                            .into());
                        }
                        if initial.len() != extent {
                            return Err(format!(
                                "{where_}: initial texels do not match the attachment"
                            )
                            .into());
                        }
                    }
                    "dontcare" => {
                        // A `dontcare` attachment carries no clear colour — the
                        // pass neither reads nor writes its texels. The
                        // constrained wildcard channel's reference colour is
                        // the exception (`research/docs/23` §3.3, v67): it is
                        // the second colour the resolve's mixes are built
                        // from, not a colour this pass clears the attachment to.
                        if attachment.clear_hex.is_some() && case.wildcard_allowed_texels.is_none()
                        {
                            return Err(format!(
                                "{where_}: a dontcare load carries no clear colour"
                            )
                            .into());
                        }
                        if attachment.initial_hex.is_some() {
                            return Err(format!(
                                "{where_}: a dontcare load carries no initial bytes"
                            )
                            .into());
                        }
                    }
                    other => {
                        return Err(format!("{where_}: unknown attachment load op {other:?}").into())
                    }
                }
                parsed.push((*attachment, None));
            }
            _other => {
                return Err(format!("{where_}: a discarded attachment cannot be compared").into());
            }
        }
    }
    // Core admission refuses an all-discarded pass
    // (`AllRenderAttachmentsDiscarded`), so the suite has to keep at least one
    // attachment on the observable surface or "nothing landed" would pass as
    // "landed correctly". The stored depth attachment is a landing too
    // (`research/docs/23` §3.3, v43/v45), which is what makes the depth-only
    // shape — every colour attachment discarded, the depth surface kept —
    // expressible.
    let depth_landing = case
        .depth
        .as_ref()
        .is_some_and(|depth| depth.store.as_deref() == Some("store"));
    if stored.is_empty() && !depth_landing {
        return Err(format!(
            "{where_}: every colour attachment discards, leaving no observable landing point"
        )
        .into());
    }
    // The two reviewed MRT locations write two different byte strings, so a
    // dual case whose locations read back the same texels could not show that
    // both outputs landed (`4080c0ff` vs `ff8040c0`). A discarded location
    // carries no expectation, so it takes no part in the comparison.
    if multiple && parsed[0].1 == parsed[1].1 {
        return Err(format!("{where_}: the two locations read back the same texel").into());
    }
    let mut rails = BTreeSet::new();
    for rail in &case.capture_rails {
        if !RENDER_RAILS.contains(&rail.as_str()) {
            return Err(format!("{where_}: unknown capture rail {rail:?}").into());
        }
        if !rails.insert(rail) {
            return Err(format!("{where_}: duplicate capture rail {rail:?}").into());
        }
    }
    if rails.is_empty() {
        return Err(format!("{where_}: capture_rails cannot be empty").into());
    }
    // Every attachment resolves against the declaring case's own table: one of
    // its declared views has to be the attachment, it has to be read-only (a
    // compute pass that wrote the view the render pass stores would make the
    // order inexpressible), and its byte range has to agree with the extent
    // the attachment restates.
    let declaring = suite
        .cases
        .iter()
        .find(|declared| declared.id == case.declaring_case)
        .ok_or(format!(
            "{where_}: unknown declaring case {}",
            case.declaring_case
        ))?;
    if dispatch_sequence(declaring).len() != 1
        || declaring.command_buffers.is_some()
        || declaring.programs.is_some()
    {
        return Err(format!(
            "{where_}: the declaring case must be one pass over its whole view pool"
        )
        .into());
    }
    let mut seen_identities = BTreeSet::new();
    let mut seen_allocations = BTreeSet::new();
    for (attachment, _texels) in &parsed {
        if !seen_identities.insert((attachment.allocation, attachment.view)) {
            return Err(format!("{where_}: duplicate attachment identity").into());
        }
        if !seen_allocations.insert(attachment.allocation) {
            return Err(
                format!("{where_}: the attachments have to name distinct allocations").into(),
            );
        }
        let matches = declaring
            .buffers
            .iter()
            .filter(|buffer| {
                buffer.allocation == attachment.allocation && buffer.view == attachment.view
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!(
                "{where_}: the declaring case has to declare exactly the attachment view"
            )
            .into());
        }
        let declared = matches[0];
        if declared.access != "read" {
            return Err(
                format!("{where_}: the declaring pass must only read the attachment view").into(),
            );
        }
        // A discarded attachment's declaring view is pinned by the extent the
        // attachment restates; a stored one has already proved its expectation
        // covers exactly that extent.
        if declared.length != attachment.width * attachment.height * 4 {
            return Err(
                format!("{where_}: attachment extent disagrees with the declaring view").into(),
            );
        }
        // A rule-expected attachment reports the digest of its declaring
        // allocation's whole image as well as the view's own plane, so the view
        // has to *be* the image: the R5a fixture declares offset 0 and a
        // view that fills its allocation (`research/docs/23` §73).
        if case.expected_rule.is_some()
            && (declared.offset != 0 || declared.allocation_size != declared.length)
        {
            return Err(format!(
                "{where_}: a rule-expected attachment's declaring view has to be its whole \
                 allocation"
            )
            .into());
        }
    }
    Ok(())
}

fn case_programs(case: &Case) -> Vec<CaseProgram> {
    case.programs.clone().unwrap_or_else(|| {
        vec![CaseProgram {
            entry: case.entry.clone(),
            air: case.air.clone(),
            metal: case.metal.clone(),
            buffer_slots: None,
        }]
    })
}

fn dispatch_sequence(case: &Case) -> Vec<CaseDispatch> {
    case.dispatches.clone().unwrap_or_else(|| {
        vec![CaseDispatch {
            grid: case.grid,
            local: case.local,
            bindings: None,
            program: None,
        }]
    })
}

/// Dispatch indices per command buffer. Legacy fixtures submit one command
/// buffer; the v9 suite splits the same sequence across several.
fn case_command_buffers(case: &Case) -> Vec<Vec<usize>> {
    case.command_buffers
        .clone()
        .unwrap_or_else(|| vec![(0..dispatch_sequence(case).len()).collect()])
}

fn validate_case_command_buffers(suite: &str, case: &Case) -> Result<()> {
    let dispatches = dispatch_sequence(case);
    let Some(groups) = &case.command_buffers else {
        if suite == "compute-buffer-v9" {
            return Err("v9 fixture requires command buffer groups".into());
        }
        return Ok(());
    };
    if suite != "compute-buffer-v9" {
        return Err("command buffer groups are only qualified by the v9 suite".into());
    }
    if !(2..=4).contains(&groups.len()) {
        return Err("v9 fixture needs two to four command buffers".into());
    }
    let mut expected = 0usize;
    for group in groups {
        if group.is_empty() {
            return Err("command buffer group cannot be empty".into());
        }
        for index in group {
            if *index != expected {
                return Err("command buffer groups must partition the dispatch order".into());
            }
            expected += 1;
        }
    }
    if expected != dispatches.len() {
        return Err("command buffer groups must partition the dispatch order".into());
    }
    Ok(())
}

/// Command buffers may each report writes for the same view; the capture
/// reports the final per-view landing, so later writes overlay earlier ones
/// and every written view must end up fully covered.
fn merge_writebacks(
    case: &Case,
    entries: impl IntoIterator<Item = (u64, u64, u64, Vec<u8>)>,
) -> Result<Vec<Writeback>> {
    let mut views = BTreeMap::new();
    for buffer in &case.buffers {
        let length = usize::try_from(buffer.length)?;
        views.insert(
            (buffer.allocation, buffer.view),
            (buffer.offset, vec![0_u8; length], vec![false; length]),
        );
    }
    for (allocation, view, offset, bytes) in entries {
        let (view_offset, data, covered) = views
            .get_mut(&(allocation, view))
            .ok_or("writeback references an unknown view")?;
        let start = usize::try_from(
            offset
                .checked_sub(*view_offset)
                .ok_or("writeback starts before its view")?,
        )?;
        let end = start
            .checked_add(bytes.len())
            .ok_or("writeback range overflow")?;
        data.get_mut(start..end)
            .ok_or("writeback exceeds its view")?
            .copy_from_slice(&bytes);
        covered[start..end].fill(true);
    }
    let mut writebacks = Vec::new();
    for ((allocation, view), (offset, data, covered)) in views {
        if covered.iter().all(|value| *value) {
            writebacks.push(Writeback::encoded(allocation, view, offset, &data));
        } else if covered.iter().any(|value| *value) {
            return Err(format!("writebacks do not cover view {view} exactly").into());
        }
    }
    Ok(writebacks)
}

fn validate_program(program: &CaseProgram) -> Result<()> {
    let (air_path, air_hash, metal_path, metal_hash) = match program.entry.as_str() {
        "read_texture_2d" => (
            "../examples/metal-smoke/shaders/kernel_read_texture_2d.ll",
            "3e969b61d3149bc9351f44c56de6fb85a403557cbcbee7240602539ca794c8df",
            "shaders/read_texture_2d.metal",
            "da21ca69d76018f2911aaf6867f517fca8e41b20d531b6b43df30931563499ee",
        ),
        "read_texture_2d_cell" => (
            "../examples/metal-smoke/shaders/kernel_read_texture_2d_cell.ll",
            "80fe6866bac049de9c1c2b33d9f15a3a133b68c321dfdb16721c991f8dfc23c9",
            "shaders/read_texture_2d_cell.metal",
            "6517da4354381bb46706ec3395d3e449ff08499df37c0c1f2a620a0c04161237",
        ),
        "copy_word" => (
            "../examples/metal-smoke/shaders/kernel_copy_word.ll",
            "292c3e1ff300fd08bf5e39aaa9abe352842eced807138f863e05056f39c56d99",
            "shaders/copy_word.metal",
            "7bfa419aef6eb0abcbec045c1bc15651b2d8f0a7591e07448edc6de6522141bc",
        ),
        // v43's declaring pass: the same reviewed copy over three bindings, so
        // the pass also *declares* the depth attachment's view the render pass
        // then stores (`research/docs/23` §3.3, v43).
        "copy_word_with_witness" => (
            "../examples/metal-smoke/shaders/kernel_copy_word_with_witness.ll",
            "f24e33124da1c228bf4766d32496d8d7ede4fc29d8e6c582c889a36343dfc18e",
            "shaders/copy_word_with_witness.metal",
            "c116fec300f1369069fbcf19d5fbb95e8c5ad07475757c19930075a19ad4367a",
        ),
        // v60's declaring pass: the same reviewed copy over four bindings, so
        // the pass also declares both the depth and the stencil landing the
        // combined render pass stores (`research/docs/23` §3.3, v60).
        "copy_word_with_witnesses" => (
            "../examples/metal-smoke/shaders/kernel_copy_word_with_witnesses.ll",
            "a06dcfcf052e51a8b30e42942bee50d620f53e5cca3107bb6779e0d42f43c37e",
            "shaders/copy_word_with_witnesses.metal",
            "256da53df3f30d52f0b864d545d015bdaa7e8e45c055f4eb9e1f8a1c3963a335",
        ),
        "kernel_dispatch_threads_boundary_barrier" => (
            "../examples/metal-smoke/shaders/kernel_dispatch_threads_boundary_barrier.ll",
            "95076cf4199734f848fd6d761dce13addc7b55354b4d8ee2be16e59287ea5945",
            "shaders/indexed_boundary.metal",
            "7684e493a8704127e39dace5476a006fac564224909c667a57fb5ac9d8291b06",
        ),
        "transform_3d" => (
            "shaders/transform_3d.ll",
            "32bb9a29fef9825972b61cb982106b2bcb7c582413e50350eabc7834532b4df2",
            "shaders/transform_3d.metal",
            "5637cf50a3de44568ff7d3b09341e84111e2a9f6ff9b617181c6368efeacaf9b",
        ),
        "mix_3d" => (
            "shaders/mix_3d.ll",
            "cccc601c6f14d5c76808f927118d77cdcb9e4824591c0492faf735197afaf95f",
            "shaders/mix_3d.metal",
            "e3fa76b0027e6d20e4649fb6e7c07c0ca1618a9ae88fa13815337d2aa7c99bf5",
        ),
        "remap_3d" => (
            "shaders/remap_3d.ll",
            "5388b13783b13a616a3b6952e0c939a120e5d1961e060dd15c11cb54083092ec",
            "shaders/remap_3d.metal",
            "0d715fe43e72fd96218f3fefc9a582c8634092fa10cc79a544869b5dee025a76",
        ),
        "copy_3d" => (
            "shaders/copy_3d.ll",
            "9f379575b8f9ed45e62df27c24761d0030e257f45c6241c649b5caae73cbe9cb",
            "shaders/copy_3d.metal",
            "3d8d71178abe03067508183a87f8c5c6843f1a3092e7f1cb52471ecaaaf0593f",
        ),
        "mrt_declare" => (
            "shaders/mrt_declare.ll",
            "0a5b6740a2839cc4c47a829a7c9badb1bb7d6557df031620a1bf17d7a04393c9",
            "shaders/mrt_declare.metal",
            "c6eeddad6686351c7ec616267f0975f7cc559ee85569a3396c83f64407eff689",
        ),
        // v24: the four-read declaring pass. One submission declares every
        // attachment view of the four-location fixture (and of any shorter
        // list that names this case).
        "mrt_declare4" => (
            "shaders/mrt_declare4.ll",
            "5bf093fb4ad3890e7ee513591e6b943755db1c6850f091580b658e52a092ccd9",
            "shaders/mrt_declare4.metal",
            "b1c51bdf4817b21c9e476eabc627ecb83e727384e3bf2436ac62377702f50a41",
        ),
        _ => return Err("unknown shader entry".into()),
    };
    if program.air.path != air_path
        || program.air.sha256 != air_hash
        || program.metal.path != metal_path
        || program.metal.sha256 != metal_hash
    {
        return Err("unreviewed shader identity".into());
    }
    Ok(())
}
fn validate_case_programs(case: &Case) -> Result<()> {
    let programs = case_programs(case);
    let layout_change = case.id.starts_with("layout_chain_");
    let subsets = case.id.starts_with("subset_chain_");
    if subsets {
        let expected: &[&str] = if case.id == "subset_chain_two" {
            &["transform_3d", "copy_3d"]
        } else {
            &["transform_3d", "copy_3d", "remap_3d"]
        };
        if case.programs.is_none()
            || programs
                .iter()
                .map(|program| program.entry.as_str())
                .collect::<Vec<_>>()
                != expected
            || programs[0].entry != case.entry
            || programs[0].air != case.air
            || programs[0].metal != case.metal
        {
            return Err("unreviewed subset program table".into());
        }
    } else if case.id.starts_with("pipeline_chain_") || layout_change {
        if case.programs.is_none()
            || programs.len() != 2
            || programs[0].entry != "transform_3d"
            || programs[1].entry != if layout_change { "remap_3d" } else { "mix_3d" }
            || programs[0].entry != case.entry
            || programs[0].air != case.air
            || programs[0].metal != case.metal
        {
            return Err("unreviewed program table".into());
        }
    } else if case.programs.is_some() {
        return Err("legacy fixture cannot carry program table".into());
    }
    for (index, program) in programs.iter().enumerate() {
        validate_program(program)?;
        if layout_change || subsets {
            let expected = if index == 0 {
                vec![(0, "read_write", 120), (2, "read", 4), (5, "write", 120)]
            } else if subsets && index == 1 {
                vec![(4, "read", 120), (9, "write", 120)]
            } else {
                vec![(1, "read", 4), (3, "read", 120), (7, "write", 120)]
            };
            let expected = expected
                .into_iter()
                .map(|(binding, access, length)| BufferSlot {
                    binding,
                    access: access.into(),
                    length,
                })
                .collect::<Vec<_>>();
            if program.buffer_slots.as_ref() != Some(&expected) {
                return Err("unreviewed per-program layout".into());
            }
        } else if program.buffer_slots.is_some() {
            return Err("legacy program cannot declare a layout".into());
        }
    }
    Ok(())
}

type CaseShape = (
    &'static str,
    [u64; 3],
    [u64; 3],
    &'static [(u32, &'static str, u64)],
);

fn case_shape(id: &str) -> Result<CaseShape> {
    let copy = (
        "copy_word",
        [1, 1, 1],
        [1, 1, 1],
        &[(0, "read", 4), (1, "write", 4)][..],
    );
    let indexed = |local| {
        (
            "kernel_dispatch_threads_boundary_barrier",
            [10, 3, 1],
            local,
            &[(0, "write", 120)][..],
        )
    };
    let transform = |local| {
        (
            "transform_3d",
            [5, 3, 2],
            local,
            &[(0, "read_write", 120), (2, "read", 4), (5, "write", 120)][..],
        )
    };
    Ok(match id {
        // v11: the texture case binds a sampled texture plus one write-only
        // output buffer; textures are validated by the texture section.
        "sampled_texture_first_texel" => (
            "read_texture_2d",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "write", 64)][..],
        ),
        // v12: the same 4x4 grid runs once as a single 4x4 group and once as
        // sixteen 1x1 groups. Both forms read texel(x, y), so the two shapes
        // must agree on [100..115] for a 4x4 R32Uint image holding 0..15.
        "texture_cell_local_4x4" => (
            "read_texture_2d_cell",
            [4, 4, 1],
            [4, 4, 1],
            &[(0, "write", 64)][..],
        ),
        "texture_cell_local_1x1" => (
            "read_texture_2d_cell",
            [4, 4, 1],
            [1, 1, 1],
            &[(0, "write", 64)][..],
        ),
        // v29: the v12 kernel over two different payloads — ascending and
        // descending cell values — so the writeback has to follow the
        // texture's own bytes instead of a constant (`research/docs/26`
        // §21.3).
        "sampled_cell_ascending_content" | "sampled_cell_descending_content" => (
            "read_texture_2d_cell",
            [4, 4, 1],
            [4, 4, 1],
            &[(0, "write", 64)][..],
        ),
        // v13: the declaring pass of the render suite reads the attachment's
        // own allocation (the whole 2x2x4 image) and copies its first word into
        // a second allocation, so the render submission also proves the
        // declaring pass really read the attachment view. The attachment
        // allocation carries no guard bytes: it is the attachment.
        "render_declaring_copy_word" | "heap_placement_copy_word" | "icb_dispatch_copy_word" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16), (1, "write", 4)][..],
        ),
        // v18/v19: the declaring pass reads both attachment allocations and
        // writes their xor into its own output view, so one submission proves
        // it read two whole-allocation views the render pass then stores into
        // (v19 stores one and discards the other).
        "render_declaring_two_attachments" | "render_declaring_store_and_discard" => (
            "mrt_declare",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16), (1, "read", 16), (2, "write", 4)][..],
        ),
        // v24: one pass declares up to four attachment views (bindings 0..3,
        // each one word of a 16-byte view) and writes their xor into its own
        // output view.
        // v27: the v13 copy_word shape with a 4x4 attachment view (64 bytes).
        // v82: the same shape over the same view, declaring the one repeated
        // texel the multisampled load's seed pass clears every sample with
        // (`research/docs/23` §82).
        "render_declaring_quad_extent" | "render_declaring_multisample_seed" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 64), (1, "write", 4)][..],
        ),
        // R1b: the same copy_word shape over the wider attachment views — the
        // 16x16 case's 1024 bytes and the 64x64 boundary's 16384 bytes — so one
        // submission declares the view each render pass stores into
        // (`research/docs/23` §70).
        "render_declaring_attachment_16x16" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 1024), (1, "write", 4)][..],
        ),
        "render_declaring_attachment_64x64" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16384), (1, "write", 4)][..],
        ),
        // R5a: the same copy_word shape over the reviewed window's own
        // boundary — the 2048x2048 attachment's 16 MiB view, declared with a
        // repeated pattern rather than spelled out (`research/docs/23` §73).
        "render_declaring_attachment_2048x2048" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16777216), (1, "write", 4)][..],
        ),
        // v43: the same 4x4 attachment view, plus the depth attachment's own
        // view as a second read, and the 4-byte output view the reviewed
        // kernel writes. One submission therefore declares every view the
        // depth-storing render pass touches (`research/docs/23` §3.3, v43).
        // v57d: the same v43 shape again, declaring the device-gated pair's own
        // depth landing view so both edge cases resolve against one resource.
        "render_declaring_depth_store" | "render_declaring_depth_resolve" => (
            "copy_word_with_witness",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 64), (1, "write", 4), (2, "read", 64)][..],
        ),
        // v83-v86: the stage-buffer cases' declaring passes. The first is the
        // witness kernel with the colour attachment's own 2x2 view (sixteen
        // bytes), the copy landing and the *writable stage buffer's* pool view
        // as its third read — the view the render pass's writeback lands in
        // (`research/docs/23` §3.3, v86). The second is the plain copy kernel
        // over the same attachment, because the borrowed stage buffer's bytes
        // travel through the owner's own registration instead of a pool view.
        "render_declaring_stage_buffer_sink" => (
            "copy_word_with_witness",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16), (1, "write", 4), (2, "read", 16)][..],
        ),
        "render_declaring_stage_buffer_lease" => (
            "copy_word",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 16), (1, "write", 4)][..],
        ),
        // v49: the same read pair with the *stencil* surface's own one-byte
        // extent (4x4 texels = 16 bytes) as the third read binding, so one
        // submission declares the colour attachment's view and the stencil
        // landing the render pass stores into.
        "render_declaring_stencil_store" => (
            "copy_word_with_witness",
            [1, 1, 1],
            [1, 1, 1],
            &[(0, "read", 64), (1, "write", 4), (2, "read", 16)][..],
        ),
        // v60: the same kernel with both landings declared — the depth view
        // (64 bytes) and the stencil view (16 bytes) as the third and fourth
        // reads — so one submission declares both views the combined render
        // pass stores into (`research/docs/23` §3.3, v60).
        "render_declaring_stencil_resolve" => (
            "copy_word_with_witnesses",
            [1, 1, 1],
            [1, 1, 1],
            &[
                (0, "read", 64),
                (1, "write", 4),
                (2, "read", 64),
                (3, "read", 16),
            ][..],
        ),
        "render_declaring_four_attachments" => (
            "mrt_declare4",
            [1, 1, 1],
            [1, 1, 1],
            &[
                (0, "read", 16),
                (1, "read", 16),
                (2, "read", 16),
                (3, "read", 16),
                (4, "write", 4),
            ][..],
        ),
        // v25: the same four-read kernel with only three attachment views; the
        // fourth read is a 4-byte scratch view, because a declared view the
        // render pass does not attach has to keep its guard bytes.
        "render_declaring_three_attachments" => (
            "mrt_declare4",
            [1, 1, 1],
            [1, 1, 1],
            &[
                (0, "read", 16),
                (1, "read", 16),
                (2, "read", 16),
                (3, "read", 4),
                (4, "write", 4),
            ][..],
        ),
        "copy_word" | "copy_seed_a" | "copy_seed_b" | "copy_pingpong" => copy,
        // v29: the lease cases bind the v1 kernel's own two slots; only the
        // read view's source arm differs (`research/docs/23` §90, R9i).
        "staged_lease_copy_word" | "borrowed_lease_copy_word" => copy,
        // v10: two disjoint views of one allocation. The reversed pair binds
        // the source above the destination so an offset mix-up cannot pass.
        "alias_disjoint_pair" | "alias_disjoint_pair_reversed" => copy,
        "indexed_boundary" | "indexed_tail" => indexed([8, 2, 1]),
        "indexed_full" => indexed([5, 3, 1]),
        "indexed_small_grid" => indexed([16, 4, 1]),
        "indexed_unit" => indexed([1, 1, 1]),
        "transform_tail"
        | "transform_twice"
        | "transform_three_times"
        | "transform_eight_times"
        | "transform_pingpong_two"
        | "transform_pingpong_three"
        | "transform_pingpong_eight"
        | "pipeline_chain_two"
        | "pipeline_chain_three"
        | "pipeline_chain_eight"
        | "layout_chain_two"
        | "layout_chain_three"
        | "layout_chain_eight" => transform([4, 2, 2]),
        "transform_small_grid" => transform([8, 4, 4]),
        "subset_chain_two" => (
            "transform_3d",
            [5, 3, 2],
            [4, 2, 2],
            &[
                (0, "read_write", 120),
                (2, "read", 4),
                (5, "write", 120),
                (8, "write", 120),
            ],
        ),
        "subset_chain_four" | "subset_chain_eight" => (
            "transform_3d",
            [5, 3, 2],
            [4, 2, 2],
            &[
                (0, "read_write", 120),
                (2, "read", 4),
                (5, "write", 120),
                (8, "write", 120),
                (9, "write", 120),
            ],
        ),
        _ => return Err("unknown case identity".into()),
    })
}

fn validate_case_dispatches(case: &Case) -> Result<()> {
    let count = match case.id.as_str() {
        "transform_twice"
        | "transform_pingpong_two"
        | "copy_pingpong"
        | "pipeline_chain_two"
        | "layout_chain_two"
        | "subset_chain_two" => 2,
        "subset_chain_four" => 4,
        "transform_three_times"
        | "transform_pingpong_three"
        | "pipeline_chain_three"
        | "layout_chain_three" => 3,
        "transform_eight_times"
        | "transform_pingpong_eight"
        | "pipeline_chain_eight"
        | "layout_chain_eight"
        | "subset_chain_eight" => 8,
        _ => {
            if case.dispatches.is_some() {
                return Err("single-pass fixture cannot carry a sequence".into());
            }
            return Ok(());
        }
    };
    let dispatches = case
        .dispatches
        .as_ref()
        .ok_or("sequence fixture requires dispatches")?;
    if dispatches.len() != count {
        return Err("wrong sequence dispatch count".into());
    }
    let locals = [[4, 2, 2], [8, 4, 4], [1, 1, 1]];
    let layout_change = case.id.starts_with("layout_chain_");
    let subsets = case.id.starts_with("subset_chain_");
    let mixed = case.id.starts_with("pipeline_chain_") || layout_change;
    let pingpong = case.id.contains("pingpong") || mixed;
    for (i, dispatch) in dispatches.iter().enumerate() {
        let program = if subsets {
            Some([0, 1, 2, 1][i % 4])
        } else {
            mixed.then_some(i % 2)
        };
        if dispatch.program != program {
            return Err("unreviewed program selection".into());
        }
        let (grid, local) = if case.id == "copy_pingpong" {
            ([1, 1, 1], [1, 1, 1])
        } else {
            ([5, 3, 2], locals[i % locals.len()])
        };
        if dispatch.grid != grid || dispatch.local != local {
            return Err("unreviewed sequence dispatch shape".into());
        }
        if pingpong || subsets {
            let expected = if subsets {
                let indices: &[usize] = match i % 4 {
                    0 => &[0, 1, 2],
                    1 => &[2, 3],
                    2 => &[1, 3, 0],
                    _ => &[0, 4],
                };
                indices
                    .iter()
                    .map(|&index| {
                        case.buffers
                            .get(index)
                            .map(|b| b.view)
                            .ok_or("missing subset resource")
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                let mut expected: Vec<_> = case.buffers.iter().map(|buffer| buffer.view).collect();
                let last = if case.id == "copy_pingpong" { 1 } else { 2 };
                if expected.len() <= last {
                    return Err("missing pingpong resource".into());
                }
                if i % 2 == 1 {
                    if layout_change {
                        expected.rotate_left(1);
                    } else {
                        expected.swap(0, last);
                    }
                }
                expected
            };
            if dispatch.bindings.as_ref() != Some(&expected) {
                return Err("unreviewed pingpong binding map".into());
            }
            for (slot, view_id) in selected_slots(case, dispatch).iter().zip(&expected) {
                let view = case
                    .buffers
                    .iter()
                    .find(|view| view.view == *view_id)
                    .ok_or("unknown mapped resource")?;
                if view.length != slot.length {
                    return Err("mapped resource extent differs from selected layout".into());
                }
            }
        } else if dispatch.bindings.is_some() {
            return Err("non-rebinding fixture cannot carry binding maps".into());
        }
    }
    if case.grid != dispatches[0].grid || case.local != dispatches[0].local {
        return Err("sequence first dispatch does not match case".into());
    }
    Ok(())
}

fn selected_slots(case: &Case, dispatch: &CaseDispatch) -> Vec<BufferSlot> {
    if let Some(programs) = &case.programs {
        if let Some(slots) = &programs[dispatch.program.unwrap_or(0)].buffer_slots {
            return slots.clone();
        }
    }
    case.buffers
        .iter()
        .map(|b| BufferSlot {
            binding: b.binding,
            access: b.access.clone(),
            length: b.length,
        })
        .collect()
}

fn ever_writable(case: &Case) -> BTreeSet<u64> {
    if let Some(dispatches) = &case.dispatches {
        dispatches
            .iter()
            .flat_map(|dispatch| {
                selected_slots(case, dispatch)
                    .into_iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.access != "read")
                    .map(|(i, _)| {
                        dispatch
                            .bindings
                            .as_ref()
                            .map_or(case.buffers[i].view, |map| map[i])
                    })
            })
            .collect()
    } else {
        case.buffers
            .iter()
            .filter(|b| b.access != "read")
            .map(|b| b.view)
            .collect()
    }
}

fn verify_transform_contract(pipeline: &CompiledComputePipeline) -> Result<()> {
    let bindings = &pipeline.contract.buffer_bindings;
    if bindings
        .iter()
        .map(|b| (b.metal_binding, b.access))
        .collect::<Vec<_>>()
        != [
            (0, BufferAccess::ReadWrite),
            (2, BufferAccess::Read),
            (5, BufferAccess::Write),
        ]
    {
        return Err("3D fixture sparse/access reflection mismatch".into());
    }
    verify_xyz_access(&bindings[0].footprint)?;
    verify_xyz_access(&bindings[2].footprint)?;
    if bindings[1].footprint != (FootprintProof::Static { max_bytes: 4 }) {
        return Err("3D fixture scalar bias reach mismatch".into());
    }
    Ok(())
}

fn verify_copy_contract(pipeline: &CompiledComputePipeline) -> Result<()> {
    let bindings = &pipeline.contract.buffer_bindings;
    if bindings
        .iter()
        .map(|binding| (binding.metal_binding, binding.access))
        .collect::<Vec<_>>()
        != [(4, BufferAccess::Read), (9, BufferAccess::Write)]
    {
        return Err("copy sparse layout/access reflection mismatch".into());
    }
    for binding in bindings {
        verify_xyz_access(&binding.footprint)?;
    }
    Ok(())
}

fn verify_xyz_access(footprint: &FootprintProof) -> Result<()> {
    let FootprintProof::Affine { accesses } = footprint else {
        return Err("3D fixture must carry an affine footprint".into());
    };
    if accesses.is_empty() {
        return Err("3D fixture has no proven accesses".into());
    }
    for access in accesses {
        let mut strides = [0u64; 3];
        for term in &access.terms {
            let slot = strides
                .get_mut(usize::from(term.axis))
                .ok_or("3D fixture unknown axis")?;
            *slot = slot
                .checked_add(term.stride)
                .ok_or("3D fixture stride overflow")?;
        }
        if access.base_offset != 0 || access.access_size != 4 || strides != [4, 20, 60] {
            return Err("3D fixture footprint must prove 120-byte XYZ reach".into());
        }
    }
    Ok(())
}

fn case_trace(
    device_epoch: DeviceEpoch,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    views: &[BufferView],
    textures: &[TextureView],
    dispatches: &[CaseDispatch],
) -> Result<ComputeTrace> {
    let passes = dispatches
        .iter()
        .map(|dispatch| -> Result<ComputePass> {
            let selected = &programs[dispatch.program.unwrap_or(0)];
            let expected = selected_slots(case, dispatch);
            if selected.contract.buffer_bindings.len() != expected.len()
                || selected.contract.buffer_bindings.iter().zip(&expected).any(
                    |(actual, expected)| {
                        actual.metal_binding != expected.binding
                            || expected.access
                                != match actual.access {
                                    BufferAccess::Read => "read",
                                    BufferAccess::Write => "write",
                                    BufferAccess::ReadWrite => "read_write",
                                    BufferAccess::Unused => "unused",
                                }
                    },
                )
            {
                return Err("source/fixture selected layout mismatch".into());
            }
            let buffers = selected
                .contract
                .buffer_bindings
                .iter()
                .enumerate()
                .map(|(index, slot)| {
                    let view_id = dispatch
                        .bindings
                        .as_ref()
                        .map_or(views[index].view_id.get(), |map| map[index]);
                    let mut resource = views
                        .iter()
                        .find(|view| view.view_id.get() == view_id)
                        .expect("validated binding map")
                        .clone();
                    resource.metal_binding = slot.metal_binding;
                    resource.access = slot.access;
                    resource
                })
                .collect();
            Ok(ComputePass {
                pipeline: selected.pipeline_id,
                buffers,
                dispatch: Dispatch {
                    kind: DispatchKind::ThreadsExact,
                    grid: dispatch.grid,
                    threads_per_threadgroup: dispatch.local,
                },
                textures: textures.to_vec(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // The provider rejects unused pipeline metadata, so each command buffer
    // carries only the programs its own passes reference.
    let mut pipelines: Vec<CompiledComputePipeline> = Vec::new();
    for pass in &passes {
        if pipelines
            .iter()
            .any(|program| program.pipeline_id == pass.pipeline)
        {
            continue;
        }
        let program = programs
            .iter()
            .find(|program| program.pipeline_id == pass.pipeline)
            .ok_or("pass pipeline is not part of the compiled program table")?;
        pipelines.push(program.clone());
    }
    Ok(ComputeTrace {
        schema_version: PROVIDER_SCHEMA_VERSION,
        device_epoch,
        operation_id: OperationId::new(operation),
        pipelines,
        encoder_dispatch_type: DispatchType::Serial,
        passes: passes.into_iter().map(TracePass::Compute).collect(),
        completion_policy: CompletionPolicy::HostReadback,
        heap: case_heap_payload(case)?.map(Box::new),
        indirect: compute_icb_payload(case)?.map(Box::new),
    })
}

/// The heap identifier one capture's placements share.
///
/// The suite's heap section names a slab by its size, not by an identity (the
/// comparator only requires a non-zero id, `research/docs/25` §5.1), so the
/// capture tool picks a stable one; every placement in a case's payload uses
/// it, which is what makes the provider's `same_slab` observation meaningful.
const HEAP_PLACEMENT_ID: HeapId = HeapId::new(61);

/// Translate a compute case's indirect section into the trace payload
/// (`research/docs/25` §4.3). The first increment replays exactly one dispatch,
/// and the encoded threadgroups have to be the pass's own group count: the
/// provider checks the same equality, so a suite cannot replay a dispatch whose
/// footprint proofs were computed for a different shape.
fn compute_icb_payload(case: &Case) -> Result<Option<IndirectCommandPayload>> {
    let Some(icb) = &case.icb else {
        return Ok(None);
    };
    if icb.kind != "dispatch" {
        return Err(format!(
            "case {}: a compute case's indirect section replays a dispatch",
            case.id
        )
        .into());
    }
    if !icb.kinds.iter().any(|kind| kind == "dispatch") {
        return Err(format!(
            "case {}: the indirect buffer does not admit its own command kind",
            case.id
        )
        .into());
    }
    let dispatches = dispatch_sequence(case);
    if dispatches.len() != 1 {
        return Err(format!(
            "case {}: the first indirect increment replays exactly one dispatch",
            case.id
        )
        .into());
    }
    let (x, y, z) = (
        icb.command.x.ok_or_else(|| -> Box<dyn Error> {
            format!("case {}: the dispatch command needs x", case.id).into()
        })?,
        icb.command.y.ok_or_else(|| -> Box<dyn Error> {
            format!("case {}: the dispatch command needs y", case.id).into()
        })?,
        icb.command.z.ok_or_else(|| -> Box<dyn Error> {
            format!("case {}: the dispatch command needs z", case.id).into()
        })?,
    );
    if icb.command.vertex_count.is_some() || icb.command.instance_count.is_some() {
        return Err(format!(
            "case {}: a dispatch command carries no draw parameters",
            case.id
        )
        .into());
    }
    Ok(Some(IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: icb.max_commands,
            kinds: vec![IndirectCommandKind::Dispatch],
        },
        command: IndirectCommandDescriptor::Dispatch {
            threadgroups: [x, y, z],
        },
        range: IndirectCommandRange {
            start: icb.range.start,
            count: icb.range.count,
        },
    }))
}

/// Translate a render case's indirect section into the trace payload
/// (`research/docs/25` §4.3). The `draw` and `draw_indexed` kinds reach a
/// render case in the first increment; the command parameters are required by
/// kind, and a parameter of another kind is a typed refusal rather than a
/// silently ignored field.
/// The bytes one render input view carries.
///
/// The object rail creates its own device buffer per stream, exactly as the
/// trace rail does, so the view has to hold trace-owned bytes: a lease-backed
/// stream is a later increment's landing on both rails
/// (`research/docs/23` §3.3).
fn component_bytes<'a>(view: &'a BufferView, label: &str) -> Result<&'a [u8]> {
    match &view.source {
        BufferSource::OwnedBytes(bytes) => Ok(bytes),
        _ => Err(format!("{label}: the object rail executes trace-owned bytes only").into()),
    }
}

fn render_icb_payload(case: &RenderCase) -> Result<Option<IndirectCommandPayload>> {
    let Some(icb) = &case.icb else {
        return Ok(None);
    };
    let command = match icb.kind.as_str() {
        "draw" => {
            if !icb.kinds.iter().any(|kind| kind == "draw") {
                return Err(format!(
                    "render case {}: the indirect buffer does not admit its own command kind",
                    case.id
                )
                .into());
            }
            let vertex_count = icb.command.vertex_count.ok_or_else(|| -> Box<dyn Error> {
                format!(
                    "render case {}: the draw command needs vertex_count",
                    case.id
                )
                .into()
            })?;
            let instance_count = icb
                .command
                .instance_count
                .ok_or_else(|| -> Box<dyn Error> {
                    format!(
                        "render case {}: the draw command needs instance_count",
                        case.id
                    )
                    .into()
                })?;
            if icb.command.index_count.is_some()
                || icb.command.x.is_some()
                || icb.command.y.is_some()
                || icb.command.z.is_some()
            {
                return Err(format!(
                    "render case {}: a draw command carries no index or dispatch parameters",
                    case.id
                )
                .into());
            }
            IndirectCommandDescriptor::Draw {
                vertex_count,
                instance_count,
            }
        }
        "draw_indexed" => {
            if !icb.kinds.iter().any(|kind| kind == "draw_indexed") {
                return Err(format!(
                    "render case {}: the indirect buffer does not admit its own command kind",
                    case.id
                )
                .into());
            }
            let index_count = icb.command.index_count.ok_or_else(|| -> Box<dyn Error> {
                format!(
                    "render case {}: the draw_indexed command needs index_count",
                    case.id
                )
                .into()
            })?;
            let instance_count = icb
                .command
                .instance_count
                .ok_or_else(|| -> Box<dyn Error> {
                    format!(
                        "render case {}: the draw_indexed command needs instance_count",
                        case.id
                    )
                    .into()
                })?;
            if icb.command.vertex_count.is_some()
                || icb.command.x.is_some()
                || icb.command.y.is_some()
                || icb.command.z.is_some()
            {
                return Err(format!(
                    "render case {}: a draw_indexed command carries no vertex or dispatch parameters",
                    case.id
                )
                .into());
            }
            IndirectCommandDescriptor::DrawIndexed {
                index_count,
                instance_count,
            }
        }
        other => {
            return Err(format!(
                "render case {}: the first indirect increment replays draws only, not {other}",
                case.id
            )
            .into());
        }
    };
    Ok(Some(IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands: icb.max_commands,
            kinds: vec![command.kind()],
        },
        command,
        range: IndirectCommandRange {
            start: icb.range.start,
            count: icb.range.count,
        },
    }))
}

/// Turn the provider's replay record into the report segment. The record comes
/// from the execution path that ran, so a rail that fell back to a direct draw
/// cannot claim an indirect replay (`research/docs/25` §5.1).
fn icb_segment(observations: Vec<IcbReplayObservation>) -> Result<IcbSegment> {
    let observation = observations
        .first()
        .ok_or("indirect case reported no replay observation")?;
    let kind = match observation.kind {
        IndirectCommandKind::Draw => "draw",
        IndirectCommandKind::DrawIndexed => "draw_indexed",
        IndirectCommandKind::Dispatch => "dispatch",
    };
    Ok(IcbSegment {
        kind,
        start: observation.start,
        count: observation.count,
        commands: observation.commands,
    })
}

/// The one storage-mode name a heap section may spell (`research/docs/25`
/// §4.2). Kept in one place so the trace and object rails refuse the same
/// vocabulary.
fn heap_storage_mode(heap: &HeapCase, case_id: &str) -> Result<StorageMode> {
    match heap.storage_mode.as_str() {
        "owned_bytes" => Ok(StorageMode::OwnedBytes),
        "staged_lease" => Ok(StorageMode::StagedLease),
        "borrowed_no_copy" => Ok(StorageMode::BorrowedNoCopy),
        other => Err(format!("case {case_id}: unknown heap storage mode {other:?}").into()),
    }
}

/// Translate a suite case's heap section into the trace payload
/// (`research/docs/25` §4.2). The structural rules (one allocation per
/// placement, no overlap, placements fit the slab) stay core admission's job;
/// this only refuses the shapes the suite cannot spell at all.
fn case_heap_payload(case: &Case) -> Result<Option<HeapPayload>> {
    let Some(heap) = &case.heap else {
        return Ok(None);
    };
    // The provider maps placements to the trace's owned allocations in
    // ascending allocation order (`research/docs/25` §6 Step 3), so the suite
    // has to name them in that order and cover every owned allocation. Texture
    // placement is outside the first increment.
    if !case.textures.is_empty() {
        return Err(format!(
            "case {}: texture placement is outside the first heap increment",
            case.id
        )
        .into());
    }
    let mut owned: Vec<u64> = case
        .buffers
        .iter()
        .map(|buffer| buffer.allocation)
        .collect();
    owned.sort_unstable();
    owned.dedup();
    if owned.len() != heap.placements.len() {
        return Err(format!(
            "case {}: the heap section has {} placements for {} owned allocations",
            case.id,
            heap.placements.len(),
            owned.len()
        )
        .into());
    }
    for (placement, allocation) in heap.placements.iter().zip(&owned) {
        if placement.allocation != *allocation {
            return Err(format!(
                "case {}: heap placements must be in ascending allocation order",
                case.id
            )
            .into());
        }
    }
    let storage_mode = heap_storage_mode(heap, &case.id)?;
    let placements = heap
        .placements
        .iter()
        .map(|placement| HeapPlacement {
            heap_id: HEAP_PLACEMENT_ID,
            offset: placement.offset,
            resource: HeapResource::Buffer {
                byte_size: placement.byte_size,
            },
        })
        .collect();
    Ok(Some(HeapPayload {
        descriptor: HeapDescriptor {
            size: heap.size,
            storage_mode,
            allows_aliasing: heap.allows_aliasing,
        },
        placements,
    }))
}

/// Turn the provider's last submission's placement records into the report
/// segment (`research/docs/25` §5.1). The records come from the `vkBind*`
/// results the provider observed, not from the suite's request, so a provider
/// that did not place the resources cannot fake this segment.
fn heap_segment(
    observations: Vec<RawHeapPlacement>,
    remap: Option<&BTreeMap<AllocationId, u64>>,
) -> Result<HeapSegment> {
    let first = observations
        .first()
        .ok_or("heap case reported no placement observations")?;
    let same_slab = observations
        .iter()
        .all(|observation| observation.heap == first.heap);
    Ok(HeapSegment {
        heap: first.heap,
        same_slab,
        placements: observations
            .iter()
            .map(|observation| HeapPlacementReport {
                allocation: remap
                    .and_then(|map| map.get(&AllocationId::new(observation.allocation)))
                    .copied()
                    .unwrap_or(observation.allocation),
                offset: observation.offset,
                byte_size: observation.byte_size,
            })
            .collect(),
    })
}

/// The culling state one render case declares (`research/docs/23` §3.3, v39).
fn case_cull(case: &RenderCase) -> Result<Option<RenderPassCull>> {
    let Some(cull) = &case.cull else {
        return Ok(None);
    };
    let mode = match cull.mode.as_str() {
        "none" => CullMode::None,
        "front" => CullMode::Front,
        "back" => CullMode::Back,
        other => {
            return Err(format!("render case {}: unsupported cull mode {other:?}", case.id).into())
        }
    };
    let winding = match cull.winding.as_str() {
        "clockwise" => Winding::Clockwise,
        "counter_clockwise" => Winding::CounterClockwise,
        other => {
            return Err(format!("render case {}: unsupported winding {other:?}", case.id).into())
        }
    };
    Ok(Some(RenderPassCull { mode, winding }))
}

/// The blend state one render case declares (`research/docs/23` §3.3, v40).
fn case_blend(case: &RenderCase) -> Result<Option<RenderPassBlend>> {
    let Some(definitions) = &case.blend else {
        return Ok(None);
    };
    let mut attachments = Vec::with_capacity(definitions.len());
    for definition in definitions {
        let factor = |name: &str| -> Result<BlendFactor> {
            Ok(match name {
                "zero" => BlendFactor::Zero,
                "one" => BlendFactor::One,
                "source_alpha" => BlendFactor::SourceAlpha,
                "one_minus_source_alpha" => BlendFactor::OneMinusSourceAlpha,
                other => {
                    return Err(format!(
                        "render case {}: unsupported blend factor {other:?}",
                        case.id
                    )
                    .into())
                }
            })
        };
        attachments.push(BlendAttachment {
            source_rgb: factor(&definition.source_rgb)?,
            destination_rgb: factor(&definition.destination_rgb)?,
            source_alpha: factor(&definition.source_alpha)?,
            destination_alpha: factor(&definition.destination_alpha)?,
            operation: match definition.operation.as_str() {
                "add" => BlendOperation::Add,
                other => {
                    return Err(format!(
                        "render case {}: unsupported blend operation {other:?}",
                        case.id
                    )
                    .into())
                }
            },
        });
    }
    Ok(Some(RenderPassBlend { attachments }))
}

/// The depth attachment and depth state one render case declares
/// (`research/docs/23` §3.3, v36).
///
/// The case's spellings are the reviewed ones: `depth32float`, a `clear` with
/// its depth or a `load`, and a `less`/`always` compare with an explicit write
/// flag. A test without an attachment is refused here as well as by the
/// contract, so the refusal names the case rather than the pass index.
fn case_depth(
    case: &RenderCase,
    where_: &str,
) -> Result<(Option<RenderDepthAttachment>, Option<DepthTest>)> {
    let depth = match &case.depth {
        None => None,
        Some(depth) => {
            let format = match depth.format.as_str() {
                "depth32float" => DepthFormat::Depth32Float,
                other => return Err(format!("{where_}: unsupported depth format {other:?}").into()),
            };
            let load = match depth.load.as_str() {
                "clear" => {
                    let clear = depth.clear_depth.ok_or(format!(
                        "{where_}: a cleared depth attachment needs clear_depth"
                    ))?;
                    DepthLoadOp::clear(clear as f32)
                }
                "load" => {
                    if depth.clear_depth.is_some() {
                        return Err(format!(
                            "{where_}: a loading depth attachment carries no clear_depth"
                        )
                        .into());
                    }
                    DepthLoadOp::Load
                }
                other => {
                    return Err(format!("{where_}: unsupported depth load op {other:?}").into())
                }
            };
            Some(RenderDepthAttachment {
                format,
                width: depth.width,
                height: depth.height,
                load,
                // The v43 pair travels together (`research/docs/23` §3.3): a
                // pass that keeps its depth surface names where the texels land,
                // and the reviewed shape is validated before this point, so a
                // half-stated pair cannot reach the trace.
                store: match depth.store.as_deref() {
                    None => None,
                    Some("store") => Some(DepthStoreOp::Store),
                    Some(other) => {
                        return Err(
                            format!("{where_}: unsupported depth store action {other:?}").into(),
                        )
                    }
                },
                identity: match (depth.store.is_some(), depth.allocation, depth.view) {
                    (true, Some(allocation), Some(view)) => Some(RenderDepthIdentity {
                        allocation_id: AllocationId::new(allocation),
                        view_id: ViewId::new(view),
                    }),
                    (true, ..) => {
                        return Err(format!(
                            "{where_}: a stored depth attachment needs its allocation and view"
                        )
                        .into())
                    }
                    (false, None, None) => None,
                    (false, ..) => {
                        return Err(format!(
                            "{where_}: a discarded depth attachment carries no identity"
                        )
                        .into())
                    }
                },
            })
        }
    };
    let test = match &case.depth_test {
        None => None,
        Some(test) => Some(DepthTest {
            compare: match test.compare.as_str() {
                "less" => CompareFunction::Less,
                "always" => CompareFunction::Always,
                other => {
                    return Err(format!("{where_}: unsupported depth compare {other:?}").into())
                }
            },
            write: test.write,
        }),
    };
    if test.is_some() && depth.is_none() {
        return Err(format!("{where_}: a depth test needs a depth attachment").into());
    }
    Ok((depth, test))
}

/// The stencil attachment and stencil state one render case declares
/// (`research/docs/23` §3.3, v47).
///
/// The case's spellings mirror `compare.py` and the Swift oracle: `stencil8`
/// with a `clear` value, and the reviewed state's comparison, operations, masks
/// and reference. A state without an attachment is refused here as well as by
/// the contract, so the refusal names the case rather than the pass index.
fn case_stencil(
    case: &RenderCase,
    where_: &str,
) -> Result<(Option<RenderStencilAttachment>, Option<StencilTest>)> {
    let stencil = match &case.stencil {
        None => None,
        Some(stencil) => {
            let format = match stencil.format.as_str() {
                "stencil8" => StencilFormat::Stencil8,
                other => {
                    return Err(format!("{where_}: unsupported stencil format {other:?}").into())
                }
            };
            let load = match stencil.load.as_str() {
                "clear" => StencilLoadOp::clear(stencil.clear_value.ok_or(format!(
                    "{where_}: a cleared stencil attachment needs clear_value"
                ))?),
                "load" => {
                    if stencil.clear_value.is_some() {
                        return Err(format!(
                            "{where_}: a loading stencil attachment carries no clear_value"
                        )
                        .into());
                    }
                    StencilLoadOp::Load
                }
                other => {
                    return Err(format!("{where_}: unsupported stencil load op {other:?}").into())
                }
            };
            Some(RenderStencilAttachment {
                format,
                width: stencil.width,
                height: stencil.height,
                load,
                // The v49 pair travels together (`research/docs/23` §3.3): a
                // pass that keeps its stencil surface names where the texels
                // land, and the reviewed shape is validated before this point.
                store: match stencil.store.as_deref() {
                    None => None,
                    Some("store") => Some(StoreOp::Store),
                    Some("dontcare") => Some(StoreOp::DontCare),
                    Some(other) => {
                        return Err(
                            format!("{where_}: unsupported stencil store action {other:?}").into(),
                        )
                    }
                },
                identity: match (stencil.store.is_some(), stencil.allocation, stencil.view) {
                    (true, Some(allocation), Some(view)) => Some(RenderStencilIdentity {
                        allocation_id: AllocationId::new(allocation),
                        view_id: ViewId::new(view),
                    }),
                    (true, ..) => {
                        return Err(format!(
                            "{where_}: a stored stencil attachment needs its allocation and view"
                        )
                        .into())
                    }
                    (false, None, None) => None,
                    (false, ..) => {
                        return Err(format!(
                            "{where_}: a discarded stencil attachment carries no identity"
                        )
                        .into())
                    }
                },
            })
        }
    };
    let test = match &case.stencil_test {
        None => None,
        Some(test) => {
            let compare = match test.compare.as_str() {
                "equal" => StencilCompare::Equal,
                "always" => StencilCompare::Always,
                other => {
                    return Err(format!("{where_}: unsupported stencil compare {other:?}").into())
                }
            };
            let operation = |name: &str, spelling: &str| -> Result<StencilOp> {
                Ok(match spelling {
                    "keep" => StencilOp::Keep,
                    "replace" => StencilOp::Replace,
                    "increment_wrap" => StencilOp::IncrementWrap,
                    other => {
                        return Err(
                            format!("{where_}: unsupported stencil {name} op {other:?}").into()
                        )
                    }
                })
            };
            Some(StencilTest {
                compare,
                fail_op: operation("fail", &test.fail_op)?,
                depth_fail_op: operation("depth fail", &test.depth_fail_op)?,
                pass_op: operation("pass", &test.pass_op)?,
                read_mask: test.read_mask,
                write_mask: test.write_mask,
                reference: test.reference,
            })
        }
    };
    if test.is_some() && stencil.is_none() {
        return Err(format!("{where_}: a stencil test needs a stencil attachment").into());
    }
    Ok((stencil, test))
}

/// Execute one render case on the Vulkan trace rail.
///
/// The trace is the declaring case's own pass followed by the render pass, i.e.
/// the shape `research/docs/23` §3.6 admits: the attachment references a view
/// the trace declares, the declaring pass only *reads* it, and the render rail
/// executes after the compute sequence. The attachment's bytes leave through
/// the existing writeback channel — the render rail pushes one
/// `BufferWriteback` for the view the trace declares — so this reports the
/// attachment's allocation image and that one writeback and nothing else. The
/// declaring pass's own landing belongs to the declaring case, which is what
/// keeps an attachment observation from being confusable with a buffer
/// writeback.
// The render runner's own signature is the case's whole input set: the
// declaring pass's programs, the render registration, the pass's identity and
// the lease channels a stage buffer may source its bytes from. Splitting it into
// a context struct would put the same fields behind one more name.
#[allow(clippy::too_many_arguments)]
fn run_render_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    declaring: &Case,
    case: &RenderCase,
    render_pipeline: &CompiledComputePipeline,
    operation: u64,
    guard: u8,
    leases: &LeaseHandles,
) -> Result<CaseResult> {
    let attachments = render_attachment_shapes(case)?;
    let (depth_attachment, depth_test) = case_depth(case, &format!("render case {}", case.id))?;
    // The stencil sibling (`research/docs/23` §3.3, v47): the same pass-level
    // state the depth pair carries, one byte wide, plus the rail-owned surface
    // the reviewed state masks with.
    let (stencil_attachment, stencil_test) =
        case_stencil(case, &format!("render case {}", case.id))?;
    let cull = case_cull(case)?;
    let blend = case_blend(case)?;

    // The declaring pass's own resource table: one backing image and one
    // `AllocationRecord` per allocation (`docs/23` §4.1).
    let mut allocations: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut recorded = BTreeSet::new();
    let mut resources = ResourceTableSnapshot::new();
    for buffer in &declaring.buffers {
        let initial = buffer.initial_bytes()?;
        let start = usize::try_from(buffer.offset)?;
        let position = match allocations
            .iter()
            .position(|(allocation, _)| *allocation == buffer.allocation)
        {
            Some(position) => position,
            None => {
                allocations.push((
                    buffer.allocation,
                    vec![guard; usize::try_from(buffer.allocation_size)?],
                ));
                allocations.len() - 1
            }
        };
        if recorded.insert(buffer.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(buffer.allocation),
                owner_epoch: provider.device_epoch(),
                size: buffer.allocation_size,
            })?;
        }
        allocations[position].1[start..start + initial.len()].copy_from_slice(&initial);
    }
    let views = declaring
        .buffers
        .iter()
        .map(|buffer| {
            let access = match buffer.access.as_str() {
                "read" => BufferAccess::Read,
                "write" => BufferAccess::Write,
                "read_write" => BufferAccess::ReadWrite,
                _ => return Err("unsupported access".into()),
            };
            let (_, backing) = allocations
                .iter()
                .find(|(allocation, _)| *allocation == buffer.allocation)
                .ok_or("unknown fixture allocation")?;
            let start = usize::try_from(buffer.offset)?;
            let end = start + usize::try_from(buffer.length)?;
            Ok(BufferView {
                view_id: ViewId::new(buffer.view),
                metal_binding: buffer.binding,
                allocation_id: AllocationId::new(buffer.allocation),
                offset: buffer.offset,
                length: buffer.length,
                access,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(backing[start..end].to_vec()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // The pass's own stage buffers (`research/docs/23` §3.3, v83-v86): one
    // view per declared slot, resolved through the same three-arm source
    // channel a compute case's views use (R9i). The lease imports have to be
    // made before the trace is built and admitted, because the snapshot the
    // admission reads has to carry the same reservations the trace's
    // `BufferSource` names; the owner windows a borrowed view maps are kept
    // alive in `owner_windows` until this case's submission has been waited
    // for.
    let mut case_leases = CaseLeases::default();
    let mut owner_windows = Vec::new();
    // A lease-armed stage buffer's allocation is the *owner's* registration, so
    // the trace's own table has to carry it before the reservation can be
    // admitted: `ResourceTableSnapshot::insert_lease` resolves the lease's
    // allocation, and a snapshot without it refuses the reservation by name
    // (`research/docs/23` §90, R9i). An owned stage buffer declares no owner
    // window and adds no record.
    for definition in &case.stage_buffers {
        if stage_buffer_storage_mode(definition, &format!("render case {}", case.id))?
            == BufferSourceKind::OwnedBytes
        {
            continue;
        }
        let allocation = AllocationId::new(definition.allocation);
        if resources.allocation(allocation).is_none() {
            resources.insert_allocation(AllocationRecord {
                allocation_id: allocation,
                owner_epoch: provider.device_epoch(),
                size: definition
                    .allocation_size
                    .ok_or("a leased stage buffer states the owner window's length")?,
            })?;
        }
    }
    let stage_buffers = stage_buffer_views(
        case,
        leases,
        provider,
        guard,
        &mut case_leases,
        &mut owner_windows,
    )?;
    for reservation in &case_leases.reservations {
        resources.insert_lease(*reservation)?;
    }
    // One declared view per attachment, in location order: every attachment
    // resolves against that view exactly as `validate_render_case` pinned.
    let declared_views = attachments
        .iter()
        .map(|(attachment, _)| {
            views
                .iter()
                .find(|view| {
                    view.view_id == ViewId::new(attachment.view)
                        && view.allocation_id == AllocationId::new(attachment.allocation)
                })
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "the declaring pass does not declare the attachment view {}",
                        attachment.view
                    )
                    .into()
                })
        })
        .collect::<Result<Vec<_>>>()?;

    // The stored depth attachment's own landing view (`research/docs/23` §3.3,
    // v43): its identity is resolved against the declaring pass's table exactly
    // as a colour attachment's is, because the depth texels leave through the
    // same byte-keyed writeback channel. The declaring pass *reads* the depth
    // view — the reviewed kernel carries a third read binding for it — so the
    // view is in the trace's pool and its offset and length are the ones the
    // writeback has to cover.
    // The stored stencil attachment's own landing view (`research/docs/23`
    // §3.3, v49): its identity resolves against the declaring pass's table
    // exactly as the depth landing's does, because the stencil texels leave
    // through the same byte-keyed writeback channel.
    let stencil_view = match case
        .stencil
        .as_ref()
        .and_then(|stencil| stencil.store.as_deref())
    {
        None => None,
        Some("store") => {
            let definition = case
                .stencil
                .as_ref()
                .ok_or("a stored stencil attachment needs its definition")?;
            let allocation = definition
                .allocation
                .ok_or("a stored stencil attachment needs its allocation")?;
            let view = definition
                .view
                .ok_or("a stored stencil attachment needs its view")?;
            Some(
                views
                    .iter()
                    .find(|candidate| {
                        candidate.view_id == ViewId::new(view)
                            && candidate.allocation_id == AllocationId::new(allocation)
                    })
                    .cloned()
                    .ok_or_else(|| -> Box<dyn Error> {
                        format!(
                            "the declaring pass does not declare the stencil view {view} of \
                             render case {}",
                            case.id
                        )
                        .into()
                    })?,
            )
        }
        Some(other) => {
            return Err(format!(
                "render case {}: unsupported stencil store action {other:?}",
                case.id
            )
            .into())
        }
    };

    let depth_view = match case.depth.as_ref().and_then(|depth| depth.store.as_deref()) {
        None => None,
        Some("store") => {
            let definition = case
                .depth
                .as_ref()
                .ok_or("a stored depth attachment needs its definition")?;
            let allocation = definition
                .allocation
                .ok_or("a stored depth attachment needs its allocation")?;
            let view = definition
                .view
                .ok_or("a stored depth attachment needs its view")?;
            Some(
                views
                    .iter()
                    .find(|candidate| {
                        candidate.view_id == ViewId::new(view)
                            && candidate.allocation_id == AllocationId::new(allocation)
                    })
                    .cloned()
                    .ok_or_else(|| -> Box<dyn Error> {
                        format!(
                            "the declaring pass does not declare the depth view {view} of render \
                             case {}",
                            case.id
                        )
                        .into()
                    })?,
            )
        }
        Some(other) => {
            return Err(format!(
                "render case {}: unsupported depth store action {other:?}",
                case.id
            )
            .into())
        }
    };

    let mut trace = case_trace(
        provider.device_epoch(),
        programs,
        declaring,
        operation,
        &views,
        &[],
        &dispatch_sequence(declaring),
    )?;
    // The indirect section replays the render pass's own triangle from one
    // encoded command (`research/docs/25` §6 Step 4); the attachment shape and
    // every other validation stay the ones the direct case uses. The declaring
    // case's own `icb` section belongs to its compute submission, not to this
    // trace, so it is replaced here rather than inherited.
    trace.indirect = render_icb_payload(case)?.map(Box::new);
    // The pass's own vertex and index streams (`research/docs/23` §3.3). They
    // carry their bytes, so the declaring case needs no extra binding and the
    // reviewed-shape validation already pinned what the draw reads.
    let (vertex_buffers, indices) = render_inputs(case, &format!("render case {}", case.id))?;
    // A clearing pass carries its colour; a loading pass carries nothing and
    // uploads the declaring view's own bytes (`research/docs/23` §3.3), so
    // each attachment's load operation is the fixture's own choice.
    let color_attachments = attachments
        .iter()
        .map(|(attachment, _)| {
            let load = match attachment.load.as_str() {
                "clear" => LoadOp::Clear(ClearColor::new(
                    unhex(
                        attachment
                            .clear_hex
                            .as_deref()
                            .ok_or("a clear attachment needs clear_hex")?,
                    )?
                    .try_into()
                    .map_err(|_| -> Box<dyn Error> { "a clear colour is four bytes".into() })?,
                )),
                "load" => LoadOp::Load,
                "dontcare" => LoadOp::DontCare,
                other => {
                    return Err(
                        format!("render case {}: unknown load op {other:?}", case.id).into(),
                    );
                }
            };
            // The fixture's store operation travels with the attachment: a
            // `dontcare` entry renders but its writes are discarded by the
            // provider (`docs/23` §3.6, v19).
            let store = match attachment.store.as_str() {
                "store" => StoreOp::Store,
                "dontcare" => StoreOp::DontCare,
                other => {
                    return Err(
                        format!("render case {}: unsupported store op {other:?}", case.id).into(),
                    );
                }
            };
            Ok(RenderAttachment {
                view_id: ViewId::new(attachment.view),
                allocation_id: AllocationId::new(attachment.allocation),
                format: attachment_format(&attachment.format)?,
                width: attachment.width,
                height: attachment.height,
                load,
                store,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let present = match &case.present {
        Some(definition) => Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(attachments[0].0.allocation),
                view_id: ViewId::new(attachments[0].0.view),
                format: attachment_format(&attachments[0].0.format)?,
                width: attachments[0].0.width,
                height: attachments[0].0.height,
                image_count: definition.image_count,
                initial: match &definition.initial_hex {
                    Some(hex) => InitialState::Sentinel(unhex(hex)?),
                    None => InitialState::Undefined,
                },
            },
            source: ViewId::new(attachments[0].0.view),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        }),
        None => None,
    };
    trace.pipelines.push(render_pipeline.clone());
    let scissor = match case.scissor {
        Some([x, y, width, height]) => Some([
            u32::try_from(x)?,
            u32::try_from(y)?,
            u32::try_from(width)?,
            u32::try_from(height)?,
        ]),
        None => None,
    };
    // The pass-wide multisample state (`research/docs/23` §3.3, v51): the
    // reviewed fixture's four-sample raster. `validate_render_case` refused
    // every other count before this point, so the mapping is total over the
    // shapes that can reach it.
    let multisample = case_multisample(case)?;
    // The sampled textures the reviewed sampling pair reads
    // (`research/docs/23` §3.3, v70): each entry's position is its binding, and
    // its bytes travel with the trace exactly as a compute binding's do. Every
    // other case binds none, which is the shape every pre-v70 pass carries.
    let textures = case
        .fragment_textures
        .as_ref()
        .map(|definitions| {
            definitions
                .iter()
                .enumerate()
                .map(|(binding, definition)| {
                    Ok(TextureView {
                        view_id: ViewId::new(definition.view),
                        metal_binding: u32::try_from(binding)?,
                        allocation_id: AllocationId::new(definition.allocation),
                        texture_type: TextureType::D2,
                        format: TextureFormat::Rgba8Unorm,
                        width: definition.width,
                        height: definition.height,
                        depth: 1,
                        array_length: 1,
                        sample_count: 1,
                        access: TextureAccess::Sampled,
                        source: TextureSource::OwnedBytes(definition.texels()?),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    trace.passes.push(TracePass::Render(RenderPassDescriptor {
        stage_buffers,
        pipeline: render_pipeline.pipeline_id,
        color_attachments,
        viewport: [
            u32::try_from(case.viewport[0])?,
            u32::try_from(case.viewport[1])?,
            u32::try_from(case.viewport[2])?,
            u32::try_from(case.viewport[3])?,
        ],
        scissor,
        multisample,
        // The reviewed depth-resolve case states its filter; every other case
        // leaves the field absent, which the rails execute as "resolve
        // nothing" (`research/docs/23` §3.3, v57).
        depth_resolve: case_depth_resolve(case)?,
        // The reviewed stencil-resolve case states its filter; every other
        // case leaves the field absent, which the rails execute as "resolve
        // nothing" (`research/docs/23` §3.3, v60).
        stencil_resolve: case_stencil_resolve(case)?,
        vertices: u32::try_from(case.vertices)?,
        vertex_buffers,
        indices,
        // The reviewed instanced case carries two; every pre-v31 case leaves
        // the field at its single-instance default
        // (`research/docs/23` §3.3, v31).
        instance_count: u32::try_from(case.instance_count)?,
        // The reviewed base-vertex fixture declares one; every pre-v34 case
        // leaves the offset at zero (`research/docs/23` §3.3, v34).
        base_vertex: u32::try_from(case.base_vertex)?,
        // The reviewed depth fixture declares both; every pre-v36 case leaves
        // them absent, which is the shape the rails execute as "no depth
        // surface" (`research/docs/23` §3.3, v36).
        depth: depth_attachment,
        depth_test,
        // The reviewed stencil fixture declares the surface and the state;
        // every other case leaves both absent, which the rails execute as "no
        // stencil surface at all" (`research/docs/23` §3.3, v47).
        stencil: stencil_attachment,
        stencil_test,
        // The reviewed cull fixture declares the state; every other case leaves
        // it absent, which the rails execute as "keep every triangle"
        // (`research/docs/23` §3.3, v39).
        cull,
        blend,
        // The reviewed render-sampler case binds its fragment texture here;
        // every other case leaves the list empty, which the rails execute as
        // "the fragment stage samples nothing" (`research/docs/23` §3.3, v70).
        textures,
        present,
    }));

    let admitted = provider
        .capabilities()
        .validate_trace(trace.clone(), resources.clone())
        .map_err(|error| format!("admit {}: {error:?}", case.id))?;
    let output = provider
        .submit(admitted)
        .map_err(|error| format!("submit {}: {error:?}", case.id))?;
    output.validate_for_trace(&trace)?;
    let CompletionDisposition::CompletedVisible { token } = output.completion else {
        return Err("render capture requires completed visible results".into());
    };
    if provider
        .wait(token, Duration::ZERO)
        .map_err(|error| format!("wait: {error:?}"))?
        != output.completion
    {
        return Err("provider completion observation changed".into());
    }
    let mut landed = BTreeMap::new();
    // The views whose writebacks this case observes: the attachments' own
    // declaring views, the stored depth and stencil landings, and the writable
    // stage buffers' pool views (`research/docs/23` §3.3, v86). Every other
    // writeback belongs to the declaring pass and is not this case's landing.
    let writable_stage_buffers = case
        .stage_buffers
        .iter()
        .filter(|definition| {
            stage_buffer_access(&definition.access, &format!("render case {}", case.id))
                .is_ok_and(|access| access.is_writable())
        })
        .map(|definition| {
            (
                AllocationId::new(definition.allocation),
                ViewId::new(definition.view),
            )
        })
        .collect::<Vec<_>>();
    for write in output.writebacks {
        let (_, backing) = allocations
            .iter_mut()
            .find(|(id, _)| *id == write.allocation_id.get())
            .ok_or("unknown writeback allocation")?;
        let start = usize::try_from(write.offset)?;
        backing[start..start + write.bytes.len()].copy_from_slice(&write.bytes);
        if declared_views
            .iter()
            .chain(depth_view.iter())
            .chain(stencil_view.iter())
            .any(|view| view.view_id == write.view_id && view.allocation_id == write.allocation_id)
            || writable_stage_buffers.contains(&(write.allocation_id, write.view_id))
        {
            landed.insert((write.allocation_id, write.view_id), write);
        }
    }
    provider
        .release_completion(token)
        .map_err(|error| format!("release completion: {error:?}"))?;
    // The lease imports belong to this case alone: a later case may reuse a view
    // id, and the registry refuses a duplicate import until the previous one is
    // released. This case's single submission has been waited for and its
    // completion released, which is the point the no-copy registry's own
    // retirement rule waits on (`research/docs/23` §90, R9i).
    for lease_id in &case_leases.staged {
        leases
            .staged
            .release_staged_lease(*lease_id)
            .map_err(|error| format!("render case {}: release staged lease: {error:?}", case.id))?;
    }
    for lease_id in &case_leases.borrowed {
        leases
            .borrowed
            .release_borrowed_lease(*lease_id)
            .map_err(|error| {
                format!("render case {}: release borrowed lease: {error:?}", case.id)
            })?;
    }
    // One writeback and one allocation image per attachment, in location
    // order. Each writeback has to cover the exact range its declaring view
    // states: a rail that landed a different range cannot be reported as this
    // case's texels. A discarded attachment's bytes disappear with the pass,
    // so no writeback and no allocation image are owed for it
    // (`docs/23` §3.6, v19).
    let mut writebacks = Vec::new();
    let mut images = Vec::new();
    for (attachment, _) in &attachments {
        if attachment.store != "store" {
            continue;
        }
        let declared = declared_views
            .iter()
            .find(|view| {
                view.view_id == ViewId::new(attachment.view)
                    && view.allocation_id == AllocationId::new(attachment.allocation)
            })
            .ok_or("the render rail landed no attachment writeback")?;
        let write = landed
            .get(&(
                AllocationId::new(attachment.allocation),
                ViewId::new(attachment.view),
            ))
            .ok_or("the render rail landed no attachment writeback")?;
        if write.offset != declared.offset || write.bytes.len() as u64 != declared.length {
            return Err(format!(
                "render case {}: the attachment writeback covers {}..{} instead of {}..{}",
                case.id,
                write.offset,
                write.offset + write.bytes.len() as u64,
                declared.offset,
                declared.offset + declared.length
            )
            .into());
        }
        let image = allocations
            .iter()
            .find(|(id, _)| *id == attachment.allocation)
            .ok_or("the attachment allocation is missing")?
            .1
            .clone();
        let (writeback, observation) = attachment_observation(
            case.expected_rule.as_deref(),
            case.readback_windows.as_deref().unwrap_or_default(),
            (
                attachment.allocation,
                attachment.view,
                write.offset,
                attachment.width,
            ),
            &write.bytes,
            &image,
        )?;
        writebacks.push(writeback);
        images.push(observation);
    }
    // The stored depth attachment's own landing, after the colour ones
    // (`research/docs/23` §3.3, v43): one writeback covering the exact view the
    // declaring pass declared, and one allocation image holding that view's
    // bytes. A rail that does not read the depth surface back lands no
    // writeback for it, so this is where "the channel exists" is observed
    // rather than assumed.
    if let Some(declared) = &depth_view {
        let definition = case
            .depth
            .as_ref()
            .ok_or("a stored depth attachment needs its definition")?;
        let allocation = definition
            .allocation
            .ok_or("a stored depth attachment needs its allocation")?;
        let view = definition
            .view
            .ok_or("a stored depth attachment needs its view")?;
        let write = landed
            .get(&(AllocationId::new(allocation), ViewId::new(view)))
            .ok_or("the render rail landed no depth writeback")?;
        if write.offset != declared.offset || write.bytes.len() as u64 != declared.length {
            return Err(format!(
                "render case {}: the depth writeback covers {}..{} instead of {}..{}",
                case.id,
                write.offset,
                write.offset + write.bytes.len() as u64,
                declared.offset,
                declared.offset + declared.length
            )
            .into());
        }
        let expected = unhex(
            definition
                .expected_hex
                .as_deref()
                .ok_or("a stored depth attachment needs expected_hex")?,
        )?;
        if write.bytes != expected {
            return Err(format!(
                "render case {}: the depth readback is {} against the reviewed {}",
                case.id,
                hex(&write.bytes),
                hex(&expected)
            )
            .into());
        }
        let image = allocations
            .iter()
            .find(|(id, _)| *id == allocation)
            .ok_or("the depth allocation is missing")?
            .1
            .clone();
        writebacks.push(Writeback::encoded(
            allocation,
            view,
            write.offset,
            &write.bytes,
        ));
        images.push(Allocation::observed(allocation, &image));
    }
    // The stored stencil attachment's own landing, after the depth one
    // (`research/docs/23` §3.3, v49): one writeback covering the exact view the
    // declaring pass declared, and one allocation image holding that view's
    // bytes.
    if let Some(declared) = &stencil_view {
        let definition = case
            .stencil
            .as_ref()
            .ok_or("a stored stencil attachment needs its definition")?;
        let allocation = definition
            .allocation
            .ok_or("a stored stencil attachment needs its allocation")?;
        let view = definition
            .view
            .ok_or("a stored stencil attachment needs its view")?;
        let write = landed
            .get(&(AllocationId::new(allocation), ViewId::new(view)))
            .ok_or("the render rail landed no stencil writeback")?;
        if write.offset != declared.offset || write.bytes.len() as u64 != declared.length {
            return Err(format!(
                "render case {}: the stencil writeback covers {}..{} instead of {}..{}",
                case.id,
                write.offset,
                write.offset + write.bytes.len() as u64,
                declared.offset,
                declared.offset + declared.length
            )
            .into());
        }
        let expected = unhex(
            definition
                .expected_hex
                .as_deref()
                .ok_or("a stored stencil attachment needs expected_hex")?,
        )?;
        if write.bytes != expected {
            return Err(format!(
                "render case {}: the stencil readback is {} against the reviewed {}",
                case.id,
                hex(&write.bytes),
                hex(&expected)
            )
            .into());
        }
        let image = allocations
            .iter()
            .find(|(id, _)| *id == allocation)
            .ok_or("the stencil allocation is missing")?
            .1
            .clone();
        writebacks.push(Writeback::encoded(
            allocation,
            view,
            write.offset,
            &write.bytes,
        ));
        images.push(Allocation::observed(allocation, &image));
    }
    // The writable stage buffers' own landings (`research/docs/23` §3.3, v86),
    // after the colour, depth and stencil ones: one writeback covering the
    // exact view the declaring pass declared, and one allocation image holding
    // the bytes the writeback left there. The landing travels through the same
    // byte-keyed channel the attachments use, so a rail that executed the
    // write but reported nothing cannot pass this case.
    let case_where = format!("render case {}", case.id);
    for definition in &case.stage_buffers {
        let access = stage_buffer_access(&definition.access, &case_where)?;
        if !access.is_writable() {
            continue;
        }
        let declared = views
            .iter()
            .find(|view| {
                view.view_id == ViewId::new(definition.view)
                    && view.allocation_id == AllocationId::new(definition.allocation)
            })
            .ok_or("the writable stage buffer's view is not in the trace's pool")?;
        let write = landed
            .get(&(
                AllocationId::new(definition.allocation),
                ViewId::new(definition.view),
            ))
            .ok_or_else(|| -> Box<dyn Error> {
                format!(
                    "render case {}: the pass landed no writeback for the writable stage buffer \
                     {}/{}",
                    case.id, definition.stage, definition.index
                )
                .into()
            })?;
        if write.offset != declared.offset || write.bytes.len() as u64 != declared.length {
            return Err(format!(
                "render case {}: the writable stage buffer {}/{} landed {}..{} instead of \
                 {}..{}",
                case.id,
                definition.stage,
                definition.index,
                write.offset,
                write.offset + write.bytes.len() as u64,
                declared.offset,
                declared.offset + declared.length
            )
            .into());
        }
        let expected = unhex(
            definition
                .expected_hex
                .as_deref()
                .ok_or("a writable stage buffer states the bytes it lands")?,
        )?;
        if write.bytes != expected {
            return Err(format!(
                "render case {}: the writable stage buffer {}/{} landed {} against the reviewed \
                 {}",
                case.id,
                definition.stage,
                definition.index,
                hex(&write.bytes),
                hex(&expected)
            )
            .into());
        }
        let image = allocations
            .iter()
            .find(|(id, _)| *id == definition.allocation)
            .ok_or("the writable stage buffer's allocation is missing")?
            .1
            .clone();
        writebacks.push(Writeback::encoded(
            definition.allocation,
            definition.view,
            write.offset,
            &write.bytes,
        ));
        images.push(Allocation::observed(definition.allocation, &image));
    }
    eprintln!(
        "render case completed: {} attachments={} bytes={}",
        case.id,
        attachments.len(),
        writebacks
            .iter()
            .map(Writeback::observed_bytes)
            .sum::<usize>()
    );
    let storage_modes = stage_buffer_storage_modes(case)?;
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks,
        allocations: images,
        copy_in: None,
        copy_out: None,
        group_counts: None,
        present: None,
        heap: None,
        icb: None,
        storage_modes,
    })
}

fn run_object_case(
    device: &objects::Device,
    programs: &[objects::Pipeline],
    case: &Case,
    guard: u8,
    async_execution: bool,
    counters: &mut dyn FnMut() -> (usize, usize),
) -> Result<(CaseResult, BTreeMap<AllocationId, u64>)> {
    // Fixture IDs are report labels only. The object API creates and validates
    // its own allocation/view identities before they are mapped back here.
    let mut resources = BTreeMap::new();
    let mut report_ids = BTreeMap::new();
    // One device Buffer per allocation. A v10 fixture binds several disjoint
    // views of the same allocation, and sharing the object is what makes the
    // provider exercise ranged aliasing instead of seeing unrelated
    // allocations. Guard bytes outside every view stay in the shared image so
    // an offset or extent mistake stays observable.
    let mut images = BTreeMap::<u64, Vec<u8>>::new();
    for definition in &case.buffers {
        let size = usize::try_from(definition.allocation_size)?;
        let offset = usize::try_from(definition.offset)?;
        let bytes = definition.initial_bytes()?;
        let image = images
            .entry(definition.allocation)
            .or_insert_with(|| vec![guard; size]);
        image[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
    let mut allocation_buffers = BTreeMap::<u64, objects::Buffer>::new();
    for definition in &case.buffers {
        let buffer = match allocation_buffers.get(&definition.allocation) {
            Some(existing) => existing.clone(),
            None => {
                let image = images
                    .remove(&definition.allocation)
                    .ok_or("missing allocation image")?;
                let created = device.new_buffer_with_bytes(image)?;
                allocation_buffers.insert(definition.allocation, created.clone());
                created
            }
        };
        let view = buffer.view(
            usize::try_from(definition.offset)?,
            usize::try_from(definition.length)?,
        )?;
        report_ids.insert(
            (view.allocation_id(), view.view_id()),
            (definition.allocation, definition.view),
        );
        resources.insert(definition.view, (buffer, view));
    }
    let queue = device.new_command_queue();
    let dispatches = dispatch_sequence(case);
    let groups = case_command_buffers(case);
    let narrow = |dimensions: [u64; 3]| -> Result<Size> {
        Ok(Size::new(
            u32::try_from(dimensions[0])?,
            u32::try_from(dimensions[1])?,
            u32::try_from(dimensions[2])?,
        )?)
    };
    // v11: sampled textures become object-API Texture handles, bound by the
    // same Metal argument index as the AIR fixture declares.
    let mut object_textures = BTreeMap::new();
    for texture in &case.textures {
        let initial = unhex(&texture.initial_hex)?;
        let created = device.new_texture_with_bytes(
            TextureFormat::R32Uint,
            texture.width,
            texture.height,
            initial,
        )?;
        object_textures.insert(texture.binding, created);
    }
    // A heap section names one slab and covers every owned allocation. The
    // object rail places each whole allocation at its declared offset and hands
    // the heap to the command, so commit publishes the placement payload
    // exactly as the trace rail's `case_heap_payload` does.
    let heap = match &case.heap {
        Some(definition) => {
            let heap = device.new_heap(
                definition.size,
                heap_storage_mode(definition, &case.id)?,
                definition.allows_aliasing,
            )?;
            for placement in &definition.placements {
                let buffer = allocation_buffers
                    .get(&placement.allocation)
                    .ok_or("heap placement names an allocation the case does not own")?;
                heap.place(buffer, placement.offset)?;
            }
            Some(heap)
        }
        None => None,
    };
    // A compute case's indirect section replays its one dispatch from an
    // encoded command. The payload is built by the same translator the trace
    // rail uses, so the two rails carry byte-identical requests.
    let icb = match &case.icb {
        Some(_) => {
            let payload = compute_icb_payload(case)?.ok_or("missing compute indirect payload")?;
            Some(device.new_indirect_command_buffer(
                payload.command.kind(),
                payload.buffer.max_commands,
                payload.buffer.kinds,
                payload.range,
                payload.command,
            )?)
        }
        None => None,
    };
    let mut reported = Vec::new();
    let mut group_counts = Vec::with_capacity(groups.len());
    // Each command buffer commits and completes before the next one records,
    // which matches Metal's serial queue boundary and re-snapshots the landed
    // bytes for the following command.
    for group in &groups {
        let before = counters();
        let command = queue.command_buffer();
        // Several dispatches on one encoder exercise snapshot-at-dispatch behavior,
        // including changed pipelines, binding tables and later first use.
        let mut encoder = command.compute_command_encoder()?;
        for index in group {
            let dispatch = dispatches
                .get(*index)
                .ok_or("command buffer dispatch index out of range")?;
            encoder.clear_buffers()?;
            encoder.clear_textures()?;
            encoder.set_compute_pipeline_state(&programs[dispatch.program.unwrap_or(0)])?;
            for (binding, texture) in &object_textures {
                encoder.set_texture(*binding, texture)?;
            }
            let slots = selected_slots(case, dispatch);
            let views = dispatch
                .bindings
                .clone()
                .unwrap_or_else(|| case.buffers.iter().map(|buffer| buffer.view).collect());
            for (slot, view) in slots.iter().zip(views) {
                let (_, view) = resources.get(&view).ok_or("unknown object fixture view")?;
                encoder.set_buffer(slot.binding, view)?;
            }
            if let Some(icb) = &icb {
                encoder.dispatch_indirect(icb, narrow(dispatch.grid)?, narrow(dispatch.local)?)?;
            } else {
                encoder.dispatch_threads(narrow(dispatch.grid)?, narrow(dispatch.local)?)?;
            }
        }
        encoder.end_encoding()?;
        if let Some(heap) = &heap {
            command.set_heap(heap)?;
        }
        command.commit()?;
        if async_execution {
            if command.status()? != metal_api_core::CommandBufferStatus::Committed {
                return Err("async object commit did not leave the command pending".into());
            }
            if !matches!(
                command.submission()?.completion,
                CompletionDisposition::Submitted { .. }
            ) {
                return Err("async object commit did not return a submitted token".into());
            }
        }
        command.wait_until_completed()?;
        if command.status()? != metal_api_core::CommandBufferStatus::Completed {
            return Err("object command did not reach Completed".into());
        }
        let output = command.submission()?;
        if !matches!(
            output.completion,
            CompletionDisposition::CompletedVisible { .. }
        ) {
            return Err("object capture requires completed visible results".into());
        }
        for write in output.writebacks {
            let &(allocation, view) = report_ids
                .get(&(write.allocation_id, write.view_id))
                .ok_or("unknown object writeback identity")?;
            reported.push((allocation, view, write.offset, write.bytes));
        }
        let after = counters();
        group_counts.push(GroupCounts {
            copy_in: u32::try_from(after.0 - before.0)?,
            copy_out: u32::try_from(after.1 - before.1)?,
        });
    }
    let writebacks = merge_writebacks(case, reported)?;
    let mut allocations = Vec::new();
    for (allocation, buffer) in &allocation_buffers {
        // Observe the object's actual host landing, rather than replaying the
        // returned writebacks into a second synthetic allocation. Exactly one
        // entry per allocation: several views may share it.
        allocations.push(Allocation::observed(*allocation, &buffer.read()?));
    }
    allocations.sort_by_key(|allocation| allocation.allocation);
    eprintln!(
        "objects command completed: {} command_buffers={} passes={}",
        case.id,
        groups.len(),
        dispatches.len()
    );
    let allocation_ids = report_ids
        .iter()
        .map(|((allocation, _), (fixture_allocation, _))| (*allocation, *fixture_allocation))
        .collect::<BTreeMap<_, _>>();
    Ok((
        CaseResult {
            id: case.id.clone(),
            completion: "CompletedVisible",
            writebacks,
            allocations,
            copy_in: None,
            copy_out: None,
            group_counts: case.command_buffers.as_ref().map(|_| group_counts),
            present: None,
            heap: None,
            icb: None,
            // The object API carries no lease channel yet
            // (`research/docs/23` §90): a case that declares one is marked for
            // the trace rails and never reaches this path.
            storage_modes: None,
        },
        allocation_ids,
    ))
}

/// Run one render case on the object API: the declaring compute pass's own
/// dispatch and the render pass ride in the same command buffer, so the
/// attachment view the compute pass declares is the one the render pass stores
/// into (`research/docs/23` §3.6). The observation is the attachment's own
/// allocation and writeback, exactly as the trace rail reports it.
fn run_object_render_case(
    device: &objects::Device,
    programs: &[objects::Pipeline],
    declaring: &Case,
    case: &RenderCase,
    render_pipeline: &objects::RenderPipeline,
    guard: u8,
    async_execution: bool,
) -> Result<CaseResult> {
    let attachments = render_attachment_shapes(case)?;
    let mut images = BTreeMap::<u64, Vec<u8>>::new();
    for definition in &declaring.buffers {
        let size = usize::try_from(definition.allocation_size)?;
        let offset = usize::try_from(definition.offset)?;
        let bytes = definition.initial_bytes()?;
        let image = images
            .entry(definition.allocation)
            .or_insert_with(|| vec![guard; size]);
        image[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
    let mut allocation_buffers = BTreeMap::<u64, objects::Buffer>::new();
    let mut resources = BTreeMap::<u64, (objects::Buffer, objects::BufferView)>::new();
    let mut report_ids = BTreeMap::<(AllocationId, ViewId), (u64, u64)>::new();
    for definition in &declaring.buffers {
        let buffer = match allocation_buffers.get(&definition.allocation) {
            Some(existing) => existing.clone(),
            None => {
                let image = images
                    .remove(&definition.allocation)
                    .ok_or("missing allocation image")?;
                let created = device.new_buffer_with_bytes(image)?;
                allocation_buffers.insert(definition.allocation, created.clone());
                created
            }
        };
        let view = buffer.view(
            usize::try_from(definition.offset)?,
            usize::try_from(definition.length)?,
        )?;
        report_ids.insert(
            (view.allocation_id(), view.view_id()),
            (definition.allocation, definition.view),
        );
        resources.insert(definition.view, (buffer, view));
    }

    let queue = device.new_command_queue();
    let command = queue.command_buffer();
    let dispatches = dispatch_sequence(declaring);
    let narrow = |dimensions: [u64; 3]| -> Result<Size> {
        Ok(Size::new(
            u32::try_from(dimensions[0])?,
            u32::try_from(dimensions[1])?,
            u32::try_from(dimensions[2])?,
        )?)
    };
    let mut object_textures = BTreeMap::new();
    for texture in &declaring.textures {
        let initial = unhex(&texture.initial_hex)?;
        let created = device.new_texture_with_bytes(
            TextureFormat::R32Uint,
            texture.width,
            texture.height,
            initial,
        )?;
        object_textures.insert(texture.binding, created);
    }

    // The declaring pass is one submission with one dispatch (`compare.py`
    // `_render_plan` pins that shape), so it records into the same command
    // buffer the render encoder opens below.
    let mut compute = command.compute_command_encoder()?;
    for dispatch in &dispatches {
        compute.clear_buffers()?;
        compute.clear_textures()?;
        compute.set_compute_pipeline_state(&programs[dispatch.program.unwrap_or(0)])?;
        for (binding, texture) in &object_textures {
            compute.set_texture(*binding, texture)?;
        }
        let slots = selected_slots(declaring, dispatch);
        let views = dispatch
            .bindings
            .clone()
            .unwrap_or_else(|| declaring.buffers.iter().map(|buffer| buffer.view).collect());
        for (slot, view) in slots.iter().zip(views) {
            let (_, view) = resources.get(&view).ok_or("unknown object fixture view")?;
            compute.set_buffer(slot.binding, view)?;
        }
        compute.dispatch_threads(narrow(dispatch.grid)?, narrow(dispatch.local)?)?;
    }
    compute.end_encoding()?;

    // One declared view per attachment, in location order, and one load
    // operation each: the object rail's attachment shape follows the fixture —
    // a clearing case carries its colour, a loading case keeps the bytes the
    // attachment view holds at commit and the rail uploads them
    // (`research/docs/23` §3.3).
    let attachment_views = attachments
        .iter()
        .map(|(attachment, _)| {
            resources
                .get(&attachment.view)
                .ok_or_else(|| {
                    format!(
                        "the declaring pass does not declare the attachment view {}",
                        attachment.view
                    )
                    .into()
                })
                .map(|(_, view)| view.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    let attachment_loads = attachments
        .iter()
        .map(|(attachment, _)| match attachment.load.as_str() {
            "clear" => Ok(objects::RenderAttachmentLoad::Clear(
                unhex(
                    attachment
                        .clear_hex
                        .as_deref()
                        .ok_or("a clear attachment needs clear_hex")?,
                )?
                .try_into()
                .map_err(|_| -> Box<dyn Error> { "a clear colour is four bytes".into() })?,
            )),
            "load" => Ok(objects::RenderAttachmentLoad::Load),
            "dontcare" => Ok(objects::RenderAttachmentLoad::DontCare),
            other => Err(format!("render case {}: unknown load op {other:?}", case.id).into()),
        })
        .collect::<Result<Vec<_>>>()?;
    // The recorded attachment list is positional: entry `i` is location `i`,
    // which is the M6 object API's `draw_*_with_attachments` shape. The
    // fixture's store operation travels with each entry the same way the trace
    // rail carries it (`docs/23` §3.6, v19).
    let recorded = attachments
        .iter()
        .zip(attachment_views.iter())
        .zip(attachment_loads.iter())
        .map(|(((attachment, _), view), load)| {
            let store = match attachment.store.as_str() {
                "store" => StoreOp::Store,
                "dontcare" => StoreOp::DontCare,
                other => {
                    return Err(
                        format!("render case {}: unsupported store op {other:?}", case.id).into(),
                    );
                }
            };
            Ok(objects::RenderColorAttachment {
                view,
                format: attachment_format(&attachment.format)?,
                load: *load,
                store,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let present =
        match &case.present {
            Some(definition) => Some(match &definition.initial_hex {
                Some(hex) => objects::PresentInitial::Sentinel(unhex(hex)?.try_into().map_err(
                    |_| -> Box<dyn Error> { "a present sentinel is four bytes".into() },
                )?),
                None => objects::PresentInitial::Undefined,
            }),
            None => None,
        };
    // A render case's indirect section replays its single triangle draw from
    // one encoded command. The same translator the trace rail uses builds the
    // payload, so the two rails carry byte-identical requests.
    let icb = match &case.icb {
        Some(_) => {
            let payload = render_icb_payload(case)?.ok_or("missing render indirect payload")?;
            Some(device.new_indirect_command_buffer(
                payload.command.kind(),
                payload.buffer.max_commands,
                payload.buffer.kinds,
                payload.range,
                payload.command,
            )?)
        }
        None => None,
    };
    let mut render = command.render_command_encoder()?;
    render.set_render_pipeline_state(render_pipeline)?;
    // The pass's scissor, when the case declares one, is encoder state on this
    // rail exactly as it is on Metal's (`research/docs/23` §3.3, v30).
    let scissor = match case.scissor {
        Some([x, y, width, height]) => Some([
            u32::try_from(x)?,
            u32::try_from(y)?,
            u32::try_from(width)?,
            u32::try_from(height)?,
        ]),
        None => None,
    };
    render.set_scissor(scissor)?;
    // The extent the recorded pass states is its colour attachment's — or, for
    // the zero-colour-attachment depth pass (`research/docs/23` §3.3, v46), its
    // depth attachment's: the depth surface is the whole raster, and the object
    // API's draw entries take that extent as their own argument.
    let (width, height) = match attachments.first() {
        Some((attachment, _)) => (attachment.width, attachment.height),
        None => {
            let depth = case
                .depth
                .as_ref()
                .ok_or("a render case without a colour attachment needs a depth attachment")?;
            (depth.width, depth.height)
        }
    };
    // The pass's own streams (`research/docs/23` §3.3): the object API binds
    // the views the case declares, and the encoder carries their bytes into the
    // trace at commit. A vertex-input case is a direct indexed draw by
    // construction, so it never reaches the indirect arm.
    let (vertex_buffers, indices) = render_inputs(case, &format!("render case {}", case.id))?;
    // The reviewed render-sampler case binds its texture through the object
    // API's own entry (`research/docs/23` §3.3, v70): the encoder's binding is
    // what the pass the command commits carries, exactly as a vertex stream's
    // is. Every other case binds none.
    let mut object_textures = Vec::new();
    if let Some(definitions) = &case.fragment_textures {
        for (index, definition) in definitions.iter().enumerate() {
            let texture = device.new_texture_with_bytes(
                TextureFormat::Rgba8Unorm,
                definition.width,
                definition.height,
                definition.texels()?,
            )?;
            render.set_fragment_texture(u32::try_from(index)?, &texture)?;
            object_textures.push(texture);
        }
    }
    let mut object_streams = Vec::with_capacity(vertex_buffers.len());
    for view in &vertex_buffers {
        let buffer = device.new_buffer_with_bytes(
            component_bytes(view, &format!("render case {} vertex stream", case.id))?.to_vec(),
        )?;
        let stream = buffer.view(usize::try_from(view.offset)?, usize::try_from(view.length)?)?;
        object_streams.push(stream);
    }
    let object_index = match &indices {
        Some(indices) => {
            let buffer = device.new_buffer_with_bytes(
                component_bytes(
                    &indices.view,
                    &format!("render case {} index buffer", case.id),
                )?
                .to_vec(),
            )?;
            let view = buffer.view(
                usize::try_from(indices.view.offset)?,
                usize::try_from(indices.view.length)?,
            )?;
            Some((view, indices.format))
        }
        None => None,
    };
    let families = object_state_families(case);
    if let Some(icb) = &icb {
        // An indirect draw replays one attachment: the MRT shape and the ICB
        // shape are mutually exclusive, which `validate_render_case` pins. The
        // entry itself carries no state family at all, so a case that declares
        // one is refused here instead of having it silently dropped — the
        // sibling of the guard the indexed ladder states below
        // (`research/docs/23` §3.3, v54 review N1).
        if !object_entry_carries_no_state(&families) {
            return Err(format!(
                "render case {}: an indirect replay carries no state, but the case declares \
                 {families:?}",
                case.id
            )
            .into());
        }
        render.draw_indirect(
            icb,
            recorded[0].view,
            recorded[0].format,
            width,
            height,
            recorded[0].load,
            present,
        )?;
    } else if let Some((index, format)) = &object_index {
        // The recording ladder's entries each carry one state family (plus the
        // counts they all take). A case that declares a combination no single
        // entry carries is refused here by name, instead of being recorded
        // through the first matching entry with the rest silently dropped —
        // the failure mode the v54 review found (`research/docs/23` §3.3, v54
        // review H1/M1).
        if !object_entry_admits(&families) {
            return Err(format!(
                "render case {}: the object rails have no single entry for the declared state \
                 {families:?}",
                case.id
            )
            .into());
        }
        for (binding, stream) in object_streams.iter().enumerate() {
            render.set_vertex_buffer(u32::try_from(binding)?, stream)?;
        }
        render.set_index_buffer(index, *format)?;
        // The instance count belongs to the draw call, exactly as Metal's
        // `drawIndexedPrimitives(...:instanceCount:)` spells it
        // (`research/docs/23` §3.3, v32): a single-instance case keeps the
        // pre-v32 entry point and its bytes, while the reviewed instanced case
        // takes the second one.
        let index_count = u32::try_from(case.vertices)?;
        let cull = case_cull(case)?;
        let blend = case_blend(case)?;

        // The object rail's own stencil pair (`research/docs/23` §3.3, v47/v48):
        // the same rail-owned surface and the same state the trace contract names,
        // converted once so the recording entry takes exactly what the fixture
        // declares.
        let object_stencil = {
            let (stencil, test) = case_stencil(case, &format!("render case {}", case.id))?;
            // The identity the recording names is the *object* view the
            // declaring pass bound, exactly as the depth landing does
            // (`research/docs/23` §3.3, v44/v49). It is resolved before the
            // conversion because the lookup can fail.
            let object_identity = match stencil.as_ref().and_then(|stencil| stencil.identity) {
                Some(identity) => {
                    let (_, view) = resources
                        .get(&identity.view_id.get())
                        .ok_or("the declaring pass does not declare the stencil view")?;
                    Some(RenderStencilIdentity {
                        allocation_id: view.allocation_id(),
                        view_id: view.view_id(),
                    })
                }
                None => None,
            };
            (
                stencil.map(|stencil| objects::RenderStencilAttachment {
                    width: stencil.width,
                    height: stencil.height,
                    load: match stencil.load {
                        metal_api_core::provider::StencilLoadOp::Clear(value) => {
                            objects::RenderStencilLoad::Clear(value)
                        }
                        metal_api_core::provider::StencilLoadOp::Load => {
                            objects::RenderStencilLoad::Load
                        }
                    },
                    store: stencil.store,
                    identity: object_identity,
                }),
                test.map(|test| objects::RenderStencilTest {
                    compare: test.compare,
                    fail_op: test.fail_op,
                    depth_fail_op: test.depth_fail_op,
                    pass_op: test.pass_op,
                    read_mask: test.read_mask,
                    write_mask: test.write_mask,
                    reference: test.reference,
                }),
            )
        };
        if case.depth.is_some() && case.multisample.is_none() {
            // The reviewed depth case opens the surface through the object
            // API's depth entry (`research/docs/23` §3.3, v36/v37): the pass
            // descriptor the trace rails carry and the object rail's own draw
            // entry name the same rail-owned surface and state.
            let (depth, depth_test) = case_depth(case, &format!("render case {}", case.id))?;
            let depth = depth.ok_or("a depth case needs a depth attachment")?;
            let depth = objects::RenderDepthAttachment {
                width: depth.width,
                height: depth.height,
                load: match depth.load {
                    DepthLoadOp::Clear(bits) => {
                        objects::RenderDepthLoad::Clear(f32::from_bits(bits))
                    }
                    DepthLoadOp::Load => objects::RenderDepthLoad::Load,
                },
                // The v44 recording carries the same store action and landing
                // identity the trace contract does (`research/docs/23` §3.3,
                // v43/v44), so a stored depth surface is observable on both
                // object rails exactly as it is on the trace rails. The
                // identity the recording names is the *object* view the
                // declaring pass bound — the fixture's own view id maps to it
                // through the resources table, exactly as the colour
                // attachment's does — while the report maps it back for the
                // comparison.
                store: depth.store,
                identity: match depth.identity {
                    Some(identity) => {
                        let (_, view) = resources
                            .get(&identity.view_id.get())
                            .ok_or("the declaring pass does not declare the depth view")?;
                        Some(RenderDepthIdentity {
                            allocation_id: view.allocation_id(),
                            view_id: view.view_id(),
                        })
                    }
                    None => None,
                },
            };
            let depth_test = depth_test.map(|test| objects::RenderDepthTest {
                compare: test.compare,
                write: test.write,
            });
            render.draw_indexed_primitives_with_depth(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.instance_count)?,
                depth,
                depth_test,
                present,
            )?;
        } else if let Some(multisample) = case_multisample(case)? {
            // The reviewed multisample case runs on the object rails too
            // (`research/docs/23` §3.3, v51/v52): the encoder records the same
            // pass-wide raster the trace contract names, so the pass it
            // becomes is the one the other rails execute. A case that also
            // opens a depth surface takes the combined entry (`§3.3`, v53/v54)
            // — the rail-owned surface the pass tests and writes, never keeps
            // — unless the case states its depth resolve (`§3.3`, v57/v58):
            // then the stored surface takes the resolving entry, which names
            // the same landing identity the trace contract carries.
            //
            // A stencil surface beside the raster takes the combined entry
            // from v56 on (`research/docs/23` §3.3, v55/v56); the contract's
            // own admission already refused a stored surface and a combined
            // depth-stencil surface.
            let (depth, depth_test) = case_depth(case, &format!("render case {}", case.id))?;
            let (stencil, stencil_test) = case_stencil(case, &format!("render case {}", case.id))?;
            match (depth, stencil) {
                (Some(depth), Some(stencil)) => {
                    // Both faces of the combined depth-stencil surface at once
                    // (`research/docs/23` §3.3, v66/v68/v70): the recording
                    // entry carries the same surface the trace rails execute —
                    // one surface the pass tests and writes through both faces
                    // — in whichever of the two reviewed shapes the case
                    // states. The v68 entry carries the pair kept by neither
                    // face (observed through the colour resolve); the v70
                    // resolving entry carries the v60 stored pair, both faces
                    // kept through their own resolves. The pair keeps both
                    // faces or neither, so a half-resolved pair belongs to no
                    // reviewed family set and `object_entry_admits` refuses it
                    // before this arm is reached; the match below is what makes
                    // the recorded pass the one the trace contract states.
                    let depth_resolve = case_depth_resolve(case)?;
                    let stencil_resolve = case_stencil_resolve(case)?;
                    let object_depth = objects::RenderDepthAttachment {
                        width: depth.width,
                        height: depth.height,
                        load: match depth.load {
                            DepthLoadOp::Clear(bits) => {
                                objects::RenderDepthLoad::Clear(f32::from_bits(bits))
                            }
                            DepthLoadOp::Load => objects::RenderDepthLoad::Load,
                        },
                        store: depth.store,
                        // The landing travels exactly as the depth-only arm's
                        // does (`research/docs/23` §3.3, v43/v44): the stored
                        // pair names the object view the declaring pass bound,
                        // while the rail-owned v66 pair has no identity to map
                        // and keeps the absent shape.
                        identity: match depth.identity {
                            Some(identity) => {
                                let (_, view) = resources
                                    .get(&identity.view_id.get())
                                    .ok_or("the declaring pass does not declare the depth view")?;
                                Some(RenderDepthIdentity {
                                    allocation_id: view.allocation_id(),
                                    view_id: view.view_id(),
                                })
                            }
                            None => None,
                        },
                    };
                    let object_depth_test = depth_test.map(|test| objects::RenderDepthTest {
                        compare: test.compare,
                        write: test.write,
                    });
                    let object_stencil = objects::RenderStencilAttachment {
                        width: stencil.width,
                        height: stencil.height,
                        load: match stencil.load {
                            metal_api_core::provider::StencilLoadOp::Clear(value) => {
                                objects::RenderStencilLoad::Clear(value)
                            }
                            metal_api_core::provider::StencilLoadOp::Load => {
                                objects::RenderStencilLoad::Load
                            }
                        },
                        store: stencil.store,
                        // The stencil face's landing is the depth face's rule
                        // one byte wide (`research/docs/23` §3.3, v49/v70).
                        identity: match stencil.identity {
                            Some(identity) => {
                                let (_, view) = resources.get(&identity.view_id.get()).ok_or(
                                    "the declaring pass does not declare the stencil view",
                                )?;
                                Some(RenderStencilIdentity {
                                    allocation_id: view.allocation_id(),
                                    view_id: view.view_id(),
                                })
                            }
                            None => None,
                        },
                    };
                    let object_stencil_test = stencil_test.map(|test| objects::RenderStencilTest {
                        compare: test.compare,
                        fail_op: test.fail_op,
                        depth_fail_op: test.depth_fail_op,
                        pass_op: test.pass_op,
                        read_mask: test.read_mask,
                        write_mask: test.write_mask,
                        reference: test.reference,
                    });
                    match (depth_resolve, stencil_resolve) {
                        (Some(depth_filter), Some(stencil_filter)) => {
                            // The v60 stored pair's own entry
                            // (`research/docs/23` §3.3, v60/v70): both faces
                            // kept, each through the filter the case states, so
                            // the pass the object rail records is the resolving
                            // pass the trace rails execute.
                            render.draw_indexed_primitives_with_multisample_depth_stencil_resolve(
                                &recorded,
                                width,
                                height,
                                index_count,
                                u32::try_from(case.instance_count)?,
                                multisample,
                                object_depth,
                                depth_filter.filter,
                                object_stencil,
                                stencil_filter.filter,
                                object_depth_test,
                                object_stencil_test,
                                present,
                            )?;
                        }
                        (None, None) => {
                            render.draw_indexed_primitives_with_multisample_depth_stencil(
                                &recorded,
                                width,
                                height,
                                index_count,
                                u32::try_from(case.instance_count)?,
                                multisample,
                                object_depth,
                                object_stencil,
                                object_depth_test,
                                object_stencil_test,
                                present,
                            )?;
                        }
                        // The pair keeps both faces or neither, so exactly one
                        // resolve is no reviewed shape at all — the family gate
                        // above refuses it by name, and this arm keeps the
                        // ladder total for a directly constructed case.
                        _ => {
                            return Err(format!(
                                "render case {}: the combined pair resolves both faces or \
                                 neither",
                                case.id
                            )
                            .into())
                        }
                    }
                }
                (Some(depth), None) => {
                    let resolve = case_depth_resolve(case)?;
                    let object_depth = objects::RenderDepthAttachment {
                        width: depth.width,
                        height: depth.height,
                        load: match depth.load {
                            DepthLoadOp::Clear(bits) => {
                                objects::RenderDepthLoad::Clear(f32::from_bits(bits))
                            }
                            DepthLoadOp::Load => objects::RenderDepthLoad::Load,
                        },
                        store: depth.store,
                        // The stored surface beside the raster reaches this
                        // arm only with its resolve (`validate_render_case`),
                        // and then names the object view the declaring pass
                        // bound — the same landing the trace contract carries;
                        // the rail-owned shape keeps the identity absent.
                        identity: match (&resolve, depth.identity) {
                            (Some(_), Some(identity)) => {
                                let (_, view) = resources
                                    .get(&identity.view_id.get())
                                    .ok_or("the declaring pass does not declare the depth view")?;
                                Some(RenderDepthIdentity {
                                    allocation_id: view.allocation_id(),
                                    view_id: view.view_id(),
                                })
                            }
                            _ => None,
                        },
                    };
                    let object_depth_test = depth_test.map(|test| objects::RenderDepthTest {
                        compare: test.compare,
                        write: test.write,
                    });
                    if let Some(resolve) = resolve {
                        // The stored surface's own tail (`research/docs/23`
                        // §3.3, v57/v58): the entry carries the filter the
                        // case states, so the pass the object rail records is
                        // the resolving pass the trace rails execute.
                        render.draw_indexed_primitives_with_multisample_depth_resolve(
                            &recorded,
                            width,
                            height,
                            index_count,
                            u32::try_from(case.instance_count)?,
                            object_depth,
                            object_depth_test,
                            multisample,
                            resolve.filter,
                            present,
                        )?;
                    } else {
                        render.draw_indexed_primitives_with_multisample_depth(
                            &recorded,
                            width,
                            height,
                            index_count,
                            u32::try_from(case.instance_count)?,
                            object_depth,
                            object_depth_test,
                            multisample,
                            present,
                        )?;
                    }
                }
                (None, Some(stencil)) => {
                    let object_stencil = objects::RenderStencilAttachment {
                        width: stencil.width,
                        height: stencil.height,
                        load: match stencil.load {
                            metal_api_core::provider::StencilLoadOp::Clear(value) => {
                                objects::RenderStencilLoad::Clear(value)
                            }
                            metal_api_core::provider::StencilLoadOp::Load => {
                                objects::RenderStencilLoad::Load
                            }
                        },
                        store: stencil.store,
                        identity: None,
                    };
                    let object_stencil_test = stencil_test.map(|test| objects::RenderStencilTest {
                        compare: test.compare,
                        fail_op: test.fail_op,
                        depth_fail_op: test.depth_fail_op,
                        pass_op: test.pass_op,
                        read_mask: test.read_mask,
                        write_mask: test.write_mask,
                        reference: test.reference,
                    });
                    render.draw_indexed_primitives_with_multisample_stencil(
                        &recorded,
                        width,
                        height,
                        index_count,
                        u32::try_from(case.instance_count)?,
                        object_stencil,
                        object_stencil_test,
                        multisample,
                        present,
                    )?;
                }
                (None, None) => {
                    render.draw_indexed_primitives_with_multisample(
                        &recorded,
                        width,
                        height,
                        index_count,
                        u32::try_from(case.instance_count)?,
                        multisample,
                        present,
                    )?;
                }
            }
        } else if let (Some(stencil), stencil_test) = &object_stencil {
            // The reviewed stencil case runs on the object rails too
            // (`research/docs/23` §3.3, v47/v48): the encoder records the same
            // rail-owned surface and the same state the trace contract names,
            // so the pass it becomes is the one the other rails execute.
            render.draw_indexed_primitives_with_stencil(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.instance_count)?,
                *stencil,
                *stencil_test,
                present,
            )?;
        } else if let Some(blend) = &blend {
            // The reviewed blending case runs on the object rails too
            // (`research/docs/23` §3.3, v40/v42): the encoder states the same
            // per-attachment blend the trace contract names.
            render.draw_indexed_primitives_with_blend(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.instance_count)?,
                &blend.attachments,
                present,
            )?;
        } else if let Some(cull) = cull {
            // The reviewed culling case runs on the object rails too
            // (`research/docs/23` §3.3, v39/v41): the encoder records the same
            // mode and winding the trace contract names, through the object
            // API's own culling entry.
            render.draw_indexed_primitives_with_cull(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.instance_count)?,
                cull,
                present,
            )?;
        } else if case.base_vertex != 0 {
            // The offset belongs to the draw call, exactly as Metal's
            // `drawIndexedPrimitives(…:baseVertex:baseInstance:)` spells it
            // (`research/docs/23` §3.3, v35): a zero-offset case keeps the
            // pre-v35 entry points and their bytes.
            render.draw_indexed_primitives_base_vertex_with_attachments(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.base_vertex)?,
                u32::try_from(case.instance_count)?,
                present,
            )?;
        } else if case.instance_count > 1 {
            render.draw_indexed_primitives_instanced_with_attachments(
                &recorded,
                width,
                height,
                index_count,
                u32::try_from(case.instance_count)?,
                present,
            )?;
        } else {
            render.draw_indexed_primitives_with_attachments(
                &recorded,
                width,
                height,
                index_count,
                present,
            )?;
        }
    } else if case.instance_count > 1 {
        // The milestone's `vertex_id` triangle has no instanced entry point on
        // the object API yet: recording it with one instance would draw a
        // different pass than the trace rails, so it is refused here instead of
        // silently narrowed (`research/docs/23` §3.3, v32).
        return Err(format!(
            "render case {}: the object rails have no instanced vertex_id shape",
            case.id
        )
        .into());
    } else {
        // The milestone's `vertex_id` triangle binds no stream and no index
        // buffer, so it is the one shape that records through the
        // single-attachment `draw_render_pass` entry point. Like an indirect
        // replay, that entry carries no state family, so a case that declares
        // one is refused rather than narrowed (`research/docs/23` §3.3, v54
        // review N1).
        if !object_entry_carries_no_state(&families) {
            return Err(format!(
                "render case {}: the vertex_id milestone carries no state, but the case \
                 declares {families:?}",
                case.id
            )
            .into());
        }
        render.draw_render_pass(
            recorded[0].view,
            recorded[0].format,
            width,
            height,
            recorded[0].load,
            present,
        )?;
    }
    render.end_encoding()?;

    command.commit()?;
    if async_execution {
        if command.status()? != metal_api_core::CommandBufferStatus::Committed {
            return Err("async object render commit did not leave the command pending".into());
        }
        if !matches!(
            command.submission()?.completion,
            CompletionDisposition::Submitted { .. }
        ) {
            return Err("async object render commit did not return a submitted token".into());
        }
    }
    command.wait_until_completed()?;
    if command.status()? != metal_api_core::CommandBufferStatus::Completed {
        return Err("object render command did not reach Completed".into());
    }
    let output = command.submission()?;
    if !matches!(
        output.completion,
        CompletionDisposition::CompletedVisible { .. }
    ) {
        return Err("object render capture requires completed visible results".into());
    }
    // One writeback and one allocation image per attachment, in location
    // order. Each writeback has to cover the exact range its declaring view
    // states, so a rail that landed a different range cannot be reported as
    // this case's texels. A discarded attachment's bytes disappear with the
    // pass, so no writeback and no allocation image are owed for it
    // (`docs/23` §3.6, v19).
    let mut writebacks = Vec::new();
    let mut images_report = Vec::new();
    for (attachment, _) in &attachments {
        if attachment.store != "store" {
            continue;
        }
        let landed = output
            .writebacks
            .iter()
            .find(|write| {
                report_ids.get(&(write.allocation_id, write.view_id))
                    == Some(&(attachment.allocation, attachment.view))
            })
            .ok_or("the object render rail landed no attachment writeback")?;
        let declared = declaring
            .buffers
            .iter()
            .find(|buffer| {
                buffer.allocation == attachment.allocation && buffer.view == attachment.view
            })
            .ok_or("the declaring pass does not declare the attachment view")?;
        if landed.offset != declared.offset || landed.bytes.len() as u64 != declared.length {
            return Err(format!(
                "render case {}: the attachment writeback covers {}..{} instead of {}..{}",
                case.id,
                landed.offset,
                landed.offset + landed.bytes.len() as u64,
                declared.offset,
                declared.offset + declared.length
            )
            .into());
        }
        let image = allocation_buffers
            .get(&attachment.allocation)
            .ok_or("the attachment allocation is missing")?
            .read()?;
        let (writeback, observation) = attachment_observation(
            case.expected_rule.as_deref(),
            case.readback_windows.as_deref().unwrap_or_default(),
            (
                attachment.allocation,
                attachment.view,
                landed.offset,
                attachment.width,
            ),
            &landed.bytes,
            &image,
        )?;
        writebacks.push(writeback);
        images_report.push(observation);
    }
    // The stored depth attachment's own landing (`research/docs/23` §3.3,
    // v43/v44), reported exactly as the trace rail reports it: the object API's
    // recording names the same view, so the same writeback and the same
    // allocation image are owed here.
    if let Some(definition) = &case.depth {
        if definition.store.as_deref() == Some("store") {
            let allocation = definition
                .allocation
                .ok_or("a stored depth attachment needs its allocation")?;
            let view = definition
                .view
                .ok_or("a stored depth attachment needs its view")?;
            let landed = output
                .writebacks
                .iter()
                .find(|write| {
                    report_ids.get(&(write.allocation_id, write.view_id))
                        == Some(&(allocation, view))
                })
                .ok_or("the object render rail landed no depth writeback")?;
            let declared = declaring
                .buffers
                .iter()
                .find(|buffer| buffer.allocation == allocation && buffer.view == view)
                .ok_or("the declaring pass does not declare the depth attachment view")?;
            if landed.offset != declared.offset || landed.bytes.len() as u64 != declared.length {
                return Err(format!(
                    "render case {}: the depth writeback covers {}..{} instead of {}..{}",
                    case.id,
                    landed.offset,
                    landed.offset + landed.bytes.len() as u64,
                    declared.offset,
                    declared.offset + declared.length
                )
                .into());
            }
            let expected = unhex(
                definition
                    .expected_hex
                    .as_deref()
                    .ok_or("a stored depth attachment needs expected_hex")?,
            )?;
            if landed.bytes != expected {
                return Err(format!(
                    "render case {}: the depth readback is {} against the reviewed {}",
                    case.id,
                    hex(&landed.bytes),
                    hex(&expected)
                )
                .into());
            }
            let image = allocation_buffers
                .get(&allocation)
                .ok_or("the depth allocation is missing")?
                .read()?;
            writebacks.push(Writeback::encoded(
                allocation,
                view,
                landed.offset,
                &landed.bytes,
            ));
            images_report.push(Allocation::observed(allocation, &image));
        }
    }
    // The stored stencil attachment's own landing (`research/docs/23` §3.3,
    // v49), reported exactly as the trace rail reports it.
    if let Some(definition) = &case.stencil {
        if definition.store.as_deref() == Some("store") {
            let allocation = definition
                .allocation
                .ok_or("a stored stencil attachment needs its allocation")?;
            let view = definition
                .view
                .ok_or("a stored stencil attachment needs its view")?;
            let landed = output
                .writebacks
                .iter()
                .find(|write| {
                    report_ids.get(&(write.allocation_id, write.view_id))
                        == Some(&(allocation, view))
                })
                .ok_or("the object render rail landed no stencil writeback")?;
            let declared = declaring
                .buffers
                .iter()
                .find(|buffer| buffer.allocation == allocation && buffer.view == view)
                .ok_or("the declaring pass does not declare the stencil attachment view")?;
            if landed.offset != declared.offset || landed.bytes.len() as u64 != declared.length {
                return Err(format!(
                    "render case {}: the stencil writeback covers {}..{} instead of {}..{}",
                    case.id,
                    landed.offset,
                    landed.offset + landed.bytes.len() as u64,
                    declared.offset,
                    declared.offset + declared.length
                )
                .into());
            }
            let expected = unhex(
                definition
                    .expected_hex
                    .as_deref()
                    .ok_or("a stored stencil attachment needs expected_hex")?,
            )?;
            if landed.bytes != expected {
                return Err(format!(
                    "render case {}: the stencil readback is {} against the reviewed {}",
                    case.id,
                    hex(&landed.bytes),
                    hex(&expected)
                )
                .into());
            }
            let image = allocation_buffers
                .get(&allocation)
                .ok_or("the stencil allocation is missing")?
                .read()?;
            writebacks.push(Writeback::encoded(
                allocation,
                view,
                landed.offset,
                &landed.bytes,
            ));
            images_report.push(Allocation::observed(allocation, &image));
        }
    }
    eprintln!(
        "objects render case completed: {} attachments={} bytes={}",
        case.id,
        attachments.len(),
        writebacks
            .iter()
            .map(Writeback::observed_bytes)
            .sum::<usize>()
    );
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks,
        allocations: images_report,
        copy_in: None,
        copy_out: None,
        group_counts: None,
        present: None,
        heap: None,
        icb: None,
        storage_modes: None,
    })
}

fn run_case(
    provider: &dyn PipelineProvider,
    programs: &[CompiledComputePipeline],
    case: &Case,
    operation: u64,
    guard: u8,
    counters: &mut dyn FnMut() -> (usize, usize),
    leases: &LeaseHandles,
) -> Result<CaseResult> {
    let mut resources = ResourceTableSnapshot::new();
    // One backing image and one AllocationRecord per allocation. A v10 fixture
    // binds two disjoint views of the same allocation, so every view's initial
    // bytes land in that allocation's single image and the trace carries one
    // record with two view ranges.
    let mut allocations: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut recorded = BTreeSet::new();
    for buffer in &case.buffers {
        let initial = buffer.initial_bytes()?;
        let start = usize::try_from(buffer.offset)?;
        let position = match allocations
            .iter()
            .position(|(allocation, _)| *allocation == buffer.allocation)
        {
            Some(position) => position,
            None => {
                allocations.push((
                    buffer.allocation,
                    vec![guard; usize::try_from(buffer.allocation_size)?],
                ));
                allocations.len() - 1
            }
        };
        if recorded.insert(buffer.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(buffer.allocation),
                owner_epoch: provider.device_epoch(),
                size: buffer.allocation_size,
            })?;
        }
        allocations[position].1[start..start + initial.len()].copy_from_slice(&initial);
    }
    // The source arm every view declares (`research/docs/23` §90, R9i): an
    // owned view reads the case's own image as the run rewrites it, while a
    // lease-armed view names one import the provider already holds. The
    // imports are made here, before the trace is built, because the snapshot
    // the admission reads has to carry the same reservations the trace's
    // `BufferSource` names; the owner windows a borrowed view maps are kept
    // alive in `owner_windows` until this case's last submission has been
    // waited for.
    let mut case_leases = CaseLeases::default();
    let mut owner_windows = Vec::new();
    let view_sources = case_sources(
        case,
        &allocations,
        provider,
        leases,
        guard,
        &mut case_leases,
        &mut owner_windows,
    )?;
    for reservation in &case_leases.reservations {
        resources.insert_lease(*reservation)?;
    }
    // v11: sampled textures are their own allocations; the provider uploads
    // them once per submission (`research/docs/18` step 1).
    let mut case_textures = Vec::with_capacity(case.textures.len());
    for texture in &case.textures {
        let initial = unhex(&texture.initial_hex)?;
        if recorded.insert(texture.allocation) {
            resources.insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(texture.allocation),
                owner_epoch: provider.device_epoch(),
                size: u64::try_from(initial.len())?,
            })?;
        }
        let access = match texture.access.as_str() {
            "sampled" => TextureAccess::Sampled,
            "storage" => TextureAccess::Storage,
            _ => return Err("unsupported texture access".into()),
        };
        let format = match texture.format.as_str() {
            "r32_uint" => TextureFormat::R32Uint,
            _ => return Err("unsupported texture format".into()),
        };
        case_textures.push(TextureView {
            view_id: ViewId::new(texture.view),
            metal_binding: texture.binding,
            allocation_id: AllocationId::new(texture.allocation),
            texture_type: TextureType::D2,
            format,
            width: texture.width,
            height: texture.height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access,
            source: TextureSource::OwnedBytes(initial),
        });
    }
    // Every command buffer snapshots the bytes landed so far, so a later
    // command reads what an earlier command wrote.
    let case_views = |allocations: &[(u64, Vec<u8>)]| -> Result<Vec<BufferView>> {
        case.buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                let access = match buffer.access.as_str() {
                    "read" => BufferAccess::Read,
                    "write" => BufferAccess::Write,
                    "read_write" => BufferAccess::ReadWrite,
                    _ => return Err("unsupported access".into()),
                };
                let (_, backing) = allocations
                    .iter()
                    .find(|(allocation, _)| *allocation == buffer.allocation)
                    .ok_or("unknown fixture allocation")?;
                let start = usize::try_from(buffer.offset)?;
                let end = start + usize::try_from(buffer.length)?;
                Ok(BufferView {
                    view_id: ViewId::new(buffer.view),
                    metal_binding: buffer.binding,
                    allocation_id: AllocationId::new(buffer.allocation),
                    offset: buffer.offset,
                    length: buffer.length,
                    access,
                    attribute_stride: None,
                    source: match &view_sources[index] {
                        CaseSource::Owned => BufferSource::OwnedBytes(backing[start..end].to_vec()),
                        CaseSource::Lease(source) => source.clone(),
                    },
                })
            })
            .collect()
    };
    let dispatches = dispatch_sequence(case);
    let groups = case_command_buffers(case);
    let initial_views = case_views(&allocations)?;
    // Refusal guards run on the complete sequence before any submission.
    let guard_trace = case_trace(
        provider.device_epoch(),
        programs,
        case,
        operation,
        &initial_views,
        &case_textures,
        &dispatches,
    )?;
    if case.entry == "transform_3d" {
        let mut short = guard_trace.clone();
        let shortened = short.passes[0]
            .as_compute()
            .ok_or("guard trace pass is not a compute pass")?
            .buffers[0]
            .view_id;
        for pass in short
            .passes
            .iter_mut()
            .filter_map(TracePass::as_compute_mut)
        {
            if let Some(view) = pass
                .buffers
                .iter_mut()
                .find(|view| view.view_id == shortened)
            {
                view.length = 119;
                if let BufferSource::OwnedBytes(bytes) = &mut view.source {
                    bytes.truncate(119);
                }
            }
        }
        let rejected = provider.capabilities().admit(&short, &resources);
        if !matches!(rejected, Err(ref error) if error.slug == "buffer_footprint_exceeds_view") {
            return Err(format!("3D fixture must refuse 119-byte view: {rejected:?}").into());
        }
    }
    if programs.len() > 1 {
        let mut forged = guard_trace.clone();
        forged.pipelines[1].contract.buffer_bindings[0].footprint =
            FootprintProof::Static { max_bytes: 1 };
        let input = provider
            .capabilities()
            .validate_trace(forged, resources.clone())
            .map_err(|error| format!("malformed late-pipeline refusal fixture: {error:?}"))?;
        if !matches!(provider.submit(input), Err(error) if error.slug == "pipeline_contract_mismatch"
            && error.completion == CompletionDisposition::NotSubmitted)
        {
            return Err(
                "provider failed to reject second-pipeline forged metadata before submission"
                    .into(),
            );
        }
        let mut unknown = guard_trace.clone();
        let original_id = unknown.pipelines[1].pipeline_id;
        let missing = metal_api_core::provider::PipelineId::new(u64::MAX);
        unknown.pipelines[1].pipeline_id = missing;
        for pass in unknown
            .passes
            .iter_mut()
            .filter_map(TracePass::as_compute_mut)
        {
            if pass.pipeline == original_id {
                pass.pipeline = missing;
            }
        }
        let input = provider
            .capabilities()
            .validate_trace(unknown, resources.clone())
            .map_err(|error| format!("malformed unknown-pipeline refusal fixture: {error:?}"))?;
        if !matches!(provider.submit(input), Err(error) if error.slug == "unknown_pipeline"
            && error.completion == CompletionDisposition::NotSubmitted)
        {
            return Err(
                "provider failed to reject unknown second pipeline before submission".into(),
            );
        }
        eprintln!("Checked second-pipeline refusal guards: {}", case.id);
    }
    let mut reported = Vec::new();
    let mut group_counts = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let before = counters();
        let views = case_views(&allocations)?;
        let selected = group
            .iter()
            .map(|position| {
                dispatches
                    .get(*position)
                    .cloned()
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "command buffer dispatch index out of range".into()
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let trace_operation = if groups.len() == 1 {
            operation
        } else {
            operation * 100 + index as u64 + 1
        };
        let trace = case_trace(
            provider.device_epoch(),
            programs,
            case,
            trace_operation,
            &views,
            &case_textures,
            &selected,
        )?;
        let admitted = provider
            .capabilities()
            .validate_trace(trace.clone(), resources.clone())
            .map_err(|error| format!("admit {}: {error:?}", case.id))?;
        let output = provider
            .submit(admitted)
            .map_err(|error| format!("submit {}: {error:?}", case.id))?;
        output.validate_for_trace(&trace)?;
        let CompletionDisposition::CompletedVisible { token } = output.completion else {
            return Err("provider capture requires completed visible results".into());
        };
        if provider
            .wait(token, Duration::ZERO)
            .map_err(|error| format!("wait: {error:?}"))?
            != output.completion
        {
            return Err("provider completion observation changed".into());
        }
        for write in output.writebacks {
            let (_, backing) = allocations
                .iter_mut()
                .find(|(id, _)| *id == write.allocation_id.get())
                .ok_or("unknown writeback allocation")?;
            let start = usize::try_from(write.offset)?;
            backing[start..start + write.bytes.len()].copy_from_slice(&write.bytes);
            reported.push((
                write.allocation_id.get(),
                write.view_id.get(),
                write.offset,
                write.bytes,
            ));
        }
        provider
            .release_completion(token)
            .map_err(|error| format!("release completion: {error:?}"))?;
        let after = counters();
        group_counts.push(GroupCounts {
            copy_in: u32::try_from(after.0 - before.0)?,
            copy_out: u32::try_from(after.1 - before.1)?,
        });
    }
    let writebacks = merge_writebacks(case, reported)?;
    allocations.sort_by_key(|(id, _)| *id);
    // The lease imports belong to this case alone: a later case may reuse a
    // view id, and the registry refuses a duplicate import until the previous
    // one is released. Every submission above has been waited for and its
    // completion released, which is the point the no-copy registry's own
    // retirement rule waits on (`research/docs/23` §90, R9i).
    for lease_id in &case_leases.staged {
        leases
            .staged
            .release_staged_lease(*lease_id)
            .map_err(|error| format!("case {}: release staged lease: {error:?}", case.id))?;
    }
    for lease_id in &case_leases.borrowed {
        leases
            .borrowed
            .release_borrowed_lease(*lease_id)
            .map_err(|error| format!("case {}: release borrowed lease: {error:?}", case.id))?;
    }
    // The arms are reported from the declarations that were actually used, and
    // only for a case that declares one: a capture of a pre-R9i fixture keeps
    // the bytes it always had, while a lease case states which arm each view
    // ran with — the observation the comparator checks instead of trusting the
    // trace's own naming.
    let declared_modes = case
        .buffers
        .iter()
        .map(|buffer| Ok((buffer.view, buffer_storage_mode(buffer, &case.id)?)))
        .collect::<Result<Vec<_>>>()?;
    let storage_modes = declared_modes
        .iter()
        .any(|(_, mode)| *mode != BufferSourceKind::OwnedBytes)
        .then(|| {
            declared_modes
                .iter()
                .map(|(view, mode)| StorageModeObservation {
                    view: *view,
                    mode: storage_mode_name(*mode),
                })
                .collect()
        });
    Ok(CaseResult {
        id: case.id.clone(),
        completion: "CompletedVisible",
        writebacks,
        allocations: allocations
            .into_iter()
            .map(|(allocation, bytes)| Allocation::observed(allocation, &bytes))
            .collect(),
        copy_in: None,
        copy_out: None,
        group_counts: case.command_buffers.as_ref().map(|_| group_counts),
        present: None,
        heap: None,
        icb: None,
        storage_modes,
    })
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(format!("input exceeds size limit: {}", path.display()).into());
    }
    Ok(bytes)
}

fn verified_source(directory: &Path, source: &Source) -> Result<Vec<u8>> {
    let path = directory.join(&source.path);
    let bytes = read_bounded(&path, MAX_SOURCE_BYTES)?;
    if hex(&Sha256::digest(&bytes)) != source.sha256 {
        return Err(format!("source digest mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The second colour of a constrained wildcard texel's mix set
/// (`research/docs/23` §3.3, v67/v82).
///
/// A case states it one of two ways. A `clear` or `dontcare` multisample raster
/// carries the case-declared `clear_hex`, which is the colour the pass opens
/// the raster from (or the reference the fixture names beside `dontcare`). A
/// *loaded* multisample raster carries its declared window instead: the
/// multisampled load is executed by a seed pass whose clear value is one colour
/// for the whole attachment, so the window has to be one repeated texel and
/// that texel is the colour every sample starts from. Both spellings are the
/// fixture's own colours, and the byte string this returns is what the mix set
/// is built from.
fn wildcard_reference(
    case: &RenderCase,
    attachment: &RenderAttachmentDefinition,
) -> Result<Vec<u8>> {
    if case.multisample.is_some() && attachment.load == "load" {
        let previous = unhex(
            attachment
                .initial_hex
                .as_deref()
                .ok_or("a loaded multisample raster needs the seed its window is made of")?,
        )?;
        if previous.len() < 4
            || previous
                .chunks_exact(4)
                .any(|texel| texel != &previous[..4])
        {
            return Err(
                "a seeded multisample raster loads one repeated texel; a per-texel seed is not \
                 a shape the reviewed rails execute"
                    .into(),
            );
        }
        return Ok(previous[..4].to_vec());
    }
    unhex(
        attachment
            .clear_hex
            .as_deref()
            .ok_or("a constrained wildcard texel needs the reference colour of its mixes")?,
    )
}

fn unhex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2)
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("hex data must be lowercase, with two digits per byte".into());
    }
    Ok(value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
            digit(pair[0]) * 16 + digit(pair[1])
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One render case of the reviewed suite, for the object-entry family rules
    /// (`research/docs/23` §3.3, v54 review H1/M1).
    fn render_case(id: &str) -> RenderCase {
        let suite: Suite =
            serde_json::from_str(include_str!("../../../../conformance/suite-v28.json")).unwrap();
        suite
            .render_cases
            .into_iter()
            .find(|case| case.id == id)
            .expect("the reviewed case exists")
    }

    /// The reviewed rule's own arithmetic (R5a, `research/docs/23` §73).
    ///
    /// The texels are spelled out by hand rather than read back from the
    /// helper, so a moved channel or a swapped axis fails here instead of
    /// silently changing what every capture of the wide case reports.
    #[test]
    fn the_reviewed_rule_is_the_texel_coordinates_little_endian() {
        for (x, y, expect) in [
            (0_u64, 0_u64, [0x00, 0x00, 0x00, 0x00]),
            (1, 0, [0x01, 0x00, 0x00, 0x00]),
            (0, 1, [0x00, 0x00, 0x01, 0x00]),
            (255, 3, [0xff, 0x00, 0x03, 0x00]),
            (256, 3, [0x00, 0x01, 0x03, 0x00]),
            (2047, 2047, [0xff, 0x07, 0xff, 0x07]),
        ] {
            assert_eq!(
                rule_texel(XY_U16LE_V1, x, y).unwrap(),
                expect,
                "texel ({x}, {y})"
            );
        }
        // A rule that is not the reviewed one is refused by name, and the
        // addressing ceiling is the rule's own: both coordinates travel as
        // `u16`s, so a wider axis could not be spelled injectively.
        assert!(rule_texel("other_rule", 0, 0).is_err());
        assert!(rule_texel(XY_U16LE_V1, RULE_ADDRESS_CEILING, 0).is_err());
    }

    /// The closed-form clear check: the rule reaches a colour exactly when the
    /// `x` and `y` its halves name are inside the extent.
    #[test]
    fn the_rule_reaches_exactly_the_colours_inside_the_window() {
        let texel = rule_texel(XY_U16LE_V1, 5, 7).unwrap();
        assert!(rule_reaches_colour(XY_U16LE_V1, &texel, 2048, 2048).unwrap());
        // One axis outside the window is enough: the same colour is a texel of
        // a wider plane only.
        assert!(!rule_reaches_colour(XY_U16LE_V1, &texel, 5, 2048).unwrap());
        assert!(!rule_reaches_colour(XY_U16LE_V1, &texel, 2048, 7).unwrap());
        assert!(rule_reaches_colour(XY_U16LE_V1, &[0x00, 0x00, 0x01, 0x00], 2048, 2048).unwrap());
    }

    /// One changed texel of the wide plane changes the digest the capture
    /// reports, and a window that leaves the plane is refused instead of
    /// reported.
    #[test]
    fn one_changed_texel_changes_the_reported_digest() {
        let plane = rule_bytes(XY_U16LE_V1, 2048, 2048).unwrap();
        assert_eq!(plane.len(), 2048 * 2048 * 4);
        let digest = plane_sha256(&plane);
        let mut changed = plane.clone();
        // The second byte of the texel at (1024, 1024): only the digest can
        // catch it, which is what the wide case's observation rests on.
        changed[(1024 * 2048 + 1024) * 4 + 1] ^= 0x01;
        assert_ne!(plane_sha256(&changed), digest);
        assert_ne!(changed, plane);

        let window = ReadbackWindowDefinition {
            x: 1984,
            y: 1984,
            width: 64,
            height: 64,
        };
        let observed = window.observe(&plane, 2048).unwrap();
        assert_eq!(observed.bytes_hex.len(), 64 * 64 * 4 * 2);
        assert!(observed
            .bytes_hex
            .starts_with(&hex(&rule_texel(XY_U16LE_V1, 1984, 1984).unwrap())));
        let outside = ReadbackWindowDefinition {
            x: 1984,
            y: 1984,
            width: 65,
            height: 64,
        };
        assert!(outside.observe(&plane, 2048).is_err());
    }

    #[test]
    fn the_object_entry_family_rules_refuse_every_unreviewed_combination() {
        // The single-family entries are what the reviewed fixtures use.
        for id in [
            "scissor_left_half_4x4",
            "depth_pair_4x4",
            "stencil_increment_pair_4x4",
            "blend_alpha_quad_4x4",
            "cull_back_half_quad_4x4",
        ] {
            let case = render_case(id);
            let families = object_state_families(&case);
            assert!(
                object_entry_admits(&families),
                "{id} declares one reviewed family: {families:?}"
            );
        }
        // The v54 increment's combined entry carries the raster and the depth
        // surface, and nothing else.
        let combined = object_state_families(&render_case("msaa_depth_pair_4x4"));
        assert_eq!(combined, vec!["multisample", "depth"]);
        assert!(object_entry_admits(&combined));
        // The reviewed colour-only raster keeps its own entry.
        let colour_only = object_state_families(&render_case("msaa_edge_4x4"));
        assert_eq!(colour_only, vec!["multisample"]);
        assert!(object_entry_admits(&colour_only));

        // The v66 combined pair declares three families at once, and the v68
        // recording entry carries both faces beside the raster, so the object
        // rails admit it by name and its marker names all five rails
        // (`research/docs/23` §3.3, v66/v68).
        let combined_pair = object_state_families(&render_case("msaa_ds_pair_4x4"));
        assert_eq!(combined_pair, vec!["multisample", "depth", "stencil"]);
        assert!(object_entry_admits(&combined_pair));

        // The v60 stored combined shape keeps both faces through their own
        // resolves, and the v70 recording entry carries exactly that shape
        // (`research/docs/23` §3.3, v60/v70): the object rails admit it by
        // name, so its marker can name the object rails beside the trace ones.
        let stored_combined =
            object_state_families(&render_case("msaa_stencil_resolve_sample0_4x4"));
        assert_eq!(
            stored_combined,
            vec![
                "multisample",
                "depth",
                "depth_resolve",
                "stencil",
                "stencil_resolve"
            ]
        );
        assert!(object_entry_admits(&stored_combined));

        // A half-resolved pair is no reviewed shape: the v60 fixtures resolve
        // both faces, and the ladder has one entry per whole shape, so a case
        // that keeps its two faces through only one resolve is refused by name
        // (`research/docs/23` §3.3, v70).
        let mut half_resolved = render_case("msaa_stencil_resolve_sample0_4x4");
        half_resolved.stencil_resolve = None;
        let families = object_state_families(&half_resolved);
        assert_eq!(
            families,
            vec!["multisample", "depth", "depth_resolve", "stencil"]
        );
        assert!(!object_entry_admits(&families));
        let mut stencil_only = render_case("msaa_stencil_resolve_sample0_4x4");
        stencil_only.depth_resolve = None;
        let families = object_state_families(&stencil_only);
        assert_eq!(
            families,
            vec!["multisample", "depth", "stencil", "stencil_resolve"]
        );
        assert!(!object_entry_admits(&families));

        // The v54 review's worst case: a raster plus a vertex offset. No entry
        // carries both, so the object rail refuses it by name instead of
        // recording a pass with the offset dropped.
        let mut offset = render_case("msaa_depth_pair_4x4");
        offset.base_vertex = 1;
        let families = object_state_families(&offset);
        assert_eq!(families, vec!["multisample", "depth", "base_vertex"]);
        assert!(!object_entry_admits(&families));

        // The gate's other hole the review found: a stencil surface plus a
        // blend state, which the ladder used to record through the stencil
        // entry with the blend dropped.
        let mut blended = render_case("blend_alpha_quad_4x4");
        blended.stencil = Some(StencilAttachmentDefinition {
            format: "stencil8".into(),
            width: 4,
            height: 4,
            load: "clear".into(),
            clear_value: Some(0),
            store: None,
            allocation: None,
            view: None,
            expected_hex: None,
        });
        let families = object_state_families(&blended);
        assert_eq!(families, vec!["stencil", "blend"]);
        assert!(!object_entry_admits(&families));

        // The admission table itself, over every subset of the seven families
        // (the v54 review's N3): a widening in the implementation that this
        // list does not state fails the "everything else is refused" half.
        let reviewed: Vec<Vec<&str>> = vec![
            vec![],
            vec!["multisample"],
            vec!["depth"],
            vec!["stencil"],
            vec!["blend"],
            vec!["cull"],
            vec!["base_vertex"],
            vec!["multisample", "depth"],
            vec!["multisample", "depth", "depth_resolve"],
            vec!["multisample", "stencil"],
            vec!["multisample", "depth", "stencil"],
            vec![
                "multisample",
                "depth",
                "depth_resolve",
                "stencil",
                "stencil_resolve",
            ],
        ];
        let all = [
            "multisample",
            "depth",
            "depth_resolve",
            "stencil",
            "stencil_resolve",
            "blend",
            "cull",
            "base_vertex",
        ];
        for mask in 0..(1_u32 << all.len()) {
            let subset: Vec<&str> = all
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(_, family)| *family)
                .collect();
            let expected = reviewed.contains(&subset);
            assert_eq!(
                object_entry_admits(&subset),
                expected,
                "the admission table disagrees about {subset:?}"
            );
        }
        // The two state-free entries admit exactly the empty family list.
        assert!(object_entry_carries_no_state(&[]));
        for family in all {
            assert!(!object_entry_carries_no_state(&[family]));
        }
    }

    fn suite() -> Suite {
        serde_json::from_str(include_str!("../../../../conformance/suite.json")).unwrap()
    }
    #[test]
    fn suite_rejects_ranges_and_data_that_cannot_describe_the_shared_case() {
        let mut s = suite();
        validate_suite(&s).unwrap();
        s.cases[0].buffers[0].offset = u64::MAX;
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].buffers[0].initial_hex = Some("00".into());
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].buffers[0].allocation_size = u64::MAX;
        assert!(validate_suite(&s).is_err());
    }
    #[test]
    fn suite_cannot_silently_expand_the_qualified_shader_dispatch() {
        let mut s = suite();
        s.cases[1].grid = [11, 3, 1];
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[1].id = s.cases[0].id.clone();
        assert!(validate_suite(&s).is_err());
        let mut s = suite();
        s.cases[0].air.sha256 = "0".repeat(64);
        assert!(validate_suite(&s).is_err());
        assert!(unhex("0aFF").is_err());
        assert!(unhex("0").is_err());
    }

    #[test]
    fn v9_splits_the_reviewed_sequence_across_command_buffers() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v9.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        for case in &s.cases {
            let groups = case_command_buffers(case);
            assert_eq!(
                groups.len(),
                if case.id == "subset_chain_four" {
                    2
                } else {
                    groups.len()
                }
            );
            let flattened: Vec<usize> = groups.iter().flatten().copied().collect();
            assert_eq!(
                flattened,
                (0..dispatch_sequence(case).len()).collect::<Vec<_>>()
            );
            assert!(groups.iter().all(|group| !group.is_empty()));
        }
        let mut s = load();
        s.cases[0].command_buffers = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![1], vec![0]]);
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![0, 1], vec![]]);
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].command_buffers = Some(vec![vec![0], vec![1], vec![2]]);
        assert!(validate_suite(&s).is_err());
        let mut legacy = suite();
        legacy.cases[0].command_buffers = Some(vec![vec![0]]);
        assert!(validate_suite(&legacy).is_err());
    }

    #[test]
    fn v2_requires_sparse_bindings_read_write_access_and_ordered_outputs() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v2.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let mut s = load();
        s.cases[6].buffers[1].binding = 1;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].buffers[0].access = "write".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].expected_writebacks.reverse();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[6].buffers[0].length = 119;
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn v2_cases_cannot_be_mislabeled_as_v1_or_change_fixed_3d_shape() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v2.json"))
                .unwrap()
        };
        let mut s = load();
        s.suite = "compute-buffer-v1".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[7].grid = [6, 3, 2];
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].id = "copy_word".into();
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn serial_suite_admits_only_reviewed_dispatch_sequences() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v3.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let mut s = load();
        s.cases[0].dispatches = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap().pop();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[2].dispatches.as_mut().unwrap().push(CaseDispatch {
            grid: [5, 3, 2],
            local: [4, 2, 2],
            bindings: None,
            program: None,
        });
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[1].dispatches.as_mut().unwrap()[1].grid = [6, 3, 2];
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn a_single_pass_case_cannot_silently_acquire_extra_gpu_work() {
        let mut s = suite();
        s.cases[0].dispatches = Some(vec![CaseDispatch {
            grid: [1, 1, 1],
            local: [1, 1, 1],
            bindings: None,
            program: None,
        }]);
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn pingpong_case_requires_exact_view_permutations_and_final_writebacks() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v4.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        assert_eq!(ever_writable(&s.cases[3]), BTreeSet::from([200, 201]));
        for map in [vec![410, 420, 410], vec![420, 410, 400], vec![410, 420]] {
            let mut s = load();
            s.cases[0].dispatches.as_mut().unwrap()[1].bindings = Some(map);
            assert!(validate_suite(&s).is_err());
        }
        let mut s = load();
        s.cases[3].expected_writebacks.pop();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap()[1].bindings = None;
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn mixed_suite_rejects_missing_unreviewed_or_unselected_programs() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v5.json"))
                .unwrap()
        };
        validate_suite(&load()).unwrap();
        let mut s = load();
        s.cases[0].programs = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1].metal.sha256 = "0".repeat(64);
        assert!(validate_suite(&s).is_err());
        for program in [None, Some(0), Some(2)] {
            let mut s = load();
            s.cases[0].dispatches.as_mut().unwrap()[1].program = program;
            assert!(validate_suite(&s).is_err());
        }
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap().reverse();
        assert!(validate_suite(&s).is_err());
    }

    #[test]
    fn differing_layout_requires_selected_slot_numbers_access_and_lengths() {
        let load = || {
            serde_json::from_str::<Suite>(include_str!("../../../../conformance/suite-v6.json"))
                .unwrap()
        };
        let s = load();
        validate_suite(&s).unwrap();
        let dispatch = &s.cases[0].dispatches.as_ref().unwrap()[1];
        let slots = selected_slots(&s.cases[0], dispatch);
        assert_eq!(
            slots
                .iter()
                .map(|slot| (slot.binding, slot.access.as_str(), slot.length))
                .collect::<Vec<_>>(),
            [(1, "read", 4), (3, "read", 120), (7, "write", 120)]
        );
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1].buffer_slots = None;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .binding = 0;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[1]
            .access = "read_write".into();
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .length = 120;
        assert!(validate_suite(&s).is_err());
        let mut s = load();
        s.cases[0].dispatches.as_mut().unwrap()[1].bindings = Some(vec![400, 420, 410]);
        assert!(validate_suite(&s).is_err());
    }

    fn subset_suite() -> Suite {
        serde_json::from_str(include_str!("../../../../conformance/suite-v7.json")).unwrap()
    }

    #[test]
    fn subset_suite_requires_reviewed_pool_sizes_and_dispatch_sequences() {
        let suite = subset_suite();
        validate_suite(&suite).unwrap();
        assert_eq!(
            suite
                .cases
                .iter()
                .map(|case| case.buffers.len())
                .collect::<Vec<_>>(),
            [4, 5, 5]
        );
        for (case_index, dispatch_index, mappings) in [
            (0, 1, vec![400, 410]),
            (1, 3, vec![410, 430]),
            (2, 4, vec![400, 420, 410]),
            (0, 1, vec![400, 430, 410]),
        ] {
            let mut invalid = subset_suite();
            invalid.cases[case_index].dispatches.as_mut().unwrap()[dispatch_index].bindings =
                Some(mappings);
            assert!(validate_suite(&invalid).is_err());
        }
        for program in [None, Some(0), Some(2), Some(usize::MAX)] {
            let mut invalid = subset_suite();
            invalid.cases[0].dispatches.as_mut().unwrap()[1].program = program;
            assert!(validate_suite(&invalid).is_err());
        }
        let mut invalid = subset_suite();
        invalid.cases[0].buffers.pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[1].dispatches.as_mut().unwrap().pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[2].dispatches.as_mut().unwrap()[7].local = [1, 1, 1];
        assert!(validate_suite(&invalid).is_err());
    }

    #[test]
    fn subset_programs_require_exact_sources_and_selected_slot_layouts() {
        let suite = subset_suite();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance");
        for case in &suite.cases {
            for program in case_programs(case) {
                verified_source(&directory, &program.air).unwrap();
                verified_source(&directory, &program.metal).unwrap();
            }
        }
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1].metal.sha256 = "0".repeat(64);
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .binding = 8;
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].programs.as_mut().unwrap()[1]
            .buffer_slots
            .as_mut()
            .unwrap()[0]
            .access = "read_write".into();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[1].programs.as_mut().unwrap().pop();
        assert!(validate_suite(&invalid).is_err());
        let mut invalid = subset_suite();
        invalid.cases[0].buffers[3].length = 4;
        assert!(validate_suite(&invalid).is_err());
    }

    #[test]
    fn subset_expected_results_cover_late_and_temporarily_unbound_writes() {
        let suite = subset_suite();
        assert_eq!(
            ever_writable(&suite.cases[0]),
            BTreeSet::from([400, 410, 430])
        );
        assert_eq!(
            ever_writable(&suite.cases[1]),
            BTreeSet::from([400, 410, 430, 440])
        );
        for view in [400, 410, 430, 440] {
            let mut invalid = subset_suite();
            invalid.cases[1]
                .expected_writebacks
                .retain(|write| write.view != view);
            assert!(validate_suite(&invalid).is_err());
        }
    }

    fn fixture_pipelines(case: &Case) -> Vec<CompiledComputePipeline> {
        use metal_api_core::provider::{
            BufferBindingContract, FunctionIdentity, FunctionSource, PipelineContract, PipelineId,
        };
        case_programs(case)
            .into_iter()
            .enumerate()
            .map(|(index, program)| CompiledComputePipeline {
                device_epoch: DeviceEpoch::new(1),
                pipeline_id: PipelineId::new(index as u64 + 1),
                function: FunctionIdentity {
                    logical_digest: SemanticDigest::new("test", vec![1]).unwrap(),
                    entry_name: program.entry,
                    source: FunctionSource::MetalSource,
                },
                contract: PipelineContract {
                    dispatch_kind: DispatchKind::ThreadsExact,
                    required_local_size: None,
                    fixed_grid: Some(case.grid),
                    push_constant_offset: 0,
                    push_constant_bytes: 0,
                    buffer_bindings: program
                        .buffer_slots
                        .unwrap()
                        .into_iter()
                        .map(|slot| BufferBindingContract {
                            metal_binding: slot.binding,
                            access: match slot.access.as_str() {
                                "read" => BufferAccess::Read,
                                "write" => BufferAccess::Write,
                                "read_write" => BufferAccess::ReadWrite,
                                _ => unreachable!(),
                            },
                            footprint: FootprintProof::Static {
                                max_bytes: slot.length,
                            },
                        })
                        .collect(),
                    texture_bindings: Vec::new(),
                    shader_capabilities: Vec::new(),
                    translator_revision: None,
                },
                // A fixture compute pipeline carries the compute half only; a
                // render pass's entry is the one `register_render_pipeline`
                // hands back, and that one carries the render half.
                render: None,
            })
            .collect()
    }

    #[test]
    fn subset_trace_retains_all_initial_resources_and_binds_only_selected_views() {
        let suite = subset_suite();
        validate_suite(&suite).unwrap();
        for case in &suite.cases {
            let mut programs = fixture_pipelines(case);
            let views = case
                .buffers
                .iter()
                .map(|buffer| BufferView {
                    view_id: ViewId::new(buffer.view),
                    allocation_id: AllocationId::new(buffer.allocation),
                    metal_binding: buffer.binding,
                    offset: buffer.offset,
                    length: buffer.length,
                    access: BufferAccess::Unused,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(buffer.initial_bytes().unwrap()),
                })
                .collect::<Vec<_>>();
            let trace = case_trace(
                DeviceEpoch::new(1),
                &programs,
                case,
                1,
                &views,
                &[],
                &dispatch_sequence(case),
            )
            .unwrap();
            let resources = trace.serial_resources().unwrap();
            assert_eq!(resources.len(), case.buffers.len());
            assert_eq!(
                trace.passes[0]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .len(),
                3
            );
            assert_eq!(
                trace.passes[1]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .len(),
                2
            );
            assert_eq!(
                trace.passes[1]
                    .as_compute()
                    .expect("capture pass is a compute pass")
                    .buffers
                    .iter()
                    .map(|view| (view.metal_binding, view.view_id.get()))
                    .collect::<Vec<_>>(),
                [(4, 400), (9, 430)]
            );
            assert!(!trace.passes[1]
                .as_compute()
                .expect("capture pass is a compute pass")
                .buffers
                .iter()
                .any(|view| view.view_id.get() == 410));
            for (initial, collected) in views.iter().zip(&resources) {
                assert_eq!(initial.view_id, collected.view_id);
                assert_eq!(initial.source, collected.source);
                assert_eq!(initial.offset, collected.offset);
            }
            assert_eq!(
                resources
                    .iter()
                    .filter(|view| view.access.is_writable())
                    .map(|view| view.view_id.get())
                    .collect::<BTreeSet<_>>(),
                ever_writable(case)
            );
            programs[1].contract.buffer_bindings[0].access = BufferAccess::ReadWrite;
            assert!(case_trace(
                DeviceEpoch::new(1),
                &programs,
                case,
                1,
                &views,
                &[],
                &dispatch_sequence(case)
            )
            .is_err());
        }
    }

    #[test]
    fn copy_contract_checks_both_sparse_accesses_and_xyz_reach() {
        use metal_api_core::provider::{AffineAccess, AffineTerm};
        let mut pipeline = fixture_pipelines(&subset_suite().cases[0]).remove(1);
        for binding in &mut pipeline.contract.buffer_bindings {
            binding.footprint = FootprintProof::Affine {
                accesses: vec![AffineAccess {
                    base_offset: 0,
                    access_size: 4,
                    terms: vec![
                        AffineTerm { axis: 0, stride: 4 },
                        AffineTerm {
                            axis: 1,
                            stride: 20,
                        },
                        AffineTerm {
                            axis: 2,
                            stride: 60,
                        },
                    ],
                }],
            };
        }
        verify_copy_contract(&pipeline).unwrap();
        for index in 0..2 {
            let mut invalid = pipeline.clone();
            invalid.contract.buffer_bindings[index].footprint =
                FootprintProof::Static { max_bytes: 120 };
            assert!(verify_copy_contract(&invalid).is_err());
        }
        pipeline.contract.buffer_bindings[0].metal_binding = 0;
        assert!(verify_copy_contract(&pipeline).is_err());
    }
}
