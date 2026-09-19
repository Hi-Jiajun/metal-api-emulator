//! Offscreen render rail for the native provider (`research/docs/23` §6 Steps 6
//! and 7, §6 Step 3.3 for the vertex-input half).
//!
//! The rail answers the one question Step 6 owns on the Apple side: can the
//! native provider build a colour attachment, a render pass descriptor, a
//! two-entry render pipeline state and a full-screen-triangle draw out of one
//! reviewed MSL module, and read the attachment's texels back byte for byte.
//! Step 7 is the trace path over it: [`plan_trace`] decides ordering, the
//! reviewed allowlist, the attachment landing and the load op from values
//! alone, [`TraceRenderPlan::writeback`] and [`merge_writebacks`] put the texels
//! into the same writeback channel a compute pass uses, and `native.rs` calls
//! both around `encode_offscreen_render`.
//!
//! The vertex-input increment adds the caller-held half of the same contract
//! (`docs/23` §3.3): a pass may bind vertex streams and an index buffer whose
//! views carry their own bytes, which this rail reads out of the pass itself,
//! proves the footprint of on the host ([`plan_vertex_input`]) and translates
//! into an `MTLVertexDescriptor` plus `setVertexBuffer` /
//! `drawIndexedPrimitives`. Those bytes are uploaded here rather than through the
//! compute rail's pool, which only carries views a compute binding declares
//! (`docs/23` §3.6). Two reviewed modules exist for the two shapes — `vertex_id`
//! positions ([`REVIEWED_SOURCE`]) and `[[stage_in]]` positions
//! ([`REVIEWED_VERTEX_SOURCE`]) — and the pipeline's [`VertexLayout`] is what
//! selects which one may compile, so a trace cannot reach source the rail did not
//! review.
//!
//! The lease increment (`docs/23` §72, R3d) widens where those streams' bytes
//! may come from, to the same three arms the Vulkan rail resolves (R3c):
//! declared bytes ([`BufferSource::OwnedBytes`], unchanged), the provider's own
//! staged copy ([`BufferSource::StagedLease`]) and the owner's mapping
//! ([`BufferSource::BorrowedNoCopy`], mapped with `newBufferWithBytesNoCopy:`
//! instead of copied). [`resolve_render_input`] decides the arm before any
//! Metal object exists, [`RenderInputRetains`] holds every imported no-copy
//! lease from that decision until the pass's command buffer is terminal, and
//! every unreadable arm is refused by name.
//!
//! **The encoder body has still never run on an Apple GPU.** What is
//! different from the pre-flip state is the evidence: CI run `34774478149`
//! (`native-oracle-build`, commit `fb4f8da`) ran the oracle's `--render-selftest`
//! — the single-device check in `conformance/RENDER-CAPTURE.md` §5, over the
//! same reviewed module and the same `runRenderCase` a suite would use — on an
//! Apple Paravirtual device, read the 2x2 attachment back as `4080c0ff` four
//! times and printed `render_selftest: PASS`. That is the flip condition the
//! provider's render bits name (`native.rs`, `ProviderCapabilities`). The two
//! rules this rail shares with its sibling on the Vulkan side
//! (`crates/metal-api-vulkan/src/render.rs`, which does execute on Lavapipe and
//! the RTX 5060) are the fixed 2x2 extent and the byte/255 fragment constants
//! that keep 8-bit UNORM rounding away from a half-integer tie.
//!
//! Shape reuse: this rail consumes the core values — [`RenderPassDescriptor`]
//! and [`RenderPipelineContract`] — and re-runs their validators instead of
//! restating the first increment's field rules. What it adds is what only a
//! device can answer: the pixel-format mapping, the clear-value decoding per
//! format, and the fact that one reviewed MSL module is the only source it will
//! compile.
#![allow(dead_code)] // the trace path does not reach this rail yet (Step 4/7)

#[cfg(target_os = "macos")]
use crate::icb;
use crate::refusal;
use metal_api_core::provider::{
    AffineAccess, AffineTerm, AllocationId, AttachmentFormat, BorrowedLeaseRegistry, BorrowedView,
    BufferAccess, BufferSource, BufferView, BufferWriteback, ClearColor, ComputeTrace,
    ContractError, DepthResolveFilter, DepthStoreOp, DepthTest, DeviceEpoch, FieldValue,
    FootprintProof, IndexBufferBinding, IndexFormat, IndirectCommandDescriptor, LeaseId,
    LeaseRegistry, LoadOp, PipelineId, PresentDescriptor, PresentMode, ProviderError,
    ProviderErrorClass, ProviderPhase, RenderPassBlend, RenderPassCull, RenderPassDescriptor,
    RenderPipelineContract, RenderPipelineStage, ResourceTableSnapshot, SampleCount, SamplerPolicy,
    StencilResolveFilter, StencilTest, StoreOp, TextureAccess, TextureFormat, TextureSource,
    TextureType, TextureView, TracePass, VertexFormat, VertexLayout, VertexStep, ViewId,
    FULL_SCREEN_TRIANGLE_VERTICES,
};
use std::collections::BTreeMap;
use std::sync::Arc;

// The depth compare function is only named by the encoder body, which exists
// on macOS alone; the plan's own `DepthTest` travels unchanged everywhere.
#[cfg(target_os = "macos")]
use foreign_types::ForeignType;
#[cfg(target_os = "macos")]
#[allow(deprecated)] // `MTLFeatureSet` is the binding's only 2D-texture-size table.
use metal::{
    Buffer, CommandQueue, CompileOptions, DepthStencilDescriptor, Device,
    IndirectCommandBufferDescriptor, MTLBlendFactor, MTLBlendOperation, MTLClearColor,
    MTLColorWriteMask, MTLCommandBufferStatus, MTLCompareFunction, MTLCullMode, MTLFeatureSet,
    MTLIndexType, MTLIndirectCommandType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRegion, MTLResourceOptions, MTLSize, MTLStencilOperation, MTLStorageMode,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLVertexFormat, MTLVertexStepFunction,
    MTLViewport, MTLWinding, NSInteger, NSRange, NSUInteger,
    RenderPassDescriptor as MetalRenderPassDescriptor, RenderPipelineDescriptor,
    RenderPipelineState, StencilDescriptor, Texture, TextureDescriptor, VertexDescriptor,
};
#[cfg(target_os = "macos")]
use metal_api_core::provider::{
    BlendFactor, BlendOperation, ColorWriteMask, CompareFunction, CullMode as ContractCullMode,
    StencilCompare, StencilOp, Winding as ContractWinding,
};
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl};

/// The reviewed render fixture: one MSL module carrying both stage entries.
///
/// Exact byte equality is the same discipline the compute allowlist uses
/// (`lib.rs::bounded_contract`): a matching entry name cannot establish the
/// footprint of caller-supplied source, so the rail compiles these bytes and
/// nothing else.
pub(crate) const REVIEWED_SOURCE: &str =
    include_str!("../../../conformance/shaders/render_offscreen_2x2.metal");

/// The reviewed indexed fixture: the same two stages, but the vertex stage
/// reads its position from `[[stage_in]]`, i.e. from a caller-held vertex
/// stream, and the pass draws through an index buffer.
///
/// A second constant rather than a second entry pair in one module, because the
/// two shapes need different pipeline state: the `vertex_id` module carries no
/// `MTLVertexDescriptor` at all, while the indexed one is meaningless without
/// the descriptor the pass's stream declares. Both files are pinned by exact
/// bytes for the same reason [`REVIEWED_SOURCE`] is: a matching entry name
/// cannot establish the footprint of caller-supplied source.
pub(crate) const REVIEWED_VERTEX_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2.metal");

/// Vertex entry of the reviewed module: positions from `vertex_id`, no vertex
/// buffer (`VertexLayout::None`).
pub(crate) const VERTEX_ENTRY: &str = "render_fullscreen_triangle";

/// Fragment entry of the reviewed module: the fixed colour texel.
pub(crate) const FRAGMENT_ENTRY: &str = "render_solid_rgba8";

/// Vertex entry of the reviewed indexed module: the position arrives through
/// `[[stage_in]]`, i.e. through the vertex descriptor the pipeline carries.
pub(crate) const QUAD_VERTEX_ENTRY: &str = "render_quad_vertex";

/// The reviewed dual module: the indexed vertex entry once more, but a
/// fragment stage that writes two colour outputs, one per MRT location.
pub(crate) const REVIEWED_DUAL_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_dual.metal");

/// Fragment entry of the reviewed dual module: two `[[color(n)]]` outputs,
/// location 0 the vertex-input texel and location 1 the second MRT texel.
pub(crate) const DUAL_FRAGMENT_ENTRY: &str = "render_solid_rgba8_dual";

/// The reviewed single-channel float module: the indexed vertex stage plus a
/// one-component fragment stage (`conformance/shaders/quad_indexed_2x2_r32f.metal`).
/// An `r32float` attachment takes a one-component store, so the four-component
/// module the other single-output shapes use cannot stand in for it
/// (`research/docs/23` §3.3, v22).
pub(crate) const REVIEWED_R32F_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_r32f.metal");

/// Fragment entry of the reviewed single-channel float module.
pub(crate) const R32F_FRAGMENT_ENTRY: &str = "render_solid_r32f";

/// The reviewed four-location module: the indexed vertex stage plus a fragment
/// stage that writes all `MAX_COLOR_ATTACHMENTS` colour locations
/// (`conformance/shaders/quad_indexed_2x2_quad.metal`). The four texels are
/// pairwise distinct, so a capture that landed one target twice cannot pass
/// (`research/docs/23` §3.3, v24).
pub(crate) const REVIEWED_QUAD_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_quad.metal");

/// Fragment entry of the reviewed four-location module.
pub(crate) const QUAD_FRAGMENT_ENTRY: &str = "render_solid_rgba8_quad";

/// The reviewed three-location module: the indexed vertex stage plus a fragment
/// stage that writes three colour locations
/// (`conformance/shaders/quad_indexed_2x2_triple.metal`). Three is not the
/// ceiling, but it is its own shape (`research/docs/23` §3.3, v25).
pub(crate) const REVIEWED_TRIPLE_SOURCE: &str =
    include_str!("../../../conformance/shaders/quad_indexed_2x2_triple.metal");

/// Fragment entry of the reviewed three-location module.
pub(crate) const TRIPLE_FRAGMENT_ENTRY: &str = "render_solid_rgba8_triple";

/// The reviewed instanced module (`research/docs/23` §3.3, v31): a vertex stage
/// that reads the caller-held quad positions, a per-instance tint and the
/// `instance_id` builtin, plus a fragment stage that stores the forwarded tint.
pub(crate) const REVIEWED_INSTANCED_SOURCE: &str =
    include_str!("../../../conformance/shaders/instanced_quad_2x2.metal");

/// Vertex entry of the reviewed instanced module.
pub(crate) const INSTANCED_VERTEX_ENTRY: &str = "render_instanced_quad_vertex";

/// Fragment entry of the reviewed instanced module.
pub(crate) const INSTANCED_FRAGMENT_ENTRY: &str = "render_instanced_tint";

/// The reviewed depth module (`research/docs/23` §3.3, v36): a vertex stage
/// that reads a caller-held `float32x3` position (so the caller chooses each
/// triangle's depth) and a `float32x4` tint, plus a fragment stage that stores
/// the forwarded tint.
pub(crate) const REVIEWED_DEPTH_SOURCE: &str =
    include_str!("../../../conformance/shaders/depth_pair_4x4.metal");

/// Vertex entry of the reviewed depth module.
pub(crate) const DEPTH_VERTEX_ENTRY: &str = "render_depth_pair_vertex";

/// Fragment entry of the reviewed depth module.
pub(crate) const DEPTH_FRAGMENT_ENTRY: &str = "render_depth_pair_tint";

/// The reviewed zero-colour-attachment depth module (`research/docs/23` §3.3,
/// v46): the depth pair's vertex stage beside a **void** fragment stage, so a
/// pass with no colour attachment still tests and writes depth.
pub(crate) const REVIEWED_DEPTH_ONLY_SOURCE: &str =
    include_str!("../../../conformance/shaders/depth_only_4x4.metal");

/// Vertex entry of the reviewed zero-colour-attachment depth module.
pub(crate) const DEPTH_ONLY_VERTEX_ENTRY: &str = "render_depth_only_vertex";

/// Fragment entry of the reviewed zero-colour-attachment depth module.
pub(crate) const DEPTH_ONLY_FRAGMENT_ENTRY: &str = "render_depth_only_fragment";

/// The reviewed render-sampler module (`research/docs/23` §3.3, v70): the
/// milestone's `vertex_id` geometry with one `float2` varying holding the
/// geometry's own normalised coordinate, and a fragment stage that samples the
/// pass's own texture binding at that coordinate through a `constexpr`
/// nearest/clamp sampler.
///
/// The module shares the milestone's *shape* — `VertexLayout::None`, one
/// `Rgba8Unorm` location — so it is selected by its entry pair rather than by
/// the layout (`reviewed_module_for`): a registration has no pass to look at,
/// and the pass that binds a texture is exactly the one that names these two
/// entries.
pub(crate) const REVIEWED_SAMPLED_SOURCE: &str =
    include_str!("../../../conformance/shaders/render_sampled_4x4.metal");

/// The sampled textures the reviewed render-sampler module reads
/// (`research/docs/23` §3.3, v102).
///
/// The module spells one `texture2d<float>` argument, so one is the number of
/// pass bindings it executes; the contract admits more because a *translated*
/// fragment stage may name two or three, and this rail refuses the difference
/// by name (`render_texture_stage_unsupported`) instead of binding surfaces
/// nothing reads.
pub(crate) const REVIEWED_SAMPLED_TEXTURE_COUNT: usize = 1;

/// The Metal texture index the reviewed render-sampler module reads
/// (`research/docs/23` §3.3, v104).
///
/// The module's `texture2d<float>` argument is `[[texture(0)]]`, so a pass
/// whose own binding sits at another index has no reviewed module behind it:
/// the contract's list is indexed rather than positional since `v104`, which is
/// what makes this rail's window a statement rather than an assumption.
pub(crate) const REVIEWED_SAMPLED_TEXTURE_BINDING: u32 = 0;

/// Vertex entry of the reviewed render-sampler module.
pub(crate) const SAMPLED_VERTEX_ENTRY: &str = "render_sampled_quad_vertex";

/// Fragment entry of the reviewed render-sampler module.
pub(crate) const SAMPLED_FRAGMENT_ENTRY: &str = "render_sampled_texel";

/// The reviewed stage-buffer module (`research/docs/23` §83, R9g): the
/// native rail's sibling of the Vulkan pair the R9/R9c increments execute —
/// a vertex stage that reads its three positions from its own `[[buffer(0)]]`
/// argument (so the pipeline binds no `MTLVertexDescriptor`), and a fragment
/// stage whose one `[[buffer(0)]]` argument is the whole stored texel.
///
/// The module shares the milestone's shape — `VertexLayout::None`, one
/// `Rgba8Unorm` location — so it is selected by its entry pair, exactly as the
/// render sampler's pair is: a registration has no pass to look at, and the
/// pass that binds the two slots is exactly the one that names these entries.
pub(crate) const REVIEWED_STAGE_BUFFER_SOURCE: &str =
    include_str!("../../../conformance/shaders/render_stage_buffer_2x2.metal");

/// Vertex entry of the reviewed stage-buffer module.
pub(crate) const STAGE_BUFFER_VERTEX_ENTRY: &str = "render_stage_buffer_vertex";

/// Fragment entry of the reviewed stage-buffer module.
pub(crate) const STAGE_BUFFER_FRAGMENT_ENTRY: &str = "render_stage_buffer_tint";

/// The vertex stage's binding inside its own `[[buffer(N)]]` index space
/// (`setVertexBuffer(_:offset:index:)`), spelled by the reviewed module.
pub(crate) const STAGE_BUFFER_VERTEX_BINDING: u32 = 0;

/// The fragment stage's binding inside its own `[[buffer(N)]]` index space
/// (`setFragmentBuffer(_:offset:index:)`). The two stages' index spaces are
/// independent — the same fact `research/docs/23` §83.2 states for the
/// contract — so vertex `0` and fragment `0` are two different slots.
pub(crate) const STAGE_BUFFER_FRAGMENT_BINDING: u32 = 0;

/// Bytes the reviewed vertex stage reads: three `float2` positions.
pub(crate) const STAGE_BUFFER_VERTEX_BYTES: u64 = 24;

/// Bytes the reviewed fragment stage reads: one `float4` tint.
pub(crate) const STAGE_BUFFER_FRAGMENT_BYTES: u64 = 16;

/// The reviewed writable stage-buffer module (`research/docs/23` §92, R9k):
/// the native rail's sibling of the two shapes R9f executes on the Vulkan rail
/// through translated modules — the write half's `source`/`sink` pair and the
/// affine half's strided `positions[vertex_id]` read.
///
/// The module shares the milestone's shape — `VertexLayout::None`, one
/// `Rgba8Unorm` location — so it is selected by its entry pair, exactly as the
/// render sampler's and the R9g stage-buffer pair are: a registration has no
/// pass to look at, and the pass that binds these slots is exactly the one that
/// names these entries.
pub(crate) const REVIEWED_STAGE_BUFFER_WRITE_SOURCE: &str =
    include_str!("../../../conformance/shaders/render_stage_buffer_write_2x2.metal");

/// Vertex entry of the reviewed writable stage-buffer module: three positions
/// from the stage's own `[[buffer(0)]]`, read with a vertex-index stride.
pub(crate) const STAGE_BUFFER_WRITE_VERTEX_ENTRY: &str = "render_stage_buffer_write_vertex";

/// Fragment entry of the reviewed writable stage-buffer module: one readable
/// `source`, one writable `sink` and one read-write `accumulator`, each a
/// `float4` in its own `[[buffer(N)]]` argument (`research/docs/23` §92, R9k).
pub(crate) const STAGE_BUFFER_WRITE_FRAGMENT_ENTRY: &str = "render_stage_buffer_write_tint";

/// The writable module's vertex-stage binding inside that stage's own
/// `[[buffer(N)]]` index space: the positions the draw's vertices index into.
pub(crate) const STAGE_BUFFER_WRITE_VERTEX_BINDING: u32 = 0;

/// Bytes one vertex-index step of the readable positions reaches: two `float`
/// components, four bytes each, eight bytes apart — the two-access affine set
/// [`STAGE_BUFFER_WRITE_VERTEX_ACCESSES`] states.
pub(crate) const STAGE_BUFFER_WRITE_VERTEX_STRIDE: u64 = 8;

/// The writable module's fragment-stage binding of the readable `source`.
pub(crate) const STAGE_BUFFER_WRITE_SOURCE_BINDING: u32 = 0;

/// The writable module's fragment-stage binding of the write-only `sink`.
pub(crate) const STAGE_BUFFER_WRITE_SINK_BINDING: u32 = 1;

/// The writable module's fragment-stage binding of the read-write
/// `accumulator`.
pub(crate) const STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING: u32 = 2;

/// Bytes the readable `source` argument is (one `float4`), which is also the
/// extent the `sink` argument is written over and the extent the `accumulator`
/// argument is read and written over.
pub(crate) const STAGE_BUFFER_WRITE_TEXEL_BYTES: u64 = 16;

/// The reflected reach of the module's vertex-stage `[[buffer(0)]]` argument
/// (`research/docs/23` §3.3, v86): two four-byte accesses strided by
/// [`STAGE_BUFFER_WRITE_VERTEX_STRIDE`] over the draw's vertex index, axis 0 of
/// [`metal_api_core::provider::RENDER_AFFINE_AXES`]. The set is the one
/// `crates/metal-api-vulkan/tests/fixtures/render_stage_buffer_positions.vert.ll`
/// translates to for the same read, so the two rails pair one measurement.
pub(crate) const STAGE_BUFFER_WRITE_VERTEX_ACCESSES: [ReviewedAffineAccess; 2] = [
    ReviewedAffineAccess {
        base_offset: 0,
        access_size: 4,
        terms: &[ReviewedAffineTerm { axis: 0, stride: 8 }],
    },
    ReviewedAffineAccess {
        base_offset: 4,
        access_size: 4,
        terms: &[ReviewedAffineTerm { axis: 0, stride: 8 }],
    },
];

/// The byte reach one reviewed `[[buffer(N)]]` argument states
/// (`research/docs/23` §3.3, v86/v92).
///
/// The two arms are the two the contract's [`FootprintProof`] states, one
/// level down: a fixed extent the module's own read reaches, or the reflected
/// `constant + stride * invocation index` access set. A reviewed module whose
/// read depends on the draw states the affine arm, and the registration pairs
/// it with an affine declaration of the same set — a static declaration is a
/// ceiling the draw's own count can outgrow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReviewedStageBufferReach {
    /// A fixed byte extent, the shape every pre-R9k reviewed argument states.
    Static { max_bytes: u64 },
    /// The reflected affine accesses, in the contract's own terms.
    Affine {
        accesses: &'static [ReviewedAffineAccess],
    },
}

impl ReviewedStageBufferReach {
    /// The byte extent this reach covers over a draw's `[vertex, instance]`
    /// counts, by the same arithmetic the core contract evaluates its
    /// declarations with (`research/docs/23` §3.3, v86): the maximum over the
    /// accesses of `base + size + Σ (count - 1) * stride`. `None` means the
    /// expression overflows `u64`, which is a proof this rail refuses by name
    /// rather than flattens.
    fn required_bytes(&self, counts: [u64; 2]) -> Option<u64> {
        match self {
            Self::Static { max_bytes } => Some(*max_bytes),
            Self::Affine { accesses } => {
                let mut required = 0_u64;
                for access in *accesses {
                    let mut end = access.base_offset.checked_add(access.access_size)?;
                    for term in access.terms {
                        let count = counts.get(usize::from(term.axis))?;
                        let maximum = count.saturating_sub(1);
                        end = end.checked_add(maximum.checked_mul(term.stride)?)?;
                    }
                    required = required.max(end);
                }
                Some(required)
            }
        }
    }

    /// The reach as the contract's own access set, which is what the
    /// registration pairs an affine declaration against: a static extent is one
    /// term-less access over its whole width, exactly as a translated static
    /// range becomes one in the Vulkan rail's reflection
    /// (`crates/metal-api-vulkan/src/render.rs`, `reflected_affine_accesses`).
    fn accesses(&self) -> Vec<AffineAccess> {
        match self {
            Self::Static { max_bytes } => vec![AffineAccess {
                base_offset: 0,
                access_size: *max_bytes,
                terms: Vec::new(),
            }],
            Self::Affine { accesses } => accesses
                .iter()
                .map(|access| AffineAccess {
                    base_offset: access.base_offset,
                    access_size: access.access_size,
                    terms: access
                        .terms
                        .iter()
                        .map(|term| AffineTerm {
                            axis: term.axis,
                            stride: term.stride,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Whether this reach is the affine arm, which is what a vertex stage that
    /// reads its positions out of a stage buffer states — and therefore what
    /// bounds its `vertex_id` values instead of the reviewed full-screen
    /// triangle's own three-vertex count ([`FULL_SCREEN_TRIANGLE_VERTICES`]).
    pub(crate) const fn is_affine(&self) -> bool {
        matches!(self, Self::Affine { .. })
    }
}

/// One access of a reviewed argument's affine reach, in the contract's own
/// terms but as a `'static` value a module table can carry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReviewedAffineAccess {
    pub(crate) base_offset: u64,
    pub(crate) access_size: u64,
    pub(crate) terms: &'static [ReviewedAffineTerm],
}

/// One `stride * axis` term of a reviewed affine reach.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReviewedAffineTerm {
    pub(crate) axis: u8,
    pub(crate) stride: u64,
}

/// One normalized affine access: base offset, access size and the ascending
/// `(axis, stride)` terms (`research/docs/23` §3.3, v86).
type NormalizedAffineAccess = (u64, u64, Vec<(u8, u64)>);

/// The order- and duplicate-insensitive form of one affine access set: what the
/// two ends of the registration pairing compare.
///
/// The same normal form the Vulkan rail's translated pairing uses
/// (`crates/metal-api-vulkan/src/render.rs`, `affine_access_set`): a
/// declaration is free to state the module's accesses in any order, and the
/// term list inside one access is a set too, so both degrees of freedom are
/// dropped before "the same access set" becomes one comparison.
fn affine_access_set(accesses: &[AffineAccess]) -> Vec<NormalizedAffineAccess> {
    let mut normalized = accesses
        .iter()
        .map(|access| {
            let mut terms = access
                .terms
                .iter()
                .map(|term| (term.axis, term.stride))
                .collect::<Vec<_>>();
            terms.sort_unstable();
            (access.base_offset, access.access_size, terms)
        })
        .collect::<Vec<_>>();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

/// One `[[buffer(N)]]` argument a reviewed module reads, as the registration
/// gate pairs it with the contract's declaration (`research/docs/23` §83,
/// R9g).
///
/// The reviewed module is the native rail's half of the reflection a
/// translated stage carries: its argument list is fixed by the pinned source
/// bytes, so this table is what the registration pairs the contract's
/// `StageBufferBinding` list against — stage, index, access and the byte
/// reach the module's own read and write state ([`ReviewedStageBufferReach`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReviewedStageBufferSlot {
    /// Which stage reads the argument.
    pub(crate) stage: RenderPipelineStage,
    /// The `[[buffer(index)]]` number inside that stage's own index space.
    pub(crate) index: u32,
    /// How the stage uses the bytes: the read-only arms of the pre-R9k
    /// modules, and the write and read-write arms of the R9k writable module.
    pub(crate) access: BufferAccess,
    /// The reach the module's own read and write states; the contract's
    /// declaration has to cover a static arm and *be* an affine one.
    pub(crate) reach: ReviewedStageBufferReach,
}

/// One reviewed render module and the (vertex-input shape, colour-format
/// shape) pair it was written for.
///
/// The [`VertexLayout`] and the pipeline's [`RenderPipelineContract`]
/// `color_formats` together select the entry pair: a pipeline whose layout
/// binds streams and whose format list names one attachment can only be the
/// module whose vertex stage reads `[[stage_in]]` and whose fragment stage
/// returns one colour, a `VertexLayout::None` single-output pipeline only the
/// module that derives positions from `vertex_id`, and an indexed pipeline
/// whose two formats are both `Rgba8Unorm` only the dual module whose fragment
/// stage writes both locations. Both the entry pair and the source bytes of
/// that one module are then the allowlist, so neither a renamed entry nor an
/// edited file can execute.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReviewedModule {
    /// The exact module bytes this rail compiles.
    pub(crate) source: &'static str,
    /// The module's path, for refusals and the documentation's sake.
    pub(crate) path: &'static str,
    /// The vertex entry the module carries.
    pub(crate) vertex_entry: &'static str,
    /// The fragment entry the module carries.
    pub(crate) fragment_entry: &'static str,
    /// Whether the module's vertex stage reads a caller-held stream.
    pub(crate) binds_buffers: bool,
    /// The `[[buffer(N)]]` arguments the module's stages read, in canonical
    /// order (vertex bindings before fragment ones, ascending inside each
    /// stage), which is the order the contract states them in too. Empty for
    /// every module that reads none — the whole pre-R9g set.
    pub(crate) stage_buffers: &'static [ReviewedStageBufferSlot],
}

/// The reviewed modules, one per (vertex-input shape, colour-format shape)
/// pair this rail executes.
pub(crate) const REVIEWED_MODULES: [ReviewedModule; 12] = [
    ReviewedModule {
        source: REVIEWED_SOURCE,
        path: "conformance/shaders/render_offscreen_2x2.metal",
        vertex_entry: VERTEX_ENTRY,
        fragment_entry: FRAGMENT_ENTRY,
        binds_buffers: false,
        stage_buffers: &[],
    },
    ReviewedModule {
        source: REVIEWED_VERTEX_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    ReviewedModule {
        source: REVIEWED_DUAL_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_dual.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: DUAL_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    ReviewedModule {
        source: REVIEWED_R32F_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_r32f.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: R32F_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    ReviewedModule {
        source: REVIEWED_QUAD_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_quad.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: QUAD_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    ReviewedModule {
        source: REVIEWED_TRIPLE_SOURCE,
        path: "conformance/shaders/quad_indexed_2x2_triple.metal",
        vertex_entry: QUAD_VERTEX_ENTRY,
        fragment_entry: TRIPLE_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    // The reviewed instanced fixture (`research/docs/23` §3.3, v31): the one
    // module whose vertex stage reads `instance_id` and a per-instance stream,
    // and whose fragment stage stores the tint that stage forwarded.
    ReviewedModule {
        source: REVIEWED_INSTANCED_SOURCE,
        path: "conformance/shaders/instanced_quad_2x2.metal",
        vertex_entry: INSTANCED_VERTEX_ENTRY,
        fragment_entry: INSTANCED_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    // The reviewed depth fixture (`research/docs/23` §3.3, v36): the module
    // whose vertex stage reads a position whose z the caller chose, so the
    // depth state is what decides which triangle survives.
    ReviewedModule {
        source: REVIEWED_DEPTH_SOURCE,
        path: "conformance/shaders/depth_pair_4x4.metal",
        vertex_entry: DEPTH_VERTEX_ENTRY,
        fragment_entry: DEPTH_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    // The reviewed zero-colour-attachment depth fixture (`research/docs/23`
    // §3.3, v46): the same vertex stage with a fragment stage that generates no
    // output, which is the shape a pass with no colour attachment compiles.
    ReviewedModule {
        source: REVIEWED_DEPTH_ONLY_SOURCE,
        path: "conformance/shaders/depth_only_4x4.metal",
        vertex_entry: DEPTH_ONLY_VERTEX_ENTRY,
        fragment_entry: DEPTH_ONLY_FRAGMENT_ENTRY,
        binds_buffers: true,
        stage_buffers: &[],
    },
    // The reviewed render-sampler fixture (`research/docs/23` §3.3, v70): the
    // milestone's shape with its own entry pair, selected by the registration's
    // entries rather than by the layout it shares with module 0.
    ReviewedModule {
        source: REVIEWED_SAMPLED_SOURCE,
        path: "conformance/shaders/render_sampled_4x4.metal",
        vertex_entry: SAMPLED_VERTEX_ENTRY,
        fragment_entry: SAMPLED_FRAGMENT_ENTRY,
        binds_buffers: false,
        stage_buffers: &[],
    },
    // The reviewed stage-buffer fixture (`research/docs/23` §83, R9g): the one
    // module whose stages read `[[buffer(N)]]` arguments, and the only module
    // whose `stage_buffers` table is non-empty. It shares the milestone's
    // `vertex_id` + one-`Rgba8Unorm`-location shape, so its entry pair is what
    // selects it — exactly as the render sampler's pair is selected.
    ReviewedModule {
        source: REVIEWED_STAGE_BUFFER_SOURCE,
        path: "conformance/shaders/render_stage_buffer_2x2.metal",
        vertex_entry: STAGE_BUFFER_VERTEX_ENTRY,
        fragment_entry: STAGE_BUFFER_FRAGMENT_ENTRY,
        binds_buffers: false,
        stage_buffers: &[
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Vertex,
                index: STAGE_BUFFER_VERTEX_BINDING,
                access: BufferAccess::Read,
                reach: ReviewedStageBufferReach::Static {
                    max_bytes: STAGE_BUFFER_VERTEX_BYTES,
                },
            },
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_FRAGMENT_BINDING,
                access: BufferAccess::Read,
                reach: ReviewedStageBufferReach::Static {
                    max_bytes: STAGE_BUFFER_FRAGMENT_BYTES,
                },
            },
        ],
    },
    // The reviewed writable stage-buffer fixture (`research/docs/23` §92,
    // R9k): the one module whose stage buffers are not all read-only — its
    // fragment stage writes `[[buffer(1)]]` and reads and writes
    // `[[buffer(2)]]` — and the one whose vertex stage reads its positions with
    // a vertex-index stride instead of from a fixed three-record extent. It
    // shares the milestone's `vertex_id` + one-`Rgba8Unorm`-location shape, so
    // its entry pair is what selects it.
    ReviewedModule {
        source: REVIEWED_STAGE_BUFFER_WRITE_SOURCE,
        path: "conformance/shaders/render_stage_buffer_write_2x2.metal",
        vertex_entry: STAGE_BUFFER_WRITE_VERTEX_ENTRY,
        fragment_entry: STAGE_BUFFER_WRITE_FRAGMENT_ENTRY,
        binds_buffers: false,
        stage_buffers: &[
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Vertex,
                index: STAGE_BUFFER_WRITE_VERTEX_BINDING,
                access: BufferAccess::Read,
                reach: ReviewedStageBufferReach::Affine {
                    accesses: &STAGE_BUFFER_WRITE_VERTEX_ACCESSES,
                },
            },
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_SOURCE_BINDING,
                access: BufferAccess::Read,
                reach: ReviewedStageBufferReach::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_SINK_BINDING,
                access: BufferAccess::Write,
                reach: ReviewedStageBufferReach::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
            ReviewedStageBufferSlot {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING,
                access: BufferAccess::ReadWrite,
                reach: ReviewedStageBufferReach::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
        ],
    },
];

/// The reviewed module a pipeline's vertex-input shape and colour-format list
/// select, or `None` for a shape no module was reviewed for.
///
/// The single-output modules are format-agnostic: the pipeline's attachment
/// format is pipeline state, not module source, so one module serves every
/// admitted single format (`SUPPORTED_COLOR_FORMATS` below). The dual module
/// is reviewed for exactly two `Rgba8Unorm` locations, because its fragment
/// stage writes two byte strings that only those two locations decode the
/// reviewed way; any other two-format list, any list above this rail's cap
/// and any `vertex_id` pipeline with more than one location has no reviewed
/// module and is refused by [`review_contract`] and [`plan`].
pub(crate) fn reviewed_module(
    layout: &VertexLayout,
    color_formats: &[AttachmentFormat],
) -> Option<&'static ReviewedModule> {
    // The four-component colour modules are layout- and storage-class-agnostic:
    // the same `float4` store lands in whichever channel order *and* storage
    // width each attachment declares, because both are the `MTLPixelFormat`'s
    // and not the module's. One module therefore serves every admitted
    // four-component format — the two 8-bit UNORM layouts and the eight-byte
    // `Rgba16Float` (`research/docs/23` §3.3 v26, §78) — while the
    // single-channel float module stays the one format-specific stage and the
    // MRT modules stay reviewed for their own 8-bit format lists. Every shape
    // that fits no reviewed module is refused rather than matched
    // approximately.
    let colour4 = |format: &AttachmentFormat| {
        matches!(
            format,
            AttachmentFormat::Rgba8Unorm
                | AttachmentFormat::Bgra8Unorm
                | AttachmentFormat::Rgba16Float
        )
    };
    let unorm8 = |format: &AttachmentFormat| {
        matches!(
            format,
            AttachmentFormat::Rgba8Unorm | AttachmentFormat::Bgra8Unorm
        )
    };
    match (layout, color_formats) {
        // The `vertex_id` module stores one `float4`, so the shape it serves is
        // the four-component class: a single-channel `R32Float` attachment next
        // to this module would read back bytes the store's shape never
        // described, which is the pairing both rails refuse by name instead of
        // executing.
        (VertexLayout::None, [single]) if colour4(single) => Some(&REVIEWED_MODULES[0]),
        // The depth module is selected by the layout's own shape: one stream
        // with two attributes — a `float32x3` position and a `float32x4` tint —
        // is the shape its vertex stage reads, so it is matched before the
        // instanced and plain single-output arms
        // (`research/docs/23` §3.3, v36).
        //
        // An *empty* format list is that shape with no colour target at all:
        // the zero-colour-attachment depth pass, whose fragment stage generates
        // no output (`research/docs/23` §3.3, v46).
        (VertexLayout::Buffers(buffers), [])
            if buffers.len() == 1 && buffers[0].attributes.len() == 2 =>
        {
            Some(&REVIEWED_MODULES[8])
        }
        (VertexLayout::Buffers(buffers), [single])
            if unorm8(single) && buffers.len() == 1 && buffers[0].attributes.len() == 2 =>
        {
            Some(&REVIEWED_MODULES[7])
        }
        // The instanced module is selected by the layout's own step function,
        // not by the format list alone: a per-instance binding is the shape its
        // vertex stage was written for, so it is matched before the plain
        // single-output arm (`research/docs/23` §3.3, v31).
        (VertexLayout::Buffers(buffers), [single])
            if unorm8(single)
                && buffers
                    .iter()
                    .any(|buffer| buffer.step == VertexStep::PerInstance) =>
        {
            Some(&REVIEWED_MODULES[6])
        }
        (VertexLayout::Buffers(_), [AttachmentFormat::R32Float]) => Some(&REVIEWED_MODULES[3]),
        (VertexLayout::Buffers(_), [single]) if colour4(single) => Some(&REVIEWED_MODULES[1]),
        (VertexLayout::Buffers(_), [first, second]) if unorm8(first) && unorm8(second) => {
            Some(&REVIEWED_MODULES[2])
        }
        (VertexLayout::Buffers(_), [first, second, third])
            if unorm8(first) && unorm8(second) && unorm8(third) =>
        {
            Some(&REVIEWED_MODULES[5])
        }
        (VertexLayout::Buffers(_), formats)
            if formats.len() == usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap_or(usize::MAX)
                && formats.iter().all(unorm8) =>
        {
            Some(&REVIEWED_MODULES[4])
        }
        _ => None,
    }
}

/// The reviewed module one registration executes, entry names included
/// (`research/docs/23` §3.3, v70).
///
/// [`reviewed_module`] answers the *shape* question — which module a
/// (vertex-input layout, colour-format list) pair compiles — and that answer is
/// unique for every shape except the render sampler's, which shares the
/// milestone's `vertex_id` + one-`Rgba8Unorm`-location shape, and the
/// stage-buffer module's, which shares it too (`research/docs/23` §83, R9g).
/// Their own entry pairs are what tell the three apart, so this is the question
/// the registration gate and the plan both ask: a registration names entries,
/// and entries are the only thing that distinguishes a sampling or a
/// stage-buffer pipeline from a solid one before a pass exists to look at.
pub(crate) fn reviewed_module_for(
    contract: &RenderPipelineContract,
) -> Option<&'static ReviewedModule> {
    let module = reviewed_module(&contract.vertex_layout, &contract.color_formats)?;
    if contract.vertex_entry == module.vertex_entry
        && contract.fragment_entry == module.fragment_entry
    {
        return Some(module);
    }
    if contract.vertex_entry == SAMPLED_VERTEX_ENTRY
        && contract.fragment_entry == SAMPLED_FRAGMENT_ENTRY
        && matches!(contract.vertex_layout, VertexLayout::None)
        && contract.color_formats == [AttachmentFormat::Rgba8Unorm]
    {
        return Some(&REVIEWED_MODULES[9]);
    }
    if contract.vertex_entry == STAGE_BUFFER_VERTEX_ENTRY
        && contract.fragment_entry == STAGE_BUFFER_FRAGMENT_ENTRY
        && matches!(contract.vertex_layout, VertexLayout::None)
        && contract.color_formats == [AttachmentFormat::Rgba8Unorm]
    {
        return Some(&REVIEWED_MODULES[10]);
    }
    // The writable stage-buffer pair (`research/docs/23` §92, R9k) is selected
    // by its entry pair for the same reason: it shares the milestone's shape,
    // and the pass that binds a `Write` or `ReadWrite` slot is exactly the one
    // that names these two entries.
    if contract.vertex_entry == STAGE_BUFFER_WRITE_VERTEX_ENTRY
        && contract.fragment_entry == STAGE_BUFFER_WRITE_FRAGMENT_ENTRY
        && matches!(contract.vertex_layout, VertexLayout::None)
        && contract.color_formats == [AttachmentFormat::Rgba8Unorm]
    {
        return Some(&REVIEWED_MODULES[11]);
    }
    None
}

/// Colour attachments this rail executes today: two. The core contract admits
/// the full MRT shape (up to `metal_api_core::provider::MAX_COLOR_ATTACHMENTS`,
/// now 4) while this rail's capability bit stays at 2: the reviewed dual
/// module (`REVIEWED_DUAL_SOURCE`) writes exactly two locations, and no module
/// writes three or four, so a pass above two is refused rather than rendered
/// partially. The value is restated here because a capability value has to be
/// spelled by the provider that declares it (`research/docs/23` §4.2).
pub(crate) const MAX_COLOR_ATTACHMENTS: u32 = 4;

/// Vertex streams one render pass may bind. The same value the core contract
/// states (`metal_api_core::provider::MAX_VERTEX_BUFFERS`), restated for the
/// same reason [`MAX_COLOR_ATTACHMENTS`] is: a capability value belongs to the
/// provider that declares it (`research/docs/23` §3.3, §4.2).
pub(crate) const MAX_VERTEX_BUFFERS: u32 = metal_api_core::provider::MAX_VERTEX_BUFFERS as u32;

/// The largest attachment extent this rail's review covers, per axis.
///
/// The milestone's window was 4×4 (`research/docs/23` §1.3): a size the
/// reviewed modules execute and a conformance case measured. R1b
/// (`research/docs/23` §70) widens it to the first family a real frame needs —
/// the reviewed fixtures pin 16×16 and the 64×64 boundary — and the declared
/// window is this ceiling clamped by the device's own 2D texture limit
/// ([`attachment_dimension_window`], [`device_attachment_dimension_limit`]).
/// Widening the ceiling further is a deliberate change that owes a boundary
/// fixture at the new value. R5a (`research/docs/23` §73) takes that step for
/// the desktop sizes the guest profile measured: the 2048×2048 boundary, which
/// every Apple GPU family's 16384-texel 2D limit covers.
pub(crate) const REVIEWED_ATTACHMENT_CEILING: [u64; 2] = [2048, 2048];

/// The attachment window a device with this 2D texture limit declares.
///
/// R1b (`research/docs/23` §70): per axis, the smaller of the reviewed ceiling
/// above and the device's own limit. The capability snapshot publishes this
/// value, so core admission refuses a wider attachment by name
/// (`attachment_dimension_limit`, carrying the maximum it crossed) instead of
/// letting the rail plan a texture the device cannot open. Pure so the clamp is
/// testable on a host without Metal, exactly as the Vulkan rail's
/// `attachment_dimension_window` is.
pub(crate) fn attachment_dimension_window(device_2d_texture_limit: u64) -> [u64; 2] {
    [
        device_2d_texture_limit.min(REVIEWED_ATTACHMENT_CEILING[0]),
        device_2d_texture_limit.min(REVIEWED_ATTACHMENT_CEILING[1]),
    ]
}

/// The 2D texture ceiling every admissible Apple GPU family states.
///
/// `metal::MTLFeatureSet::max_2d_texture_size` answers 16384 for every macOS
/// GPU family, and this rail admits Apple4+ devices only; the fallback in
/// [`device_attachment_dimension_limit`] uses this value so a device that
/// answers no legacy macOS feature set still declares the documented ceiling
/// rather than 0.
pub(crate) const APPLE_2D_TEXTURE_CEILING: u64 = 16_384;

/// The stage-buffer bits the provider declares (`research/docs/23` §83, §92,
/// R9g/R9k).
///
/// The bits name this rail's own window: the reviewed
/// `conformance/shaders/render_stage_buffer_2x2.metal` module, whose vertex
/// stage reads its three positions and whose fragment stage its one `float4`
/// from their own `[[buffer(0)]]` arguments — the slots the encoder fills with
/// `setVertexBuffer(_:offset:index:)` and `setFragmentBuffer(_:offset:index:)`
/// — and the writable arm `conformance/shaders/render_stage_buffer_write_2x2.metal`
/// adds beside it (the readable source, the write-only sink and the read-write
/// accumulator).
///
/// Flip evidence (`research/docs/23` §83, §92): two `native-oracle-build`
/// readings on Apple Paravirtual devices, both from the "…when a Metal device
/// is eligible" steps whose probe answered `eligible: true` rather than the
/// `SKIP` an ineligible runner prints. The first is CI run `35225235455` (commit
/// `6b86a50`, archived in `evidence/apple-native-stage-buffer-2026-09-17/`),
/// whose `--stage-buffer-selftest` printed
/// `stage_buffer_selftest: PASS (stage_buffer_positions_2x2 4080c0ff… /
/// 00ff00ff… / 4080c0ff×4)` — the three runs whose covered texel moved with the
/// bytes each stage's own `[[buffer(0)]]` carried. The second is CI run
/// `35233984141` (job `native-oracle-build`, commit `81f9599`, this branch's own
/// base), which read both halves on one device — macOS 15.7.9 (Build 24G830):
/// the same `stage_buffer_selftest: PASS (…)` line and
/// `stage_buffer_write_selftest: PASS (stage_buffer_write_2x2 frames=[…]
/// sinks=[…] accumulators=[…])`, the writable module landing the frame, the
/// sink and the accumulator. The same job's native provider capture still
/// skipped both suite cases ("…binds no stage buffers; the case is marked for
/// vulkan"), which is the pre-flip state this increment removes. Before the
/// flip both fields were at their defaults, so core admission refused a
/// stage-buffer pass with `render_stage_buffer_unsupported` instead of
/// executing a path no device reading had confirmed.
///
/// What stays refused after the flip is the shape the reviewed modules do not
/// cover, and each refusal is by name rather than a silent drop: a pipeline
/// whose stages come from a *translated* AIR pair ([`reviewed_module_for`]
/// selects the reviewed MSL modules alone, so the pair has no module to pair
/// its declarations with), a declaration the selected module does not read
/// (`render_stage_buffer_stage_unsupported`), an access or extent the module's
/// own argument disagrees with (`render_stage_reflection_mismatch`), an
/// `Unbounded` footprint, and an affine reach past the declared window. What
/// the flip does *not* claim is a device reading for every shape the rail now
/// executes: the writable pair's reading is the `--stage-buffer-write-selftest`
/// run rather than a suite case (its suite case pins AIR), and a presenting
/// pass that binds a writable slot executes without a device reading of its
/// own.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StageBufferCapabilityBits {
    pub(crate) supports_render_stage_buffers: bool,
    pub(crate) max_render_stage_buffers: u32,
    /// The per-stage window this rail states, or `0` for the list-bound-only
    /// reading (`research/docs/23` §117, E-SB2). The reviewed modules bind one
    /// slot per stage, so this rail declares none: a pair that spreads thirteen
    /// declarations over its stages is refused by name by core admission rather
    /// than executed against slots no Apple reading measured.
    pub(crate) max_render_stage_buffers_per_stage: u32,
    /// Whether this rail executes a pair whose two stages each read a
    /// `[[buffer(n)]]` argument of the same Metal index (`research/docs/23`
    /// §3.3, E-TX9). Declared beside the pair above because it is the same
    /// face: the reviewed pair already binds its vertex stage's bytes through
    /// `setVertexBuffer(_:offset:index:)` at set 1 and its fragment stage's
    /// through `setFragmentBuffer(_:offset:index:)` at set 2, which is exactly
    /// the arrangement the folded shape needs, and the Apple device readings
    /// (`--stage-buffer-selftest`, `--stage-buffer-write-selftest`) are the
    /// readings for the shape those two stages state.
    pub(crate) supports_render_stage_buffer_namespace_split: bool,
    /// Whether this rail executes a stage buffer whose declared footprint is
    /// `FootprintProof::BindingRange` — the reach the translation could not
    /// state (`research/docs/23` §3.3, E-SB3). This rail keeps the default:
    /// its reviewed pair reads each `[[buffer(N)]]` argument at the extent its
    /// own pinned bytes state ([`ReviewedStageBufferReach`]), and executing a
    /// declaration nothing measured would mean binding an argument whose read
    /// window the module's own source does not carry — so a registration that
    /// states the arm is refused by name
    /// (`render_stage_buffer_binding_range_unsupported`) rather than run
    /// against a window the rail cannot account for. The Vulkan rail is the
    /// half that publishes the bit, because a `STORAGE_BUFFER` descriptor's
    /// whole range is a window it can bind with the device's
    /// `robustBufferAccess` enabling it.
    pub(crate) supports_render_stage_buffer_binding_range: bool,
}

/// The one spelling of the stage-buffer bits, so the macOS snapshot and the
/// flip condition cannot drift apart — the shape `crate::heap` and
/// `crate::icb` state their own pending bits in.
pub(crate) fn stage_buffer_capability_bits() -> StageBufferCapabilityBits {
    StageBufferCapabilityBits {
        supports_render_stage_buffers: true,
        max_render_stage_buffers: MAX_RENDER_STAGE_BUFFERS,
        max_render_stage_buffers_per_stage: 0,
        // The reviewed pair binds the two stages at different slots, so the
        // folded shape is the arrangement this rail already executes. No new
        // native implementation arrives with the bit: the declaration names
        // the shape the two device readings above measured.
        supports_render_stage_buffer_namespace_split: true,
        // The whole-binding arm stays at the fail-closed default
        // (`research/docs/23` §3.3, E-SB3): this rail's reviewed modules read
        // each argument at the extent their own source states, so the arm has
        // no route here and the registration is refused by name.
        supports_render_stage_buffer_binding_range: false,
    }
}

/// Colour formats this rail can build an `MTLTexture` and a pipeline state from
/// — the core contract's admitted set, without `R32Uint` ([`pixel_format`]
/// refuses that one).
///
/// `Rgba16Float` arrived with the gate-3 census (`research/docs/23` §78): the
/// desktop load's `MTLPixelFormatRGBA16Float` attachments are the same
/// four-component class the reviewed single-output modules store, and the
/// device side is Metal itself — `MTLPixelFormat::RGBA16Float` is core Metal 2,
/// so there is no per-device probe to ask, unlike the Vulkan rail's
/// `vkGetPhysicalDeviceImageFormatProperties` question.
pub(crate) const SUPPORTED_COLOR_FORMATS: [AttachmentFormat; 4] = AttachmentFormat::ADMITTED;

/// The render bits the provider declares, in one value so the macOS capability
/// snapshot (`native.rs`) and the host-side unit tests cannot drift.
///
/// Every field is the rail's own limit, so capability admission and this rail
/// agree by construction; the unit tests assert that agreement against core's
/// [`metal_api_core::provider::ProviderCapabilities::admit`], which is the only
/// place the two could otherwise diverge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RenderCapabilityBits {
    pub(crate) supports_render_passes: bool,
    pub(crate) max_color_attachments: u32,
    pub(crate) max_attachment_dimension: [u64; 2],
    pub(crate) supported_color_formats: Vec<AttachmentFormat>,
    /// The superset fragment interface (2026-09-20, the third door behind
    /// census v46's `stage_buffer_footprint` bucket). Declared beside the three
    /// attachment-side fields it narrows, and always `false` here: Apple has no
    /// oracle for a module that stores a colour location the pass does not
    /// attach, and this rail's reviewed-module table selects a stage by the
    /// colour format list's *exact* shape (`reviewed_module`), so the shape
    /// matches no arm and is refused by name.
    pub(crate) supports_render_fragment_output_superset: bool,
    /// The 16-bit shader capability pair (2026-09-20, census v48's LPF
    /// pipeline). Always `false` here, and it is this rail's own answer rather
    /// than a missing Apple-side reading: the reviewed modules are MSL text
    /// this rail authored, none of them narrows a float to `half` and reads its
    /// bits back, and this rail has no translation front end at all — a module
    /// whose *SPIR-V* declares `Float16`/`Int16` is a module no `ReviewedModule`
    /// arm was written for, so a registration that names one is refused by name
    /// (`native_render_source_not_reviewed`). Declaring the bit would promise a
    /// shape this rail's module table refuses, so the snapshot keeps the
    /// contract's fail-closed default.
    pub(crate) supports_render_half_capabilities: bool,
    /// Render-sampler bits, declared next to the render bits for the same
    /// reason: the snapshot and the rail cannot disagree about what this
    /// provider samples (`research/docs/23` §3.3, v70). The three fields come
    /// from [`render_texture_capability_bits`], so their flip condition is one
    /// observation rather than a second set of inline literals that could
    /// drift from the comment.
    pub(crate) supports_render_texture_sampling: bool,
    pub(crate) max_render_textures: u32,
    /// The per-stage sampled-texture window, declared beside the list bound it
    /// narrows (`research/docs/23` §3.3, E-TC1). Always `0` here: the reviewed
    /// modules sample one texture argument, so the rail states no wider window
    /// and a fragment stage past the list bound is refused by name.
    pub(crate) max_render_textures_per_stage: u32,
    pub(crate) supported_render_texture_formats: Vec<TextureFormat>,
    /// The gathered-extent shape's bit (`research/docs/23` §3.3, E-TX10),
    /// declared beside the three render-sampler fields it narrows.
    pub(crate) supports_render_texture_gathered_extent: bool,
    /// The gathered extent's *no-copy* arm (`research/docs/23` §111, E-TX12),
    /// declared beside the bit above for the same reason: it is the other half
    /// of the same face, and a rail may execute one without the other.
    pub(crate) supports_render_texture_gathered_extent_no_copy: bool,
    /// The landing-view arm (`research/docs/23` §115 之后的增量，E-TX13). Kept
    /// beside the two gathered-extent bits so the snapshot has one place that
    /// answers "does this rail execute the arm" — this one always says no.
    pub(crate) supports_render_attachment_landing_view: bool,
    /// The kept-frame landing entry (`research/docs/23` §115 之后的增量，
    /// E-TX14/R4b). Kept beside the landing-view bit for the same reason, and
    /// also always `false` here: this rail has no owner-window write route, so
    /// it refuses the entry by name.
    pub(crate) supports_render_kept_frame_landing: bool,
    /// The texel space (2026-09-19, census v43's `texture_state` axis). Kept
    /// beside the render-sampler bits for the same reason, and also always
    /// `false` here: the reviewed MSL modules spell one `constexpr sampler` in
    /// the normalized space and take no `[[sampler(n)]]` argument at all, so
    /// the rail refuses such a pass by name.
    pub(crate) supports_render_pixel_coordinate_sampler: bool,
    /// The pass-entry snapshot arm (`research/docs/23` §118, E-TX15). Kept
    /// beside the two bits above for the same reason, and always `false` here:
    /// Apple has no oracle for a fragment reading the attachment it writes, so
    /// this rail refuses the arm by name.
    pub(crate) supports_render_pass_entry_snapshot: bool,
    /// Present bits, declared next to the render bits for the same reason: the
    /// snapshot and the rail cannot disagree about what this provider runs.
    /// The four fields come from [`present_capability_bits`], so their flip
    /// condition is one observation (`research/docs/24` §6 Step 7) rather than
    /// a second set of inline literals that could drift from the comment.
    pub(crate) supports_presentation: bool,
    pub(crate) max_present_targets: u32,
    pub(crate) supported_present_modes: Vec<PresentMode>,
    pub(crate) max_present_image_count: u32,
}

/// The present bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side unit tests cannot drift.
///
/// Split out from the render bits because the flip condition is a separate
/// single-device observation: the Swift oracle's `--present-selftest` on an
/// Apple GPU, exactly as the render bits rest on `--render-selftest`
/// (`research/docs/24` §6 Step 7).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PresentCapabilityBits {
    pub(crate) supports_presentation: bool,
    pub(crate) max_present_targets: u32,
    pub(crate) supported_present_modes: Vec<PresentMode>,
    pub(crate) max_present_image_count: u32,
}

/// The first presentation increment's target cap, spelled once so the snapshot
/// and its tests cannot drift from core's value (`research/docs/24` §3.1).
pub(crate) const MAX_PRESENT_TARGETS: u32 = metal_api_core::provider::MAX_PRESENT_TARGETS as u32;

/// The first presentation increment's image cap, same rule as
/// [`MAX_PRESENT_TARGETS`].
pub(crate) const MAX_PRESENT_IMAGE_COUNT: u32 = metal_api_core::provider::MAX_PRESENT_IMAGE_COUNT;

/// The stage-buffer cap the first stage-buffer increment declares, same rule as
/// [`MAX_PRESENT_TARGETS`]: core's own ceiling, spelled once so the snapshot
/// and its tests cannot drift from the contract's value
/// (`research/docs/23` §3.3, v83).
pub(crate) const MAX_RENDER_STAGE_BUFFERS: u32 =
    metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS as u32;

/// The largest instance count the instancing increment executes
/// (`research/docs/23` §3.3, v31). Same rule as the Vulkan rail's ceiling: the
/// reviewed fixture draws two instances and the declared window is four.
pub(crate) const MAX_RENDER_INSTANCES: u32 = 4;

/// The render bits this provider declares as of the Step 7 flip, with the
/// attachment window R1b made device-gated.
///
/// Flip evidence (`research/docs/23` §4.2, §6 Steps 6-7;
/// `conformance/RENDER-CAPTURE.md` §5): CI run `34774478149` — job
/// `native-oracle-build` at commit `fb4f8da` — ran `native-oracle
/// --render-selftest` on an Apple Paravirtual device, whose report and log read
/// `4080c0ff` four times and ended with `render_selftest: PASS`. A green run
/// whose log said `SKIP` would not be that evidence, because it reports a runner
/// without an eligible device rather than an executed reviewed path.
///
/// The attachment window is the reviewed ceiling clamped by the device's own
/// 2D texture limit (`research/docs/23` §70): the snapshot and the rail's
/// device-half refusal read the same number
/// ([`device_attachment_dimension_limit`] through the provider), so a device
/// narrower than the review declares its own limit instead of the ceiling.
pub(crate) fn capability_bits(device_2d_texture_limit: u64) -> RenderCapabilityBits {
    let present = present_capability_bits();
    let render_texture = render_texture_capability_bits();
    RenderCapabilityBits {
        supports_render_passes: true,
        max_color_attachments: MAX_COLOR_ATTACHMENTS,
        max_attachment_dimension: attachment_dimension_window(device_2d_texture_limit),
        supported_color_formats: SUPPORTED_COLOR_FORMATS.to_vec(),
        // The superset fragment interface (2026-09-20, the third door behind
        // census v46's `stage_buffer_footprint` bucket) is this rail's own
        // boundary rather than a missing measurement: `reviewed_module` selects
        // its MSL module by the layout's and the colour format list's *exact*
        // shape, so a module that stores a location the pass does not attach
        // matches no arm and the registration is refused by name. Declaring
        // the bit would promise a shape this rail's module table refuses, so it
        // keeps the contract's fail-closed default.
        supports_render_fragment_output_superset: false,
        // The 16-bit shader capability pair keeps the same fail-closed default
        // (2026-09-20, census v48's LPF pipeline), and for the same kind of
        // reason: this rail executes MSL modules it authored, none of which
        // narrows a float to `half`, and it has no SPIR-V front end that could
        // read a translated module's `OpCapability Float16`/`Int16` at all.
        supports_render_half_capabilities: false,
        supports_render_texture_sampling: render_texture.supports_render_texture_sampling,
        max_render_textures: render_texture.max_render_textures,
        // The window is the bits' own field rather than a literal here
        // (`research/docs/23` §3.3, E-TC1): the rail states it once, beside the
        // reviewed count its own walk applies.
        max_render_textures_per_stage: render_texture.max_render_textures_per_stage,
        supported_render_texture_formats: render_texture.supported_render_texture_formats,
        supports_render_texture_gathered_extent: render_texture
            .supports_render_texture_gathered_extent,
        supports_render_texture_gathered_extent_no_copy: render_texture
            .supports_render_texture_gathered_extent_no_copy,
        // The landing-view arm (`research/docs/23` §115 之后的增量，E-TX13) is
        // refused by this rail's own store walk: its owner-window channel is an
        // input channel with no route that writes one, so the snapshot keeps the
        // fail-closed default beside the two bits above.
        supports_render_attachment_landing_view: false,
        supports_render_kept_frame_landing: false,
        // The texel space has no module on this rail either (2026-09-19,
        // census v43's `texture_state` axis): the reviewed MSL modules state one
        // `constexpr sampler` in the normalized space, so the snapshot keeps the
        // fail-closed default and core admission refuses such a pass by name.
        supports_render_pixel_coordinate_sampler: false,
        // The pass-entry snapshot arm (`research/docs/23` §118, E-TX15) is
        // refused by this rail's own texture walk: Apple has no oracle for a
        // fragment reading the attachment it writes, so the snapshot keeps the
        // fail-closed default beside the two bits above.
        supports_render_pass_entry_snapshot: false,
        supports_presentation: present.supports_presentation,
        max_present_targets: present.max_present_targets,
        supported_present_modes: present.supported_present_modes,
        max_present_image_count: present.max_present_image_count,
    }
}

/// The render-sampler bits the provider declares (`research/docs/23` §3.3,
/// v70).
///
/// The bits name this rail's own window: one `rgba8_unorm` binding whose
/// texture shares the render area's extent, uploaded into a shared-storage
/// `MTLTexture` and sampled through the reviewed fragment stage's `constexpr`
/// nearest/clamp sampler. Flip evidence: the reviewed
/// `conformance/shaders/render_sampled_4x4.metal` module and the plan gates the
/// host-side tests below pin; the macOS `--render-selftest` run of the same
/// case is what the CI job's oracle reports next.
pub(crate) fn render_texture_capability_bits() -> RenderTextureCapabilityBits {
    RenderTextureCapabilityBits {
        supports_render_texture_sampling: true,
        // The *rail's* window, not the contract's cap
        // (`research/docs/23` §3.3, v102): this rail executes the reviewed
        // MSL module, which samples one texture argument, so one is how many
        // bindings it can execute today. Core's own ceiling
        // ([`metal_api_core::provider::MAX_RENDER_TEXTURES`]) admits the wider
        // declarations a translated module names; declaring them here would
        // promise a shape this rail's plan refuses by name.
        max_render_textures: REVIEWED_SAMPLED_TEXTURE_COUNT as u32,
        // The per-stage window stays undeclared (`research/docs/23` §3.3,
        // E-TC1): the reviewed module samples one texture argument, so the
        // list bound above is the whole rule and a stage that declares
        // thirteen is refused by name by core admission instead of being
        // executed against slots no Apple reading sized. The rail's own walk
        // states the same window one face over
        // (`REVIEWED_SAMPLED_TEXTURE_COUNT`) when a directly-constructed
        // request skips admission.
        max_render_textures_per_stage: 0,
        supported_render_texture_formats: SUPPORTED_RENDER_TEXTURE_FORMATS.to_vec(),
        // The gathered extent stays refused (`research/docs/23` §3.3, E-TX10):
        // this rail answers *every* sampled source of another extent with
        // `render_texture_extent_unsupported` (the rail's own texture walk),
        // and Apple has no oracle for the shape — the reviewed module samples
        // the render area's own texel centres and declares no coordinate of its
        // own. Declaring the bit would promise a shape this rail's plan refuses
        // by name, so it keeps the consumer's fail-closed default.
        supports_render_texture_gathered_extent: false,
        // The gathered extent's no-copy arm (`research/docs/23` §111, E-TX12):
        // this rail has no reading of the shape at all — every sampled source of
        // another extent is refused by name, and the reviewed module declares no
        // coordinate of its own, so there is neither an Apple oracle for the
        // destination grid's index nor a Metal primitive that expresses it. The
        // bit keeps the consumer's fail-closed default.
        supports_render_texture_gathered_extent_no_copy: false,
        supports_render_kept_frame_landing: false,
        supports_render_attachment_landing_view: false,
        // The pass-entry snapshot arm (`research/docs/23` §118, E-TX15): this
        // rail has no Apple oracle for the shape at all, so the bit keeps the
        // consumer's fail-closed default.
        supports_render_pass_entry_snapshot: false,
    }
}

/// The render-sampler bits, in the same shape [`PresentCapabilityBits`] uses:
/// one value so the macOS snapshot and the host-side tests cannot drift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RenderTextureCapabilityBits {
    pub(crate) supports_render_texture_sampling: bool,
    pub(crate) max_render_textures: u32,
    /// The per-stage window this rail states, or `0` for the list-bound-only
    /// reading (`research/docs/23` §3.3, E-TC1). The reviewed modules sample
    /// one texture argument, so this rail declares none: a fragment stage that
    /// declares a wider list — thirteen declarations in the widest shape the
    /// census has read — is refused by name by core admission rather than
    /// executed against slots no Apple reading measured.
    pub(crate) max_render_textures_per_stage: u32,
    pub(crate) supported_render_texture_formats: Vec<TextureFormat>,
    /// Whether this rail executes a sampled source whose extent is not the
    /// render area's (`research/docs/23` §3.3, E-TX10). `false` here is the
    /// declaration, not a missing measurement: the rail's texture walk refuses
    /// the shape by name.
    pub(crate) supports_render_texture_gathered_extent: bool,
    /// Whether this rail executes the gathered extent's *no-copy* arm — a
    /// sampled source whose extent is not the render area's and whose bytes are
    /// the owner's no-copy window (`research/docs/23` §111, E-TX12). `false`
    /// here is the declaration, not a missing measurement: the rail refuses
    /// every source of another extent by name, and Apple has no oracle for the
    /// destination grid's index.
    pub(crate) supports_render_texture_gathered_extent_no_copy: bool,
    /// Whether this rail's snapshot declares the landing-view arm. It never
    /// does: `store_action` refuses the arm by name.
    pub(crate) supports_render_attachment_landing_view: bool,
    /// Whether this rail's snapshot declares the kept-frame landing entry
    /// (`research/docs/23` §115 之后的增量，E-TX14/R4b). It never does: the
    /// entry is refused by name before any plan exists.
    pub(crate) supports_render_kept_frame_landing: bool,
    /// Whether this rail's snapshot declares the pass-entry snapshot arm
    /// (`research/docs/23` §118, E-TX15). It never does: the arm is refused by
    /// name before any plan exists.
    pub(crate) supports_render_pass_entry_snapshot: bool,
}

/// The first render-sampler increment's binding cap, spelled once so the
/// snapshot and its tests cannot drift from core's value
/// (`research/docs/23` §3.3, v70/v102).
///
/// This is the *contract's* ceiling, which the pass-side shape rules restate
/// for a directly-constructed pass. The snapshot the provider publishes
/// declares the rail's own window beside it
/// ([`REVIEWED_SAMPLED_TEXTURE_COUNT`]), because those are two different
/// questions: the contract admits what a translated module may name, the rail
/// declares what its reviewed modules execute.
pub(crate) const MAX_RENDER_TEXTURES: u32 = metal_api_core::provider::MAX_RENDER_TEXTURES as u32;

/// The texture formats this rail's reviewed sampling table names: one
/// `rgba8_unorm` texel.
///
/// The Vulkan rail widened the same table to every lane the contract's
/// [`TextureFormat::RENDER_SAMPLED`] names beyond the first: the second 8-bit
/// byte order, the narrow lanes, the eight-byte half-float lane and the two
/// single-component float lanes (`research/docs/23` §107/§113/§119). This
/// rail's table stays at the one format its review covers, so a `bgra8_unorm`,
/// `r8_unorm`, `rg8_unorm`, `rgba16_float`, `r32_float` or `r16_float` sampled
/// texture is refused here by name at admission
/// (`render_texture_format_unsupported`) instead of being executed as an
/// unmeasured claim. The reviewed MSL sibling samples a `texture2d<float>` —
/// the pixel format is the plan's own fact — so the mechanical widening would
/// be the pixel format's own name, and the Apple-side self-test reading is what
/// would have to land with it, exactly as the present/stage-buffer flips state.
///
/// The one-dimensional arm has a second reason to stay refused on this rail
/// (2026-09-19, census b10's `texture_shape` bucket): the reviewed MSL module
/// samples a `texture2d<float>`, and no Apple-side reading states the Metal 1D
/// equivalence a `texture1d_array<float, sample>` module would need. The shape
/// therefore keeps the rail's own `render_texture_shape_unsupported` refusal
/// (`metal-api-core`'s `MAX_RENDER_TEXTURE_DIMENSION_1D` is declared only by
/// the snapshots whose rail executes the arm).
pub(crate) const SUPPORTED_RENDER_TEXTURE_FORMATS: [TextureFormat; 1] = [TextureFormat::Rgba8Unorm];

/// The present bits this provider declares as of the present-track flip.
///
/// Flip evidence (`research/docs/24` §6 Step 7): CI run `34781060564`, job
/// `native-oracle-build`, step "Run native present self-test when a Metal
/// device is eligible". The probe reported an eligible Apple Paravirtual
/// device (`supports_apple4: true`), the oracle then ran the reviewed 2x2
/// present equivalent (`load: load`, `fefefefe`-sentinel preset) and printed
/// `present_selftest: PASS (4080c0ff4080c0ff4080c0ff4080c0ff)` — four
/// expected texels, never the sentinel. A green job whose log said `SKIP` is
/// not that evidence: it reports a runner without an eligible device, not an
/// executed reviewed present path. Before the flip these bits were all at
/// their defaults, so core admission refused a present-bearing trace with
/// `present_targets_unsupported` instead of running the offscreen render and
/// silently dropping the present (`research/docs/24` §4.2); the test below
/// keeps that refusal path pinned on a constructed pre-flip snapshot.
pub(crate) fn present_capability_bits() -> PresentCapabilityBits {
    PresentCapabilityBits {
        supports_presentation: true,
        max_present_targets: MAX_PRESENT_TARGETS,
        supported_present_modes: PresentMode::ADMITTED.to_vec(),
        max_present_image_count: MAX_PRESENT_IMAGE_COUNT,
    }
}

/// The vertex-input bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side unit tests cannot drift.
///
/// Split out from the render bits for the same reason the present bits are: the
/// three values are the *rail's own* limits (as many streams as the contract
/// caps, the formats this rail has an observation for —
/// [`DECLARED_VERTEX_FORMATS`] — and the index widths it translates into
/// `MTLIndexType`), so capability admission and this rail's declaration agree
/// by construction instead of by a second list that could drift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VertexInputCapabilityBits {
    pub(crate) max_vertex_buffers: u32,
    pub(crate) supported_vertex_formats: Vec<VertexFormat>,
    pub(crate) supported_index_formats: Vec<IndexFormat>,
    /// The superset vertex interface's bit (`research/docs/23` §3.3, E-TX11),
    /// declared beside the three vertex-input fields it narrows.
    pub(crate) supports_render_vertex_interface_superset: bool,
    /// The layout-free count above the milestone's three vertices
    /// (2026-09-19, census v45's `vertex_span` bucket). Declared beside the
    /// superset bit for the same reason: both are vertex-input-arm questions
    /// this rail's reviewed module table answers `false` to.
    pub(crate) supports_render_vertex_count_above_triangle: bool,
}

/// The vertex-input bits this provider declares as of the vertex-input flip.
///
/// Flip condition (`research/docs/23` §3.3, §6 Step 3.3;
/// `conformance/RENDER-CAPTURE.md` §8): the Swift oracle's `--vertex-selftest`
/// on an Apple GPU, i.e. the same shape as the render bits' `--render-selftest`
/// evidence. That self-test builds the indexed reviewed module, binds the
/// fixture's `float32x2` stream and `uint16` index buffer through an
/// `MTLVertexDescriptor`, draws with `drawIndexedPrimitives` and reads the 2x2
/// attachment back as `4080c0ff` four times, printing `vertex_selftest: PASS
/// (4080c0ff)`. A green job whose log said `SKIP` is not that evidence: it
/// reports a runner without an eligible device rather than an executed reviewed
/// path.
///
/// Before the flip these three bits were all at their defaults, so core
/// admission refused a vertex-bearing trace with `vertex_buffer_limit` /
/// `index_format_unsupported` instead of executing it with positions the trace
/// did not ask for; the test below keeps that refusal path pinned on a
/// constructed pre-flip snapshot. The host-side half this rail owns is
/// [`plan_vertex_input`]: the stream bytes, the stride and index footprints and
/// the index values are all proved before a device object exists, and the
/// remaining Apple-only question is whether Metal executes exactly that plan.
pub(crate) fn vertex_input_capability_bits() -> VertexInputCapabilityBits {
    VertexInputCapabilityBits {
        max_vertex_buffers: MAX_VERTEX_BUFFERS,
        supported_vertex_formats: DECLARED_VERTEX_FORMATS.to_vec(),
        supported_index_formats: IndexFormat::ADMITTED.to_vec(),
        // The superset interface stays refused (`research/docs/23` §3.3,
        // E-TX11): this rail has no reflection to hold a layout against —
        // `reviewed_module` selects its MSL module by the layout's *exact*
        // shape (one stream with two attributes, the instanced pair, the
        // depth pair, …), so a layout that declares attributes no reviewed
        // module reads matches no arm and the registration is refused by name.
        // Declaring the bit would promise a shape this rail's module table
        // refuses, so it keeps the consumer's fail-closed default.
        supports_render_vertex_interface_superset: false,
        // The layout-free count above the milestone's three vertices
        // (2026-09-19, census v45's `vertex_span` bucket) keeps the same
        // fail-closed default for the same kind of reason: the one
        // `vertex_id` module this rail's table holds reads a *three-entry*
        // position table by index (`conformance/shaders/render_offscreen_2x2.metal`),
        // so a count above three would read a position the module does not
        // carry. Declaring the bit would promise a shape no reviewed module
        // here can execute.
        supports_render_vertex_count_above_triangle: false,
    }
}

/// The vertex formats this provider *declares* (`research/docs/23` §103,
/// E-VF1).
///
/// The four 32-bit storages are the window the `--vertex-selftest` observation
/// behind the vertex-input flip built: the reviewed indexed module over one
/// `float32x2` stream and a `uint16` index buffer. The contract's four
/// normalized storages arrived afterwards, and its scalar `float32` lane after
/// them. Their descriptor mapping is already
/// total ([`vertex_format`] and [`metal_vertex_format`]), and the
/// `MTLVertexFormat` each one names is the reviewed one — but what core
/// admission reads is this *declaration*, and this rail's discipline is that a
/// capability follows an Apple-side observation rather than a table (the
/// `--stage-buffer-selftest` / `--heap-selftest` flips record the same rule).
///
/// Flip condition: an Apple device reading of the widened shapes — the
/// `--vertex-selftest` shape extended to a case whose descriptor declares
/// `UChar4Normalized` and its siblings, and one that declares `Float` — landing
/// the fixture's own bytes on the macOS runner. This local round cannot produce
/// that reading (no push, no Apple device), so the four normalized storages and
/// the scalar lane stay out of the declaration: a trace that declares one is
/// refused at admission with `vertex_format_unsupported` rather than executed on
/// an unobserved path. The scalar lane is refused by *name* rather than by
/// absence: the reviewed modules this rail's plan selects are the same
/// `float32x2`/`float32x3`/`float32x4` readers, so nothing here would read a
/// scalar member even if the descriptor could be built.
///
/// The Vulkan rail declares every value of the contract's list
/// (`crates/metal-api-vulkan/src/provider.rs`): the normalized four and the
/// scalar lane are Vulkan's *required* vertex input formats, so that rail's
/// declaration needs no device answer, and its execution is measured on
/// Lavapipe by `tests/render_normalized_vertex_e2e.rs` and
/// `tests/render_scalar_vertex_e2e.rs`.
pub(crate) const DECLARED_VERTEX_FORMATS: [VertexFormat; 4] = [
    VertexFormat::Float32x2,
    VertexFormat::Float32x3,
    VertexFormat::Float32x4,
    VertexFormat::Uint32,
];

/// The instancing bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v31).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InstancingCapabilityBits {
    pub(crate) supports_render_instancing: bool,
    pub(crate) max_render_instances: u32,
}

/// The instancing bits this provider declares as of the v31 flip.
///
/// Flip condition (`research/docs/23` §3.3): the reviewed `instanced_pair_4x4`
/// case on the Apple rail, i.e. the macOS CI job's `native-metal` capture of
/// the suite that names every rail. That run builds the reviewed MSL module,
/// declares a `per_instance` stream through `MTLVertexDescriptor` (`
/// stepFunction = .perInstance`, `stepRate = 1`), draws `instanceCount: 2` and
/// reads back the red left half and the green right half the fixture pins. The
/// ceiling follows the Vulkan rail's: the reviewed two instances rounded up to
/// four, so a wider draw stays refused by core admission rather than silently
/// narrowed.
///
/// Before this flip both bits were at their defaults, so core admission refused
/// a multi-instance pass (or a per-instance layout) with
/// `render_instancing_unsupported` / `vertex_step_unsupported` instead of
/// executing it once; the host-side half this rail owns is
/// [`plan_vertex_input`], which proves the per-instance footprint before a
/// device object exists.
pub(crate) fn instancing_capability_bits() -> InstancingCapabilityBits {
    InstancingCapabilityBits {
        supports_render_instancing: true,
        max_render_instances: MAX_RENDER_INSTANCES,
    }
}

/// The multisample bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v51).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MultisampleCapabilityBits {
    pub(crate) supports_render_multisample: bool,
    pub(crate) max_render_sample_count: u32,
    /// The reviewed 2/4/8 counts the device admits, as a bitmask over the
    /// contract codes: bit `i` = [`SampleCount`] code `i`. The capture runner
    /// reads the device-gated sample-count cases against this mask, because
    /// the ceiling alone cannot say which of the counts a device lacks
    /// (`research/docs/23` §3.3, v61).
    pub(crate) render_sample_counts: u32,
}

/// The selected device's own 2D texture limit, in texels per axis (R1b,
/// `research/docs/23` §70).
///
/// metal-rs carries the limit as a table on `MTLFeatureSet` rather than as a
/// `Device` property, so the device's answer is the largest 2D texture size
/// among the macOS GPU feature sets the device itself reports — the same shape
/// [`device_multisample_capability_bits`] uses for the sample counts. Every
/// macOS GPU family states a 16384 ceiling, and this rail only admits Apple4+
/// devices (`native.rs`: a named device with unified memory), so a device that
/// answers no macOS feature set at all keeps that documented Apple-family
/// ceiling instead of declaring 0, which would refuse every render pass.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // `MTLFeatureSet` is the binding's only 2D-size table; see above.
pub(crate) fn device_attachment_dimension_limit(device: &Device) -> u64 {
    const MACOS_GPU_FEATURE_SETS: [MTLFeatureSet; 5] = [
        MTLFeatureSet::macOS_GPUFamily1_v1,
        MTLFeatureSet::macOS_GPUFamily1_v2,
        MTLFeatureSet::macOS_GPUFamily1_v3,
        MTLFeatureSet::macOS_GPUFamily1_v4,
        MTLFeatureSet::macOS_GPUFamily2_v1,
    ];
    let limit = MACOS_GPU_FEATURE_SETS
        .iter()
        .filter(|feature_set| device.supports_feature_set(**feature_set))
        .map(|feature_set| feature_set.max_2d_texture_size())
        .max()
        .unwrap_or(0);
    if limit == 0 {
        APPLE_2D_TEXTURE_CEILING
    } else {
        u64::from(limit)
    }
}

/// Refuse a render attachment the device's own 2D texture limit cannot carry.
///
/// R1b (`research/docs/23` §70): the device half of the declared window,
/// answered from the trace's own attachment extents before any Metal object
/// exists. The snapshot already declares the device-gated window, so admission
/// refuses a wider attachment as `attachment_dimension_limit`; this refusal is
/// the rail's own second line for a trace that skipped admission, and the
/// provider runs it before [`plan_trace`]'s reviewed-ceiling check so the
/// device's answer is the one a caller sees when both halves are exceeded. The
/// requested extent and the limit it crossed are both reported; nothing is
/// narrowed.
pub(crate) fn refuse_attachment_extent_over_device_limit(
    trace: &ComputeTrace,
    device_2d_texture_limit: u64,
) -> Result<(), ProviderError> {
    let refusal = |width: u64, height: u64| {
        capability_refusal("attachment_extent_device_limit")
            .with_field("width", FieldValue::Unsigned(width))
            .with_field("height", FieldValue::Unsigned(height))
            .with_field(
                "maximum_width",
                FieldValue::Unsigned(device_2d_texture_limit),
            )
            .with_field(
                "maximum_height",
                FieldValue::Unsigned(device_2d_texture_limit),
            )
    };
    for pass in trace.render_passes() {
        let extents = pass
            .color_attachments
            .iter()
            .map(|attachment| (attachment.width, attachment.height))
            .chain(pass.depth.iter().map(|depth| (depth.width, depth.height)))
            .chain(
                pass.stencil
                    .iter()
                    .map(|stencil| (stencil.width, stencil.height)),
            );
        for (width, height) in extents {
            if width > device_2d_texture_limit || height > device_2d_texture_limit {
                return Err(refusal(width, height));
            }
        }
    }
    Ok(())
}

/// The multisample bits this provider declares, from the device's own answer
/// (`research/docs/23` §3.3, v51/v61).
///
/// The v61 increment widens the reviewed raster family to 2x/4x/8x, so the
/// ceiling is no longer a fixed reviewed count but the largest of the three
/// rasters the device's `supportsTextureSampleCount:` probe admits. A device
/// that admits none of them keeps both bits at their defaults, so core
/// admission refuses a multisampled pass with
/// `render_multisample_unsupported` instead of executing it as a single-sample
/// draw; the host-side half this rail owns is [`plan`], which holds the state
/// to the counts the encoder builds and the texture creation in
/// [`multisample_attachment_textures`].
#[cfg(target_os = "macos")]
pub(crate) fn device_multisample_capability_bits(device: &Device) -> MultisampleCapabilityBits {
    let counts = [
        (8u32, 1 << SampleCount::Eight.code()),
        (4, 1 << SampleCount::Four.code()),
        (2, 1 << SampleCount::Two.code()),
    ];
    let mask = counts
        .iter()
        .filter(|(count, _)| device.supports_texture_sample_count(u64::from(*count)))
        .fold(0, |mask, (_, bit)| mask | bit);
    let ceiling = counts
        .into_iter()
        .find(|(count, _)| device.supports_texture_sample_count(u64::from(*count)))
        .map(|(count, _)| count);
    MultisampleCapabilityBits {
        supports_render_multisample: ceiling.is_some(),
        max_render_sample_count: ceiling.unwrap_or(0),
        render_sample_counts: mask,
    }
}

/// The depth-resolve bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v57c).
///
/// The mode mask is the contract's per-filter bit layout: bit `i` =
/// [`DepthResolveFilter`] code `i` (`metal-api-core`:
/// `ProviderCapabilities::depth_resolve_modes`). The v57e
/// `--depth-resolve-selftest` run measured an Apple Paravirtual device
/// executing all three filters — the mixed column's `min=0000003f` and
/// `max=6666663f` against `sample0=0000003f` (`f4d70e4`, CI run
/// `35112569688`) — so the mask declares Sample0|Min|Max (`0b111`) instead of
/// the fail-closed Sample0-only value the pre-evidence increments used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DepthResolveCapabilityBits {
    pub(crate) supports_render_depth_resolve: bool,
    pub(crate) depth_resolve_modes: u32,
}

/// The Sample0 mode bit, spelled once so the probe and the snapshot cannot
/// disagree about which filter the rail admits.
pub(crate) const DEPTH_RESOLVE_SAMPLE0_BIT: u32 = 1u32 << DepthResolveFilter::Sample0.code();

/// The Min mode bit, declared on the same v57e self-test evidence as Sample0
/// and Max: the mixed column's reduction really lands the nearest depth.
pub(crate) const DEPTH_RESOLVE_MIN_BIT: u32 = 1u32 << DepthResolveFilter::Min.code();

/// The Max mode bit, declared on the same v57e self-test evidence as Sample0
/// and Min: the mixed column's reduction really lands the furthest depth.
pub(crate) const DEPTH_RESOLVE_MAX_BIT: u32 = 1u32 << DepthResolveFilter::Max.code();

/// The depth-resolve bits derived from the device itself, queried exactly once
/// when the provider is created (`native.rs`).
///
/// `metal` 0.33 models depth32-float resolve as a macOS platform fact:
/// `depth32_float_capabilities` reports `Resolve` for every macOS feature set
/// (its `device.rs`, and the SDK has no finer per-device selector), while the
/// crate's `supports_msaa_depth_resolve` is iOS-gated and says false on macOS.
/// The feature-set API is deprecated in the SDK, so the probe asks the modern
/// `supportsFamily:` question for the Apple family the provider's admission
/// already requires — a real device query rather than a compile-time constant,
/// combined with the platform fact the crate models. The v57e
/// `--depth-resolve-selftest` CI output landed the evidence the mask was
/// waiting for: the Apple Paravirtual device executed all three filters and
/// printed `depth_resolve_selftest: PASS` (`f4d70e4`, CI run `35112569688`),
/// so an eligible device now declares Sample0|Min|Max (`0b111`) and an
/// ineligible one still declares `0` (`research/docs/23` §3.3, v57c/v57e).
#[cfg(target_os = "macos")]
pub(crate) fn device_depth_resolve_capability_bits(device: &Device) -> DepthResolveCapabilityBits {
    let supports = device.supports_family(metal::MTLGPUFamily::Apple4);
    DepthResolveCapabilityBits {
        supports_render_depth_resolve: supports,
        depth_resolve_modes: if supports {
            DEPTH_RESOLVE_SAMPLE0_BIT | DEPTH_RESOLVE_MIN_BIT | DEPTH_RESOLVE_MAX_BIT
        } else {
            0
        },
    }
}

/// The stencil-resolve bits the provider declares, in one value so the macOS
/// capability snapshot and the host-side tests cannot drift
/// (`research/docs/23` §3.3, v60).
///
/// The mode mask is the contract's per-filter bit layout: bit `i` =
/// [`StencilResolveFilter`] code `i`. The v59 `--stencil-resolve-selftest`
/// run measured an Apple Paravirtual device executing both filters — the
/// mixed column's `depth_resolved_sample(min)=01` and
/// `depth_resolved_sample(max)=00` against `sample0=01` (`2b877b8`, CI run
/// `35120171655`) — so the mask declares Sample0|DepthResolvedSample (`0b11`)
/// and an ineligible device still declares `0` (`research/docs/23` §3.3, v60).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StencilResolveCapabilityBits {
    pub(crate) supports_render_stencil_resolve: bool,
    pub(crate) stencil_resolve_modes: u32,
}

/// The Sample0 mode bit, spelled once so the probe and the snapshot cannot
/// disagree about which filter the rail admits.
pub(crate) const STENCIL_RESOLVE_SAMPLE0_BIT: u32 = 1u32 << StencilResolveFilter::Sample0.code();

/// The DepthResolvedSample mode bit, declared on the same v59 self-test
/// evidence as Sample0: the mixed column's reduction really follows the depth
/// resolve's selected sample.
pub(crate) const STENCIL_RESOLVE_DRS_BIT: u32 =
    1u32 << StencilResolveFilter::DepthResolvedSample.code();

/// The stencil-resolve bits derived from the device itself, queried exactly
/// once when the provider is created (`native.rs`).
///
/// The probe asks the same Apple-family question the depth resolve asks; the
/// v59 `--stencil-resolve-selftest` CI output landed the evidence the mask was
/// waiting for: the Apple Paravirtual device executed both filters and printed
/// `stencil_resolve_selftest: PASS` (`2b877b8`, CI run `35120171655`), so an
/// eligible device now declares Sample0|DepthResolvedSample (`0b11`) and an
/// ineligible one still declares `0` (`research/docs/23` §3.3, v60).
#[cfg(target_os = "macos")]
pub(crate) fn device_stencil_resolve_capability_bits(
    device: &Device,
) -> StencilResolveCapabilityBits {
    let supports = device.supports_family(metal::MTLGPUFamily::Apple4);
    StencilResolveCapabilityBits {
        supports_render_stencil_resolve: supports,
        stencil_resolve_modes: if supports {
            STENCIL_RESOLVE_SAMPLE0_BIT | STENCIL_RESOLVE_DRS_BIT
        } else {
            0
        },
    }
}

/// The texel the reviewed fragment writes, as the UNORM8 bytes an admitted
/// attachment stores. Both this rail and the Swift oracle's render self-test
/// have to recognise it, and it is what makes "the draw ran and covered the
/// whole attachment" falsifiable: a 2x2 attachment whose four texels are all
/// `40 80 c0 ff` cannot be the sentinel a `LoadOp::Clear` leaves behind.
pub(crate) const EXPECTED_TEXEL_BYTES: [u8; 4] = [0x40, 0x80, 0xc0, 0xff];

/// The pixel format this rail maps an admitted attachment format onto.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderPixelFormat {
    Rgba8Unorm,
    Bgra8Unorm,
    R32Float,
    /// `MTLPixelFormat::RGBA16Float` / `VK_FORMAT_R16G16B16A16_SFLOAT`
    /// (`research/docs/23` §78).
    ///
    /// The four-component class's second storage width: the reviewed
    /// single-output modules all store a `float4`, and the attachment's own
    /// format decides whether that store lands as four `UNORM8` channels or as
    /// four half floats. Eight bytes per texel is what makes this rail's
    /// readback extent and `Load` upload an attachment's own fact.
    Rgba16Float,
}

impl RenderPixelFormat {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Rgba8Unorm => "rgba8_unorm",
            Self::Bgra8Unorm => "bgra8_unorm",
            Self::R32Float => "r32_float",
            Self::Rgba16Float => "rgba16_float",
        }
    }
}

/// The clear value or previous contents an encoder has to start from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RenderLoadAction {
    /// Clear every texel, with the components decoded from the contract bytes.
    Clear([f64; 4]),
    /// Keep the attachment's previous contents.
    Load,
    /// Leave the attachment's previous contents undefined: the encoder sets
    /// `MTLLoadAction::DontCare` and presets nothing, so the pass neither
    /// reads nor overwrites the pre-pass bytes before drawing
    /// (`research/docs/23` §3.1, v20).
    DontCare,
}

/// The store action this rail sets. `Store` lands the attachment's texels on
/// the observable surface through the readback; `DontCare` is the contract's
/// `StoreOp::DontCare` (`research/docs/23` §3.6, v19): the attachment still
/// renders, but its bytes disappear from the observable surface — the encoder
/// sets `MTLStoreAction::DontCare` and reads nothing back, so a discarded
/// attachment cannot pass as "landed correctly". Core admission still refuses
/// a pass whose every attachment discards, so one `Store` action always keeps
/// the pass observable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderStoreAction {
    Store,
    /// Discard the attachment's writes; no readback and no writeback leave this
    /// attachment's location.
    DontCare,
}

/// Map an admitted attachment format onto the pixel format this rail builds.
pub(crate) fn pixel_format(format: AttachmentFormat) -> Result<RenderPixelFormat, ProviderError> {
    match format {
        AttachmentFormat::Rgba8Unorm => Ok(RenderPixelFormat::Rgba8Unorm),
        AttachmentFormat::Bgra8Unorm => Ok(RenderPixelFormat::Bgra8Unorm),
        AttachmentFormat::R32Float => Ok(RenderPixelFormat::R32Float),
        AttachmentFormat::Rgba16Float => Ok(RenderPixelFormat::Rgba16Float),
        // Expressible in the contract for symmetry with the sampled-texture
        // rail, refused by the first render increment: its texels are integers
        // while `LoadOp::Clear` and the fragment output carry colour bytes
        // (`metal_api_core::provider::AttachmentFormat::R32Uint`). The refusal
        // reuses core admission's slug and class rather than inventing a second
        // spelling for one fact.
        AttachmentFormat::R32Uint => Err(capability_refusal("attachment_format_unsupported")
            .with_field("format", FieldValue::Unsigned(u64::from(format.code())))),
    }
}

/// Decode the contract's clear bytes into `MTLClearColor` components.
///
/// The contract carries bytes in memory order, so the mapping is per format: the
/// two 8-bit UNORM formats differ in channel order, and the single-channel float
/// format's four bytes are the red component itself. Writing the value into the
/// attachment is the driver's job; this only fixes which component each byte
/// means (`research/docs/23` §3.5).
pub(crate) fn clear_components(clear: ClearColor, format: RenderPixelFormat) -> [f64; 4] {
    let bytes = clear.as_bytes();
    match format {
        // Memory order is R, G, B, A.
        RenderPixelFormat::Rgba8Unorm => [
            unorm8(bytes[0]),
            unorm8(bytes[1]),
            unorm8(bytes[2]),
            unorm8(bytes[3]),
        ],
        // Memory order is B, G, R, A, so red and blue swap on the way into the
        // component order `MTLClearColor` carries: one colour has two byte
        // strings, one per format.
        RenderPixelFormat::Bgra8Unorm => [
            unorm8(bytes[2]),
            unorm8(bytes[1]),
            unorm8(bytes[0]),
            unorm8(bytes[3]),
        ],
        // One channel: the four bytes are the little-endian IEEE-754 bits of the
        // red component. Green and blue stay zero and alpha is unused, which is
        // what a single-channel attachment stores.
        RenderPixelFormat::R32Float => [
            f64::from(f32::from_le_bytes(narrow_four(bytes))),
            0.0,
            0.0,
            1.0,
        ],
        // Four half floats, in the format's memory order (R, G, B, A
        // little-endian halves). The widening is exact, so the components the
        // encoder hands `MTLClearColor` are the very values the clear's bytes
        // name and the driver rounds each of them into the half the readback
        // compares (`research/docs/23` §78,
        // `metal_api_core::provider::half_to_f32`).
        RenderPixelFormat::Rgba16Float => [
            f64::from(half_from_memory_order(bytes, 0)),
            f64::from(half_from_memory_order(bytes, 1)),
            f64::from(half_from_memory_order(bytes, 2)),
            f64::from(half_from_memory_order(bytes, 3)),
        ],
    }
}

/// One half-precision component of a `Rgba16Float` clear, from its
/// little-endian pair of bytes.
///
/// A payload shorter than four halves cannot reach here — admission and the
/// plan both compare the clear's length with its attachment's own texel width —
/// so a hand-built value that somehow got past them reads zero rather than
/// panicking inside an encoder call.
fn half_from_memory_order(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    let pair = bytes.get(offset..offset + 2).unwrap_or(&[0, 0]);
    metal_api_core::provider::half_to_f32(u16::from_le_bytes([pair[0], pair[1]]))
}

/// The first four bytes of a clear payload, or four zeros for a value no
/// admission could have built.
///
/// The same fail-closed-to-zero rule as [`half_from_memory_order`]: the four
/// byte formats' payloads are checked against their attachment's texel width
/// before an encoder exists, and this keeps the decode total anyway.
fn narrow_four(bytes: &[u8]) -> [u8; 4] {
    let mut four = [0_u8; 4];
    if let Some(pair) = bytes.get(..4) {
        four.copy_from_slice(pair);
    }
    four
}

fn unorm8(byte: u8) -> f64 {
    f64::from(byte) / 255.0
}

/// The load action an encoder has to set for this pass.
pub(crate) fn load_action(
    load: LoadOp,
    format: RenderPixelFormat,
) -> Result<RenderLoadAction, ProviderError> {
    match load {
        LoadOp::Clear(clear) => Ok(RenderLoadAction::Clear(clear_components(clear, format))),
        LoadOp::Load => Ok(RenderLoadAction::Load),
        LoadOp::DontCare => Ok(RenderLoadAction::DontCare),
        // The provider-resident load (`research/docs/23` §76, R7) keeps the
        // bytes of the image the provider owns under the attachment's own
        // identity, so the encoder opens it with `MTLStoreActionLoad` exactly
        // as a trace-declared `Load` does. The difference between the two arms
        // is where the bytes come from — a resident load uploads nothing,
        // because they are already in the provider's texture — and that is
        // decided by the plan, which refuses the whole trace by name unless the
        // provider resolved a live image for the identity
        // (`resident_target_undeclared` and the registry's own refusals).
        LoadOp::Resident => Ok(RenderLoadAction::Load),
    }
}

/// The store action an encoder has to set for this pass.
///
/// Both contract variants are admitted now (`research/docs/23` §3.6, v19):
/// core admission refuses the all-discarded pass before a plan is built, so
/// any pass that reaches this function still stores at least one attachment.
/// The retained `Result` spells that this is the mapping the encoder depends
/// on, not a second policy.
pub(crate) fn store_action(store: StoreOp) -> Result<RenderStoreAction, ProviderError> {
    Ok(match store {
        StoreOp::Store => RenderStoreAction::Store,
        StoreOp::DontCare => RenderStoreAction::DontCare,
        // A resident store keeps the pass's bytes in the provider's own image
        // (`research/docs/23` §76, R7), which means the encoder still has to
        // state `MTLStoreActionStore`: the texture is what has to keep them.
        // What this arm does *not* do is publish a writeback — that is the
        // plan's `publishes` bit, which the readback and the writeback channel
        // both read, so "kept in the provider's image" cannot pass as "landed
        // through the buffer channel" or the other way round.
        StoreOp::Resident => RenderStoreAction::Store,
        // The owner-window store (`research/docs/23` §114, E-TX8) lands the
        // frame in the owner's registered window, and this rail's own window
        // channel is the render *input* one (`research/docs/23` §72, R3d): it
        // imports an owner mapping the device reads, and it has no route that
        // writes one back. The arm is refused rather than executed as a plain
        // store, because the caller that declared it asked for the guest's own
        // pages to hold the frame and a writeback nobody lands would leave them
        // holding the pass's previous bytes.
        StoreOp::Borrowed => {
            return Err(capability_refusal("render_attachment_landing_unsupported")
                .with_field("source", FieldValue::Text("borrowed_no_copy".to_owned()))
                .with_detail(
                    "a borrowed store lands the pass's frame in the owner's registered window; \
                     this rail carries the owner's window as an input and has no landing route \
                     that writes one",
                ))
        }
        // The landing-view arm (`research/docs/23` §115 之后的增量，E-TX13) names
        // the window through a *second* view declaration, and this rail's window
        // channel is an input channel: it imports an owner mapping the device
        // reads and has no route that writes one back. Refused by name for the
        // same reason the borrowed arm beside it is — and the field says which
        // of the two declarations the caller meant, so a reader of the refusal
        // does not have to guess.
        StoreOp::BorrowedLanding(view) => {
            return Err(capability_refusal("render_attachment_landing_unsupported")
                .with_field("source", FieldValue::Text("landing_view".to_owned()))
                .with_field("landing_view", FieldValue::Unsigned(view.view_id.get()))
                .with_field(
                    "landing_allocation",
                    FieldValue::Unsigned(view.allocation_id.get()),
                )
                .with_detail(
                    "a landing-view store lands the pass's frame in the owner's registered \
                     window a second view declaration names; this rail carries the owner's \
                     window as an input and has no landing route that writes one",
                ))
        }
    })
}

/// The vertex format this rail hands its `MTLVertexAttributeDescriptor`.
///
/// A value of its own rather than `MTLVertexFormat` directly, so the descriptor
/// translation is host-testable: `MTLVertexFormat` only exists on macOS, while
/// the binding index, the stride and the attribute offsets are decided here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderVertexFormat {
    Float2,
    Float3,
    Float4,
    Uint,
    /// `MTLVertexFormat::UChar2Normalized` (`research/docs/23` §103, E-VF1).
    UChar2Normalized,
    /// `MTLVertexFormat::UChar4Normalized`.
    UChar4Normalized,
    /// `MTLVertexFormat::UShort2Normalized`.
    UShort2Normalized,
    /// `MTLVertexFormat::UShort4Normalized`.
    UShort4Normalized,
    /// `MTLVertexFormat::Float` — the scalar lane the contract appended on
    /// 2026-09-20 (census v46's `vertex_format` bucket). Mapped so the
    /// descriptor translation stays total; not declared, for the reason
    /// [`DECLARED_VERTEX_FORMATS`] states.
    Float,
}

impl RenderVertexFormat {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Float2 => "float32x2",
            Self::Float3 => "float32x3",
            Self::Float4 => "float32x4",
            Self::Uint => "uint32",
            // The spellings are the contract's own (`VertexFormat`), so a
            // refusal that quotes this rail's plan and a suite that names the
            // storage read the same word (`research/docs/23` §103).
            Self::UChar2Normalized => "unorm8x2",
            Self::UChar4Normalized => "unorm8x4",
            Self::UShort2Normalized => "unorm16x2",
            Self::UShort4Normalized => "unorm16x4",
            Self::Float => "float32x1",
        }
    }
}

/// Map an admitted vertex format onto the format this rail builds.
///
/// Total over the contract's eight values, so a ninth wire code arriving
/// without a mapping fails to compile here rather than silently becoming a
/// different descriptor. The four normalized storages are mapped from the
/// `MTLVertexFormat`s that name them exactly — `UChar2Normalized` and its three
/// siblings carry the same `c / 255` / `c / 65535` conversion the contract
/// states, so no arm here has to add or remove a normalization
/// (`research/docs/23` §103). The scalar lane maps onto the same
/// `MTLVertexFormat::Float` its contract name spells, with no conversion
/// between the fetched bytes and the member (2026-09-20, census v46's
/// `vertex_format` bucket).
///
/// Being mapped is not being *declared*: which of these the provider publishes
/// is [`vertex_input_capability_bits`], and that window is where the Apple
/// reading lives.
pub(crate) const fn vertex_format(format: VertexFormat) -> RenderVertexFormat {
    match format {
        VertexFormat::Float32x2 => RenderVertexFormat::Float2,
        VertexFormat::Float32x3 => RenderVertexFormat::Float3,
        VertexFormat::Float32x4 => RenderVertexFormat::Float4,
        VertexFormat::Uint32 => RenderVertexFormat::Uint,
        VertexFormat::Unorm8x2 => RenderVertexFormat::UChar2Normalized,
        VertexFormat::Unorm8x4 => RenderVertexFormat::UChar4Normalized,
        VertexFormat::Unorm16x2 => RenderVertexFormat::UShort2Normalized,
        VertexFormat::Unorm16x4 => RenderVertexFormat::UShort4Normalized,
        VertexFormat::Float32x1 => RenderVertexFormat::Float,
    }
}

/// The index width this rail hands `drawIndexedPrimitives(indexType:)`.
///
/// The sibling of [`RenderVertexFormat`]: a host-visible value so the
/// `IndexFormat` → `MTLIndexType` mapping is testable without a device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderIndexType {
    Uint16,
    Uint32,
}

impl RenderIndexType {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
        }
    }

    /// Bytes one index of this width occupies, restated from the contract value
    /// the mapping came from so the footprint proof and the width cannot drift.
    pub(crate) const fn bytes(self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Map an admitted index format onto the width this rail draws with. Total for
/// the same reason [`vertex_format`] is.
pub(crate) const fn index_type(format: IndexFormat) -> RenderIndexType {
    match format {
        IndexFormat::Uint16 => RenderIndexType::Uint16,
        IndexFormat::Uint32 => RenderIndexType::Uint32,
    }
}

/// The step function this rail hands its `MTLVertexBufferLayoutDescriptor`
/// (`research/docs/23` §3.3, v31).
///
/// The sibling of [`RenderVertexFormat`]: a host-visible value so the
/// `VertexStep` → `MTLVertexStepFunction` mapping (and the footprint proof that
/// reads it) is testable without a device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RenderVertexStep {
    PerVertex,
    PerInstance,
}

impl RenderVertexStep {
    /// Stable spelling used by tests and refusals.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::PerVertex => "per_vertex",
            Self::PerInstance => "per_instance",
        }
    }
}

/// Map an admitted step onto the function this rail builds. Total for the same
/// reason [`vertex_format`] is: the contract's closed pair is exactly the two
/// `MTLVertexStepFunction`s the rails execute.
pub(crate) const fn vertex_step(step: VertexStep) -> RenderVertexStep {
    match step {
        VertexStep::PerVertex => RenderVertexStep::PerVertex,
        VertexStep::PerInstance => RenderVertexStep::PerInstance,
    }
}

/// One vertex attribute as the descriptor builder needs it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedVertexAttribute {
    /// Shader-visible attribute location (`[[attribute(n)]]`).
    pub(crate) location: u32,
    /// Byte offset inside one vertex.
    pub(crate) offset: u64,
    /// The translated format.
    pub(crate) format: RenderVertexFormat,
}

/// One vertex stream the pass binds, resolved before any Metal object exists.
///
/// The bytes are the view's own (`BufferSource::OwnedBytes`), i.e. exactly the
/// `view.length` bytes the trace declared at the view, and `offset` is where
/// that view starts inside its allocation. The macOS encoder builds one
/// MTLBuffer with the bytes at that offset and binds it at the same offset, the
/// convention the compute pool's merged images follow (`native.rs`): a stream
/// reads the byte range the trace named instead of being silently re-based at
/// zero.
#[derive(Debug)]
pub(crate) struct PlannedVertexStream<'a> {
    /// Binding index: entry `i` of the pipeline layout is binding `i`, which is
    /// also its `MTLVertexBufferLayoutDescriptor` index and the
    /// `setVertexBuffer(_:offset:index:)` index.
    pub(crate) buffer_index: u32,
    /// Bytes between consecutive vertices.
    pub(crate) stride: u64,
    /// How often the stream advances (`research/docs/23` §3.3, v31): once per
    /// vertex or once per instance. The footprint proof below reads it to
    /// decide which count the stream has to cover.
    pub(crate) step: RenderVertexStep,
    /// Attributes this stream is read through.
    pub(crate) attributes: Vec<PlannedVertexAttribute>,
    /// Where this stream's bytes come from (`research/docs/23` §72, R3d): the
    /// bytes the view declares, the provider's staged copy of the owner's
    /// lease, or the owner's own mapping.
    pub(crate) source: PlannedInputSource<'a>,
    /// The view's offset inside its allocation.
    pub(crate) offset: u64,
}

impl PlannedVertexStream<'_> {
    /// The offset the encoder binds this stream at.
    ///
    /// An uploaded stream is bound at the view's own offset inside its
    /// allocation; a no-copy stream is bound at its offset inside the owner
    /// reservation the encoder maps. Either way the binding starts on the first
    /// byte the trace named.
    pub(crate) fn binding_offset(&self) -> u64 {
        source_binding_offset(&self.source, self.offset)
    }
}

/// The index buffer the pass draws through, resolved the same way.
#[derive(Debug)]
pub(crate) struct PlannedIndexStream<'a> {
    /// The view the indices come from.
    pub(crate) view_id: ViewId,
    /// Where this stream's bytes come from, exactly as a vertex stream's do.
    pub(crate) source: PlannedInputSource<'a>,
    /// The view's offset inside its allocation.
    pub(crate) offset: u64,
    /// The translated index width.
    pub(crate) format: RenderIndexType,
    /// Indices the draw consumes, i.e. `RenderPassDescriptor::vertices` in the
    /// indexed shape.
    pub(crate) index_count: u32,
    /// The highest index the draw reads plus one: the number of vertices the
    /// bound streams have to cover for this draw.
    pub(crate) vertex_span: u64,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34), i.e. Metal's `baseVertex`. The stream has to cover
    /// `base_vertex + vertex_span` records, and the draw call carries the
    /// offset whenever it is non-zero.
    pub(crate) base_vertex: u64,
}

impl PlannedIndexStream<'_> {
    /// The offset the encoder binds the index buffer at, by the same rule the
    /// vertex streams use.
    pub(crate) fn binding_offset(&self) -> u64 {
        source_binding_offset(&self.source, self.offset)
    }
}

/// One stage buffer the pass binds, resolved before any Metal object exists
/// (`research/docs/23` §83/R9g, §92/R9k).
///
/// The entry names its stage explicitly: the two stages' `[[buffer(N)]]` index
/// spaces are independent, so the index alone cannot say which namespace it
/// fills. `bytes` is the reviewed module's own reach over *this* draw — the
/// fact the registration paired the declaration against, evaluated over the
/// draw's counts for the affine arm — and the encoder binds the resolved buffer
/// at `index` in the stage's own namespace.
///
/// A writable binding is also a landing (`research/docs/23` §3.3, v86/v92): the
/// view identity, offset and length here are the ones the pass's own view
/// declares, which is what the post-fence readback publishes as one complete
/// [`BufferWriteback`] for that view.
#[derive(Debug)]
pub(crate) struct PlannedStageBuffer<'a> {
    /// Which stage reads or writes the binding.
    pub(crate) stage: RenderPipelineStage,
    /// The `[[buffer(index)]]` number inside that stage's namespace, i.e. the
    /// `setVertexBuffer(_:offset:index:)` / `setFragmentBuffer(_:offset:index:)`
    /// index the encoder binds.
    pub(crate) index: u32,
    /// Bytes the reviewed module's read and write reach over this draw.
    pub(crate) bytes: u64,
    /// Where this binding's bytes come from, through the same three-armed
    /// channel every other render input uses.
    pub(crate) source: PlannedInputSource<'a>,
    /// The view's offset inside its allocation, which is also the offset a
    /// writeback of this view carries.
    pub(crate) offset: u64,
    /// The identity of the view this binding lands in, for a writable binding.
    /// Read-only bindings carry it too, so the plan's one binding table is the
    /// whole record of what the pass bound.
    pub(crate) view_id: ViewId,
    pub(crate) allocation_id: AllocationId,
    /// How the stage uses the bytes. A writable binding is the arm the rail
    /// reads back after the pass's fence and publishes through the writeback
    /// channel.
    pub(crate) access: BufferAccess,
    /// Bytes the view declares, which is the extent a writeback publishes
    /// (`BufferWriteback` carries the view's whole extent, not the module's
    /// reach inside it).
    pub(crate) length: u64,
}

impl PlannedStageBuffer<'_> {
    /// The offset the encoder binds this binding at, by the same rule the
    /// vertex streams use.
    pub(crate) fn binding_offset(&self) -> u64 {
        source_binding_offset(&self.source, self.offset)
    }
}

/// Resolve a pass's vertex streams and index buffer from the pass itself.
///
/// A render input declares its own bytes (`research/docs/23` §3.6): entry `i` of
/// [`RenderPassDescriptor::vertex_buffers`] is a read-only [`BufferView`] whose
/// `source` carries the vertices, and an index binding carries its view beside
/// the width. Neither has to be declared by a compute pass, so this rail uploads
/// what the pass hands it instead of resolving a name against a pool.
///
/// Two rules are checked here because a driver answers both with undefined
/// behaviour instead of an error:
///
/// * the stream's bytes are ones this rail can read
///   ([`resolve_render_input`]): the bytes the view declares, the provider's
///   staged copy of an owner lease, or the owner's own mapping when the device
///   can take one. Nothing else has a path through this rail, so an arm this
///   rail cannot resolve is refused instead of executed against bytes the rail
///   does not have;
/// * the declared range covers every vertex and index the draw reads (the
///   footprint proof `research/docs/23` §3.3 asks for: Metal would read past the
///   buffer, or index a stream out of range, without refusing). Which count the
///   proof is against depends on the draw: a non-indexed draw reads its vertex
///   count in order, while an indexed draw reads the vertices its index values
///   select, so the refusal names the index rather than the stream in that case.
///
/// The proof reads the resolved window, which for a no-copy input is the
/// owner's own pages: a footprint proved over a copy of them could pass while
/// the device reads different bytes.
///
/// What is *not* checked here is the binding label: the entry's position is the
/// binding index both rails use, and core admission already holds each view's
/// own `metal_binding` to it ([`validate_vertex_buffer_binding`]), which `plan`
/// re-runs before this function.
///
/// `vertex_positions_carried_by_stage_buffer` is the one draw shape a stream
/// cannot bound: a `vertex_id`-shaped module that reads its positions out of a
/// stage buffer states its own affine reach, and that reach — not the reviewed
/// full-screen triangle's three records — is what covers the indices the draw
/// names (`research/docs/23` §92, R9k). The flag is the caller's answer for the
/// module it selected, so this proof and the stage-buffer proof read one
/// measurement of the draw.
fn plan_vertex_input<'a>(
    pass: &'a RenderPassDescriptor,
    pipeline: &RenderPipelineContract,
    leases: Option<&RenderLeaseContext<'_>>,
    vertex_positions_carried_by_stage_buffer: bool,
) -> Result<(Vec<PlannedVertexStream<'a>>, Option<PlannedIndexStream<'a>>), ProviderError> {
    let mut streams = Vec::with_capacity(pass.vertex_buffers.len());
    // One layout entry per bound stream, in binding order; core admission
    // refuses a pass and a layout that disagree about the count
    // (`ContractError::VertexLayoutBindingMismatch`), which `plan` re-runs
    // before this function, so the zip covers every bound stream.
    for (buffer_index, (view, layout)) in pass
        .vertex_buffers
        .iter()
        .zip(pipeline.vertex_layout.buffers())
        .enumerate()
    {
        let source = resolve_render_input(view, leases, RenderInputRole::Vertex)?;
        streams.push(PlannedVertexStream {
            buffer_index: u32::try_from(buffer_index)
                .map_err(|_| capability_refusal("vertex_buffer_limit"))?,
            stride: layout.stride,
            step: vertex_step(layout.step),
            attributes: layout
                .attributes
                .iter()
                .map(|attribute| PlannedVertexAttribute {
                    location: attribute.location,
                    offset: attribute.offset,
                    format: vertex_format(attribute.format),
                })
                .collect(),
            source,
            offset: view.offset,
        });
    }
    // A per-instance stream's record count is the draw's instance count, and a
    // per-vertex stream's is the vertex span the draw reads; the two proofs
    // below therefore ask each stream for the right count
    // (`research/docs/23` §3.3, v31). The instance count comes from the pass,
    // which `plan` has already validated as at least one.
    let instance_count = u64::from(pass.instance_count);
    for (buffer_index, stream) in streams.iter().enumerate() {
        if stream.step != RenderVertexStep::PerInstance {
            continue;
        }
        // Saturating for the same reason the per-vertex proof is: the product
        // only has to decide whether the stream covers the count.
        let required = instance_count.saturating_mul(stream.stride);
        if u64::try_from(stream.source.len()).unwrap_or(u64::MAX) < required {
            return Err(vertex_footprint_refusal(
                buffer_index,
                stream.stride,
                required,
                stream.source.len(),
            )
            .with_field("step", FieldValue::Text(stream.step.name().to_owned()))
            .with_detail("a per-instance stream has to cover one record per instance"));
        }
    }
    let indices = match &pass.indices {
        None => None,
        Some(binding) => Some(plan_index_stream(
            binding,
            pass.vertices,
            u64::from(pass.base_vertex),
            leases,
        )?),
    };
    match &indices {
        // An indexed draw reads the vertices its index values select, so each
        // stream has to cover that span. The refusal names the index that
        // reached past the stream, because that is what the trace has to change.
        Some(indices) => {
            // The offset takes part in the proof: the vertex a draw reads is
            // `base_vertex + index`, so a span that fits on its own can still
            // reach past the stream once the offset is added
            // (`research/docs/23` §3.3, v34).
            let required_span = indices.vertex_span.saturating_add(indices.base_vertex);
            for (buffer_index, stream) in streams.iter().enumerate() {
                // A per-instance stream is proved against the instance count
                // above, so the vertex span does not apply to it
                // (`research/docs/23` §3.3, v31).
                if stream.step == RenderVertexStep::PerInstance {
                    continue;
                }
                // `stride == 0` cannot reach here (`plan` re-runs the layout
                // validator), so `checked_div` is only the safe spelling of the
                // quotient: a zero stride would be refused upstairs rather than
                // read as an unbounded stream.
                let covered = u64::try_from(stream.source.len())
                    .unwrap_or(u64::MAX)
                    .checked_div(stream.stride)
                    .unwrap_or(0);
                if required_span > covered {
                    return Err(index_value_refusal(highest_index(required_span), covered)
                        .with_field(
                            "buffer_index",
                            FieldValue::Unsigned(u64::try_from(buffer_index).unwrap_or(u64::MAX)),
                        )
                        .with_field("base_vertex", FieldValue::Unsigned(indices.base_vertex))
                        .with_field("view", FieldValue::Unsigned(indices.view_id.get())));
                }
            }
            // The `vertex_id` shape binds no stream to bound its index values:
            // the reviewed module generates exactly
            // `FULL_SCREEN_TRIANGLE_VERTICES` positions, so an index at or above
            // that count would read a position the module does not carry. A
            // reviewed module that reads its positions *out of a stage buffer*
            // states its own reach instead (`research/docs/23` §92, R9k): the
            // affine declaration paired with that slot is proven over the same
            // `base_vertex + highest index + 1` count
            // ([`plan_stage_buffers`]), so the record every `vertex_id` names is
            // covered by the buffer's own proof rather than by this clamp.
            if pass.vertex_buffers.is_empty()
                && !vertex_positions_carried_by_stage_buffer
                && required_span > u64::from(FULL_SCREEN_TRIANGLE_VERTICES)
            {
                return Err(index_value_refusal(
                    highest_index(required_span),
                    u64::from(FULL_SCREEN_TRIANGLE_VERTICES),
                )
                .with_field("base_vertex", FieldValue::Unsigned(indices.base_vertex))
                .with_field("view", FieldValue::Unsigned(indices.view_id.get())));
            }
        }
        // A non-indexed draw reads vertices `0..vertices` in order, so every
        // stream has to cover the pass's own count.
        None => {
            // The layout-free count is this rail's own window (2026-09-19,
            // census v45's `vertex_span` bucket). `streams` is empty exactly
            // when the pipeline declares `VertexLayout::None`, and the one
            // `vertex_id` module this rail's table holds reads a *three-entry*
            // position table by index
            // (`conformance/shaders/render_offscreen_2x2.metal`), so a wider
            // count would read a position it does not carry. Core admission
            // refuses the shape by name for the snapshot this rail publishes
            // (`render_vertex_count_window_unsupported`); this second check is
            // the directly-constructed request's, and it names the rail rather
            // than the snapshot, the same way the texel-space refusal in
            // [`plan`] does.
            if streams.is_empty() && pass.vertices != FULL_SCREEN_TRIANGLE_VERTICES {
                return Err(capability_refusal("render_vertex_count_window_unsupported")
                    .with_field("vertices", FieldValue::Unsigned(u64::from(pass.vertices)))
                    .with_field("rail", FieldValue::Text("native".to_owned()))
                    .with_detail(
                        "this rail's reviewed `vertex_id` module reads a three-entry position \
                         table by index, so a layout-free draw whose count is not the \
                         milestone's three would read a position the module does not carry; the \
                         Vulkan rail's reviewed and translated modules are total functions of \
                         the index, which is where the widened count runs",
                    ));
            }
            for (buffer_index, stream) in streams.iter().enumerate() {
                // The per-vertex arm of the same split: a per-instance stream
                // covers one record per instance instead of one per vertex.
                if stream.step == RenderVertexStep::PerInstance {
                    continue;
                }
                // Saturating, because the proof only has to decide whether the
                // stream covers the count: an unrepresentable product is by
                // definition larger than any buffer this provider admits.
                let required = u64::from(pass.vertices).saturating_mul(stream.stride);
                if u64::try_from(stream.source.len()).unwrap_or(u64::MAX) < required {
                    return Err(vertex_footprint_refusal(
                        buffer_index,
                        stream.stride,
                        required,
                        stream.source.len(),
                    ));
                }
            }
        }
    }
    Ok((streams, indices))
}

/// The highest index value a span of `vertex_span` vertices reads.
fn highest_index(vertex_span: u64) -> u32 {
    u32::try_from(vertex_span.saturating_sub(1)).unwrap_or(u32::MAX)
}

/// The index buffer of one pass, with its footprint and its index values proved.
///
/// The bytes are the binding's own view, exactly as a vertex stream's are
/// ([`plan_vertex_input`]): an index buffer declares its source instead of
/// naming a view a compute pass happens to carry, and [`resolve_render_input`]
/// decides which of the three sources that declaration is.
fn plan_index_stream<'a>(
    binding: &'a IndexBufferBinding,
    index_count: u32,
    base_vertex: u64,
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<PlannedIndexStream<'a>, ProviderError> {
    let view = &binding.view;
    let source = resolve_render_input(view, leases, RenderInputRole::Index)?;
    let bytes = source.proof_bytes();
    let format = index_type(binding.format);
    let width = usize::try_from(format.bytes()).unwrap_or(usize::MAX);
    let needed = usize::try_from(index_count)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| index_footprint_refusal(width, usize::MAX, bytes.len()))?;
    if bytes.len() < needed {
        return Err(index_footprint_refusal(width, needed, bytes.len()));
    }
    let highest = match format {
        RenderIndexType::Uint16 => bytes[..needed]
            .chunks_exact(2)
            .map(|chunk| u32::from(u16::from_le_bytes([chunk[0], chunk[1]])))
            .max(),
        RenderIndexType::Uint32 => bytes[..needed]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .max(),
    };
    // `u64`, so an index of `u32::MAX` is a span rather than an overflow: the
    // footprint proof below refuses it like any other index that reaches past
    // the streams.
    let vertex_span = match highest {
        None => 0,
        Some(value) => u64::from(value) + 1,
    };
    Ok(PlannedIndexStream {
        base_vertex,
        view_id: view.view_id,
        source,
        offset: view.offset,
        format,
        index_count,
        vertex_span,
    })
}

/// One render input whose source this rail cannot read (`research/docs/23`
/// §72, R3d), refused under the name this rail published before the lease
/// channel existed.
fn render_input_refusal(
    role: RenderInputRole,
    view: ViewId,
    storage_mode: &'static str,
    detail: &'static str,
) -> ProviderError {
    capability_refusal(role.slug())
        .with_field("view", FieldValue::Unsigned(view.get()))
        .with_field("storage_mode", FieldValue::Text(storage_mode.to_owned()))
        .with_detail(detail)
}

/// Resolve a pass's stage buffer bindings from the pass itself
/// (`research/docs/23` §83/R9g, §92/R9k).
///
/// Both halves of the pairing are answered before this runs: the registration
/// gate paired the contract's declarations with the reviewed module's own
/// `[[buffer(N)]]` arguments ([`validate_reviewed_stage_buffers`]), and
/// `validate_against` paired the pass's views with the declarations — so every
/// view here is one the module reads or writes, at its own slot, with the
/// declared extent covering the module's reach. What this walk adds is the
/// rail's own three-armed source channel ([`resolve_render_input`]): a binding
/// declares its bytes, names a staged lease, or names an owner window to map,
/// exactly as a vertex stream does, and the proof repeats the vertex rule that
/// the *resolved* input covers the module's reach — a footprint proved over a
/// copy could pass while the device reads different bytes.
///
/// An affine reach is evaluated over the draw's own counts by the same
/// arithmetic the core contract uses (`render_affine_axis_counts` /
/// `render_affine_required_bytes`, `crates/metal-api-core/src/provider.rs`,
/// `research/docs/23` §3.3, v86): axis 0 is `vertices` for a non-indexed draw
/// and `base_vertex + highest index + 1` for an indexed one — the highest index
/// read from the *resolved* index bytes the vertex-stream proof already
/// resolved ([`PlannedIndexStream::vertex_span`]), so the two proofs share one
/// measurement of the draw — with axis 1 the pass's instance count. An
/// expression that cannot be evaluated here (one that overflows `u64`) is
/// refused by name rather than admitted against a bound nothing states.
fn plan_stage_buffers<'a>(
    pass: &'a RenderPassDescriptor,
    module: &ReviewedModule,
    leases: Option<&RenderLeaseContext<'_>>,
    indices: Option<&PlannedIndexStream<'a>>,
    affine_counts: Option<[u64; 2]>,
) -> Result<Vec<PlannedStageBuffer<'a>>, ProviderError> {
    // The two invocation counts an affine reach is bounded by
    // (`render_affine_axis_counts`): the vertex indices the draw can name and
    // its instances. `affine_counts` is the contract's own answer, stated by
    // [`RenderPipelineContract::affine_axis_counts`] over the index bytes this
    // rail resolved ([`resolve_affine_index_bytes`]), so the rail's proof and
    // the contract's bound are one arithmetic. A pass whose declarations
    // evaluate no index arithmetic falls back to the span
    // [`plan_index_stream`] already measured — the same numbers, from the same
    // bytes — and a pass with no index at all counts `vertices` itself.
    let counts = match (affine_counts, indices) {
        (Some(counts), _) => Some(counts),
        (None, Some(indices)) => indices
            .base_vertex
            .checked_add(indices.vertex_span)
            .map(|vertices| [vertices, u64::from(pass.instance_count)]),
        (None, None) => Some([u64::from(pass.vertices), u64::from(pass.instance_count)]),
    };
    let mut bindings = Vec::with_capacity(pass.stage_buffers.len());
    for binding in &pass.stage_buffers {
        // The slot the registration paired this declaration with. Both walks
        // run before any plan exists (`validate_against` pairs the pass with
        // the declarations, `review_contract` pairs the declarations with the
        // module), so a missing slot is refused rather than unwrapped — a rail
        // that cannot name the module's own read never binds the slot.
        let slot = module
            .stage_buffers
            .iter()
            .find(|slot| slot.stage == binding.stage && slot.index == binding.view.metal_binding)
            .ok_or_else(|| {
                capability_refusal("render_stage_reflection_mismatch")
                    .with_field("stage", FieldValue::Text(binding.stage.name().to_owned()))
                    .with_field(
                        "index",
                        FieldValue::Unsigned(u64::from(binding.view.metal_binding)),
                    )
                    .with_field("field", FieldValue::Text("bindings".to_owned()))
                    .with_detail(
                        "the reviewed module reads no `[[buffer(N)]]` argument at this stage's \
                         slot, so the rail has no read extent the binding's bytes were proven \
                         against",
                    )
            })?;
        let source = resolve_render_input(&binding.view, leases, RenderInputRole::StageBuffer)?;
        // The module's own reach over this draw: a fixed extent as it stands,
        // an affine one evaluated over the counts above. An evaluation that
        // overflows is a proof this rail refuses by name, exactly as the
        // contract refuses the evaluation it cannot state.
        let required = counts
            .and_then(|counts| slot.reach.required_bytes(counts))
            .ok_or_else(|| {
                capability_refusal("render_stage_buffer_footprint_unsupported")
                    .with_field("stage", FieldValue::Text(binding.stage.name().to_owned()))
                    .with_field(
                        "binding",
                        FieldValue::Unsigned(u64::from(binding.view.metal_binding)),
                    )
                    .with_field("view", FieldValue::Unsigned(binding.view.view_id.get()))
                    .with_detail(
                        "the reviewed module's affine reach cannot be evaluated over this draw's \
                         own counts, so the rail has no bound the binding's bytes are proven \
                         against",
                    )
            })?;
        if u64::try_from(source.len()).unwrap_or(u64::MAX) < required {
            return Err(
                capability_refusal("render_stage_buffer_footprint_unsupported")
                    .with_field("stage", FieldValue::Text(binding.stage.name().to_owned()))
                    .with_field(
                        "binding",
                        FieldValue::Unsigned(u64::from(binding.view.metal_binding)),
                    )
                    .with_field("view", FieldValue::Unsigned(binding.view.view_id.get()))
                    .with_field("required_bytes", FieldValue::Unsigned(required))
                    .with_field(
                        "resolved_bytes",
                        FieldValue::Unsigned(u64::try_from(source.len()).unwrap_or(u64::MAX)),
                    )
                    .with_detail(
                        "the reviewed module's reach over this draw passes the bytes this rail \
                         resolved for the binding, and the device reads the resolved mapping \
                         rather than a copy of it",
                    ),
            );
        }
        bindings.push(PlannedStageBuffer {
            stage: binding.stage,
            index: binding.view.metal_binding,
            bytes: required,
            source,
            offset: binding.view.offset,
            view_id: binding.view.view_id,
            allocation_id: binding.view.allocation_id,
            access: binding.view.access,
            length: binding.view.length,
        });
    }
    Ok(bindings)
}

/// Which of a pass's render inputs a refusal is about.
#[derive(Clone, Copy)]
enum RenderInputRole {
    /// A vertex stream of the pass's layout.
    Vertex,
    /// The pass's index buffer.
    Index,
    /// The previous contents a `LoadOp::Load` attachment uploads
    /// (`research/docs/23` §74, R5b).
    Attachment,
    /// A sampled texture the pass's fragment stage reads
    /// (`research/docs/23` §75, R5c).
    Texture,
    /// A stage buffer the pipeline declares and the pass binds
    /// (`research/docs/23` §83, R9g).
    StageBuffer,
}

impl RenderInputRole {
    /// The capability slug this role's unreadable source is refused with. The
    /// two stream names are the ones this rail published before the lease
    /// channel existed, so a capture that could not read a stream keeps its
    /// slug; the attachment name is the one the Vulkan rail publishes for the
    /// same source arm, and the texture role keeps the sampler's own source
    /// name, so both rails answer a capture with one name
    /// (`research/docs/23` §75, R5c).
    const fn slug(self) -> &'static str {
        match self {
            Self::Vertex => VERTEX_SLUG,
            Self::Index => INDEX_SLUG,
            Self::Attachment => ATTACHMENT_LOAD_SLUG,
            Self::Texture => TEXTURE_SLUG,
            Self::StageBuffer => STAGE_BUFFER_SLUG,
        }
    }
}

/// The lease channel one render plan resolves its inputs through
/// (`research/docs/23` §72, R3d).
///
/// The two registries are the provider's own — the same pair the compute rail
/// resolves its pool views through, so an imported lease is one object with one
/// retain count — and the admitted snapshot is the authority on which
/// reservations exist: [`LeaseRegistry::view_bytes`] and
/// [`BorrowedLeaseRegistry::view_pointer`] compare the imported reservation
/// against it and refuse a view that falls outside it. The alignment is the
/// device's own `newBufferWithBytesNoCopy:` requirement, or zero when this
/// device cannot map an owner window at all.
pub(crate) struct RenderLeaseContext<'a> {
    pub(crate) staging: &'a LeaseRegistry,
    pub(crate) borrowed: &'a Arc<BorrowedLeaseRegistry>,
    pub(crate) resources: &'a ResourceTableSnapshot,
    pub(crate) device_epoch: DeviceEpoch,
    pub(crate) host_import_alignment: u64,
}

/// Where one render input's bytes come from (`research/docs/23` §72, R3d).
#[derive(Clone, Debug)]
pub(crate) enum PlannedInputSource<'a> {
    /// The bytes the view itself declares (`BufferSource::OwnedBytes`),
    /// unchanged from the pre-lease increments. The encoder uploads them into a
    /// buffer that starts at the view.
    Declared(&'a [u8]),
    /// The provider's staged copy of an owner lease
    /// (`BufferSource::StagedLease`); the encoder uploads it exactly like
    /// declared bytes, but the bytes are one submission's copy of the owner's
    /// window rather than the trace's own.
    Staged(Vec<u8>),
    /// The owner's own mapping (`BufferSource::BorrowedNoCopy`): the encoder
    /// maps the whole reservation with `newBufferWithBytesNoCopy:` and binds
    /// this view at [`BorrowedView::offset`] inside it, so no byte is copied.
    NoCopy {
        lease: LeaseId,
        window: BorrowedView,
    },
}

impl PlannedInputSource<'_> {
    /// The bytes the rail's footprint proof reads.
    ///
    /// A no-copy window is read through the owner's mapping, because that is
    /// where the proof's bytes are: the import contract keeps the mapping
    /// readable at this address until the provider releases the import, so this
    /// reads the same bytes the device will read rather than a copy of them.
    /// A snapshot-style implementation cannot pass a proof taken this way.
    fn proof_bytes(&self) -> &[u8] {
        match self {
            Self::Declared(bytes) => bytes,
            Self::Staged(bytes) => bytes,
            // SAFETY: the window was resolved by the no-copy registry for an
            // imported lease, whose contract keeps the owner's mapping readable
            // over exactly this window until the import is released.
            Self::NoCopy { window, .. } => unsafe {
                std::slice::from_raw_parts(window.pointer as *const u8, window.len)
            },
        }
    }

    /// The length of the window this input binds.
    fn len(&self) -> usize {
        match self {
            Self::Declared(bytes) => bytes.len(),
            Self::Staged(bytes) => bytes.len(),
            Self::NoCopy { window, .. } => window.len,
        }
    }

    /// The no-copy lease this input reads, when it is one. Every other arm has
    /// no owner mapping to retain.
    const fn borrowed_lease(&self) -> Option<LeaseId> {
        match self {
            Self::Declared(_) | Self::Staged(_) => None,
            Self::NoCopy { lease, .. } => Some(*lease),
        }
    }
}

/// The offset the encoder binds one input at.
///
/// An uploaded input carries the view's bytes placed at the view's own offset
/// inside its allocation, and the binding uses that same offset — the
/// convention the compute pool's merged images follow (`native.rs`: an
/// allocation image is bound at `view.offset`). A no-copy input is bound at its
/// offset inside the owner reservation the encoder maps, because that mapping
/// (not the view) is what starts at address zero. Both arms therefore start the
/// binding on the first byte the trace named instead of re-basing it at zero.
fn source_binding_offset(source: &PlannedInputSource<'_>, view_offset: u64) -> u64 {
    match source {
        PlannedInputSource::Declared(_) | PlannedInputSource::Staged(_) => view_offset,
        PlannedInputSource::NoCopy { window, .. } => {
            u64::try_from(window.offset).unwrap_or(u64::MAX)
        }
    }
}

/// Resolve one render input's source into the bytes the rail will bind
/// (`research/docs/23` §72, R3d).
///
/// `OwnedBytes` resolves to the view's own bytes exactly as before. A
/// `StagedLease` resolves through the provider's [`LeaseRegistry`], which holds
/// the owner's staged copy; the rail uploads those bytes into a buffer of its
/// own. A `BorrowedNoCopy` resolves through the shared
/// [`BorrowedLeaseRegistry`], which hands back the owner's address and never
/// copies. Every unresolvable arm is refused by name before any Metal object
/// exists: a submission with no lease channel, a device that cannot map owner
/// memory, a reservation or mapping that misses the import's alignment rules,
/// and the registry's own `lease_not_imported` / `lease_not_admitted` /
/// `lease_snapshot_mismatch` / `lease_epoch_mismatch` /
/// `lease_range_out_of_bounds`.
fn resolve_render_input<'a>(
    view: &'a BufferView,
    leases: Option<&RenderLeaseContext<'_>>,
    role: RenderInputRole,
) -> Result<PlannedInputSource<'a>, ProviderError> {
    match &view.source {
        BufferSource::OwnedBytes(bytes) => Ok(PlannedInputSource::Declared(bytes)),
        // The multi-window guest source (`research/docs/23` §74, E-TX6) is
        // refused by name on this rail for the reason every unflipped arm is:
        // the arm's reading is the gather the Vulkan rail performs out of the
        // owner's imported mappings, and the Apple-side reading its flip would
        // owe — a host plan that reads several windows of one view, plus the
        // device evidence behind it — is a later increment. Stating the
        // boundary keeps a native caller from getting a different meaning for
        // the same declaration instead of a refusal.
        BufferSource::GuestRuns(_) => Err(render_input_refusal(
            role,
            view.view_id,
            "guest_runs",
            "the bytes are an ordered list of the owner's guest runs, and this rail does not \
             read the guest-runs arm yet: the arm is executed by the Vulkan rail \
             (`research/docs/23` §74, E-TX6), and the Apple-side reading its flip would owe is \
             a later increment",
        )),
        BufferSource::StagedLease(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    role,
                    view.view_id,
                    "staged_lease",
                    "the render submission carries no lease channel, so a lease-backed render \
                     input cannot be read",
                )
            })?;
            let bytes = leases.staging.view_bytes(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            Ok(PlannedInputSource::Staged(bytes))
        }
        BufferSource::BorrowedNoCopy(lease_id) => {
            let leases = leases.ok_or_else(|| {
                render_input_refusal(
                    role,
                    view.view_id,
                    "borrowed_no_copy",
                    "the render submission carries no lease channel, so a lease-backed render \
                     input cannot be read",
                )
            })?;
            if leases.host_import_alignment == 0 {
                return Err(capability_refusal("storage_mode_unsupported")
                    .with_field("view", FieldValue::Unsigned(view.view_id.get()))
                    .with_field(
                        "storage_mode",
                        FieldValue::Text("borrowed_no_copy".to_owned()),
                    )
                    .with_detail(
                        "this device cannot map an owner window, so a no-copy render input has \
                         no path through this rail",
                    ));
            }
            let window = leases.borrowed.view_pointer(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            check_no_copy_window(*lease_id, window, leases.host_import_alignment)?;
            Ok(PlannedInputSource::NoCopy {
                lease: *lease_id,
                window,
            })
        }
    }
}

/// Resolve one sampled texture's bytes into the source the encoder uploads
/// (`research/docs/23` §75, R5c).
///
/// The render sampler's third declaration of the same three arms: a texture
/// carries its whole byte extent (no offset and length, unlike a buffer view),
/// so the lease window is the texture's own tightly packed extent at the
/// reservation's start — the window rule core's registries state for
/// `TextureSource`. `OwnedBytes` behaves byte for byte as before, a
/// `StagedLease` uploads the provider's own copy of the owner's window, and a
/// `BorrowedNoCopy` uploads the owner's own pages, so a rewrite after the
/// import reaches the sampled texels instead of leaving the import's first copy
/// behind.
///
/// The widened `TextureSource` arm this rail does not carry is the
/// trace-produced one (`research/docs/23` §110, E-TX3): it is refused by name
/// rather than resolved through a guess, exactly as the lease arms are refused
/// when the submission carries no lease channel.
fn resolve_render_texture_source<'a>(
    view: &'a TextureView,
    leases: Option<&RenderLeaseContext<'_>>,
    binding: usize,
) -> Result<PlannedInputSource<'a>, ProviderError> {
    match &view.source {
        TextureSource::OwnedBytes(bytes) => Ok(PlannedInputSource::Declared(bytes)),
        // The trace's own production (`research/docs/23` §110, E-TX3) is
        // refused by name on this rail for the reason every unflipped arm is:
        // the arm's resolution needs the trace's own writebacks, and the
        // Apple-side execution that would carry them has no reading yet. The
        // Vulkan rail executes the arm; this rail states the boundary instead
        // of uploading bytes no pass of this submission produced.
        TextureSource::TraceView => Err(texture_source_refusal(
            binding,
            view.view_id,
            "trace_view",
            "the texels are the trace's own earlier GPU output, and this rail's trace path \
             does not resolve the produced-bytes arm yet: the arm is executed by the Vulkan \
             rail (`research/docs/23` §110), and the Apple-side reading its flip would owe \
             is a later increment",
        )),
        // The pass-entry snapshot arm (`research/docs/23` §118, E-TX15) is
        // refused by name for the reason every unflipped arm is, plus one of
        // its own: what the arm promises is a device-side image copy taken
        // before the render pass opens and bound as the sampled view, and this
        // rail has no Apple oracle for a fragment reading the attachment the
        // same pass writes — the reading its flip would owe is a later
        // increment's. The bit stays `false` beside it, so a consumer refuses
        // the declaration during admission rather than reaching this walk.
        TextureSource::PassEntrySnapshot => Err(texture_source_refusal(
            binding,
            view.view_id,
            "pass_entry_snapshot",
            "the texels are the pass's own colour attachment as it stands when the pass \
             opens, and this rail's trace path does not carry the before-the-pass image copy \
             the arm states: the arm is executed by the Vulkan rail (`research/docs/23` \
             §118), and Apple has no oracle for the shape",
        )),
        TextureSource::StagedLease(lease_id) => {
            let leases = leases.ok_or_else(|| {
                texture_source_refusal(
                    binding,
                    view.view_id,
                    "staged_lease",
                    "the render submission carries no lease channel, so a lease-backed render \
                     texture cannot be read",
                )
            })?;
            let bytes = leases.staging.texture_bytes(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            Ok(PlannedInputSource::Staged(bytes))
        }
        TextureSource::BorrowedNoCopy(lease_id) => {
            let leases = leases.ok_or_else(|| {
                texture_source_refusal(
                    binding,
                    view.view_id,
                    "borrowed_no_copy",
                    "the render submission carries no lease channel, so a lease-backed render \
                     texture cannot be read",
                )
            })?;
            if leases.host_import_alignment == 0 {
                return Err(capability_refusal("storage_mode_unsupported")
                    .with_field("view", FieldValue::Unsigned(view.view_id.get()))
                    .with_field(
                        "storage_mode",
                        FieldValue::Text("borrowed_no_copy".to_owned()),
                    )
                    .with_detail(
                        "this device cannot read an owner window, so a no-copy render texture \
                         has no path through this rail",
                    ));
            }
            let window = leases.borrowed.texture_pointer(
                *lease_id,
                view,
                leases.device_epoch,
                leases.resources,
            )?;
            check_no_copy_window(*lease_id, window, leases.host_import_alignment)?;
            Ok(PlannedInputSource::NoCopy {
                lease: *lease_id,
                window,
            })
        }
    }
}

/// One sampled texture's source refusal (`research/docs/23` §75, R5c).
///
/// The fields are the sampler's own — the binding index the view's own label
/// holds it to, the view's identity and the storage mode it arrived under — so
/// a capture reads the same two names the Vulkan rail publishes.
fn texture_source_refusal(
    binding: usize,
    view: ViewId,
    storage_mode: &'static str,
    detail: &'static str,
) -> ProviderError {
    capability_refusal(TEXTURE_SLUG)
        .with_field("binding", FieldValue::Unsigned(binding as u64))
        .with_field("view", FieldValue::Unsigned(view.get()))
        .with_field("storage_mode", FieldValue::Text(storage_mode.to_owned()))
        .with_detail(detail)
}

/// The whole-page rules every no-copy window has to meet
/// (`research/docs/23` §72/§75).
///
/// `newBufferWithBytesNoCopy:` maps whole pages: the reservation's base address
/// and length have to sit on the device's import alignment, and the view inside
/// it on the 4-byte rule the compute rail states for the same mapping. The
/// first two are also what `import_borrowed_lease` checked for this provider;
/// re-asking them here keeps the refusal beside the window that would be
/// mapped, and the checks are the ones the encoder body depends on.
fn check_no_copy_window(
    lease_id: LeaseId,
    window: BorrowedView,
    host_import_alignment: u64,
) -> Result<(), ProviderError> {
    if !u64::try_from(window.base_len)
        .unwrap_or(u64::MAX)
        .is_multiple_of(host_import_alignment)
    {
        return Err(crate::lease_length_refusal(
            lease_id,
            u64::try_from(window.base_len).unwrap_or(u64::MAX),
            host_import_alignment,
        ));
    }
    let alignment = usize::try_from(host_import_alignment).unwrap_or(usize::MAX);
    if !window.base_pointer.is_multiple_of(alignment) {
        return Err(crate::lease_alignment_refusal(
            lease_id,
            window.base_pointer,
            host_import_alignment,
        ));
    }
    if !window.offset.is_multiple_of(4) {
        return Err(crate::lease_offset_refusal(
            lease_id,
            u64::try_from(window.offset).unwrap_or(u64::MAX),
        ));
    }
    Ok(())
}

/// The storage modes a stream view can carry, as the refusal spells them.
fn storage_mode_name(source: &BufferSource) -> &'static str {
    match source {
        BufferSource::OwnedBytes(_) => "owned_bytes",
        BufferSource::StagedLease(_) => "staged_lease",
        BufferSource::BorrowedNoCopy(_) => "borrowed_no_copy",
        // The guest-runs arm this rail refuses by name
        // (`research/docs/23` §74, E-TX6).
        BufferSource::GuestRuns(_) => "guest_runs",
    }
}

fn vertex_footprint_refusal(
    buffer_index: usize,
    stride: u64,
    covered: u64,
    available: usize,
) -> ProviderError {
    capability_refusal("render_vertex_footprint_unsupported")
        .with_field(
            "buffer_index",
            FieldValue::Unsigned(u64::try_from(buffer_index).unwrap_or(u64::MAX)),
        )
        .with_field("stride", FieldValue::Unsigned(stride))
        .with_field("required_bytes", FieldValue::Unsigned(covered))
        .with_field(
            "available_bytes",
            FieldValue::Unsigned(u64::try_from(available).unwrap_or(u64::MAX)),
        )
        .with_detail(
            "the declared vertex view does not cover every vertex the draw reads; a bound \
             stream would be read past its end",
        )
}

fn index_footprint_refusal(width: usize, required: usize, available: usize) -> ProviderError {
    capability_refusal("render_index_footprint_unsupported")
        .with_field(
            "index_bytes",
            FieldValue::Unsigned(u64::try_from(width).unwrap_or(u64::MAX)),
        )
        .with_field(
            "required_bytes",
            FieldValue::Unsigned(u64::try_from(required).unwrap_or(u64::MAX)),
        )
        .with_field(
            "available_bytes",
            FieldValue::Unsigned(u64::try_from(available).unwrap_or(u64::MAX)),
        )
        .with_detail("the declared index view does not cover the indices the draw consumes")
}

fn index_value_refusal(highest: u32, vertices_covered: u64) -> ProviderError {
    capability_refusal("render_index_value_out_of_range")
        .with_field("highest_index", FieldValue::Unsigned(u64::from(highest)))
        .with_field("vertices_covered", FieldValue::Unsigned(vertices_covered))
        .with_detail(
            "an index selects a vertex the bound streams do not cover; the driver would \
             read outside the stream instead of refusing",
        )
}

/// Slug of a vertex stream this rail cannot read.
const VERTEX_SLUG: &str = "render_vertex_buffer_unsupported";

/// Slug of an index stream this rail cannot read.
const INDEX_SLUG: &str = "render_index_buffer_unsupported";

/// Slug of a loading attachment's previous contents this rail cannot read
/// (`research/docs/23` §74, R5b).
///
/// The two stream slugs above are the names this rail published before the
/// lease channel existed; an attachment's previous contents are a third
/// declaration the rail resolves through the same channel, so they get their
/// own name rather than sharing the load operation's
/// (`attachment_load_op_unsupported`, which still refuses a `Load` with no
/// declaration at all).
const ATTACHMENT_LOAD_SLUG: &str = "render_attachment_load_source_unsupported";

/// Slug of a sampled texture this rail cannot read (`research/docs/23` §75,
/// R5c).
///
/// The render sampler's source arm is the name the Vulkan rail published for
/// the same fact before the lease channel reached textures, so the two rails
/// keep answering a capture with one name: a texture whose bytes could not be
/// resolved is refused under it, exactly as the attachment's previous contents
/// are refused under their own.
const TEXTURE_SLUG: &str = "render_texture_source_unsupported";

/// Slug of a stage buffer this rail cannot read (`research/docs/23` §83, R9g).
///
/// The name the Vulkan rail publishes for the same fact, so both rails answer
/// a capture whose stage buffer's bytes could not be resolved with one slug.
const STAGE_BUFFER_SLUG: &str = "render_stage_buffer_source_unsupported";

/// One offscreen render pass to execute.
///
/// The pass and the pipeline are the core values themselves, so this rail cannot
/// drift from the contract's field set: a field the contract adds is a field
/// admission already validates.
pub(crate) struct OffscreenRenderRequest<'a> {
    /// The trace's render pass entry.
    pub(crate) pass: &'a RenderPassDescriptor,
    /// The registered render pipeline the pass names.
    pub(crate) pipeline: &'a RenderPipelineContract,
    /// The MSL module to compile. The pipeline's [`VertexLayout`] and
    /// `color_formats` select which reviewed module is the only one accepted
    /// here — `vertex_id` positions ([`REVIEWED_SOURCE`]), single-output
    /// `[[stage_in]]` positions ([`REVIEWED_VERTEX_SOURCE`]) or the dual
    /// `[[stage_in]]` module ([`REVIEWED_DUAL_SOURCE`]) — so a caller cannot
    /// pair one shape's descriptor with another shape's module.
    pub(crate) source: &'a str,
    /// One entry per colour attachment, in location order: where the tightly
    /// packed texels that attachment already holds come from, for
    /// [`LoadOp::Load`]. Required exactly then, refused for a clear. The source
    /// is resolved by the caller ([`resolve_render_input`] through the lease
    /// channel) or, on the trace path, by [`plan_trace_with_leases`] from the
    /// attachment's declaring view (`research/docs/23` §74, R5b).
    pub(crate) initial: Vec<Option<PlannedInputSource<'a>>>,
    /// The provider's answer for the attachment's resident declaration, one
    /// entry per colour attachment in location order (`research/docs/23` §76,
    /// R7): `true` where the provider resolved a live image for that
    /// attachment's own `(allocation, view)` identity. The list is empty for
    /// every caller with no resident registry — the shape every pre-R7 caller
    /// states — and a pass that declares the resident target then is refused by
    /// name, so the arm can never be executed as a fresh per-pass attachment
    /// the trace did not ask for. A non-empty list has to carry one entry per
    /// colour attachment, exactly as [`Self::initial`] does.
    pub(crate) resident: Vec<bool>,
}

/// One sampled texture a render pass binds (`research/docs/23` §3.3, v70): the
/// source of its tightly packed texel bytes, the extent the pass's render area
/// shares with it, and the Metal index the fragment stage's own
/// `[[texture(n)]]` argument names (`v104`) — the index the encoder binds it
/// at, rather than the entry's position in the list. The source's three arms
/// are the three
/// [`TextureSource`] arms, resolved before any Metal object exists
/// (`research/docs/23` §75, R5c).
#[derive(Debug)]
pub(crate) struct PlannedTexture<'a> {
    pub(crate) source: PlannedInputSource<'a>,
    pub(crate) extent: [u32; 2],
    pub(crate) binding: u32,
}

/// Everything the encoder needs, decided before the first Metal object exists.
#[derive(Debug)]
pub(crate) struct RenderPlan<'a> {
    /// The reviewed module this plan compiles. The entry pair below is read out
    /// of the contract, which [`review_contract`] has already compared with the
    /// module, so the two cannot disagree about what runs.
    pub(crate) source: &'a str,
    /// The module's path, for diagnostics.
    pub(crate) module_path: &'static str,
    pub(crate) vertex_entry: &'a str,
    pub(crate) fragment_entry: &'a str,
    /// One entry per colour attachment, in location order: the pixel format,
    /// the load/store actions and the previous bytes the encoder writes into
    /// each attachment before the pass opens.
    pub(crate) attachments: Vec<PlannedAttachment<'a>>,
    /// The pass's sampled textures, in canonical binding order
    /// (`research/docs/23` §3.3, v70/v104): the reviewed sampling pair's one
    /// `rgba8_unorm` surface whose extent is the render area's own, carried as
    /// the bytes the encoder uploads into its own `MTLTexture` and bound at the
    /// Metal index the entry states. Empty for every pre-v70 plan.
    pub(crate) textures: Vec<PlannedTexture<'a>>,
    /// Attachment extent in texels, as `[width, height]`.
    pub(crate) extent: [u32; 2],
    /// `[origin_x, origin_y, width, height]`, copied from the validated pass.
    pub(crate) viewport: [u32; 4],
    /// The pass's scissor rectangle, or `None` for the whole viewport
    /// (`research/docs/23` §3.3, v29).
    pub(crate) scissor: Option<[u32; 4]>,
    /// The pass's culling state, or `None` for "keep every triangle"
    /// (`research/docs/23` §3.3, v39).
    pub(crate) cull: Option<RenderPassCull>,
    /// The pass's blend state, or `None` for "write the fragment output"
    /// (`research/docs/23` §3.3, v40).
    pub(crate) blend: Option<RenderPassBlend>,
    /// The pass-wide multisample raster (`research/docs/23` §3.3, v51), or
    /// `None` for the single-sample raster every pre-v51 pass ran. When
    /// present, the encoder creates a four-sample texture per colour location,
    /// renders into it and resolves it into the attachment's own shared
    /// texture with `storeAction = .multisampleResolve`; the readback observes
    /// the resolve target exactly as the trace's attachment view.
    pub(crate) multisample: Option<SampleCount>,
    /// The rail-owned depth attachment this pass opens, or `None` for a pass
    /// with no depth surface (`research/docs/23` §3.3, v36).
    pub(crate) depth: Option<PlannedDepth>,
    /// The depth resolve a multisampled pass states, carried as the filter the
    /// encoder sets on the depth attachment (`research/docs/23` §3.3, v57c).
    /// `None` is every pass that does not resolve — the pre-v57 shapes and the
    /// single-sample stored depth.
    pub(crate) depth_resolve: Option<DepthResolveFilter>,
    /// The stencil resolve a multisampled pass states, carried as the filter
    /// the encoder sets on the stencil attachment (`research/docs/23` §3.3,
    /// v60). `None` is every pass that does not resolve — the pre-v60 shapes
    /// and the single-sample stored stencil.
    pub(crate) stencil_resolve: Option<StencilResolveFilter>,
    /// The rail-owned stencil attachment this pass opens, or `None` for a pass
    /// with no stencil surface (`research/docs/23` §3.3, v47).
    pub(crate) stencil: Option<PlannedStencil>,
    pub(crate) vertices: u32,
    /// Instances the draw runs (`research/docs/23` §3.3, v31): Metal's
    /// `drawPrimitives(vertexCount:instanceCount:)` second count. `1` for every
    /// pre-v31 pass.
    pub(crate) instance_count: u32,
    /// One entry per bound vertex stream, in binding order, with the sources and
    /// footprints [`plan_vertex_input`] resolved and proved.
    pub(crate) vertex_streams: Vec<PlannedVertexStream<'a>>,
    /// The index buffer of an indexed draw, resolved from the pass's own view.
    pub(crate) indices: Option<PlannedIndexStream<'a>>,
    /// The pass's stage buffers, in the contract's canonical order
    /// (`research/docs/23` §83, R9g): one entry per `[[buffer(N)]]` argument
    /// the reviewed module reads, naming its stage, the slot the encoder binds
    /// and the bytes the module reaches there. Empty for every pass that
    /// declares none, which is every pre-R9g plan.
    pub(crate) stage_buffers: Vec<PlannedStageBuffer<'a>>,
    /// The flat byte shape of the **depth** surface this plan reads back, if
    /// any: `depth32float` is four bytes per texel over the pass extent
    /// (`metal_api_core::provider::DEPTH_BYTES_PER_TEXEL`).
    ///
    /// The colour attachments carry their own [`TexelExtent`] beside their
    /// format (`research/docs/23` §78), because a pass may mix widths; this one
    /// is the depth surface's and the depth resolve target's, which share the
    /// depth format.
    pub(crate) depth_texel: TexelExtent,
}

impl RenderPlan<'_> {
    /// The no-copy leases this pass's inputs read, in binding order
    /// (`research/docs/23` §72/§74, R3d/R5b).
    ///
    /// The list a submission retains before it maps a single owner window and
    /// retires once the pass's command buffer is terminal. One lease named by
    /// two streams appears twice, which is the retain count the registry needs:
    /// both bindings read the same mapping. A loading attachment's previous
    /// contents are the third input that can name one: the encoder uploads them
    /// out of the owner's mapping, and a sampled texture's bytes are the fourth
    /// (`research/docs/23` §75, R5c): the encoder uploads them the same way, one
    /// hold per window, so a pass that binds several textures retains each lease
    /// once per window it appears in.
    pub(crate) fn borrowed_leases(&self) -> Vec<LeaseId> {
        let mut leases = Vec::new();
        for stream in &self.vertex_streams {
            if let Some(lease) = stream.source.borrowed_lease() {
                leases.push(lease);
            }
        }
        if let Some(index) = &self.indices {
            if let Some(lease) = index.source.borrowed_lease() {
                leases.push(lease);
            }
        }
        for stage in &self.stage_buffers {
            if let Some(lease) = stage.source.borrowed_lease() {
                leases.push(lease);
            }
        }
        for attachment in &self.attachments {
            if let Some(source) = &attachment.initial {
                if let Some(lease) = source.borrowed_lease() {
                    leases.push(lease);
                }
            }
        }
        for texture in &self.textures {
            if let Some(lease) = texture.source.borrowed_lease() {
                leases.push(lease);
            }
        }
        leases
    }
}

/// The depth attachment a plan opens (`research/docs/23` §3.3, v36/v43).
///
/// Rail-owned like the Vulkan rail's: no trace identity lives here, so the plan
/// carries the shape the encoder creates and opens, and the trace's own landing
/// is resolved beside it ([`TraceRenderPlan::depth_landing`]). The store action
/// is what decides whether the surface outlives the pass: a storing one is read
/// back through the same `getBytes` shape the colour attachments use, and every
/// pre-v43 shape — no statement at all, or the explicit discard — keeps the
/// surface rail-owned and disappears with the pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedDepth {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `Some(bits)` for a clear (the `f32`'s own bits), `None` for `Load`.
    pub(crate) clear_bits: Option<u32>,
    /// The pass's depth state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub(crate) test: Option<DepthTest>,
    /// The store action the trace stated, or `None` for the pre-v43 shape
    /// (`research/docs/23` §3.3, v43). A storing surface is the only one this
    /// rail reads back, which is also why its texture is created with shared
    /// storage ([`depth_texture`]).
    pub(crate) store: Option<DepthStoreOp>,
}

impl PlannedDepth {
    /// Whether the pass keeps this surface — and therefore reads it back
    /// (`research/docs/23` §3.3, v43).
    pub(crate) fn storing(&self) -> bool {
        self.store == Some(DepthStoreOp::Store)
    }
}

/// The stencil attachment a plan opens (`research/docs/23` §3.3, v47/v49).
///
/// Rail-owned like the depth surface, so the plan carries the shape the encoder
/// creates and opens while the trace's own landing is resolved beside it
/// ([`TraceRenderPlan::stencil_landing`]). The store action is what decides
/// whether the surface outlives the pass: a storing one is read back through
/// the same `getBytes` shape the colour attachments use, one byte per texel,
/// and every pre-v49 shape — no statement at all, or the explicit discard —
/// keeps the surface rail-owned and disappears with the pass. What the trace
/// always states is the extent, the clear value or previous-contents load op,
/// and the stencil state the pass's draw tests and writes with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PlannedStencil {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `Some(value)` for a clear, `None` for `Load`.
    pub(crate) clear_value: Option<u8>,
    /// The pass's stencil state, or `None` for "the attachment exists and
    /// nothing tests it".
    pub(crate) test: Option<StencilTest>,
    /// The store action the trace stated, or `None` for the pre-v49 shape
    /// (`research/docs/23` §3.3, v49). A storing surface is the only one this
    /// rail reads back, which is also why its texture is created with shared
    /// storage ([`stencil_texture`]).
    pub(crate) store: Option<StoreOp>,
}

impl PlannedStencil {
    /// Whether the pass keeps this surface — and therefore reads it back
    /// (`research/docs/23` §3.3, v49).
    pub(crate) fn storing(&self) -> bool {
        self.store == Some(StoreOp::Store)
    }
}

/// The flat byte shape of one texture this rail reads back, uploads through
/// `replaceRegion` or hands to `getBytes`.
///
/// Per format, not per rail: a pass's attachments do not have to share a texel
/// width — `Rgba16Float` stores eight bytes per texel beside the four-byte
/// formats (`research/docs/23` §78) — so the byte extent and the row pitch are
/// answered from the format that carries them. The pass extent is shared, so
/// the two numbers are pure functions of `(extent, bytes per texel)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TexelExtent {
    /// Tightly packed bytes over the whole extent.
    pub(crate) bytes: usize,
    /// Bytes of one texture row.
    pub(crate) row_pitch: usize,
}

impl TexelExtent {
    /// The flat extent of one `extent`-sized surface of `bytes_per_texel`-byte
    /// texels, or `None` when the arithmetic leaves the host's `usize`.
    pub(crate) fn flat(extent: [u32; 2], bytes_per_texel: u64) -> Option<Self> {
        let texels = u64::from(extent[0]).checked_mul(u64::from(extent[1]))?;
        Some(Self {
            bytes: usize::try_from(texels.checked_mul(bytes_per_texel)?).ok()?,
            row_pitch: usize::try_from(u64::from(extent[0]).checked_mul(bytes_per_texel)?).ok()?,
        })
    }
}

/// One colour attachment of a planned pass, resolved before any Metal object
/// exists.
#[derive(Debug)]
pub(crate) struct PlannedAttachment<'a> {
    /// The pixel format this rail builds the attachment's texture and pipeline
    /// state with.
    pub(crate) format: RenderPixelFormat,
    /// The flat byte shape of this attachment's own texture: its format's texel
    /// width times the pass extent, and one row of it (`research/docs/23` §78).
    ///
    /// Per attachment rather than per plan, because a pass may carry an
    /// eight-byte `Rgba16Float` location beside four-byte ones: the readback,
    /// the `Load` upload and the row pitch `getBytes`/`replaceRegion` take are
    /// all facts of the attachment's own format.
    pub(crate) texel: TexelExtent,
    /// The clear value or previous contents the attachment starts from.
    pub(crate) load: RenderLoadAction,
    /// The store action the encoder sets: `Store` keeps the attachment's texels
    /// where they are — in a texture this rail created for a trace-declared
    /// store, and in the provider's own image for a resident store
    /// (`research/docs/23` §76, R7) — while `DontCare` discards the attachment.
    pub(crate) store: RenderStoreAction,
    /// Whether the attachment's bytes leave through the buffer writeback
    /// channel, which is exactly the trace-declared [`StoreOp::Store`]
    /// (`research/docs/23` §3.6/§76): a resident store keeps them in the
    /// provider's image and a discarded attachment drops them, so neither
    /// produces a readback and neither lands a writeback. The encoder's
    /// readback list and [`TraceRenderPlan::writebacks`] both read this bit, so
    /// the two cannot disagree about which attachments left the pass.
    pub(crate) publishes: bool,
    /// Whether the attachment renders into the provider's resident image for
    /// its own `(allocation, view)` identity rather than a fresh per-pass
    /// texture (`research/docs/23` §76, R7). The provider hands that texture to
    /// [`encode_offscreen_render_with_resident`]; a plan that declares one and
    /// is encoded without it is refused by name instead of rendered into a
    /// fresh image the trace never named.
    pub(crate) resident: bool,
    /// The tightly packed texels a [`LoadOp::Load`] uploads before the pass
    /// opens, resolved to the window they come from (`research/docs/23` §74,
    /// R5b); `None` for a clear. A no-copy source keeps the owner's own
    /// mapping, so the upload reads the pages the footprint proof read and the
    /// plan names the lease its submission has to retain.
    pub(crate) initial: Option<PlannedInputSource<'a>>,
    /// The one colour a multisampled `Load` seeds every sample of the
    /// attachment with before the measured pass opens it
    /// (`research/docs/23` §82, v82), in the pass's own component order.
    ///
    /// Metal's load route cannot upload into a multisampled texture either —
    /// a blit copy is single-sample at both ends, exactly as the Vulkan rail's
    /// copy is — so the encoder records a seed render pass whose load action is
    /// `Clear` and whose store action keeps the seeded samples, and the
    /// measured pass then opens the texture with `MTLLoadAction::Load`. A clear
    /// value is one colour for the whole attachment, so the declared window has
    /// to be one repeated texel and has to come from a window this submission
    /// owns; both deviations are refused by name
    /// ([`multisample_seed`]). `Some` exactly for that shape.
    pub(crate) seed: Option<[f64; 4]>,
}

impl PlannedAttachment<'_> {
    /// The bytes the encoder uploads for a loading attachment: the resolved
    /// window itself, which for a no-copy source is the owner's own mapping
    /// (`research/docs/23` §74, R5b).
    pub(crate) fn initial_bytes(&self) -> Option<&[u8]> {
        self.initial.as_ref().map(|source| source.proof_bytes())
    }
}

/// The one colour a multisampled `Load` seeds every sample with
/// (`research/docs/23` §82, v82).
///
/// Metal's copy commands are single-sample at both ends exactly as Vulkan's
/// are (`replaceRegion` into a multisampled texture is not a route this rail
/// takes either), so the load is executed by a *seed pass*: the encoder opens
/// the same multisampled texture with `MTLLoadAction::Clear` — which writes the
/// clear colour to every sample — stores it, and the measured pass then opens
/// it with `MTLLoadAction::Load`. The clear colour is one value for the whole
/// attachment, so the declaration has to be one repeated texel of its own
/// format width, read from a window this submission owns.
///
/// The two deviations carry the Vulkan rail's own names, so a capture can read
/// the same slug on either rail (`research/docs/23` §82):
///
/// - a window that is not one repeated texel
///   (`render_multisample_load_nonuniform_unsupported`): a per-texel seed needs
///   a full-coverage fragment shader this rail does not own — the reviewed
///   modules are the trace's, not the rail's;
/// - an owner's own mapping (`render_multisample_load_borrowed_unsupported`):
///   the seed is host state read before the encoder exists, so a borrowed
///   window would be a snapshot of the owner's pages rather than a device read
///   of them. The staged arm states the same bytes through the provider's own
///   copy.
fn multisample_seed(
    source: &PlannedInputSource<'_>,
    attachment: usize,
    bytes_per_texel: u64,
    format: RenderPixelFormat,
) -> Result<[f64; 4], ProviderError> {
    if matches!(source, PlannedInputSource::NoCopy { .. }) {
        return Err(
            capability_refusal("render_multisample_load_borrowed_unsupported")
                .with_field("attachment", FieldValue::Unsigned(attachment as u64))
                .with_field(
                    "storage_mode",
                    FieldValue::Text("borrowed_no_copy".to_owned()),
                )
                .with_detail(
                    "a multisampled `Load` is executed by a seed pass whose clear value is host \
                     state; a borrowed seed would be read here instead of by the device at \
                     execution",
                ),
        );
    }
    let bytes = source.proof_bytes();
    let width = usize::try_from(bytes_per_texel).unwrap_or(usize::MAX);
    if width == 0 || bytes.len() < width {
        return Err(args_refusal("render_attachment_initial_mismatch")
            .with_field("attachment", FieldValue::Unsigned(attachment as u64))
            .with_field("byte_length", FieldValue::Unsigned(bytes.len() as u64))
            .with_detail("a multisampled load's seed window holds at least one texel"));
    }
    let (texel, rest) = bytes.split_at(width);
    if rest.chunks(width).any(|chunk| chunk != texel) {
        return Err(
            capability_refusal("render_multisample_load_nonuniform_unsupported")
                .with_field("attachment", FieldValue::Unsigned(attachment as u64))
                .with_detail(
                    "the seed route states one clear value for the whole raster, so the loaded \
                     window has to be one repeated texel; a per-texel seed needs a \
                     full-coverage fragment shader this rail does not own",
                ),
        );
    }
    let clear = ClearColor::from_bytes(texel).ok_or_else(|| {
        args_refusal("render_attachment_initial_mismatch")
            .with_field("attachment", FieldValue::Unsigned(attachment as u64))
            .with_field("byte_length", FieldValue::Unsigned(texel.len() as u64))
            .with_detail("a seed texel is one to eight bytes")
    })?;
    Ok(clear_components(clear, format))
}

/// Validate a render request against the contract and the rail's own allowlist.
///
/// Runs entirely without a device, so every refusal here is testable on a host
/// that cannot load Metal. Nothing outside the request is read: the pass carries
/// its own streams' bytes and its own attachments, so this call answers the
/// same way for a trace pass and for the device-level helper's trace-less
/// request.
pub(crate) fn plan<'a>(
    request: &OffscreenRenderRequest<'a>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<RenderPlan<'a>, ProviderError> {
    // A pass that states the texel space is refused here, before any reviewed
    // module is planned (2026-09-19, census v43's `texture_state` axis). The
    // rail's own capability snapshot keeps the bit at its default, so core
    // admission already answers this shape; this second check is the
    // directly-constructed request's, and it names the rail rather than the
    // snapshot. The reviewed MSL modules spell one `constexpr sampler` in the
    // normalized space and take no `[[sampler(n)]]` argument at all, so there
    // is no module here a texel coordinate could be executed against.
    if let Some(sampler) = request
        .pass
        .samplers
        .iter()
        .find(|sampler| sampler.coordinates.is_pixel())
    {
        return Err(
            capability_refusal("render_pixel_coordinate_sampler_unsupported")
                .with_field(
                    "sampler_binding",
                    FieldValue::Unsigned(u64::from(sampler.metal_binding)),
                )
                .with_field("rail", FieldValue::Text("native".to_owned()))
                .with_detail(
                    "this rail's reviewed modules sample through their own `constexpr sampler` in \
                 the normalized space, so a pass whose runtime sampler states the texel space \
                 has no module behind it here; the Vulkan rail's translated arm is where the \
                 space runs, through the fragment module's explicit-LOD sibling",
                ),
        );
    }
    plan_with_leases(request, None, depth_resolve_modes, stencil_resolve_modes)
}

/// One binding of `pipeline` whose affine declaration reads the draw's index
/// axis, if any declares one.
///
/// The counts a pass states are the pass's own — the index bytes and the two
/// fields are the same whichever binding is asked — so any binding of that
/// shape answers the whole walk. A pipeline whose declarations all state a
/// static footprint has none, and the bound then reads no index at all.
fn affine_index_axis_binding(
    pipeline: &RenderPipelineContract,
) -> Option<(RenderPipelineStage, u32)> {
    pipeline
        .stage_buffers
        .iter()
        .find(|binding| {
            matches!(
                &binding.footprint,
                FootprintProof::Affine { accesses }
                    if accesses
                        .iter()
                        .any(|access| access.terms.iter().any(|term| term.axis == 0))
            )
        })
        .map(|binding| (binding.stage, binding.index))
}

/// The pass's index bytes, resolved for the contract's affine bound when the
/// pipeline needs them (`research/docs/23` §92, R9k).
///
/// An affine stage-buffer footprint is bounded by the draw's own
/// `base_vertex + highest index + 1`, and that arithmetic reads the index
/// buffer's bytes. A view the trace carries states them itself; a lease view
/// does not, and the contract's default arm refuses that declaration by name.
/// This rail owns the same registries its vertex and stage-buffer proofs read
/// through, so it resolves the window here and lets the contract — the one
/// place that states the count rule — evaluate the bound.
///
/// `None` is every other shape: a pipeline whose declarations state a static
/// footprint evaluates no index arithmetic
/// ([`RenderPipelineContract::stage_buffers`]), and a pass with no index buffer
/// names its vertices by `vertices`. Resolving nothing is a decision of the
/// declaration, not a fallback: the contract is never handed a guess in place
/// of bytes.
fn resolve_affine_index_bytes(
    pipeline: &RenderPipelineContract,
    pass: &RenderPassDescriptor,
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<Option<Vec<u8>>, ProviderError> {
    let Some(binding) = pass.indices.as_ref() else {
        return Ok(None);
    };
    // A view the trace carries states its own bytes: the contract reads them
    // itself, and handing the same window over as if it had been resolved would
    // be one measurement stated twice.
    if matches!(binding.view.source, BufferSource::OwnedBytes(_)) {
        return Ok(None);
    }
    if affine_index_axis_binding(pipeline).is_none() {
        return Ok(None);
    }
    let source = resolve_render_input(&binding.view, leases, RenderInputRole::Index)?;
    Ok(Some(source.proof_bytes().to_vec()))
}

/// Validate a render request against the contract and the rail's own allowlist,
/// resolving the pass's vertex and index inputs through `leases`.
///
/// The trace path's entry point: a render input may declare its bytes, name a
/// staged lease or name an owner window to map, and only this call has the
/// registries that can resolve the last two (`research/docs/23` §72, R3d).
/// [`plan`] is the same decision with no lease channel, which is the
/// device-level helper's shape and every test that plans declared bytes.
pub(crate) fn plan_with_leases<'a>(
    request: &OffscreenRenderRequest<'a>,
    leases: Option<&RenderLeaseContext<'_>>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<RenderPlan<'a>, ProviderError> {
    let attachments = &request.pass.color_attachments;
    // The rail's own extent-equality gate comes before the contract's viewport
    // rule: every colour location renders into the pass's one raster, so two
    // attachments of different extents cannot both land. The contract would
    // report the same shape only as a viewport disagreement, which names no
    // attachment; this refusal names the two extents instead.
    if let Some(first) = attachments.first() {
        for (index, other) in attachments.iter().enumerate().skip(1) {
            if other.width != first.width || other.height != first.height {
                return Err(args_refusal("render_attachment_extent_mismatch")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("width", FieldValue::Unsigned(other.width))
                    .with_field("height", FieldValue::Unsigned(other.height))
                    .with_field("first_width", FieldValue::Unsigned(first.width))
                    .with_field("first_height", FieldValue::Unsigned(first.height))
                    .with_detail("every colour attachment of one render pass shares one extent"));
            }
        }
    }
    // The pass's own shape rules and the pipeline/attachment format agreement
    // belong to the contract (`research/docs/23` §3.1, §3.2), not to this
    // rail.
    request.pass.validate().map_err(contract_refusal)?;
    // An affine stage-buffer footprint is bounded by the draw's own index
    // values (`research/docs/23` §92, R9k), so the contract's pairing is asked
    // with the index bytes this rail can read: a lease view is resolved through
    // the same registries the vertex and stage-buffer proofs use, and the
    // contract states the bound instead of refusing the declaration by name.
    // A pass whose declarations evaluate no index arithmetic hands over `None`,
    // which is the strict arm this rail published before the resolved entry
    // existed.
    let resolved_index = resolve_affine_index_bytes(request.pipeline, request.pass, leases)?;
    request
        .pipeline
        .validate_against(request.pass, resolved_index.as_deref())
        .map_err(contract_refusal)?;
    // The same entry states the counts the rail's own plan evaluates the affine
    // reaches over, so the rail's proof and the contract's bound are one
    // arithmetic rather than two that have to agree (`research/docs/23` §92,
    // R9k). A pass whose declarations evaluate no index arithmetic hands over
    // `None` and keeps the cheap span arithmetic
    // [`plan_vertex_input`] already measured.
    let affine_counts = match resolved_index.as_deref() {
        None => None,
        Some(bytes) => {
            let (stage, index) = affine_index_axis_binding(request.pipeline)
                .expect("resolved bytes are only read for an index-axis declaration");
            Some(
                request
                    .pipeline
                    .affine_axis_counts(request.pass, stage, index, Some(bytes))
                    .map_err(contract_refusal)?,
            )
        }
    };
    // The MRT contract admits `MAX_COLOR_ATTACHMENTS` attachments and a
    // matching format list, and this rail's reviewed modules cover one, two and
    // the full ceiling. A wider pass (reachable only through a
    // directly-constructed request, since core admission refuses it) is refused
    // instead of silently rendering only the first few locations (wave3 R1).
    if attachments.len() > usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap_or(usize::MAX) {
        return Err(mrt_attachment_count_refusal(attachments.len()));
    }
    // The provider's answer for the pass's resident declarations
    // (`research/docs/23` §76, R7). The registry itself lives in the provider
    // — it owns the images and decides whether an identity holds bytes a load
    // may read — so this rail is handed one bit per attachment: whether that
    // attachment's identity resolved to a live image. An empty list is the
    // shape every caller with no registry states, and the two ways the list and
    // the trace can disagree are refused below, in both directions.
    let residents = if request.resident.is_empty() {
        None
    } else if request.resident.len() == attachments.len() {
        Some(request.resident.as_slice())
    } else {
        return Err(
            capability_refusal("resident_target_undeclared").with_detail(format!(
                "the resident-target list must carry one entry per colour attachment: {} \
             attachments, {} entries",
                attachments.len(),
                request.resident.len()
            )),
        );
    };
    // A present action hands exactly one attachment on to its target, and the
    // present encoder renders into that one texture: a present pass with a
    // second location would be dropped, so it is refused under the single-
    // attachment slug instead.
    let present = request.pass.present.is_some();
    if present && attachments.len() != 1 {
        return Err(capability_refusal("render_attachment_count_unsupported")
            .with_field(
                "attachments",
                FieldValue::Unsigned(attachments.len() as u64),
            )
            .with_field("maximum", FieldValue::Unsigned(1))
            .with_detail("a present pass hands exactly one colour attachment on to its target"));
    }
    // A present action keeps its own provider-owned target (the R4a registry's
    // image, `research/docs/24` §6 Step 7), while the resident target *is* the
    // provider's image for the attachment's identity (`research/docs/23` §76,
    // R7). Two registries would own one identity, so a pass that declares both
    // is refused by name before either image exists — the same slug, class and
    // detail the Vulkan rail's present entry states.
    if present {
        if let Some(attachment) = attachments
            .iter()
            .find(|attachment| attachment.declares_resident_target())
        {
            return Err(capability_refusal("resident_target_present_unsupported")
                .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                .with_field(
                    "allocation",
                    FieldValue::Unsigned(attachment.allocation_id.get()),
                )
                .with_detail(
                    "the present action keeps its own provider-owned target; a pass that \
                     declares the render rail's resident target beside it is outside this \
                     increment",
                ));
        }
    }
    // A multisampled raster resolves into a single-sample landing the pass
    // owns, while a resident target *is* the landing
    // (`research/docs/23` §76, R7). The two shapes meet only through the
    // present rail's resolve, which this increment does not widen: a pass that
    // declares a resident target beside a multisample raster is refused by name
    // before the first Metal object exists, instead of resolving into a texture
    // the trace believes it named. The same slug, class and detail the Vulkan
    // rail's multisample gate states.
    if let Some(multisample) = request.pass.multisample {
        if attachments
            .iter()
            .any(|attachment| attachment.declares_resident_target())
        {
            return Err(capability_refusal("resident_target_multisample_unsupported")
                .with_field(
                    "samples",
                    FieldValue::Unsigned(u64::from(multisample.sample_count.samples())),
                )
                .with_detail(
                    "a multisampled pass resolves into a single-sample landing of its own; the \
                     resident target is the landing of a single-sample raster in this increment",
                ));
        }
    }
    // A present pass renders into the provider-owned target alone: it opens no
    // depth surface, so a trace that names one — stored or not — asks for a
    // state this shape cannot execute. Refusing here keeps the depth attachment
    // from being silently dropped instead of opened, which is what the
    // offscreen path would do with it (`research/docs/23` §3.3, v43; the same
    // slug, class and detail the Vulkan rail's present entry states).
    if present {
        if let Some(depth) = &request.pass.depth {
            return Err(capability_refusal("render_present_depth_unsupported")
                .with_field(
                    "store",
                    FieldValue::Text(
                        match depth.store {
                            Some(DepthStoreOp::Store) => "store",
                            Some(DepthStoreOp::DontCare) => "dontcare",
                            None => "unstated",
                        }
                        .to_owned(),
                    ),
                )
                .with_detail(
                    "the present rail renders into one provider-owned colour target and opens no \
                     depth surface",
                ));
        }
        // The stencil surface is the depth rule's sibling
        // (`research/docs/23` §3.3, v49): the present shape hands its one
        // colour attachment's texels on and opens no stencil surface, so a
        // trace that names one — stored or not — is refused instead of
        // executed with the attachment dropped, or, once a storing surface
        // names a landing, with those bytes silently left behind. This is the
        // same slug, class and detail the Vulkan rail's present entry states.
        if let Some(stencil) = &request.pass.stencil {
            return Err(capability_refusal("render_present_stencil_unsupported")
                .with_field(
                    "store",
                    FieldValue::Text(
                        match stencil.store {
                            Some(StoreOp::Store) => "store",
                            Some(StoreOp::DontCare) => "dontcare",
                            Some(StoreOp::Resident) => "resident",
                            // Both owner-window stores are the colour
                            // attachment's arms (`research/docs/23` §115 及其后的
                            // 增量，E-TX8/E-TX13), refused by core admission for
                            // the stencil surface.
                            Some(StoreOp::Borrowed) => "borrowed",
                            Some(StoreOp::BorrowedLanding(_)) => "borrowed_landing",
                            None => "unstated",
                        }
                        .to_owned(),
                    ),
                )
                .with_detail(
                    "the present rail renders into one provider-owned colour target and opens no \
                     stencil surface",
                ));
        }
    }
    // The pass's raster is its colour attachment's — or, for the
    // zero-colour-attachment depth pass (`research/docs/23` §3.3, v46), its
    // depth attachment's: the depth surface is the whole raster, and the
    // pipeline's empty colour-format list is what states that no colour
    // target exists beside it. A stencil surface is never the raster on its
    // own: it is a second surface beside the pass's colour or depth one, and
    // the contract refuses a colour-less pass with no depth surface
    // (`EmptyAttachmentList`) before this rail reaches the fallback
    // (`research/docs/23` §3.3, v47).
    let raster = match attachments.first() {
        Some(attachment) => [attachment.width, attachment.height],
        None => {
            let depth = request
                .pass
                .depth
                .as_ref()
                .ok_or_else(|| contract_refusal(ContractError::EmptyAttachmentList))?;
            let identity = depth.identity.ok_or_else(|| {
                args_refusal("render_depth_state_unsupported").with_detail(
                    "a pass with no colour attachment renders into its stored depth surface",
                )
            })?;
            if depth.store != Some(DepthStoreOp::Store) || identity.view_id.is_zero() {
                return Err(args_refusal("render_depth_state_unsupported").with_detail(
                    "a pass with no colour attachment renders into its stored depth surface",
                ));
            }
            [depth.width, depth.height]
        }
    };
    // The pipeline's (vertex-input shape, colour-format list) pair selects the
    // one reviewed module this call may compile; the (module, entry pair) pair
    // is then the whole allowlist, re-checked by `review_contract` below.
    let module = reviewed_module_for(request.pipeline);
    match module {
        Some(module) if request.source == module.source => {}
        Some(module) => {
            return Err(
                allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                    "a {} layout with {:?} compiles the bytes of `{}` and nothing else",
                    layout_name(&request.pipeline.vertex_layout),
                    request.pipeline.color_formats,
                    module.path
                )),
            );
        }
        None => {
            return Err(
                allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                    "no reviewed module carries the {:?} shape with a {} layout",
                    request.pipeline.color_formats,
                    layout_name(&request.pipeline.vertex_layout)
                )),
            );
        }
    }
    review_contract(request.pipeline)?;
    for attachment in attachments {
        if !SUPPORTED_COLOR_FORMATS.contains(&attachment.format) {
            return Err(
                capability_refusal("attachment_format_unsupported").with_field(
                    "format",
                    FieldValue::Unsigned(u64::from(attachment.format.code())),
                ),
            );
        }
    }
    if raster[0] > REVIEWED_ATTACHMENT_CEILING[0] || raster[1] > REVIEWED_ATTACHMENT_CEILING[1] {
        // The slug and fields capability admission uses for this fact
        // (`metal_api_core::provider::ProviderCapabilities::admit_render_passes`).
        // R1b (`research/docs/23` §70): this is the reviewed-ceiling half of the
        // declared window; the device half is answered by the provider before
        // planning (`native.rs`, `refuse_attachment_extent_over_device_limit`),
        // so the two halves cannot disagree about which limit was crossed.
        return Err(capability_refusal("attachment_dimension_limit")
            .with_field("width", FieldValue::Unsigned(raster[0]))
            .with_field("height", FieldValue::Unsigned(raster[1]))
            .with_field(
                "maximum_width",
                FieldValue::Unsigned(REVIEWED_ATTACHMENT_CEILING[0]),
            )
            .with_field(
                "maximum_height",
                FieldValue::Unsigned(REVIEWED_ATTACHMENT_CEILING[1]),
            ));
    }
    // Bounded by the check above, so these conversions cannot lose a bit.
    let extent = [raster[0] as u32, raster[1] as u32];
    // The declaration states the sampler state the pass's samples were lowered
    // against (`research/docs/23` §3.3, v100), and this rail's reviewed MSL
    // module *carries* that state in its own `constexpr sampler`: the one it
    // spells is the one it executes. A declaration naming another filtering or
    // addressing mode describes a sampler no module behind it carries, so it is
    // refused by name with both halves rather than executed as the reviewed
    // state and reported under another. A second reviewed module per state is
    // what would lift this, not a silently substituted sampler.
    //
    // The Vulkan rail's family widened in §109 (the mipmapped filters and the
    // mirror/clamp-to-zero address modes); this table does **not** follow it.
    // The reviewed MSL module is a compile-time artifact whose sampler is
    // spelled in its own source, so every state outside the one it carries —
    // including the newly named ones — keeps its refusal until a module per
    // state is reviewed on the Apple side.
    for declared in &request.pipeline.textures {
        // The sampler-free texel-fetch arm is the render face's own
        // (`research/docs/23` §3.3, v105), and no reviewed MSL module reads a
        // texture without a sampler: the reviewed pair's `constexpr sampler` is
        // what its `sample` calls go through. The declaration is refused by
        // name here instead of being executed as one of the sampled shapes this
        // rail's modules carry — the Vulkan rail's translated arm is where a
        // fetch runs, under the module's own `OpImageFetch`.
        if declared.access == TextureAccess::Fetched {
            return Err(capability_refusal("render_texture_access_unsupported")
                .with_field(
                    "binding",
                    FieldValue::Unsigned(u64::from(declared.metal_binding)),
                )
                .with_field("declared_access", FieldValue::Text("Fetched".to_owned()))
                .with_field("module_access", FieldValue::Text("sampled".to_owned()))
                .with_detail(
                    "the reviewed render modules sample every texture they read through their \
                     own `constexpr sampler`, so a sampler-free texel fetch has no reviewed \
                     module behind it on this rail",
                ));
        }
        if declared.sampler == Some(SamplerPolicy::reviewed_render_sampler()) {
            continue;
        }
        // A runtime `[[sampler(n)]]` argument is the sibling shape
        // (`research/docs/23` §3.3, v102): the reviewed MSL module spells its
        // own `constexpr sampler` and takes no sampler argument, so the state
        // the pass states for that index has no module behind it. The refusal
        // carries both halves of the pairing and both states — the request's,
        // when it bound one, and the reviewed module's — so a caller can tell
        // "a runtime sampler this rail has no module for" apart from "a state
        // no review covered".
        if let Some(sampler_binding) = declared.runtime_sampler {
            let half = |policy: Option<SamplerPolicy>| match policy {
                Some(policy) => (
                    FieldValue::Text(format!("{:?}", policy.filter)),
                    FieldValue::Text(format!("{:?}", policy.address)),
                ),
                None => (
                    FieldValue::Text("None".to_owned()),
                    FieldValue::Text("None".to_owned()),
                ),
            };
            let bound = request
                .pass
                .samplers
                .iter()
                .find(|sampler| sampler.metal_binding == sampler_binding)
                .map(|sampler| sampler.policy);
            let (filter, address) = half(bound);
            let (module_filter, module_address) =
                half(Some(SamplerPolicy::reviewed_render_sampler()));
            return Err(capability_refusal("render_runtime_sampler_unsupported")
                .with_field(
                    "binding",
                    FieldValue::Unsigned(u64::from(declared.metal_binding)),
                )
                .with_field(
                    "sampler_binding",
                    FieldValue::Unsigned(u64::from(sampler_binding)),
                )
                .with_field("filter", filter)
                .with_field("address", address)
                .with_field("module_filter", module_filter)
                .with_field("module_address", module_address)
                .with_detail(
                    "the reviewed MSL module spells its own `constexpr sampler` and takes no \
                     `[[sampler(n)]]` argument, so a runtime sampler has no reviewed module to \
                     execute it; the Vulkan rail's translated arm is where one runs, with the \
                     state the translation's own reflection names",
                ));
        }
        let half = |policy: Option<SamplerPolicy>| match policy {
            Some(policy) => (
                FieldValue::Text(format!("{:?}", policy.filter)),
                FieldValue::Text(format!("{:?}", policy.address)),
            ),
            None => (
                FieldValue::Text("None".to_owned()),
                FieldValue::Text("None".to_owned()),
            ),
        };
        let (filter, address) = half(declared.sampler);
        let (module_filter, module_address) = half(Some(SamplerPolicy::reviewed_render_sampler()));
        return Err(capability_refusal("render_texture_sampler_unsupported")
            .with_field(
                "binding",
                FieldValue::Unsigned(u64::from(declared.metal_binding)),
            )
            .with_field("filter", filter)
            .with_field("address", address)
            .with_field("module_filter", module_filter)
            .with_field("module_address", module_address));
    }
    // The render sampler (`research/docs/23` §3.3, v70/v104) is one decision in
    // two halves, so both are answered together: the reviewed sampling module
    // samples the pass's own texture binding, and a pass that binds a texture
    // runs only through that module. A pass that binds none keeps the empty
    // list every pre-v70 plan carried.
    let textures =
        if request.pass.textures.is_empty() {
            if request.source == REVIEWED_SAMPLED_SOURCE {
                return Err(capability_refusal("render_texture_binding_required").with_detail(
                "the reviewed sampling pair samples the pass's own texture binding; this pass \
                 binds none",
            ));
            }
            Vec::new()
        } else {
            if request.source != REVIEWED_SAMPLED_SOURCE {
                return Err(capability_refusal("render_texture_stage_unsupported").with_detail(
                "the pass binds a render texture but its fragment stage is not the reviewed \
                 sampling module",
            ));
            }
            // The reviewed MSL module samples exactly one
            // `MTLTexture` argument, so a pass that binds more is a shape no
            // reviewed module carries (`research/docs/23` §3.3, v102). The
            // contract admits up to `MAX_RENDER_TEXTURES` — which is what lets
            // a *translated* stage state two or three — and this rail answers
            // the half it cannot execute by name rather than binding surfaces
            // nothing reads.
            if request.pass.textures.len() != REVIEWED_SAMPLED_TEXTURE_COUNT {
                return Err(capability_refusal("render_texture_stage_unsupported")
                    .with_field(
                        "textures",
                        FieldValue::Unsigned(request.pass.textures.len() as u64),
                    )
                    .with_field(
                        "bindings",
                        FieldValue::Unsigned(REVIEWED_SAMPLED_TEXTURE_COUNT as u64),
                    )
                    .with_detail(
                        "the reviewed MSL sampling module reads one sampled texture, so a pass \
                         that binds another number has no reviewed module behind it; the wider \
                         shapes are the translated arm's, where the module's own reflection \
                         names the textures it reads",
                    ));
            }
            // The index is the other half of the same window (`v104`): the
            // reviewed MSL module reads `[[texture(0)]]`, so a binding the
            // contract's indexed list places at another Metal index would be
            // encoded into an argument the module does not read, and the
            // descriptor it does read would stay unbound. Refused by name, with
            // both indexes, rather than encoded as the low one.
            let binding = request.pass.textures[0].metal_binding;
            if binding != REVIEWED_SAMPLED_TEXTURE_BINDING {
                return Err(capability_refusal("render_texture_stage_unsupported")
                    .with_field("binding", FieldValue::Unsigned(u64::from(binding)))
                    .with_field(
                        "module_binding",
                        FieldValue::Unsigned(u64::from(REVIEWED_SAMPLED_TEXTURE_BINDING)),
                    )
                    .with_detail(
                        "the reviewed MSL sampling module reads the texture at its own low \
                         index, so a binding at another Metal index has no reviewed module \
                         behind it",
                    ));
            }
            resolve_render_textures(request.pass, extent, leases)?
        };
    // The pass extent is shared, but the texel width is not: `depth32float`
    // stores four bytes per texel, the colour attachments store their own
    // format's width — `Rgba16Float` eight where the UNORM layouts store four
    // (`research/docs/23` §78) — and the stencil surface is the third width,
    // one byte per texel (`STENCIL_BYTES_PER_TEXEL`), whose readback computes
    // its own extent beside the store it serves ([`read_stencil_texels`],
    // `research/docs/23` §3.3, v49).
    let depth_texel = TexelExtent::flat(extent, metal_api_core::provider::DEPTH_BYTES_PER_TEXEL)
        .ok_or_else(|| capability_refusal("attachment_dimension_limit"))?;
    // The vertex-input half: the streams with their bytes and their footprints.
    // Planned after the attachment because a stream is the draw's own input,
    // exactly as the attachment is its output.
    // A reviewed module that reads its positions out of a stage buffer
    // (`research/docs/23` §92, R9k) is the one shape whose `vertex_id` values a
    // stream never bounds: the flag below is what tells the index proof which
    // reach covers them.
    let vertex_positions_carried_by_stage_buffer = module.is_some_and(|module| {
        module
            .stage_buffers
            .iter()
            .any(|slot| slot.stage == RenderPipelineStage::Vertex && slot.reach.is_affine())
    });
    let (vertex_streams, indices) = plan_vertex_input(
        request.pass,
        request.pipeline,
        leases,
        vertex_positions_carried_by_stage_buffer,
    )?;
    if request.initial.len() != attachments.len() {
        return Err(
            args_refusal("render_attachment_initial_mismatch").with_detail(format!(
                "{} attachments need one previous-bytes entry each, got {}",
                attachments.len(),
                request.initial.len()
            )),
        );
    }
    let mut planned_attachments = Vec::with_capacity(attachments.len());
    // The rasters this rail executes (`research/docs/23` §3.3, v51/v61): only
    // an admitted two-, four- or eight-sample raster has a seed route, and a
    // state of one sample falls through to the refusal below exactly as it did
    // before.
    let multisampled = matches!(
        request.pass.multisample.map(|state| state.sample_count),
        Some(SampleCount::Two | SampleCount::Four | SampleCount::Eight)
    );
    for (index, (attachment, previous)) in attachments
        .iter()
        .zip(request.initial.iter().cloned())
        .enumerate()
    {
        // The trace's own declaration and the provider's answer have to agree
        // in both directions (`research/docs/23` §76, R7). A pass that declares
        // the resident target and was handed no image for it would be executed
        // as a per-pass attachment this rail created — a clear where the trace
        // asked for the provider's own bytes — and an image resolved for an
        // attachment that declares no residency would render the provider's
        // image where the trace declared a per-pass one. Both are refused here,
        // before any Metal object exists, rather than resolved into "whichever
        // image came first".
        let declares_resident = attachment.declares_resident_target();
        let resident = residents.is_some_and(|list| list[index]);
        match (declares_resident, resident) {
            (true, false) => {
                return Err(capability_refusal("resident_target_undeclared")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_detail(
                        "the pass declares the provider-resident target and the provider \
                         resolved no image for it",
                    ));
            }
            (false, true) => {
                return Err(capability_refusal("resident_target_undeclared")
                    .with_field("attachment", FieldValue::Unsigned(index as u64))
                    .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                    .with_field(
                        "allocation",
                        FieldValue::Unsigned(attachment.allocation_id.get()),
                    )
                    .with_detail(
                        "the provider resolved a resident target for this attachment and the \
                         pass declares neither a resident load nor a resident store",
                    ));
            }
            _ => {}
        }
        let format = pixel_format(attachment.format)?;
        let load = load_action(attachment.load, format)?;
        let store = store_action(attachment.store)?;
        // The attachment's own texel width, not the pass's: the `Load` window
        // it uploads and the readback it lands are both measured in this
        // format's texels (`research/docs/23` §78).
        let texel = TexelExtent::flat(extent, attachment.format.bytes_per_texel())
            .ok_or_else(|| capability_refusal("attachment_dimension_limit"))?;
        // A resident *load*'s previous contents arrive in the provider's own
        // image, so the trace declares none for it: a caller that hands bytes
        // over anyway named two sources for one attachment
        // (`research/docs/23` §76, R7), and the rail refuses instead of letting
        // one of them silently win. A `LoadOp::Load` beside a *resident store*
        // is the other way round — its bytes are the trace's own declaration,
        // uploaded into the provider's image before the draw — so it keeps its
        // `initial` source like any trace-declared load.
        if attachment.loads_resident_target() && previous.is_some() {
            return Err(capability_refusal("resident_target_undeclared")
                .with_field("attachment", FieldValue::Unsigned(index as u64))
                .with_field("load_op", FieldValue::Text("resident".to_owned()))
                .with_detail(
                    "a `LoadOp::Resident` attachment keeps the provider image's own contents; \
                     the trace declares no previous bytes for it",
                ));
        }
        let initial = match (load, previous, present, resident) {
            // The resident load's bytes are the provider's image, which is the
            // only arm that reaches the encoder with `.load` and no upload.
            (RenderLoadAction::Load, None, _, true) => None,
            (RenderLoadAction::Clear(_), None, _, _) => None,
            (RenderLoadAction::DontCare, None, _, _) => None,
            (RenderLoadAction::Load, Some(source), _, _) if source.len() == texel.bytes => {
                Some(source)
            }
            (RenderLoadAction::Load, Some(source), _, _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(format!(
                        "LoadOp::Load needs {} tightly packed bytes of its own format, got {}",
                        texel.bytes,
                        source.len(),
                    )),
                );
            }
            (RenderLoadAction::Load, None, true, _) => None,
            (RenderLoadAction::Load, None, false, false) => {
                return Err(args_refusal("render_attachment_initial_mismatch")
                    .with_detail("LoadOp::Load needs the attachment's previous texels"));
            }
            (RenderLoadAction::Clear(_), Some(_), _, _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(
                        "LoadOp::Clear writes every texel, so initial bytes are refused",
                    ),
                );
            }
            (RenderLoadAction::DontCare, Some(_), _, _) => {
                return Err(
                    args_refusal("render_attachment_initial_mismatch").with_detail(
                        "LoadOp::DontCare reads and presets no pre-pass bytes, so initial bytes \
                         are refused",
                    ),
                );
            }
        };
        planned_attachments.push(PlannedAttachment {
            format,
            texel,
            load,
            store,
            // The landing the pass publishes is the trace-declared
            // `StoreOp::Store` alone (`research/docs/23` §3.6/§76): a resident
            // store keeps its bytes in the provider's image and a discarded
            // attachment drops them, so neither one produces a readback.
            publishes: attachment.store == StoreOp::Store,
            resident: declares_resident,
            // The multisampled load's seed (`research/docs/23` §82, v82): a
            // four-sample surface cannot be uploaded into, so the plan states
            // the one colour the encoder's seed pass clears every sample with.
            // The same rule the Vulkan rail states, with the same names.
            seed: match (load, &initial) {
                (RenderLoadAction::Load, Some(source)) if multisampled => Some(multisample_seed(
                    source,
                    index,
                    attachment.format.bytes_per_texel(),
                    format,
                )?),
                (RenderLoadAction::Load, None) if multisampled => {
                    return Err(capability_refusal("render_multisample_load_unsupported")
                        .with_detail(
                            "a multisampled `Load` is seeded by a clear the plan has to state; \
                             this attachment declares no previous bytes",
                        ));
                }
                _ => None,
            },
            initial,
        });
    }
    // The depth resolve (`research/docs/23` §3.3, v57c) only means something
    // beside a multisample raster that keeps its depth surface: the resolve is
    // the reduction of the stored four-sample texels, so a resolve without
    // both is refused instead of silently ignored. The core contract refuses
    // the same shape with `DepthResolveWithoutStoredDepth`; this is the
    // value-level second line of defence for a directly-constructed request,
    // in the same place the Vulkan rail re-asserts it.
    if request.pass.depth_resolve.is_some() {
        let stored = request.pass.multisample.is_some()
            && request
                .pass
                .depth
                .as_ref()
                .is_some_and(|depth| depth.store == Some(DepthStoreOp::Store));
        if !stored {
            let store = request
                .pass
                .depth
                .as_ref()
                .and_then(|depth| depth.store)
                .map(DepthStoreOp::code);
            return Err(contract_refusal(
                ContractError::DepthResolveWithoutStoredDepth { store },
            ));
        }
    }
    // The stencil resolve (`research/docs/23` §3.3, v60) is the depth
    // resolve's sibling one byte wide: it only means something beside a
    // multisample raster that keeps its stencil surface, and the
    // `DepthResolvedSample` filter names the sample the depth resolve
    // selected. The core contract refuses the same shapes with
    // `StencilResolveWithoutStoredStencil` and
    // `StencilResolveWithoutDepthResolve`; this is the value-level second
    // line of defence for a directly-constructed request.
    if let Some(resolve) = request.pass.stencil_resolve {
        let stored = request.pass.multisample.is_some()
            && request
                .pass
                .stencil
                .as_ref()
                .is_some_and(|stencil| stencil.store == Some(StoreOp::Store));
        if !stored {
            let store = request
                .pass
                .stencil
                .as_ref()
                .and_then(|stencil| stencil.store);
            return Err(contract_refusal(
                ContractError::StencilResolveWithoutStoredStencil { store },
            ));
        }
        if resolve.filter == StencilResolveFilter::DepthResolvedSample
            && request.pass.depth_resolve.is_none()
        {
            return Err(contract_refusal(
                ContractError::StencilResolveWithoutDepthResolve,
            ));
        }
        let mask = 1u32 << u32::from(resolve.filter.code());
        if stencil_resolve_modes & mask == 0 {
            return Err(
                capability_refusal("render_stencil_resolve_filter_unsupported")
                    .with_field(
                        "filter",
                        FieldValue::Unsigned(u64::from(resolve.filter.code())),
                    )
                    .with_field(
                        "modes",
                        FieldValue::Unsigned(u64::from(stencil_resolve_modes)),
                    ),
            );
        }
    }
    let module = module.expect("the source check above refused a shape without a module");
    // The pass's stage buffers (`research/docs/23` §83/R9g, §92/R9k): the
    // registration gate paired every declaration with one of the module's own
    // `[[buffer(N)]]` arguments, so this walk resolves the pass's views through
    // the same three-armed source channel the vertex and index streams use and
    // proves the module's reach against the bytes it resolved — over the draw's
    // own counts for an affine reach, read from the same resolved index stream
    // the vertex proof just took. Empty for every pass that declares none,
    // which is every pre-R9g plan.
    let stage_buffers = plan_stage_buffers(
        request.pass,
        module,
        leases,
        indices.as_ref(),
        affine_counts,
    )?;
    Ok(RenderPlan {
        source: module.source,
        module_path: module.path,
        vertex_entry: request.pipeline.vertex_entry.as_str(),
        fragment_entry: request.pipeline.fragment_entry.as_str(),
        attachments: planned_attachments,
        // The pass's sampled textures, resolved above
        // (`research/docs/23` §3.3, v70). Empty for every pre-v70 plan, which
        // is the shape the encoder's texture bind branches on.
        textures,
        extent,
        viewport: request.pass.viewport,
        scissor: request.pass.scissor,
        cull: request.pass.cull,
        blend: request.pass.blend.clone(),
        // The multisample raster (`research/docs/23` §3.3, v51/v61). The
        // contract already refused a single-sample state; the rail re-asserts
        // the counts its encoder knows how to build, so a directly-constructed
        // request cannot reach `newTextureWithDescriptor` with a raster this
        // increment does not execute. The attachment's load was decided above:
        // a `clear`/`dontcare` opens the image from its own action, and a
        // `Load` carries the seed the encoder's seed pass clears every sample
        // with (`research/docs/23` §82, v82).
        multisample: match request.pass.multisample {
            Some(multisample)
                if matches!(
                    multisample.sample_count,
                    SampleCount::Two | SampleCount::Four | SampleCount::Eight
                ) =>
            {
                // A stored multisampled depth surface is admitted from v57c on,
                // through the resolve the pass then has to state: its texels
                // are only observable as the resolve's reduction, so a stored
                // surface without one stays refused, and a filter the device
                // does not report is refused by the same per-filter question
                // the capability snapshot answered (`research/docs/23` §3.3,
                // v57c).
                if request
                    .pass
                    .depth
                    .as_ref()
                    .is_some_and(|depth| depth.store == Some(DepthStoreOp::Store))
                {
                    match request.pass.depth_resolve {
                        Some(resolve) => {
                            let mask = 1u32 << u32::from(resolve.filter.code());
                            if depth_resolve_modes & mask == 0 {
                                return Err(capability_refusal(
                                    "render_depth_resolve_filter_unsupported",
                                )
                                .with_field(
                                    "filter",
                                    FieldValue::Unsigned(u64::from(resolve.filter.code())),
                                )
                                .with_field(
                                    "modes",
                                    FieldValue::Unsigned(u64::from(depth_resolve_modes)),
                                ));
                            }
                        }
                        None => {
                            return Err(capability_refusal(
                                "render_multisample_depth_store_unsupported",
                            )
                            .with_detail(
                                "a multisampled depth surface cannot be kept without a depth \
                                 resolve",
                            ));
                        }
                    }
                }
                // The stencil sibling (`research/docs/23` §3.3, v55/v60): a
                // kept stencil surface needs the stencil resolve, and a
                // combined depth-stencil surface is one texture both faces
                // share, so its two store decisions have to agree: both
                // rail-owned with no resolve — the v66 write-then-test pair —
                // or both kept through their resolves, the v60 shape
                // (`research/docs/23` §3.3, v60/v66).
                if let (Some(depth), Some(stencil)) = (&request.pass.depth, &request.pass.stencil) {
                    let rail_owned = !depth.is_stored() && !stencil.is_stored();
                    let resolved = depth.is_stored()
                        && stencil.is_stored()
                        && request.pass.depth_resolve.is_some()
                        && request.pass.stencil_resolve.is_some();
                    if !rail_owned && !resolved {
                        return Err(capability_refusal(
                            "render_stencil_combined_surface_unsupported",
                        )
                        .with_detail(
                            "the multisample raster opens one depth-stencil surface: \
                                     the two faces keep both or neither — both rail-owned with \
                                     no resolve, or both stored through their resolves",
                        ));
                    }
                }
                if request
                    .pass
                    .stencil
                    .as_ref()
                    .is_some_and(|stencil| stencil.store == Some(StoreOp::Store))
                {
                    // The stored surface is admitted from v60 on, through the
                    // resolve the pass then has to state: its texels are only
                    // observable as the resolve's reduction, so a stored
                    // surface without one stays refused, and a filter the
                    // device does not report is refused by the same per-filter
                    // question the capability snapshot answered
                    // (`research/docs/23` §3.3, v60).
                    match request.pass.stencil_resolve {
                        Some(resolve) => {
                            let mask = 1u32 << u32::from(resolve.filter.code());
                            if stencil_resolve_modes & mask == 0 {
                                return Err(capability_refusal(
                                    "render_stencil_resolve_filter_unsupported",
                                )
                                .with_field(
                                    "filter",
                                    FieldValue::Unsigned(u64::from(resolve.filter.code())),
                                )
                                .with_field(
                                    "modes",
                                    FieldValue::Unsigned(u64::from(stencil_resolve_modes)),
                                ));
                            }
                        }
                        None => {
                            return Err(capability_refusal(
                                "render_multisample_stencil_store_unsupported",
                            )
                            .with_detail(
                                "a multisampled stencil surface cannot be kept without a \
                                 stencil resolve",
                            ))
                        }
                    }
                }
                Some(multisample.sample_count)
            }
            Some(_) => {
                return Err(capability_refusal("render_multisample_state_unsupported")
                    .with_detail(
                        "the multisample increment executes the two-, four- and eight-sample \
                         rasters only",
                    ));
            }
            None => None,
        },
        depth: request.pass.depth.as_ref().map(|depth| PlannedDepth {
            width: u32::try_from(depth.width).unwrap_or(u32::MAX),
            height: u32::try_from(depth.height).unwrap_or(u32::MAX),
            clear_bits: depth.load.clear_depth().map(f32::to_bits),
            test: request.pass.depth_test,
            // The store action is the pass's own statement
            // (`research/docs/23` §3.3, v43): the pre-v43 shapes leave it
            // absent, and core admission has already held the storing shape to
            // naming a landing identity.
            store: depth.store,
        }),
        depth_resolve: request.pass.depth_resolve.map(|resolve| resolve.filter),
        stencil_resolve: request.pass.stencil_resolve.map(|resolve| resolve.filter),
        stencil: request.pass.stencil.as_ref().map(|stencil| PlannedStencil {
            width: u32::try_from(stencil.width).unwrap_or(u32::MAX),
            height: u32::try_from(stencil.height).unwrap_or(u32::MAX),
            // The same decoding the depth load op uses: the clear arm
            // carries the value every stencil texel starts from, and `Load`
            // keeps the attachment's previous contents
            // (`research/docs/23` §3.3, v47).
            clear_value: stencil.load.clear_value(),
            // The state is the pass's own, and core admission has already
            // refused a test with no attachment to test
            // (`StencilTestWithoutAttachment`).
            test: request.pass.stencil_test,
            // The store action is the pass's own statement
            // (`research/docs/23` §3.3, v49): the pre-v49 shapes leave it
            // absent, and core admission has already held the storing shape to
            // naming a landing identity.
            store: stencil.store,
        }),
        vertices: request.pass.vertices,
        instance_count: request.pass.instance_count,
        vertex_streams,
        indices,
        stage_buffers,
        depth_texel,
    })
}

/// The vertex-input shape of a pipeline, as the refusals spell it.
pub(crate) fn layout_name(layout: &VertexLayout) -> &'static str {
    match layout {
        VertexLayout::None => "vertex_id",
        VertexLayout::Buffers(_) => "vertex-buffer",
    }
}

/// The rail's review gate for a render pipeline contract.
///
/// Each reviewed module carries exactly one vertex entry and one fragment entry,
/// and the contract's [`VertexLayout`] and `color_formats` say which module it
/// may be: a single-output contract names the module its layout selects, and a
/// dual-format contract with a stream layout the dual module. A shape no
/// module was reviewed for — a dual-format `vertex_id` contract, a wider
/// format list, or a dual format pair that is not two `Rgba8Unorm` locations —
/// is refused with the same slug, class and phase the compute allowlist gives
/// an unreviewed kernel (`lib.rs::bounded_contract`,
/// `native_shader_not_allowlisted`): a matching file name, an edited module or
/// a recompiled one must not be enough to run different source
/// (`research/docs/23` §6 Step 7).
///
/// A registration that declares stage buffer bindings is answered by the
/// module's own `[[buffer(N)]]` table instead (`research/docs/23` §83, R9g):
/// the declaration half is paired field by field with the arguments the
/// reviewed module actually reads ([`validate_reviewed_stage_buffers`]), so a
/// contract can no longer be refused merely for carrying the face — but a
/// declaration no reviewed module accounts for still is, by name. Registration
/// (`NativeMetalProvider::register_render_pipeline`) and [`plan`] both run it,
/// so the refusal is reachable before a submission as well as inside one.
pub(crate) fn review_contract(contract: &RenderPipelineContract) -> Result<(), ProviderError> {
    if let Some(module) = reviewed_module_for(contract) {
        return validate_reviewed_stage_buffers(contract, module);
    }
    // The refusal names the entry pair the shape carries, so a caller can fix
    // the registration without reading the rail: the shape's own module is the
    // one a matching pair compiles, and the render-sampler shape shares the
    // milestone's — which is exactly why the entry pair, not the shape, is
    // what a caller has to correct here (`research/docs/23` §3.3, v70).
    let Some(module) = reviewed_module(&contract.vertex_layout, &contract.color_formats) else {
        return Err(
            allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
                "no reviewed module carries the {:?} shape with a {} layout",
                contract.color_formats,
                layout_name(&contract.vertex_layout),
            )),
        );
    };
    Err(
        allowlist_refusal("native_render_source_not_reviewed").with_detail(format!(
            "a {} layout with {:?} compiles `{}`, which carries {:?} and {:?}",
            layout_name(&contract.vertex_layout),
            contract.color_formats,
            module.path,
            module.vertex_entry,
            module.fragment_entry,
        )),
    )
}

/// Pair a registration's stage-buffer declarations with the reviewed module's
/// own `[[buffer(N)]]` arguments (`research/docs/23` §83, R9g).
///
/// The reviewed module is this rail's half of the pairing a translated stage
/// states through its reflection: its argument list is fixed by the pinned
/// source bytes ([`ReviewedStageBufferSlot`]), and the contract declares what
/// each stage reaches there. The two halves have to agree field by field, and
/// each disagreement keeps the vocabulary the Vulkan rails established:
///
/// * a declaration at a slot the reviewed module never reads would be a
///   binding silently dropped — `render_stage_buffer_stage_unsupported`, the
///   same slug and arm the Vulkan rail's reviewed-versus-translated gate uses;
/// * a declaration whose access or byte extent disagrees with the module's own
///   argument is a reflection mismatch — `render_stage_reflection_mismatch`,
///   with the module's read extent in the `reflected_bytes` field the R9c arm
///   fills, because the pass's view is proven against the declaration and the
///   module reads the pinned bytes;
/// * a module argument no declaration covers would leave a descriptor the
///   stage reads undefined — `render_stage_buffer_binding_required`, the
///   reviewed pair's own refusal on the Vulkan rail.
///
/// A registration that declares no stage buffers and names a module that reads
/// none keeps the exact pre-R9g answer: there is nothing to pair, and the walk
/// has no state to report.
fn validate_reviewed_stage_buffers(
    contract: &RenderPipelineContract,
    module: &ReviewedModule,
) -> Result<(), ProviderError> {
    let entry = |stage: RenderPipelineStage| match stage {
        RenderPipelineStage::Vertex => contract.vertex_entry.as_str(),
        RenderPipelineStage::Fragment => contract.fragment_entry.as_str(),
    };
    let access_name = |access: BufferAccess| match access {
        BufferAccess::Read => "read",
        BufferAccess::Write => "write",
        BufferAccess::ReadWrite => "read_write",
        BufferAccess::Unused => "unused",
    };
    let mismatch = |declared: &metal_api_core::provider::StageBufferBinding| {
        capability_refusal("render_stage_reflection_mismatch")
            .with_field("stage", FieldValue::Text(declared.stage.name().to_owned()))
            .with_field("entry", FieldValue::Text(entry(declared.stage).to_owned()))
            .with_field("field", FieldValue::Text("bindings".to_owned()))
            .with_field("index", FieldValue::Unsigned(u64::from(declared.index)))
    };
    for declared in &contract.stage_buffers {
        let Some(slot) = module
            .stage_buffers
            .iter()
            .find(|slot| slot.stage == declared.stage && slot.index == declared.index)
        else {
            return Err(capability_refusal("render_stage_buffer_stage_unsupported")
                .with_field("stage", FieldValue::Text(declared.stage.name().to_owned()))
                .with_field("binding", FieldValue::Unsigned(u64::from(declared.index)))
                .with_detail(format!(
                    "the reviewed module `{}` reads no `[[buffer({})]]` argument in its {} \
                         stage, so this declaration would bind a slot the shader never reads",
                    module.path,
                    declared.index,
                    declared.stage.name(),
                )));
        };
        if declared.access != slot.access {
            return Err(mismatch(declared)
                .with_field(
                    "declared_access",
                    FieldValue::Text(access_name(declared.access).to_owned()),
                )
                .with_field(
                    "reflected_access",
                    FieldValue::Text(access_name(slot.access).to_owned()),
                )
                .with_detail(
                    "the reviewed module's argument and the contract's declaration classify this \
                     binding's access differently, so the bytes the pass binds are not the \
                     interface the module states",
                ));
        }
        // The two footprint arms state the same measurement at different
        // levels, so they are paired differently (`research/docs/23` §3.3,
        // v86/v92) — the rule the Vulkan rail's translated pairing states,
        // restated for a reviewed slot table. A *static* declaration is a
        // ceiling the module's fixed reach has to fit under, while an *affine*
        // declaration is the measurement itself: the two ends have to be the
        // same access set, because a ceiling a draw can outgrow is not what an
        // affine reach needs.
        match (&declared.footprint, slot.reach) {
            (
                FootprintProof::Static { max_bytes },
                ReviewedStageBufferReach::Static {
                    max_bytes: reflected,
                },
            ) => {
                if *max_bytes < reflected {
                    return Err(mismatch(declared)
                        .with_field("declared_bytes", FieldValue::Unsigned(*max_bytes))
                        .with_field("reflected_bytes", FieldValue::Unsigned(reflected))
                        .with_detail(
                            "the reviewed module reads past the declared extent, and the pass's \
                             view is proven against the declaration rather than the module",
                        ));
                }
            }
            (
                FootprintProof::Static { max_bytes },
                ReviewedStageBufferReach::Affine { accesses },
            ) => {
                return Err(mismatch(declared)
                    .with_field("declared_bytes", FieldValue::Unsigned(*max_bytes))
                    .with_field(
                        "reflected_accesses",
                        FieldValue::Unsigned(accesses.len() as u64),
                    )
                    .with_detail(
                        "the reviewed module's reach grows with the draw's own vertex index, and \
                         the declaration states one static byte extent: a ceiling the draw can \
                         outgrow is not the measurement this module states",
                    ));
            }
            (FootprintProof::Affine { accesses }, reach) => {
                let reflected = reach.accesses();
                if affine_access_set(accesses) != affine_access_set(&reflected) {
                    return Err(mismatch(declared)
                        .with_field(
                            "declared_accesses",
                            FieldValue::Unsigned(accesses.len() as u64),
                        )
                        .with_field(
                            "reflected_accesses",
                            FieldValue::Unsigned(reflected.len() as u64),
                        )
                        .with_detail(
                            "the declaration and the module state different affine access sets, \
                             and this arm pairs two measurements of one module rather than a \
                             ceiling with a reach",
                        ));
                }
            }
            (FootprintProof::Unbounded, _) => {
                return Err(mismatch(declared).with_detail(
                    "the declared footprint is unbounded, which the render contract does not \
                     admit for a stage buffer at all",
                ));
            }
            // The whole-binding arm is the Vulkan rail's (`research/docs/23`
            // §3.3, E-SB3), and this rail keeps its own refusal by name rather
            // than a shared one: the reviewed pair reads each `[[buffer(N)]]`
            // argument at the extent its own pinned source states
            // ([`ReviewedStageBufferReach`]), so a declaration that states no
            // reach at all would have to be executed against a window this rail
            // cannot account for — the opposite of what its reviewed modules
            // carry. The capability frame publishes the same answer
            // ([`stage_buffer_capability_bits`] keeps the bit closed), so a
            // consumer never hands this rail the shape; a registration that
            // states it anyway is refused here, by name, before any Metal
            // object exists.
            (FootprintProof::BindingRange, _) => {
                return Err(capability_refusal(
                    "render_stage_buffer_binding_range_unsupported",
                )
                .with_field("stage", FieldValue::Text(declared.stage.name().to_owned()))
                .with_field("entry", FieldValue::Text(entry(declared.stage).to_owned()))
                .with_field("field", FieldValue::Text("bindings".to_owned()))
                .with_field("index", FieldValue::Unsigned(u64::from(declared.index)))
                .with_field(
                    "module",
                    FieldValue::Text(module.path.to_owned()),
                )
                .with_detail(
                    "the declared footprint states the whole-binding arm, where the translation \
                     did not state how far into the binding the module reaches: this rail's \
                     reviewed stages read each `[[buffer(N)]]` argument at the extent their own \
                     source pins, so there is no route here that executes a declaration nothing \
                     measured. The Vulkan rail executes the arm on a device whose \
                     `robustBufferAccess` was enabled, binding the pass's own view whole",
                ));
            }
        }
    }
    for slot in module.stage_buffers {
        let declared = contract
            .stage_buffers
            .iter()
            .any(|declared| declared.stage == slot.stage && declared.index == slot.index);
        if !declared {
            return Err(capability_refusal("render_stage_buffer_binding_required")
                .with_field("stage", FieldValue::Text(slot.stage.name().to_owned()))
                .with_field("binding", FieldValue::Unsigned(u64::from(slot.index)))
                .with_detail(format!(
                    "the reviewed module `{}` reads its {} stage's `[[buffer({})]]` argument, \
                         so a registration that names this module has to declare the slot",
                    module.path,
                    slot.stage.name(),
                    slot.index,
                )));
        }
    }
    Ok(())
}

/// Resolve one pass's sampled textures into the rail's own plan shape
/// (`research/docs/23` §3.3, v70/v104).
///
/// The Vulkan rail's window, restated for a directly-constructed pass: exactly
/// one `rgba8_unorm` 2D single-sample surface whose extent equals the render
/// area. The extent rule is what makes the fixture's expectation
/// driver-independent — the interpolated varying stands on a texel centre only
/// when the texture and the render area share their extent — so a texture of
/// another size is refused by name instead of sampled as a filtered read the
/// review never covered.
///
/// The texture's bytes are resolved through the same three-arm channel the
/// streams and the loading attachments use (`research/docs/23` §75, R5c): the
/// trace's own bytes, the provider's staged copy of an owner lease, or the
/// owner's own mapping, whose pages the encoder uploads as they stand. The
/// resolution runs before the first Metal object exists.
fn resolve_render_textures<'a>(
    pass: &'a RenderPassDescriptor,
    extent: [u32; 2],
    leases: Option<&RenderLeaseContext<'_>>,
) -> Result<Vec<PlannedTexture<'a>>, ProviderError> {
    if pass.textures.len() > MAX_RENDER_TEXTURES as usize {
        return Err(capability_refusal("render_texture_limit")
            .with_field(
                "requested",
                FieldValue::Unsigned(pass.textures.len() as u64),
            )
            .with_field(
                "maximum",
                FieldValue::Unsigned(u64::from(MAX_RENDER_TEXTURES)),
            ));
    }
    let mut textures = Vec::with_capacity(pass.textures.len());
    for view in &pass.textures {
        // Every refusal names the binding the view itself states (`v104`): the
        // pass's list may skip an index, so the entry's position is the list's
        // own order rather than the argument the module reads.
        let binding = view.metal_binding;
        if view.format != TextureFormat::Rgba8Unorm {
            return Err(capability_refusal("render_texture_format_unsupported")
                .with_field("binding", FieldValue::Unsigned(u64::from(binding)))
                .with_field("format", FieldValue::Text(format!("{:?}", view.format)))
                .with_detail(
                    "the reviewed sampling module reads one rgba8_unorm surface; this rail's \
                     table names that one format, and the other 8-bit byte order is the Vulkan \
                     rail's widened arm (`research/docs/23` §107), exactly as the narrow lanes, \
                     the eight-byte half-float lane and the two single-component float lanes \
                     beside it are (§113, §119), pending the Apple-side reading each flip would \
                     owe",
                ));
        }
        if view.texture_type != TextureType::D2
            || view.sample_count != 1
            || view.depth != 1
            || view.array_length != 1
        {
            return Err(capability_refusal("render_texture_shape_unsupported")
                .with_field("binding", FieldValue::Unsigned(u64::from(binding)))
                .with_field(
                    "texture_type",
                    FieldValue::Text(format!("{:?}", view.texture_type)),
                )
                .with_field("sample_count", FieldValue::Unsigned(view.sample_count))
                .with_detail(
                    "the reviewed sampling module reads a single-sample 2D surface: the Metal 1D \
                     and 3D equivalences the widened arms beside it would need are shapes this \
                     rail has no Apple-side reading for, so a `D1`/`D1Array` LUT and a `D3` \
                     volume keep this refusal by name while the snapshot declares no window for \
                     either of them",
                ));
        }
        let source = resolve_render_texture_source(
            view,
            leases,
            usize::try_from(binding).unwrap_or(usize::MAX),
        )?;
        let width = u32::try_from(view.width).unwrap_or(u32::MAX);
        let height = u32::try_from(view.height).unwrap_or(u32::MAX);
        if width == 0 || height == 0 {
            return Err(contract_refusal(ContractError::ZeroDimension {
                field: "render texture",
                axis: 0,
            }));
        }
        if [width, height] != extent {
            return Err(capability_refusal("render_texture_extent_unsupported")
                .with_field("binding", FieldValue::Unsigned(u64::from(binding)))
                .with_field("width", FieldValue::Unsigned(u64::from(width)))
                .with_field("height", FieldValue::Unsigned(u64::from(height)))
                .with_field("render_width", FieldValue::Unsigned(u64::from(extent[0])))
                .with_field("render_height", FieldValue::Unsigned(u64::from(extent[1])))
                .with_detail(
                    "the reviewed sampling shape samples a texture of the render area's own \
                     extent, so every fragment stands on a texel centre",
                ));
        }
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|texels| texels.checked_mul(view.format.bytes_per_texel()))
            .ok_or_else(|| contract_refusal(ContractError::ArithmeticOverflow("render texture")))?;
        let actual = u64::try_from(source.len()).unwrap_or(u64::MAX);
        if actual != expected {
            return Err(contract_refusal(ContractError::SourceLengthMismatch {
                view: view.view_id,
                expected,
                actual,
            }));
        }
        textures.push(PlannedTexture {
            source,
            extent: [width, height],
            binding,
        });
    }
    Ok(textures)
}

/// Where the previous contents an offscreen `LoadOp::Load` pass uploads before
/// it opens come from (`research/docs/23` §74, R5b).
///
/// The declaring case's view is the only channel that carries an attachment's
/// previous contents: [`metal_api_core::provider::RenderAttachment`] restates
/// the view's identity and shape and names no bytes, so a `Load` resolves them
/// from the declaration itself — the bytes it owns
/// ([`BufferSource::OwnedBytes`]), the provider's staged copy of an owner lease,
/// or the owner's own mapping, which is the same three-arm resolution a render
/// stream goes through ([`resolve_render_input`]). The no-copy arm keeps the
/// owner's mapping, so the bytes the encoder uploads are the pages the owner
/// wrote last — a snapshot-style import cannot pass that, and the plan names the
/// lease its submission has to retain.
///
/// `Clear` and `DontCare` resolve no bytes: a clear writes every texel and a
/// `DontCare` attachment declares its pre-pass contents undefined, so neither
/// shape presets the attachment (`research/docs/23` §3.1, v20).
pub(crate) fn previous_source<'a>(
    load: LoadOp,
    view: &'a BufferView,
    leases: Option<&RenderLeaseContext<'_>>,
    attachment: usize,
) -> Result<Option<PlannedInputSource<'a>>, ProviderError> {
    match load {
        // The refusal carries the location it could not read, so a capture can
        // tell which attachment of a multi-attachment pass the source arm
        // belongs to.
        LoadOp::Load => resolve_render_input(view, leases, RenderInputRole::Attachment)
            .map(Some)
            .map_err(|error| {
                error.with_field("attachment", FieldValue::Unsigned(attachment as u64))
            }),
        LoadOp::Clear(_) | LoadOp::DontCare => Ok(None),
        // A resident load declares no view at all (`research/docs/23` §76, R7):
        // its bytes are the provider's own image for the attachment's identity,
        // which the registry resolved before this plan existed, so there is
        // nothing to resolve out of the trace's declaration. A caller that hands
        // bytes over anyway is refused by name in
        // [`plan_with_leases`], not silently resolved here. The `attachment`
        // index stays part of the signature because the trace path's own
        // refusals carry it.
        LoadOp::Resident => Ok(None),
    }
}

/// Refuse a trace whose compute passes would be reordered against a render
/// pass's stores.
///
/// The trace path executes every compute pass before every render pass, in trace
/// order within each group (see [`plan_trace`] and
/// `NativeMetalProvider::execute_render_passes`). Compute passes that come
/// before a render pass therefore run in the order the trace asked for, and so
/// do render passes among themselves. The one shape that would silently change
/// meaning is a compute pass that *follows* a render pass and binds a view that
/// render pass stores: it would observe pre-render bytes where the trace's
/// serial order defines post-render ones. Core admission already refuses the
/// write/write half of the pair (`AttachmentComputeConflict`,
/// `attachment_resource_conflict`); this is the read half, refused rather than
/// executed in an order the bytes would not reflect.
///
/// Core admission owns both halves now (`ContractError::
/// RenderPassOrderUnsupported`, slug `render_pass_order_unsupported`, review
/// item I4, 2026-09-14), and every submitted trace reached it through
/// `ProviderCapabilities::admit`. This walk stays as the rail's own defense for
/// a value-level plan: it compares view identities, so it is at least as strict
/// as the contract's byte ranges and never admits a trace the contract refused.
pub(crate) fn refuse_reordered_render_reads(trace: &ComputeTrace) -> Result<(), ProviderError> {
    let mut render_written = BTreeMap::<ViewId, usize>::new();
    for (index, entry) in trace.passes.iter().enumerate() {
        match entry {
            TracePass::Render(pass) => {
                for attachment in &pass.color_attachments {
                    render_written.entry(attachment.view_id).or_insert(index);
                }
            }
            // A landing-only entry writes the owner's window, not a view of
            // this trace's pool, so it registers nothing for the ordering walk
            // below (`research/docs/23` §115 之后的增量，E-TX14/R4b). It is the
            // rail that refuses the entry by name, in its own plan gate.
            TracePass::Landing(_) => {}
            TracePass::Compute(pass) => {
                let bound = pass
                    .buffers
                    .iter()
                    .map(|view| view.view_id)
                    .chain(pass.textures.iter().map(|texture| texture.view_id));
                for view in bound {
                    if let Some(render_pass) = render_written.get(&view) {
                        return Err(refusal(
                            ProviderPhase::Resolve,
                            ProviderErrorClass::Capability,
                            "render_pass_order_unsupported",
                        )
                        .with_field("pass", FieldValue::Unsigned(index as u64))
                        .with_field("render_pass", FieldValue::Unsigned(*render_pass as u64))
                        .with_field("view", FieldValue::Unsigned(view.get()))
                        .with_detail(
                            "a compute pass that follows a render pass storing this view \
                             would observe pre-render bytes, because this increment runs \
                             every compute pass before every render pass",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn capability_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Capability, slug)
}

fn compile_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Compile, ProviderErrorClass::Compile, slug)
}

/// The review gate: an unreviewed source or entry is a capability refusal, the
/// same class the compute allowlist gives an unreviewed kernel
/// (`lib.rs::bounded_contract`, `native_shader_not_allowlisted`).
fn allowlist_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Compile, ProviderErrorClass::Capability, slug)
}

/// A structural refusal about a fact only this rail models, hence a slug of its
/// own: the contract's pass carries no "previous contents" field, so the
/// `LoadOp::Load` agreement is the rail's to state.
fn args_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Args, slug)
}

/// The refusal for a pass whose attachment count this rail cannot execute.
///
/// The capability bit reports `MAX_COLOR_ATTACHMENTS`, so a wider pass is
/// refused here with the rail's own limit instead of silently rendering only
/// the first few locations. With the reviewed modules covering one, two and the
/// ceiling, this gate is the fail-closed half for a directly-constructed
/// request that skipped core admission (wave3 R1, v24).
fn mrt_attachment_count_refusal(attachments: usize) -> ProviderError {
    capability_refusal("render_mrt_attachment_count_unsupported")
        .with_field("attachments", FieldValue::Unsigned(attachments as u64))
        .with_field(
            "maximum",
            FieldValue::Unsigned(MAX_COLOR_ATTACHMENTS as u64),
        )
        .with_detail(
            "the reviewed modules write one, two or MAX_COLOR_ATTACHMENTS colour locations; \
             a wider pass would silently drop its later locations",
        )
}

/// Map a core contract error onto the refusal admission already uses.
///
/// `metal-api-core`'s exhaustive mapping (`contract_error_refusal`) is private to
/// that crate, so this table repeats only the variants a render pass and its
/// pipeline can raise here, with the same class and slug. It is a mirror, not a
/// second policy — and in the real flow it is defence in depth, because a trace
/// reaches a provider through `ProviderCapabilities::admit` first.
fn contract_refusal(error: ContractError) -> ProviderError {
    use ContractError as E;
    let (class, slug) = match &error {
        E::UnsupportedAttachmentFormat(_) => (
            ProviderErrorClass::Capability,
            "attachment_format_unsupported",
        ),
        E::UnsupportedAttachmentLoadOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_load_op_unsupported",
        ),
        E::UnsupportedAttachmentStoreOp(_) => (
            ProviderErrorClass::Capability,
            "attachment_store_op_unsupported",
        ),
        E::AttachmentLimitExceeded { .. } => (
            ProviderErrorClass::Capability,
            "attachment_count_unsupported",
        ),
        E::ViewportOutsideAttachment { .. } => (
            ProviderErrorClass::Capability,
            "viewport_extent_unsupported",
        ),
        E::DrawVertexCountMismatch { .. } => {
            (ProviderErrorClass::Capability, "draw_shape_unsupported")
        }
        // Everything else a pass or a pipeline can raise here is structural: the
        // class and slug core admission gives it.
        _ => (ProviderErrorClass::Args, "trace_contract_invalid"),
    };
    refusal(ProviderPhase::Resolve, class, slug).with_detail(error.to_string())
}

/// The refusal a render pass gets when it names a pipeline this context never
/// registered as a render pipeline.
///
/// One counter and one id namespace serve both rails, so the wording is the
/// mirror of [`crate::native`]'s `unknown_pipeline`: a compute pass naming a
/// render registration and a render pass naming a compute registration are both
/// resource refusals, with the rail that refused them as the slug.
fn unknown_render_pipeline(id: PipelineId) -> ProviderError {
    refusal(
        ProviderPhase::Resolve,
        ProviderErrorClass::Resource,
        "unknown_render_pipeline",
    )
    .with_field("pipeline", FieldValue::Unsigned(id.get()))
}

/// One render pass of a trace, planned before any device object exists.
///
/// The pass and the contract are borrowed from values that outlive execution —
/// the trace itself and the provider's render registrations — while the landing
/// view points into the serial view pool the encoder is built from. The planned
/// [`RenderPlan`] is the same value the encoder body consumes, so planning once
/// and encoding later cannot disagree.
#[derive(Debug)]
pub(crate) struct TraceRenderPlan<'a> {
    pub(crate) pass: &'a RenderPassDescriptor,
    pub(crate) contract: &'a RenderPipelineContract,
    /// One entry per colour attachment, in location order: the pool view whose
    /// identity covers the attachment, which is the writeback channel each
    /// attachment's texels leave through. `None` is the one arm that publishes
    /// no writeback *and* needs no declared view to read its previous bytes: a
    /// resident store keeps its bytes in the provider's image
    /// (`research/docs/23` §76, R7), so it needs no landing even when the trace
    /// asks for a host readback. Every other attachment — stored, discarded or
    /// loading — keeps the pre-R7 rule and resolves its view.
    pub(crate) landings: Vec<Option<&'a BufferView>>,
    /// The landing view of a stored depth attachment, or `None` for every
    /// shape whose depth surface does not outlive its pass
    /// (`research/docs/23` §3.3, v43): the pool view whose identity is the
    /// depth identity, which is the writeback channel the stored texels leave
    /// through. A pass with no depth attachment and one that discards it both
    /// land here as `None`, exactly as they state nothing about a landing.
    pub(crate) depth_landing: Option<&'a BufferView>,
    /// The landing view of a stored stencil attachment, or `None` for every
    /// shape whose stencil surface does not outlive its pass
    /// (`research/docs/23` §3.3, v49): the pool view whose identity is the
    /// stencil identity, which is the writeback channel the stored one-byte
    /// texels leave through. A pass with no stencil attachment and one that
    /// discards it both land here as `None`, exactly as they state nothing
    /// about a landing.
    pub(crate) stencil_landing: Option<&'a BufferView>,
    pub(crate) plan: RenderPlan<'a>,
    /// The present action hanging off this pass, if any, with its sentinel
    /// already expanded to the target's whole texel extent so the macOS
    /// present path only has to upload it (`research/docs/24` §3.1, §6 Step 7).
    pub(crate) present: Option<PresentPlan<'a>>,
}

/// One present action, planned before the first Metal object exists.
///
/// The descriptor is borrowed from the pass it hands on, and the sentinel is
/// the expanded byte string the present path presets into the target before
/// the render runs. `None` means the target declares [`InitialState::Undefined`]
/// and holds whatever the device gave it (`research/docs/24` §3.1).
#[derive(Debug)]
pub(crate) struct PresentPlan<'a> {
    /// The present descriptor the pass carries.
    pub(crate) descriptor: &'a PresentDescriptor,
    /// The sentinel texel replicated across the whole target, or `None`.
    pub(crate) sentinel: Option<Vec<u8>>,
}

impl TraceRenderPlan<'_> {
    /// The writebacks this pass's readbacks become: one [`BufferWriteback`] per
    /// *stored* landing view, in location order, followed by the stored depth
    /// attachment's own when the pass has one (`research/docs/23` §3.3, v43)
    /// and then the stored stencil attachment's own (`research/docs/23` §3.3,
    /// v49). The writable stage buffers come last, after the attachments' own
    /// landings (`research/docs/23` §3.3, v86/v92): their bytes are the pass's
    /// other observable output, read beside the texels that same submission
    /// wrote.
    ///
    /// The view identity, allocation and offset are each landing view's own, so
    /// resource admission, lease bookkeeping and readback consumers need no
    /// second path: every attachment lands exactly where a compute pass writing
    /// the same view would (`research/docs/23` §6 Step 7). A discarded
    /// attachment has no readback and no writeback: its landing view stays in
    /// the plan for the load-side resolution, but its bytes never leave the
    /// pass, so it cannot present a blank readback as "landed correctly"
    /// (`research/docs/23` §3.6, v19). The caller's `readback` therefore carries
    /// one entry per stored attachment, matching the filtered landings, plus
    /// the depth texels exactly when the pass stores that surface, and the
    /// stencil texels exactly when it stores that one.
    ///
    /// The depth and stencil writebacks are the same shape as the colour ones —
    /// the landing view's own identity and offset, and the surface's own
    /// `depth32float` or one-byte `stencil8` texels — so neither needs a second
    /// channel. The list they are appended to need not be in identity order
    /// itself: every caller folds it through [`merge_writebacks`], which is
    /// where the canonical order the core contract states is established.
    ///
    /// A *resident* store is the third arm that lands no writeback
    /// (`research/docs/23` §76, R7): its bytes stay in the provider's own
    /// image, where the pass that later loads them observes them, so it is
    /// filtered out here exactly as a discarded attachment is — and the encoder
    /// filters its readback by the same bit, which is what keeps the two lists
    /// paired. The pairing is by *publishing* attachment in location order, not
    /// by store action: both the stored and the resident arm state Metal's
    /// `Store`, so pairing on the action would shift every later attachment's
    /// bytes into the wrong writeback.
    pub(crate) fn writebacks(&self, readback: RenderReadback) -> Vec<BufferWriteback> {
        let mut writebacks: Vec<BufferWriteback> = Vec::new();
        let mut readbacks = readback.attachments.into_iter();
        for (landing, attachment) in self.landings.iter().zip(&self.plan.attachments) {
            if !attachment.publishes {
                continue;
            }
            // The plan resolves one landing view per publishing attachment, and
            // the encoder reads back exactly those attachments in location
            // order, so the two iterators advance together. The `debug_assert`
            // keeps a hand-built plan from silently pairing bytes with the next
            // attachment's view.
            debug_assert!(
                landing.is_some(),
                "a publishing attachment is planned with its landing view"
            );
            if let (Some(landing), Some(bytes)) = (landing, readbacks.next()) {
                writebacks.push(BufferWriteback {
                    view_id: landing.view_id,
                    allocation_id: landing.allocation_id,
                    offset: landing.offset,
                    bytes,
                });
            }
        }
        // A discarded or absent depth surface has no readback, so it adds no
        // writeback: the bytes never left the pass (`research/docs/23` §3.3,
        // v43).
        if let (Some(landing), Some(texels)) = (self.depth_landing, readback.depth) {
            writebacks.push(BufferWriteback {
                view_id: landing.view_id,
                allocation_id: landing.allocation_id,
                offset: landing.offset,
                bytes: texels,
            });
        }
        // A discarded or absent stencil surface is the same story one byte
        // wide (`research/docs/23` §3.3, v49): no readback, no writeback.
        if let (Some(landing), Some(texels)) = (self.stencil_landing, readback.stencil) {
            writebacks.push(BufferWriteback {
                view_id: landing.view_id,
                allocation_id: landing.allocation_id,
                offset: landing.offset,
                bytes: texels,
            });
        }
        // A writable stage buffer is a landing like a stored attachment
        // (`research/docs/23` §3.3, v86/v92): one complete writeback for the
        // view the trace declared, in the pass's canonical binding order and in
        // the same byte-keyed channel. The encoder reads back exactly the
        // writable bindings, in that order, so the two lists advance together —
        // the `debug_assert` keeps a hand-built plan from publishing one
        // binding's bytes under the next one's identity.
        let writable = self
            .plan
            .stage_buffers
            .iter()
            .filter(|binding| binding.access.is_writable());
        debug_assert_eq!(
            readback.stage_buffers.len(),
            writable.clone().count(),
            "the encoder reads back one stage buffer per writable binding",
        );
        for (binding, landing) in writable.zip(readback.stage_buffers) {
            writebacks.push(BufferWriteback {
                view_id: binding.view_id,
                allocation_id: binding.allocation_id,
                offset: binding.offset,
                bytes: landing.bytes,
            });
        }
        writebacks
    }
}

/// Plan every render pass of a trace, without a device.
///
/// Four decisions have to be made before the first Metal object exists, and all
/// of them are answerable from values: the order the rails run in
/// ([`refuse_reordered_render_reads`], whose rule core admission also states as
/// part of the contract), the reviewed allowlist, each attachment's landing
/// view, and the previous bytes a loading pass uploads ([`previous_bytes`]).
/// `pool` is [`ComputeTrace::serial_resources`], the same pool the encoder
/// binds, and `contracts` holds the render contracts the provider registered
/// for the pipeline ids this trace names — a caller-supplied table entry is
/// checked against those registrations in `native.rs`, where the registry
/// lives. The pool's only job here is each attachment's landing view: a render
/// input carries its own bytes, so the streams a draw reads are resolved from
/// the pass itself ([`plan_vertex_input`]).
pub(crate) fn plan_trace<'a>(
    trace: &'a ComputeTrace,
    pool: &'a [BufferView],
    contracts: &'a BTreeMap<PipelineId, RenderPipelineContract>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<Vec<TraceRenderPlan<'a>>, ProviderError> {
    plan_trace_with_leases(
        trace,
        pool,
        contracts,
        None,
        None,
        depth_resolve_modes,
        stencil_resolve_modes,
    )
}

/// Plan every render pass of a trace, resolving lease-backed inputs through
/// `leases` (`research/docs/23` §72, R3d) and the resident targets through
/// `residents` (`research/docs/23` §76, R7).
///
/// The trace path's entry point: `native.rs` hands the provider's own lease
/// channel in, so a pass whose vertex or index view names a staged lease or an
/// owner window is resolved before the first Metal object exists, and the
/// provider's resident registry's answer per render pass, in trace order, one
/// bit per colour attachment: whether that attachment's resident declaration
/// resolved to a live provider image. [`plan_trace`] is the same plan with no
/// channel, which refuses those arms by name — the shape a caller with no
/// registries gets.
pub(crate) fn plan_trace_with_leases<'a>(
    trace: &'a ComputeTrace,
    pool: &'a [BufferView],
    contracts: &'a BTreeMap<PipelineId, RenderPipelineContract>,
    leases: Option<&RenderLeaseContext<'_>>,
    residents: Option<&[Vec<bool>]>,
    depth_resolve_modes: u32,
    stencil_resolve_modes: u32,
) -> Result<Vec<TraceRenderPlan<'a>>, ProviderError> {
    // A landing-only entry has no rail here (`research/docs/23` §115 之后的增量，
    // E-TX14/R4b). The native provider keeps resident images (its registry is the
    // sibling of the Vulkan one), but its owner-window channel is an *input*
    // route: there is no code path that writes an owner's pages, which is why
    // `StoreOp::Borrowed` has been refused by name since E-TX8. Delivering a kept
    // frame would need that write route, so the entry is refused here — before
    // the early return below, which a landing-only trace would otherwise take
    // (it carries no render pass at all).
    if let Some(landing) = trace.landings().next() {
        return Err(capability_refusal("kept_frame_landing_unsupported")
            .with_field("view", FieldValue::Unsigned(landing.frame.view_id.get()))
            .with_field(
                "allocation",
                FieldValue::Unsigned(landing.frame.allocation_id.get()),
            )
            .with_field("source", FieldValue::Text("native_rail".to_owned()))
            .with_detail(
                "a landing-only entry writes a frame the provider kept into an owner's \
                 registered window; this rail has no owner-window write route, so it refuses \
                 the entry instead of landing it somewhere else",
            ));
    }
    if !trace.has_render_passes() {
        return Ok(Vec::new());
    }
    // One resident list per render pass is the provider's answer for the whole
    // trace, so a list that does not line up with the passes is refused before
    // any of them is planned instead of being read as "the remaining passes
    // declare nothing" (`research/docs/23` §76, R7).
    if let Some(residents) = residents {
        if residents.len() != trace.render_passes().count() {
            return Err(
                capability_refusal("resident_target_undeclared").with_detail(format!(
                    "the resident-target list must carry one entry per render pass: {} passes, {} \
                 entries",
                    trace.render_passes().count(),
                    residents.len()
                )),
            );
        }
    }
    refuse_reordered_render_reads(trace)?;
    let mut planned = Vec::with_capacity(trace.render_passes().count());
    for (pass_index, pass) in trace.render_passes().enumerate() {
        let contract = contracts
            .get(&pass.pipeline)
            .ok_or_else(|| unknown_render_pipeline(pass.pipeline))?;
        // An indirect draw replays its pass through `MTLIndirectRenderCommand`
        // state, and the first indirect increment builds a command that carries
        // the pipeline state and the draw counts — not the vertex streams a
        // caller-held layout reads, and not the stage buffers a pass binds
        // beside them (`research/docs/23` §83, R9g). A pass that binds either
        // is refused here rather than replayed from buffers nothing bound
        // (`research/docs/25` §6 Step 7b; the same slug the ICB rail uses for
        // a shape it cannot replay).
        if matches!(
            trace.indirect.as_ref().map(|indirect| indirect.command),
            Some(IndirectCommandDescriptor::Draw { .. })
        ) && (!pass.vertex_buffers.is_empty()
            || pass.indices.is_some()
            || !pass.stage_buffers.is_empty())
        {
            return Err(capability_refusal("icb_command_unsupported").with_detail(
                "an indirect draw replays the vertex_id shape; a pass that binds caller-held \
                 vertex, index or stage buffer streams is not part of the first indirect \
                 increment",
            ));
        }
        // An attachment that no buffer view covers has no landing rail: the
        // texels would have nowhere to go, so the pass is refused instead of
        // being executed and dropped. Each declared view is resolved before its
        // load op because a loading pass reads its previous bytes from the same
        // declaration (`research/docs/23` §3.3). The one attachment that needs
        // no declaration is a resident store (`research/docs/23` §76, R7): its
        // bytes stay in the provider's own image and it publishes no writeback,
        // so it neither lands through the pool nor uploads anything out of it.
        // Every other arm — stored, discarded, or loading for its previous
        // bytes — keeps the pre-R7 rule unchanged.
        let resident = residents
            .and_then(|passes| passes.get(pass_index))
            .map(|list| list.as_slice());
        let mut landings = Vec::with_capacity(pass.color_attachments.len());
        let mut previous = Vec::with_capacity(pass.color_attachments.len());
        for (index, attachment) in pass.color_attachments.iter().enumerate() {
            let declared = pool.iter().find(|view| {
                view.view_id == attachment.view_id && view.allocation_id == attachment.allocation_id
            });
            // A resident store is the arm that publishes no writeback, so it
            // needs no landing view even when the trace asks for a host
            // readback. A resident *load* beside a trace-declared store is the
            // resident store's sibling one step earlier: it publishes, so it
            // resolves its landing exactly as any storing attachment does — and
            // a `LoadOp::Load` beside a resident store keeps the pre-R7 rule
            // too, because its previous bytes come from that same declaration.
            let landing_needed = !attachment.declares_resident_target()
                || attachment.store == StoreOp::Store
                || attachment.load == LoadOp::Load;
            let landing = if landing_needed {
                Some(declared.ok_or_else(|| {
                    capability_refusal("render_attachment_landing_unsupported")
                        .with_field("view", FieldValue::Unsigned(attachment.view_id.get()))
                        .with_field(
                            "allocation",
                            FieldValue::Unsigned(attachment.allocation_id.get()),
                        )
                        .with_detail(
                            "attachment bytes land through the buffer writeback channel, \
                         and this trace declares no buffer view covering the attachment",
                        )
                })?)
            } else {
                None
            };
            // An offscreen `Load` uploads the contents the declaring view
            // carries before the pass opens — its own bytes, a staged lease's
            // copy, or the owner's mapping (`research/docs/23` §74, R5b). A
            // present pass's `Load` keeps the target's own initial state, which
            // the present path supplies, so it resolves no bytes
            // (`research/docs/24` §3.1), and a resident load's bytes are the
            // provider's image, which `previous_source` states as "nothing to
            // resolve" (`research/docs/23` §76, R7).
            let source = match (pass.present.is_none(), landing) {
                (true, Some(declared)) => {
                    previous_source(attachment.load, declared, leases, index)?
                }
                // The one arm with no landing view is a resident declaration
                // that neither publishes nor loads trace-declared bytes, so
                // there is no declaration to resolve anything out of.
                _ => None,
            };
            landings.push(landing);
            previous.push(source);
        }
        // The stored depth attachment's landing view, resolved before the pass
        // runs for the same reason the colour ones are: the texels it receives
        // have to be named by the trace, and a storing surface without a
        // declaration is refused instead of executed and dropped
        // (`research/docs/23` §3.3, v43). Every other shape — no depth
        // attachment, or one the pass discards with itself — states no landing
        // and resolves none.
        let depth_landing = match pass.depth.as_ref() {
            Some(depth) => match (depth.store, depth.identity) {
                (Some(DepthStoreOp::Store), Some(identity)) => Some(
                    pool.iter()
                        .find(|view| {
                            view.view_id == identity.view_id
                                && view.allocation_id == identity.allocation_id
                        })
                        .ok_or_else(|| {
                            capability_refusal("render_depth_landing_unsupported")
                                .with_field("view", FieldValue::Unsigned(identity.view_id.get()))
                                .with_field(
                                    "allocation",
                                    FieldValue::Unsigned(identity.allocation_id.get()),
                                )
                                .with_detail(
                                    "a stored depth attachment's texels land through the buffer \
                                     writeback channel, and this trace declares no buffer view \
                                     covering the attachment",
                                )
                        })?,
                ),
                _ => None,
            },
            None => None,
        };
        // The stored stencil attachment's landing view is the depth rule one
        // byte wide (`research/docs/23` §3.3, v49): resolved before the pass
        // runs because the texels it receives have to be named by the trace,
        // and refused by name when no declaration covers them, rather than
        // executed and dropped. Every other shape — no stencil attachment, or
        // one the pass discards with itself — states no landing and resolves
        // none.
        let stencil_landing = match pass.stencil.as_ref() {
            Some(stencil) => match (stencil.store, stencil.identity) {
                (Some(StoreOp::Store), Some(identity)) => Some(
                    pool.iter()
                        .find(|view| {
                            view.view_id == identity.view_id
                                && view.allocation_id == identity.allocation_id
                        })
                        .ok_or_else(|| {
                            capability_refusal("render_stencil_landing_unsupported")
                                .with_field("view", FieldValue::Unsigned(identity.view_id.get()))
                                .with_field(
                                    "allocation",
                                    FieldValue::Unsigned(identity.allocation_id.get()),
                                )
                                .with_detail(
                                    "a stored stencil attachment's texels land through the buffer \
                                     writeback channel, and this trace declares no buffer view \
                                     covering the attachment",
                                )
                        })?,
                ),
                _ => None,
            },
            None => None,
        };
        let plan_of_pass = plan_with_leases(
            &OffscreenRenderRequest {
                pass,
                pipeline: contract,
                source: reviewed_module_for(contract).map_or("", |module| module.source),
                initial: previous,
                // The provider's own answer for this pass, in the same
                // location order as the attachments (`research/docs/23` §76,
                // R7). An absent list is "no resident registry", which
                // `plan_with_leases` refuses a resident declaration with by
                // name.
                resident: resident.map(|list| list.to_vec()).unwrap_or_default(),
            },
            leases,
            depth_resolve_modes,
            stencil_resolve_modes,
        )?;
        let present = pass.present.as_ref().map(|descriptor| {
            let sentinel = descriptor.target.initial.sentinel().map(|texel| {
                let texels = usize::try_from(
                    u64::from(plan_of_pass.extent[0])
                        .saturating_mul(u64::from(plan_of_pass.extent[1])),
                )
                .unwrap_or(0);
                texel.repeat(texels)
            });
            PresentPlan {
                descriptor,
                sentinel,
            }
        });
        planned.push(TraceRenderPlan {
            pass,
            contract,
            landings,
            depth_landing,
            stencil_landing,
            plan: plan_of_pass,
            present,
        });
    }
    Ok(planned)
}

/// Fold compute and render writebacks into the one canonical list a submission
/// returns.
///
/// Compute and render writebacks share one channel and one rule: one complete
/// writeback per written view, keyed by identity. A view both rails could have
/// written is refused by core admission (`AttachmentComputeConflict`), and the
/// render rail runs last, so a repeated key keeps the bytes the render pass
/// ended with. The attachment's own view is in the serial pool as a written view
/// (`ComputeTrace::serial_resources`), which is exactly why the merge — and not
/// the compute readback — is what lands its texels.
pub(crate) fn merge_writebacks(
    compute: Vec<BufferWriteback>,
    render: Vec<BufferWriteback>,
) -> Vec<BufferWriteback> {
    let mut merged = BTreeMap::new();
    for writeback in compute.into_iter().chain(render) {
        merged.insert((writeback.allocation_id, writeback.view_id), writeback);
    }
    merged.into_values().collect()
}

/// Keeps every owner mapping one planned pass reads imported until that pass's
/// command buffer is terminal (`research/docs/23` §72, R3d).
///
/// The order is the compute rail's: retain before the first mapping is made,
/// retire once Metal has retired the work. This rail is synchronous —
/// `encode_into_and_readback` commits and waits before it returns — so every
/// exit from the encode call is a retirement point: the caller drops this guard
/// at the end of the pass it belongs to, after the wait. A failure that never
/// reached the queue drops it too; nothing was queued to read the mapping, and
/// the retain is this rail's own, so releasing it blocks nobody.
pub(crate) struct RenderInputRetains {
    registry: Arc<BorrowedLeaseRegistry>,
    lease_ids: Vec<LeaseId>,
    /// Whether the holds are still outstanding. Cleared by [`Self::retire`].
    armed: bool,
}

impl RenderInputRetains {
    /// Retain every no-copy lease the plan reads, before a single buffer is
    /// mapped. A plan whose inputs are all uploaded bytes retains nothing.
    pub(crate) fn retain(
        registry: &Arc<BorrowedLeaseRegistry>,
        plan: &RenderPlan<'_>,
    ) -> Result<Self, ProviderError> {
        let lease_ids = plan.borrowed_leases();
        registry.retain_all(&lease_ids)?;
        Ok(Self {
            registry: Arc::clone(registry),
            lease_ids,
            armed: true,
        })
    }

    /// Metal can no longer read the owner's mappings: drop every hold.
    pub(crate) fn retire(&mut self) {
        if self.armed {
            self.registry.retire_all(&self.lease_ids);
            self.armed = false;
        }
    }
}

impl Drop for RenderInputRetains {
    fn drop(&mut self) {
        self.retire();
    }
}

/// The texels one offscreen render pass hands back (`research/docs/23` §3.3,
/// v43/v49).
///
/// `attachments` carries one entry per *stored* colour attachment, in location
/// order: the shape the readback channel had before the depth attachment could
/// be observed. `depth` carries the stored depth surface's own tightly packed
/// `depth32float` texels — four bytes per texel over the pass's extent, the
/// same region [`read_texels`] reads a colour attachment from — or `None` when
/// the pass discards its depth attachment, which is what every pre-v43 trace
/// states. `stencil` carries the stored stencil surface's own tightly packed
/// `stencil8` texels — one byte per texel over the same region, read by
/// [`read_stencil_texels`] — or `None` when the pass discards its stencil
/// attachment, which is what every pre-v49 trace states.
#[derive(Debug)]
pub(crate) struct RenderReadback {
    pub(crate) attachments: Vec<Vec<u8>>,
    pub(crate) depth: Option<Vec<u8>>,
    pub(crate) stencil: Option<Vec<u8>>,
    /// The bytes every *writable* stage buffer holds after the pass, in
    /// canonical binding order (`research/docs/23` §3.3, v86/v92). Empty for
    /// every pass whose stage buffers are read-only, which is the shape every
    /// pre-R9k pass states.
    pub(crate) stage_buffers: Vec<StageBufferReadback>,
}

/// One writable stage buffer's bytes, read once the pass's fence has signalled
/// (`research/docs/23` §3.3, v86/v92).
///
/// The entry names the binding the way the contract does — stage plus the index
/// inside that stage's own namespace — so the writeback the plan publishes is
/// resolved against the pass's own view without re-deriving the slot from a
/// position.
#[derive(Debug)]
pub(crate) struct StageBufferReadback {
    pub(crate) stage: RenderPipelineStage,
    pub(crate) index: u32,
    pub(crate) bytes: Vec<u8>,
}

/// Execute one offscreen render pass and return its tightly packed texel bytes,
/// one readback per colour attachment in location order, the stored depth
/// surface's own when the pass has one and the stored stencil surface's own
/// when it has that one.
///
/// Not verified on an Apple GPU: the check that would verify this encoder body
/// is the Rust provider's own render path in a committed suite, and the macOS
/// oracle's `--render-selftest` run `34774478149` is the observation behind the
/// capability flip (`conformance/RENDER-CAPTURE.md` §5, §6).
///
/// This entry point has no trace, but a render input carries its own bytes, so a
/// pass that binds vertex or index streams still executes from the bytes it
/// declared (`research/docs/23` §3.6). What such a call does not have is an
/// attachment landing view, which is why the trace path is
/// [`plan_trace`] + [`encode_offscreen_render`]: the texels of this entry point
/// are returned to the caller instead of landing in a pooled view.
#[cfg(target_os = "macos")]
pub(crate) fn execute_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    request: &OffscreenRenderRequest<'_>,
) -> Result<RenderReadback, ProviderError> {
    // The depth-resolve modes the rail re-asserts come from the same device
    // probe the capability snapshot published, so a directly-constructed
    // request is refused by the same mask admission used.
    let planned = plan(
        request,
        device_depth_resolve_capability_bits(device).depth_resolve_modes,
        device_stencil_resolve_capability_bits(device).stencil_resolve_modes,
    )?;
    encode_offscreen_render(device, queue, &planned)
}

/// Encode, commit and read back one already planned pass.
///
/// Split from [`execute_offscreen_render`] so the trace path can plan once
/// ([`plan_trace`], before the compute command buffer is committed) and then
/// encode that same decision, instead of planning a second, possibly different,
/// pass. The attachments are fresh per pass: they are created here and dropped
/// with the readbacks, which is the offscreen shape (`research/docs/23` §6
/// Step 7).
#[cfg(target_os = "macos")]
pub(crate) fn encode_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
) -> Result<RenderReadback, ProviderError> {
    // The pre-R7 shape: every attachment is a fresh texture this rail creates
    // and drops with the readbacks. A plan that declares a resident target has
    // to arrive through [`encode_offscreen_render_with_resident`], which is
    // where the provider hands its own images over; encoding it here would
    // render the pass into a fresh attachment the trace never named, which is
    // exactly the silent downgrade the arm exists to prevent
    // (`research/docs/23` §76, R7).
    refuse_unresolved_resident_targets(planned, &[])?;
    objc::rc::autoreleasepool(|| {
        let attachments = attachment_textures(device, planned, &[])?;
        encode_into_and_readback(device, queue, planned, &attachments, None)
    })
}

/// Encode, commit and read back one already planned offscreen pass whose
/// resident attachments render into the provider's own images
/// (`research/docs/23` §76, R7).
///
/// `residents` is the provider's answer for the plan's colour attachments, in
/// location order: `Some(texture)` exactly for the attachments the plan
/// declares as resident, `None` for every attachment this rail creates itself.
/// The provider resolved those images — and refused every way an identity can
/// be gone — before the plan existed (`native.rs::resolve_render_residents`),
/// so this call is the encoder half of the same decision: it renders the pass
/// into those textures, reads back the publishing attachments, and leaves the
/// resident ones' bytes in the provider's images.
#[cfg(target_os = "macos")]
pub(crate) fn encode_offscreen_render_with_resident(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    residents: &[Option<Texture>],
) -> Result<RenderReadback, ProviderError> {
    refuse_unresolved_resident_targets(planned, residents)?;
    objc::rc::autoreleasepool(|| {
        let attachments = attachment_textures(device, planned, residents)?;
        encode_into_and_readback(device, queue, planned, &attachments, None)
    })
}

/// Refuse a plan the encoder cannot match its resident images to.
///
/// Two ways the two lists can disagree, both refused by name before the first
/// Metal object of this call exists (`research/docs/23` §76, R7): an attachment
/// the plan declares resident without a provider image would be rendered into a
/// fresh texture under the resident name, and an image handed over for an
/// attachment the plan declares as a per-pass one would render the provider's
/// bytes where the trace asked for a fresh attachment.
#[cfg(target_os = "macos")]
fn refuse_unresolved_resident_targets(
    planned: &RenderPlan<'_>,
    residents: &[Option<Texture>],
) -> Result<(), ProviderError> {
    if residents.is_empty() {
        if let Some(attachment) = planned
            .attachments
            .iter()
            .position(|attachment| attachment.resident)
        {
            return Err(capability_refusal("resident_target_undeclared")
                .with_field("attachment", FieldValue::Unsigned(attachment as u64))
                .with_detail(
                    "the plan declares the provider-resident target and the caller handed over no \
                 provider image for it",
                ));
        }
        return Ok(());
    }
    if residents.len() != planned.attachments.len()
        || residents
            .iter()
            .zip(&planned.attachments)
            .any(|(texture, attachment)| texture.is_some() != attachment.resident)
    {
        return Err(
            capability_refusal("resident_target_undeclared").with_detail(format!(
            "the provider's resident images must carry one entry per colour attachment, `Some` \
             exactly where the plan declares one: {} attachments, {} entries",
            planned.attachments.len(),
            residents.len()
        )),
        );
    }
    Ok(())
}

/// Encode, commit and read back one already planned present pass into the
/// caller-held target texture.
///
/// The target is the present action's own texture, held by (allocation, view)
/// across submissions (`research/docs/24` §6 Step 7); this function renders the
/// pass's attachment into it and reads the target back after `wait`, but it
/// neither creates nor destroys the texture. The acquire/present counts live in
/// `native.rs`, which calls this between its two counter increments.
///
/// The whole readback is returned rather than the target's texels alone
/// (`research/docs/23` §92, R9k): a present pass may bind stage buffers beside
/// its attachment, and a writable one lands exactly as it does on the offscreen
/// rail — the caller publishes the attachment and every writable binding
/// through one writeback list.
#[cfg(target_os = "macos")]
pub(crate) fn encode_present_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    target: &Texture,
) -> Result<RenderReadback, ProviderError> {
    objc::rc::autoreleasepool(|| {
        encode_into_and_readback(device, queue, planned, std::slice::from_ref(target), None)
    })
}

/// Encode, commit and read back one already planned offscreen pass whose
/// full-screen triangle is replayed from one `MTLIndirectCommandBuffer`
/// (`research/docs/25` §6 Step 7b).
///
/// The pass shape rules are the ones [`encode_offscreen_render`] already
/// enforces — this entry point only swaps the direct draw for an indirect
/// replay, so a pass that is not admitted as an offscreen render cannot reach
/// it. The draw command's vertex and instance counts are the ones the
/// [`crate::icb::plan_replay`] narrowed to a non-indexed draw.
#[cfg(target_os = "macos")]
pub(crate) fn encode_indirect_offscreen_render(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    replay: &icb::IcbPlan,
    residents: &[Option<Texture>],
) -> Result<RenderReadback, ProviderError> {
    // `plan_trace` refuses an indirect draw whose pass binds streams, because
    // the replay shape this rail builds carries the pipeline state and the draw
    // counts rather than the streams a caller-held layout reads. This is the
    // same rule one level down, for a caller that reaches the encoder without a
    // trace plan.
    if !planned.vertex_streams.is_empty()
        || planned.indices.is_some()
        || !planned.stage_buffers.is_empty()
    {
        return Err(capability_refusal("icb_command_unsupported").with_detail(
            "an indirect draw replays the vertex_id shape; a pass that binds caller-held vertex, \
             index or stage buffer streams is not part of the first indirect increment",
        ));
    }
    // The replay renders into the same textures the direct draw does, so the
    // resident images arrive through the same channel and are matched by the
    // same rule (`research/docs/23` §76, R7).
    refuse_unresolved_resident_targets(planned, residents)?;
    objc::rc::autoreleasepool(|| {
        let attachments = attachment_textures(device, planned, residents)?;
        encode_into_and_readback(device, queue, planned, &attachments, Some(*replay))
    })
}

/// The shared encoder body of the offscreen and present rails: build the
/// reviewed pipeline, render the pass into `targets`, wait for a terminal
/// command-buffer status, and read every attachment's texels back — the stored
/// depth and stencil surfaces' own included when the plan has them
/// (`research/docs/23` §3.3, v43/v49).
#[cfg(target_os = "macos")]
fn encode_into_and_readback(
    device: &Device,
    queue: &CommandQueue,
    planned: &RenderPlan<'_>,
    targets: &[Texture],
    indirect: Option<icb::IcbPlan>,
) -> Result<RenderReadback, ProviderError> {
    let pipeline = render_pipeline_state(device, planned)?;
    // The sampled textures the reviewed sampling pair reads
    // (`research/docs/23` §3.3, v70): one shared-storage `MTLTexture` per
    // binding, filled with the trace's own texels, bound on the encoder below
    // and kept in this local for the same reason the depth texture is — the
    // encoder references it until it ends.
    let sampled_textures = sampled_textures(device, planned)?;
    // A multisampled pass renders into its own four-sample textures
    // (`research/docs/23` §3.3, v51): one per colour location, created beside
    // the resolve targets `targets` already holds. The resolve targets are what
    // the readback below observes; the pass descriptor names both, and the
    // textures stay in this local for the same reason the depth and stencil
    // textures do — the encoder references them until it ends.
    let multisample_targets = planned
        .multisample
        .map(|_| multisample_attachment_textures(device, planned))
        .transpose()?;
    // The pass descriptor is autoreleased; it only has to outlive the
    // encoder creation below.
    let pass = MetalRenderPassDescriptor::new();
    // One descriptor entry per colour location: the pass's `colorAttachments`
    // array is indexed by location, so entry `i` carries attachment `i`'s
    // texture and its own load/store actions.
    for (index, (attachment, target)) in planned.attachments.iter().zip(targets).enumerate() {
        let color = pass
            .color_attachments()
            .object_at(index as u64)
            .ok_or_else(|| resource_refusal("metal_render_attachment_descriptor_unavailable"))?;
        // A multisampled location names *two* textures: the four-sample surface
        // the fragments land in and the single-sample resolve target — the
        // attachment's own texture, which is what the trace observes
        // (`research/docs/23` §3.3, v51). A single-sample location names one,
        // exactly as every pre-v51 pass did.
        let multisampled = multisample_targets
            .as_ref()
            .map(|textures| &textures[index]);
        color.set_texture(Some(multisampled.unwrap_or(target)));
        if let Some(_multisampled) = multisampled {
            color.set_resolve_texture(Some(target));
        }
        match attachment.load {
            RenderLoadAction::Clear(components) => {
                color.set_load_action(MTLLoadAction::Clear);
                color.set_clear_color(MTLClearColor::new(
                    components[0],
                    components[1],
                    components[2],
                    components[3],
                ));
            }
            RenderLoadAction::Load => color.set_load_action(MTLLoadAction::Load),
            // `MTLLoadAction::DontCare` mirrors the contract: the pre-pass
            // contents are undefined, the pass neither reads nor presets them,
            // and the draw alone defines what the attachment stores
            // (`research/docs/23` §3.1, v20).
            RenderLoadAction::DontCare => color.set_load_action(MTLLoadAction::DontCare),
        }
        // A discarded attachment still renders, but its bytes are not kept:
        // the store action is what makes the attachment disappear from the
        // observable surface, and the readback below skips it
        // (`research/docs/23` §3.6, v19).
        match (multisampled, attachment.store) {
            // The resolve is the only landing a multisampled location has:
            // `.multisampleResolve` writes the resolved texels into the resolve
            // texture and does not keep the four-sample surface, which is
            // exactly the "resolve into the attachment view" shape the trace
            // states (`research/docs/23` §3.3, v51).
            (Some(_), RenderStoreAction::Store) => {
                color.set_store_action(MTLStoreAction::MultisampleResolve)
            }
            (Some(_), RenderStoreAction::DontCare) => {
                color.set_store_action(MTLStoreAction::DontCare)
            }
            (None, RenderStoreAction::Store) => color.set_store_action(MTLStoreAction::Store),
            (None, RenderStoreAction::DontCare) => color.set_store_action(MTLStoreAction::DontCare),
        }
    }
    // The depth attachment is the rail's own texture, created when the plan
    // declares one (`research/docs/23` §3.3, v36): the pass descriptor opens it
    // with the plan's load operation and stores or discards it after the draw
    // exactly as the trace stated (v43). A storing surface is read back below.
    // The texture and the depth-stencil state stay in these locals until the
    // readback below: the pass descriptor and the encoder reference them, and
    // the rail's objects are autoreleased at the end of the pool.
    // A resolving pass names two depth textures: the four-sample surface the
    // fragments land in (private, never read back) and the single-sample
    // landing the resolve writes — the v43 shared-storage readback texture,
    // which is what the readback below observes (`research/docs/23` §3.3,
    // v57c). Every non-resolving shape keeps the one texture `depth_texture`
    // builds, exactly as before.
    // The combined depth-stencil shape (`research/docs/23` §3.3, v60) names
    // one four-sample `depth32Float_stencil8` texture from both attachment
    // descriptors, so the two targets below share the one surface instead of
    // the two single-face textures the other shapes build.
    let combined = planned.depth.is_some() && planned.stencil.is_some();
    let raster_samples = planned.multisample.map(|count| u64::from(count.samples()));
    let combined_target = if combined {
        planned
            .depth
            .as_ref()
            .map(|depth| {
                let samples = raster_samples.ok_or_else(|| {
                    capability_refusal("render_multisample_state_unsupported").with_detail(
                        "a combined depth-stencil surface states a multisampled raster",
                    )
                })?;
                combined_depth_stencil_surface(device, depth, samples)
            })
            .transpose()?
    } else {
        None
    };
    let depth_resolving = planned.depth_resolve.is_some();
    let depth_target = planned
        .depth
        .as_ref()
        .map(|depth| {
            if combined_target.is_some() {
                Ok(combined_target
                    .clone()
                    .expect("the combined target exists beside the combined plan"))
            } else if depth_resolving {
                let samples = raster_samples.ok_or_else(|| {
                    capability_refusal("render_multisample_state_unsupported")
                        .with_detail("a depth resolve states a multisampled raster")
                })?;
                multisample_depth_surface(device, depth, samples)
            } else {
                depth_texture(device, depth, planned.multisample)
            }
        })
        .transpose()?;
    let depth_resolve_target = if depth_resolving {
        planned
            .depth
            .as_ref()
            .map(|depth| depth_texture(device, depth, None))
            .transpose()?
    } else {
        None
    };
    if let (Some(depth), Some(texture)) = (&planned.depth, &depth_target) {
        let attachment = pass
            .depth_attachment()
            .ok_or_else(|| resource_refusal("metal_render_depth_descriptor_unavailable"))?;
        attachment.set_texture(Some(texture));
        match depth.clear_bits {
            Some(bits) => {
                attachment.set_load_action(MTLLoadAction::Clear);
                attachment.set_clear_depth(f64::from(f32::from_bits(bits)));
            }
            None => attachment.set_load_action(MTLLoadAction::Load),
        }
        // The store action is the pass's own statement
        // (`research/docs/23` §3.3, v43/v57c): a resolving stored surface
        // lands in its resolve target through `.multisampleResolve`, a stored
        // single-sample surface keeps its own texels, and every other shape
        // discards the rail-owned surface with the pass.
        attachment.set_store_action(match (depth.storing(), planned.depth_resolve) {
            (true, Some(_)) => MTLStoreAction::MultisampleResolve,
            (true, None) => MTLStoreAction::Store,
            (false, _) => MTLStoreAction::DontCare,
        });
        // The two depth-resolve fields are a `metal` 0.33 binding gap: the
        // depth attachment descriptor has no setter for them, so the rail
        // sends the two selectors itself. The filter ordinal is the contract's
        // code (`MTLMultisampleDepthResolveFilterSample0/Min/Max` = 0/1/2),
        // and the resolve target is the single-sample landing above.
        if let (Some(filter), Some(landing)) =
            (planned.depth_resolve, depth_resolve_target.as_ref())
        {
            unsafe {
                let _: () = msg_send![attachment, setResolveTexture: Some(landing.as_ref())];
                let _: () = msg_send![attachment, setDepthResolveFilter: u64::from(filter.code())];
            }
        }
    }
    // The stencil attachment is rail-owned in the same way (`research/docs/23`
    // §3.3, v47): created when the plan declares one, opened with the plan's
    // load operation and the pass's own clear value, and stored or discarded
    // after the draw exactly as the trace stated (v49). A storing surface has
    // to survive the pass for its texels to leave it, which is what the
    // readback below observes; every pre-v49 shape discards the rail-owned
    // surface with the pass. The texture stays in this local for the same
    // reason the depth texture does: the pass descriptor references it until
    // the encoder is done.
    let stencil_target = planned
        .stencil
        .as_ref()
        .map(|stencil| {
            if let Some(combined) = &combined_target {
                Ok(combined.clone())
            } else {
                stencil_texture(device, stencil, planned.multisample)
            }
        })
        .transpose()?;
    // The single-sample landing the stencil resolve writes into — the v49
    // shared-storage readback texture, which is what the readback below
    // observes (`research/docs/23` §3.3, v60). Every non-resolving shape
    // carries none.
    let stencil_resolve_target = if planned.stencil_resolve.is_some() {
        planned
            .stencil
            .as_ref()
            .map(|stencil| stencil_texture(device, stencil, None))
            .transpose()?
    } else {
        None
    };
    if let (Some(stencil), Some(texture)) = (&planned.stencil, &stencil_target) {
        let attachment = pass
            .stencil_attachment()
            .ok_or_else(|| resource_refusal("metal_render_stencil_descriptor_unavailable"))?;
        attachment.set_texture(Some(texture));
        match stencil.clear_value {
            Some(value) => {
                attachment.set_load_action(MTLLoadAction::Clear);
                attachment.set_clear_stencil(u32::from(value));
            }
            None => attachment.set_load_action(MTLLoadAction::Load),
        }
        attachment.set_store_action(match (stencil.storing(), planned.stencil_resolve) {
            (true, Some(_)) => MTLStoreAction::MultisampleResolve,
            (true, None) => MTLStoreAction::Store,
            (false, _) => MTLStoreAction::DontCare,
        });
        // The two stencil-resolve fields are a `metal` 0.33 binding gap: the
        // stencil attachment descriptor has no setter for them, so the rail
        // sends the two selectors itself. The resolve target is the base
        // class's `resolveTexture` — not a `stencilResolveTexture` selector,
        // which the v57c depth work showed does not exist — and the filter
        // ordinal is the contract's code
        // (`MTLMultisampleStencilResolveFilterSample0/DepthResolvedSample` =
        // 0/1).
        if let (Some(filter), Some(landing)) =
            (planned.stencil_resolve, stencil_resolve_target.as_ref())
        {
            unsafe {
                let _: () = msg_send![attachment, setResolveTexture: Some(landing.as_ref())];
                let _: () =
                    msg_send![attachment, setStencilResolveFilter: u64::from(filter.code())];
            }
        }
    }
    // Metal carries the depth and the stencil state in one descriptor
    // (`research/docs/23` §3.3, v36/v47): built when the pass opens either
    // surface, because a stencil-only pass has no depth test to state and a
    // depth-only pass no stencil state. The depth half is the pass's own test,
    // or `Always` with no write for an attachment nothing tests; the stencil
    // half arms the front and back descriptors with the same state, exactly as
    // the contract states it for both faces.
    let depth_stencil_state = (planned.depth.is_some() || planned.stencil.is_some()).then(|| {
        let descriptor = DepthStencilDescriptor::new();
        let (compare, write) = match planned.depth.as_ref().and_then(|depth| depth.test) {
            Some(test) => (
                match test.compare {
                    CompareFunction::Less => MTLCompareFunction::Less,
                    CompareFunction::Always => MTLCompareFunction::Always,
                },
                test.write,
            ),
            // An attachment with no test still clears; every fragment passes.
            // A pass with no depth attachment at all keeps these same defaults,
            // spelled out because the two halves travel in one object.
            None => (MTLCompareFunction::Always, false),
        };
        descriptor.set_depth_compare_function(compare);
        descriptor.set_depth_write_enabled(write);
        if let Some(test) = planned.stencil.as_ref().and_then(|stencil| stencil.test) {
            let stencil_descriptor = StencilDescriptor::new();
            stencil_descriptor.set_stencil_compare_function(metal_stencil_compare(test.compare));
            stencil_descriptor.set_stencil_failure_operation(metal_stencil_operation(test.fail_op));
            stencil_descriptor
                .set_depth_failure_operation(metal_stencil_operation(test.depth_fail_op));
            stencil_descriptor
                .set_depth_stencil_pass_operation(metal_stencil_operation(test.pass_op));
            stencil_descriptor.set_read_mask(u32::from(test.read_mask));
            stencil_descriptor.set_write_mask(u32::from(test.write_mask));
            descriptor.set_front_face_stencil(Some(&stencil_descriptor));
            descriptor.set_back_face_stencil(Some(&stencil_descriptor));
        }
        device.new_depth_stencil_state(&descriptor)
    });
    // The command buffer and the encoder are autoreleased and the rail is
    // synchronous, so neither has to be retained: nothing here outlives this
    // pool.
    let command = queue.new_command_buffer();
    // The multisampled load's seed pass (`research/docs/23` §82, v82) is the
    // first encoder of the command buffer: one `CLEAR`-opened, `STORE`d colour
    // attachment per seeded location — the very multisampled textures the
    // measured encoder below names — so every sample of those texels holds the
    // plan's own seed when the measured pass opens them with
    // `MTLLoadAction::Load`. The load action is the whole work: no pipeline
    // state, no draw, and therefore no pipeline the rail would have to own.
    if planned
        .attachments
        .iter()
        .any(|attachment| attachment.seed.is_some())
    {
        let multisampled = multisample_targets.as_ref().ok_or_else(|| {
            capability_refusal("render_multisample_load_unsupported").with_detail(
                "a seeded multisampled load needs the n-sample textures the seed pass clears",
            )
        })?;
        let seed_pass = MetalRenderPassDescriptor::new();
        for (index, attachment) in planned.attachments.iter().enumerate() {
            let Some(components) = attachment.seed else {
                continue;
            };
            let color = seed_pass
                .color_attachments()
                .object_at(index as u64)
                .ok_or_else(|| {
                    resource_refusal("metal_render_attachment_descriptor_unavailable")
                })?;
            color.set_texture(Some(&multisampled[index]));
            color.set_load_action(MTLLoadAction::Clear);
            color.set_clear_color(MTLClearColor::new(
                components[0],
                components[1],
                components[2],
                components[3],
            ));
            color.set_store_action(MTLStoreAction::Store);
        }
        let seed_encoder = command.new_render_command_encoder(seed_pass);
        seed_encoder.end_encoding();
    }
    let encoder = command.new_render_command_encoder(pass);
    encoder.set_render_pipeline_state(&pipeline);
    // Metal's depth state is encoder state (`research/docs/23` §3.3, v36): the
    // compare function and the write enable are the pass's own, exactly as the
    // contract states them.
    if let Some(state) = &depth_stencil_state {
        encoder.set_depth_stencil_state(state);
    }
    // The stencil reference value is encoder state too, not pipeline state
    // (`research/docs/23` §3.3, v47): Metal takes it per draw call, so the
    // pass's own value is set once before the draw exactly as the contract
    // states it.
    if let Some(test) = planned.stencil.as_ref().and_then(|stencil| stencil.test) {
        encoder.set_stencil_reference_value(u32::from(test.reference));
    }
    // Culling is encoder state too (`research/docs/23` §3.3, v39): the mode and
    // the winding are the pass's own, and a pass without the state keeps
    // Metal's defaults (cull none, counter-clockwise front) exactly.
    if let Some(cull) = &planned.cull {
        encoder.set_cull_mode(match cull.mode {
            ContractCullMode::None => MTLCullMode::None,
            ContractCullMode::Front => MTLCullMode::Front,
            ContractCullMode::Back => MTLCullMode::Back,
        });
        encoder.set_front_facing_winding(match cull.winding {
            ContractWinding::Clockwise => MTLWinding::Clockwise,
            ContractWinding::CounterClockwise => MTLWinding::CounterClockwise,
        });
    }
    // The vertex streams the plan resolved, bound at the same indices the
    // pipeline's `MTLVertexDescriptor` names. The MTLBuffers are kept for the
    // whole call: they have to outlive the encoder that reads them, and the
    // plan's bytes do not.
    let mut stream_buffers = Vec::with_capacity(planned.vertex_streams.len() + 1);
    for stream in &planned.vertex_streams {
        let offset = NSUInteger::try_from(stream.binding_offset()).unwrap_or(NSUInteger::MAX);
        let buffer = stream_buffer(device, &stream.source, stream.binding_offset())?;
        encoder.set_vertex_buffer(
            NSUInteger::from(stream.buffer_index),
            Some(buffer.as_ref()),
            offset,
        );
        stream_buffers.push(buffer);
    }
    // The pass's stage buffers (`research/docs/23` §83/R9g, §92/R9k), bound at
    // the slot each stage's own `[[buffer(N)]]` namespace names: a vertex
    // binding is a `setVertexBuffer(_:offset:index:)` and a fragment binding a
    // `setFragmentBuffer(_:offset:index:)`, so a vertex `0` and a fragment `0`
    // fill two different slots. A writable binding is bound exactly the same
    // way — Metal has no read-only view of a buffer — and the readback below
    // reads its bytes back out of the same `MTLBuffer` once the fence has
    // signalled, which is what makes the write a landing rather than a dropped
    // side effect (`research/docs/23` §3.3, v86/v92). The MTLBuffers are kept
    // for the whole call beside the streams' own, for the same reason: they
    // have to outlive the encoder that reads them — and, for the writable ones,
    // the readback that observes what it wrote.
    let mut stage_buffers = Vec::with_capacity(planned.stage_buffers.len());
    for stage in &planned.stage_buffers {
        let offset = NSUInteger::try_from(stage.binding_offset()).unwrap_or(NSUInteger::MAX);
        let buffer = stream_buffer(device, &stage.source, stage.binding_offset())?;
        match stage.stage {
            RenderPipelineStage::Vertex => encoder.set_vertex_buffer(
                NSUInteger::from(stage.index),
                Some(buffer.as_ref()),
                offset,
            ),
            RenderPipelineStage::Fragment => encoder.set_fragment_buffer(
                NSUInteger::from(stage.index),
                Some(buffer.as_ref()),
                offset,
            ),
        }
        stage_buffers.push(buffer);
    }
    // The viewport is the pass's own rect (`research/docs/23` §3.3, v100): the
    // covering default is inside the attachments' extent, and a pass that
    // declares a smaller or offset rect states the rect this encoder records,
    // so the texels it does not cover keep the load's own bytes. The scissor
    // below it is the pass's own rectangle when it declares one, and the render
    // area otherwise (`research/docs/23` §3.3, v29).
    encoder.set_viewport(MTLViewport {
        originX: f64::from(planned.viewport[0]),
        originY: f64::from(planned.viewport[1]),
        width: f64::from(planned.viewport[2]),
        height: f64::from(planned.viewport[3]),
        znear: 0.0,
        zfar: 1.0,
    });
    let [scissor_x, scissor_y, scissor_width, scissor_height] =
        planned
            .scissor
            .unwrap_or([0, 0, planned.extent[0], planned.extent[1]]);
    encoder.set_scissor_rect(metal::MTLScissorRect {
        x: metal::NSUInteger::from(scissor_x),
        y: metal::NSUInteger::from(scissor_y),
        width: metal::NSUInteger::from(scissor_width),
        height: metal::NSUInteger::from(scissor_height),
    });
    // The sampled textures are bound before the draw at the index each entry's
    // own view states (`research/docs/23` §3.3, v70/v104): the plan carries the
    // Metal `[[texture(n)]]` argument the fragment stage reads, which is the
    // contract's indexed binding rather than the entry's position.
    for (sampled, texture) in planned.textures.iter().zip(&sampled_textures) {
        encoder.set_fragment_texture(u64::from(sampled.binding), Some(texture.as_ref()));
    }
    match indirect {
        None => match &planned.indices {
            // An indexed draw names its index buffer in the draw call, which is
            // where Metal takes it: the contract's index binding becomes one
            // `drawIndexedPrimitives(indexCount:indexType:indexBuffer:
            // indexBufferOffset:)`, with the count the pass carries in the
            // indexed shape.
            Some(indices) => {
                let offset =
                    NSUInteger::try_from(indices.binding_offset()).unwrap_or(NSUInteger::MAX);
                let buffer = stream_buffer(device, &indices.source, indices.binding_offset())?;
                if indices.base_vertex == 0 {
                    encoder.draw_indexed_primitives_instanced(
                        MTLPrimitiveType::Triangle,
                        u64::from(indices.index_count),
                        metal_index_type(indices.format),
                        buffer.as_ref(),
                        offset,
                        u64::from(planned.instance_count),
                    );
                } else {
                    // The offset belongs to the draw call
                    // (`research/docs/23` §3.3, v34): the base-vertex entry
                    // states it beside the instance count, and a zero-offset
                    // draw keeps the pre-v34 entry point exactly.
                    let base_vertex =
                        NSInteger::try_from(indices.base_vertex).unwrap_or(NSInteger::MAX);
                    encoder.draw_indexed_primitives_instanced_base_instance(
                        MTLPrimitiveType::Triangle,
                        u64::from(indices.index_count),
                        metal_index_type(indices.format),
                        buffer.as_ref(),
                        offset,
                        u64::from(planned.instance_count),
                        base_vertex,
                        0,
                    );
                }
                stream_buffers.push(buffer);
            }
            None => {
                // The instanced draw call (`research/docs/23` §3.3, v31): the
                // same entry point with the pass's second count, which is `1`
                // for every pre-v31 pass and therefore byte-identical to the
                // single-instance call it replaces.
                encoder.draw_primitives_instanced(
                    MTLPrimitiveType::Triangle,
                    0,
                    u64::from(planned.vertices),
                    u64::from(planned.instance_count),
                )
            }
        },
        Some(replay) => {
            let icb::IcbCommand::Draw {
                vertex_count,
                instance_count,
            } = replay.command
            else {
                return Err(capability_refusal("icb_command_unsupported")
                    .with_field("kind", FieldValue::Text("dispatch".to_owned()))
                    .with_detail("the first indirect increment replays non-indexed draws only"));
            };
            let descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.set_command_types(MTLIndirectCommandType::Draw);
            let buffer = device.new_indirect_command_buffer_with_descriptor(
                &descriptor,
                u64::from(replay.max_commands),
                MTLResourceOptions::StorageModeShared,
            );
            if buffer.as_ptr().is_null() {
                return Err(resource_refusal(
                    "metal_indirect_command_buffer_allocation_failed",
                ));
            }
            let command = buffer.indirect_render_command_at_index(u64::from(replay.range.start));
            command.set_render_pipeline_state(&pipeline);
            command.draw_primitives(
                MTLPrimitiveType::Triangle,
                0,
                u64::from(vertex_count),
                u64::from(instance_count),
                0,
            );
            encoder.execute_commands_in_buffer(
                &buffer,
                NSRange::new(u64::from(replay.range.start), u64::from(replay.range.count)),
            );
        }
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        // `MTLCommandBufferStatus` is a `#[repr(u32)]` enum, so the numeric
        // status is the diagnostic a field-less error can carry.
        let status = command.status() as u32;
        return Err(resource_refusal("metal_render_command_failed")
            .with_detail(format!("command buffer ended with status {status}")));
    }
    let attachments = planned
        .attachments
        .iter()
        .zip(targets)
        // The readback is the *publishing* arm's, which is the trace-declared
        // `StoreOp::Store` (`research/docs/23` §76, R7): a resident store keeps
        // its bytes in the provider's image and a discarded attachment drops
        // them, so neither one yields texels here — and the same bit filters
        // `TraceRenderPlan::writebacks`, which is what keeps the two lists
        // paired attachment by attachment.
        .filter(|(attachment, _)| attachment.publishes)
        // Each attachment is read with its own format's texel width: a stored
        // `Rgba16Float` location lands eight bytes per texel where its
        // neighbours land four (`research/docs/23` §78).
        .map(|(attachment, target)| read_texels(target, planned, attachment.texel))
        .collect::<Result<Vec<Vec<u8>>, ProviderError>>()?;
    // The stored depth surface's texels leave through the same `getBytes` shape
    // the colour attachments use (`research/docs/23` §3.3, v43): the texture is
    // shared storage exactly when the pass stores it (`depth_texture`), and the
    // region is the pass's own extent, which the contract holds to the colour
    // attachments'. A discarded or absent surface keeps its bytes on the
    // device and reads nothing back.
    // A resolving pass observes the resolve target — the single-sample landing
    // `depth_resolve_target` holds — while every non-resolving stored surface
    // is read back from its own texture (`research/docs/23` §3.3, v43/v57c).
    let depth = match (&planned.depth, &depth_target, &depth_resolve_target) {
        (Some(depth), _, Some(landing)) if depth.storing() => {
            Some(read_texels(landing, planned, planned.depth_texel)?)
        }
        (Some(depth), Some(texture), None) if depth.storing() => {
            Some(read_texels(texture, planned, planned.depth_texel)?)
        }
        _ => None,
    };
    // The stored stencil surface's texels leave through the same `getBytes`
    // shape one byte wide (`research/docs/23` §3.3, v49): the texture is shared
    // storage exactly when the pass stores it (`stencil_texture`), and the
    // region is the pass's own extent, which the contract holds to the stencil
    // surface's just as it holds the depth surface's. A discarded or absent
    // surface keeps its bytes on the device and reads nothing back.
    let stencil = match (&planned.stencil, &stencil_target, &stencil_resolve_target) {
        // A resolving pass observes the resolve target — the single-sample
        // landing `stencil_resolve_target` holds — while every non-resolving
        // stored surface is read back from its own texture
        // (`research/docs/23` §3.3, v49/v60).
        (Some(stencil), _, Some(landing)) if stencil.storing() => {
            Some(read_stencil_texels(landing, planned)?)
        }
        (Some(stencil), Some(texture), None) if stencil.storing() => {
            Some(read_stencil_texels(texture, planned)?)
        }
        _ => None,
    };
    // The writable stage buffers' bytes, read once the fence has signalled
    // (`research/docs/23` §3.3, v86/v92). Both source arms are read the way the
    // bytes already live: an uploaded binding lives in this rail's own shared
    // `MTLBuffer`, and a no-copy binding's `MTLBuffer` *is* the owner's mapping,
    // so `contents()` reaches the bytes the device just wrote either way. The
    // read publishes the *whole* view — `BufferWriteback` carries the view's own
    // extent — at the offset the binding was placed at, which is the extent the
    // writeback contract requires for a written view.
    let mut stage_readbacks = Vec::with_capacity(planned.stage_buffers.len());
    for (binding, buffer) in planned.stage_buffers.iter().zip(&stage_buffers) {
        if !binding.access.is_writable() {
            continue;
        }
        let offset = usize::try_from(binding.binding_offset())
            .map_err(|_| resource_refusal("metal_render_stream_offset_overflow"))?;
        let length = usize::try_from(binding.length)
            .map_err(|_| resource_refusal("metal_render_stream_offset_overflow"))?;
        // SAFETY: the `MTLBuffer` was created (uploaded) or mapped (no-copy)
        // over at least `offset + length` bytes — the plan resolved the view's
        // own window — and it is still alive here because `stage_buffers` owns
        // every handle until the end of this call.
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(offset), length).to_vec()
        };
        stage_readbacks.push(StageBufferReadback {
            stage: binding.stage,
            index: binding.index,
            bytes,
        });
    }
    Ok(RenderReadback {
        attachments,
        depth,
        stencil,
        stage_buffers: stage_readbacks,
    })
}

/// The colour attachments this rail renders into, one texture per location.
///
/// `usage = RenderTarget` states what the texture is for, and the shared storage
/// mode is what makes the texels CPU-visible for the readback on the
/// unified-memory device the provider admits — the same reason the sampled
/// texture rail uses shared storage (`research/docs/16` §4.8).
///
/// `residents` is the provider's answer for the plan's attachments, in location
/// order (`research/docs/23` §76, R7): a location the plan declares resident
/// renders into the provider's own texture, which the encoder neither creates
/// nor uploads into — its previous contents are the bytes the provider kept, and
/// the plan resolved no `initial` source for it. Every other location is the
/// fresh shared-storage texture this rail has always built.
#[cfg(target_os = "macos")]
fn attachment_textures(
    device: &Device,
    planned: &RenderPlan<'_>,
    residents: &[Option<Texture>],
) -> Result<Vec<Texture>, ProviderError> {
    let mut textures = Vec::with_capacity(planned.attachments.len());
    for (index, attachment) in planned.attachments.iter().enumerate() {
        let texture = if attachment.resident {
            residents
                .get(index)
                .and_then(|texture| texture.clone())
                .ok_or_else(|| {
                    capability_refusal("resident_target_undeclared")
                        .with_field("attachment", FieldValue::Unsigned(index as u64))
                        .with_detail(
                            "the plan declares the provider-resident target and the caller \
                             handed over no provider image for it",
                        )
                })?
        } else {
            present_target_texture(device, attachment.format, planned.extent)?
        };
        // The upload reads the resolved window, which for a no-copy source is
        // the owner's own mapping: an owner that rewrites its pages after the
        // import changes what the preset uploads, exactly as it changes what a
        // bound stream reads (`research/docs/23` §74, R5b).
        //
        // A resident attachment's own bytes are the provider's image, and the
        // one resident shape that still uploads is a `LoadOp::Load` beside a
        // resident store: those declared previous bytes are what define the
        // image before the draw, exactly as they define a fresh per-pass
        // attachment (`research/docs/23` §76, R7).
        if let Some(source) = &attachment.initial {
            // A multisampled `Load` is seeded by the encoder's own seed pass
            // (`research/docs/23` §82, v82): those bytes define the n-sample
            // surface, not the single-sample texture this loop holds, which the
            // resolve overwrites whole — so the upload is skipped rather than
            // written into bytes the pass never reads.
            if planned.multisample.is_none() {
                upload_texels(&texture, planned, attachment.texel, source.proof_bytes());
            }
        }
        textures.push(texture);
    }
    Ok(textures)
}

/// The multisampled surfaces one multisampled pass renders into
/// (`research/docs/23` §3.3, v51/v61), one per colour location.
///
/// The textures are render targets with shared storage for the same reason
/// every attachment texture is: the rail is synchronous and the resolved
/// texels, not these, are what leaves through the readback. The sample count is
/// the one the plan carried — the pass's own 2x/4x/8x raster — so the raster
/// state and the textures cannot disagree.
#[cfg(target_os = "macos")]
fn multisample_attachment_textures(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<Vec<Texture>, ProviderError> {
    let samples = match planned.multisample {
        Some(count @ (SampleCount::Two | SampleCount::Four | SampleCount::Eight)) => {
            u64::from(count.samples())
        }
        _ => {
            return Err(
                capability_refusal("render_multisample_state_unsupported").with_detail(
                    "the multisample increment creates two-, four- and eight-sample surfaces \
                     only",
                ),
            );
        }
    };
    let mut textures = Vec::with_capacity(planned.attachments.len());
    for attachment in &planned.attachments {
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_pixel_format(metal_pixel_format(attachment.format));
        descriptor.set_width(u64::from(planned.extent[0]));
        descriptor.set_height(u64::from(planned.extent[1]));
        descriptor.set_mipmap_level_count(1);
        descriptor.set_sample_count(samples);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let pointer: *mut metal::MTLTexture =
            unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
        if pointer.is_null() {
            return Err(resource_refusal(
                "metal_render_multisample_target_allocation_failed",
            ));
        }
        textures.push(unsafe { Texture::from_ptr(pointer) });
    }
    Ok(textures)
}

/// One render target texture, built for an offscreen attachment or a present
/// target. A present target is created once and then reused across submissions,
/// which is why this creation is split from the initial upload
/// (`research/docs/24` §6 Step 7).
#[cfg(target_os = "macos")]
pub(crate) fn present_target_texture(
    device: &Device,
    format: RenderPixelFormat,
    extent: [u32; 2],
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(metal_pixel_format(format));
    descriptor.set_width(u64::from(extent[0]));
    descriptor.set_height(u64::from(extent[1]));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_target_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// One contract blend factor as the `MTLBlendFactor` it names
/// (`research/docs/23` §3.3, v40/v100).
///
/// Every arm is total over the contract's list: the two families core refuses
/// by name still have a `MTLBlendFactor` of the same meaning, and the contract
/// is where that decision lives. Metal reads none of them unless the colour
/// attachment's blend enable is set, which is the condition the refusals
/// state.
#[cfg(target_os = "macos")]
const fn metal_blend_factor(factor: BlendFactor) -> MTLBlendFactor {
    match factor {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SourceAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSourceAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::SourceColor => MTLBlendFactor::SourceColor,
        BlendFactor::OneMinusSourceColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::DestinationColor => MTLBlendFactor::DestinationColor,
        BlendFactor::OneMinusDestinationColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::DestinationAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDestinationAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::SourceAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        BlendFactor::BlendColor => MTLBlendFactor::BlendColor,
        BlendFactor::OneMinusBlendColor => MTLBlendFactor::OneMinusBlendColor,
        BlendFactor::BlendAlpha => MTLBlendFactor::BlendAlpha,
        BlendFactor::OneMinusBlendAlpha => MTLBlendFactor::OneMinusBlendAlpha,
        BlendFactor::Source1Color => MTLBlendFactor::Source1Color,
        BlendFactor::OneMinusSource1Color => MTLBlendFactor::OneMinusSource1Color,
        BlendFactor::Source1Alpha => MTLBlendFactor::Source1Alpha,
        BlendFactor::OneMinusSource1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

/// One contract blend operation as the `MTLBlendOperation` it names
/// (`research/docs/23` §3.3, v40/v100).
#[cfg(target_os = "macos")]
const fn metal_blend_operation(operation: BlendOperation) -> MTLBlendOperation {
    match operation {
        BlendOperation::Add => MTLBlendOperation::Add,
        BlendOperation::Subtract => MTLBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOperation::Min => MTLBlendOperation::Min,
        BlendOperation::Max => MTLBlendOperation::Max,
    }
}

/// One contract write mask as Metal's own `MTLColorWriteMask`
/// (`research/docs/23` §3.3, v100).
///
/// The contract's spelling is already Metal's — alpha first from the low end —
/// so this is the identity on the four channel bits; naming it keeps the two
/// rails' mappings symmetric, because Vulkan's bit order is not this one.
#[cfg(target_os = "macos")]
const fn metal_color_write_mask(mask: ColorWriteMask) -> MTLColorWriteMask {
    MTLColorWriteMask::from_bits_truncate(mask.bits() as NSUInteger)
}

/// One contract stencil comparison as the `MTLCompareFunction` it names
/// (`research/docs/23` §3.3, v47).
///
/// The two admitted values are the ones both APIs spell identically, so the
/// mapping is total over the contract's own list — the same shape the Vulkan
/// rail's `vk_stencil_compare` has.
#[cfg(target_os = "macos")]
const fn metal_stencil_compare(compare: StencilCompare) -> MTLCompareFunction {
    match compare {
        StencilCompare::Equal => MTLCompareFunction::Equal,
        StencilCompare::Always => MTLCompareFunction::Always,
    }
}

/// One contract stencil operation as the `MTLStencilOperation` it names
/// (`research/docs/23` §3.3, v47).
#[cfg(target_os = "macos")]
const fn metal_stencil_operation(operation: StencilOp) -> MTLStencilOperation {
    match operation {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
    }
}

/// The rail-owned depth texture of a pass that declares one
/// (`research/docs/23` §3.3, v36/v43).
///
/// Shared storage when the pass stores the surface, because `getBytes` reads no
/// `Private` texture and the readback is what makes the stored texels
/// observable — the same reason the colour attachments are shared
/// (`research/docs/16` §4.8). The pre-v43 shapes keep `Private`: nothing reads
/// those texels back, and the rail-owned surface disappears with the pass.
#[cfg(target_os = "macos")]
fn depth_texture(
    device: &Device,
    depth: &PlannedDepth,
    multisample: Option<SampleCount>,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    // A multisampled pass creates its depth surface with the raster's own
    // sample count (`research/docs/23` §3.3, v53/v61): Metal refuses an
    // encoder whose depth texture's sample count disagrees with
    // `rasterSampleCount`, so the two come from one decision. The surface
    // stays `Private` because a multisampled depth surface is rail-owned in
    // this increment — keeping it would need the depth resolve filter the
    // increment after this one reviews.
    if let Some(multisample) = multisample {
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_sample_count(u64::from(multisample.samples()));
    } else {
        descriptor.set_texture_type(MTLTextureType::D2);
    }
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(if depth.storing() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Private
    });
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_depth_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The multisampled depth surface a resolving pass renders into
/// (`research/docs/23` §3.3, v57c/v61).
///
/// The resolve's landing is the single-sample shared texture
/// [`depth_texture`] builds for a stored surface, so this surface is never
/// read back and stays private — the same reason the four-sample colour
/// surfaces are private in the Swift oracle. Metal refuses an encoder whose
/// depth texture's sample count disagrees with `rasterSampleCount`, so the two
/// come from one decision exactly as the non-resolving multisampled surface
/// does. The count is the raster's own, carried in from the plan.
#[cfg(target_os = "macos")]
fn multisample_depth_surface(
    device: &Device,
    depth: &PlannedDepth,
    samples: u64,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2Multisample);
    descriptor.set_sample_count(samples);
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal(
            "metal_render_multisample_depth_allocation_failed",
        ));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The rail-owned stencil texture of a pass that declares one
/// (`research/docs/23` §3.3, v47/v49).
///
/// Shared storage when the pass stores the surface, for the same reason a
/// storing depth surface is shared: `getBytes` reads no `Private` texture, and
/// the readback is what makes the stored texels observable
/// (`research/docs/16` §4.8). Every pre-v49 shape keeps `Private`: the stencil
/// fixture observes a discarded surface's effect through the colour attachment
/// the mask decides, and the rail-owned surface disappears with the pass.
#[cfg(target_os = "macos")]
fn stencil_texture(
    device: &Device,
    stencil: &PlannedStencil,
    multisample: Option<SampleCount>,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    // A multisampled pass creates its stencil surface with the raster's own
    // sample count (`research/docs/23` §3.3, v55/v61), exactly as the depth
    // surface does: Metal refuses an encoder whose attachment disagrees with
    // `rasterSampleCount`. The surface stays `Private` because a multisampled
    // stencil surface is rail-owned in this increment — keeping it would need
    // the stencil resolve the increment after this one reviews.
    if let Some(multisample) = multisample {
        descriptor.set_texture_type(MTLTextureType::D2Multisample);
        descriptor.set_sample_count(u64::from(multisample.samples()));
    } else {
        descriptor.set_texture_type(MTLTextureType::D2);
    }
    descriptor.set_pixel_format(MTLPixelFormat::Stencil8);
    descriptor.set_width(u64::from(stencil.width));
    descriptor.set_height(u64::from(stencil.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(if stencil.storing() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Private
    });
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_stencil_allocation_failed"));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// The combined depth-stencil surface of a pass that opens both faces
/// (`research/docs/23` §3.3, v60/v61).
///
/// Metal binds one texture to both attachment descriptors, so the combined
/// shape creates one multisampled `depth32Float_stencil8` surface the depth and
/// stencil halves share; the resolve landings below are separate single-sample
/// textures. The surface is private — its texels leave through the resolves,
/// never through `getBytes`.
#[cfg(target_os = "macos")]
fn combined_depth_stencil_surface(
    device: &Device,
    depth: &PlannedDepth,
    samples: u64,
) -> Result<Texture, ProviderError> {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2Multisample);
    descriptor.set_sample_count(samples);
    descriptor.set_pixel_format(MTLPixelFormat::Depth32Float_Stencil8);
    descriptor.set_width(u64::from(depth.width));
    descriptor.set_height(u64::from(depth.height));
    descriptor.set_mipmap_level_count(1);
    descriptor.set_usage(MTLTextureUsage::RenderTarget);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    let pointer: *mut metal::MTLTexture =
        unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
    if pointer.is_null() {
        return Err(resource_refusal(
            "metal_render_combined_depth_stencil_allocation_failed",
        ));
    }
    Ok(unsafe { Texture::from_ptr(pointer) })
}

/// One MTLBuffer holding a stream view's bytes, for a vertex or index binding.
///
/// The two upload arms build the image this rail built before the lease channel
/// existed: the view's bytes placed at `offset` (the view's own offset inside
/// its allocation) and bound at that same offset. For the reviewed fixture the
/// offset is zero, so the image is exactly the declared bytes; for a view that
/// starts above the allocation's first byte the stream still reads the byte
/// range the trace named instead of being silently re-based at zero.
///
/// The third arm maps the owner's reservation instead of copying anything: the
/// mapping is what starts at address zero, and the caller binds it at the
/// view's offset inside it (`PlannedInputSource::NoCopy`, `native.rs`'s
/// `ResolvedBuffer::Borrowed` bindings).
#[cfg(target_os = "macos")]
fn stream_buffer(
    device: &Device,
    source: &PlannedInputSource<'_>,
    offset: u64,
) -> Result<Buffer, ProviderError> {
    let PlannedInputSource::NoCopy { window, .. } = source else {
        return upload_stream_buffer(device, offset, source.proof_bytes());
    };
    // SAFETY: the window was resolved by the provider's no-copy registry for an
    // imported lease, whose contract keeps the owner's mapping readable at this
    // address until the provider releases the import — which this rail does not
    // do before the pass's command buffer reached a terminal status
    // (`RenderInputRetains`). A nil deallocator leaves the owner responsible for
    // its own pages.
    let pointer: *mut metal::MTLBuffer = unsafe {
        msg_send![device.as_ref(),
            newBufferWithBytesNoCopy:window.base_pointer as *mut std::ffi::c_void
            length:window.base_len
            options:MTLResourceOptions::StorageModeShared
            deallocator:std::ptr::null::<std::ffi::c_void>()]
    };
    if pointer.is_null() {
        return Err(resource_refusal(
            "metal_render_no_copy_stream_buffer_failed",
        ));
    }
    Ok(unsafe { Buffer::from_ptr(pointer) })
}

/// One MTLBuffer holding uploaded bytes at the offset the binding uses.
#[cfg(target_os = "macos")]
fn upload_stream_buffer(
    device: &Device,
    offset: u64,
    bytes: &[u8],
) -> Result<Buffer, ProviderError> {
    let start = usize::try_from(offset)
        .map_err(|_| resource_refusal("metal_render_stream_offset_overflow"))?;
    let end = start
        .checked_add(bytes.len())
        .ok_or_else(|| resource_refusal("metal_render_stream_offset_overflow"))?;
    let mut image = vec![0_u8; end];
    image[start..end].copy_from_slice(bytes);
    let pointer: *mut metal::MTLBuffer = unsafe {
        msg_send![device.as_ref(),
            newBufferWithBytes:image.as_ptr().cast::<std::ffi::c_void>()
            length:image.len()
            options:MTLResourceOptions::StorageModeShared]
    };
    if pointer.is_null() {
        return Err(resource_refusal("metal_render_stream_buffer_failed"));
    }
    Ok(unsafe { Buffer::from_ptr(pointer) })
}

/// Upload tightly packed texels into a texture's whole extent.
///
/// `replace_region` takes the source stride and owns the texture-side layout,
/// so this upload cannot repeat the Vulkan rail's defect: there the host had to
/// guess the destination row pitch, while Metal keeps that distance inside the
/// driver (`research/docs/16` §4.8). The present path uses this for the
/// sentinel preset that makes "the present never happened" falsifiable
/// (`research/docs/24` §3.1).
#[cfg(target_os = "macos")]
pub(crate) fn upload_texels(
    texture: &Texture,
    planned: &RenderPlan<'_>,
    texel: TexelExtent,
    bytes: &[u8],
) {
    texture.replace_region(
        region(planned),
        0,
        bytes.as_ptr().cast(),
        NSUInteger::try_from(texel.row_pitch).unwrap_or(NSUInteger::MAX),
    );
}

/// The two-stage pipeline state of the reviewed module.
#[cfg(target_os = "macos")]
fn render_pipeline_state(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<RenderPipelineState, ProviderError> {
    let options = CompileOptions::new();
    // The plan's own module, not the milestone's: a vertex-input pipeline is
    // built from `quad_indexed_2x2.metal`, and compiling the `vertex_id` module
    // for it would fail on the entry name rather than run the reviewed quad
    // (`research/docs/23` §3.3).
    let library = device
        .new_library_with_source(planned.source, &options)
        .map_err(|error| {
            compile_refusal("metal_render_library_compile_failed").with_detail(error)
        })?;
    let vertex = library
        .get_function(planned.vertex_entry, None)
        .map_err(|error| {
            compile_refusal("metal_render_vertex_function_missing").with_detail(error)
        })?;
    let fragment = library
        .get_function(planned.fragment_entry, None)
        .map_err(|error| {
            compile_refusal("metal_render_fragment_function_missing").with_detail(error)
        })?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(vertex.as_ref()));
    descriptor.set_fragment_function(Some(fragment.as_ref()));
    // The pipeline's raster sample count follows the pass's own multisample
    // state (`research/docs/23` §3.3, v51/v61): Metal refuses a pipeline whose
    // `rasterSampleCount` disagrees with the attachments the encoder binds, so
    // the two come from one decision. A single-sample pass keeps the default
    // exactly as every pre-v51 pass did.
    if let Some(multisample) = planned.multisample {
        descriptor.set_raster_sample_count(u64::from(multisample.samples()));
    }
    // The vertex descriptor is what makes `[[attribute(n)]]` mean a byte range
    // of a bound stream: the MSL module names the attribute locations, the
    // descriptor says which binding, stride, offset and format each one reads.
    // A `VertexLayout::None` pipeline carries none — its vertex stage takes
    // `vertex_id` and reads no stream — which is why the descriptor is built
    // only from a plan that has streams.
    if !planned.vertex_streams.is_empty() {
        let vertex_descriptor = VertexDescriptor::new();
        for stream in &planned.vertex_streams {
            let layout = vertex_descriptor
                .layouts()
                .object_at(NSUInteger::from(stream.buffer_index))
                .ok_or_else(|| {
                    resource_refusal("metal_render_vertex_layout_descriptor_unavailable")
                })?;
            layout.set_stride(NSUInteger::try_from(stream.stride).unwrap_or(NSUInteger::MAX));
            // The binding's own step function (`research/docs/23` §3.3, v31).
            // A per-instance stream advances once per instance, which is
            // Metal's default `stepRate` of one; a per-vertex stream keeps the
            // pre-v31 behavior exactly.
            layout.set_step_function(metal_vertex_step(stream.step));
            if stream.step == RenderVertexStep::PerInstance {
                layout.set_step_rate(1);
            }
            for attribute in &stream.attributes {
                let target = vertex_descriptor
                    .attributes()
                    .object_at(NSUInteger::from(attribute.location))
                    .ok_or_else(|| {
                        resource_refusal("metal_render_vertex_attribute_descriptor_unavailable")
                    })?;
                target.set_format(metal_vertex_format(attribute.format));
                target
                    .set_offset(NSUInteger::try_from(attribute.offset).unwrap_or(NSUInteger::MAX));
                target.set_buffer_index(NSUInteger::from(stream.buffer_index));
            }
        }
        descriptor.set_vertex_descriptor(Some(vertex_descriptor));
    }
    // A pass that opens a depth attachment compiles its pipeline against that
    // attachment's format; a pre-v36 pass declares none
    // (`research/docs/23` §3.3, v36).
    // The combined depth-stencil shape compiles both slots against the one
    // `depth32Float_stencil8` format the two attachment descriptors share
    // (`research/docs/23` §3.3, v60); the single-face shapes keep their own
    // formats.
    let combined = planned.depth.is_some() && planned.stencil.is_some();
    if planned.depth.is_some() {
        descriptor.set_depth_attachment_pixel_format(if combined {
            MTLPixelFormat::Depth32Float_Stencil8
        } else {
            MTLPixelFormat::Depth32Float
        });
    }
    // The stencil slot is the same rule one surface over
    // (`research/docs/23` §3.3, v47): a pass that opens a stencil attachment
    // compiles its pipeline against that attachment's format, and a pass
    // without one leaves the slot at its default exactly as before.
    if planned.stencil.is_some() {
        descriptor.set_stencil_attachment_pixel_format(if combined {
            MTLPixelFormat::Depth32Float_Stencil8
        } else {
            MTLPixelFormat::Stencil8
        });
    }
    // One pipeline attachment per colour location: entry `i` states the pixel
    // format the reviewed fragment's output `i` is compiled against, which the
    // plan already forced to agree with the pass's attachment list
    // (`research/docs/23` §3.3). A pass that states blend state states it here,
    // per location, exactly as Metal's colour attachment descriptor indexes it
    // (`research/docs/23` §3.3, v40).
    for (index, attachment) in planned.attachments.iter().enumerate() {
        let color = descriptor
            .color_attachments()
            .object_at(index as u64)
            .ok_or_else(|| resource_refusal("metal_render_pipeline_attachment_unavailable"))?;
        color.set_pixel_format(metal_pixel_format(attachment.format));
        if let Some(blend) = planned
            .blend
            .as_ref()
            .and_then(|blend| blend.attachments.get(index))
        {
            // The entry is Metal's `MTLRenderPipelineColorAttachmentDescriptor`
            // (`research/docs/23` §3.3, v40/v100): whether this attachment
            // blends at all, the two operations (Metal spells them
            // `rgbBlendOperation` and `alphaBlendOperation`), the four factors,
            // and the write mask, which Metal applies after blending and reads
            // whether or not the entry blends.
            color.set_blending_enabled(blend.enabled);
            color.set_rgb_blend_operation(metal_blend_operation(blend.operation));
            color.set_alpha_blend_operation(metal_blend_operation(blend.alpha_operation));
            color.set_source_rgb_blend_factor(metal_blend_factor(blend.source_rgb));
            color.set_destination_rgb_blend_factor(metal_blend_factor(blend.destination_rgb));
            color.set_source_alpha_blend_factor(metal_blend_factor(blend.source_alpha));
            color.set_destination_alpha_blend_factor(metal_blend_factor(blend.destination_alpha));
            color.set_write_mask(metal_color_write_mask(blend.write_mask));
        }
    }
    device
        .new_render_pipeline_state(descriptor.as_ref())
        .map_err(|error| compile_refusal("metal_render_pipeline_compile_failed").with_detail(error))
}

/// Read one texture's texels back into host memory, tightly packed.
///
/// `texel` is the read source's own format width (`research/docs/23` §78): a
/// colour attachment carries its [`PlannedAttachment::texel`], the depth surface
/// and its resolve target carry the plan's `depth32float` extent, and the
/// region is the pass's extent in both cases — the contract holds every
/// surface to the colour attachments' extent.
#[cfg(target_os = "macos")]
fn read_texels(
    texture: &Texture,
    planned: &RenderPlan<'_>,
    texel: TexelExtent,
) -> Result<Vec<u8>, ProviderError> {
    let mut texels = vec![0_u8; texel.bytes];
    texture.get_bytes(
        texels.as_mut_ptr().cast(),
        NSUInteger::try_from(texel.row_pitch).unwrap_or(NSUInteger::MAX),
        region(planned),
        0,
    );
    Ok(texels)
}

/// Read the stored stencil surface back into host memory, tightly packed.
///
/// `Stencil8` carries one byte per texel — `metal-api-core`'s
/// `STENCIL_BYTES_PER_TEXEL` — so this surface's flat byte extent is the texel
/// extent itself rather than the four-byte-per-texel arithmetic the colour and
/// depth readbacks share, and one texture row is `width` bytes wide
/// (`research/docs/23` §3.3, v49). The region is the pass's own extent, which
/// the contract holds to the stencil surface's exactly as it holds it to every
/// colour attachment's.
#[cfg(target_os = "macos")]
fn read_stencil_texels(
    texture: &Texture,
    planned: &RenderPlan<'_>,
) -> Result<Vec<u8>, ProviderError> {
    let texel_bytes = u64::from(planned.extent[0]).saturating_mul(u64::from(planned.extent[1]));
    let Ok(bytes) = usize::try_from(texel_bytes) else {
        return Err(capability_refusal("attachment_dimension_limit"));
    };
    let mut texels = vec![0_u8; bytes];
    texture.get_bytes(
        texels.as_mut_ptr().cast(),
        NSUInteger::try_from(u64::from(planned.extent[0])).unwrap_or(NSUInteger::MAX),
        region(planned),
        0,
    );
    Ok(texels)
}

#[cfg(target_os = "macos")]
/// The sampled textures one plan carries, as the shared-storage `MTLTexture`s
/// the encoder binds (`research/docs/23` §3.3, v70).
///
/// Each texture is created with `ShaderRead` usage — the sampling stage is the
/// only reader — and filled with the plan's own bytes through
/// `replaceRegion`, one tightly packed `width * 4`-byte row per texel row, which
/// is the same upload shape the rail's attachment presets use. A no-copy
/// texture's upload reads the owner's own mapping, so an owner that rewrites
/// its pages after the import changes the texels the pass samples
/// (`research/docs/23` §75, R5c).
#[cfg(target_os = "macos")]
fn sampled_textures(
    device: &Device,
    planned: &RenderPlan<'_>,
) -> Result<Vec<Texture>, ProviderError> {
    let mut textures = Vec::with_capacity(planned.textures.len());
    for sampled in &planned.textures {
        let [width, height] = sampled.extent;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(metal_pixel_format(RenderPixelFormat::Rgba8Unorm));
        descriptor.set_width(u64::from(width));
        descriptor.set_height(u64::from(height));
        descriptor.set_mipmap_level_count(1);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        let pointer: *mut metal::MTLTexture =
            unsafe { msg_send![device.as_ref(), newTextureWithDescriptor: descriptor.as_ref()] };
        if pointer.is_null() {
            return Err(resource_refusal("metal_render_texture_allocation_failed"));
        }
        let texture = unsafe { Texture::from_ptr(pointer) };
        texture.replace_region(
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: u64::from(width),
                    height: u64::from(height),
                    depth: 1,
                },
            },
            0,
            sampled.source.proof_bytes().as_ptr().cast(),
            NSUInteger::try_from(u64::from(width) * 4).unwrap_or(NSUInteger::MAX),
        );
        textures.push(texture);
    }
    Ok(textures)
}

#[cfg(target_os = "macos")]
fn region(planned: &RenderPlan<'_>) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: u64::from(planned.extent[0]),
            height: u64::from(planned.extent[1]),
            depth: 1,
        },
    }
}

#[cfg(target_os = "macos")]
const fn metal_pixel_format(format: RenderPixelFormat) -> MTLPixelFormat {
    match format {
        RenderPixelFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        RenderPixelFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        RenderPixelFormat::R32Float => MTLPixelFormat::R32Float,
        RenderPixelFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
    }
}

/// The `MTLVertexFormat` one planned attribute format names.
#[cfg(target_os = "macos")]
const fn metal_vertex_format(format: RenderVertexFormat) -> MTLVertexFormat {
    match format {
        RenderVertexFormat::Float2 => MTLVertexFormat::Float2,
        RenderVertexFormat::Float3 => MTLVertexFormat::Float3,
        RenderVertexFormat::Float4 => MTLVertexFormat::Float4,
        // The contract's `uint32` is Metal's scalar `UInt`, not `UInt2`: the
        // attribute is one 32-bit unsigned integer, and `UInt` is the format
        // whose width matches `VertexFormat::Uint32::bytes()`.
        RenderVertexFormat::Uint => MTLVertexFormat::UInt,
        // The four normalized storages, by their `MTLVertexFormat` names
        // (`research/docs/23` §103): Metal normalizes each stored integer by
        // its own maximum, which is the conversion the contract states, so the
        // bytes Metal fetches are the bytes the fixture stored.
        RenderVertexFormat::UChar2Normalized => MTLVertexFormat::UChar2Normalized,
        RenderVertexFormat::UChar4Normalized => MTLVertexFormat::UChar4Normalized,
        RenderVertexFormat::UShort2Normalized => MTLVertexFormat::UShort2Normalized,
        RenderVertexFormat::UShort4Normalized => MTLVertexFormat::UShort4Normalized,
        // The scalar lane is Metal's own `Float` — one 32-bit component, no
        // normalization (2026-09-20, census v46's `vertex_format` bucket), the
        // width `VertexFormat::Float32x1::bytes()` states.
        RenderVertexFormat::Float => MTLVertexFormat::Float,
    }
}

/// The `MTLVertexStepFunction` one planned step names
/// (`research/docs/23` §3.3, v31).
#[cfg(target_os = "macos")]
const fn metal_vertex_step(step: RenderVertexStep) -> MTLVertexStepFunction {
    match step {
        RenderVertexStep::PerVertex => MTLVertexStepFunction::PerVertex,
        RenderVertexStep::PerInstance => MTLVertexStepFunction::PerInstance,
    }
}

/// The `MTLIndexType` one planned index width names.
#[cfg(target_os = "macos")]
const fn metal_index_type(format: RenderIndexType) -> MTLIndexType {
    match format {
        RenderIndexType::Uint16 => MTLIndexType::UInt16,
        RenderIndexType::Uint32 => MTLIndexType::UInt32,
    }
}

#[cfg(target_os = "macos")]
fn resource_refusal(slug: &'static str) -> ProviderError {
    refusal(ProviderPhase::Resolve, ProviderErrorClass::Resource, slug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_api_core::provider::{
        AcquirePolicy, AliasMode, AllocationId, AllocationRecord, AttachmentLandingView,
        BlendAttachment, BlendFactor, BlendOperation, BorrowedLease, BufferAccess,
        BufferBindingContract, BufferLease, BufferSource, ColorWriteMask, CompareFunction,
        CompiledComputePipeline, CompletionPolicy, ComputePass, DepthFormat, DepthLoadOp,
        DepthStoreOp, DeviceEpoch, Dispatch, DispatchKind, DispatchType, FootprintProof,
        FunctionIdentity, FunctionSource, IndirectCommandBufferDescriptor, IndirectCommandKind,
        IndirectCommandPayload, IndirectCommandRange, InitialState, KeptFrame, KeptFrameLanding,
        LeaseId, LeaseRegistry, LeaseReservation, MultisampleDepthResolve, MultisampleState,
        MultisampleStencilResolve, OperationId, PipelineContract, PresentTarget,
        ProviderCapabilities, RenderAttachment, RenderDepthAttachment, RenderDepthIdentity,
        RenderPassBlend, RenderPipelineStage, RenderStencilAttachment, RenderStencilIdentity,
        ResourceTableSnapshot, SamplerAddressMode, SamplerFilter, SamplerPolicy, SemanticDigest,
        StageBufferBinding, StageBufferView, StagedLease, StencilCompare, StencilFormat,
        StencilLoadOp, StencilOp, StencilResolveFilter, StencilTest, StorageMode, TextureAccess,
        VertexAttribute, VertexBufferLayout, VertexLayout, ViewId, PROVIDER_SCHEMA_VERSION,
    };

    /// The texels the reviewed fragment writes, as `MTLClearColor` components.
    const EXPECTED_TEXEL_COMPONENTS: [f64; 4] = [64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0];

    /// The milestone's pass: one 2x2 `Rgba8Unorm` attachment, stored, and drawn
    /// as the full-screen triangle.
    fn milestone_pass(load: LoadOp) -> RenderPassDescriptor {
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
            pipeline: PipelineId::new(3),
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(7),
                allocation_id: AllocationId::new(9),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load,
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            scissor: None,
            vertices: 3,
            vertex_buffers: Vec::new(),
            indices: None,
            instance_count: 1,
            textures: Vec::new(),
            present: None,
        }
    }

    /// The three positions the reviewed stage-buffer fixture reads, as the
    /// vertex stage's own `[[buffer(0)]]` bytes: (-1,1), (0.25,1), (-1,-0.25)
    /// in Metal's clip space, little-endian `float32x2`.
    fn stage_buffer_positions() -> Vec<u8> {
        let positions: [[f32; 2]; 3] = [[-1.0, 1.0], [0.25, 1.0], [-1.0, -0.25]];
        positions
            .iter()
            .flatten()
            .flat_map(|component| component.to_le_bytes())
            .collect()
    }

    /// The fragment stage's own `[[buffer(0)]]` bytes: one `float4` holding the
    /// same texel the reviewed solids store (`40 80 c0 ff` after the UNORM
    /// quantisation, byte/255 on purpose — `research/docs/23` §3.5).
    fn stage_buffer_tint() -> Vec<u8> {
        EXPECTED_TEXEL_COMPONENTS
            .iter()
            .flat_map(|component| (*component as f32).to_le_bytes())
            .collect()
    }

    /// The reviewed stage-buffer pair as a registration (`research/docs/23`
    /// §83, R9g): both `[[buffer(0)]]` arguments declared, each with the static
    /// extent the module's own read reaches (24 bytes of positions, 16 of
    /// tint).
    fn stage_buffer_declarations() -> Vec<StageBufferBinding> {
        vec![
            StageBufferBinding {
                stage: RenderPipelineStage::Vertex,
                index: STAGE_BUFFER_VERTEX_BINDING,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static {
                    max_bytes: STAGE_BUFFER_VERTEX_BYTES,
                },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_FRAGMENT_BINDING,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static {
                    max_bytes: STAGE_BUFFER_FRAGMENT_BYTES,
                },
            },
        ]
    }

    fn stage_buffer_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: stage_buffer_declarations(),
            vertex_entry: STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
            fragment_entry: STAGE_BUFFER_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        }
    }

    /// The milestone's 2x2 attachment with the stage-buffer pair's two views
    /// bound, the way a trace hands them over.
    fn stage_buffer_pass(load: LoadOp) -> RenderPassDescriptor {
        let mut pass = milestone_pass(load);
        pass.stage_buffers = vec![
            StageBufferView {
                stage: RenderPipelineStage::Vertex,
                view: BufferView {
                    view_id: ViewId::new(61),
                    metal_binding: STAGE_BUFFER_VERTEX_BINDING,
                    allocation_id: AllocationId::new(63),
                    offset: 0,
                    length: STAGE_BUFFER_VERTEX_BYTES,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(stage_buffer_positions()),
                },
            },
            StageBufferView {
                stage: RenderPipelineStage::Fragment,
                view: BufferView {
                    view_id: ViewId::new(62),
                    metal_binding: STAGE_BUFFER_FRAGMENT_BINDING,
                    allocation_id: AllocationId::new(64),
                    offset: 0,
                    length: STAGE_BUFFER_FRAGMENT_BYTES,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(stage_buffer_tint()),
                },
            },
        ];
        pass
    }

    fn stage_buffer_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_STAGE_BUFFER_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        }
    }

    /// The trace-table entry the reviewed stage-buffer pair's registration hands
    /// back: the milestone's own pipeline id, so the pass this fixture states
    /// names it, the reviewed module's vertex entry and the stage-buffer
    /// contract itself (`research/docs/23` §83, R9g).
    fn stage_buffer_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(3),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-stage-buffer-fixture", vec![61])
                    .unwrap(),
                entry_name: STAGE_BUFFER_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(stage_buffer_pipeline()),
        }
    }

    /// The reviewed stage-buffer trace and the resource namespace it needs
    /// (`research/docs/23` §83, R9g): the milestone's declaration pass, then the
    /// 2x2 render pass whose two `[[buffer(0)]]` slots the case's own views
    /// fill. This is the shape core admission's third render gate walks, so the
    /// capability snapshot's stage-buffer pair is read against it.
    fn stage_buffer_trace(load: LoadOp) -> (ComputeTrace, ResourceTableSnapshot) {
        let (mut trace, mut resources) = milestone_trace(load);
        trace.pipelines[1] = stage_buffer_table_entry();
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the milestone trace ends with its render pass");
        };
        *pass = stage_buffer_pass(load);
        for (allocation, size) in [
            (AllocationId::new(63), STAGE_BUFFER_VERTEX_BYTES),
            (AllocationId::new(64), STAGE_BUFFER_FRAGMENT_BYTES),
        ] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(3),
                    size,
                })
                .unwrap();
        }
        (trace, resources)
    }

    /// One plan's stage-buffer binding table, as the evidence line spells it
    /// (`research/docs/23` §83, R9g): the slot each binding fills in its own
    /// stage's `[[buffer(N)]]` namespace, the offset the encoder binds it at,
    /// the bytes the reviewed module reads there and which of the three source
    /// arms the bytes came from.
    fn stage_buffer_table(plan: &RenderPlan<'_>) -> String {
        plan.stage_buffers
            .iter()
            .map(|stage| {
                let source = match &stage.source {
                    PlannedInputSource::Declared(bytes) => {
                        format!("declared({} bytes)", bytes.len())
                    }
                    PlannedInputSource::Staged(bytes) => {
                        format!("staged_lease({} bytes)", bytes.len())
                    }
                    PlannedInputSource::NoCopy { lease, window } => format!(
                        "borrowed_no_copy(lease={} window={} len={})",
                        lease.get(),
                        window.offset,
                        window.len,
                    ),
                };
                format!(
                    "{} [[buffer({})]] offset={} bytes={} source={}",
                    stage.stage.name(),
                    stage.index,
                    stage.binding_offset(),
                    stage.bytes,
                    source,
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// The reviewed stage-buffer pair executes end to end on the planning half
    /// (`research/docs/23` §83, R9g): the registration pairs both declarations
    /// with the module's own `[[buffer(0)]]` arguments, and the plan resolves
    /// both views into a binding table naming each stage's own slot. The
    /// encoder half is behind `cfg(target_os = "macos")`, so the table is what
    /// this host can read — the same split every other render fixture uses.
    #[test]
    fn the_stage_buffer_pair_is_reviewed_and_plans_its_bindings() {
        let contract = stage_buffer_pipeline();
        let selected = reviewed_module_for(&contract).expect("the stage-buffer pair is reviewed");
        assert_eq!(selected.source, REVIEWED_STAGE_BUFFER_SOURCE);
        assert_eq!(
            selected.path,
            "conformance/shaders/render_stage_buffer_2x2.metal"
        );
        review_contract(&contract).expect("the registration pairs with the module's arguments");

        let pass = stage_buffer_pass(LoadOp::Clear(sentinel()));
        let request = stage_buffer_request(&pass, &contract);
        let planned = plan_pass(&request).expect("the reviewed pair plans its two bindings");
        let [vertex, fragment] = planned.stage_buffers.as_slice() else {
            panic!("the reviewed module reads one argument per stage");
        };
        assert_eq!(vertex.stage, RenderPipelineStage::Vertex);
        assert_eq!(vertex.index, STAGE_BUFFER_VERTEX_BINDING);
        assert_eq!(vertex.bytes, STAGE_BUFFER_VERTEX_BYTES);
        assert_eq!(vertex.binding_offset(), 0);
        assert_eq!(vertex.source.proof_bytes(), stage_buffer_positions());
        assert_eq!(fragment.stage, RenderPipelineStage::Fragment);
        assert_eq!(fragment.index, STAGE_BUFFER_FRAGMENT_BINDING);
        assert_eq!(fragment.bytes, STAGE_BUFFER_FRAGMENT_BYTES);
        assert_eq!(fragment.source.proof_bytes(), stage_buffer_tint());
        // Both halves declare their own bytes, so the plan maps nothing.
        assert!(planned.borrowed_leases().is_empty());
        eprintln!(
            "native stage-buffer plan bindings: {}",
            stage_buffer_table(&planned)
        );
    }

    /// Every disagreement between a declaration and the reviewed module's own
    /// argument list is refused by name (`research/docs/23` §83, R9g), in the
    /// vocabulary the Vulkan rails established rather than a new slug — and the
    /// capability snapshot keeps the bit off until an Apple device run lands.
    #[test]
    fn the_stage_buffer_registration_refuses_every_disagreement_by_name() {
        // A declaration at a slot the module never reads: the binding would be
        // silently dropped, which is the arm the Vulkan rail refuses under the
        // same slug.
        let mut declared = stage_buffer_pipeline();
        declared.stage_buffers.push(StageBufferBinding {
            stage: RenderPipelineStage::Fragment,
            index: 4,
            access: BufferAccess::Read,
            footprint: FootprintProof::Static { max_bytes: 16 },
        });
        let refused = review_contract(&declared).expect_err("a slot the module never reads");
        eprintln!("native stage-buffer slot the module never reads refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_buffer_stage_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("stage"),
            Some(&FieldValue::Text("fragment".to_owned()))
        );
        assert_eq!(
            refused.fields.get("binding"),
            Some(&FieldValue::Unsigned(4))
        );

        // The declaration half of the pairing: an extent under the module's own
        // read would let the shader read past the bytes the pass proved.
        let mut short = stage_buffer_pipeline();
        short.stage_buffers[1].footprint = FootprintProof::Static { max_bytes: 8 };
        let refused = review_contract(&short).expect_err("an extent under the module's read");
        eprintln!("native stage-buffer short extent refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_bytes"),
            Some(&FieldValue::Unsigned(8))
        );
        assert_eq!(
            refused.fields.get("reflected_bytes"),
            Some(&FieldValue::Unsigned(STAGE_BUFFER_FRAGMENT_BYTES))
        );

        // The whole-binding arm is the Vulkan rail's (`research/docs/23` §3.3,
        // E-SB3): the reviewed pair reads each `[[buffer(N)]]` argument at the
        // extent its own pinned source states, so a declaration that states no
        // reach at all has no route here — refused by its own name and slug
        // rather than executed against a window the module's source does not
        // carry, and the snapshot keeps the bit closed so a consumer never
        // hands this rail the shape.
        let mut whole_binding = stage_buffer_pipeline();
        whole_binding.stage_buffers[1].footprint = FootprintProof::BindingRange;
        let refused = review_contract(&whole_binding)
            .expect_err("the whole-binding arm has no route on this rail");
        eprintln!("native whole-binding arm refused: {refused:?}");
        assert_eq!(
            refused.slug,
            "render_stage_buffer_binding_range_unsupported"
        );
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("stage"),
            Some(&FieldValue::Text("fragment".to_owned()))
        );
        assert_eq!(
            refused.fields.get("index"),
            Some(&FieldValue::Unsigned(u64::from(
                STAGE_BUFFER_FRAGMENT_BINDING
            )))
        );

        // The other direction: naming the module but declaring none of its
        // arguments leaves descriptors the stages read undefined.
        let mut undeclared = stage_buffer_pipeline();
        undeclared.stage_buffers.clear();
        let refused =
            review_contract(&undeclared).expect_err("the module's arguments must be declared");
        eprintln!("native stage-buffer binding required refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_buffer_binding_required");
        assert_eq!(
            refused.fields.get("stage"),
            Some(&FieldValue::Text("vertex".to_owned()))
        );

        // The bufferless modules keep the pre-R9g answer for a declaration they
        // cannot read: the milestone's own pair has no `[[buffer(N)]]`
        // argument, so a registration that declares one is refused rather than
        // executed with the binding dropped.
        let declared = RenderPipelineContract {
            stage_buffers: vec![StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: 0,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static { max_bytes: 16 },
            }],
            textures: Vec::new(),
            ..milestone_pipeline()
        };
        let refused =
            review_contract(&declared).expect_err("a declaration under a bufferless module");
        eprintln!("native bufferless declaration refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_buffer_stage_unsupported");

        // The pass half stays the contract's own pair rule: a view the
        // registration never declared is refused before this rail is asked —
        // the same `trace_contract_invalid` the pre-R9g test pinned.
        let milestone = milestone_pipeline();
        let mut pass = stage_buffer_pass(LoadOp::Clear(sentinel()));
        pass.stage_buffers.truncate(1);
        let request = milestone_request(&pass, &milestone, None);
        let refused = plan_pass(&request).expect_err("a pass that binds an undeclared slot");
        eprintln!("native undeclared stage-buffer binding refused: {refused:?}");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);

        // The snapshot and the rail now declare the same window: the pair of
        // declarations above is what the Apple device readings
        // (`stage_buffer_selftest` / `stage_buffer_write_selftest`) flipped, so
        // the bits are the production spelling's own and core admission admits
        // the shape instead of refusing it by name. The pre-flip refusal stays
        // pinned on a constructed snapshot
        // (`the_pre_flip_snapshot_refuses_a_stage_buffer_trace`).
        let bits = capability_bits(2048);
        assert!(!bits.supports_render_passes || bits.max_color_attachments > 0);
        let capabilities = capabilities(&bits);
        assert!(capabilities.supports_render_stage_buffers);
        assert_eq!(
            capabilities.max_render_stage_buffers,
            MAX_RENDER_STAGE_BUFFERS
        );
        // The folded shape's bit is the same face's (`research/docs/23` §3.3,
        // E-TX9): the reviewed pair already binds its two stages at set 1 and
        // set 2, so the snapshot declares the arrangement those Apple device
        // readings measured.
        assert!(capabilities.supports_render_stage_buffer_namespace_split);
        assert!(capabilities.declares_render_stage_buffer_namespace_split());
        assert!(
            !capabilities_before_the_stage_buffer_flip(&capability_bits(APPLE_2D_TEXTURE_CEILING))
                .supports_render_stage_buffer_namespace_split,
            "the pre-flip declaration keeps the shape bit closed"
        );
        // The whole-binding arm stays closed on this rail (`research/docs/23`
        // §3.3, E-SB3), and its predicate answers the same thing: the reviewed
        // pair has no route that executes a declaration nothing measured.
        assert!(!capabilities.supports_render_stage_buffer_binding_range);
        assert!(!capabilities.declares_render_stage_buffer_binding_range());
    }

    /// A stage buffer resolves through the same three-armed source channel a
    /// vertex stream does (`research/docs/23` §83, R9g): a staged lease becomes
    /// the provider's own copy and retains nothing, because only the no-copy
    /// arm has an owner mapping to hold.
    #[test]
    fn a_stage_buffer_reads_its_staged_lease() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(41);
        let reservation = lease_registration(
            lease_id,
            AllocationId::new(63),
            STAGE_BUFFER_VERTEX_BYTES,
            epoch,
        );
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(reservation, stage_buffer_positions())
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(63),
                owner_epoch: epoch,
                size: STAGE_BUFFER_VERTEX_BYTES,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the reservation covers its view");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };

        let mut pass = stage_buffer_pass(LoadOp::Clear(sentinel()));
        pass.stage_buffers[0].view.source = BufferSource::StagedLease(lease_id);
        let contract = stage_buffer_pipeline();
        let request = stage_buffer_request(&pass, &contract);
        let planned = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the staged window holds the reviewed positions");
        assert!(
            matches!(
                planned.stage_buffers[0].source,
                PlannedInputSource::Staged(_)
            ),
            "a staged lease resolves into the provider's own copy: {:?}",
            planned.stage_buffers[0].source
        );
        assert_eq!(
            planned.stage_buffers[0].source.proof_bytes(),
            stage_buffer_positions()
        );
        assert!(
            planned.borrowed_leases().is_empty(),
            "a staged arm has no owner mapping to retain"
        );
        eprintln!(
            "native stage-buffer staged-lease plan bindings: {}",
            stage_buffer_table(&planned)
        );

        // The staged copy is the provider's: releasing it is the owner's
        // decision, and the same declaration is refused by name until it is
        // imported again — under the stage buffer's own source slug.
        staging
            .release(lease_id)
            .expect("the fixture import is released");
        let refused = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released staged lease cannot be read");
        eprintln!("native released stage-buffer lease refused: {refused:?}");
        assert_eq!(refused.slug, "lease_not_imported");
    }

    // ---------------------------------------------------------------------
    // The writable stage-buffer half (`research/docs/23` §92, R9k)
    // ---------------------------------------------------------------------

    /// The writable fixture's view and allocation identities, one pair per
    /// binding: the strided positions, the readable `source`, the write-only
    /// `sink` and the read-write `accumulator`.
    const WRITE_STAGE_BUFFER_POSITIONS_VIEW: ViewId = ViewId::new(71);
    const WRITE_STAGE_BUFFER_POSITIONS_ALLOCATION: AllocationId = AllocationId::new(73);
    const WRITE_STAGE_BUFFER_SOURCE_VIEW: ViewId = ViewId::new(74);
    const WRITE_STAGE_BUFFER_SOURCE_ALLOCATION: AllocationId = AllocationId::new(76);
    const WRITE_STAGE_BUFFER_SINK_VIEW: ViewId = ViewId::new(77);
    const WRITE_STAGE_BUFFER_SINK_ALLOCATION: AllocationId = AllocationId::new(79);
    const WRITE_STAGE_BUFFER_ACCUMULATOR_VIEW: ViewId = ViewId::new(81);
    const WRITE_STAGE_BUFFER_ACCUMULATOR_ALLOCATION: AllocationId = AllocationId::new(83);

    /// The readable `source` payload: the same `float4` the R9g pair binds, so
    /// the attachment and the sink carry the texel the R9f write half carries.
    fn write_stage_buffer_source() -> Vec<u8> {
        stage_buffer_tint()
    }

    /// The read-write `accumulator`'s previous bytes: `0.25` in every
    /// component. The stage adds one to whatever it holds, so a rail that bound
    /// zeros would publish `1.0` where this publishes `1.25` — the read half of
    /// the access is what the difference observes.
    fn write_stage_buffer_accumulator() -> Vec<u8> {
        [0.25_f32; 4]
            .iter()
            .flat_map(|component| component.to_le_bytes())
            .collect()
    }

    /// What the accumulator holds after the pass: one added to every component
    /// of its previous bytes, as the fixture's own `f32` arithmetic states it.
    fn write_stage_buffer_accumulator_after() -> Vec<u8> {
        [1.25_f32; 4]
            .iter()
            .flat_map(|component| component.to_le_bytes())
            .collect()
    }

    /// The writable fixture's declarations (`research/docs/23` §92, R9k): the
    /// vertex positions as the reflected affine pair of accesses, then the
    /// fragment stage's read, write and read-write slots, each in canonical
    /// order (vertex bindings first, ascending inside each stage).
    fn write_stage_buffer_declarations() -> Vec<StageBufferBinding> {
        let access = |base_offset| AffineAccess {
            base_offset,
            access_size: 4,
            terms: vec![AffineTerm {
                axis: 0,
                stride: STAGE_BUFFER_WRITE_VERTEX_STRIDE,
            }],
        };
        vec![
            StageBufferBinding {
                stage: RenderPipelineStage::Vertex,
                index: STAGE_BUFFER_WRITE_VERTEX_BINDING,
                access: BufferAccess::Read,
                footprint: FootprintProof::Affine {
                    accesses: vec![access(0), access(4)],
                },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_SOURCE_BINDING,
                access: BufferAccess::Read,
                footprint: FootprintProof::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_SINK_BINDING,
                access: BufferAccess::Write,
                footprint: FootprintProof::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
            StageBufferBinding {
                stage: RenderPipelineStage::Fragment,
                index: STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING,
                access: BufferAccess::ReadWrite,
                footprint: FootprintProof::Static {
                    max_bytes: STAGE_BUFFER_WRITE_TEXEL_BYTES,
                },
            },
        ]
    }

    fn write_stage_buffer_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: write_stage_buffer_declarations(),
            vertex_entry: STAGE_BUFFER_WRITE_VERTEX_ENTRY.to_owned(),
            fragment_entry: STAGE_BUFFER_WRITE_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        }
    }

    /// One stage buffer view of the writable fixture, at the access its slot
    /// declares and with the bytes the case states.
    fn write_stage_buffer_view(
        stage: RenderPipelineStage,
        index: u32,
        view_id: ViewId,
        allocation_id: AllocationId,
        access: BufferAccess,
        bytes: Vec<u8>,
    ) -> StageBufferView {
        StageBufferView {
            stage,
            view: BufferView {
                view_id,
                metal_binding: index,
                allocation_id,
                offset: 0,
                length: u64::try_from(bytes.len()).expect("fixture length"),
                access,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(bytes),
            },
        }
    }

    /// The writable fixture's pass: the milestone's 2x2 attachment with the
    /// four stage buffer views bound, in canonical order.
    fn write_stage_buffer_pass(load: LoadOp, positions: Vec<u8>) -> RenderPassDescriptor {
        let mut pass = milestone_pass(load);
        pass.stage_buffers = vec![
            write_stage_buffer_view(
                RenderPipelineStage::Vertex,
                STAGE_BUFFER_WRITE_VERTEX_BINDING,
                WRITE_STAGE_BUFFER_POSITIONS_VIEW,
                WRITE_STAGE_BUFFER_POSITIONS_ALLOCATION,
                BufferAccess::Read,
                positions,
            ),
            write_stage_buffer_view(
                RenderPipelineStage::Fragment,
                STAGE_BUFFER_WRITE_SOURCE_BINDING,
                WRITE_STAGE_BUFFER_SOURCE_VIEW,
                WRITE_STAGE_BUFFER_SOURCE_ALLOCATION,
                BufferAccess::Read,
                write_stage_buffer_source(),
            ),
            write_stage_buffer_view(
                RenderPipelineStage::Fragment,
                STAGE_BUFFER_WRITE_SINK_BINDING,
                WRITE_STAGE_BUFFER_SINK_VIEW,
                WRITE_STAGE_BUFFER_SINK_ALLOCATION,
                BufferAccess::Write,
                // The sink starts as zeros, so a rail that executed the pass
                // but landed nothing would publish these bytes.
                vec![0; STAGE_BUFFER_WRITE_TEXEL_BYTES as usize],
            ),
            write_stage_buffer_view(
                RenderPipelineStage::Fragment,
                STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING,
                WRITE_STAGE_BUFFER_ACCUMULATOR_VIEW,
                WRITE_STAGE_BUFFER_ACCUMULATOR_ALLOCATION,
                BufferAccess::ReadWrite,
                write_stage_buffer_accumulator(),
            ),
        ];
        pass
    }

    fn write_stage_buffer_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_STAGE_BUFFER_WRITE_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        }
    }

    /// One contract access as the evidence line spells it.
    fn access_name(access: BufferAccess) -> &'static str {
        match access {
            BufferAccess::Read => "read",
            BufferAccess::Write => "write",
            BufferAccess::ReadWrite => "read_write",
            BufferAccess::Unused => "unused",
        }
    }

    /// One plan's writable binding table, as the evidence line spells it
    /// (`research/docs/23` §92, R9k): the slot, the offset the encoder binds,
    /// the bytes the module's own reach covers over this draw, the access, the
    /// view the binding lands in and which of the three source arms the bytes
    /// came from.
    fn write_stage_buffer_table(plan: &RenderPlan<'_>) -> String {
        plan.stage_buffers
            .iter()
            .map(|stage| {
                let source = match &stage.source {
                    PlannedInputSource::Declared(bytes) => {
                        format!("declared({} bytes)", bytes.len())
                    }
                    PlannedInputSource::Staged(bytes) => {
                        format!("staged_lease({} bytes)", bytes.len())
                    }
                    PlannedInputSource::NoCopy { lease, window } => format!(
                        "borrowed_no_copy(lease={} window={} len={})",
                        lease.get(),
                        window.offset,
                        window.len,
                    ),
                };
                format!(
                    "{} [[buffer({})]] offset={} bytes={} access={} view={}/{} len={} source={}",
                    stage.stage.name(),
                    stage.index,
                    stage.binding_offset(),
                    stage.bytes,
                    access_name(stage.access),
                    stage.view_id.get(),
                    stage.allocation_id.get(),
                    stage.length,
                    source,
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// The index bytes the affine half's indexed runs draw through:
    /// `[0, 1, 2]` as little-endian `uint16`, the same triple the core
    /// contract's own affine unit test states.
    fn write_stage_buffer_indices() -> Vec<u8> {
        [0_u16, 1, 2]
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect()
    }

    /// The writable fixture, end to end on the planning half (`research/docs/23`
    /// §92, R9k): the registration pairs all four declarations with the
    /// reviewed module's own `[[buffer(N)]]` arguments — the affine vertex slot
    /// against the reflected access pair, and the fragment stage's read, write
    /// and read-write slots against their own accesses — and the plan resolves
    /// every view into a binding table naming its slot, its access and the
    /// extent the module's reach covers over the draw.
    #[test]
    fn the_writable_stage_buffer_pair_is_reviewed_and_plans_its_bindings() {
        let contract = write_stage_buffer_pipeline();
        let selected = reviewed_module_for(&contract).expect("the writable pair is reviewed");
        assert_eq!(selected.source, REVIEWED_STAGE_BUFFER_WRITE_SOURCE);
        assert_eq!(
            selected.path,
            "conformance/shaders/render_stage_buffer_write_2x2.metal"
        );
        review_contract(&contract).expect("the registration pairs with the module's arguments");

        let positions = stage_buffer_positions();
        let pass = write_stage_buffer_pass(LoadOp::Clear(sentinel()), positions.clone());
        // A non-indexed three-vertex draw: the affine reach's highest access
        // ends at `4 + 4 + (3 - 1) * 8 = 24` bytes, the payload's own length.
        let request = write_stage_buffer_request(&pass, &contract);
        let planned = plan_pass(&request).expect("the reviewed writable pair plans its bindings");
        let [vertex, source, sink, accumulator] = planned.stage_buffers.as_slice() else {
            panic!("the reviewed module reads and writes four arguments");
        };
        assert_eq!(vertex.stage, RenderPipelineStage::Vertex);
        assert_eq!(vertex.index, STAGE_BUFFER_WRITE_VERTEX_BINDING);
        assert_eq!(vertex.access, BufferAccess::Read);
        assert_eq!(vertex.bytes, 24, "the affine reach over three vertices");
        assert_eq!(vertex.view_id, WRITE_STAGE_BUFFER_POSITIONS_VIEW);
        assert_eq!(
            vertex.allocation_id,
            WRITE_STAGE_BUFFER_POSITIONS_ALLOCATION
        );
        assert_eq!(vertex.length, 24);
        assert_eq!(vertex.source.proof_bytes(), positions);
        assert_eq!(source.index, STAGE_BUFFER_WRITE_SOURCE_BINDING);
        assert_eq!(source.access, BufferAccess::Read);
        assert_eq!(source.bytes, STAGE_BUFFER_WRITE_TEXEL_BYTES);
        assert_eq!(sink.index, STAGE_BUFFER_WRITE_SINK_BINDING);
        assert_eq!(sink.access, BufferAccess::Write);
        assert_eq!(sink.bytes, STAGE_BUFFER_WRITE_TEXEL_BYTES);
        assert_eq!(sink.view_id, WRITE_STAGE_BUFFER_SINK_VIEW);
        assert_eq!(accumulator.index, STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING);
        assert_eq!(accumulator.access, BufferAccess::ReadWrite);
        assert_eq!(accumulator.view_id, WRITE_STAGE_BUFFER_ACCUMULATOR_VIEW);
        eprintln!(
            "native writable stage-buffer plan bindings: {}",
            write_stage_buffer_table(&planned)
        );

        // The indexed run (`[0, 1, 2]` through `base_vertex = 1`) reaches
        // `base_vertex + highest index + 1 = 4` vertices, so the same affine
        // access pair asks for `4 + 4 + (4 - 1) * 8 = 32` bytes — the
        // arithmetic the core contract's own `render_affine_axis_counts` /
        // `render_affine_required_bytes` state, read from the pass's own index
        // bytes. A 32-byte view covers it; the 24-byte payload above does not.
        let mut indexed = write_stage_buffer_pass(LoadOp::Clear(sentinel()), vec![0x11; 32]);
        indexed.indices = Some(IndexBufferBinding {
            format: IndexFormat::Uint16,
            view: BufferView {
                view_id: ViewId::new(85),
                metal_binding: 0,
                allocation_id: AllocationId::new(86),
                offset: 0,
                length: 6,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(write_stage_buffer_indices()),
            },
        });
        indexed.base_vertex = 1;
        let request = write_stage_buffer_request(&indexed, &contract);
        let planned = plan_pass(&request).expect("the indexed draw's affine reach is covered");
        assert_eq!(planned.stage_buffers[0].bytes, 32);
        assert_eq!(
            planned
                .indices
                .as_ref()
                .expect("the draw is indexed")
                .base_vertex,
            1
        );
        eprintln!(
            "native writable stage-buffer indexed plan bindings: {}",
            write_stage_buffer_table(&planned)
        );

        let mut short = write_stage_buffer_pass(LoadOp::Clear(sentinel()), positions);
        short.indices = indexed.indices.clone();
        short.base_vertex = 1;
        let request = write_stage_buffer_request(&short, &contract);
        let refused = plan_pass(&request).expect_err("a 24-byte view under a 32-byte reach");
        eprintln!("native indexed affine reach refused: {refused:?}");
        // The pairing gate is what refuses the shape, and its detail carries
        // the two numbers: the extent the draw's own counts make the
        // declaration reach, and the view's own length. Core admission maps
        // this `ContractError` to `render_stage_buffer_footprint_unsupported`
        // (`contract_error_refusal`), while this rail's own mapping spells the
        // same fact through its established `trace_contract_invalid` arm with
        // the message in the detail — the reading R9g pinned for the core
        // pairing as well.
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);
        let detail = refused.detail.clone().unwrap_or_default();
        assert!(
            detail.contains("reads 32 bytes") && detail.contains("declares 24"),
            "the refusal names the draw-bound reach and the view: {detail}"
        );
    }

    /// A writable stage buffer lands through the one writeback channel
    /// (`research/docs/23` §3.3, v86/v92): the plan publishes one complete
    /// writeback per writable binding — the pass's own view, at the view's own
    /// offset, carrying the whole view's bytes — while every read-only binding
    /// publishes nothing. The bytes are the readback the encoder took after the
    /// fence, which is what makes "the stage wrote" and "the rail observed it"
    /// one fact rather than two.
    #[test]
    fn the_writable_stage_buffer_landings_are_published_through_the_writeback_channel() {
        let contract = write_stage_buffer_pipeline();
        let mut pass = write_stage_buffer_pass(LoadOp::Clear(sentinel()), stage_buffer_positions());
        // The sink sits four bytes into its allocation, so the offset half of
        // the writeback contract is observable beside the length half: the
        // published range is the view's own `4..20`, not the allocation's
        // `0..16`.
        pass.stage_buffers[2].view.offset = 4;
        let request = write_stage_buffer_request(&pass, &contract);
        let plan = plan_pass(&request).expect("the writable pair plans");
        let attachment = BufferView {
            view_id: ViewId::new(7),
            metal_binding: 0,
            allocation_id: AllocationId::new(9),
            offset: 0,
            length: 16,
            access: BufferAccess::Write,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 16]),
        };
        let planned = TraceRenderPlan {
            pass: &pass,
            contract: &contract,
            landings: vec![Some(&attachment)],
            depth_landing: None,
            stencil_landing: None,
            plan,
            present: None,
        };
        let frame = EXPECTED_TEXEL_BYTES.repeat(4);
        let sink = write_stage_buffer_source();
        let accumulator = write_stage_buffer_accumulator_after();
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![frame.clone()],
            depth: None,
            stencil: None,
            stage_buffers: vec![
                StageBufferReadback {
                    stage: RenderPipelineStage::Fragment,
                    index: STAGE_BUFFER_WRITE_SINK_BINDING,
                    bytes: sink.clone(),
                },
                StageBufferReadback {
                    stage: RenderPipelineStage::Fragment,
                    index: STAGE_BUFFER_WRITE_ACCUMULATOR_BINDING,
                    bytes: accumulator.clone(),
                },
            ],
        });
        eprintln!("native writable stage-buffer writebacks: {writebacks:?}");
        let [colour, sink_writeback, accumulator_writeback] = writebacks.as_slice() else {
            panic!("one attachment and two writable bindings land");
        };
        // The attachment keeps its own landing, first, exactly as it did before
        // this increment.
        assert_eq!(colour.view_id, ViewId::new(7));
        assert_eq!(colour.allocation_id, AllocationId::new(9));
        assert_eq!(colour.bytes, frame);
        // The write-only sink publishes the bytes the stage wrote over it.
        assert_eq!(sink_writeback.view_id, WRITE_STAGE_BUFFER_SINK_VIEW);
        assert_eq!(
            sink_writeback.allocation_id,
            WRITE_STAGE_BUFFER_SINK_ALLOCATION
        );
        assert_eq!(sink_writeback.offset, 4);
        assert_eq!(sink_writeback.bytes, sink);
        assert_ne!(
            sink_writeback.bytes,
            vec![0; STAGE_BUFFER_WRITE_TEXEL_BYTES as usize],
            "the sink starts as zeros, so a rail that landed nothing would publish them"
        );
        // The read-write accumulator publishes the previous bytes *plus* one,
        // which is the arm's read half made observable: a rail that bound zeros
        // would publish `1.0` here.
        assert_eq!(
            accumulator_writeback.view_id,
            WRITE_STAGE_BUFFER_ACCUMULATOR_VIEW
        );
        assert_eq!(accumulator_writeback.bytes, accumulator);
        assert_ne!(
            accumulator_writeback.bytes,
            write_stage_buffer_accumulator(),
            "the accumulator's previous bytes took part in the write"
        );
        // The two read-only bindings publish nothing: a writeback for them
        // would be a "bytes left the pass" claim no stage made.
        assert!(writebacks.iter().all(|writeback| {
            writeback.view_id != WRITE_STAGE_BUFFER_POSITIONS_VIEW
                && writeback.view_id != WRITE_STAGE_BUFFER_SOURCE_VIEW
        }));
    }

    /// Every disagreement between a declaration and the writable module's own
    /// argument table is refused by name (`research/docs/23` §92, R9k), in the
    /// vocabulary the Vulkan rail's translated pairing established rather than
    /// a new slug — and the pass half keeps the contract's own pair rule.
    #[test]
    fn the_writable_stage_buffer_face_refuses_what_it_cannot_land() {
        // One: the declaration says the readable `source` is written while the
        // module only reads it — the same disagreement from the access side
        // that the Vulkan rail's write-half refusal states.
        let mut written_source = write_stage_buffer_pipeline();
        written_source.stage_buffers[1].access = BufferAccess::Write;
        let refused = review_contract(&written_source).expect_err("a written source");
        eprintln!("native writable stage-buffer written source refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("declared_access"),
            Some(&FieldValue::Text("write".to_owned()))
        );
        assert_eq!(
            refused.fields.get("reflected_access"),
            Some(&FieldValue::Text("read".to_owned()))
        );

        // Two: the declaration says the sink is read-write while the module
        // only writes it. The writeback machinery is access-agnostic, so this
        // is the *declaration* half that has to be refused: a `ReadWrite` the
        // stage never reads is not the interface the module states.
        let mut read_write_sink = write_stage_buffer_pipeline();
        read_write_sink.stage_buffers[2].access = BufferAccess::ReadWrite;
        let refused = review_contract(&read_write_sink).expect_err("a read-written sink");
        eprintln!("native writable stage-buffer read-written sink refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_access"),
            Some(&FieldValue::Text("read_write".to_owned()))
        );
        assert_eq!(
            refused.fields.get("reflected_access"),
            Some(&FieldValue::Text("write".to_owned()))
        );

        // Three: a static declaration for the affine slot — the positions the
        // draw indexes into are a measurement over the draw's own vertices, and
        // one byte ceiling cannot state it.
        let mut static_positions = write_stage_buffer_pipeline();
        static_positions.stage_buffers[0].footprint = FootprintProof::Static { max_bytes: 24 };
        let refused = review_contract(&static_positions).expect_err("a static positions reach");
        eprintln!("native writable stage-buffer static positions refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_bytes"),
            Some(&FieldValue::Unsigned(24))
        );
        assert_eq!(
            refused.fields.get("reflected_accesses"),
            Some(&FieldValue::Unsigned(2))
        );

        // Four: an affine declaration that is not the module's access set —
        // here the same two sizes with the stride halved, which is a module
        // nobody wrote rather than a ceiling.
        let mut half_stride = write_stage_buffer_pipeline();
        let FootprintProof::Affine { accesses } = &mut half_stride.stage_buffers[0].footprint
        else {
            panic!("the fixture declares the affine arm");
        };
        for access in accesses.iter_mut() {
            access.terms[0].stride = STAGE_BUFFER_WRITE_VERTEX_STRIDE / 2;
        }
        let refused = review_contract(&half_stride).expect_err("another stride");
        eprintln!("native writable stage-buffer half stride refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_accesses"),
            Some(&FieldValue::Unsigned(2))
        );
        assert_eq!(
            refused.fields.get("reflected_accesses"),
            Some(&FieldValue::Unsigned(2))
        );

        // Five: a static extent under the write slot's own reach, the pre-R9k
        // mismatch one level down.
        let mut short_sink = write_stage_buffer_pipeline();
        short_sink.stage_buffers[2].footprint = FootprintProof::Static { max_bytes: 8 };
        let refused = review_contract(&short_sink).expect_err("a sink the stage writes past");
        eprintln!("native writable stage-buffer short sink refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_bytes"),
            Some(&FieldValue::Unsigned(8))
        );
        assert_eq!(
            refused.fields.get("reflected_bytes"),
            Some(&FieldValue::Unsigned(STAGE_BUFFER_WRITE_TEXEL_BYTES))
        );

        // Six: an unbounded declaration, which the render contract does not
        // admit for a stage buffer at all.
        let mut unbounded = write_stage_buffer_pipeline();
        unbounded.stage_buffers[3].footprint = FootprintProof::Unbounded;
        let refused = review_contract(&unbounded).expect_err("an unbounded accumulator");
        eprintln!("native writable stage-buffer unbounded refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");

        // Seven: the reviewed *read-only* pair keeps its pre-R9k answer — a
        // writable declaration has no writer behind it, because R9g's module
        // only reads its slots.
        let mut read_only = stage_buffer_pipeline();
        read_only.stage_buffers[1].access = BufferAccess::Write;
        let refused = review_contract(&read_only).expect_err("a writable read-only module");
        eprintln!("native writable declaration under the R9g pair refused: {refused:?}");
        assert_eq!(refused.slug, "render_stage_reflection_mismatch");
        assert_eq!(
            refused.fields.get("declared_access"),
            Some(&FieldValue::Text("write".to_owned()))
        );
        assert_eq!(
            refused.fields.get("reflected_access"),
            Some(&FieldValue::Text("read".to_owned()))
        );

        // Eight: the pass half stays the contract's own rule — a sink view the
        // pass classifies as read-only beside a `Write` declaration is refused
        // by core before this rail is asked (`trace_contract_invalid`).
        let contract = write_stage_buffer_pipeline();
        let mut pass = write_stage_buffer_pass(LoadOp::Clear(sentinel()), stage_buffer_positions());
        pass.stage_buffers[2].view.access = BufferAccess::Read;
        let request = write_stage_buffer_request(&pass, &contract);
        let refused = plan_pass(&request).expect_err("a read-only sink view");
        eprintln!("native writable stage-buffer read-only sink refused: {refused:?}");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);

        // The writable arm is inside the same declared window the read-only
        // arm is (`stage_buffer_capability_bits`): the snapshot declares the
        // pair and the cap, and the refusals above are the shapes the reviewed
        // modules still answer by name. The pre-flip refusal stays pinned on a
        // constructed snapshot
        // (`the_pre_flip_snapshot_refuses_a_stage_buffer_trace`).
        let bits = capability_bits(2048);
        let capabilities = capabilities(&bits);
        assert!(capabilities.supports_render_stage_buffers);
        assert_eq!(
            capabilities.max_render_stage_buffers,
            MAX_RENDER_STAGE_BUFFERS
        );
    }

    /// The affine bound over an *indexed draw whose index view is a lease* is
    /// three states (`research/docs/23` §92 R9k; §96 E-L1).
    ///
    /// The contract's count rule (`render_affine_axis_counts`) reads index
    /// bytes, and a lease view's bytes are not declared by the trace. The core's
    /// default arm therefore refuses that pairing by name, which is the refusal
    /// [`DeviceCapabilities::admit_render_passes`] still states before any
    /// provider runs — a snapshot owns no registry, so it cannot resolve
    /// anything. The rail owns the registries: [`resolve_affine_index_bytes`]
    /// reads the lease's window and hands it to the contract's resolved entry,
    /// which states the same bound the rail's own proof evaluates.
    ///
    /// The three states live in one test because they are one decision: the
    /// rail's resolution, the count the resolved bytes state, and the by-name
    /// refusal a caller with no lease channel keeps.
    #[test]
    fn the_affine_bound_over_a_lease_index_view_is_stated_from_resolved_bytes() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(51);
        let index_allocation = AllocationId::new(86);
        let reservation = lease_registration(lease_id, index_allocation, 6, epoch);
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(reservation, write_stage_buffer_indices())
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: index_allocation,
                owner_epoch: epoch,
                size: 6,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the reservation covers its view");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let binding = IndexBufferBinding {
            format: IndexFormat::Uint16,
            view: BufferView {
                view_id: ViewId::new(85),
                metal_binding: 0,
                allocation_id: index_allocation,
                offset: 0,
                length: 6,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::StagedLease(lease_id),
            },
        };
        // The rail's own half: the staged lease resolves, and the span the
        // affine reach is bounded by comes out of the provider's copy of the
        // owner's bytes.
        let resolved = plan_index_stream(&binding, 3, 1, Some(&leases))
            .expect("the rail resolves a staged lease's index bytes");
        assert_eq!(resolved.vertex_span, 3);
        let slot = REVIEWED_MODULES
            .iter()
            .find(|module| module.source == REVIEWED_STAGE_BUFFER_WRITE_SOURCE)
            .expect("the writable module is in the table")
            .stage_buffers[0];
        let counts = [resolved.base_vertex + resolved.vertex_span, 1];
        eprintln!("native lease-index affine counts: {counts:?}");
        assert_eq!(
            slot.reach.required_bytes(counts),
            Some(32),
            "base_vertex + highest index + 1 = 4 records, two four-byte accesses eight \
             bytes apart"
        );

        // The core half: the same pass, refused by the contract pairing that
        // runs before any rail resolution exists. The refusal carries the
        // stage-buffer footprint vocabulary and no fields — `ContractError`
        // refusals spell their facts into the detail, which is exactly the
        // boundary this increment records.
        let contract = write_stage_buffer_pipeline();
        let mut pass = write_stage_buffer_pass(LoadOp::Clear(sentinel()), vec![0x11; 32]);
        pass.indices = Some(binding);
        pass.base_vertex = 1;
        let error = contract
            .validate_against(&pass, None)
            .expect_err("a lease index view leaves the affine bound unprovable");
        assert!(
            matches!(
                error,
                ContractError::StageBufferFootprintProofUnsupported {
                    stage: RenderPipelineStage::Vertex,
                    index: 0,
                }
            ),
            "the unprovable evaluation names the vertex positions slot: {error:?}"
        );
        let refused = contract_refusal(error);
        // The mapping this rail states for the same error, and the mapping core
        // admission states for it (`contract_error_refusal`,
        // `render_stage_buffer_footprint_unsupported` / Capability, which is
        // the slug a trace sees when `admit_render_passes` refuses it before
        // any provider runs). Both are one variant; the two callers spell it
        // in their own established vocabulary.
        eprintln!("native lease-index affine refused: {refused:?}");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(refused.class, ProviderErrorClass::Args);
        assert!(
            refused.fields.is_empty(),
            "the core pairing states the refused slot in its detail: {:?}",
            refused.detail
        );
        // The rail's half, reading one: with the registries in hand the same
        // request is admitted, and the counts it evaluates the reach over are
        // the contract's own answer for the resolved window — `[4, 1]`, the same
        // pair the span arithmetic above stated.
        let request = write_stage_buffer_request(&pass, &contract);
        let planned = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the rail resolves the lease index bytes and states the bound");
        let resolved_counts = contract
            .affine_axis_counts(
                &pass,
                RenderPipelineStage::Vertex,
                0,
                Some(&write_stage_buffer_indices()),
            )
            .expect("the resolved window states the draw's own counts");
        eprintln!("native lease-index affine counts (core entry): {resolved_counts:?}");
        assert_eq!(resolved_counts, counts);
        assert_eq!(planned.stage_buffers[0].bytes, 32);

        // The rail's half, reading two: without a lease channel nothing is
        // resolved, and the refusal is the index view's own name — the window
        // that could not be read is the fact a caller can act on.
        let refused = plan_with_leases(&request, None, 0, 0)
            .expect_err("a lease index view cannot be read without the registries");
        eprintln!("native lease-index affine plan refused: {refused:?}");
        assert_eq!(refused.slug, "render_index_buffer_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );

        // And the strict contract arm the snapshot states keeps its own name
        // for a caller that resolved nothing: `None` is not a guess.
        let refused = contract_refusal(
            contract
                .validate_against(&pass, None)
                .expect_err("the strict arm refuses the lease view"),
        );
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert!(
            refused
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("does not evaluate")),
            "the strict arm names the proof it cannot evaluate: {:?}",
            refused.detail
        );

        staging
            .release(lease_id)
            .expect("the fixture import is released");
    }
    fn milestone_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: VERTEX_ENTRY.to_owned(),
            fragment_entry: FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        }
    }

    /// The sentinel `LoadOp::Clear` leaves behind. Every channel has to differ
    /// from the fragment's texel, or "the pass never ran" would read back as a
    /// pass.
    fn sentinel() -> ClearColor {
        ClearColor::new([0xfe, 0xfe, 0xfe, 0xfe])
    }

    /// The reviewed render-sampler fixture's contract and its 4×4 pass
    /// (`research/docs/23` §3.3, v70/v100).
    fn sampled_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: SAMPLED_VERTEX_ENTRY.to_owned(),
            fragment_entry: SAMPLED_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            // The declaration the pass's binding pairs with (`research/docs/23`
            // §3.3, v100): the reviewed MSL module samples one `rgba8_unorm`
            // surface through its own `constexpr sampler`, which is nearest
            // filtering with clamped addressing — the one state this rail
            // executes, and the state a declaration has to repeat.
            textures: vec![metal_api_core::provider::TextureBindingContract::sampled(
                0,
                metal_api_core::provider::TextureFormat::Rgba8Unorm,
                metal_api_core::provider::SamplerPolicy::reviewed_render_sampler(),
            )],
        }
    }

    /// One `size`×`size` sampled texture view whose sixteen texels are pairwise
    /// distinct.
    fn sampled_texture_view(size: u64) -> metal_api_core::provider::TextureView {
        let bytes = (0..size as u8)
            .flat_map(|y| (0..size as u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect::<Vec<_>>();
        metal_api_core::provider::TextureView {
            view_id: ViewId::new(83),
            metal_binding: 0,
            allocation_id: AllocationId::new(53),
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            width: size,
            height: size,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: TextureAccess::Sampled,
            source: TextureSource::OwnedBytes(bytes),
        }
    }

    fn sampled_pass(size: u64) -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments[0].width = size;
        pass.color_attachments[0].height = size;
        pass.viewport = [0, 0, size as u32, size as u32];
        pass.textures = vec![sampled_texture_view(size)];
        pass
    }

    fn milestone_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
        initial: Option<&'a [u8]>,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_SOURCE,
            initial: vec![initial.map(PlannedInputSource::Declared)],
            resident: Vec::new(),
        }
    }

    /// [`plan`] for the device-level helper's shape.
    ///
    /// The helper has no trace, but a render input carries its own bytes, so
    /// this is the same call the trace path makes; only the attachment's landing
    /// view belongs to [`plan_trace`].
    fn plan_pass<'a>(
        request: &OffscreenRenderRequest<'a>,
    ) -> Result<RenderPlan<'a>, ProviderError> {
        plan(request, 0, 0)
    }

    /// The render sampler's shape selection and its plan gates
    /// (`research/docs/23` §3.3, v70).
    ///
    /// Host-side: the encoder body lives behind `cfg(target_os = "macos")`, so
    /// what these tests pin is everything answerable from values — which
    /// reviewed module a registration selects, what the plan admits, and what
    /// it refuses by name — which is also the whole falsification surface the
    /// macOS run then executes.
    #[test]
    fn the_render_sampler_is_selected_by_its_entries_and_gated_by_its_shape() {
        let sampled = sampled_pipeline();
        let selected = reviewed_module_for(&sampled).expect("the sampled pair is reviewed");
        assert_eq!(selected.source, REVIEWED_SAMPLED_SOURCE);
        assert_eq!(
            selected.path,
            "conformance/shaders/render_sampled_4x4.metal"
        );
        review_contract(&sampled).expect("the sampled registration passes the allowlist");
        // The milestone shares the sampled pair's shape, so the entries are the
        // only thing that tells the two apart — and a crossed pair is refused.
        let milestone = milestone_pipeline();
        assert_eq!(
            reviewed_module_for(&milestone).map(|module| module.path),
            Some("conformance/shaders/render_offscreen_2x2.metal")
        );
        let crossed = RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: VERTEX_ENTRY.to_owned(),
            fragment_entry: SAMPLED_FRAGMENT_ENTRY.to_owned(),
            textures: Vec::new(),
            ..sampled_pipeline()
        };
        assert!(reviewed_module_for(&crossed).is_none());
        assert_eq!(
            review_contract(&crossed).unwrap_err().slug,
            "native_render_source_not_reviewed"
        );

        // The reviewed shape plans: one 4x4 texture whose extent is the render
        // area's own.
        let pass = sampled_pass(4);
        let plan = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &sampled,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .expect("the reviewed sampling shape is planned");
        assert_eq!(plan.textures.len(), 1);
        assert_eq!(plan.textures[0].extent, [4, 4]);
        assert_eq!(plan.textures[0].source.len(), 64);
        assert_eq!(plan.vertex_entry, SAMPLED_VERTEX_ENTRY);
        assert_eq!(plan.fragment_entry, SAMPLED_FRAGMENT_ENTRY);

        // The pair and the binding are one decision in both directions.
        let unbound = {
            let mut pass = sampled_pass(4);
            pass.textures = Vec::new();
            pass
        };
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &unbound,
            pipeline: &sampled,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("unbound refused: {error:?}");
        // Since v100 the pair rules answer first (`research/docs/23` §3.3,
        // v100): the registration declared a texture this pass does not bind,
        // so no rail ever reaches an unbound descriptor.
        assert_eq!(error.slug, "trace_contract_invalid");
        assert!(error
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("declares texture binding 0")));
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &milestone,
            source: REVIEWED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("uncoupled refused: {error:?}");
        // The other direction of the same pair (`research/docs/23` §3.3,
        // v100): a pass that binds a texture its registration never declared is
        // refused by the contract before the rail asks which module runs.
        assert_eq!(error.slug, "trace_contract_invalid");
        assert!(error
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("declares no texture there")));

        // Another extent puts some fragment's sample on a texel boundary or
        // inside a neighbour, which is a filtered read the review never covered.
        let other_extent = {
            let mut pass = sampled_pass(4);
            pass.textures = vec![sampled_texture_view(2)];
            pass
        };
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &other_extent,
            pipeline: &sampled,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("extent refused: {error:?}");
        assert_eq!(error.slug, "render_texture_extent_unsupported");

        // The owner's no-copy window is refused by this rail's *source* walk
        // (`research/docs/23` §111, E-TX12): the Vulkan rail's gathered sibling
        // has no Metal expression — this rail's reviewed module declares no
        // coordinate of its own — and this rail has no channel that could read a
        // lease-backed texture at all, so the arm stops one question earlier
        // than the extent walk. The snapshot keeps the same answer as a
        // declaration beside it (`supports_render_texture_gathered_extent_no_copy`
        // stays `false`), which is what a consumer reads before submitting.
        let no_copy_extent = {
            let mut pass = sampled_pass(4);
            let mut view = sampled_texture_view(2);
            view.source = TextureSource::BorrowedNoCopy(LeaseId::new(41));
            pass.textures = vec![view];
            pass
        };
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &no_copy_extent,
            pipeline: &sampled,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("no-copy extent refused: {error:?}");
        assert_eq!(error.slug, "render_texture_source_unsupported");
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );

        // The reviewed fragment stage reads one `rgba8_unorm` 2D surface, and
        // the first increment uploads trace-owned bytes only.
        let other_format = {
            let mut pass = sampled_pass(4);
            let mut view = sampled_texture_view(4);
            view.format = TextureFormat::Bgra8Unorm;
            pass.textures = vec![view];
            pass
        };
        // The declaration states the same format the pass binds, so the pair
        // rules admit the trace and the rail's own format window is what
        // refuses (`research/docs/23` §3.3, v100).
        let mut bgra_pipeline = sampled_pipeline();
        bgra_pipeline.textures[0].format = TextureFormat::Bgra8Unorm;
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &other_format,
            pipeline: &bgra_pipeline,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        assert_eq!(error.slug, "render_texture_format_unsupported");
        let leased = {
            let mut pass = sampled_pass(4);
            let mut view = sampled_texture_view(4);
            view.source = TextureSource::StagedLease(metal_api_core::provider::LeaseId::new(7));
            pass.textures = vec![view];
            pass
        };
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &leased,
            pipeline: &sampled,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        assert_eq!(error.slug, "render_texture_source_unsupported");
    }

    /// The lanes the *other* rail widened are outside this one and stay refused
    /// here (`research/docs/23` §113/§107/§119).
    ///
    /// Metal can express every one of these formats (`.r8Unorm` / `.rg8Unorm` /
    /// `.rgba16Float` / `.r32Float` / `.r16Float`), but this rail executes its
    /// reviewed MSL module and declares exactly the formats that review covers;
    /// a snapshot that listed a format it refuses would be a claim without a
    /// reading. So every lane beyond the reviewed `rgba8_unorm` texel keeps the
    /// format refusal, with the same slug and the same fields as every other
    /// unadmitted format, and the refusal happens before the first Metal object
    /// exists (the plan step is where it lands).
    #[test]
    fn the_formats_the_reviewed_module_does_not_read_stay_refused_by_name() {
        let sampled = sampled_pipeline();
        for format in [
            TextureFormat::R8Unorm,
            TextureFormat::R8G8Unorm,
            TextureFormat::Rgba16Float,
            TextureFormat::R32Float,
            TextureFormat::R16Float,
        ] {
            let mut pass = sampled_pass(4);
            let mut view = sampled_texture_view(4);
            view.format = format;
            // One byte and two bytes per texel (`TextureFormat::bytes_per_texel`):
            // the arm's own byte extent, so the contract's length rule is not
            // the one that answers.
            view.source =
                TextureSource::OwnedBytes(vec![0x5a; (4 * 4 * format.bytes_per_texel()) as usize]);
            pass.textures = vec![view];
            let mut pipeline = sampled.clone();
            pipeline.textures[0].format = format;
            let error = plan_pass(&OffscreenRenderRequest {
                pass: &pass,
                pipeline: &pipeline,
                source: REVIEWED_SAMPLED_SOURCE,
                initial: vec![None],
                resident: Vec::new(),
            })
            .unwrap_err();
            eprintln!("{format:?} refused: {error:?}");
            assert_eq!(error.slug, "render_texture_format_unsupported");
            assert_eq!(
                error.fields.get("format"),
                Some(&metal_api_core::provider::FieldValue::Text(format!(
                    "{format:?}"
                ))),
                "the refusal names the format it read"
            );
            assert_eq!(
                error.fields.get("binding"),
                Some(&metal_api_core::provider::FieldValue::Unsigned(0))
            );
        }

        // The one-dimensional arm (2026-09-19, census b10's `texture_shape`
        // bucket) is the same table's other half: no Apple-side reading states
        // the Metal 1D equivalence the arm would need, so a `D1Array` view
        // bound beside the reviewed module keeps the shape's own refusal — and
        // the snapshot declares no one-dimensional window at all, which is the
        // fail-closed direction every consumer of the field reads.
        assert_eq!(
            capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
                .max_render_texture_dimension_1d,
            0,
            "the native snapshot declares no one-dimensional sampled window"
        );
        let mut pass = sampled_pass(4);
        let mut view = sampled_texture_view(4);
        view.texture_type = TextureType::D1Array;
        view.height = 1;
        view.source = TextureSource::OwnedBytes(vec![0x5a; 4 * 4]);
        pass.textures = vec![view];
        // The declaration restates the view's type, which is the pairing the
        // contract holds the two to — so what answers the shape here is the
        // rail's own gate rather than the structural rule that would catch a
        // declaration naming another type.
        let mut one_dim = sampled.clone();
        one_dim.textures[0].texture_type = TextureType::D1Array;
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &one_dim,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("D1Array refused: {error:?}");
        assert_eq!(error.slug, "render_texture_shape_unsupported");
        assert_eq!(
            error.fields.get("texture_type"),
            Some(&metal_api_core::provider::FieldValue::Text(
                "D1Array".to_owned()
            ))
        );

        // The three-dimensional arm (2026-09-20, the `D3` sampled texture arm)
        // is the same table's third axis: a volume's `float3` coordinate has no
        // reviewed MSL sibling here either, so a `D3` view bound beside the
        // reviewed module keeps the shape's own refusal by name — and the
        // snapshot declares no three-dimensional window at all, which is the
        // fail-closed direction every consumer of the field reads.
        assert_eq!(
            capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
                .max_render_texture_dimension_3d,
            0,
            "the native snapshot declares no three-dimensional sampled window"
        );
        let mut pass = sampled_pass(4);
        let mut view = sampled_texture_view(4);
        view.texture_type = TextureType::D3;
        view.depth = 2;
        view.source = TextureSource::OwnedBytes(vec![0x5a; 4 * 4 * 2 * 4]);
        pass.textures = vec![view];
        // The declaration restates the view's type, the pairing the contract
        // holds the two to — so what answers the shape is the rail's own gate
        // rather than the structural rule that would catch a declaration naming
        // another type.
        let mut volume = sampled.clone();
        volume.textures[0].texture_type = TextureType::D3;
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &volume,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("D3 refused: {error:?}");
        assert_eq!(error.slug, "render_texture_shape_unsupported");
        assert_eq!(
            error.fields.get("texture_type"),
            Some(&metal_api_core::provider::FieldValue::Text("D3".to_owned()))
        );
    }

    /// The declaration repeats the state the reviewed module's own `constexpr
    /// sampler` carries (`research/docs/23` §3.3, v100).
    ///
    /// This rail executes the reviewed MSL module, whose sampler state lives in
    /// the module itself: the rail cannot create another one without another
    /// reviewed module, so a declaration naming another filtering or addressing
    /// mode is refused with both halves rather than executed as the reviewed
    /// state under a name that says otherwise.
    #[test]
    fn a_render_sampler_declaration_must_repeat_the_reviewed_module() {
        let pass = sampled_pass(4);
        for (filter, address) in [
            (SamplerFilter::Linear, SamplerAddressMode::ClampToEdge),
            (SamplerFilter::Nearest, SamplerAddressMode::Repeat),
        ] {
            let mut declaring = sampled_pipeline();
            declaring.textures[0].sampler = Some(SamplerPolicy { filter, address });
            let error = plan_pass(&OffscreenRenderRequest {
                pass: &pass,
                pipeline: &declaring,
                source: REVIEWED_SAMPLED_SOURCE,
                initial: vec![None],
                resident: Vec::new(),
            })
            .unwrap_err();
            eprintln!("{filter:?}/{address:?} refused: {error:?}");
            assert_eq!(error.slug, "render_texture_sampler_unsupported");
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.fields.get("binding"), Some(&FieldValue::Unsigned(0)));
            assert_eq!(
                error.fields.get("filter"),
                Some(&FieldValue::Text(format!("{filter:?}")))
            );
            assert_eq!(
                error.fields.get("address"),
                Some(&FieldValue::Text(format!("{address:?}")))
            );
            assert_eq!(
                error.fields.get("module_filter"),
                Some(&FieldValue::Text("Nearest".to_owned()))
            );
            assert_eq!(
                error.fields.get("module_address"),
                Some(&FieldValue::Text("ClampToEdge".to_owned()))
            );
        }
        // Control: the state the reviewed module carries admits the very same
        // pass, so the refusal above is about the state rather than the shape.
        plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &sampled_pipeline(),
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .expect("the reviewed state admits the reviewed pass");
    }

    /// The texel space has no rail here (2026-09-19, census v43's
    /// `texture_state` axis): the reviewed MSL modules spell one `constexpr
    /// sampler` in the normalized space and take no `[[sampler(n)]]` argument
    /// at all, so the snapshot keeps the fail-closed default and a
    /// directly-constructed pass that states the space is refused by name
    /// before any Metal object exists.
    #[test]
    fn the_texel_space_is_never_declared_and_a_pass_that_states_it_is_refused() {
        assert!(!capability_bits(16384).supports_render_pixel_coordinate_sampler);
        assert!(!capabilities(&capability_bits(16384))
            .declares_render_pixel_coordinate_sampler_support());

        let mut pass = sampled_pass(4);
        pass.samplers = vec![
            metal_api_core::provider::RenderSamplerBinding::with_coordinates(
                0,
                SamplerPolicy {
                    filter: SamplerFilter::Linear,
                    address: SamplerAddressMode::ClampToZero,
                },
                metal_api_core::provider::SamplerCoordinates::Pixel,
            ),
        ];
        let pipeline = sampled_pipeline();
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("texel space refused: {error:?}");
        assert_eq!(error.slug, "render_pixel_coordinate_sampler_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("rail"),
            Some(&FieldValue::Text("native".to_owned()))
        );
        assert_eq!(
            error.fields.get("sampler_binding"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// The reviewed window in the runtime-sampler increment
    /// (`research/docs/23` §3.3, v102): this rail has no MSL module that takes
    /// a `[[sampler(n)]]` argument, and its reviewed module samples one
    /// `MTLTexture`, so both shapes are refused by name instead of executed
    /// through a module that does not carry them.
    #[test]
    fn a_runtime_sampler_and_a_second_texture_stay_outside_the_reviewed_table() {
        // A declaration that pairs its texture with a runtime sampler argument:
        // the reviewed module spells its own `constexpr sampler`, so the state
        // the request states has no module behind it. The refusal names both
        // halves of the pairing and both states.
        let mut runtime = sampled_pipeline();
        runtime.textures[0].sampler = None;
        runtime.textures[0].runtime_sampler = Some(0);
        let mut pass = sampled_pass(4);
        pass.samplers = vec![metal_api_core::provider::RenderSamplerBinding::new(
            0,
            SamplerPolicy {
                filter: SamplerFilter::Linear,
                address: SamplerAddressMode::Repeat,
            },
        )];
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &runtime,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("runtime sampler refused: {error:?}");
        assert_eq!(error.slug, "render_runtime_sampler_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.fields.get("binding"), Some(&FieldValue::Unsigned(0)));
        assert_eq!(
            error.fields.get("sampler_binding"),
            Some(&FieldValue::Unsigned(0))
        );
        assert_eq!(
            error.fields.get("filter"),
            Some(&FieldValue::Text("Linear".to_owned()))
        );
        assert_eq!(
            error.fields.get("address"),
            Some(&FieldValue::Text("Repeat".to_owned()))
        );
        assert_eq!(
            error.fields.get("module_filter"),
            Some(&FieldValue::Text("Nearest".to_owned()))
        );
        assert_eq!(
            error.fields.get("module_address"),
            Some(&FieldValue::Text("ClampToEdge".to_owned()))
        );

        // A second sampled texture: the reviewed module reads one
        // `texture2d<float>` argument, so the wider contract — which core
        // admits for a translated stage's sake — is refused by name here.
        let mut second = sampled_texture_view(4);
        second.metal_binding = 1;
        second.view_id = metal_api_core::provider::ViewId::new(84);
        second.allocation_id = metal_api_core::provider::AllocationId::new(54);
        let mut two = sampled_pass(4);
        two.textures.push(second);
        let mut wide = sampled_pipeline();
        wide.textures
            .push(metal_api_core::provider::TextureBindingContract::sampled(
                1,
                metal_api_core::provider::TextureFormat::Rgba8Unorm,
                SamplerPolicy::reviewed_render_sampler(),
            ));
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &two,
            pipeline: &wide,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .unwrap_err();
        eprintln!("two textures refused: {error:?}");
        assert_eq!(error.slug, "render_texture_stage_unsupported");
        assert_eq!(error.fields.get("textures"), Some(&FieldValue::Unsigned(2)));
        assert_eq!(error.fields.get("bindings"), Some(&FieldValue::Unsigned(1)));

        // Control: the reviewed shape itself still plans, so the two refusals
        // above are about the wider shapes rather than about the fixture.
        plan_pass(&OffscreenRenderRequest {
            pass: &sampled_pass(4),
            pipeline: &sampled_pipeline(),
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        })
        .expect("the reviewed shape plans");
    }

    /// The render bits the provider declares now carry the sampler
    /// (`research/docs/23` §3.3, v70): the three values are the rail's own
    /// window, so they and the plan gates cannot disagree.
    #[test]
    fn render_texture_capability_bits_name_the_reviewed_window() {
        let bits = render_texture_capability_bits();
        assert!(bits.supports_render_texture_sampling);
        assert_eq!(bits.max_render_textures, 1);
        // The per-stage window stays undeclared (`research/docs/23` §3.3,
        // E-TC1): this rail's reviewed module samples one texture argument, so
        // the list bound is the whole rule and a fragment stage that declares
        // thirteen sampled textures is refused by name by core admission
        // instead of being executed against slots no Apple reading sized.
        assert_eq!(bits.max_render_textures_per_stage, 0);
        assert!(!capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .declares_render_texture_per_stage_ceiling());
        assert_eq!(
            bits.supported_render_texture_formats,
            vec![TextureFormat::Rgba8Unorm]
        );
        // The narrow lanes and the eight-byte half-float lane belong to the
        // other rail (`research/docs/23` §113/§107): Metal can express
        // `.r8Unorm` / `.rg8Unorm` / `.rgba16Float`, but this rail's table is
        // the reviewed MSL module's window, so a snapshot that listed any of
        // them would claim a shape whose Apple-side reading does not exist yet.
        for wider in [
            TextureFormat::R8Unorm,
            TextureFormat::R8G8Unorm,
            TextureFormat::Rgba16Float,
        ] {
            assert!(
                !bits.supported_render_texture_formats.contains(&wider),
                "{wider:?} belongs to the widened rail, not this one"
            );
        }
        // The gathered extent is this rail's own boundary rather than a
        // missing measurement (`research/docs/23` §3.3, E-TX10): every sampled
        // source of another extent is refused by name, so the declaration
        // stays at the contract's fail-closed default beside the three fields
        // above.
        assert!(!bits.supports_render_texture_gathered_extent);
        // The gathered extent's no-copy arm (`research/docs/23` §111, E-TX12):
        // the same walk refuses the owner's window of another extent, and this
        // rail has no Metal expression for the destination grid's index, so the
        // declaration stays at the default beside the bit above.
        assert!(!bits.supports_render_texture_gathered_extent_no_copy);
        let declared = capability_bits(APPLE_2D_TEXTURE_CEILING);
        assert_eq!(
            declared.supports_render_texture_sampling,
            bits.supports_render_texture_sampling
        );
        assert_eq!(declared.max_render_textures, bits.max_render_textures);
        assert_eq!(
            declared.max_render_textures_per_stage,
            bits.max_render_textures_per_stage
        );
        assert_eq!(
            declared.supported_render_texture_formats,
            bits.supported_render_texture_formats
        );
        assert!(!declared.supports_render_texture_gathered_extent);
        assert_eq!(
            declared.supports_render_texture_gathered_extent,
            bits.supports_render_texture_gathered_extent
        );
        assert!(!declared.supports_render_texture_gathered_extent_no_copy);
        assert_eq!(
            declared.supports_render_texture_gathered_extent_no_copy,
            bits.supports_render_texture_gathered_extent_no_copy
        );
    }

    /// The refusal of a (source, entry pair) triple the rail does not review.
    fn allowlist_refusal(source: &str, vertex: &str, fragment: &str) -> ProviderError {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: vertex.to_owned(),
            fragment_entry: fragment.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::None,
            textures: Vec::new(),
        };
        let request = OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source,
            initial: vec![None],
            resident: Vec::new(),
        };
        plan_pass(&request).map(|_| ()).unwrap_err()
    }

    #[test]
    fn plan_accepts_the_milestone_shape_and_fixes_the_readback_extent() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap();
        assert_eq!(planned.vertex_entry, VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, FRAGMENT_ENTRY);
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(planned.viewport, [0, 0, 2, 2]);
        assert_eq!(planned.vertices, 3);
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert_eq!(attachment.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(attachment.store, RenderStoreAction::Store);
        // 2x2 texels of a 4-byte format: 16 bytes, two rows of 8. The extent
        // is the attachment's own (`research/docs/23` §78).
        assert_eq!(attachment.texel.bytes, 16);
        assert_eq!(attachment.texel.row_pitch, 8);
        assert_eq!(planned.depth_texel.bytes, 16);
        assert_eq!(planned.depth_texel.row_pitch, 8);
        assert_eq!(attachment.initial_bytes(), None);
        let RenderLoadAction::Clear(components) = attachment.load else {
            panic!("the milestone clears its attachment");
        };
        assert_eq!(components, [254.0 / 255.0; 4]);
    }

    /// The clear payload of the eight-byte format the census named: four
    /// little-endian halves, in the format's memory order (`research/docs/23`
    /// §78).
    fn rgba16float_clear() -> ClearColor {
        ClearColor::from_bytes(&[0x00, 0x3c, 0x00, 0x38, 0x66, 0x2e, 0x00, 0x30])
            .expect("one eight-byte texel")
    }

    /// `Rgba16Float`'s own plan facts (`research/docs/23` §78): the reviewed
    /// single-output module serves it, the readback extent is the attachment's
    /// own width rather than the pass's, and a `Load` of the same attachment is
    /// held to that width.
    #[test]
    fn plan_reads_an_rgba16float_attachment_at_its_own_texel_width() {
        let mut pass = milestone_pass(LoadOp::Clear(rgba16float_clear()));
        pass.color_attachments[0].format = AttachmentFormat::Rgba16Float;
        let mut pipeline = milestone_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::Rgba16Float];
        let planned = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect("the reviewed single-output module serves the four-component float format");
        assert_eq!(planned.source, REVIEWED_SOURCE);
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert_eq!(attachment.format, RenderPixelFormat::Rgba16Float);
        // 2x2 texels of an eight-byte format: 32 bytes, two rows of 16 — twice
        // the four-byte class's extent for the same raster.
        assert_eq!(attachment.texel.bytes, 32);
        assert_eq!(attachment.texel.row_pitch, 16);
        // The depth surface keeps its own four-byte width beside it.
        assert_eq!(planned.depth_texel.bytes, 16);
        assert_eq!(planned.depth_texel.row_pitch, 8);
        let RenderLoadAction::Clear(components) = attachment.load else {
            panic!("the fixture clears its attachment");
        };
        assert_eq!(components, [1.0, 0.5, f64::from(0.099_975_586_f32), 0.125]);

        // A `Load` uploads the attachment's own texels: the eight-byte payload
        // lands, and a four-byte one is refused by name with the width it was
        // measured against, exactly as core admission measures it.
        let mut loading = milestone_pass(LoadOp::Load);
        loading.color_attachments[0].format = AttachmentFormat::Rgba16Float;
        let previous = vec![0x5a; 32];
        let planned = plan_pass(&milestone_request(&loading, &pipeline, Some(&previous)))
            .expect("a 2x2 Rgba16Float attachment loads 32 bytes");
        assert_eq!(planned.attachments[0].initial_bytes(), Some(&previous[..]));
        let short = vec![0x5a; 16];
        let refused = plan_pass(&milestone_request(&loading, &pipeline, Some(&short)))
            .expect_err("a four-byte payload cannot state an eight-byte format's texels");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "render_attachment_initial_mismatch");
    }

    /// The four-component modules are exactly the ones that serve 16-bit float:
    /// the single-channel format's reviewed stage is the indexed one, so a
    /// `vertex_id` pipeline that names `R32Float` is refused instead of
    /// compiled with a `float4` store (`research/docs/23` §78).
    #[test]
    fn the_vertex_id_module_serves_the_four_component_formats_only() {
        assert!(reviewed_module(&VertexLayout::None, &[AttachmentFormat::Rgba16Float]).is_some());
        assert!(reviewed_module(&VertexLayout::None, &[AttachmentFormat::R32Float]).is_none());

        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments[0].format = AttachmentFormat::R32Float;
        let mut pipeline = milestone_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::R32Float];
        let refused = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect_err("the milestone module stores a float4 and this format has one channel");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "native_render_source_not_reviewed");
    }

    #[test]
    fn plan_refuses_an_unreviewed_source_or_entry() {
        let edited = REVIEWED_SOURCE.replace("192.0", "191.0");
        assert_ne!(edited, REVIEWED_SOURCE, "the fixture still writes 192.0");
        for (label, error) in [
            (
                "edited source",
                allowlist_refusal(&edited, VERTEX_ENTRY, FRAGMENT_ENTRY),
            ),
            (
                "wrong vertex entry",
                allowlist_refusal(REVIEWED_SOURCE, "render_triangle", FRAGMENT_ENTRY),
            ),
            (
                "wrong fragment entry",
                allowlist_refusal(REVIEWED_SOURCE, VERTEX_ENTRY, "render_solid"),
            ),
        ] {
            assert_eq!(error.slug, "native_render_source_not_reviewed", "{label}");
            assert_eq!(error.class, ProviderErrorClass::Capability, "{label}");
            assert_eq!(error.phase, ProviderPhase::Compile, "{label}");
        }
    }

    #[test]
    fn reviewed_fixture_matches_the_expected_texel_bytes() {
        for literal in ["64.0 / 255.0", "128.0 / 255.0", "192.0 / 255.0"] {
            assert!(
                REVIEWED_SOURCE.contains(literal),
                "the reviewed fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_SOURCE.contains("0.5"),
            "the reviewed fixture must not carry a half-integer tie constant"
        );
        for (component, byte) in EXPECTED_TEXEL_COMPONENTS.iter().zip(EXPECTED_TEXEL_BYTES) {
            // A byte/255 constant is far from a tie, so every driver lands the
            // same byte (`research/docs/23` §3.5).
            assert_eq!((component * 255.0).round(), f64::from(byte));
        }
    }

    #[test]
    fn attachment_formats_map_to_the_admitted_pixel_formats() {
        for (format, expected) in [
            (AttachmentFormat::Rgba8Unorm, RenderPixelFormat::Rgba8Unorm),
            (AttachmentFormat::Bgra8Unorm, RenderPixelFormat::Bgra8Unorm),
            (AttachmentFormat::R32Float, RenderPixelFormat::R32Float),
            (
                AttachmentFormat::Rgba16Float,
                RenderPixelFormat::Rgba16Float,
            ),
        ] {
            assert_eq!(pixel_format(format).unwrap(), expected);
            assert!(SUPPORTED_COLOR_FORMATS.contains(&format));
        }
        assert_eq!(RenderPixelFormat::Rgba8Unorm.name(), "rgba8_unorm");
        assert_eq!(RenderPixelFormat::Bgra8Unorm.name(), "bgra8_unorm");
        assert_eq!(RenderPixelFormat::R32Float.name(), "r32_float");
        assert_eq!(RenderPixelFormat::Rgba16Float.name(), "rgba16_float");
    }

    #[test]
    fn r32uint_is_refused_as_a_colour_attachment() {
        let error = pixel_format(AttachmentFormat::R32Uint).unwrap_err();
        assert_eq!(error.slug, "attachment_format_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert!(!SUPPORTED_COLOR_FORMATS.contains(&AttachmentFormat::R32Uint));
    }

    #[test]
    fn clear_components_decode_by_format_byte_order() {
        // One colour, two byte strings: memory order is R,G,B,A for the first
        // format and B,G,R,A for the second.
        assert_eq!(
            clear_components(
                ClearColor::new([0x40, 0x80, 0xc0, 0xff]),
                RenderPixelFormat::Rgba8Unorm
            ),
            EXPECTED_TEXEL_COMPONENTS
        );
        assert_eq!(
            clear_components(
                ClearColor::new([0xc0, 0x80, 0x40, 0xff]),
                RenderPixelFormat::Bgra8Unorm
            ),
            EXPECTED_TEXEL_COMPONENTS
        );
        // The single-channel format's four bytes are the red component itself.
        assert_eq!(
            clear_components(
                ClearColor::new(0.5_f32.to_le_bytes()),
                RenderPixelFormat::R32Float
            ),
            [0.5, 0.0, 0.0, 1.0]
        );
        // The eight-byte format's payload is four little-endian halves, and the
        // components are the values they name: `0x3c00` is 1.0, `0x3800` is
        // 0.5, and `0x2e66` is the half nearest 0.1 — a value the f32 of 0.1 is
        // *not*, which is the rounding a 16-bit float attachment observes
        // (`research/docs/23` §78).
        assert_eq!(
            clear_components(
                ClearColor::from_bytes(&[0x00, 0x3c, 0x00, 0x38, 0x66, 0x2e, 0x00, 0x30])
                    .expect("one eight-byte texel"),
                RenderPixelFormat::Rgba16Float
            ),
            [1.0, 0.5, f64::from(0.099_975_586_f32), f64::from(0.125_f32),]
        );
    }

    #[test]
    fn clear_load_and_dont_care_map_to_distinct_load_actions() {
        assert_eq!(
            load_action(
                LoadOp::Clear(ClearColor::new([0, 0, 0, 0xff])),
                RenderPixelFormat::Rgba8Unorm
            )
            .unwrap(),
            RenderLoadAction::Clear([0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(
            load_action(LoadOp::Load, RenderPixelFormat::Rgba8Unorm).unwrap(),
            RenderLoadAction::Load
        );
        assert_eq!(
            load_action(LoadOp::DontCare, RenderPixelFormat::Rgba8Unorm).unwrap(),
            RenderLoadAction::DontCare
        );
    }

    #[test]
    fn store_dont_care_is_admitted() {
        assert_eq!(
            store_action(StoreOp::Store).unwrap(),
            RenderStoreAction::Store
        );
        assert_eq!(
            store_action(StoreOp::DontCare).unwrap(),
            RenderStoreAction::DontCare
        );
    }

    #[test]
    fn a_borrowed_store_is_refused_by_name() {
        // The owner-window store (`research/docs/23` §114, E-TX8) lands the
        // pass's frame in the owner's *registered window*. This rail carries
        // the owner's window as a render **input** (`research/docs/23` §72,
        // R3d) and has no landing route that writes one, so the arm stays
        // refused by name rather than executing as a plain store: a plain
        // store is not what the caller declared, and the guest's pages would
        // be left holding the pass's previous bytes. E-TX9b pins the refusal
        // so the boundary is a measured reading of the native rail rather than
        // an incidental code path.
        let error = store_action(StoreOp::Borrowed).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("source"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );

        // The landing-view arm (`research/docs/23` §115 之后的增量，E-TX13) is
        // refused for the same reason with its own `source`: the rail has no
        // route that writes an owner's window, and the refusal names *which*
        // declaration the caller meant so a reader does not have to guess
        // whether the window came from the attachment's own view or from a
        // second one.
        let landing = metal_api_core::provider::AttachmentLandingView {
            allocation_id: AllocationId::new(77),
            view_id: ViewId::new(78),
        };
        let error = store_action(StoreOp::BorrowedLanding(landing)).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("source"),
            Some(&FieldValue::Text("landing_view".to_owned()))
        );
        assert_eq!(
            error.fields.get("landing_view"),
            Some(&FieldValue::Unsigned(landing.view_id.get()))
        );
        assert_eq!(
            error.fields.get("landing_allocation"),
            Some(&FieldValue::Unsigned(landing.allocation_id.get()))
        );
    }

    #[test]
    fn plan_refuses_an_attachment_beyond_the_declared_extent() {
        // The rail's reviewed ceiling is `REVIEWED_ATTACHMENT_CEILING` (R1b,
        // `research/docs/23` §70); one texel beyond it is refused rather than
        // silently clamped, with the maximum the plan measured against.
        let over = REVIEWED_ATTACHMENT_CEILING[0] + 1;
        let over_u32 = u32::try_from(over).expect("the ceiling fits the viewport");
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [0, 0, over_u32, over_u32];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = over;
        attachment.height = over;
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "attachment_dimension_limit");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("maximum_width"),
            Some(&FieldValue::Unsigned(REVIEWED_ATTACHMENT_CEILING[0]))
        );
    }

    #[test]
    fn plan_accepts_a_four_by_four_attachment() {
        // The extent ceiling the capability reports is executable: a 4x4
        // attachment plans with sixty-four texel bytes.
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [0, 0, 4, 4];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = 4;
        attachment.height = 4;
        let pipeline = milestone_pipeline();
        let plan = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect("a four-by-four attachment is within the declared extent");
        assert_eq!(plan.extent, [4, 4]);
        assert_eq!(plan.attachments[0].texel.bytes, 64);
    }

    #[test]
    fn plan_carries_a_declared_viewport_and_a_masked_blend_entry() {
        // v100: the plan is the one description both the encoder and the
        // pipeline are built from, so the pass's own rect and its
        // per-attachment blend state have to reach it unchanged. The two
        // rails' device readings are the same fields' execution.
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [1, 1, 1, 1];
        pass.blend = Some(RenderPassBlend {
            attachments: vec![BlendAttachment {
                enabled: false,
                source_rgb: BlendFactor::One,
                destination_rgb: BlendFactor::Zero,
                source_alpha: BlendFactor::One,
                destination_alpha: BlendFactor::Zero,
                operation: BlendOperation::Add,
                alpha_operation: BlendOperation::Subtract,
                write_mask: ColorWriteMask::RED.union(ColorWriteMask::ALPHA),
            }],
        });
        let pipeline = milestone_pipeline();
        let plan = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect("a rect inside the attachment is the increment's shape");
        assert_eq!(plan.viewport, [1, 1, 1, 1]);
        assert_eq!(plan.extent, [2, 2]);
        let blend = plan
            .blend
            .as_ref()
            .expect("the entry travels with the plan");
        assert!(!blend.attachments[0].enabled);
        assert_eq!(
            blend.attachments[0].alpha_operation,
            BlendOperation::Subtract
        );
        assert_eq!(
            blend.attachments[0].write_mask,
            ColorWriteMask::RED.union(ColorWriteMask::ALPHA)
        );
        assert_eq!(blend.attachments[0].write_mask.bits(), 0x9);
    }

    #[test]
    fn plan_accepts_the_reviewed_window_boundary() {
        // R1b (`research/docs/23` §70) pinned 64×64; R5a (§73) moves the
        // boundary to 2048×2048. The boundary extent the new fixture pins plans
        // with the whole texel count, so the declared window is executable on
        // the host before any Metal object exists.
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.viewport = [
            0,
            0,
            REVIEWED_ATTACHMENT_CEILING[0] as u32,
            REVIEWED_ATTACHMENT_CEILING[1] as u32,
        ];
        let attachment = &mut pass.color_attachments[0];
        attachment.width = REVIEWED_ATTACHMENT_CEILING[0];
        attachment.height = REVIEWED_ATTACHMENT_CEILING[1];
        let pipeline = milestone_pipeline();
        let plan = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect("the reviewed boundary extent is within the declared window");
        assert_eq!(
            plan.extent,
            [
                REVIEWED_ATTACHMENT_CEILING[0] as u32,
                REVIEWED_ATTACHMENT_CEILING[1] as u32
            ]
        );
        assert_eq!(
            plan.attachments[0].texel.bytes,
            usize::try_from(REVIEWED_ATTACHMENT_CEILING[0] * REVIEWED_ATTACHMENT_CEILING[1] * 4)
                .expect("the reviewed window's texel bytes fit a host usize")
        );
    }

    #[test]
    fn the_device_half_refuses_an_extent_beyond_the_devices_own_limit() {
        // R1b (`research/docs/23` §70): the declared window's device half is
        // the provider's own refusal, asked before the reviewed-ceiling check,
        // so a trace that skipped admission is refused with the device's answer
        // instead of an extent the device could never open. The check is a
        // value-level one, so it runs on a host without Metal.
        let mut refused_trace = milestone_trace(LoadOp::Clear(sentinel())).0;
        let pass = render_pass_mut(&mut refused_trace);
        pass.color_attachments[0].width = 8;
        pass.color_attachments[0].height = 8;
        pass.viewport = [0, 0, 8, 8];

        let refused = refuse_attachment_extent_over_device_limit(&refused_trace, 4)
            .expect_err("an extent beyond the device's own limit is a refusal");
        eprintln!("refused: {refused:?}");
        assert_eq!(refused.slug, "attachment_extent_device_limit");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("width"), Some(&FieldValue::Unsigned(8)));
        assert_eq!(
            refused.fields.get("maximum_width"),
            Some(&FieldValue::Unsigned(4))
        );
        assert_eq!(
            refused.fields.get("maximum_height"),
            Some(&FieldValue::Unsigned(4))
        );
        // The same trace on a device at least as wide as the review passes the
        // device half; the reviewed ceiling is what bounds it then.
        refuse_attachment_extent_over_device_limit(&refused_trace, APPLE_2D_TEXTURE_CEILING)
            .expect("an eight-texel extent is inside every reviewed device's limit");

        // The boundary extent itself: inside a device at the Apple ceiling,
        // refused by a device one texel narrower.
        let mut boundary_trace = milestone_trace(LoadOp::Clear(sentinel())).0;
        let pass = render_pass_mut(&mut boundary_trace);
        pass.color_attachments[0].width = REVIEWED_ATTACHMENT_CEILING[0];
        pass.color_attachments[0].height = REVIEWED_ATTACHMENT_CEILING[1];
        pass.viewport = [0, 0, 64, 64];
        refuse_attachment_extent_over_device_limit(&boundary_trace, APPLE_2D_TEXTURE_CEILING)
            .expect("the reviewed boundary is inside the device's own limit");
        let refused = refuse_attachment_extent_over_device_limit(
            &boundary_trace,
            REVIEWED_ATTACHMENT_CEILING[0] - 1,
        )
        .expect_err("a narrower device refuses the reviewed boundary");
        assert_eq!(refused.slug, "attachment_extent_device_limit");
        assert_eq!(
            refused.fields.get("maximum_width"),
            Some(&FieldValue::Unsigned(REVIEWED_ATTACHMENT_CEILING[0] - 1))
        );
    }

    #[test]
    fn plan_refuses_a_pipeline_whose_format_disagrees_with_the_attachment() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let mut pipeline = milestone_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::Bgra8Unorm];
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
    }

    /// The reviewed dual shape plans two attachments, one per colour location,
    /// with the second location's texel distinct from the first's.
    #[test]
    fn plan_accepts_the_dual_shape_and_plans_two_attachments() {
        let pass = dual_pass();
        let pipeline = dual_pipeline();
        let planned = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap();
        assert_eq!(
            planned.module_path,
            "conformance/shaders/quad_indexed_2x2_dual.metal"
        );
        assert_eq!(planned.vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, DUAL_FRAGMENT_ENTRY);
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(planned.vertices, 6);
        let [first, second] = planned.attachments.as_slice() else {
            panic!("the dual shape renders two attachments");
        };
        assert_eq!(first.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(second.format, RenderPixelFormat::Rgba8Unorm);
        assert!(matches!(first.load, RenderLoadAction::Clear(_)));
        assert!(matches!(second.load, RenderLoadAction::Clear(_)));
        assert_eq!(first.store, RenderStoreAction::Store);
        assert_eq!(second.store, RenderStoreAction::Store);
        assert_eq!(first.initial_bytes(), None);
        assert_eq!(second.initial_bytes(), None);
        assert_eq!(first.texel.bytes, 16);
        assert_eq!(first.texel.row_pitch, 8);
        assert_eq!(second.texel, first.texel);
    }

    /// The v19 store increment: the same dual shape with location 1 discarded
    /// plans both attachments, but the discarded one carries `DontCare` rather
    /// than a landing observation.
    #[test]
    fn plan_admits_a_discarded_attachment_beside_a_stored_one() {
        let mut pass = dual_pass();
        pass.color_attachments[1].store = StoreOp::DontCare;
        let pipeline = dual_pipeline();
        let planned = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap();
        let [first, second] = planned.attachments.as_slice() else {
            panic!("the dual shape renders two attachments");
        };
        assert_eq!(first.store, RenderStoreAction::Store);
        assert_eq!(second.store, RenderStoreAction::DontCare);
        assert!(matches!(first.load, RenderLoadAction::Clear(_)));
        assert!(matches!(second.load, RenderLoadAction::Clear(_)));
    }

    /// The zero-colour-attachment depth pass (`research/docs/23` §3.3, v46): a
    /// pass with no colour attachment at all plans its raster from the stored
    /// depth attachment, compiles the reviewed module whose fragment stage
    /// generates no output, and carries no colour attachment to read back.
    ///
    /// Runs without a device, so the shape rules are the whole check here; the
    /// macOS CI executes the compiled pipeline.
    #[test]
    fn plan_admits_a_zero_colour_attachment_depth_pass() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments.clear();
        // The reviewed depth-only module reads the depth pair's stream, so the
        // pass binds one: the same two-attribute layout its vertex stage was
        // written for.
        pass.vertex_buffers = vec![BufferView {
            view_id: ViewId::new(31),
            metal_binding: 0,
            allocation_id: AllocationId::new(33),
            offset: 0,
            length: 192,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 192]),
        }];
        pass.indices = Some(IndexBufferBinding {
            view: quad_index_view(),
            format: IndexFormat::Uint16,
        });
        pass.vertices = 6;
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: DEPTH_STORE_ALLOCATION,
                view_id: DEPTH_STORE_VIEW,
            }),
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        let pipeline = RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: DEPTH_ONLY_VERTEX_ENTRY.to_owned(),
            fragment_entry: DEPTH_ONLY_FRAGMENT_ENTRY.to_owned(),
            color_formats: Vec::new(),
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 32,
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
            }]),
            textures: Vec::new(),
        };
        let request = OffscreenRenderRequest {
            pass: &pass,
            pipeline: &pipeline,
            source: REVIEWED_DEPTH_ONLY_SOURCE,
            initial: Vec::new(),
            resident: Vec::new(),
        };
        let planned = plan(&request, 0, 0).expect("the zero-colour depth pass plans");
        assert!(
            planned.attachments.is_empty(),
            "a pass with no colour attachment plans no colour readback"
        );
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(
            planned.depth.as_ref().map(|depth| depth.store),
            Some(Some(DepthStoreOp::Store))
        );

        // The module pairing is the whole allowlist: the depth pair's source is
        // refused for this shape, exactly as any other unreviewed module is.
        let mismatched = OffscreenRenderRequest {
            source: REVIEWED_DEPTH_SOURCE,
            ..request
        };
        let error =
            plan(&mismatched, 0, 0).expect_err("the pair module is not this shape's module");
        assert_eq!(error.slug, "native_render_source_not_reviewed");
    }

    /// The v47 stencil shape (`research/docs/23` §3.3, v47): one colour
    /// attachment beside a rail-owned `stencil8` surface the pass clears to
    /// zero, and the reviewed state the depth pair's stream plus one colour
    /// target compiles under.
    ///
    /// Runs without a device, so the shape rules and the three planned fields
    /// are the whole check here; the macOS CI compiles the pipeline state and
    /// the depth-stencil state this plan builds.
    #[test]
    fn plan_admits_a_stencil_pass_and_plans_the_reviewed_state() {
        let pass = stencil_pass();
        let pipeline = stencil_pipeline();
        let planned =
            plan(&stencil_request(&pass, &pipeline), 0, 0).expect("the stencil pass plans");
        // The colour attachment is the raster, the stencil surface the second
        // one beside it: the pair module is what this shape compiles.
        assert_eq!(planned.extent, [2, 2]);
        assert_eq!(
            planned.module_path,
            "conformance/shaders/depth_pair_4x4.metal"
        );
        assert_eq!(planned.vertex_entry, DEPTH_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, DEPTH_FRAGMENT_ENTRY);
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the stencil shape renders one colour attachment");
        };
        assert_eq!(attachment.format, RenderPixelFormat::Rgba8Unorm);
        assert_eq!(attachment.store, RenderStoreAction::Store);
        // The rail-owned surface's three planned fields, each the contract's
        // own value: the extent, the clear the load op carries, and the state
        // the draw tests and writes with. This fixture states no store action,
        // so the surface never leaves the pass and carries no landing
        // (`research/docs/23` §3.3, v49).
        let stencil = planned
            .stencil
            .expect("the pass opens the rail-owned stencil surface");
        assert_eq!(stencil.width, 2);
        assert_eq!(stencil.height, 2);
        assert_eq!(stencil.clear_value, Some(0));
        assert_eq!(stencil.test, Some(STENCIL_TEST));
        assert_eq!(stencil.store, None);
    }

    /// The contract's own rule, one level below the plan
    /// (`research/docs/23` §3.3, v47): stencil state with no stencil
    /// attachment has nothing to test, and core admission refuses it before
    /// this rail plans anything — the same shape the depth test states.
    #[test]
    fn plan_refuses_stencil_state_without_a_stencil_attachment() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.stencil_test = Some(STENCIL_TEST);
        assert_eq!(
            plan(&milestone_request(&pass, &milestone_pipeline(), None), 0, 0)
                .unwrap_err()
                .slug,
            "trace_contract_invalid"
        );
        // The contract states the refusal by name, so the rail's slug is the
        // one core admission uses rather than a second spelling of the same
        // fact.
        assert_eq!(
            pass.validate(),
            Err(ContractError::StencilTestWithoutAttachment)
        );
    }

    /// One pass whose stored multisampled depth surface states the resolve the
    /// device admits (`research/docs/23` §3.3, v57c).
    fn depth_resolving_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store: Some(DepthStoreOp::Store),
            identity: Some(RenderDepthIdentity {
                allocation_id: DEPTH_STORE_ALLOCATION,
                view_id: DEPTH_STORE_VIEW,
            }),
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Sample0,
        });
        pass
    }

    /// The v60 combined shape: a stored multisampled depth surface and a
    /// stored stencil surface, both resolved (`research/docs/23` §3.3, v60).
    fn combined_stencil_resolving_pass() -> RenderPassDescriptor {
        let mut pass = depth_resolving_pass();
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 2,
            height: 2,
            load: StencilLoadOp::clear(0),
            store: Some(StoreOp::Store),
            identity: Some(RenderStencilIdentity {
                allocation_id: AllocationId::new(941),
                view_id: ViewId::new(951),
            }),
        });
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Always,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::IncrementWrap,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        pass.stencil_resolve = Some(MultisampleStencilResolve {
            filter: StencilResolveFilter::Sample0,
        });
        pass
    }

    /// The v60 combined shape plans through the device mask, carrying both
    /// filters the encoder sets (`research/docs/23` §3.3, v60).
    #[test]
    fn plan_admits_the_combined_stencil_resolve_in_the_device_mask() {
        let pass = combined_stencil_resolving_pass();
        let pipeline = milestone_pipeline();
        let planned = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            STENCIL_RESOLVE_SAMPLE0_BIT,
        )
        .expect("the combined resolve the device admits plans");
        assert_eq!(planned.depth_resolve, Some(DepthResolveFilter::Sample0));
        assert_eq!(planned.stencil_resolve, Some(StencilResolveFilter::Sample0));
    }

    /// A stencil filter outside the device mask is refused by the per-filter
    /// question the capability snapshot answered (`research/docs/23` §3.3,
    /// v60).
    #[test]
    fn plan_refuses_a_stencil_resolve_filter_the_device_does_not_report() {
        let pass = combined_stencil_resolving_pass();
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a stencil resolve outside the mask is refused");
        assert_eq!(refused.slug, "render_stencil_resolve_filter_unsupported");
    }

    /// The v66 rail-owned combined pair plans: one combined surface carries
    /// both faces, neither is kept, and the plan carries no resolve — the
    /// write-then-test shape whose third triangle fails against the value the
    /// second one's depth failure wrote (`research/docs/23` §3.3, v66).
    #[test]
    fn plan_admits_the_rail_owned_combined_pair() {
        let mut pass = combined_stencil_resolving_pass();
        pass.depth_resolve = None;
        pass.stencil_resolve = None;
        {
            let depth = pass
                .depth
                .as_mut()
                .expect("the fixture opens a depth attachment");
            depth.load = DepthLoadOp::clear(1.0);
            depth.store = None;
            depth.identity = None;
        }
        {
            let stencil = pass
                .stencil
                .as_mut()
                .expect("the fixture opens a stencil attachment");
            stencil.store = None;
            stencil.identity = None;
        }
        pass.stencil_test = Some(StencilTest {
            compare: StencilCompare::Equal,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::IncrementWrap,
            pass_op: StencilOp::Keep,
            read_mask: 0xff,
            write_mask: 0xff,
            reference: 0,
        });
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0)
            .expect("the rail-owned combined pair plans");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.depth_resolve, None);
        assert_eq!(planned.stencil_resolve, None);
    }

    /// A pair that keeps one face while dropping the other is refused by name
    /// rather than read as either reviewed shape (`research/docs/23` §3.3,
    /// v66).
    #[test]
    fn plan_refuses_a_lopsided_combined_pair() {
        let mut pass = combined_stencil_resolving_pass();
        // The stencil half drops its store and its resolve while the depth
        // half keeps both: the one surface's two faces disagree.
        pass.stencil_resolve = None;
        {
            let stencil = pass
                .stencil
                .as_mut()
                .expect("the fixture opens a stencil attachment");
            stencil.store = None;
            stencil.identity = None;
        }
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a lopsided combined pair is refused");
        // The contract owns the agreement rule, so the plan reports its own
        // spelling before the rail's re-assert can run.
        assert_eq!(refused.slug, "trace_contract_invalid");
    }

    /// The v57c shape: a stored multisampled depth surface whose resolve the
    /// device mask admits plans, and the plan carries the filter the encoder
    /// sets on the depth attachment (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_admits_a_stored_multisampled_depth_resolve_in_the_device_mask() {
        let pass = depth_resolving_pass();
        let pipeline = milestone_pipeline();
        let planned = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect("a stored multisampled depth resolve the device admits plans");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.depth_resolve, Some(DepthResolveFilter::Sample0));
        let depth = planned
            .depth
            .expect("the resolving pass opens its depth surface");
        assert_eq!(depth.store, Some(DepthStoreOp::Store));
    }

    /// The device mask is the per-filter question the capability snapshot
    /// answered: a resolve whose filter the device does not report is refused
    /// by name before any Metal object exists, exactly as the Vulkan rail
    /// refuses it (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_refuses_a_depth_resolve_filter_the_device_does_not_report() {
        let mut pass = depth_resolving_pass();
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Min,
        });
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a Min resolve outside the Sample0-only mask is refused");
        assert_eq!(refused.slug, "render_depth_resolve_filter_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.fields.get("filter"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            refused.fields.get("modes"),
            Some(&FieldValue::Unsigned(u64::from(DEPTH_RESOLVE_SAMPLE0_BIT)))
        );
    }

    /// A stored multisampled depth surface without its resolve keeps the
    /// pre-v57c refusal: the surface's texels are only observable as the
    /// resolve's reduction, so keeping them without one stays impossible
    /// (`research/docs/23` §3.3, v57c). The core contract refuses the shape
    /// first, so the plan reports the contract's own spelling.
    #[test]
    fn plan_refuses_a_stored_multisampled_depth_without_a_resolve() {
        let mut pass = depth_resolving_pass();
        pass.depth_resolve = None;
        let pipeline = milestone_pipeline();
        let refused = plan(
            &milestone_request(&pass, &pipeline, None),
            DEPTH_RESOLVE_SAMPLE0_BIT,
            0,
        )
        .expect_err("a stored multisampled depth without a resolve is refused");
        assert_eq!(refused.slug, "trace_contract_invalid");
        assert_eq!(
            pass.validate(),
            Err(ContractError::MultisampleDepthStoreUnsupported)
        );
    }

    /// The v20 load increment (`research/docs/23` §3.1): a `LoadOp::DontCare`
    /// attachment plans as `MTLLoadAction::DontCare` with no preset bytes, so
    /// the encoder neither reads nor overwrites the pre-pass contents.
    #[test]
    fn plan_admits_a_dont_care_load_and_presets_nothing() {
        let pass = milestone_pass(LoadOp::DontCare);
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap();
        assert_eq!(planned.attachments.len(), 1);
        assert_eq!(planned.attachments[0].load, RenderLoadAction::DontCare);
        assert_eq!(
            planned.attachments[0].initial_bytes(),
            None,
            "a DontCare attachment carries no pre-pass bytes"
        );
    }

    /// A pass whose only attachment discards is refused by the contract before
    /// this rail plans anything: core admission's `AllRenderAttachmentsDiscarded`
    /// reaches `plan` as `trace_contract_invalid`, so a discarded pass can never
    /// present a blank readback as "landed correctly" (`research/docs/23` §3.6).
    #[test]
    fn plan_refuses_a_pass_whose_only_attachment_discards() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.color_attachments[0].store = StoreOp::DontCare;
        let pipeline = milestone_pipeline();
        let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
        assert_eq!(error.slug, "trace_contract_invalid");
        assert_eq!(error.class, ProviderErrorClass::Args);
    }

    /// The MRT contract admits up to `MAX_COLOR_ATTACHMENTS`, and this rail's
    /// reviewed modules cover one, two and the ceiling: a wider pass is refused
    /// instead of being rendered partially (wave3 R1).
    #[test]
    fn plan_refuses_a_pass_beyond_the_attachment_ceiling() {
        let maximum = usize::try_from(MAX_COLOR_ATTACHMENTS).unwrap();
        let mut pass = dual_pass();
        for _ in 2..(maximum + 1) {
            pass.color_attachments.push(pass.color_attachments[0]);
        }
        let mut pipeline = dual_pipeline();
        pipeline.color_formats = vec![AttachmentFormat::Rgba8Unorm; maximum + 1];
        let error = plan(&dual_request(&pass, &pipeline), 0, 0).unwrap_err();
        // Core admission owns the ceiling now that the rail's own maximum is
        // the contract's: the wider pass is refused as a trace-contract shape
        // before the rail's gate can see it (`attachment_count_unsupported`).
        assert_eq!(error.slug, "attachment_count_unsupported");
        // The rail's own gate stays as the fail-closed half for a directly
        // constructed request that skipped admission; its shape is asserted
        // directly so the two gates cannot drift apart.
        let refusal = mrt_attachment_count_refusal(maximum + 1);
        assert_eq!(refusal.slug, "render_mrt_attachment_count_unsupported");
        assert_eq!(refusal.class, ProviderErrorClass::Capability);
        assert_eq!(
            refusal.fields.get("attachments"),
            Some(&FieldValue::Unsigned((maximum + 1) as u64))
        );
        assert_eq!(
            refusal.fields.get("maximum"),
            Some(&FieldValue::Unsigned(maximum as u64))
        );
    }

    /// Every colour location renders into the pass's one raster, so two
    /// attachments of different extents are refused by name before the
    /// contract's viewport rule reports the same shape as a viewport
    /// disagreement.
    #[test]
    fn plan_refuses_attachments_that_disagree_about_the_extent() {
        let mut pass = dual_pass();
        pass.color_attachments[1].height = 1;
        let error = plan(&dual_request(&pass, &dual_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_extent_mismatch");
        assert_eq!(error.class, ProviderErrorClass::Args);
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(1))
        );
        assert_eq!(error.fields.get("width"), Some(&FieldValue::Unsigned(2)));
        assert_eq!(error.fields.get("height"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            error.fields.get("first_height"),
            Some(&FieldValue::Unsigned(2))
        );
    }

    /// A present action hands its one attachment on to the target texture, so a
    /// present pass with a second location is refused under the
    /// single-attachment slug instead of having location 1 dropped.
    #[test]
    fn plan_refuses_a_present_pass_with_two_attachments() {
        let mut pass = dual_pass();
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(7),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Undefined,
            },
            source: ViewId::new(7),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        let error = plan(&dual_request(&pass, &dual_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_count_unsupported");
        assert_eq!(
            error.fields.get("attachments"),
            Some(&FieldValue::Unsigned(2))
        );
        assert_eq!(error.fields.get("maximum"), Some(&FieldValue::Unsigned(1)));
    }

    /// A present pass renders into the provider-owned target alone: it opens no
    /// depth surface, so a pass that names one — stored or not — is refused
    /// instead of having the attachment silently dropped, the same refusal the
    /// Vulkan rail's present entry states (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_refuses_a_present_pass_with_a_depth_attachment() {
        for (store, expected) in [
            (None, "unstated"),
            (Some(DepthStoreOp::DontCare), "dontcare"),
            (Some(DepthStoreOp::Store), "store"),
        ] {
            let mut pass = milestone_present_pass();
            pass.depth = Some(RenderDepthAttachment {
                format: DepthFormat::Depth32Float,
                width: 2,
                height: 2,
                load: DepthLoadOp::clear(1.0),
                store,
                identity: match store {
                    Some(DepthStoreOp::Store) => Some(RenderDepthIdentity {
                        allocation_id: DEPTH_STORE_ALLOCATION,
                        view_id: DEPTH_STORE_VIEW,
                    }),
                    _ => None,
                },
            });
            pass.depth_test = Some(DepthTest {
                compare: CompareFunction::Less,
                write: true,
            });
            let pipeline = milestone_pipeline();
            let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
            assert_eq!(error.slug, "render_present_depth_unsupported");
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(
                error.fields.get("store"),
                Some(&FieldValue::Text(expected.to_owned()))
            );
            assert_eq!(
                error.detail.as_deref(),
                Some(
                    "the present rail renders into one provider-owned colour target and opens \
                     no depth surface"
                )
            );
        }
    }

    /// A present pass renders into the provider-owned target alone: it opens no
    /// stencil surface either, so a pass that names one — stored or not — is
    /// refused instead of having the attachment dropped, the same refusal the
    /// Vulkan rail's present entry states (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_refuses_a_present_pass_with_a_stencil_attachment() {
        for (store, expected) in [
            (None, "unstated"),
            (Some(StoreOp::DontCare), "dontcare"),
            (Some(StoreOp::Store), "store"),
        ] {
            let mut pass = milestone_present_pass();
            pass.stencil = Some(RenderStencilAttachment {
                format: StencilFormat::Stencil8,
                width: 2,
                height: 2,
                load: StencilLoadOp::clear(0),
                store,
                identity: match store {
                    Some(StoreOp::Store) => Some(RenderStencilIdentity {
                        allocation_id: STENCIL_STORE_ALLOCATION,
                        view_id: STENCIL_STORE_VIEW,
                    }),
                    _ => None,
                },
            });
            pass.stencil_test = Some(STENCIL_TEST);
            let pipeline = milestone_pipeline();
            let error = plan(&milestone_request(&pass, &pipeline, None), 0, 0).unwrap_err();
            assert_eq!(error.slug, "render_present_stencil_unsupported");
            assert_eq!(error.class, ProviderErrorClass::Capability);
            assert_eq!(error.phase, ProviderPhase::Resolve);
            assert_eq!(
                error.fields.get("store"),
                Some(&FieldValue::Text(expected.to_owned()))
            );
            assert_eq!(
                error.detail.as_deref(),
                Some(
                    "the present rail renders into one provider-owned colour target and opens \
                     no stencil surface"
                )
            );
        }
    }

    /// A present action beside a four-sample raster is admitted from v62 on:
    /// the encoder creates the n-sample surface and resolves it into the
    /// provider-owned present target with `storeAction = .multisampleResolve`,
    /// so the plan carries the raster and the single attachment unchanged
    /// (`research/docs/24` §3.5, v62).
    #[test]
    fn plan_admits_a_multisampled_present_pass_over_its_resolve_landing() {
        let mut pass = milestone_present_pass();
        // The reviewed multisample load shape opens the attachment from a
        // clear; the sentinel the present target carries is overwritten by the
        // resolve, exactly as the v62 fixture states.
        pass.color_attachments[0].load = LoadOp::Clear(sentinel());
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        let pipeline = milestone_pipeline();
        let planned = plan(&milestone_request(&pass, &pipeline, None), 0, 0)
            .expect("the four-sample present pass plans over its resolve landing");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        assert_eq!(planned.attachments.len(), 1);
        assert_eq!(planned.attachments[0].store, RenderStoreAction::Store);
    }

    #[test]
    fn plan_refuses_a_draw_shape_other_than_the_full_screen_triangle() {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.vertices = 6;
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        // The layout-free count above the milestone's three vertices is this
        // rail's own window (2026-09-19, census v45's `vertex_span` bucket):
        // the reviewed `vertex_id` module reads a three-entry position table by
        // index, so the refusal names the window rather than the draw shape.
        // The slug is the one core admission states for the same shape, and
        // this rail's snapshot keeps the bit undeclared.
        assert_eq!(error.slug, "render_vertex_count_window_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        // Below the triangle's three the *contract* refuses first, and that
        // refusal keeps its own name: this plan check is the widened arm's.
        let mut short = milestone_pass(LoadOp::Clear(sentinel()));
        short.vertices = 2;
        let error = plan_pass(&milestone_request(&short, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "draw_shape_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
    }

    #[test]
    fn load_requires_the_previous_texels_and_clear_refuses_them() {
        let pass = milestone_pass(LoadOp::Load);
        let pipeline = milestone_pipeline();
        let error = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
        assert_eq!(error.class, ProviderErrorClass::Args);

        let previous = [0x11_u8; 16];
        let planned = plan_pass(&milestone_request(&pass, &pipeline, Some(&previous))).unwrap();
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert_eq!(attachment.load, RenderLoadAction::Load);
        assert_eq!(attachment.initial_bytes(), Some(previous.as_slice()));

        let short = [0x11_u8; 15];
        let error = plan_pass(&milestone_request(&pass, &pipeline, Some(&short))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");

        let cleared = milestone_pass(LoadOp::Clear(sentinel()));
        let error =
            plan_pass(&milestone_request(&cleared, &pipeline, Some(&previous))).unwrap_err();
        assert_eq!(error.slug, "render_attachment_initial_mismatch");
    }

    /// The falsifiability rule `research/docs/23` §1.3 states: a pass that never
    /// ran has to be distinguishable from one that did, in every channel.
    #[test]
    fn a_cleared_attachment_cannot_imitate_the_fragment_output() {
        let pass = milestone_pass(LoadOp::Clear(sentinel()));
        let pipeline = milestone_pipeline();
        let planned = plan_pass(&milestone_request(&pass, &pipeline, None)).unwrap();
        let RenderLoadAction::Clear(components) = planned.attachments[0].load else {
            panic!("the milestone clears its attachment");
        };
        for channel in 0..4 {
            assert_ne!(
                components[channel], EXPECTED_TEXEL_COMPONENTS[channel],
                "clear channel {channel} must differ from the fragment output"
            );
        }
    }

    /// The one compute pass of the trace-path fixture: a read-only declaration
    /// of the attachment's own view.
    ///
    /// Core admission resolves every render attachment against the views the
    /// trace declares, so a render-bearing trace always carries one of these,
    /// and it is what makes the attachment's landing rail exist (see
    /// `ComputeTrace::serial_resources`).
    fn declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: ViewId::new(7),
                metal_binding: 0,
                allocation_id: AllocationId::new(9),
                offset: 0,
                length: 16,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0xfe; 16]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The compute registration `declaration_pass` names. Its contract reflects
    /// a read-only 16-byte binding, i.e. exactly the extent the 2x2 attachment
    /// restates, and the footprint proof is static so admission can compare it
    /// with the view.
    fn declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-wiring-fixture", vec![7]).unwrap(),
                entry_name: "declares_the_attachment_view".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![BufferBindingContract {
                    metal_binding: 0,
                    access: BufferAccess::Read,
                    footprint: FootprintProof::Static { max_bytes: 16 },
                }],
                texture_bindings: Vec::new(),
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry a render registration hands back, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it: the id the
    /// render pass names, the reviewed vertex entry, the most permissive
    /// exact-thread contract (nothing reads a render entry as a compute
    /// contract) and the reviewed render contract in the `render` half core
    /// admission compares the attachment against.
    fn render_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(3),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-wiring-fixture", vec![3]).unwrap(),
                entry_name: VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(RenderPipelineContract {
                stage_buffers: Vec::new(),
                vertex_entry: VERTEX_ENTRY.to_owned(),
                fragment_entry: FRAGMENT_ENTRY.to_owned(),
                color_formats: vec![AttachmentFormat::Rgba8Unorm],
                vertex_layout: VertexLayout::None,
                textures: Vec::new(),
            }),
        }
    }

    /// The milestone's trace and the resource namespace it needs: the
    /// declaration pass, then the render pass under test.
    fn milestone_trace(load: LoadOp) -> (ComputeTrace, ResourceTableSnapshot) {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(1),
            pipelines: vec![declaration_pipeline(), render_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                TracePass::Render(milestone_pass(load)),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: DeviceEpoch::new(3),
                size: 16,
            })
            .unwrap();
        (trace, resources)
    }

    /// The milestone trace's render pass, for the value-level window checks.
    fn render_pass_mut(trace: &mut ComputeTrace) -> &mut RenderPassDescriptor {
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the milestone trace carries one render pass");
        };
        pass
    }

    /// The milestone's present pass: the same 2x2 attachment, but with a
    /// `Load` that keeps the present target's sentinel and a present action
    /// handing that target on (`research/docs/24` §3.1, §3.5 shape one).
    fn milestone_present_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Load);
        pass.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(7),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Sentinel(vec![0xfe, 0xfe, 0xfe, 0xfe]),
            },
            source: ViewId::new(7),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        pass
    }

    /// The milestone's present-bearing trace and resource namespace.
    fn milestone_present_trace() -> (ComputeTrace, ResourceTableSnapshot) {
        let (mut trace, resources) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        *pass = milestone_present_pass();
        (trace, resources)
    }

    /// The stored depth attachment's own resource (`research/docs/23` §3.3,
    /// v43): allocation 940 and view 950, covering the milestone pass's 2x2
    /// `depth32float` extent — four texels of four bytes, the same length the
    /// colour attachment's landing view has.
    const DEPTH_STORE_ALLOCATION: AllocationId = AllocationId::new(940);
    const DEPTH_STORE_VIEW: ViewId = ViewId::new(950);
    /// The stored depth surface's byte length: 2x2 texels of a four-byte
    /// `depth32float` texel each.
    const DEPTH_STORE_BYTES: u64 = 16;

    /// The depth the v43 fixture's draw leaves behind: `0.5` as
    /// `depth32float`'s little-endian bytes. The clear below writes `1.0`
    /// (`00 00 80 3f`), so a pass that never stored its depth surface cannot
    /// read back as one that did.
    const DEPTH_STORE_TEXEL: [u8; 4] = [0x00, 0x00, 0x00, 0x3f];

    /// The compute declaration of the stored depth surface's landing view: one
    /// read-only view over the whole attachment, the same shape the colour
    /// attachment's own declaration pass has.
    ///
    /// A stored depth identity has to be declared by the trace: core admission
    /// resolves it against the same declarations a colour attachment resolves
    /// against and refuses an undeclared one (`AttachmentViewUnknown`), and the
    /// declaring binding is what puts the view into the serial pool — its
    /// access is what `serial_resources` upgrades to a write — which is where
    /// [`plan_trace`] resolves the landing from (`research/docs/23` §3.3, v43).
    fn depth_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: DEPTH_STORE_VIEW,
                metal_binding: 0,
                allocation_id: DEPTH_STORE_ALLOCATION,
                offset: 0,
                length: DEPTH_STORE_BYTES,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0x80; DEPTH_STORE_BYTES as usize]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The v43 trace: the milestone pass plus a depth attachment of the same
    /// 2x2 extent, and the compute declaration of its landing view.
    ///
    /// `store` is the whole v43 axis. `Some(Store)` is the shape whose texels
    /// survive the pass and therefore need a landing; `None` is the pre-v43
    /// shape and `Some(DontCare)` the explicit discard, both of which keep the
    /// surface rail-owned and land nothing.
    fn depth_store_trace(store: Option<DepthStoreOp>) -> ComputeTrace {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.depth = Some(RenderDepthAttachment {
            format: DepthFormat::Depth32Float,
            width: 2,
            height: 2,
            load: DepthLoadOp::clear(1.0),
            store,
            identity: match store {
                Some(DepthStoreOp::Store) => Some(RenderDepthIdentity {
                    allocation_id: DEPTH_STORE_ALLOCATION,
                    view_id: DEPTH_STORE_VIEW,
                }),
                _ => None,
            },
        });
        pass.depth_test = Some(DepthTest {
            compare: CompareFunction::Less,
            write: true,
        });
        // The declaration comes before the render pass, exactly as the colour
        // attachment's does: the pool it feeds is the compute rail's binding
        // set and every pass of it runs before the draw.
        trace
            .passes
            .insert(1, TracePass::Compute(depth_declaration_pass()));
        trace
    }

    /// The capability snapshot the macOS provider builds, with the render bits
    /// taken from the value under test, the stage-buffer pair from the rail's
    /// own spelling and the compute bits from `native.rs`.
    fn capabilities(bits: &RenderCapabilityBits) -> ProviderCapabilities {
        capabilities_with(
            bits,
            &vertex_input_capability_bits(),
            &stage_buffer_capability_bits(),
        )
    }

    /// The same snapshot with the stage-buffer pair closed, so a test can
    /// construct the pre-flip declaration (`native.rs` takes the pair from
    /// [`stage_buffer_capability_bits`], which is the value the flip moved).
    fn capabilities_before_the_stage_buffer_flip(
        bits: &RenderCapabilityBits,
    ) -> ProviderCapabilities {
        let mut capabilities = capabilities(bits);
        capabilities.supports_render_stage_buffers = false;
        capabilities.max_render_stage_buffers = 0;
        // The folded shape's bit arrived with the same face
        // (`research/docs/23` §3.3, E-TX9), so the pre-flip declaration keeps
        // it closed too: a snapshot that cannot fill a stage-buffer slot
        // cannot execute a pair whose two stages read one Metal index.
        capabilities.supports_render_stage_buffer_namespace_split = false;
        capabilities
    }

    /// The same snapshot with the vertex-input bits and the stage-buffer pair
    /// spelled out, so a test can construct the pre-flip declaration
    /// (`native.rs` takes all three from the rail).
    fn capabilities_with(
        bits: &RenderCapabilityBits,
        vertex: &VertexInputCapabilityBits,
        stage_buffers: &StageBufferCapabilityBits,
    ) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_render_stage_buffers: stage_buffers.supports_render_stage_buffers,
            max_render_stage_buffers: stage_buffers.max_render_stage_buffers,
            // The native rail's reviewed pair binds one slot per stage
            // (`research/docs/23` §83), so the snapshot declares no per-stage
            // window and keeps the list bound as the whole rule (§117 E-SB2).
            max_render_stage_buffers_per_stage: stage_buffers.max_render_stage_buffers_per_stage,
            // The native rail declares no one-dimensional sampled window
            // (2026-09-19, census b10's `texture_shape` bucket): no Apple-side
            // reading states the Metal 1D equivalence this generation accepts,
            // so the shape keeps its refusal by name and the field stays at the
            // arm's fail-closed default.
            max_render_texture_dimension_1d: 0,
            max_render_texture_dimension_3d: 0,
            supports_render_stage_buffer_namespace_split: stage_buffers
                .supports_render_stage_buffer_namespace_split,
            // The whole-binding arm rides the same face and keeps its
            // fail-closed default on this rail (`research/docs/23` §3.3,
            // E-SB3).
            supports_render_stage_buffer_binding_range: stage_buffers
                .supports_render_stage_buffer_binding_range,
            // The native rail's reviewed MSL modules spell one `constexpr
            // sampler` in the normalized space, and the rail refuses a runtime
            // `[[sampler(n)]]` argument by name (`render_runtime_sampler_unsupported`),
            // so the texel space (2026-09-19, census v43's `texture_state`
            // axis) has no module behind it here either: the bit keeps its
            // default and admission refuses such a pass before the rail.
            supports_render_pixel_coordinate_sampler: bits.supports_render_pixel_coordinate_sampler,
            max_passes: 8,
            supports_threads_exact: true,
            supports_threadgroups: false,
            supports_serial: true,
            supports_concurrent: false,
            max_local_size: [1024, 1024, 1024],
            max_invocations: 1024,
            max_group_count: [1024, 1024, 1024],
            max_storage_buffer_descriptors: 31,
            supports_compute_texture_sampling: false,
            max_compute_textures: 0,
            supported_compute_texture_formats: Vec::new(),
            max_buffer_range: 1024,
            max_push_constant_bytes: 0,
            alias_mode: AliasMode::DistinctViews,
            storage_modes: vec![StorageMode::OwnedBytes],
            host_readback: true,
            submit_only: false,
            supports_render_passes: bits.supports_render_passes,
            max_color_attachments: bits.max_color_attachments,
            max_attachment_dimension: bits.max_attachment_dimension,
            supported_color_formats: bits.supported_color_formats.clone(),
            supports_render_fragment_output_superset: bits.supports_render_fragment_output_superset,
            // The 16-bit shader capability pair keeps the rail's own
            // fail-closed answer (2026-09-20, census v48's LPF pipeline): no
            // reviewed MSL module narrows a float to `half`, so the pair this
            // face names has no module behind it here.
            supports_render_half_capabilities: bits.supports_render_half_capabilities,
            max_vertex_buffers: vertex.max_vertex_buffers,
            supported_vertex_formats: vertex.supported_vertex_formats.clone(),
            supported_index_formats: vertex.supported_index_formats.clone(),
            supports_render_vertex_interface_superset: vertex
                .supports_render_vertex_interface_superset,
            supports_render_vertex_count_above_triangle: vertex
                .supports_render_vertex_count_above_triangle,
            supports_render_instancing: false,
            max_render_instances: 0,
            // The test snapshot spells the pre-flip shape out for the same
            // reason the instancing pair above does: a test constructs the
            // "cannot multisample" declaration and asserts core admission
            // refuses the multisampled pass (`research/docs/23` §3.3, v51).
            supports_render_multisample: false,
            max_render_sample_count: 0,
            supports_render_depth_resolve: false,
            depth_resolve_modes: 0,
            supports_render_stencil_resolve: false,
            stencil_resolve_modes: 0,
            // The test snapshot spells the pre-flip shape out for the same
            // reason the multisample pair above does: a test constructs the
            // "cannot sample render-side textures" declaration and asserts
            // core admission refuses the texture-bearing pass
            // (`research/docs/23` §3.3, v70).
            supports_render_texture_sampling: false,
            max_render_textures: 0,
            max_render_textures_per_stage: 0,
            supported_render_texture_formats: Vec::new(),
            supports_render_texture_gathered_extent: false,
            supports_render_texture_gathered_extent_no_copy: false,
            // The kept-frame landing entry (`research/docs/23` §115 之后的增量，
            // E-TX14/R4b) is refused by this rail's own plan gate, so the test
            // snapshot spells the fail-closed default out beside the landing
            // view's bit.
            supports_render_kept_frame_landing: false,
            supports_render_attachment_landing_view: false,
            // The pass-entry snapshot arm (`research/docs/23` §118, E-TX15)
            // is refused by this rail's own plan gate, so the test snapshot
            // spells the fail-closed default out beside the two bits above.
            supports_render_pass_entry_snapshot: false,
            supports_presentation: bits.supports_presentation,
            max_present_targets: bits.max_present_targets,
            supported_present_modes: bits.supported_present_modes.clone(),
            max_present_image_count: bits.max_present_image_count,
            supports_heaps: false,
            max_heap_bytes: 0,
            supported_heap_storage_modes: Vec::new(),
            supports_heap_aliasing: false,
            supports_indirect_command_buffers: false,
            max_indirect_commands: 0,
            supported_indirect_commands: Vec::new(),
        }
    }

    /// The registrations `plan_trace` resolves the trace's pipeline ids against.
    fn milestone_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(PipelineId::new(3), milestone_pipeline())])
    }

    /// The vertex-input fixture's identities: one 32-byte vertex stream
    /// (`view` 21 inside allocation 12) and one 12-byte index buffer (`view` 22
    /// inside allocation 13). Both are separate whole-allocation views, the
    /// shape the reviewed fixture declares, so the rail's footprint proof is
    /// stated against the bytes the trace itself carries.
    const QUAD_VERTEX_VIEW: ViewId = ViewId::new(21);
    const QUAD_VERTEX_ALLOCATION: AllocationId = AllocationId::new(12);
    const QUAD_INDEX_VIEW: ViewId = ViewId::new(22);
    const QUAD_INDEX_ALLOCATION: AllocationId = AllocationId::new(13);
    /// The reviewed quad pipeline's id: one counter with the render rail's other
    /// registration, so 3 stays the milestone's triangle.
    const QUAD_PIPELINE: PipelineId = PipelineId::new(4);
    /// The reviewed dual pipeline's id, one counter further.
    const DUAL_PIPELINE: PipelineId = PipelineId::new(5);

    /// The four NDC corners of the reviewed quad, as the 32 little-endian
    /// `float32x2` bytes the stream holds.
    fn quad_vertex_bytes() -> Vec<u8> {
        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let mut bytes = Vec::with_capacity(32);
        for corner in corners {
            for component in corner {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
        bytes
    }

    /// The six `uint16` indices that select the quad's two triangles: (0,1,2)
    /// and (2,1,3), i.e. 12 little-endian bytes covering all four vertices.
    fn quad_index_bytes() -> Vec<u8> {
        let indices: [u16; 6] = [0, 1, 2, 2, 1, 3];
        indices
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<u8>>()
    }

    /// The reviewed vertex stream as the render pass declares it: a read-only
    /// view at binding 0 whose `source` carries the four NDC corners
    /// (`RenderPassDescriptor::vertex_buffers`).
    fn quad_vertex_view() -> BufferView {
        let bytes = quad_vertex_bytes();
        BufferView {
            view_id: QUAD_VERTEX_VIEW,
            metal_binding: 0,
            allocation_id: QUAD_VERTEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the fixture's lengths fit u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        }
    }

    /// The reviewed index buffer: the same shape one level over, with the
    /// `metal_binding` an index binding always carries
    /// (`IndexBufferBinding::validate_shape` refuses any other value).
    fn quad_index_view() -> BufferView {
        let bytes = quad_index_bytes();
        BufferView {
            view_id: QUAD_INDEX_VIEW,
            metal_binding: 0,
            allocation_id: QUAD_INDEX_ALLOCATION,
            offset: 0,
            length: u64::try_from(bytes.len()).expect("the fixture's lengths fit u64"),
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(bytes),
        }
    }

    /// The reviewed indexed pipeline contract: one `float32x2` stream,
    /// `[[stage_in]]` positions.
    fn quad_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: QUAD_VERTEX_ENTRY.to_owned(),
            fragment_entry: FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
                step: VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            }]),
            textures: Vec::new(),
        }
    }

    /// The reviewed indexed pass: the same 2x2 attachment, six indices over the
    /// bound stream.
    fn quad_pass() -> RenderPassDescriptor {
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
            pipeline: QUAD_PIPELINE,
            color_attachments: vec![RenderAttachment {
                view_id: ViewId::new(7),
                allocation_id: AllocationId::new(9),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                load: LoadOp::Clear(sentinel()),
                store: StoreOp::Store,
            }],
            viewport: [0, 0, 2, 2],
            scissor: None,
            vertices: 6,
            vertex_buffers: vec![quad_vertex_view()],
            indices: Some(IndexBufferBinding {
                view: quad_index_view(),
                format: IndexFormat::Uint16,
            }),
            instance_count: 1,
            textures: Vec::new(),
            present: None,
        }
    }

    /// The compute pass that declares the attachment view and both streams.
    ///
    /// The attachment lands through view 7, which only a compute declaration
    /// puts into the serial pool, so this pass is what makes the landing view
    /// real. The two streams are declared here as well — with the pass's own
    /// stream bytes, which core admission requires to agree with the render
    /// pass's views for the same identity — even though the render rail no
    /// longer needs a declaration to read them.
    fn quad_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(7),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(9),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfe; 16]),
                },
                BufferView {
                    metal_binding: 1,
                    ..quad_vertex_view()
                },
                BufferView {
                    metal_binding: 2,
                    ..quad_index_view()
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The declaration pass's compute registration: three read-only bindings
    /// whose static footprints are exactly the views' own lengths.
    fn quad_declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-vertex-fixture", vec![11]).unwrap(),
                entry_name: "declares_the_render_views".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![
                    BufferBindingContract {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 1,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 32 },
                    },
                    BufferBindingContract {
                        metal_binding: 2,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 12 },
                    },
                ],
                texture_bindings: Vec::new(),
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry of the quad registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn quad_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: QUAD_PIPELINE,
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-vertex-fixture", vec![4]).unwrap(),
                entry_name: QUAD_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(quad_pipeline()),
        }
    }

    /// The trace-table entry's compute half: the most permissive exact-thread
    /// contract, which no render pass reads as a compute contract.
    fn render_table_contract() -> PipelineContract {
        PipelineContract {
            dispatch_kind: DispatchKind::ThreadsExact,
            required_local_size: None,
            fixed_grid: None,
            push_constant_offset: 0,
            push_constant_bytes: 0,
            buffer_bindings: Vec::new(),
            texture_bindings: Vec::new(),
            shader_capabilities: Vec::new(),
            translator_revision: None,
        }
    }

    /// The vertex-input fixture's trace and resource namespace: the declaration
    /// pass, then the indexed render pass.
    fn quad_trace() -> (ComputeTrace, ResourceTableSnapshot) {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(2),
            pipelines: vec![quad_declaration_pipeline(), quad_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(quad_declaration_pass()),
                TracePass::Render(quad_pass()),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        for (allocation, size) in [
            (AllocationId::new(9), 16),
            (QUAD_VERTEX_ALLOCATION, 32),
            (QUAD_INDEX_ALLOCATION, 12),
        ] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(3),
                    size,
                })
                .unwrap();
        }
        (trace, resources)
    }

    /// The registrations `plan_trace` resolves the vertex-input trace against.
    fn quad_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(QUAD_PIPELINE, quad_pipeline())])
    }

    /// The request the trace path builds for the reviewed indexed pass.
    fn quad_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_VERTEX_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        }
    }

    /// The alignment the native rail's no-copy mapping needs from an owner
    /// window: `newBufferWithBytesNoCopy:` maps whole pages, so the reservation
    /// and its base address are held to this value. The tests spell it once
    /// instead of reading the platform's page size, so every host says the same
    /// thing.
    const OWNER_ALIGNMENT: u64 = 4096;

    /// A page-aligned owner mapping for the no-copy arm, freed when it drops.
    ///
    /// The owner of a window is outside the provider: this stands in for the
    /// owner's own pages, which the rail must read rather than copy.
    struct OwnerPages {
        pointer: *mut u8,
        length: usize,
        layout: std::alloc::Layout,
    }

    impl OwnerPages {
        /// One zeroed `length`-byte mapping at `alignment`.
        fn new(length: usize, alignment: usize) -> Self {
            let layout = std::alloc::Layout::from_size_align(length, alignment)
                .expect("the owner layout is well formed");
            // SAFETY: the layout is non-zero, and `Drop` frees the same layout.
            let pointer = unsafe { std::alloc::alloc(layout) };
            assert!(!pointer.is_null(), "page-aligned owner allocation failed");
            // SAFETY: the mapping covers `length` writable bytes.
            unsafe { std::ptr::write_bytes(pointer, 0, length) };
            Self {
                pointer,
                length,
                layout,
            }
        }

        /// The owner's address, as the no-copy registry is handed it.
        fn as_ptr(&self) -> usize {
            self.pointer as usize
        }

        /// Write the mapping's leading bytes, as an owner that writes its pages
        /// before or after the import does.
        fn write(&mut self, bytes: &[u8]) {
            assert!(
                bytes.len() <= self.length,
                "an owner write fits its own mapping"
            );
            // SAFETY: the mapping covers `bytes.len()` writable bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.pointer, bytes.len());
            }
        }

        /// The leading bytes of the mapping, as the rail's proofs read them.
        fn leading(&self, length: usize) -> &[u8] {
            // SAFETY: the mapping covers `length` readable bytes.
            unsafe { std::slice::from_raw_parts(self.pointer, length) }
        }
    }

    impl Drop for OwnerPages {
        fn drop(&mut self) {
            // SAFETY: the pointer came from `alloc` with this exact layout, and
            // the owner frees its pages only after the import is released.
            unsafe { std::alloc::dealloc(self.pointer, self.layout) };
        }
    }

    /// One lease over `allocation`'s first `length` bytes.
    fn lease_registration(
        lease_id: LeaseId,
        allocation: AllocationId,
        length: u64,
        epoch: DeviceEpoch,
    ) -> LeaseReservation {
        LeaseReservation {
            lease: BufferLease {
                lease_id,
                allocation_id: allocation,
                owner_epoch: epoch,
            },
            offset: 0,
            length,
        }
    }

    /// The reviewed indexed pass with both of its inputs naming leases.
    fn leased_quad_pass(vertex: BufferSource, index: BufferSource) -> RenderPassDescriptor {
        let mut pass = quad_pass();
        pass.vertex_buffers[0].source = vertex;
        pass.indices
            .as_mut()
            .expect("the fixture is indexed")
            .view
            .source = index;
        pass
    }

    /// The vertex-input fixture's stream views, with the sources a lease test
    /// hands them.
    ///
    /// The declaration pass and the render pass name the same two view ids, and
    /// a serial trace refuses two declarations of one view that disagree about
    /// its bytes (`SerialBufferRebinding`), so both passes spell the same
    /// source — which is exactly the invariant the trace contract states.
    fn lease_the_quad_trace(trace: &mut ComputeTrace, vertex: BufferSource, index: BufferSource) {
        for pass in &mut trace.passes {
            match pass {
                TracePass::Compute(pass) => {
                    for view in &mut pass.buffers {
                        if view.view_id == QUAD_VERTEX_VIEW {
                            view.source = vertex.clone();
                        } else if view.view_id == QUAD_INDEX_VIEW {
                            view.source = index.clone();
                        }
                    }
                }
                TracePass::Landing(_) => {}
                TracePass::Render(pass) => {
                    for view in &mut pass.vertex_buffers {
                        if view.view_id == QUAD_VERTEX_VIEW {
                            view.source = vertex.clone();
                        }
                    }
                    if let Some(indices) = &mut pass.indices {
                        indices.view.source = index.clone();
                    }
                }
            }
        }
    }

    /// A snapshot carrying the quad's two allocations at whole-page size, plus
    /// the two reservations drawn from their first bytes.
    ///
    /// The no-copy arm maps the reservation, not the view, so the fixture's
    /// reservations are page-sized exactly as the production rail's are — at
    /// the alignment the mapping itself needs, which a device test reads from
    /// the device.
    fn owner_resources(
        epoch: DeviceEpoch,
        vertex: LeaseReservation,
        index: LeaseReservation,
        alignment: u64,
    ) -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        for allocation in [QUAD_VERTEX_ALLOCATION, QUAD_INDEX_ALLOCATION] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: epoch,
                    size: alignment,
                })
                .expect("the fixture allocation is well formed");
        }
        resources
            .insert_lease(vertex)
            .expect("the vertex reservation covers its view");
        resources
            .insert_lease(index)
            .expect("the index reservation covers its view");
        resources
    }

    /// The stencil fixture's reviewed state: the v47 fixture's own values —
    /// `equal 0` with reference 0, both failure operations `keep`, and
    /// `increment_wrap` on success, over the full read and write masks
    /// (`research/docs/23` §3.3, v47). Spelled once so the pass and the
    /// assertion cannot drift.
    const STENCIL_TEST: StencilTest = StencilTest {
        compare: StencilCompare::Equal,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::Keep,
        pass_op: StencilOp::IncrementWrap,
        read_mask: 0xff,
        write_mask: 0xff,
        reference: 0,
    };

    /// The v47 fixture's pass shape: the milestone's one 2x2 `rgba8_unorm`
    /// attachment, the reviewed pair stream with six indices over it, and the
    /// rail-owned `stencil8` surface cleared to zero that the state above
    /// tests (`research/docs/23` §3.3, v47).
    ///
    /// The stream is the depth pair's two-attribute layout — a `float32x3`
    /// position and a `float32x4` tint — which is the shape the pair module's
    /// vertex stage reads; the vertex bytes themselves are the fixture's
    /// business, not the plan's, so a zeroed 192-byte view (six records of 32)
    /// stands in for them exactly as the zero-colour depth fixture does.
    fn stencil_pass() -> RenderPassDescriptor {
        let mut pass = milestone_pass(LoadOp::Clear(sentinel()));
        pass.vertex_buffers = vec![BufferView {
            view_id: ViewId::new(41),
            metal_binding: 0,
            allocation_id: AllocationId::new(43),
            offset: 0,
            length: 192,
            access: BufferAccess::Read,
            attribute_stride: None,
            source: BufferSource::OwnedBytes(vec![0; 192]),
        }];
        pass.indices = Some(IndexBufferBinding {
            view: quad_index_view(),
            format: IndexFormat::Uint16,
        });
        pass.vertices = 6;
        pass.stencil = Some(RenderStencilAttachment {
            format: StencilFormat::Stencil8,
            width: 2,
            height: 2,
            load: StencilLoadOp::clear(0),
            // The v47 shape states no store and names no landing: the surface
            // is the pass's own and disappears with it (`research/docs/23`
            // §3.3, v49).
            store: None,
            identity: None,
        });
        pass.stencil_test = Some(STENCIL_TEST);
        pass
    }

    /// The reviewed pair pipeline contract the stencil fixture declares: the
    /// depth pair's two stages compiled against one `rgba8_unorm` attachment,
    /// with the two-attribute stream layout its vertex stage was written for
    /// (`research/docs/23` §3.3, v47).
    fn stencil_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: DEPTH_VERTEX_ENTRY.to_owned(),
            fragment_entry: DEPTH_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 32,
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
            }]),
            textures: Vec::new(),
        }
    }

    /// The request the trace path builds for the reviewed stencil pass: the
    /// pair module's bytes and one previous-bytes slot per colour attachment.
    fn stencil_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_DEPTH_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        }
    }

    /// The stored stencil attachment's own resource (`research/docs/23` §3.3,
    /// v49): allocation 960 and view 970, covering the stencil fixture's 2x2
    /// `stencil8` extent — four one-byte texels.
    const STENCIL_STORE_ALLOCATION: AllocationId = AllocationId::new(960);
    const STENCIL_STORE_VIEW: ViewId = ViewId::new(970);
    /// The stored stencil surface's byte length: 2x2 texels of a single-byte
    /// `stencil8` texel each.
    const STENCIL_STORE_BYTES: u64 = 4;
    /// The stencil value the v49 fixture's draw leaves behind: `1`, which is
    /// the reviewed state's `increment_wrap` on success. The pass clears the
    /// surface to `0`, so a pass that never stored its stencil surface cannot
    /// read back as one that did.
    const STENCIL_STORE_TEXEL: u8 = 0x01;

    /// The compute declaration of the stored stencil surface's landing view:
    /// one read-only view over the whole attachment, the depth declaration's
    /// own shape one byte wide.
    ///
    /// A stored stencil identity has to be declared by the trace: core
    /// admission resolves it against the same declarations a colour attachment
    /// resolves against and refuses an undeclared one (`AttachmentViewUnknown`),
    /// and the declaring binding is what puts the view into the serial pool —
    /// its access is what `serial_resources` upgrades to a write — which is
    /// where [`plan_trace`] resolves the landing from (`research/docs/23` §3.3,
    /// v49).
    fn stencil_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![BufferView {
                view_id: STENCIL_STORE_VIEW,
                metal_binding: 0,
                allocation_id: STENCIL_STORE_ALLOCATION,
                offset: 0,
                length: STENCIL_STORE_BYTES,
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(vec![0x80; STENCIL_STORE_BYTES as usize]),
            }],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The v49 trace: the stencil fixture's reviewed pass with the store action
    /// under test, and the compute declaration of its landing view.
    ///
    /// `store` is the whole v49 axis. `Some(Store)` is the shape whose texels
    /// survive the pass and therefore need a landing; `None` is the pre-v49
    /// shape and `Some(DontCare)` the explicit discard, both of which keep the
    /// surface rail-owned and land nothing.
    fn stencil_store_trace(store: Option<StoreOp>) -> ComputeTrace {
        let mut pass = stencil_pass();
        let stencil = pass
            .stencil
            .as_mut()
            .expect("the stencil fixture declares a stencil attachment");
        stencil.store = store;
        stencil.identity = match store {
            Some(StoreOp::Store) => Some(RenderStencilIdentity {
                allocation_id: STENCIL_STORE_ALLOCATION,
                view_id: STENCIL_STORE_VIEW,
            }),
            _ => None,
        };
        ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(4),
            pipelines: vec![declaration_pipeline(), stencil_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                // The stencil declaration comes before the render pass, exactly
                // as the colour attachment's does: the pool it feeds is the
                // compute rail's binding set and every pass of it runs before
                // the draw.
                TracePass::Compute(stencil_declaration_pass()),
                TracePass::Render(pass),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        }
    }

    /// The trace-table entry of the stencil registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn stencil_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(3),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-stencil-fixture", vec![3]).unwrap(),
                entry_name: DEPTH_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(stencil_pipeline()),
        }
    }

    /// The registrations `plan_trace` resolves the stencil trace against.
    fn stencil_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(PipelineId::new(3), stencil_pipeline())])
    }

    /// The reviewed dual pipeline contract: the indexed vertex stage plus the
    /// two-location fragment stage, compiled against two `rgba8_unorm`
    /// attachments.
    fn dual_pipeline() -> RenderPipelineContract {
        RenderPipelineContract {
            stage_buffers: Vec::new(),
            vertex_entry: QUAD_VERTEX_ENTRY.to_owned(),
            fragment_entry: DUAL_FRAGMENT_ENTRY.to_owned(),
            color_formats: vec![AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm],
            vertex_layout: VertexLayout::Buffers(vec![VertexBufferLayout {
                stride: 8,
                step: VertexStep::PerVertex,
                attributes: vec![VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            }]),
            textures: Vec::new(),
        }
    }

    /// The reviewed dual pass: the indexed quad's vertex stream and index
    /// buffer, drawn into two 2x2 `rgba8_unorm` attachments.
    fn dual_pass() -> RenderPassDescriptor {
        let mut pass = quad_pass();
        pass.pipeline = DUAL_PIPELINE;
        pass.color_attachments.push(RenderAttachment {
            view_id: ViewId::new(8),
            allocation_id: AllocationId::new(10),
            format: AttachmentFormat::Rgba8Unorm,
            width: 2,
            height: 2,
            load: LoadOp::Clear(sentinel()),
            store: StoreOp::Store,
        });
        pass
    }

    /// The request the trace path builds for the reviewed dual pass.
    fn dual_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_DUAL_SOURCE,
            initial: vec![None, None],
            resident: Vec::new(),
        }
    }

    /// The compute pass that declares both attachment views and both streams.
    ///
    /// Each attachment lands through its own view (7 in allocation 9, 8 in
    /// allocation 10), which only a compute declaration puts into the serial
    /// pool, so this pass is what makes both landing rails real. The two
    /// streams are declared here as well, the same shape the quad fixture
    /// uses.
    fn dual_declaration_pass() -> ComputePass {
        ComputePass {
            pipeline: PipelineId::new(11),
            buffers: vec![
                BufferView {
                    view_id: ViewId::new(7),
                    metal_binding: 0,
                    allocation_id: AllocationId::new(9),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfe; 16]),
                },
                BufferView {
                    view_id: ViewId::new(8),
                    metal_binding: 1,
                    allocation_id: AllocationId::new(10),
                    offset: 0,
                    length: 16,
                    access: BufferAccess::Read,
                    attribute_stride: None,
                    source: BufferSource::OwnedBytes(vec![0xfd; 16]),
                },
                BufferView {
                    metal_binding: 2,
                    ..quad_vertex_view()
                },
                BufferView {
                    metal_binding: 3,
                    ..quad_index_view()
                },
            ],
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            },
            textures: Vec::new(),
        }
    }

    /// The dual declaration pass's compute registration: four read-only
    /// bindings whose static footprints are exactly the views' own lengths.
    fn dual_declaration_pipeline() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: PipelineId::new(11),
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-mrt-fixture", vec![11]).unwrap(),
                entry_name: "declares_the_render_views".to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: PipelineContract {
                dispatch_kind: DispatchKind::ThreadsExact,
                required_local_size: None,
                fixed_grid: None,
                push_constant_offset: 0,
                push_constant_bytes: 0,
                buffer_bindings: vec![
                    BufferBindingContract {
                        metal_binding: 0,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 1,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 16 },
                    },
                    BufferBindingContract {
                        metal_binding: 2,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 32 },
                    },
                    BufferBindingContract {
                        metal_binding: 3,
                        access: BufferAccess::Read,
                        footprint: FootprintProof::Static { max_bytes: 12 },
                    },
                ],
                texture_bindings: Vec::new(),
                shader_capabilities: Vec::new(),
                translator_revision: None,
            },
            render: None,
        }
    }

    /// The trace-table entry of the dual registration, minted the way
    /// `NativeMetalProvider::register_render_pipeline` mints it.
    fn dual_table_entry() -> CompiledComputePipeline {
        CompiledComputePipeline {
            device_epoch: DeviceEpoch::new(3),
            pipeline_id: DUAL_PIPELINE,
            function: FunctionIdentity {
                logical_digest: SemanticDigest::new("render-mrt-fixture", vec![5]).unwrap(),
                entry_name: QUAD_VERTEX_ENTRY.to_owned(),
                source: FunctionSource::MetalSource,
            },
            contract: render_table_contract(),
            render: Some(dual_pipeline()),
        }
    }

    /// The dual fixture's trace and resource namespace: the declaration pass,
    /// then the dual render pass with the requested load op.
    fn dual_trace(load: LoadOp) -> (ComputeTrace, ResourceTableSnapshot) {
        let mut pass = dual_pass();
        for attachment in &mut pass.color_attachments {
            attachment.load = load;
        }
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(3),
            pipelines: vec![dual_declaration_pipeline(), dual_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(dual_declaration_pass()),
                TracePass::Render(pass),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        let mut resources = ResourceTableSnapshot::new();
        for (allocation, size) in [
            (AllocationId::new(9), 16),
            (AllocationId::new(10), 16),
            (QUAD_VERTEX_ALLOCATION, 32),
            (QUAD_INDEX_ALLOCATION, 12),
        ] {
            resources
                .insert_allocation(AllocationRecord {
                    allocation_id: allocation,
                    owner_epoch: DeviceEpoch::new(3),
                    size,
                })
                .unwrap();
        }
        (trace, resources)
    }

    /// The registrations `plan_trace` resolves the dual trace against.
    fn dual_contracts() -> BTreeMap<PipelineId, RenderPipelineContract> {
        BTreeMap::from([(DUAL_PIPELINE, dual_pipeline())])
    }

    /// Shrink a declared stream to `length` bytes, keeping the view's shape
    /// self-consistent so a footprint refusal is the only rule that can fire.
    fn shorten(view: &mut BufferView, length: u64) {
        let BufferSource::OwnedBytes(bytes) = &view.source else {
            panic!("the fixture's streams are owned bytes");
        };
        let end = usize::try_from(length).expect("the fixture's lengths fit a host usize");
        view.length = length;
        view.source = BufferSource::OwnedBytes(bytes[..end].to_vec());
    }

    /// Replace a declared stream's bytes and length together, for the index
    /// values a footprint test wants to vary.
    fn replace_bytes(view: &mut BufferView, bytes: Vec<u8>) {
        view.length = u64::try_from(bytes.len()).expect("the fixture's lengths fit u64");
        view.source = BufferSource::OwnedBytes(bytes);
    }

    /// The `vertex_id` milestone pass with an index buffer bound to it: three
    /// indices select through the module's three generated positions, which is
    /// the shape core admission states for a pass with indices and no stream
    /// (`RenderPassDescriptor::validate`).
    ///
    /// The index bytes travel in the binding's own view, so the serial pool this
    /// trace returns is the compute declaration's — view 7, the attachment's
    /// landing view — and holds no stream at all.
    fn vertex_id_indexed_trace(indices: [u16; 3]) -> (ComputeTrace, Vec<BufferView>) {
        let index_bytes = indices
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<u8>>();
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the milestone fixture ends with its render pass");
        };
        pass.indices = Some(IndexBufferBinding {
            view: BufferView {
                view_id: QUAD_INDEX_VIEW,
                metal_binding: 0,
                allocation_id: QUAD_INDEX_ALLOCATION,
                offset: 0,
                length: u64::try_from(index_bytes.len()).expect("the fixture's lengths fit u64"),
                access: BufferAccess::Read,
                attribute_stride: None,
                source: BufferSource::OwnedBytes(index_bytes),
            },
            format: IndexFormat::Uint16,
        });
        let pool = trace.serial_resources().expect("admitted serial pool");
        (trace, pool)
    }

    /// The trace's table entry and the registry have to carry the same render
    /// contract.
    ///
    /// The entry's `render` half is what core admission compares a pass against
    /// (review item I3, 2026-09-14) and the registry is what the rail executes,
    /// so a fixture where the two disagree would let a value-level test pass on
    /// an agreement the provider would refuse at submit.
    #[test]
    fn the_render_table_entry_carries_the_registered_contract() {
        let entry = render_table_entry();
        let registered = milestone_contracts();
        let contract = registered
            .get(&entry.pipeline_id)
            .expect("the registry holds the entry the render pass names");
        assert_eq!(entry.render.as_ref(), Some(contract));
    }

    /// The step that flips the capability bit needs the declared bits and core
    /// admission to agree: what the snapshot says it renders and what core
    /// admits have to be the same set, or one of the two is lying.
    #[test]
    fn declared_render_capabilities_admit_what_the_rail_plans() {
        let bits = capability_bits(APPLE_2D_TEXTURE_CEILING);
        assert!(bits.supports_render_passes);
        assert_eq!(bits.max_color_attachments, MAX_COLOR_ATTACHMENTS);
        // R1b (`research/docs/23` §70): the declared window is the reviewed
        // ceiling on a device as wide as the review, and the device's own
        // limit on a narrower one.
        assert_eq!(bits.max_attachment_dimension, REVIEWED_ATTACHMENT_CEILING);
        assert_eq!(
            capability_bits(32).max_attachment_dimension,
            [32, 32],
            "a device narrower than the review declares its own 2D texture limit"
        );
        assert_eq!(
            capability_bits(4).max_attachment_dimension,
            [4, 4],
            "the milestone's 4x4 window is the floor the clamp can report"
        );
        assert_eq!(
            bits.supported_color_formats,
            SUPPORTED_COLOR_FORMATS.to_vec()
        );
        // The rail declares the full MRT shape now that its reviewed modules
        // cover one, two and the `MAX_COLOR_ATTACHMENTS` ceiling (v24). The bit
        // still has to stay at or below core's cap (wave3 R1).
        let core_max = u32::try_from(metal_api_core::provider::MAX_COLOR_ATTACHMENTS).unwrap();
        assert_eq!(core_max, 4);
        assert_eq!(bits.max_color_attachments, core_max);
        assert!(bits.max_color_attachments <= core_max);
        assert_eq!(
            bits.supported_color_formats,
            AttachmentFormat::ADMITTED.to_vec()
        );
        // The superset fragment interface is the attachment face's own
        // boundary rather than a missing measurement (2026-09-20, the third
        // door behind census v46's `stage_buffer_footprint` bucket):
        // `reviewed_module` selects its MSL module by the colour format list's
        // exact shape, so a module that stores a location the pass does not
        // attach matches no arm and is refused by name. The declaration stays
        // at the contract's fail-closed default beside the three fields above.
        assert!(!bits.supports_render_fragment_output_superset);
        assert!(!capabilities(&bits).declares_render_fragment_output_superset_support());

        // The 16-bit shader capability pair is this rail's own boundary too
        // (2026-09-20, census v48's LPF pipeline): the reviewed MSL modules are
        // the rail's own text and none of them narrows a float to `half`, so a
        // module that declares `OpCapability Float16`/`Int16` is outside this
        // rail's reviewed set and struck out here — and the snapshot's own bit
        // is what a consumer reads before it hands this provider such a module.
        assert!(!bits.supports_render_half_capabilities);
        assert!(!capabilities(&bits).declares_render_half_capabilities());

        let (trace, resources) = milestone_trace(LoadOp::Clear(sentinel()));
        capabilities(&bits)
            .admit(&trace, &resources)
            .expect("the declared bits admit the milestone trace");

        // The pre-flip snapshot refuses the same trace in render admission,
        // before any compute reservation — that refusal is what the flip
        // removes.
        let mut before_the_flip = bits.clone();
        before_the_flip.supports_render_passes = false;
        let refused = capabilities(&before_the_flip)
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "render_passes_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
    }

    /// The flipped snapshot admits the present-bearing trace in the second
    /// admission gate, and its present bits are the rail's own limits rather
    /// than a second spelling that could drift from the contract's. The flip
    /// evidence is the Apple-GPU `--present-selftest` run recorded on
    /// [`present_capability_bits`].
    #[test]
    fn declared_present_capabilities_admit_what_the_rail_plans() {
        let bits = capability_bits(APPLE_2D_TEXTURE_CEILING);
        assert!(bits.supports_presentation);
        assert_eq!(bits.max_present_targets, MAX_PRESENT_TARGETS);
        assert_eq!(bits.supported_present_modes, PresentMode::ADMITTED.to_vec());
        assert_eq!(bits.max_present_image_count, MAX_PRESENT_IMAGE_COUNT);
        // The first increment admits exactly one target, one image and FIFO
        // (`research/docs/24` §3.1), which is what those constants say.
        assert_eq!(
            bits.max_present_targets,
            metal_api_core::provider::MAX_PRESENT_TARGETS as u32
        );
        assert_eq!(
            bits.max_present_image_count,
            metal_api_core::provider::MAX_PRESENT_IMAGE_COUNT
        );

        let (trace, resources) = milestone_present_trace();
        capabilities(&bits)
            .admit(&trace, &resources)
            .expect("the declared bits admit the present milestone trace");
    }

    /// The pre-flip snapshot — the same bits with the present gate closed —
    /// still refuses the present-bearing trace in the second admission gate
    /// (`present_targets_unsupported`) before any resource action. Keeping the
    /// refusal pinned on a constructed snapshot is what makes the flipped
    /// production bits above falsifiable: it is the same trace and the same
    /// gate, only the declaration differs.
    #[test]
    fn the_pre_flip_snapshot_refuses_a_present_bearing_trace() {
        let mut bits = capability_bits(APPLE_2D_TEXTURE_CEILING);
        bits.supports_presentation = false;
        bits.max_present_targets = 0;
        bits.supported_present_modes = Vec::new();
        bits.max_present_image_count = 0;

        let (trace, resources) = milestone_present_trace();
        let refused = capabilities(&bits).admit(&trace, &resources).unwrap_err();
        assert_eq!(refused.slug, "present_targets_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        // The refusal names the number of present actions the snapshot would
        // have to drop.
        assert_eq!(
            refused.fields.get("targets"),
            Some(&FieldValue::Unsigned(1))
        );
    }

    /// The stage-buffer pair the Apple device readings flipped
    /// (`stage_buffer_capability_bits`): the declared bits are the rail's own
    /// window, the limit is core's ceiling rather than a second number, and
    /// admission admits the reviewed stage-buffer trace — the shape both
    /// readings executed on device (`--stage-buffer-selftest`,
    /// `--stage-buffer-write-selftest`).
    #[test]
    fn declared_stage_buffer_capabilities_admit_what_the_rail_plans() {
        let bits = stage_buffer_capability_bits();
        assert!(bits.supports_render_stage_buffers);
        assert_eq!(bits.max_render_stage_buffers, MAX_RENDER_STAGE_BUFFERS);
        // The per-stage window stays undeclared (`research/docs/23` §117,
        // E-SB2): this rail's reviewed modules bind one slot per stage, so the
        // list bound is the whole rule and a pair that declares thirteen slots
        // between its stages is refused by name by core admission instead of
        // being executed against slots no Apple reading sized.
        assert_eq!(bits.max_render_stage_buffers_per_stage, 0);
        assert!(!capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .declares_render_stage_buffer_per_stage_ceiling());
        // The folded shape's bit is part of the same spelling (`research/docs/23`
        // §3.3, E-TX9): the reviewed pair's two stages are bound at set 1 and
        // set 2, so the shape that needs those two namespaces declared here.
        assert!(bits.supports_render_stage_buffer_namespace_split);
        assert_eq!(
            bits.max_render_stage_buffers,
            metal_api_core::provider::MAX_RENDER_STAGE_BUFFERS as u32
        );

        let (trace, resources) = stage_buffer_trace(LoadOp::Clear(sentinel()));
        capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .admit(&trace, &resources)
            .expect("the declared bits admit the reviewed stage-buffer trace");

        // A trace whose passes bind no stage buffer keeps the pre-v83
        // admission path: the flipped gate widens nothing else.
        let (milestone, milestone_resources) = milestone_trace(LoadOp::Clear(sentinel()));
        capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .admit(&milestone, &milestone_resources)
            .expect("a pass that binds no stage buffer keeps its own admission");
    }

    /// The pre-flip snapshot — the same bits with the stage-buffer gate closed
    /// — still refuses the reviewed trace in the third render gate
    /// (`render_stage_buffer_unsupported`), after the render gate admitted the
    /// pass itself. Keeping the refusal pinned on a constructed snapshot is
    /// what makes the flipped production bits above falsifiable: it is the same
    /// trace and the same gate, only the declaration differs.
    #[test]
    fn the_pre_flip_snapshot_refuses_a_stage_buffer_trace() {
        let (trace, resources) = stage_buffer_trace(LoadOp::Clear(sentinel()));
        let refused =
            capabilities_before_the_stage_buffer_flip(&capability_bits(APPLE_2D_TEXTURE_CEILING))
                .admit(&trace, &resources)
                .unwrap_err();
        assert_eq!(refused.slug, "render_stage_buffer_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        // The refusal names the pass and the number of bindings the snapshot
        // would have to drop.
        assert_eq!(refused.fields.get("pass"), Some(&FieldValue::Unsigned(1)));
        assert_eq!(
            refused.fields.get("bindings"),
            Some(&FieldValue::Unsigned(2))
        );
    }

    /// The host-side half of the present plan: the descriptor is borrowed from
    /// the pass and the sentinel is expanded to the target's whole extent, so
    /// the macOS present path only has to upload it.
    #[test]
    fn plan_trace_plans_the_present_action_and_expands_the_sentinel() {
        let (trace, _) = milestone_present_trace();
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed present pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the present trace carries one render pass");
        };
        let present = planned
            .present
            .as_ref()
            .expect("the pass carries a present");
        assert_eq!(present.descriptor.source, ViewId::new(7));
        assert_eq!(
            present.descriptor.target.allocation_id,
            AllocationId::new(9)
        );
        assert_eq!(present.descriptor.target.view_id, ViewId::new(7));
        assert_eq!(present.descriptor.mode, PresentMode::Fifo);
        assert_eq!(present.descriptor.acquire, AcquirePolicy::Blocking);
        // The one-texel sentinel is replicated across the 2x2 target.
        assert_eq!(
            present.sentinel.as_deref(),
            Some([0xfe, 0xfe, 0xfe, 0xfe].repeat(4).as_slice())
        );
        // The present pass keeps the target's sentinel through a `Load`, which
        // is the load op the present path can honour (`research/docs/24` §3.1).
        assert!(matches!(
            planned.plan.attachments[0].load,
            RenderLoadAction::Load
        ));
    }

    /// The host-side half of the trace path: the plan a device-free host can
    /// check, which is the same decision the macOS encoder body then executes.
    #[test]
    fn plan_trace_plans_the_milestone_pass_and_its_landing_view() {
        let (trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.pass.color_attachments[0].view_id, ViewId::new(7));
        assert_eq!(planned.contract, &milestone_pipeline());
        assert_eq!(planned.plan.extent, [2, 2]);
        assert_eq!(planned.plan.attachments[0].texel.bytes, 16);
        assert_eq!(planned.plan.attachments[0].texel.row_pitch, 8);
        assert_eq!(
            planned.plan.attachments[0].format,
            RenderPixelFormat::Rgba8Unorm
        );
        assert_eq!(planned.plan.vertices, 3);
        assert_eq!(planned.plan.attachments[0].initial_bytes(), None);
        // The landing view is the declaration's own identity and range, so the
        // writeback is the one the trace asked for and no second channel is
        // invented.
        assert_eq!(
            planned.landings[0].expect("the attachment lands").view_id,
            ViewId::new(7)
        );
        assert_eq!(
            planned.landings[0]
                .expect("the attachment lands")
                .allocation_id,
            AllocationId::new(9)
        );
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
            stage_buffers: Vec::new(),
        });
        let [writeback] = writebacks.as_slice() else {
            panic!("the milestone attachment becomes one writeback");
        };
        assert_eq!(writeback.view_id, ViewId::new(7));
        assert_eq!(writeback.allocation_id, AllocationId::new(9));
        assert_eq!(writeback.offset, 0);
        assert_eq!(writeback.bytes, [0x40, 0x80, 0xc0, 0xff].repeat(4));

        // A compute-only trace keeps the pre-render path: nothing to plan, no
        // new refusal and no attachment readback.
        let mut compute_only = trace.clone();
        compute_only
            .passes
            .retain(|pass| pass.as_compute().is_some());
        assert!(plan_trace(&compute_only, &pool, &contracts, 0, 0)
            .expect("a compute-only trace plans nothing")
            .is_empty());
    }

    /// A loading pass uploads the bytes its declaring view owns: the plan keeps
    /// `Load` as the load action and carries those bytes, which is what the
    /// encoder presets into the attachment instead of clearing it
    /// (`research/docs/23` §3.3).
    #[test]
    fn plan_trace_plans_a_loading_pass_and_its_previous_bytes() {
        let (trace, _) = milestone_trace(LoadOp::Load);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the declaring view's own bytes are what a load uploads");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::Load);
        // The declaration's 16-byte range is the tightly packed 2x2 rgba8
        // texels, so the plan carries exactly those bytes.
        assert_eq!(
            planned.plan.attachments[0].initial_bytes(),
            Some([0xfe; 16].as_slice())
        );
    }

    /// A `DontCare` attachment still needs its landing declaration — the stored
    /// texels land through the writeback channel — but resolves no previous
    /// bytes, so the plan carries `DontCare` and presets nothing: the output is
    /// the draw alone (`research/docs/23` §3.1, v20).
    #[test]
    fn plan_trace_plans_a_dont_care_pass_without_declared_bytes() {
        let (trace, _) = milestone_trace(LoadOp::DontCare);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the declaring view lands the writeback; only the bytes are absent");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::DontCare);
        assert_eq!(
            planned.plan.attachments[0].initial_bytes(),
            None,
            "a DontCare attachment presets nothing"
        );
    }

    /// A lease-backed declaration is resolved through the lease channel
    /// (`research/docs/23` §74, R5b); with no channel to resolve it in — the
    /// `plan_trace` shape, which is the device-level caller — the loading pass
    /// is refused under the source's own name rather than executed as a clear.
    #[test]
    fn plan_trace_refuses_a_leased_attachment_load_without_a_channel() {
        let (trace, _) = milestone_trace(LoadOp::Load);
        let mut pool = trace.serial_resources().expect("admitted serial pool");
        let view = pool
            .iter_mut()
            .find(|view| {
                view.view_id == ViewId::new(7) && view.allocation_id == AllocationId::new(9)
            })
            .expect("the milestone trace declares the attachment view");
        view.source = BufferSource::StagedLease(LeaseId::new(5));
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_load_source_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(ViewId::new(7).get()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// A landing-only entry has no rail here (`research/docs/23` §115 之后的
    /// 增量，E-TX14/R4b). This provider keeps resident images — its registry is
    /// the sibling of the Vulkan one — but its owner-window channel is an
    /// *input* route: no code path writes an owner's pages, which is why
    /// `StoreOp::Borrowed` has been refused since E-TX8. Delivering a kept frame
    /// would need exactly that write route, so the entry is refused by name
    /// before any plan exists — and before the early return a landing-only trace
    /// would otherwise take.
    #[test]
    fn plan_trace_refuses_a_kept_frame_landing_by_name() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        trace.passes.push(TracePass::Landing(KeptFrameLanding {
            frame: KeptFrame {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(8),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
            },
            landing: AttachmentLandingView {
                allocation_id: AllocationId::new(11),
                view_id: ViewId::new(10),
            },
        }));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "kept_frame_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("source"),
            Some(&FieldValue::Text("native_rail".to_owned())),
            "the refusal names the rail that has no owner-window write route"
        );
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(8)),
            "the refusal names the kept frame it could not deliver"
        );
        // The snapshot says the same thing the rail does: the bit stays at its
        // fail-closed default on this rail.
        assert!(!capabilities(&capability_bits(16384)).supports_render_kept_frame_landing);
    }

    /// A pass-entry snapshot declaration has no rail here (`research/docs/23`
    /// §118, E-TX15). The arm promises a device-side image copy taken before
    /// the render pass opens and bound as the sampled view, and this rail has
    /// no Apple oracle for a fragment reading the attachment the same pass
    /// writes — so the declaration is refused under the arm's own name, with
    /// the rail beside it, and the snapshot's bit stays at its fail-closed
    /// default.
    #[test]
    fn plan_trace_refuses_a_pass_entry_snapshot_by_name() {
        // The rail resolves the arm at its own texture walk, so the reading is
        // the plan gate rather than the trace walk: the pass and its pipeline
        // are the reviewed sampling pair, whose declaration matches the
        // rewritten texture field for field.
        let mut pass = sampled_pass(4);
        pass.color_attachments[0].load = LoadOp::Load;
        let attachment = pass.color_attachments[0];
        let mut texture = pass.textures[0].clone();
        texture.view_id = attachment.view_id;
        texture.allocation_id = attachment.allocation_id;
        texture.format = attachment.format.as_texture_format();
        texture.width = attachment.width;
        texture.height = attachment.height;
        texture.source = TextureSource::PassEntrySnapshot;
        pass.textures = vec![texture];
        let error = plan_pass(&OffscreenRenderRequest {
            pass: &pass,
            pipeline: &sampled_pipeline(),
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![Some(PlannedInputSource::Declared(&[0x11; 64]))],
            resident: Vec::new(),
        })
        .unwrap_err();
        assert_eq!(error.slug, "render_texture_source_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("pass_entry_snapshot".to_owned())),
            "the refusal names the arm it cannot execute"
        );
        // The snapshot says the same thing the rail does: the bit stays at its
        // fail-closed default on this rail.
        assert!(!capabilities(&capability_bits(16384)).supports_render_pass_entry_snapshot);
    }

    /// The landing rail is the writeback channel: an attachment no declared view
    /// covers has nowhere to land, so it is refused instead of executed and
    /// dropped.
    #[test]
    fn plan_trace_refuses_an_attachment_without_a_landing_view() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.color_attachments[0].view_id = ViewId::new(8);
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(8)),
            "the refusal names the attachment that has no landing rail"
        );

        // A loading pass needs the declaration twice over — as its landing and
        // as the source of the bytes it uploads — so an undeclared attachment
        // is refused under the same slug rather than planned without bytes.
        let (mut loading, _) = milestone_trace(LoadOp::Load);
        let Some(TracePass::Render(pass)) = loading.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.color_attachments[0].view_id = ViewId::new(8);
        let error = plan_trace(&loading, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_landing_unsupported");
    }

    /// The v43 increment over the trace path: a depth attachment the pass
    /// stores resolves its declared view as a second landing, and the stored
    /// texels become the writeback that follows the colour ones in the same
    /// channel (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_trace_plans_a_stored_depth_pass_and_its_landing_view() {
        let trace = depth_store_trace(Some(DepthStoreOp::Store));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the stored depth surface lands in its declaring view");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        let Some(depth) = &planned.plan.depth else {
            panic!("the pass opens a depth attachment");
        };
        assert_eq!(depth.store, Some(DepthStoreOp::Store));
        // The landing is the declaration's own identity and range, so the
        // stored texels leave through the view the trace named and no second
        // channel is invented (`research/docs/23` §3.3, v43).
        let landing = planned
            .depth_landing
            .expect("a stored depth attachment names its landing view");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        assert_eq!(landing.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(landing.offset, 0);
        assert_eq!(landing.length, DEPTH_STORE_BYTES);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: Some(DEPTH_STORE_TEXEL.repeat(4)),
            stencil: None,
            stage_buffers: Vec::new(),
        });
        let [colour, depth] = writebacks.as_slice() else {
            panic!("the stored depth attachment becomes a second writeback");
        };
        assert_eq!(colour.view_id, ViewId::new(7));
        assert_eq!(colour.allocation_id, AllocationId::new(9));
        assert_eq!(colour.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(depth.view_id, DEPTH_STORE_VIEW);
        assert_eq!(depth.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(depth.offset, 0);
        assert_eq!(depth.bytes, DEPTH_STORE_TEXEL.repeat(4));
    }

    /// The v57c trace shape: the stored depth pair over a four-sample raster
    /// whose resolve the device mask admits. The stored surface's landing is
    /// the same v43 view — the resolve target is a rail-owned texture, not a
    /// second trace identity — and a mask without the filter bit refuses the
    /// pass at plan time, before any Metal object exists
    /// (`research/docs/23` §3.3, v57c).
    #[test]
    fn plan_trace_plans_a_resolving_depth_pass_when_the_device_mask_admits_it() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        pass.depth_resolve = Some(MultisampleDepthResolve {
            filter: DepthResolveFilter::Sample0,
        });
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, DEPTH_RESOLVE_SAMPLE0_BIT, 0)
            .expect("the resolving pass plans in the admitted mask");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        assert_eq!(planned.plan.multisample, Some(SampleCount::Four));
        assert_eq!(
            planned.plan.depth_resolve,
            Some(DepthResolveFilter::Sample0)
        );
        let landing = planned
            .depth_landing
            .expect("the stored depth surface names its v43 landing");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        assert_eq!(landing.allocation_id, DEPTH_STORE_ALLOCATION);

        // The same trace through a mask without the Sample0 bit is refused by
        // the per-filter question the capability snapshot answered.
        let refused = plan_trace(&trace, &pool, &contracts, 0, 0).unwrap_err();
        assert_eq!(refused.slug, "render_depth_resolve_filter_unsupported");
    }

    /// The two shapes that state no store keep the surface rail-owned: no
    /// landing is resolved, and a readback that carries no depth texels adds no
    /// second writeback — the pre-v43 byte shape exactly
    /// (`research/docs/23` §3.3, v43).
    ///
    /// The depth-only shape is the same pass with every colour attachment
    /// discarding (`research/docs/23` §3.3, v45): the plan still carries the
    /// colour attachment — it renders, its bytes disappear — and the stored
    /// depth surface is the whole observation, so the readback is a depth one
    /// and nothing else.
    #[test]
    fn plan_trace_plans_a_depth_only_pass() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        let pass = trace
            .passes
            .iter_mut()
            .find_map(|pass| match pass {
                TracePass::Render(pass) => Some(pass),
                TracePass::Compute(_) => None,
                TracePass::Landing(_) => None,
            })
            .expect("the fixture carries a render pass");
        pass.color_attachments[0].store = StoreOp::DontCare;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("a pass whose only landing is its depth surface plans");
        let [planned] = planned.as_slice() else {
            panic!("the depth trace carries one render pass");
        };
        assert_eq!(
            planned.plan.attachments[0].store,
            RenderStoreAction::DontCare
        );
        let landing = planned
            .depth_landing
            .expect("the stored depth surface is the pass's landing");
        assert_eq!(landing.view_id, DEPTH_STORE_VIEW);
        // The readback carries no colour texels — the discarded attachment
        // lands nothing — and the depth texels become the pass's one
        // writeback, in the depth view the trace declared.
        let writebacks = planned.writebacks(RenderReadback {
            attachments: Vec::new(),
            depth: Some(DEPTH_STORE_TEXEL.repeat(4)),
            stencil: None,
            stage_buffers: Vec::new(),
        });
        let [depth] = writebacks.as_slice() else {
            panic!("a depth-only pass lands exactly one writeback");
        };
        assert_eq!(depth.view_id, DEPTH_STORE_VIEW);
        assert_eq!(depth.allocation_id, DEPTH_STORE_ALLOCATION);
        assert_eq!(depth.bytes, DEPTH_STORE_TEXEL.repeat(4));
    }

    /// The discard shapes that state no depth store keep the pre-v45 refusal.
    #[test]
    fn plan_trace_keeps_a_discarded_depth_attachment_out_of_the_writebacks() {
        for store in [None, Some(DepthStoreOp::DontCare)] {
            let trace = depth_store_trace(store);
            let pool = trace.serial_resources().expect("admitted serial pool");
            let contracts = milestone_contracts();
            let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
                .expect("a discarded depth surface needs no landing");
            let [planned] = planned.as_slice() else {
                panic!("the depth trace carries one render pass");
            };
            let Some(depth) = &planned.plan.depth else {
                panic!("the pass opens a depth attachment");
            };
            assert_eq!(depth.store, store);
            assert!(
                planned.depth_landing.is_none(),
                "{store:?} keeps the depth surface rail-owned"
            );
            let writebacks = planned.writebacks(RenderReadback {
                attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
                depth: None,
                stencil: None,
                stage_buffers: Vec::new(),
            });
            let [writeback] = writebacks.as_slice() else {
                panic!("{store:?} lands the colour attachment alone");
            };
            assert_eq!(writeback.view_id, ViewId::new(7));
            assert_eq!(writeback.allocation_id, AllocationId::new(9));
            assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        }
    }

    /// A storing depth attachment no declared view covers has nowhere to land,
    /// so the pass is refused by name instead of executed and dropped — the
    /// shape the colour attachments' landing refusal already has
    /// (`research/docs/23` §3.3, v43).
    #[test]
    fn plan_trace_refuses_a_stored_depth_attachment_without_a_landing_view() {
        let mut trace = depth_store_trace(Some(DepthStoreOp::Store));
        // The depth declaration is what the landing resolution reads: without
        // it the trace declares no view covering the stored surface, and the
        // colour attachment's own declaration stays in place. Core admission
        // states the same requirement one level up, which is why the trace
        // itself no longer resolves to a pool.
        trace.passes.retain(|pass| {
            !matches!(
                pass,
                TracePass::Compute(compute)
                    if compute
                        .buffers
                        .iter()
                        .any(|view| view.view_id == DEPTH_STORE_VIEW)
            )
        });
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::AttachmentViewUnknown { .. })
        ));
        // The rail's own walk keeps the refusal for a plan that never went
        // through admission: the pool below is the shape the colour attachment
        // alone declares, so the stored depth surface is the only landing left
        // without a view.
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_depth_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(DEPTH_STORE_VIEW.get())),
            "the refusal names the depth view that has no landing rail"
        );
        assert_eq!(
            error.fields.get("allocation"),
            Some(&FieldValue::Unsigned(DEPTH_STORE_ALLOCATION.get()))
        );
    }

    /// The v49 increment over the trace path: a stencil attachment the pass
    /// stores resolves its declared view as a second landing beside the depth
    /// one, and the stored one-byte texels become the writeback that follows
    /// the colour ones in the same channel (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_plans_a_stored_stencil_pass_and_its_landing_view() {
        let trace = stencil_store_trace(Some(StoreOp::Store));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = stencil_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("the stored stencil surface lands in its declaring view");
        let [planned] = planned.as_slice() else {
            panic!("the stencil trace carries one render pass");
        };
        let Some(stencil) = &planned.plan.stencil else {
            panic!("the pass opens a stencil attachment");
        };
        assert_eq!(stencil.store, Some(StoreOp::Store));
        // The landing is the declaration's own identity and range, so the
        // stored texels leave through the view the trace named and no second
        // channel is invented (`research/docs/23` §3.3, v49).
        let landing = planned
            .stencil_landing
            .expect("a stored stencil attachment names its landing view");
        assert_eq!(landing.view_id, STENCIL_STORE_VIEW);
        assert_eq!(landing.allocation_id, STENCIL_STORE_ALLOCATION);
        assert_eq!(landing.offset, 0);
        assert_eq!(landing.length, STENCIL_STORE_BYTES);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: Some(vec![STENCIL_STORE_TEXEL; 4]),
            stage_buffers: Vec::new(),
        });
        let [colour, stencil] = writebacks.as_slice() else {
            panic!("the stored stencil attachment becomes a second writeback");
        };
        assert_eq!(colour.view_id, ViewId::new(7));
        assert_eq!(colour.allocation_id, AllocationId::new(9));
        assert_eq!(colour.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(stencil.view_id, STENCIL_STORE_VIEW);
        assert_eq!(stencil.allocation_id, STENCIL_STORE_ALLOCATION);
        assert_eq!(stencil.offset, 0);
        assert_eq!(stencil.bytes, vec![STENCIL_STORE_TEXEL; 4]);
    }

    /// The two shapes that state no stencil store keep the surface rail-owned:
    /// no landing is resolved, and a readback that carries no stencil texels
    /// adds no second writeback — the pre-v49 byte shape exactly
    /// (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_keeps_an_unstored_stencil_attachment_out_of_the_writebacks() {
        for store in [None, Some(StoreOp::DontCare)] {
            let trace = stencil_store_trace(store);
            let pool = trace.serial_resources().expect("admitted serial pool");
            let contracts = stencil_contracts();
            let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
                .expect("a discarded stencil surface needs no landing");
            let [planned] = planned.as_slice() else {
                panic!("the stencil trace carries one render pass");
            };
            let Some(stencil) = &planned.plan.stencil else {
                panic!("the pass opens a stencil attachment");
            };
            assert_eq!(stencil.store, store);
            assert!(
                planned.stencil_landing.is_none(),
                "{store:?} keeps the stencil surface rail-owned"
            );
            let writebacks = planned.writebacks(RenderReadback {
                attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
                depth: None,
                stencil: None,
                stage_buffers: Vec::new(),
            });
            let [writeback] = writebacks.as_slice() else {
                panic!("{store:?} lands the colour attachment alone");
            };
            assert_eq!(writeback.view_id, ViewId::new(7));
            assert_eq!(writeback.allocation_id, AllocationId::new(9));
            assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        }
    }

    /// A storing stencil attachment no declared view covers has nowhere to
    /// land, so the pass is refused by name instead of executed and dropped —
    /// the shape the colour and depth attachments' landing refusals already
    /// have (`research/docs/23` §3.3, v49).
    #[test]
    fn plan_trace_refuses_a_stored_stencil_attachment_without_a_landing_view() {
        let mut trace = stencil_store_trace(Some(StoreOp::Store));
        // The stencil declaration is what the landing resolution reads: without
        // it the trace declares no view covering the stored surface, and the
        // colour attachment's own declaration stays in place. Core admission
        // states the same requirement one level up, which is why the trace
        // itself no longer resolves to a pool.
        trace.passes.retain(|pass| {
            !matches!(
                pass,
                TracePass::Compute(compute)
                    if compute
                        .buffers
                        .iter()
                        .any(|view| view.view_id == STENCIL_STORE_VIEW)
            )
        });
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::AttachmentViewUnknown { .. })
        ));
        // The rail's own walk keeps the refusal for a plan that never went
        // through admission: the pool below is the shape the colour attachment
        // alone declares, so the stored stencil surface is the only landing
        // left without a view.
        let pool = vec![declaration_pass().buffers[0].clone()];
        let error = plan_trace(&trace, &pool, &stencil_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_stencil_landing_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(STENCIL_STORE_VIEW.get())),
            "the refusal names the stencil view that has no landing rail"
        );
        assert_eq!(
            error.fields.get("allocation"),
            Some(&FieldValue::Unsigned(STENCIL_STORE_ALLOCATION.get()))
        );
    }

    /// The ordering rule the trace path shares with the Vulkan rail: every
    /// compute pass runs before every render pass, so a compute pass that
    /// follows a render store of a view it binds would read pre-render bytes.
    ///
    /// Core admission states that order as part of the contract (review item
    /// I4, 2026-09-14), so the first half observes the refusal from
    /// `serial_resources` — the value-level entry point that runs admission —
    /// and the second half keeps the rail's own walk covered as defense in
    /// depth for a plan that never went through admission.
    #[test]
    fn plan_trace_refuses_a_compute_pass_that_reads_after_a_render_store() {
        let (mut trace, _) = milestone_trace(LoadOp::Clear(sentinel()));
        trace.passes.push(TracePass::Compute(declaration_pass()));
        assert!(matches!(
            trace.serial_resources(),
            Err(ContractError::RenderPassOrderUnsupported { .. })
        ));
        let error = refuse_reordered_render_reads(&trace).unwrap_err();
        assert_eq!(error.slug, "render_pass_order_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);

        // The same rule in its legal direction: a declaration that comes before
        // the store is planned, not refused.
        let (legal, _) = milestone_trace(LoadOp::Clear(sentinel()));
        let pool = legal.serial_resources().expect("admitted serial pool");
        assert_eq!(
            plan_trace(&legal, &pool, &milestone_contracts(), 0, 0)
                .unwrap()
                .len(),
            1
        );
    }

    /// A render registration is a review gate: an entry pair the reviewed module
    /// does not carry cannot be registered, the same way an unreviewed MSL
    /// fixture cannot be compiled.
    #[test]
    fn a_render_contract_has_to_name_the_reviewed_entries() {
        assert_eq!(review_contract(&milestone_pipeline()), Ok(()));
        let mut edited = milestone_pipeline();
        edited.fragment_entry = "render_solid_rgba8_v2".to_owned();
        let error = review_contract(&edited).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Compile);
    }

    /// The merge that puts the two rails' bytes in one channel: one writeback
    /// per written view, in the canonical order the submission protocol
    /// requires, with the render bytes winning because the render rail runs
    /// last.
    #[test]
    fn render_writebacks_replace_the_compute_bytes_of_the_same_view() {
        let later_view = BufferWriteback {
            view_id: ViewId::new(2),
            allocation_id: AllocationId::new(1),
            offset: 16,
            bytes: vec![0x11; 4],
        };
        let attachment = BufferWriteback {
            view_id: ViewId::new(1),
            allocation_id: AllocationId::new(1),
            offset: 0,
            bytes: vec![0xfe; 16],
        };
        let rendered = BufferWriteback {
            view_id: ViewId::new(1),
            allocation_id: AllocationId::new(1),
            offset: 0,
            bytes: [0x40, 0x80, 0xc0, 0xff].repeat(4),
        };
        let merged = merge_writebacks(vec![later_view.clone(), attachment], vec![rendered.clone()]);
        assert_eq!(merged, vec![rendered, later_view]);
    }

    /// The vertex-input bits the provider declares have to be the rail's own
    /// limits, and core admission has to admit exactly the trace the rail plans
    /// — the same agreement the render and present bits are held to. The
    /// pre-flip snapshot is the falsifiable half: with the three bits at their
    /// defaults the same trace is refused during admission instead of being
    /// executed with positions the trace did not ask for.
    #[test]
    fn declared_vertex_input_bits_admit_what_the_rail_plans() {
        let bits = vertex_input_capability_bits();
        assert_eq!(bits.max_vertex_buffers, MAX_VERTEX_BUFFERS);
        // The rail's limits are core's own values, not a second spelling that
        // could drift from the contract's.
        assert_eq!(
            bits.max_vertex_buffers,
            u32::try_from(metal_api_core::provider::MAX_VERTEX_BUFFERS).unwrap()
        );
        assert_eq!(
            bits.supported_vertex_formats,
            DECLARED_VERTEX_FORMATS.to_vec()
        );
        assert_eq!(bits.supported_index_formats, IndexFormat::ADMITTED.to_vec());
        // The superset vertex interface is this rail's own boundary rather than
        // a missing measurement (`research/docs/23` §3.3, E-TX11):
        // `reviewed_module` selects its MSL module by the layout's exact shape,
        // so a layout declaring attributes no reviewed module reads is refused
        // by name, and the declaration stays at the contract's fail-closed
        // default beside the three fields above.
        assert!(!bits.supports_render_vertex_interface_superset);
        // The declared window is a strict subset of the contract's vocabulary:
        // the four normalized storages and the scalar lane are mapped (see the
        // sibling test) but not declared, because the Apple-side reading that
        // would declare them has not been taken (`research/docs/23` §103;
        // 2026-09-20 for the scalar lane).
        assert_eq!(
            bits.supported_vertex_formats.len(),
            DECLARED_VERTEX_FORMATS.len()
        );
        assert!(
            VertexFormat::ADMITTED.len() > DECLARED_VERTEX_FORMATS.len(),
            "the contract admits more storages than this rail declares"
        );
        for format in DECLARED_VERTEX_FORMATS {
            assert!(VertexFormat::ADMITTED.contains(&format));
        }
        assert_eq!(
            bits.supported_index_formats.len(),
            IndexFormat::ADMITTED.len()
        );

        let (trace, resources) = quad_trace();
        capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .admit(&trace, &resources)
            .expect("the declared bits admit the vertex-input trace");

        // Closed stream count: the pass's binding is refused before its formats
        // are read, which is the order capability admission documents.
        let mut closed = vertex_input_capability_bits();
        closed.max_vertex_buffers = 0;
        let refused = capabilities_with(
            &capability_bits(APPLE_2D_TEXTURE_CEILING),
            &closed,
            &stage_buffer_capability_bits(),
        )
        .admit(&trace, &resources)
        .unwrap_err();
        assert_eq!(refused.slug, "vertex_buffer_limit");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(refused.phase, ProviderPhase::Resolve);
        assert_eq!(
            refused.fields.get("maximum"),
            Some(&FieldValue::Unsigned(0))
        );

        // Closed index widths: the same trace, refused one gate later.
        let mut no_indices = vertex_input_capability_bits();
        no_indices.supported_index_formats = Vec::new();
        let refused = capabilities_with(
            &capability_bits(APPLE_2D_TEXTURE_CEILING),
            &no_indices,
            &stage_buffer_capability_bits(),
        )
        .admit(&trace, &resources)
        .unwrap_err();
        assert_eq!(refused.slug, "index_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);

        // Closed vertex formats: the layout's attribute format is the fact this
        // gate reads, after the pass and the layout already agreed.
        let mut no_formats = vertex_input_capability_bits();
        no_formats.supported_vertex_formats = Vec::new();
        let refused = capabilities_with(
            &capability_bits(APPLE_2D_TEXTURE_CEILING),
            &no_formats,
            &stage_buffer_capability_bits(),
        )
        .admit(&trace, &resources)
        .unwrap_err();
        assert_eq!(refused.slug, "vertex_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
    }

    /// The storages the reviewed modules do not read are *mapped* and not
    /// *declared* (`research/docs/23` §103, E-VF1; the scalar lane on
    /// 2026-09-20, census v46's `vertex_format` bucket), and both halves are
    /// pinned here: each of the contract's five widened formats translates into
    /// the `MTLVertexFormat` that names it, while core admission refuses a trace
    /// whose layout declares one — the fail-closed arm until an Apple device
    /// reading lands.
    #[test]
    fn the_storages_the_reviewed_module_does_not_read_are_mapped_before_they_are_declared() {
        for (format, expected, name, bytes) in [
            (
                VertexFormat::Unorm8x2,
                RenderVertexFormat::UChar2Normalized,
                "unorm8x2",
                2,
            ),
            (
                VertexFormat::Unorm8x4,
                RenderVertexFormat::UChar4Normalized,
                "unorm8x4",
                4,
            ),
            (
                VertexFormat::Unorm16x2,
                RenderVertexFormat::UShort2Normalized,
                "unorm16x2",
                4,
            ),
            (
                VertexFormat::Unorm16x4,
                RenderVertexFormat::UShort4Normalized,
                "unorm16x4",
                8,
            ),
            (
                VertexFormat::Float32x1,
                RenderVertexFormat::Float,
                "float32x1",
                4,
            ),
        ] {
            assert_eq!(vertex_format(format), expected, "{name}");
            assert_eq!(expected.name(), name);
            assert_eq!(format.bytes(), bytes, "{name}");
            assert!(
                !DECLARED_VERTEX_FORMATS.contains(&format),
                "{name} is not part of the declared window yet"
            );
        }
        // The declared window is exactly what capability admission reads, so a
        // normalized attribute is refused by name instead of being executed
        // through a descriptor this rail never observed.
        let (mut trace, resources) = quad_trace();
        let entry = trace
            .pipelines
            .last_mut()
            .expect("the fixture carries a pipeline table");
        let contract = entry
            .render
            .as_mut()
            .expect("the table entry carries the render contract");
        let VertexLayout::Buffers(buffers) = &mut contract.vertex_layout else {
            unreachable!("the reviewed layout is a buffer layout")
        };
        // The same reviewed stream, declared as one `unorm8x4` attribute: a
        // four-byte storage in its own stride, so nothing but the storage
        // differs from the trace the declared window admits.
        buffers[0].stride = 4;
        buffers[0].attributes[0].format = VertexFormat::Unorm8x4;
        let refused = capabilities(&capability_bits(APPLE_2D_TEXTURE_CEILING))
            .admit(&trace, &resources)
            .unwrap_err();
        assert_eq!(refused.slug, "vertex_format_unsupported");
        assert_eq!(refused.class, ProviderErrorClass::Capability);
        assert_eq!(
            refused.fields.get("format"),
            Some(&FieldValue::Unsigned(u64::from(
                VertexFormat::Unorm8x4.code()
            )))
        );
        // (The declaration itself is the thing that moves when the Apple
        // reading lands; this test reads the same list the snapshot does rather
        // than a second copy.)
        assert!(!vertex_input_capability_bits()
            .supported_vertex_formats
            .contains(&VertexFormat::Unorm8x4));
    }

    /// The descriptor translation: what the rail hands Metal is the layout's
    /// binding order, stride and attributes, plus the digits' formats and the
    /// index width. Every value here is what the macOS encoder writes into its
    /// `MTLVertexDescriptor` and its draw call.
    #[test]
    fn plan_translates_the_indexed_layout_into_a_descriptor_plan() {
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let planned = plan(&quad_request(&pass, &pipeline), 0, 0).expect("the reviewed pass plans");

        assert_eq!(planned.source, REVIEWED_VERTEX_SOURCE);
        assert_eq!(
            planned.module_path,
            "conformance/shaders/quad_indexed_2x2.metal"
        );
        assert_eq!(planned.vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(planned.fragment_entry, FRAGMENT_ENTRY);
        assert_eq!(planned.vertices, 6);

        let [stream] = planned.vertex_streams.as_slice() else {
            panic!("the reviewed layout binds one stream");
        };
        assert_eq!(stream.buffer_index, 0);
        assert_eq!(stream.stride, 8);
        assert_eq!(
            stream.attributes,
            vec![PlannedVertexAttribute {
                location: 0,
                offset: 0,
                format: RenderVertexFormat::Float2,
            }]
        );
        assert_eq!(stream.offset, 0);
        assert_eq!(stream.source.proof_bytes(), quad_vertex_bytes());

        let indices = planned.indices.as_ref().expect("the pass is indexed");
        assert_eq!(indices.format, RenderIndexType::Uint16);
        assert_eq!(indices.index_count, 6);
        assert_eq!(indices.vertex_span, 4);
        assert_eq!(indices.offset, 0);
        assert_eq!(indices.source.proof_bytes(), quad_index_bytes());

        // The format mappings the encoder reads these values through.
        for (format, expected) in [
            (VertexFormat::Float32x2, RenderVertexFormat::Float2),
            (VertexFormat::Float32x3, RenderVertexFormat::Float3),
            (VertexFormat::Float32x4, RenderVertexFormat::Float4),
            (VertexFormat::Uint32, RenderVertexFormat::Uint),
        ] {
            assert_eq!(vertex_format(format), expected);
        }
        assert_eq!(RenderVertexFormat::Float2.name(), "float32x2");
        assert_eq!(RenderVertexFormat::Float3.name(), "float32x3");
        assert_eq!(RenderVertexFormat::Float4.name(), "float32x4");
        assert_eq!(RenderVertexFormat::Uint.name(), "uint32");
        for (format, expected) in [
            (IndexFormat::Uint16, RenderIndexType::Uint16),
            (IndexFormat::Uint32, RenderIndexType::Uint32),
        ] {
            assert_eq!(index_type(format), expected);
            // The footprint proof and the draw's width read the same value.
            assert_eq!(expected.bytes(), format.bytes());
        }
        assert_eq!(RenderIndexType::Uint16.name(), "uint16");
        assert_eq!(RenderIndexType::Uint32.name(), "uint32");
    }

    /// A stream's bytes travel with the pass, so the rail needs no compute
    /// declaration to read them (`research/docs/23` §3.6) — what it does need is
    /// a source it can resolve, which a lease-backed view only is when the
    /// submission carries a lease channel (`research/docs/23` §72, R3d). A pass
    /// planned through [`plan`] has none, so both lease arms keep the slugs this
    /// rail published before the channel existed.
    #[test]
    fn plan_refuses_a_stream_whose_bytes_this_rail_does_not_hold() {
        // The reviewed pass with its stream bytes declared by no pass at all: the
        // device-level helper's shape, which the vertex-input increment admits.
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let planned = plan_pass(&quad_request(&pass, &pipeline))
            .expect("a render input carries its own bytes");
        assert_eq!(
            planned.vertex_streams[0].source.proof_bytes(),
            quad_vertex_bytes()
        );
        assert_eq!(
            planned
                .indices
                .as_ref()
                .map(|indices| indices.source.proof_bytes()),
            Some(quad_index_bytes().as_slice())
        );

        // A lease-backed view has no bytes of its own, and this plan carries no
        // registries to resolve them from: the stream is refused by name and the
        // storage mode it arrived with is part of the refusal.
        let mut leased_pass = quad_pass();
        leased_pass.vertex_buffers[0].source = BufferSource::StagedLease(LeaseId::new(5));
        let error = plan(&quad_request(&leased_pass, &quad_pipeline()), 0, 0).unwrap_err();
        eprintln!("no channel: {error:?}");
        assert_eq!(error.slug, "render_vertex_buffer_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(QUAD_VERTEX_VIEW.get()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );

        // The index half is refused under its own slug, so a capture can tell
        // which of the two inputs the rail could not read.
        let mut borrowed_pass = quad_pass();
        borrowed_pass.indices.as_mut().unwrap().view.source =
            BufferSource::BorrowedNoCopy(LeaseId::new(6));
        let error = plan(&quad_request(&borrowed_pass, &quad_pipeline()), 0, 0).unwrap_err();
        eprintln!("no channel: {error:?}");
        assert_eq!(error.slug, "render_index_buffer_unsupported");
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
    }

    /// A staged lease resolves into the provider's own copy of the owner's
    /// window (`research/docs/23` §72, R3d).
    ///
    /// The trace path's plan is handed the same pair of registries the compute
    /// rail resolves through, so a pass whose streams name staged leases plans
    /// exactly as one that declares its own bytes — and, with no no-copy stream
    /// among them, names no lease to retain. Releasing the staged copy is what
    /// makes the same declaration unreadable, under the registry's own name.
    #[test]
    fn plan_trace_resolves_a_staged_lease_stream_into_the_providers_copy() {
        let epoch = DeviceEpoch::new(3);
        let vertex_lease = LeaseId::new(31);
        let index_lease = LeaseId::new(32);
        let vertex_reservation =
            lease_registration(vertex_lease, QUAD_VERTEX_ALLOCATION, 32, epoch);
        let index_reservation = lease_registration(index_lease, QUAD_INDEX_ALLOCATION, 12, epoch);
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(vertex_reservation, quad_vertex_bytes())
                    .expect("the staged vertex window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        staging
            .import(
                StagedLease::new(index_reservation, quad_index_bytes())
                    .expect("the staged index window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());

        let (mut trace, mut resources) = quad_trace();
        lease_the_quad_trace(
            &mut trace,
            BufferSource::StagedLease(vertex_lease),
            BufferSource::StagedLease(index_lease),
        );
        resources
            .insert_lease(vertex_reservation)
            .expect("the vertex reservation covers its view");
        resources
            .insert_lease(index_reservation)
            .expect("the index reservation covers its view");

        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = quad_contracts();
        let planned = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect("the staged windows hold the reviewed quad");
        let [planned] = planned.as_slice() else {
            panic!("the vertex-input trace carries one render pass");
        };
        let [stream] = planned.plan.vertex_streams.as_slice() else {
            panic!("the reviewed layout binds one stream");
        };
        assert!(
            matches!(stream.source, PlannedInputSource::Staged(_)),
            "a staged lease resolves into the provider's own copy: {:?}",
            stream.source
        );
        assert_eq!(stream.source.proof_bytes(), quad_vertex_bytes());
        let indices = planned.plan.indices.as_ref().expect("the pass is indexed");
        assert!(matches!(indices.source, PlannedInputSource::Staged(_)));
        assert_eq!(indices.source.proof_bytes(), quad_index_bytes());
        assert!(
            planned.plan.borrowed_leases().is_empty(),
            "a staged arm has no owner mapping to retain"
        );

        // The staged copy is the provider's; releasing it is the owner's
        // `LeaseLedger` decision, and the same declaration is refused by name
        // until it is imported again.
        staging
            .release(vertex_lease)
            .expect("the fixture import is released");
        let error = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect_err("a released staged lease cannot be read");
        eprintln!("released staged lease refused: {error:?}");
        assert_eq!(error.slug, "lease_not_imported");
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("lease"),
            Some(&FieldValue::Unsigned(vertex_lease.get()))
        );
    }

    /// A no-copy lease resolves into the owner's own pages (`research/docs/23`
    /// §72, R3d).
    ///
    /// The plan maps the owner's reservation instead of copying it, the
    /// footprint proof reads the owner's bytes — an owner that rewrites its own
    /// index page changes what the proof sees, which a snapshot-style import
    /// could not — and the plan names the leases a submission has to retain. The
    /// guard takes one hold per lease and retires them when it drops, which is
    /// the retirement point this synchronous rail has.
    #[test]
    fn plan_resolves_a_borrowed_lease_stream_into_the_owners_pages() {
        let epoch = DeviceEpoch::new(3);
        let vertex_lease = LeaseId::new(41);
        let index_lease = LeaseId::new(42);
        let vertex_reservation =
            lease_registration(vertex_lease, QUAD_VERTEX_ALLOCATION, OWNER_ALIGNMENT, epoch);
        let index_reservation =
            lease_registration(index_lease, QUAD_INDEX_ALLOCATION, OWNER_ALIGNMENT, epoch);
        let mut owner_vertices =
            OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_vertices.write(&quad_vertex_bytes());
        let mut owner_indices = OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_indices.write(&quad_index_bytes());
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        // Both owner mappings outlive the two imports below and are released
        // after every retain is back to zero, which is what the no-copy
        // registry's contract asks of its caller.
        borrowed
            .import(
                BorrowedLease::new(vertex_reservation, owner_vertices.as_ptr())
                    .expect("the owner's vertex window is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        borrowed
            .import(
                BorrowedLease::new(index_reservation, owner_indices.as_ptr())
                    .expect("the owner's index window is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let resources = owner_resources(
            epoch,
            vertex_reservation,
            index_reservation,
            OWNER_ALIGNMENT,
        );
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let pass = leased_quad_pass(
            BufferSource::BorrowedNoCopy(vertex_lease),
            BufferSource::BorrowedNoCopy(index_lease),
        );
        let pipeline = quad_pipeline();
        let request = quad_request(&pass, &pipeline);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the owner's pages hold the reviewed quad");
        let [stream] = plan.vertex_streams.as_slice() else {
            panic!("the reviewed layout binds one stream");
        };
        let PlannedInputSource::NoCopy { lease, window } = &stream.source else {
            panic!(
                "a no-copy lease resolves into the owner's mapping: {:?}",
                stream.source
            );
        };
        assert_eq!(*lease, vertex_lease);
        assert_eq!(
            window.offset, 0,
            "the view starts at the reservation's base"
        );
        assert_eq!(window.len, quad_vertex_bytes().len());
        assert_eq!(window.base_len, OWNER_ALIGNMENT as usize);
        assert_eq!(stream.binding_offset(), 0);
        assert_eq!(stream.source.proof_bytes(), quad_vertex_bytes());
        let indices = plan.indices.as_ref().expect("the pass is indexed");
        assert!(matches!(
            indices.source,
            PlannedInputSource::NoCopy { lease, .. } if lease == index_lease
        ));
        assert_eq!(indices.source.proof_bytes(), quad_index_bytes());
        assert_eq!(
            plan.borrowed_leases(),
            vec![vertex_lease, index_lease],
            "the plan names one hold per no-copy stream, in binding order"
        );

        // The probe: the owner rewrites its own index page after the import, and
        // the proof reads those bytes instead of a copy taken at import time —
        // the same falsification the Vulkan rail's e2e states with a device.
        owner_indices.write(
            &[0_u16, 1, 2, 2, 1, 37]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<u8>>(),
        );
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("an index the streams do not cover is refused");
        eprintln!("owner-rewritten index window refused: {error:?}");
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(37))
        );
        owner_indices.write(&quad_index_bytes());
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the restored owner window holds the reviewed quad again");

        // One hold per lease while the pass is in flight, and none once Metal
        // has retired it. Both spellings count: an explicit retire and the
        // guard's own drop, which is what an early return after the queue would
        // take.
        let mut retains =
            RenderInputRetains::retain(&borrowed, &plan).expect("both holds are taken");
        assert_eq!(borrowed.outstanding(vertex_lease), Some(1));
        assert_eq!(borrowed.outstanding(index_lease), Some(1));
        retains.retire();
        assert_eq!(borrowed.outstanding(vertex_lease), Some(0));
        assert_eq!(borrowed.outstanding(index_lease), Some(0));
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("both holds are taken");
        drop(retains);
        assert_eq!(borrowed.outstanding(vertex_lease), Some(0));
        assert_eq!(borrowed.outstanding(index_lease), Some(0));

        // A released import is refused by the registry's own name, exactly as a
        // released staged lease is.
        borrowed
            .release(vertex_lease)
            .expect("the fixture import is released");
        borrowed
            .release(index_lease)
            .expect("the fixture import is released");
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released no-copy import cannot be resolved");
        eprintln!("released borrowed lease refused: {error:?}");
        assert_eq!(error.slug, "lease_not_imported");
    }

    /// The milestone trace with its one declaring view naming `source`
    /// (`research/docs/23` §74, R5b).
    ///
    /// The attachment's previous contents are the declaring view's own bytes, so
    /// a lease-backed loading pass names the lease in the compute declaration —
    /// the render pass restates only the attachment's identity and shape
    /// (`research/docs/23` §3.3).
    fn lease_the_milestone_attachment(trace: &mut ComputeTrace, source: BufferSource) {
        for pass in &mut trace.passes {
            if let TracePass::Compute(pass) = pass {
                for view in &mut pass.buffers {
                    if view.view_id == ViewId::new(7) {
                        view.source = source.clone();
                    }
                }
            }
        }
    }

    /// A snapshot carrying the milestone attachment's allocation at whole-page
    /// size, plus the reservation drawn from its first bytes.
    ///
    /// The no-copy arm maps the reservation, not the view, so the fixture's
    /// reservation is page-sized exactly as the production rail's is — at the
    /// alignment the mapping itself needs, which a device test reads from the
    /// device.
    fn owner_attachment_resources(
        epoch: DeviceEpoch,
        reservation: LeaseReservation,
        alignment: u64,
    ) -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(9),
                owner_epoch: epoch,
                size: alignment,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the attachment reservation covers its view");
        resources
    }

    /// A staged lease resolves a loading attachment's previous contents into
    /// the provider's own copy (`research/docs/23` §74, R5b).
    ///
    /// The plan carries the staged bytes exactly as it carries the declaring
    /// view's own, names no lease to retain, and refuses the same declaration
    /// under the registry's own name once the copy is released.
    #[test]
    fn plan_trace_resolves_a_staged_lease_attachment_load_into_the_providers_copy() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(61);
        let reservation = lease_registration(lease_id, AllocationId::new(9), 16, epoch);
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(reservation, vec![0xfe; 16])
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = owner_attachment_resources(epoch, reservation, OWNER_ALIGNMENT);
        // The staged arm is the provider's own copy, so the device's mapping
        // ability plays no part in it.
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: 0,
        };

        let (mut trace, _) = milestone_trace(LoadOp::Load);
        lease_the_milestone_attachment(&mut trace, BufferSource::StagedLease(lease_id));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect("the staged copy carries the attachment's previous contents");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        let [attachment] = planned.plan.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        assert!(
            matches!(attachment.initial, Some(PlannedInputSource::Staged(_))),
            "a staged lease resolves into the provider's own copy: {:?}",
            attachment.initial
        );
        assert_eq!(
            attachment.initial_bytes(),
            Some([0xfe; 16].as_slice()),
            "the staged window is what the preset uploads"
        );
        assert!(
            planned.plan.borrowed_leases().is_empty(),
            "a staged arm has no owner mapping to retain"
        );

        staging
            .release(lease_id)
            .expect("the fixture import is released");
        let error = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect_err("a released staged lease cannot be read");
        eprintln!("released staged attachment lease refused: {error:?}");
        assert_eq!(error.slug, "lease_not_imported");
        assert_eq!(
            error.fields.get("lease"),
            Some(&FieldValue::Unsigned(lease_id.get()))
        );
        // The location the rail could not read is part of the refusal, so a
        // multi-attachment capture can tell which one it was.
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// A no-copy lease resolves a loading attachment's previous contents into
    /// the owner's own pages (`research/docs/23` §74, R5b).
    ///
    /// The proof and the preset read the same mapping: an owner that rewrites
    /// its pages after the import changes both, which a snapshot-style import
    /// could not, and the plan names the lease its submission has to retain.
    #[test]
    fn plan_resolves_a_borrowed_lease_attachment_load_into_the_owners_pages() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(62);
        let reservation =
            lease_registration(lease_id, AllocationId::new(9), OWNER_ALIGNMENT, epoch);
        let mut owner_attachment =
            OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_attachment.write(&[0xfe; 16]);
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, owner_attachment.as_ptr())
                    .expect("the owner's attachment window is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let resources = owner_attachment_resources(epoch, reservation, OWNER_ALIGNMENT);
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };

        let (mut trace, _) = milestone_trace(LoadOp::Load);
        lease_the_milestone_attachment(&mut trace, BufferSource::BorrowedNoCopy(lease_id));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let planned = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect("the owner's pages hold the attachment's previous contents");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        let [attachment] = planned.plan.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        let Some(PlannedInputSource::NoCopy { lease, window }) = &attachment.initial else {
            panic!(
                "a no-copy lease resolves into the owner's mapping: {:?}",
                attachment.initial
            );
        };
        assert_eq!(*lease, lease_id);
        assert_eq!(
            window.offset, 0,
            "the view starts at the reservation's base"
        );
        assert_eq!(window.len, 16);
        assert_eq!(window.base_len, OWNER_ALIGNMENT as usize);
        assert_eq!(attachment.initial_bytes(), Some([0xfe; 16].as_slice()));
        assert_eq!(
            planned.plan.borrowed_leases(),
            vec![lease_id],
            "the plan names one hold for the loading attachment's window"
        );

        // The probe: the owner rewrites its own page after the import, and the
        // preset reads those bytes instead of a copy taken at import time — the
        // same falsification the Vulkan rail's e2e states with a device.
        owner_attachment.write(&[0x37; 16]);
        let planned = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect("the rewritten owner window still holds sixteen bytes");
        assert_eq!(
            planned[0].plan.attachments[0].initial_bytes(),
            Some([0x37; 16].as_slice()),
            "the preset follows the owner's rewritten pages"
        );

        // One hold while the pass is in flight, none once Metal retired it —
        // both spellings, exactly as the stream half measures them.
        let mut retains =
            RenderInputRetains::retain(&borrowed, &planned[0].plan).expect("the hold is taken");
        assert_eq!(borrowed.outstanding(lease_id), Some(1));
        retains.retire();
        assert_eq!(borrowed.outstanding(lease_id), Some(0));
        let retains =
            RenderInputRetains::retain(&borrowed, &planned[0].plan).expect("the hold is taken");
        drop(retains);
        assert_eq!(borrowed.outstanding(lease_id), Some(0));

        // A device that cannot map an owner window refuses the same declaration
        // by the storage mode's own name rather than copying it.
        let unmappable = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: 0,
        };
        let error =
            plan_trace_with_leases(&trace, &pool, &contracts, Some(&unmappable), None, 0, 0)
                .expect_err("a device with no owner mapping cannot read the window");
        eprintln!("no owner mapping refused: {error:?}");
        assert_eq!(error.slug, "storage_mode_unsupported");
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(ViewId::new(7).get()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
    }

    /// The v82 shape at the plan level: a four-sample attachment the pass opens
    /// with `LoadOp::Load` (`research/docs/23` §82).
    ///
    /// Metal's copy commands are single-sample at both ends, so the load is
    /// executed by the encoder's own seed pass, whose clear value is host
    /// state. What these tests pin is everything answerable without a device:
    /// the one uniform texel the plan turns into components, the two shapes it
    /// refuses by name — a window that is not one repeated texel, and an
    /// owner's own mapping, which the seed would read on the host rather than
    /// through the device — and the missing-bytes shape a load with no seed
    /// cannot execute.
    #[test]
    fn plan_states_the_seed_of_a_multisampled_load_and_refuses_the_shapes_it_cannot_read() {
        let seeded: Vec<u8> = [0x22_u8, 0x44, 0x66, 0x89].repeat(4);
        let mut pass = milestone_pass(LoadOp::Load);
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        let pipeline = milestone_pipeline();
        let planned = plan_pass(&milestone_request(&pass, &pipeline, Some(&seeded)))
            .expect("one repeated texel is the seed the seed pass clears with");
        assert_eq!(planned.multisample, Some(SampleCount::Four));
        let [attachment] = planned.attachments.as_slice() else {
            panic!("the milestone renders one attachment");
        };
        // The load itself stays a load: the seed is what the encoder's own
        // render pass clears, not a change of the trace's declaration.
        assert_eq!(attachment.load, RenderLoadAction::Load);
        assert_eq!(
            attachment.seed,
            Some([
                0x22 as f64 / 255.0,
                0x44 as f64 / 255.0,
                0x66 as f64 / 255.0,
                0x89 as f64 / 255.0,
            ]),
            "the seed decodes through the attachment's own format"
        );
        assert_eq!(attachment.initial_bytes(), Some(seeded.as_slice()));

        // A per-texel seed: the clear value is one colour for the whole
        // attachment, and a full-coverage shader that could write every sample
        // is not a module this rail owns.
        let mut mixed = seeded.clone();
        mixed[7] = 0xff;
        let error = plan_pass(&milestone_request(&pass, &pipeline, Some(&mixed)))
            .expect_err("a per-texel seed is refused by name");
        eprintln!("nonuniform multisampled load refused: {error:?}");
        assert_eq!(error.slug, "render_multisample_load_nonuniform_unsupported");
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );

        // No bytes at all: the offscreen shape is refused by the load's own
        // declaration rule — a `Load` that declares nothing has nothing to
        // upload — and the multisampled route adds no second spelling of it.
        let error = plan_pass(&milestone_request(&pass, &pipeline, None))
            .expect_err("a load without declared bytes is refused by name");
        eprintln!("seedless multisampled load refused: {error:?}");
        assert_eq!(error.slug, "render_attachment_initial_mismatch");

        // The multisampled *present* shape is the one arm that reaches the
        // plan with a `Load` and no declared bytes: its n-sample surface is
        // rail-owned and the sentinel preset a single-sample present target
        // carries cannot define a multisampled image, so the shape has no seed
        // to state and is refused by name instead of opening the surface from
        // undefined memory.
        let mut presenting = milestone_pass(LoadOp::Load);
        presenting.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        presenting.present = Some(PresentDescriptor {
            target: PresentTarget {
                allocation_id: AllocationId::new(9),
                view_id: ViewId::new(7),
                format: AttachmentFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                image_count: 1,
                initial: InitialState::Undefined,
            },
            source: ViewId::new(7),
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        let error = plan_pass(&milestone_request(&presenting, &pipeline, None))
            .expect_err("a multisampled present load has no seed to state");
        eprintln!("present multisampled load refused: {error:?}");
        assert_eq!(error.slug, "render_multisample_load_unsupported");

        // The single-sample load keeps the upload it always had: the seed is
        // the multisampled route's own fact, and a pre-v82 pass states none.
        let mut single = milestone_pass(LoadOp::Load);
        single.multisample = None;
        let planned = plan_pass(&milestone_request(&single, &pipeline, Some(&seeded)))
            .expect("a single-sample load keeps its upload");
        assert_eq!(planned.attachments[0].seed, None);
    }

    /// An owner's own mapping is not a seed the multisampled route can read
    /// (`research/docs/23` §82): the clear value is host state, so the borrowed
    /// window would be a snapshot of the owner's pages rather than the device
    /// read §74 promises. The staged arm states the same bytes through the
    /// provider's own copy and is admitted.
    #[test]
    fn plan_refuses_a_borrowed_seed_and_admits_the_staged_one() {
        let epoch = DeviceEpoch::new(3);
        let seed: Vec<u8> = [0x22_u8, 0x44, 0x66, 0x89].repeat(4);
        let borrow_id = LeaseId::new(64);
        let borrow_reservation =
            lease_registration(borrow_id, AllocationId::new(9), OWNER_ALIGNMENT, epoch);
        let mut owner_attachment =
            OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_attachment.write(&seed);
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(borrow_reservation, owner_attachment.as_ptr())
                    .expect("the owner's attachment window is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let staged_id = LeaseId::new(63);
        let staged_reservation = lease_registration(staged_id, AllocationId::new(9), 16, epoch);
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(staged_reservation, seed.clone())
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let resources = owner_attachment_resources(epoch, borrow_reservation, OWNER_ALIGNMENT);
        let staged_resources =
            owner_attachment_resources(epoch, staged_reservation, OWNER_ALIGNMENT);
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let (mut trace, _) = milestone_trace(LoadOp::Load);
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        lease_the_milestone_attachment(&mut trace, BufferSource::BorrowedNoCopy(borrow_id));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = milestone_contracts();
        let error = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect_err("a borrowed seed would be read on the host");
        eprintln!("borrowed multisampled seed refused: {error:?}");
        assert_eq!(error.slug, "render_multisample_load_borrowed_unsupported");
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );

        // The staged arm states the same bytes through the provider's own copy,
        // so the seed route reads them exactly as it reads declared bytes.
        let (mut staged_trace, _) = milestone_trace(LoadOp::Load);
        let Some(TracePass::Render(pass)) = staged_trace.passes.last_mut() else {
            panic!("the fixture ends with its render pass");
        };
        pass.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        lease_the_milestone_attachment(&mut staged_trace, BufferSource::StagedLease(staged_id));
        let pool = staged_trace
            .serial_resources()
            .expect("admitted serial pool");
        let staged_leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &staged_resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let planned = plan_trace_with_leases(
            &staged_trace,
            &pool,
            &contracts,
            Some(&staged_leases),
            None,
            0,
            0,
        )
        .expect("a staged seed is the provider's own copy");
        assert_eq!(
            planned[0].plan.attachments[0].seed,
            Some([
                0x22 as f64 / 255.0,
                0x44 as f64 / 255.0,
                0x66 as f64 / 255.0,
                0x89 as f64 / 255.0,
            ])
        );
    }

    /// The sixteen texels the reviewed sampling fixture uploads
    /// (`research/docs/23` §3.3, v70): one distinct texel per position.
    fn sampled_texels() -> Vec<u8> {
        (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [x, y, x.wrapping_add(y), 0xff]))
            .collect()
    }

    /// The sixteen texels the owner's rewritten window holds
    /// (`research/docs/23` §75, R5c): four distinct channels per texel, none of
    /// them equal to the fixture's own, so "the pass read the owner's pages
    /// after the rewrite" is falsifiable per texel.
    fn rewritten_sampled_texels() -> Vec<u8> {
        (0..4u8)
            .flat_map(|y| (0..4u8).flat_map(move |x| [0x80 | x, 0x40 | y, x ^ y, 0xff]))
            .collect()
    }

    /// The reviewed sampled pass with its texture bound to `source`
    /// (`research/docs/23` §75, R5c).
    fn leased_sampled_pass(source: TextureSource) -> RenderPassDescriptor {
        let mut pass = sampled_pass(4);
        pass.textures[0].source = source;
        pass
    }

    /// The sampled pass's request, as the device-level helper's shape builds
    /// it.
    fn sampled_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_SAMPLED_SOURCE,
            initial: vec![None],
            resident: Vec::new(),
        }
    }

    /// A snapshot carrying the sampled texture's allocation at whole-page size,
    /// plus the reservation drawn from its first bytes
    /// (`research/docs/23` §75, R5c).
    ///
    /// The owner's reservation is page-sized exactly as the production rail's
    /// is — the import rules make owners align their windows — while the
    /// texture is only its first sixty-four bytes.
    fn owner_texture_resources(
        epoch: DeviceEpoch,
        reservation: LeaseReservation,
        alignment: u64,
    ) -> ResourceTableSnapshot {
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(53),
                owner_epoch: epoch,
                size: alignment,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(reservation)
            .expect("the texture reservation covers its view");
        resources
    }

    /// A staged lease resolves a sampled texture's texels into the provider's
    /// own copy (`research/docs/23` §75, R5c).
    ///
    /// The plan carries the staged bytes exactly as it carries the texture's
    /// own, names no lease to retain, and refuses the same declaration under the
    /// registry's own name once the copy is released. The window is the
    /// texture's own extent at the reservation's start, so the page-aligned
    /// padding behind it stays unread.
    #[test]
    fn plan_resolves_a_staged_lease_render_texture_into_the_providers_copy() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(71);
        let reservation =
            lease_registration(lease_id, AllocationId::new(53), OWNER_ALIGNMENT, epoch);
        let texels = sampled_texels();
        let mut staged_bytes = vec![0x5a_u8; OWNER_ALIGNMENT as usize];
        staged_bytes[..texels.len()].copy_from_slice(&texels);
        let staging = LeaseRegistry::new();
        staging
            .import(
                StagedLease::new(reservation, staged_bytes)
                    .expect("the staged window matches its reservation"),
            )
            .expect("the fixture import is accepted");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = owner_texture_resources(epoch, reservation, OWNER_ALIGNMENT);
        // The staged arm is the provider's own copy, so the device's mapping
        // ability plays no part in it.
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: 0,
        };

        let pipeline = sampled_pipeline();
        let pass = leased_sampled_pass(TextureSource::StagedLease(lease_id));
        let request = sampled_request(&pass, &pipeline);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the staged copy carries the texture's texels");
        assert!(
            matches!(plan.textures[0].source, PlannedInputSource::Staged(_)),
            "a staged lease resolves into the provider's own copy: {:?}",
            plan.textures[0].source
        );
        assert_eq!(
            plan.textures[0].source.proof_bytes(),
            texels.as_slice(),
            "the window is the texture's own extent at the reservation's start"
        );
        assert!(
            plan.borrowed_leases().is_empty(),
            "a staged arm has no owner mapping to retain"
        );

        staging
            .release(lease_id)
            .expect("the fixture import is released");
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released staged lease cannot be read");
        eprintln!("released staged texture lease refused: {error:?}");
        assert_eq!(error.slug, "lease_not_imported");
        assert_eq!(
            error.fields.get("lease"),
            Some(&FieldValue::Unsigned(lease_id.get()))
        );
    }

    /// A no-copy lease resolves a sampled texture's texels into the owner's own
    /// pages (`research/docs/23` §75, R5c).
    ///
    /// The proof and the encoder's upload read the same mapping: an owner that
    /// rewrites its pages after the import changes both, which a snapshot-style
    /// import could not, and the plan names the lease its submission has to
    /// retain.
    #[test]
    fn plan_resolves_a_borrowed_lease_render_texture_into_the_owners_pages() {
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(72);
        let reservation =
            lease_registration(lease_id, AllocationId::new(53), OWNER_ALIGNMENT, epoch);
        let texels = sampled_texels();
        let mut owner_texture = OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_texture.write(&texels);
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, owner_texture.as_ptr())
                    .expect("the owner's texture window is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let resources = owner_texture_resources(epoch, reservation, OWNER_ALIGNMENT);
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };

        let pipeline = sampled_pipeline();
        let pass = leased_sampled_pass(TextureSource::BorrowedNoCopy(lease_id));
        let request = sampled_request(&pass, &pipeline);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the owner's pages hold the texture's texels");
        let PlannedInputSource::NoCopy { lease, window } = &plan.textures[0].source else {
            panic!(
                "a no-copy lease resolves into the owner's mapping: {:?}",
                plan.textures[0].source
            );
        };
        assert_eq!(*lease, lease_id);
        assert_eq!(
            window.offset, 0,
            "the texture starts at the reservation's base"
        );
        assert_eq!(
            window.len,
            texels.len(),
            "the window is the texture's own extent, not the page-aligned reservation"
        );
        assert_eq!(window.base_len, OWNER_ALIGNMENT as usize);
        assert_eq!(plan.textures[0].source.proof_bytes(), texels.as_slice());
        assert_eq!(
            plan.borrowed_leases(),
            vec![lease_id],
            "the plan names one hold for the texture's window"
        );

        // The probe: the owner rewrites its own page after the import, and the
        // upload reads those bytes instead of a copy taken at import time — the
        // same falsification the Vulkan rail's e2e states with a device.
        let rewritten = rewritten_sampled_texels();
        owner_texture.write(&rewritten);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the rewritten owner window still holds a 4x4 surface");
        assert_eq!(
            plan.textures[0].source.proof_bytes(),
            rewritten.as_slice(),
            "the upload follows the owner's rewritten pages"
        );

        // One hold while the pass is in flight, none once Metal retired it —
        // both spellings, exactly as the attachment half measures them.
        let mut retains = RenderInputRetains::retain(&borrowed, &plan).expect("the hold is taken");
        assert_eq!(borrowed.outstanding(lease_id), Some(1));
        retains.retire();
        assert_eq!(borrowed.outstanding(lease_id), Some(0));
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("the hold is taken");
        drop(retains);
        assert_eq!(borrowed.outstanding(lease_id), Some(0));

        // A released import is the registry's own refusal.
        borrowed
            .release(lease_id)
            .expect("the owner's texture window is released");
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released no-copy import cannot be resolved");
        assert_eq!(error.slug, "lease_not_imported");
    }

    /// A lease-backed sampled texture with no channel is refused under the
    /// sampler's source name with the storage mode it arrived under, and a
    /// device that cannot read an owner window is refused under the storage
    /// mode's own name (`research/docs/23` §75, R5c).
    #[test]
    fn a_lease_backed_render_texture_is_refused_without_a_channel_or_a_mapping() {
        let pipeline = sampled_pipeline();
        let pass = leased_sampled_pass(TextureSource::BorrowedNoCopy(LeaseId::new(73)));
        let request = sampled_request(&pass, &pipeline);
        let error = plan_pass(&request).expect_err("a lease needs the channel that imported it");
        eprintln!("no channel: {error:?}");
        assert_eq!(error.slug, "render_texture_source_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(error.fields.get("binding"), Some(&FieldValue::Unsigned(0)));
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(ViewId::new(83).get()))
        );

        // A device that cannot read an owner window refuses the borrowed arm
        // under the storage mode's published name, with the window's identity,
        // exactly as the stream and attachment roles do.
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = ResourceTableSnapshot::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: DeviceEpoch::new(3),
            host_import_alignment: 0,
        };
        let error =
            plan_with_leases(&request, Some(&leases), 0, 0).expect_err("no owner mapping path");
        eprintln!("no owner mapping refused: {error:?}");
        assert_eq!(error.slug, "storage_mode_unsupported");
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(ViewId::new(83).get()))
        );
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );

        // The whole-page rule is the one every no-copy window meets, the
        // texture's own window included: a reservation that covers the texture
        // but misses the device's import alignment is refused by name instead
        // of mapped.
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(74);
        let short = lease_registration(lease_id, AllocationId::new(53), 128, epoch);
        let mut owner_texture = OwnerPages::new(OWNER_ALIGNMENT as usize, OWNER_ALIGNMENT as usize);
        owner_texture.write(&sampled_texels());
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(short, owner_texture.as_ptr())
                    .expect("a 128-byte reservation is a valid lease window"),
            )
            .expect("the fixture import is accepted");
        let mut resources = ResourceTableSnapshot::new();
        resources
            .insert_allocation(AllocationRecord {
                allocation_id: AllocationId::new(53),
                owner_epoch: epoch,
                size: OWNER_ALIGNMENT,
            })
            .expect("the fixture allocation is well formed");
        resources
            .insert_lease(short)
            .expect("the short reservation is admitted");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let pass = leased_sampled_pass(TextureSource::BorrowedNoCopy(lease_id));
        let request = sampled_request(&pass, &pipeline);
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a 128-byte reservation cannot map as a whole page");
        eprintln!("short reservation refused: {error:?}");
        assert_eq!(error.slug, "lease_length_unsupported");
        assert_eq!(error.fields.get("lease"), Some(&FieldValue::Unsigned(74)));
    }

    /// The reviewed stream collapsed onto one corner.
    ///
    /// Every fragment is degenerate, so the draw covers no texel and the
    /// attachment keeps the clear sentinel. That is the observation an owner
    /// rewrite gives a device: a rail that had snapshotted the owner's pages at
    /// import time would still draw the original quad.
    fn collapsed_quad_vertex_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for _ in 0..4 {
            bytes.extend_from_slice(&(-1.0_f32).to_le_bytes());
            bytes.extend_from_slice(&(-1.0_f32).to_le_bytes());
        }
        bytes
    }

    /// The reviewed stream's four vertices moved into the left column:
    /// `(-1,-1) (0,-1) (-1,1) (0,1)`. With the six reviewed indices the two
    /// triangles cover exactly two of the four texels, which leaves the other
    /// two for a loading attachment's previous contents to show through.
    ///
    /// The band is chosen to be symmetric under the NDC y flip the two rails
    /// disagree about — Vulkan's y points down, Metal's points up — so both
    /// rails cover the same texel column and the byte expectation stays
    /// identical (`crates/metal-api-vulkan/tests/render_e2e.rs`).
    #[cfg(target_os = "macos")]
    fn left_column_quad_vertex_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for (x, y) in [(-1.0_f32, -1.0_f32), (0.0, -1.0), (-1.0, 1.0), (0.0, 1.0)] {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        bytes
    }

    /// Name one source for the quad's vertex stream in the declaration pass and
    /// the render pass alike, which is what a serial trace asks of two
    /// declarations of one view (`SerialBufferRebinding`).
    #[cfg(target_os = "macos")]
    fn re_source_quad_vertices(trace: &mut ComputeTrace, source: BufferSource) {
        for pass in &mut trace.passes {
            match pass {
                TracePass::Landing(_) => {}
                TracePass::Compute(pass) => {
                    for view in &mut pass.buffers {
                        if view.view_id == QUAD_VERTEX_VIEW {
                            view.source = source.clone();
                        }
                    }
                }
                TracePass::Render(pass) => {
                    for view in &mut pass.vertex_buffers {
                        if view.view_id == QUAD_VERTEX_VIEW {
                            view.source = source.clone();
                        }
                    }
                }
            }
        }
    }

    /// The word the owner's page holds as the attachment's previous contents
    /// (`research/docs/23` §74, R5b): four bytes no reviewed module produces and
    /// no clear sentinel spells, so "the preset came from the owner" is
    /// falsifiable per texel.
    #[cfg(target_os = "macos")]
    const OWNER_ATTACHMENT_WORD: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

    /// A byte string as the evidence logs spell it.
    #[cfg(target_os = "macos")]
    fn hex(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The default Metal device a device-only test may run on, or `None` on a
    /// host whose device is not the unified-memory Apple GPU the provider
    /// requires (`native.rs::new`). The macOS CI job is where this returns
    /// `Some`; every other host prints the skip and returns.
    #[cfg(target_os = "macos")]
    fn eligible_apple_device() -> Option<Device> {
        let device = Device::system_default()?;
        if device.name().trim().is_empty()
            || !device.has_unified_memory()
            || !device.supports_family(metal::MTLGPUFamily::Apple4)
        {
            return None;
        }
        Some(device)
    }

    /// The no-copy render input on a real Metal device (`research/docs/23` §72,
    /// R3d).
    ///
    /// Device-only, so the macOS CI job's `cargo test -p metal-api-native` is the
    /// observation: the check is the attachment's own texels, which no host
    /// without Metal can produce. Three facts are measured here that the
    /// host-side tests cannot reach — `newBufferWithBytesNoCopy:` maps the
    /// owner's reservation, the draw reads those pages (`0x40 0x80 0xc0 0xff`
    /// over the 2x2 attachment, byte for byte the value the declared-bytes
    /// fixture lands), and an owner rewrite afterwards changes the draw's output
    /// instead of leaving the import's bytes behind. The retain guard is
    /// measured with the registry: both holds are back to zero once the pass's
    /// command buffer is terminal.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_borrowed_lease_stream_draws_the_owners_pages_on_a_device() {
        let Some(device) = eligible_apple_device() else {
            eprintln!("skipping native render-lease test: no eligible Metal device");
            return;
        };
        let queue = device.new_command_queue();
        // The mapping's own alignment is the device's page size, which is what
        // `no_copy_alignment` publishes and `import_borrowed_lease` checks; the
        // host-side tests use 4 KiB because they never map anything.
        let alignment = crate::native::page_size();
        if alignment == 0 {
            eprintln!("skipping native render-lease test: the device reports no page size");
            return;
        }
        let epoch = DeviceEpoch::new(3);
        let vertex_lease = LeaseId::new(71);
        let index_lease = LeaseId::new(72);
        let vertex_reservation =
            lease_registration(vertex_lease, QUAD_VERTEX_ALLOCATION, alignment, epoch);
        let index_reservation =
            lease_registration(index_lease, QUAD_INDEX_ALLOCATION, alignment, epoch);
        let mut owner_vertices = OwnerPages::new(alignment as usize, alignment as usize);
        owner_vertices.write(&quad_vertex_bytes());
        let mut owner_indices = OwnerPages::new(alignment as usize, alignment as usize);
        owner_indices.write(&quad_index_bytes());
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(vertex_reservation, owner_vertices.as_ptr())
                    .expect("the owner's vertex window is a valid reservation"),
            )
            .expect("the owner's vertex window is imported");
        borrowed
            .import(
                BorrowedLease::new(index_reservation, owner_indices.as_ptr())
                    .expect("the owner's index window is a valid reservation"),
            )
            .expect("the owner's index window is imported");
        let resources = owner_resources(epoch, vertex_reservation, index_reservation, alignment);
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: alignment,
        };
        let pass = leased_quad_pass(
            BufferSource::BorrowedNoCopy(vertex_lease),
            BufferSource::BorrowedNoCopy(index_lease),
        );
        let pipeline = quad_pipeline();
        let request = quad_request(&pass, &pipeline);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the owner's pages hold the reviewed quad");
        assert_eq!(plan.borrowed_leases(), vec![vertex_lease, index_lease]);

        // The pass is synchronous, so dropping the guard after the encoder
        // returns is the retirement point the provider's own `execute_render_passes`
        // takes: nothing else holds the owner's mapping when the texels arrive.
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("both holds are taken");
        assert_eq!(borrowed.outstanding(vertex_lease), Some(1));
        let readback = encode_offscreen_render(&device, &queue, &plan)
            .expect("the no-copy mapping draws the reviewed quad");
        drop(retains);
        let attachment = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!("borrowed lease attachment: {}", hex(&attachment));
        assert_eq!(
            attachment,
            EXPECTED_TEXEL_BYTES.repeat(4),
            "the device draws the owner's pages through the mapping"
        );
        assert_eq!(
            borrowed.outstanding(vertex_lease),
            Some(0),
            "the vertex hold is retired once the pass is terminal"
        );
        assert_eq!(borrowed.outstanding(index_lease), Some(0));

        // The owner rewrites its own vertex page: a rail that had snapshotted
        // the window at import time would still draw the original quad, while
        // the mapping reads the pages the device reads.
        owner_vertices.write(&collapsed_quad_vertex_bytes());
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("both holds are taken");
        let readback = encode_offscreen_render(&device, &queue, &plan)
            .expect("the rewritten window still maps");
        drop(retains);
        let collapsed = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!(
            "owner-rewritten vertex window readback: {}",
            hex(&collapsed)
        );
        assert_eq!(
            collapsed,
            [0xfe_u8; 4].repeat(4),
            "the draw follows the owner's rewritten pages down to the clear sentinel"
        );

        // A released import is the registry's own refusal, the same name the
        // host-side tests assert without a device.
        borrowed
            .release(vertex_lease)
            .expect("the owner's vertex window is released");
        borrowed
            .release(index_lease)
            .expect("the owner's index window is released");
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released no-copy import cannot be resolved");
        assert_eq!(error.slug, "lease_not_imported");
    }

    /// The no-copy attachment preset on a real Metal device (`research/docs/23`
    /// §74, R5b).
    ///
    /// Device-only, so the macOS CI job's `cargo test -p metal-api-native` is
    /// the observation: three facts the host-side tests cannot reach — the
    /// encoder uploads the owner's pages into the attachment before the pass
    /// opens, the draw leaves the owner's bytes in the texels it does not
    /// cover, and an owner rewrite afterwards changes those texels instead of
    /// leaving the import's first copy behind. The retain guard is measured
    /// with the registry as well.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_borrowed_lease_attachment_preset_reads_the_owners_pages_on_a_device() {
        let Some(device) = eligible_apple_device() else {
            eprintln!("skipping native render-lease test: no eligible Metal device");
            return;
        };
        let queue = device.new_command_queue();
        let alignment = crate::native::page_size();
        if alignment == 0 {
            eprintln!("skipping native render-lease test: the device reports no page size");
            return;
        }
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(81);
        let reservation = lease_registration(lease_id, AllocationId::new(9), alignment, epoch);
        let mut owner_attachment = OwnerPages::new(alignment as usize, alignment as usize);
        owner_attachment.write(&OWNER_ATTACHMENT_WORD.repeat(4));
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, owner_attachment.as_ptr())
                    .expect("the owner's attachment window is a valid reservation"),
            )
            .expect("the owner's attachment window is imported");
        let resources = owner_attachment_resources(epoch, reservation, alignment);
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: alignment,
        };

        // The reviewed quad trace with two changes: its attachment opens with
        // `LoadOp::Load` over the owner's window, and its stream covers only the
        // left column, so two texels keep whatever the preset uploaded.
        let (mut trace, _) = quad_trace();
        lease_the_milestone_attachment(&mut trace, BufferSource::BorrowedNoCopy(lease_id));
        re_source_quad_vertices(
            &mut trace,
            BufferSource::OwnedBytes(left_column_quad_vertex_bytes()),
        );
        for pass in &mut trace.passes {
            if let TracePass::Render(pass) = pass {
                pass.color_attachments[0].load = LoadOp::Load;
            }
        }
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = quad_contracts();
        let planned = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect("the owner's pages hold the attachment's previous contents");
        let [planned] = planned.as_slice() else {
            panic!("the quad trace carries one render pass");
        };
        assert_eq!(
            planned.plan.borrowed_leases(),
            vec![lease_id],
            "the loading attachment's own window is the one hold this pass takes"
        );

        // The pass is synchronous, so dropping the guard after the encoder
        // returns is the retirement point the provider's own
        // `execute_render_passes` takes.
        let retains =
            RenderInputRetains::retain(&borrowed, &planned.plan).expect("the hold is taken");
        assert_eq!(borrowed.outstanding(lease_id), Some(1));
        let readback = encode_offscreen_render(&device, &queue, &planned.plan)
            .expect("the owner's window presets the attachment");
        drop(retains);
        let attachment = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!("loading attachment readback: {}", hex(&attachment));
        let covered = attachment
            .chunks_exact(4)
            .filter(|texel| *texel == EXPECTED_TEXEL_BYTES)
            .count();
        let loaded = attachment
            .chunks_exact(4)
            .filter(|texel| *texel == OWNER_ATTACHMENT_WORD)
            .count();
        assert_eq!(
            (covered, loaded),
            (2, 2),
            "the left column is drawn and the rest keeps the owner's bytes: {}",
            hex(&attachment)
        );
        assert_eq!(
            borrowed.outstanding(lease_id),
            Some(0),
            "the attachment hold is retired once the pass is terminal"
        );

        // A rail that had snapshotted the owner's page at import time would
        // keep presetting the first word; the rewrite reaches the texels the
        // draw leaves alone instead.
        let rewritten_word = [0x55_u8, 0x66, 0x77, 0x88];
        owner_attachment.write(&rewritten_word.repeat(4));
        let retains =
            RenderInputRetains::retain(&borrowed, &planned.plan).expect("the hold is taken");
        let readback = encode_offscreen_render(&device, &queue, &planned.plan)
            .expect("the rewritten owner window still presets the attachment");
        drop(retains);
        let rewritten = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!(
            "owner-rewritten attachment window readback: {}",
            hex(&rewritten)
        );
        let covered = rewritten
            .chunks_exact(4)
            .filter(|texel| *texel == EXPECTED_TEXEL_BYTES)
            .count();
        let loaded = rewritten
            .chunks_exact(4)
            .filter(|texel| *texel == rewritten_word)
            .count();
        assert_eq!(
            (covered, loaded),
            (2, 2),
            "the preset follows the owner's rewritten pages: {}",
            hex(&rewritten)
        );
        assert!(
            !rewritten
                .chunks_exact(4)
                .any(|texel| texel == OWNER_ATTACHMENT_WORD),
            "no texel keeps the pre-rewrite word a snapshot would have pinned: {}",
            hex(&rewritten)
        );

        // A released import is the registry's own refusal, the same name the
        // host-side tests assert without a device.
        borrowed
            .release(lease_id)
            .expect("the owner's attachment window is released");
        let error = plan_trace_with_leases(&trace, &pool, &contracts, Some(&leases), None, 0, 0)
            .expect_err("a released no-copy import cannot be resolved");
        assert_eq!(error.slug, "lease_not_imported");
    }

    /// The no-copy sampled texture on a real Metal device (`research/docs/23`
    /// §75, R5c).
    ///
    /// Device-only, so the macOS CI job's `cargo test -p metal-api-native` is
    /// the observation: three facts the host-side tests cannot reach — the
    /// encoder uploads the owner's pages into the sampled texture, the fragment
    /// stage samples those texels into the attachment (byte for byte the value
    /// the declared-bytes fixture lands), and an owner rewrite afterwards
    /// changes the landing instead of leaving the import's first copy behind.
    /// The retain guard is measured with the registry as well.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_borrowed_lease_texture_samples_the_owners_pages_on_a_device() {
        let Some(device) = eligible_apple_device() else {
            eprintln!("skipping native render-lease test: no eligible Metal device");
            return;
        };
        let queue = device.new_command_queue();
        // The mapping's own alignment is the device's page size, which is what
        // `no_copy_alignment` publishes and `import_borrowed_lease` checks; the
        // host-side tests use 4 KiB because they never map anything.
        let alignment = crate::native::page_size();
        if alignment == 0 {
            eprintln!("skipping native render-lease test: the device reports no page size");
            return;
        }
        let epoch = DeviceEpoch::new(3);
        let lease_id = LeaseId::new(82);
        let reservation = lease_registration(lease_id, AllocationId::new(53), alignment, epoch);
        let texels = sampled_texels();
        let mut owner_texture = OwnerPages::new(alignment as usize, alignment as usize);
        owner_texture.write(&texels);
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, owner_texture.as_ptr())
                    .expect("the owner's texture window is a valid reservation"),
            )
            .expect("the owner's texture window is imported");
        let resources = owner_texture_resources(epoch, reservation, alignment);
        let staging = LeaseRegistry::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: alignment,
        };
        let pipeline = sampled_pipeline();
        let pass = leased_sampled_pass(TextureSource::BorrowedNoCopy(lease_id));
        let request = sampled_request(&pass, &pipeline);
        let plan = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect("the owner's pages hold the texture's texels");
        assert_eq!(
            plan.borrowed_leases(),
            vec![lease_id],
            "the texture's own window is the one hold this pass takes"
        );

        // The pass is synchronous, so dropping the guard after the encoder
        // returns is the retirement point the provider's own
        // `execute_render_passes` takes.
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("the hold is taken");
        assert_eq!(borrowed.outstanding(lease_id), Some(1));
        let readback = encode_offscreen_render(&device, &queue, &plan)
            .expect("the owner's texels sample into the attachment");
        drop(retains);
        let attachment = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!("borrowed lease texture readback: {}", hex(&attachment));
        assert_eq!(
            attachment,
            texels,
            "the fragment stage samples the owner's pages through the mapping: {}",
            hex(&attachment)
        );
        assert_eq!(
            borrowed.outstanding(lease_id),
            Some(0),
            "the texture hold is retired once the pass is terminal"
        );

        // A rail that had snapshotted the owner's window at import time — or
        // that had uploaded it into its own texture while building the pass —
        // would keep sampling the first texels; the rewrite reaches every texel
        // the draw reads instead.
        let rewritten = rewritten_sampled_texels();
        owner_texture.write(&rewritten);
        let retains = RenderInputRetains::retain(&borrowed, &plan).expect("the hold is taken");
        let readback = encode_offscreen_render(&device, &queue, &plan)
            .expect("the rewritten owner window still samples");
        drop(retains);
        let rewritten_attachment = readback
            .attachments
            .into_iter()
            .next()
            .expect("the pass stores its one attachment");
        eprintln!(
            "owner-rewritten texture window readback: {}",
            hex(&rewritten_attachment)
        );
        assert_eq!(
            rewritten_attachment,
            rewritten,
            "the sampled texels follow the owner's rewritten pages: {}",
            hex(&rewritten_attachment)
        );

        // A released import is the registry's own refusal, the same name the
        // host-side tests assert without a device.
        borrowed
            .release(lease_id)
            .expect("the owner's texture window is released");
        let error = plan_with_leases(&request, Some(&leases), 0, 0)
            .expect_err("a released no-copy import cannot be resolved");
        assert_eq!(error.slug, "lease_not_imported");
    }

    /// A device that cannot map an owner window refuses a no-copy render input
    /// under the name core admission and the compute rail publish for the same
    /// fact (`research/docs/23` §72, R3d): `storage_mode_unsupported`, with the
    /// storage mode the view arrived under.
    #[test]
    fn a_borrowed_render_input_is_refused_without_host_mapping() {
        let staging = LeaseRegistry::new();
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        let resources = ResourceTableSnapshot::new();
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: DeviceEpoch::new(3),
            host_import_alignment: 0,
        };
        let pass = leased_quad_pass(
            BufferSource::BorrowedNoCopy(LeaseId::new(51)),
            quad_index_view().source,
        );
        let pipeline = quad_pipeline();
        let error = plan_with_leases(&quad_request(&pass, &pipeline), Some(&leases), 0, 0)
            .expect_err("a device without a no-copy path cannot bind the owner's window");
        eprintln!("no host mapping: {error:?}");
        assert_eq!(error.slug, "storage_mode_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("borrowed_no_copy".to_owned()))
        );
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(QUAD_VERTEX_VIEW.get()))
        );
    }

    /// An owner window that misses one of the mapping's own rules is refused by
    /// name before a single page is mapped (`research/docs/23` §72, R3d).
    ///
    /// Metal maps whole pages, so the reservation's base and length are held to
    /// the import alignment and the view inside it to the 4-byte rule a binding
    /// uses; all three are the checks the compute rail states for the same
    /// mapping, spelled once in `lib.rs`.
    #[test]
    fn a_borrowed_render_input_is_refused_when_its_window_misses_the_mapping_rules() {
        let epoch = DeviceEpoch::new(3);
        let staging = LeaseRegistry::new();

        // A base address one byte past the alignment.
        let lease_id = LeaseId::new(61);
        let reservation =
            lease_registration(lease_id, QUAD_VERTEX_ALLOCATION, OWNER_ALIGNMENT, epoch);
        let resources = owner_resources(
            epoch,
            reservation,
            lease_registration(
                LeaseId::new(62),
                QUAD_INDEX_ALLOCATION,
                OWNER_ALIGNMENT,
                epoch,
            ),
            OWNER_ALIGNMENT,
        );
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(reservation, 0x2000 + 1)
                    .expect("a non-null owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let pass = leased_quad_pass(
            BufferSource::BorrowedNoCopy(lease_id),
            quad_index_view().source,
        );
        let pipeline = quad_pipeline();
        let error = plan_with_leases(&quad_request(&pass, &pipeline), Some(&leases), 0, 0)
            .expect_err("a misaligned owner base cannot be mapped");
        eprintln!("misaligned owner window refused: {error:?}");
        assert_eq!(error.slug, "lease_alignment_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("pointer"),
            Some(&FieldValue::Unsigned(0x2000 + 1))
        );
        assert_eq!(
            error.fields.get("alignment"),
            Some(&FieldValue::Unsigned(OWNER_ALIGNMENT))
        );
        assert_eq!(
            error.fields.get("lease"),
            Some(&FieldValue::Unsigned(lease_id.get()))
        );

        // A reservation that is not a whole number of pages.
        let short = lease_registration(lease_id, QUAD_VERTEX_ALLOCATION, 32, epoch);
        let mut short_resources = ResourceTableSnapshot::new();
        short_resources
            .insert_allocation(AllocationRecord {
                allocation_id: QUAD_VERTEX_ALLOCATION,
                owner_epoch: epoch,
                size: 32,
            })
            .expect("the short fixture allocation is well formed");
        short_resources
            .insert_allocation(AllocationRecord {
                allocation_id: QUAD_INDEX_ALLOCATION,
                owner_epoch: epoch,
                size: OWNER_ALIGNMENT,
            })
            .expect("the index fixture allocation is well formed");
        short_resources
            .insert_lease(short)
            .expect("the short reservation covers its view");
        short_resources
            .insert_lease(lease_registration(
                LeaseId::new(62),
                QUAD_INDEX_ALLOCATION,
                OWNER_ALIGNMENT,
                epoch,
            ))
            .expect("the index reservation covers its view");
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(short, 0x2000)
                    .expect("a page-aligned owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &short_resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let error = plan_with_leases(&quad_request(&pass, &pipeline), Some(&leases), 0, 0)
            .expect_err("a reservation that is not a page multiple cannot be mapped");
        eprintln!("non-page reservation refused: {error:?}");
        assert_eq!(error.slug, "lease_length_unsupported");
        assert_eq!(error.fields.get("length"), Some(&FieldValue::Unsigned(32)));
        assert_eq!(
            error.fields.get("alignment"),
            Some(&FieldValue::Unsigned(OWNER_ALIGNMENT))
        );

        // A view that starts off the 4-byte grid inside the reservation.
        let page = lease_registration(lease_id, QUAD_VERTEX_ALLOCATION, OWNER_ALIGNMENT, epoch);
        let resources = owner_resources(
            epoch,
            page,
            lease_registration(
                LeaseId::new(62),
                QUAD_INDEX_ALLOCATION,
                OWNER_ALIGNMENT,
                epoch,
            ),
            OWNER_ALIGNMENT,
        );
        let borrowed = Arc::new(BorrowedLeaseRegistry::new());
        borrowed
            .import(
                BorrowedLease::new(page, 0x2000)
                    .expect("a page-aligned owner pointer is a valid reservation"),
            )
            .expect("the fixture import is accepted");
        let leases = RenderLeaseContext {
            staging: &staging,
            borrowed: &borrowed,
            resources: &resources,
            device_epoch: epoch,
            host_import_alignment: OWNER_ALIGNMENT,
        };
        let mut pass = leased_quad_pass(
            BufferSource::BorrowedNoCopy(lease_id),
            quad_index_view().source,
        );
        pass.vertex_buffers[0].offset = 2;
        let error = plan_with_leases(&quad_request(&pass, &pipeline), Some(&leases), 0, 0)
            .expect_err("a view off the 4-byte grid cannot be bound");
        eprintln!("off-grid window refused: {error:?}");
        assert_eq!(error.slug, "lease_offset_unsupported");
        assert_eq!(error.fields.get("offset"), Some(&FieldValue::Unsigned(2)));
    }

    /// The footprint proof `research/docs/23` §3.3 asks for: every vertex and
    /// every index the draw reads has to be inside the bytes the trace
    /// declared, and every index value has to select a vertex the streams
    /// cover. Metal reads past a short buffer without refusing, so the rail
    /// refuses first.
    ///
    /// For an indexed draw the stream-coverage rule and the index-value rule
    /// are the same condition (`stride * (highest + 1) <= bytes`), so the
    /// refusal names the index that reached past the stream; the stream's own
    /// footprint is the refusal a non-indexed draw gets, where the pass's count
    /// is the only bound.
    #[test]
    fn plan_refuses_a_stream_that_does_not_cover_the_draw() {
        // 24 bytes cover three of the four vertices the index values select.
        let mut short_stream = quad_pass();
        shorten(&mut short_stream.vertex_buffers[0], 24);
        let error = plan(&quad_request(&short_stream, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(3))
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(3))
        );
        assert_eq!(
            error.fields.get("buffer_index"),
            Some(&FieldValue::Unsigned(0))
        );

        // A non-indexed draw reads its vertex count in order, so the same
        // stream has to cover `vertices * stride` instead of the index span.
        let mut non_indexed = quad_pass();
        non_indexed.indices = None;
        non_indexed.vertices = 4;
        shorten(&mut non_indexed.vertex_buffers[0], 24);
        let error = plan(&quad_request(&non_indexed, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_vertex_footprint_unsupported");
        assert_eq!(
            error.fields.get("required_bytes"),
            Some(&FieldValue::Unsigned(32))
        );

        // 10 bytes cannot hold the six `uint16` indices.
        let mut short_indices = quad_pass();
        shorten(&mut short_indices.indices.as_mut().unwrap().view, 10);
        let error = plan(&quad_request(&short_indices, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_footprint_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(
            error.fields.get("required_bytes"),
            Some(&FieldValue::Unsigned(12))
        );

        // An index value at or above the four vertices the stream covers.
        let mut out_of_range = quad_pass();
        replace_bytes(
            &mut out_of_range.indices.as_mut().unwrap().view,
            [0_u16, 1, 2, 2, 1, 9]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        );
        let error = plan(&quad_request(&out_of_range, &quad_pipeline()), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(9)),
            "the refusal names the largest index value the draw reads"
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(4)),
            "the refusal names the vertices the stream covers"
        );
        assert_eq!(
            error.fields.get("view"),
            Some(&FieldValue::Unsigned(QUAD_INDEX_VIEW.get()))
        );
    }

    /// The `vertex_id` shape indexed through a buffer instead of drawn in
    /// order: the same reviewed module, three indices over three generated
    /// positions. Its index values are bounded by the module's own positions,
    /// because no stream carries a vertex count.
    #[test]
    fn an_indexed_vertex_id_draw_is_bounded_by_the_modules_positions() {
        let (trace, pool) = vertex_id_indexed_trace([0, 1, 2]);
        let contracts = milestone_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("three indices over three generated positions plan");
        let [planned] = planned.as_slice() else {
            panic!("the computed milestone trace carries one render pass");
        };
        assert!(planned.plan.vertex_streams.is_empty());
        let indices = planned.plan.indices.as_ref().expect("the pass is indexed");
        assert_eq!(indices.index_count, 3);
        assert_eq!(indices.vertex_span, 3);

        // The fourth position the module does not carry is refused by value,
        // before a driver would read it.
        let (trace, pool) = vertex_id_indexed_trace([0, 1, 9]);
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_index_value_out_of_range");
        assert_eq!(
            error.fields.get("highest_index"),
            Some(&FieldValue::Unsigned(9))
        );
        assert_eq!(
            error.fields.get("vertices_covered"),
            Some(&FieldValue::Unsigned(u64::from(
                FULL_SCREEN_TRIANGLE_VERTICES
            )))
        );
    }

    /// The trace path over the vertex-input fixture: the same plan the macOS
    /// encoder consumes. The streams carry their own bytes, so only the
    /// attachment's landing view comes from the serial pool, and the writeback
    /// lands where the trace declared it.
    #[test]
    fn plan_trace_plans_the_indexed_pass_and_its_landing_view() {
        let (trace, _) = quad_trace();
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = quad_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed indexed pass plans");
        assert_eq!(planned.len(), 1);
        let [planned] = planned.as_slice() else {
            panic!("the vertex-input trace carries one render pass");
        };
        assert_eq!(planned.contract, &quad_pipeline());
        assert_eq!(planned.plan.vertices, 6);
        assert_eq!(planned.plan.vertex_streams.len(), 1);
        assert_eq!(
            planned.plan.vertex_streams[0].source.proof_bytes(),
            quad_vertex_bytes()
        );
        assert_eq!(
            planned
                .plan
                .indices
                .as_ref()
                .map(|indices| indices.index_count),
            Some(6)
        );
        assert_eq!(
            planned.landings[0].expect("the attachment lands").view_id,
            ViewId::new(7)
        );
        assert_eq!(
            planned.landings[0]
                .expect("the attachment lands")
                .allocation_id,
            AllocationId::new(9)
        );
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
            stage_buffers: Vec::new(),
        });
        let [writeback] = writebacks.as_slice() else {
            panic!("the indexed attachment becomes one writeback");
        };
        assert_eq!(writeback.view_id, ViewId::new(7));
        assert_eq!(writeback.bytes, EXPECTED_TEXEL_BYTES.repeat(4));
    }

    /// The trace path over the dual fixture: one landing view per colour
    /// location, resolved from the serial pool in location order, and one
    /// writeback per landing view when the encoder's per-attachment readbacks
    /// come back.
    #[test]
    fn plan_trace_plans_the_dual_pass_with_a_landing_per_location() {
        let (trace, _) = dual_trace(LoadOp::Clear(sentinel()));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed dual pass plans");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.contract, &dual_pipeline());
        assert_eq!(planned.plan.attachments.len(), 2);
        assert_eq!(
            planned.landings[0].expect("the attachment lands").view_id,
            ViewId::new(7)
        );
        assert_eq!(
            planned.landings[0]
                .expect("the attachment lands")
                .allocation_id,
            AllocationId::new(9)
        );
        assert_eq!(
            planned.landings[1].expect("the attachment lands").view_id,
            ViewId::new(8)
        );
        assert_eq!(
            planned.landings[1]
                .expect("the attachment lands")
                .allocation_id,
            AllocationId::new(10)
        );
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![
                EXPECTED_TEXEL_BYTES.repeat(4),
                [0xff, 0x80, 0x40, 0xc0].repeat(4),
            ],
            depth: None,
            stencil: None,
            stage_buffers: Vec::new(),
        });
        assert_eq!(writebacks.len(), 2);
        assert_eq!(writebacks[0].view_id, ViewId::new(7));
        assert_eq!(writebacks[0].bytes, EXPECTED_TEXEL_BYTES.repeat(4));
        assert_eq!(writebacks[1].view_id, ViewId::new(8));
        assert_eq!(writebacks[1].bytes, [0xff, 0x80, 0x40, 0xc0].repeat(4));
    }

    /// The v19 store increment over the trace path: location 1 discards, so
    /// its landing view stays resolved for the load side but produces no
    /// writeback — only the stored attachment's texels leave the pass
    /// (`research/docs/23` §3.6).
    #[test]
    fn plan_trace_drops_a_discarded_attachment_from_the_writebacks() {
        let (mut trace, _) = dual_trace(LoadOp::Clear(sentinel()));
        let Some(TracePass::Render(pass)) = trace.passes.last_mut() else {
            panic!("the dual fixture ends with its render pass");
        };
        pass.color_attachments[1].store = StoreOp::DontCare;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned =
            plan_trace(&trace, &pool, &contracts, 0, 0).expect("the reviewed dual pass plans");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].store, RenderStoreAction::Store);
        assert_eq!(
            planned.plan.attachments[1].store,
            RenderStoreAction::DontCare
        );
        // Both landings are still resolved — the discarded attachment keeps its
        // declaring view for the load-side resolution — but only the stored
        // attachment's readback becomes a writeback.
        assert_eq!(planned.landings.len(), 2);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![EXPECTED_TEXEL_BYTES.repeat(4)],
            depth: None,
            stencil: None,
            stage_buffers: Vec::new(),
        });
        assert_eq!(writebacks.len(), 1);
        assert_eq!(writebacks[0].view_id, ViewId::new(7));
        assert_eq!(writebacks[0].allocation_id, AllocationId::new(9));
        assert_eq!(writebacks[0].offset, 0);
        assert_eq!(writebacks[0].bytes, EXPECTED_TEXEL_BYTES.repeat(4));
    }

    /// A loading dual pass uploads each attachment's own declaring bytes: the
    /// plan resolves the two previous-bytes entries independently, location 0
    /// from view 7 and location 1 from view 8.
    #[test]
    fn plan_trace_resolves_the_previous_bytes_per_location() {
        let (trace, _) = dual_trace(LoadOp::Load);
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0)
            .expect("each declaring view's own bytes are what a load uploads");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert_eq!(planned.plan.attachments[0].load, RenderLoadAction::Load);
        assert_eq!(
            planned.plan.attachments[0].initial_bytes(),
            Some([0xfe; 16].as_slice())
        );
        assert_eq!(planned.plan.attachments[1].load, RenderLoadAction::Load);
        assert_eq!(
            planned.plan.attachments[1].initial_bytes(),
            Some([0xfd; 16].as_slice())
        );
    }

    /// The source refusal is per location: a lease-backed second declaration
    /// with no channel to resolve it is refused with the same slug, class and
    /// phase a single attachment gets, naming storage mode and location rather
    /// than executing location 1 as a clear.
    #[test]
    fn plan_trace_refuses_a_leased_second_attachment_load_without_a_channel() {
        let (trace, _) = dual_trace(LoadOp::Load);
        let mut pool = trace.serial_resources().expect("admitted serial pool");
        let view = pool
            .iter_mut()
            .find(|view| {
                view.view_id == ViewId::new(8) && view.allocation_id == AllocationId::new(10)
            })
            .expect("the dual trace declares the second attachment view");
        view.source = BufferSource::StagedLease(LeaseId::new(5));
        let contracts = dual_contracts();
        let error = plan_trace(&trace, &pool, &contracts, 0, 0).unwrap_err();
        assert_eq!(error.slug, "render_attachment_load_source_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
        assert_eq!(
            error.fields.get("storage_mode"),
            Some(&FieldValue::Text("staged_lease".to_owned()))
        );
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(1))
        );
    }

    /// A render input carries its own bytes, so the trace does not have to
    /// declare the streams through a compute binding (`research/docs/23` §3.6).
    /// What the pool — the compute rail's binding set — still carries is the
    /// attachment's landing view, which is why this trace keeps exactly one
    /// declaration and still plans the draw.
    #[test]
    fn a_trace_plans_streams_no_compute_binding_declares() {
        let trace = ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: DeviceEpoch::new(3),
            operation_id: OperationId::new(2),
            pipelines: vec![declaration_pipeline(), quad_table_entry()],
            encoder_dispatch_type: DispatchType::Serial,
            passes: vec![
                TracePass::Compute(declaration_pass()),
                TracePass::Render(quad_pass()),
            ],
            completion_policy: CompletionPolicy::HostReadback,
            heap: None,
            indirect: None,
        };
        trace
            .validate()
            .expect("the declaration pass declares the attachment only");
        let pool = trace.serial_resources().expect("admitted serial pool");
        assert_eq!(
            pool.iter().map(|view| view.view_id).collect::<Vec<_>>(),
            vec![ViewId::new(7)],
            "neither stream is part of the compute rail's binding set"
        );

        let contracts = quad_contracts();
        let planned = plan_trace(&trace, &pool, &contracts, 0, 0).expect("the draw plans");
        let [planned] = planned.as_slice() else {
            panic!("the vertex-input trace carries one render pass");
        };
        assert_eq!(
            planned.plan.vertex_streams[0].source.proof_bytes(),
            quad_vertex_bytes()
        );
        assert_eq!(
            planned
                .plan
                .indices
                .as_ref()
                .map(|indices| indices.source.proof_bytes()),
            Some(quad_index_bytes().as_slice())
        );
    }

    /// An indirect draw replays its pass through `MTLIndirectRenderCommand`
    /// state, which carries the pipeline state and the draw counts rather than
    /// the streams a caller-held layout reads: the shape is refused before any
    /// Metal object exists, with the slug the ICB rail uses for a replay it
    /// cannot build.
    #[test]
    fn plan_trace_refuses_an_indirect_draw_of_a_stream_pass() {
        let (mut trace, _) = quad_trace();
        trace.indirect = Some(Box::new(IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands: 1,
                kinds: vec![IndirectCommandKind::Draw],
            },
            command: IndirectCommandDescriptor::Draw {
                vertex_count: 6,
                instance_count: 1,
            },
            range: IndirectCommandRange { start: 0, count: 1 },
        }));
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &quad_contracts(), 0, 0).unwrap_err();
        assert_eq!(error.slug, "icb_command_unsupported");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);
    }

    /// The reviewed allowlist is a (module, layout, colour-format list) triple:
    /// a layout that binds streams may only compile the `[[stage_in]]` module
    /// for one location, a `vertex_id` pipeline only the module whose vertex
    /// stage reads no stream, and an indexed pipeline with two `rgba8_unorm`
    /// locations only the dual module. A caller cannot pair one shape's
    /// descriptor with another shape's source.
    #[test]
    fn a_vertex_layout_selects_the_reviewed_module() {
        let pass = quad_pass();
        let pipeline = quad_pipeline();
        let single = [AttachmentFormat::Rgba8Unorm];
        let dual = [AttachmentFormat::Rgba8Unorm, AttachmentFormat::Rgba8Unorm];

        assert_eq!(
            reviewed_module(&VertexLayout::None, &single).map(|module| module.source),
            Some(REVIEWED_SOURCE)
        );
        assert_eq!(
            reviewed_module(&pipeline.vertex_layout, &single).map(|module| module.source),
            Some(REVIEWED_VERTEX_SOURCE)
        );
        assert_eq!(
            reviewed_module(&pipeline.vertex_layout, &dual).map(|module| module.source),
            Some(REVIEWED_DUAL_SOURCE)
        );
        assert!(
            !reviewed_module(&VertexLayout::None, &single)
                .expect("the vertex_id shape is reviewed")
                .binds_buffers
        );
        assert!(
            reviewed_module(&pipeline.vertex_layout, &single)
                .expect("the single-output stream shape is reviewed")
                .binds_buffers
        );
        // A dual-format `vertex_id` contract has no reviewed module, and the
        // two-attachment stream shape now accepts any mix of the two 8-bit
        // layouts (v26) while refusing a list the modules cannot serve.
        assert!(reviewed_module(&VertexLayout::None, &dual).is_none());
        assert_eq!(
            reviewed_module(
                &pipeline.vertex_layout,
                &[AttachmentFormat::Rgba8Unorm, AttachmentFormat::Bgra8Unorm]
            )
            .map(|module| module.fragment_entry),
            Some(DUAL_FRAGMENT_ENTRY)
        );
        assert!(reviewed_module(
            &pipeline.vertex_layout,
            &[AttachmentFormat::Rgba8Unorm, AttachmentFormat::R32Float]
        )
        .is_none());
        assert_eq!(layout_name(&VertexLayout::None), "vertex_id");
        assert_eq!(layout_name(&pipeline.vertex_layout), "vertex-buffer");

        // The reviewed contracts are accepted, and so is the milestone's.
        assert_eq!(review_contract(&pipeline), Ok(()));
        assert_eq!(review_contract(&quad_pipeline()), Ok(()));
        assert_eq!(review_contract(&dual_pipeline()), Ok(()));

        // A dual-format `vertex_id` contract is refused by name: no reviewed
        // module derives positions from `vertex_id` and writes two locations.
        let mut wider = milestone_pipeline();
        wider.color_formats = dual.to_vec();
        let error = review_contract(&wider).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");

        // A stream layout with the `vertex_id` module's bytes, and the reverse.
        let crossed = OffscreenRenderRequest {
            source: REVIEWED_SOURCE,
            ..quad_request(&pass, &pipeline)
        };
        let error = plan(&crossed, 0, 0).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Compile);
        assert!(
            error
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("vertex-buffer layout")),
            "the refusal names the layout the source was paired with: {:?}",
            error.detail
        );

        let milestone = milestone_pipeline();
        let milestone_pass = milestone_pass(LoadOp::Clear(sentinel()));
        let crossed = OffscreenRenderRequest {
            source: REVIEWED_VERTEX_SOURCE,
            ..milestone_request(&milestone_pass, &milestone, None)
        };
        let error = plan_pass(&crossed).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");

        // The entry pair is checked against the selected module too: the
        // indexed entries with a `vertex_id` layout are not a reviewed pair.
        let mut wrong_entry = milestone_pipeline();
        wrong_entry.vertex_entry = QUAD_VERTEX_ENTRY.to_owned();
        let error = review_contract(&wrong_entry).unwrap_err();
        assert_eq!(error.slug, "native_render_source_not_reviewed");
        assert_eq!(
            error
                .detail
                .as_deref()
                .map(|detail| detail.contains(VERTEX_ENTRY)),
            Some(true)
        );
    }

    /// The indexed fixture is held to the same two falsifiability rules the
    /// `vertex_id` fixture is: the fragment writes a byte/255 texel with no
    /// half-integer tie, and the module carries exactly the reviewed entries.
    #[test]
    fn reviewed_indexed_fixture_matches_the_expected_texel_bytes() {
        for literal in ["64.0 / 255.0", "128.0 / 255.0", "192.0 / 255.0"] {
            assert!(
                REVIEWED_VERTEX_SOURCE.contains(literal),
                "the indexed fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_VERTEX_SOURCE.contains("0.5"),
            "the indexed fixture must not carry a half-integer tie constant"
        );
        for entry in [QUAD_VERTEX_ENTRY, FRAGMENT_ENTRY] {
            assert!(
                REVIEWED_VERTEX_SOURCE.contains(entry),
                "the indexed fixture no longer carries {entry}"
            );
        }
        // The vertex stage reads a stage-in attribute: a module without
        // `[[stage_in]]` would make the pass's descriptor meaningless.
        assert!(REVIEWED_VERTEX_SOURCE.contains("[[stage_in]]"));
        assert!(REVIEWED_VERTEX_SOURCE.contains("[[attribute(0)]]"));
        // The fixture's own bytes, spelled the way a suite's `initial_hex`
        // spells them, so a drift between this rail's fixture, the Swift
        // self-test and the suite is visible on a host without a GPU.
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hex(&quad_vertex_bytes()),
            "000080bf000080bf0000803f000080bf000080bf0000803f0000803f0000803f"
        );
        assert_eq!(hex(&quad_index_bytes()), "000001000200020001000300");
        // The two single-output modules share one fragment entry, which is what
        // keeps a capture from telling the draws apart by their colour.
        assert_eq!(
            REVIEWED_MODULES[0].fragment_entry,
            REVIEWED_MODULES[1].fragment_entry
        );
        assert_eq!(REVIEWED_MODULES[0].vertex_entry, VERTEX_ENTRY);
        assert_eq!(REVIEWED_MODULES[1].vertex_entry, QUAD_VERTEX_ENTRY);
    }

    /// The dual fixture is held to the same falsifiability rules: its two
    /// fragment outputs are byte/255 constants with no half-integer tie, and
    /// the module carries exactly the reviewed entry pair.
    #[test]
    fn reviewed_dual_fixture_matches_the_expected_texel_bytes() {
        for literal in [
            "64.0 / 255.0",
            "128.0 / 255.0",
            "192.0 / 255.0",
            "255.0 / 255.0",
        ] {
            assert!(
                REVIEWED_DUAL_SOURCE.contains(literal),
                "the dual fixture no longer writes {literal}"
            );
        }
        assert!(
            !REVIEWED_DUAL_SOURCE.contains("0.5"),
            "the dual fixture must not carry a half-integer tie constant"
        );
        for entry in [QUAD_VERTEX_ENTRY, DUAL_FRAGMENT_ENTRY] {
            assert!(
                REVIEWED_DUAL_SOURCE.contains(entry),
                "the dual fixture no longer carries {entry}"
            );
        }
        assert!(REVIEWED_DUAL_SOURCE.contains("[[color(0)]]"));
        assert!(REVIEWED_DUAL_SOURCE.contains("[[color(1)]]"));
        assert_eq!(REVIEWED_MODULES[2].vertex_entry, QUAD_VERTEX_ENTRY);
        assert_eq!(REVIEWED_MODULES[2].fragment_entry, DUAL_FRAGMENT_ENTRY);
        assert_eq!(
            REVIEWED_MODULES[2].path,
            "conformance/shaders/quad_indexed_2x2_dual.metal"
        );
    }

    // -----------------------------------------------------------------------
    // The provider-resident render target's value-level half
    // (`research/docs/23` §76, R7).
    //
    // The registry itself (`crate::resident`) owns the identities, the budget
    // and every named refusal; these cases pin what this rail decides around
    // it: how the two resident arms map onto Metal's actions, which arm
    // publishes a writeback, which attachment needs a landing view, and the
    // four ways a trace and its provider can disagree about the declaration.
    // -----------------------------------------------------------------------

    /// The bytes the resident chain's seed is measured against: four bytes no
    /// reviewed module writes and no clear sentinel spells, so "the chain read
    /// the provider's image" is falsifiable per texel (`research/docs/23` §76,
    /// R7). The device case in `native.rs` uses the same word.
    const RESIDENT_SEED_BYTES: [u8; 4] = [0x11, 0x22, 0x33, 0x44];

    /// One request for a resident-bearing pass: the milestone's shape with the
    /// provider's answer per attachment and, when the case hands one over, the
    /// previous bytes a trace-declared load would upload.
    fn resident_request<'a>(
        pass: &'a RenderPassDescriptor,
        pipeline: &'a RenderPipelineContract,
        resolved: &[bool],
        previous: Option<PlannedInputSource<'a>>,
    ) -> OffscreenRenderRequest<'a> {
        OffscreenRenderRequest {
            pass,
            pipeline,
            source: REVIEWED_SOURCE,
            initial: vec![previous],
            resident: resolved.to_vec(),
        }
    }

    /// The two resident arms on the value level: a resident store keeps the
    /// pass's bytes in the provider's image under Metal's `.store` while
    /// publishing nothing, and a resident load opens that image with `.load`
    /// and no upload of its own (`research/docs/23` §76, R7).
    #[test]
    fn the_resident_arms_map_onto_the_actions_they_need() {
        let pipeline = milestone_pipeline();
        let mut storing = milestone_pass(LoadOp::Clear(sentinel()));
        storing.color_attachments[0].store = StoreOp::Resident;
        let plan = plan_pass(&resident_request(&storing, &pipeline, &[true], None))
            .expect("a resident store defines the provider's image");
        let [attachment] = plan.attachments.as_slice() else {
            panic!("the milestone pass carries one attachment");
        };
        assert!(
            attachment.resident,
            "the attachment renders into the provider's image"
        );
        assert_eq!(
            attachment.store,
            RenderStoreAction::Store,
            "Metal has to keep the texels: the provider's image is what a later pass loads"
        );
        assert!(
            !attachment.publishes,
            "a resident store publishes no writeback: its bytes stay in the provider's image"
        );
        assert_eq!(attachment.initial_bytes(), None);

        let loading = milestone_pass(LoadOp::Resident);
        let plan = plan_pass(&resident_request(&loading, &pipeline, &[true], None))
            .expect("a resident load keeps the provider image's own bytes");
        let [attachment] = plan.attachments.as_slice() else {
            panic!("the milestone pass carries one attachment");
        };
        assert!(attachment.resident);
        assert_eq!(
            attachment.load,
            RenderLoadAction::Load,
            "a resident load opens the provider's image exactly as a trace-declared load does"
        );
        assert!(
            attachment.publishes,
            "a trace-declared store beside it still lands"
        );
        assert_eq!(
            attachment.initial_bytes(),
            None,
            "a resident load uploads nothing: the bytes are already in the provider's image"
        );

        // The third resident arm: a trace-declared `Load` beside a resident
        // store. Its previous bytes are the trace's own, uploaded into the
        // provider's image before the draw — the other way a resident image is
        // defined (`research/docs/23` §76, R7).
        let mut seeded = milestone_pass(LoadOp::Load);
        seeded.color_attachments[0].store = StoreOp::Resident;
        let previous = RESIDENT_SEED_BYTES.repeat(4);
        let plan = plan_pass(&resident_request(
            &seeded,
            &pipeline,
            &[true],
            Some(PlannedInputSource::Declared(&previous)),
        ))
        .expect("a load beside a resident store keeps its declared bytes");
        let [attachment] = plan.attachments.as_slice() else {
            panic!("the milestone pass carries one attachment");
        };
        assert!(attachment.resident);
        assert!(!attachment.publishes);
        assert_eq!(attachment.load, RenderLoadAction::Load);
        assert_eq!(
            attachment.initial_bytes(),
            Some(&previous[..]),
            "the declared bytes are what the encoder uploads into the provider's image"
        );
    }

    /// The two directions in which a trace and its provider can disagree about
    /// the resident declaration are refused by name, before any Metal object
    /// exists (`research/docs/23` §76, R7).
    #[test]
    fn resident_declarations_must_agree_with_the_providers_answers() {
        let pipeline = milestone_pipeline();
        // The trace declares the resident target and the caller resolved no
        // image for it: the pass would be executed as a fresh per-pass
        // attachment, which is the silent downgrade the arm exists to prevent.
        let loading = milestone_pass(LoadOp::Resident);
        let error = plan_pass(&resident_request(&loading, &pipeline, &[], None))
            .expect_err("a resident load without the provider's image is refused");
        assert_eq!(error.slug, "resident_target_undeclared");
        assert_eq!(
            error.fields.get("attachment"),
            Some(&FieldValue::Unsigned(0))
        );
        assert_eq!(error.class, ProviderErrorClass::Capability);
        assert_eq!(error.phase, ProviderPhase::Resolve);

        // The provider resolved an image for an attachment that declares no
        // residency: the pass would render the provider's bytes where the trace
        // declared its own.
        let clearing = milestone_pass(LoadOp::Clear(sentinel()));
        let error = plan_pass(&resident_request(&clearing, &pipeline, &[true], None))
            .expect_err("an image for an undeclared attachment is refused");
        assert_eq!(error.slug, "resident_target_undeclared");

        // A `LoadOp::Resident` attachment that also carries previous bytes named
        // two sources for one attachment, so neither may silently win.
        let previous = EXPECTED_TEXEL_BYTES.repeat(4);
        let error = plan_pass(&resident_request(
            &loading,
            &pipeline,
            &[true],
            Some(PlannedInputSource::Declared(&previous)),
        ))
        .expect_err("a resident load declares no previous bytes");
        assert_eq!(error.slug, "resident_target_undeclared");
        assert_eq!(
            error.fields.get("load_op"),
            Some(&FieldValue::Text("resident".to_owned()))
        );

        // The list is the provider's answer per attachment, so it has to carry
        // one entry per colour attachment.
        let error = plan_pass(&resident_request(&loading, &pipeline, &[true, false], None))
            .expect_err("a list wider than the attachment list is refused");
        assert_eq!(error.slug, "resident_target_undeclared");
    }

    /// A resident target beside a present action or a multisample raster is
    /// refused by name, with the Vulkan rail's own slugs
    /// (`research/docs/23` §76, R7).
    #[test]
    fn a_resident_target_beside_present_or_multisample_is_refused() {
        let pipeline = milestone_pipeline();
        let mut present = milestone_present_pass();
        present.color_attachments[0].load = LoadOp::Resident;
        let error = plan_pass(&resident_request(&present, &pipeline, &[true], None))
            .expect_err("a present action keeps its own target");
        assert_eq!(error.slug, "resident_target_present_unsupported");
        assert_eq!(
            error.fields.get("allocation"),
            Some(&FieldValue::Unsigned(9))
        );

        let mut multisampled = milestone_pass(LoadOp::Clear(sentinel()));
        multisampled.multisample = Some(MultisampleState {
            sample_count: SampleCount::Four,
        });
        multisampled.color_attachments[0].store = StoreOp::Resident;
        let error = plan_pass(&resident_request(&multisampled, &pipeline, &[true], None))
            .expect_err("a multisampled raster resolves into its own landing");
        assert_eq!(error.slug, "resident_target_multisample_unsupported");
        assert_eq!(error.fields.get("samples"), Some(&FieldValue::Unsigned(4)));
    }

    /// A resident store needs no landing view, and the writeback channel pairs
    /// by *publishing* attachment rather than by store action
    /// (`research/docs/23` §76, R7).
    ///
    /// This is the falsifiable half of the R7 copy-out pairing: both attachment
    /// arms state Metal's `Store`, so a rail that paired readbacks on the
    /// action would hand the stored sibling's bytes to the resident one's
    /// location — and one that dropped the publishing filter would pair them
    /// across a gap.
    #[test]
    fn a_resident_store_needs_no_landing_and_pairs_its_siblings_bytes() {
        let (mut trace, _resources) = dual_trace(LoadOp::Clear(sentinel()));
        let pass = render_pass_mut(&mut trace);
        // Location 0 keeps its frame in the provider's image, location 1 keeps
        // publishing through the writeback channel.
        pass.color_attachments[0].store = StoreOp::Resident;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let contracts = dual_contracts();
        let planned = plan_trace_with_leases(
            &trace,
            &pool,
            &contracts,
            None,
            Some(&[vec![true, false]]),
            0,
            0,
        )
        .expect("the resident store beside a stored sibling plans");
        let [planned] = planned.as_slice() else {
            panic!("the dual trace carries one render pass");
        };
        assert!(planned.plan.attachments[0].resident);
        assert!(!planned.plan.attachments[0].publishes);
        assert!(!planned.plan.attachments[1].resident);
        assert!(planned.plan.attachments[1].publishes);
        assert!(
            planned.landings[0].is_none(),
            "a resident store neither lands nor loads trace-declared bytes, so it needs no \
             declared view even though the trace has one"
        );
        let landing = planned.landings[1].expect("the stored sibling keeps its landing");
        assert_eq!(landing.view_id, ViewId::new(8));
        assert_eq!(landing.allocation_id, AllocationId::new(10));

        // The encoder reads back the publishing attachment alone, so its one
        // readback is the stored sibling's, and the writeback carries that
        // sibling's own identity.
        let stored_texels = EXPECTED_TEXEL_BYTES.repeat(4);
        let writebacks = planned.writebacks(RenderReadback {
            attachments: vec![stored_texels.clone()],
            depth: None,
            stencil: None,
            stage_buffers: Vec::new(),
        });
        let [only] = writebacks.as_slice() else {
            panic!("one publishing attachment becomes one writeback");
        };
        assert_eq!(only.view_id, ViewId::new(8));
        assert_eq!(only.allocation_id, AllocationId::new(10));
        assert_eq!(only.bytes, stored_texels);
    }

    /// The trace path without a provider registry refuses a resident
    /// declaration by name, and refuses a resident list that does not line up
    /// with the passes (`research/docs/23` §76, R7).
    #[test]
    fn the_trace_path_without_a_registry_refuses_resident_declarations() {
        let (mut trace, _resources) = milestone_trace(LoadOp::Clear(sentinel()));
        render_pass_mut(&mut trace).color_attachments[0].store = StoreOp::Resident;
        let pool = trace.serial_resources().expect("admitted serial pool");
        let error = plan_trace(&trace, &pool, &milestone_contracts(), 0, 0)
            .expect_err("no registry resolved this pass's resident declaration");
        assert_eq!(error.slug, "resident_target_undeclared");

        let error =
            plan_trace_with_leases(&trace, &pool, &milestone_contracts(), None, Some(&[]), 0, 0)
                .expect_err("the resident list has to carry one entry per render pass");
        assert_eq!(error.slug, "resident_target_undeclared");

        // The same trace plans once the provider resolved the identity, and the
        // pass publishes no writeback: every byte of it stays in the provider's
        // image.
        let contracts = milestone_contracts();
        let planned =
            plan_trace_with_leases(&trace, &pool, &contracts, None, Some(&[vec![true]]), 0, 0)
                .expect("the provider resolved the resident identity");
        let [planned] = planned.as_slice() else {
            panic!("the milestone trace carries one render pass");
        };
        assert!(planned.plan.attachments[0].resident);
        assert!(planned.landings[0].is_none());
        assert!(planned
            .writebacks(RenderReadback {
                attachments: Vec::new(),
                depth: None,
                stencil: None,
                stage_buffers: Vec::new(),
            })
            .is_empty());
    }
}
