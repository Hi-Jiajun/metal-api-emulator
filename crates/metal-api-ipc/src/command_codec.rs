//! Binary codec for the owner-to-provider command channel.
//!
//! One frame carries one [`CommandRequest`] or one [`CommandResponse`]. The
//! encoding is versioned by [`COMMAND_FRAME_MAGIC`] and every value is
//! length-delimited, so a decoder never trusts an untrusted length beyond the
//! frame itself. The codec is deliberately explicit: the core provider
//! contract stays free of serialization dependencies and the wire format
//! documents every field that crosses the process boundary.

use crate::codec::CodecError;
use crate::command::{CommandRequest, CommandResponse};
use metal_api_core::provider::{
    AcquirePolicy, AffineAccess, AffineTerm, AliasMode, AllocationId, AllocationRecord,
    AttachmentFormat, BlendAttachment, BlendFactor, BlendOperation, BufferAccess,
    BufferBindingContract, BufferLease, BufferSource, BufferView, BufferWriteback, ClearColor,
    CompareFunction, CompiledComputePipeline, CompletionDisposition, CompletionPolicy,
    CompletionReadback, CompletionToken, ComputePass, ComputeTrace, CullMode, DepthFormat,
    DepthLoadOp, DepthResolveFilter, DepthStoreOp, DepthTest, DeviceEpoch, Dispatch, DispatchKind,
    DispatchType, FieldValue, FootprintProof, FunctionIdentity, FunctionSource, HeapDescriptor,
    HeapId, HeapPayload, HeapPlacement, HeapResource, IndexBufferBinding, IndexFormat,
    IndirectCommandBufferDescriptor, IndirectCommandDescriptor, IndirectCommandKind,
    IndirectCommandPayload, IndirectCommandRange, InitialState, LeaseId, LeaseReservation, LoadOp,
    MultisampleDepthResolve, MultisampleState, MultisampleStencilResolve, OperationId,
    PipelineCompileRequest, PipelineContract, PipelineId, PresentDescriptor, PresentMode,
    PresentTarget, ProviderCapabilities, ProviderError, ProviderErrorClass, ProviderHealth,
    ProviderPhase, ProviderSubmission, QueuePriority, RenderAttachment, RenderDepthAttachment,
    RenderDepthIdentity, RenderPassBlend, RenderPassCull, RenderPassDescriptor,
    RenderPipelineContract, RenderStencilAttachment, RenderStencilIdentity, ResourceTableSnapshot,
    Retryability, SampleCount, SemanticDigest, ShaderSource, StagedLease, StencilCompare,
    StencilFormat, StencilLoadOp, StencilOp, StencilResolveFilter, StencilTest, StorageMode,
    StoreOp, SubmissionId, TextureAccess, TextureFormat, TextureSource, TextureType, TextureView,
    TracePass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexLayout, VertexStep, ViewId,
    Winding, MAX_COLOR_ATTACHMENTS, MAX_RENDER_TEXTURES, MAX_VERTEX_ATTRIBUTES, MAX_VERTEX_BUFFERS,
};
use std::io::{Read, Write};

/// Protocol magic and version. A different byte sequence is refused.
pub const COMMAND_FRAME_MAGIC: [u8; 4] = *b"MCC1";
/// Maximum encoded payload, excluding the nine-byte frame header.
pub const MAX_COMMAND_FRAME: usize = 64 * 1024 * 1024;
/// Maximum total length of a request or response reassembled from chunk
/// frames.
pub const MAX_CHUNKED_PAYLOAD: usize = 256 * 1024 * 1024;

const REQUEST_FRAME: u8 = 0x01;
const RESPONSE_FRAME: u8 = 0x02;
const CHUNK_FRAME: u8 = 0x03;

const CAPABILITIES_REQUEST: u8 = 0x01;
const COMPILE_REQUEST: u8 = 0x02;
const SUBMIT_REQUEST: u8 = 0x03;
const WAIT_REQUEST: u8 = 0x04;
const READBACK_REQUEST: u8 = 0x05;
const CANCEL_REQUEST: u8 = 0x06;
const RELEASE_PIPELINE_REQUEST: u8 = 0x07;
const RELEASE_COMPLETION_REQUEST: u8 = 0x08;
const HEALTH_REQUEST: u8 = 0x09;
const IMPORT_STAGED_LEASE_REQUEST: u8 = 0x0a;
const RELEASE_STAGED_LEASE_REQUEST: u8 = 0x0b;
const IMPORT_BORROWED_LEASE_REQUEST: u8 = 0x0c;
const RELEASE_BORROWED_LEASE_REQUEST: u8 = 0x0d;
/// Owner queue-priority marking.
///
/// This tag is the reason the channel needed no version bump: it is additive,
/// so a decoder that predates it answers
/// [`CodecError::UnknownCommandTag`] for a marked frame while every frame an
/// older owner sends still decodes here unchanged (and therefore keeps the
/// provider's queues at `QueuePriority::Default`).
const SET_QUEUE_PRIORITIES_REQUEST: u8 = 0x0e;
/// Submit a trace whose pass list uses the tagged compute/render layout.
///
/// A compute-only trace keeps travelling under [`SUBMIT_REQUEST`] with the
/// exact pre-render pass bytes, so this tag is only ever emitted when a render
/// pass is present. An older decoder answers `UnknownCommandTag` for it
/// instead of misreading the tagged pass list, which is the same additive
/// policy the queue-priority tag uses.
const SUBMIT_RENDER_REQUEST: u8 = 0x0f;
/// Submit a trace that carries a heap or an indirect-command payload.
///
/// A compute-only or render-only trace keeps travelling under
/// [`SUBMIT_REQUEST`] / [`SUBMIT_RENDER_REQUEST`] with the exact pre-heap/ICB
/// bytes, so this tag is only ever emitted when a heap or ICB payload exists.
/// An older decoder answers `UnknownCommandTag` for it instead of misreading
/// the tagged heap/ICB tail, the same additive policy the render and present
/// tags used (`research/docs/25-heaps与ICB设计.md` §4.5).
const SUBMIT_HEAP_ICB_REQUEST: u8 = 0x10;

const CAPABILITIES_RESPONSE: u8 = 0x01;
const COMPILED_RESPONSE: u8 = 0x02;
const SUBMITTED_RESPONSE: u8 = 0x03;
const OBSERVED_RESPONSE: u8 = 0x04;
const READBACK_RESPONSE: u8 = 0x05;
const RELEASED_RESPONSE: u8 = 0x06;
const HEALTH_RESPONSE: u8 = 0x07;
const IMPORTED_RESPONSE: u8 = 0x08;
/// The device queue table an owner marking installed.
const QUEUE_PRIORITIES_RESPONSE: u8 = 0x09;
/// Capabilities including the render bits the pre-render payload cannot carry.
///
/// Only a snapshot that declares at least one non-default render bit uses this
/// tag; every compute-only provider keeps sending the legacy
/// [`CAPABILITIES_RESPONSE`] bytes.
const RENDER_CAPABILITIES_RESPONSE: u8 = 0x0a;
const ERROR_RESPONSE: u8 = 0x7f;

/// Pass discriminator inside the tagged trace layout.
const PASS_KIND_COMPUTE: u8 = 0x00;
const PASS_KIND_RENDER: u8 = 0x01;
/// A render pass that also carries a present action
/// (`research/docs/24` §4.1 and §3.5, shape one: present as the render pass's
/// tail action).
///
/// The discriminator is a pass kind rather than a presence byte inside the
/// render payload so an offscreen-only render pass keeps the exact bytes the
/// render track published: a decoder that predates this tag answers
/// [`CodecError::UnknownPassTag`] for a presenting pass instead of reading the
/// present tail as the trace's completion policy. That is the same additive
/// policy [`PASS_KIND_RENDER`] itself used, and it is the shape `docs/24` §4.1
/// leaves open for the tagged payload ("extend the `SUBMIT_RENDER_REQUEST`
/// payload or add a tag").
const PASS_KIND_RENDER_PRESENT: u8 = 0x02;

/// A render pass that carries a caller-held vertex stream, a present action, or
/// both (`research/docs/23` §3.3).
///
/// The tag is followed by one feature byte, so the two optional sections that
/// may follow the base render payload stay orthogonal instead of multiplying
/// the tag space: bit 0 is the vertex/index input block and bit 1 is the
/// present tail. A decoder that predates this tag answers
/// [`CodecError::UnknownPassTag`], and an older encoder keeps writing
/// [`PASS_KIND_RENDER`] / [`PASS_KIND_RENDER_PRESENT`] for the shapes it
/// already published, so every pre-vertex frame keeps its exact bytes.
const PASS_KIND_RENDER_EXT: u8 = 0x10;

/// Feature bits carried by [`PASS_KIND_RENDER_EXT`].
const RENDER_FEATURE_VERTEX_INPUT: u8 = 0x01;
const RENDER_FEATURE_PRESENT: u8 = 0x02;
/// The scissor block (`research/docs/23` §3.3, v29): four `u32`s
/// `[x, y, width, height]` after the base payload.
const RENDER_FEATURE_SCISSOR: u8 = 0x04;
/// The instancing tail (`research/docs/23` §3.3, v31): one `u32` instance count
/// after every earlier optional section. A pass that draws one instance — the
/// only shape published before v31 — never sets the bit, so its bytes stay
/// exactly what they were, and the decoder reads the missing section as the
/// single instance the older frames meant.
const RENDER_FEATURE_INSTANCING: u8 = 0x08;
/// The base-vertex tail (`research/docs/23` §3.3, v34): one `u32` vertex offset
/// after every earlier optional section. A pass whose indices start at zero —
/// the only shape published before v34 — never sets the bit, so its bytes stay
/// exactly what they were, and the decoder reads the missing section as the
/// zero offset the older frames meant.
const RENDER_FEATURE_BASE_VERTEX: u8 = 0x10;
/// The blend state (`research/docs/23` §3.3, v40): one `(source rgb, destination
/// rgb, source alpha, destination alpha, operation)` block per colour
/// attachment, appended after every earlier optional section. A pass that
/// blends nothing — every shape published before v40 — never sets the bit, so
/// its bytes stay exactly what they were.
///
/// This is the feature byte's last free bit: a later optional section needs
/// either a second feature byte or a tag of its own, and this comment is where
/// that decision has to be made.
const RENDER_FEATURE_BLEND: u8 = 0x80;
/// The culling state (`research/docs/23` §3.3, v39): one mode byte and one
/// winding byte after every earlier optional section. A pass that culls
/// nothing — every shape published before v39 — never sets the bit, so its
/// bytes stay exactly what they were.
const RENDER_FEATURE_CULL: u8 = 0x40;
/// The depth block (`research/docs/23` §3.3, v36): the depth attachment's
/// format, extent and load operation, followed by the pass's depth state when
/// it declares one. A pass with no depth attachment — every shape published
/// before v36 — never sets the bit, so its bytes stay exactly what they were.
const RENDER_FEATURE_DEPTH: u8 = 0x20;
/// Every bit this version knows. An unknown bit is a decoder refusal rather
/// than a silently skipped section.
const RENDER_FEATURE_KNOWN: u8 = RENDER_FEATURE_VERTEX_INPUT
    | RENDER_FEATURE_PRESENT
    | RENDER_FEATURE_SCISSOR
    | RENDER_FEATURE_INSTANCING
    | RENDER_FEATURE_BASE_VERTEX
    | RENDER_FEATURE_DEPTH
    | RENDER_FEATURE_CULL
    | RENDER_FEATURE_BLEND;

/// A render pass that carries a section invented after the narrow feature byte
/// was full (`research/docs/23` §3.3, v43).
///
/// [`PASS_KIND_RENDER_EXT`]'s byte had every bit assigned by v40, when blending
/// took the last free one, so the next optional section needed either a second
/// byte or a tag of its own. This tag is that second byte: it is followed by a
/// big-endian `u16` feature word whose **low byte repeats every meaning** the
/// narrow tag's byte has and whose high byte carries the sections invented
/// after it. A decoder that predates this tag answers
/// [`CodecError::UnknownPassTag`] for the tag itself, so a wide frame cannot be
/// read as a narrow one, and an encoder that needs no wide section keeps
/// writing the narrow tag — every pre-v43 frame keeps its exact bytes.
const PASS_KIND_RENDER_EXT_WIDE: u8 = 0x11;

/// The wide feature word's first bit (`research/docs/23` §3.3, v43): the pass
/// states its depth attachment's store action, one byte after the depth block
/// and before every later section. A pass that keeps nothing never sets it, so
/// the narrow frames keep their bytes exactly.
const RENDER_WIDE_FEATURE_DEPTH_STORE: u16 = 0x0100;

/// The wide feature word's second bit (`research/docs/23` §3.3, v43): the
/// pass's depth attachment is a caller-held resource, and its `allocation` and
/// `view` follow — after the depth block and after the store action when that
/// one is present. The identity is what the stored texels land on, so it is
/// only written for the shape that keeps them.
const RENDER_WIDE_FEATURE_DEPTH_RESOURCE: u16 = 0x0200;

/// The wide feature word's third bit (`research/docs/23` §3.3, v47): the pass
/// opens a stencil attachment and, when it declares one, carries its stencil
/// state. The block follows every depth section, so a pass that states no
/// stencil keeps its bytes exactly — and a decoder that predates this bit
/// refuses the wide word instead of skipping the block.
const RENDER_WIDE_FEATURE_STENCIL: u16 = 0x0400;

/// The wide feature word's fourth bit (`research/docs/23` §3.3, v49): the pass
/// states its stencil attachment's store action, one byte after the stencil
/// block. A pass that keeps nothing never sets it.
const RENDER_WIDE_FEATURE_STENCIL_STORE: u16 = 0x0800;

/// The wide feature word's fifth bit (`research/docs/23` §3.3, v49): the pass's
/// stencil attachment is a caller-held resource, and its `allocation` and
/// `view` follow — after the stencil block and after the store action when that
/// one is present. The identity is what the stored texels land on, so it is
/// only written for the shape that keeps them.
const RENDER_WIDE_FEATURE_STENCIL_RESOURCE: u16 = 0x1000;

/// The wide feature word's sixth bit (`research/docs/23` §3.3, v51): the pass
/// states a pass-wide multisample raster, one sample count code after the
/// stencil sections and before culling. A pass that never multisamples never
/// sets it, so every pre-v51 frame keeps its exact bytes.
const RENDER_WIDE_FEATURE_MULTISAMPLE: u16 = 0x2000;

/// The wide feature word's seventh bit (`research/docs/23` §3.3, v57): the
/// pass resolves its stored multisampled depth surface, one filter code after
/// the multisample section and before culling. A pass that never resolves
/// never sets it, so every pre-v57 frame keeps its exact bytes.
const RENDER_WIDE_FEATURE_DEPTH_RESOLVE: u16 = 0x4000;

/// The wide feature word's eighth and final bit (`research/docs/23` §3.3,
/// v60): the pass resolves its stored multisampled stencil surface, one
/// filter code after the depth-resolve section and before culling. A pass that
/// never resolves never sets it, so every pre-v60 frame keeps its exact
/// bytes. With this bit the high byte is full: the next optional section needs
/// a new mechanism — a second wide word behind a tag, or a v2 frame — rather
/// than a ninth bit.
const RENDER_WIDE_FEATURE_STENCIL_RESOLVE: u16 = 0x8000;

/// A render pass that carries the sampled textures its fragment stage reads
/// (`research/docs/23` §3.3, v70).
///
/// The wide feature word is full to its last bit (`0x8000`), so the next
/// optional section needed a mechanism of its own and this tag is it — the
/// choice [`RENDER_WIDE_FEATURE_STENCIL_RESOLVE`]'s comment already named. The
/// payload is the same `u16` wide feature word (same meanings, so the two
/// tags keep sharing the section walker) followed by the texture block — a
/// `u8` count and that many full [`TextureView`]s, source and all, because the
/// texel bytes a render pass samples have to travel with the trace that binds
/// them, exactly as a compute pass's texture bindings do. A pass that binds no
/// texture keeps writing [`PASS_KIND_RENDER_EXT`] / [`PASS_KIND_RENDER_EXT_WIDE`]
/// and every pre-v70 frame keeps its exact bytes; a decoder that predates this
/// tag answers [`CodecError::UnknownPassTag`] for it.
const PASS_KIND_RENDER_SAMPLED: u8 = 0x12;

/// Every bit of the wide feature word this version knows. The low byte is the
/// narrow byte verbatim; an unknown *high* bit is a decoder refusal, exactly as
/// an unknown pass tag is, so a future section cannot be skipped silently.
const RENDER_WIDE_FEATURE_KNOWN: u16 = RENDER_FEATURE_KNOWN as u16
    | RENDER_WIDE_FEATURE_DEPTH_STORE
    | RENDER_WIDE_FEATURE_DEPTH_RESOURCE
    | RENDER_WIDE_FEATURE_STENCIL
    | RENDER_WIDE_FEATURE_STENCIL_STORE
    | RENDER_WIDE_FEATURE_STENCIL_RESOURCE
    | RENDER_WIDE_FEATURE_MULTISAMPLE
    | RENDER_WIDE_FEATURE_DEPTH_RESOLVE
    | RENDER_WIDE_FEATURE_STENCIL_RESOLVE;

/// Pipeline vertex-layout discriminators. `None` keeps the single byte the
/// pre-vertex pipeline payload wrote; `Buffers` appends the layout block.
const VERTEX_LAYOUT_NONE: u8 = 0x00;
const VERTEX_LAYOUT_BUFFERS: u8 = 0x01;
/// A [`VertexLayout::Buffers`] list whose bindings carry their own step
/// function (`research/docs/23` §3.3, v31).
///
/// The discriminator is a *second* layout tag rather than an extra byte inside
/// the existing one: a layout whose bindings all step per vertex is what every
/// pre-v31 frame wrote, so it keeps `VERTEX_LAYOUT_BUFFERS` and its exact
/// bytes, and only a layout that declares a per-instance stream takes this tag.
/// A decoder that predates it answers [`CodecError::UnknownEnumValue`] instead
/// of reading the step byte as a stride.
const VERTEX_LAYOUT_BUFFERS_STEPPED: u8 = 0x02;

/// Pipeline-table discriminator inside the tagged trace layout.
///
/// The pipeline table carries one entry per registration, and the render half
/// of an entry is what lets core admission compare an attachment with the
/// pipeline the render pass names (review item I3, 2026-09-14). The
/// discriminator is a value rather than an optional length prefix so an
/// unknown tag is refused instead of being read as "no render half", exactly
/// like [`PASS_KIND_COMPUTE`]. Only the tagged layout writes it: a
/// compute-only trace keeps the pre-render pipeline bytes, so
/// `PIPELINE_KIND_COMPUTE` is what an entry in that layout decodes to.
const PIPELINE_KIND_COMPUTE: u8 = 0x00;
const PIPELINE_KIND_RENDER: u8 = 0x01;
/// Render entry whose contract carries one colour format **per attachment**.
///
/// `PIPELINE_KIND_RENDER` is followed by exactly one format byte, so a contract
/// with two or more formats has no single byte to write. This tag replaces
/// that byte with a length-prefixed list and keeps every other field of the
/// entry in place, which is what lets the single-format entry keep its exact
/// pre-MRT bytes: an owner that registers one attachment per pipeline never
/// writes this tag, and a decoder that predates it answers
/// [`CodecError::UnknownPipelineTag`] instead of misreading the list as a
/// vertex layout.
const PIPELINE_KIND_RENDER_MRT: u8 = 0x02;

/// Maximum number of passes one tagged trace frame may carry.
///
/// The legacy layout keeps its previous (payload-bounded) behaviour; this
/// bound exists so a corrupt tagged count cannot make the decoder allocate
/// before it has read a single pass.
pub const MAX_TAGGED_TRACE_PASSES: usize = 4096;

/// Maximum colour formats one capability snapshot may declare.
///
/// [`AttachmentFormat`] is a closed four-value family, so this bound can never
/// refuse a well-formed snapshot; it only stops a corrupt count from driving
/// the decoder.
pub const MAX_SUPPORTED_COLOR_FORMATS: usize = 16;

/// Maximum present modes one capability snapshot may declare.
///
/// [`PresentMode`] is a closed four-value family (the four
/// `VkPresentModeKHR` modes, `research/docs/24` §3.1), so this bound can never
/// refuse a well-formed snapshot; like [`MAX_SUPPORTED_COLOR_FORMATS`] it only
/// stops a corrupt count from driving the decoder.
pub const MAX_SUPPORTED_PRESENT_MODES: usize = 8;

/// Maximum vertex formats one capability snapshot may declare.
///
/// [`VertexFormat`] is a closed four-value family, so this bound can never
/// refuse a well-formed snapshot; it only stops a corrupt count from driving
/// the decoder.
pub const MAX_SUPPORTED_VERTEX_FORMATS: usize = 8;

/// Maximum index widths one capability snapshot may declare. Same rule as
/// [`MAX_SUPPORTED_VERTEX_FORMATS`] over the closed two-value family.
pub const MAX_SUPPORTED_INDEX_FORMATS: usize = 4;

/// Presence tag of the capability tail's vertex-input block.
///
/// The block travels inside the extended capability frame's optional tail,
/// after the heap/ICB half (`research/docs/23` §3.3). A tag rather than a bare
/// field keeps a future third section from being read as this one.
const CAPABILITY_VERTEX_INPUT_TAIL: u8 = 0x01;

/// Presence tag of the capability tail's instancing block
/// (`research/docs/23` §3.3, v31).
///
/// The block follows the vertex-input block when the snapshot declares either
/// instancing bit, and carries the bit plus the snapshot's instance ceiling. It
/// is a separate tagged section for the same reason the vertex-input block is:
/// a snapshot that declares instancing but no vertex input still keeps the
/// decoder's position rules unambiguous.
const CAPABILITY_INSTANCING_TAIL: u8 = 0x02;

/// Presence tag of the capability tail's multisample block
/// (`research/docs/23` §3.3, v51).
///
/// The block follows the instancing block when the snapshot declares either
/// multisample bit, and carries the bit plus the snapshot's sample ceiling. It
/// is a separate tagged section for the same reason the two blocks before it
/// are: a snapshot that declares multisampling but neither vertex input nor
/// instancing still keeps the decoder's position rules unambiguous.
const CAPABILITY_MULTISAMPLE_TAIL: u8 = 0x04;

/// Presence tag of the capability tail's depth-resolve block
/// (`research/docs/23` §3.3, v57).
///
/// The block follows the multisample block when the snapshot declares either
/// depth-resolve bit, and carries the bool plus the filter bitmask (bit `i` =
/// [`DepthResolveFilter`] code `i`). It is a separate tagged section for the
/// same reason the three blocks before it are: a snapshot that declares depth
/// resolve but none of the earlier blocks still keeps the decoder's position
/// rules unambiguous.
const CAPABILITY_DEPTH_RESOLVE_TAIL: u8 = 0x08;

/// Presence tag of the capability tail's stencil-resolve block
/// (`research/docs/23` §3.3, v60).
///
/// The block follows the depth-resolve block when the snapshot declares
/// either stencil-resolve bit, and carries the bool plus the filter bitmask
/// (bit `i` = [`StencilResolveFilter`] code `i`). It is a separate tagged
/// section for the same reason the four blocks before it are: a snapshot that
/// declares stencil resolve but none of the earlier blocks still keeps the
/// decoder's position rules unambiguous.
const CAPABILITY_STENCIL_RESOLVE_TAIL: u8 = 0x10;

/// Presence tag of the capability tail's render-sampler block
/// (`research/docs/23` §3.3, v70).
///
/// The block follows the stencil-resolve block when the snapshot declares any
/// of the three render-sampler bits, and carries the bool, the binding cap and
/// the admitted texture formats. It is a separate tagged section for the same
/// reason the five blocks before it are: a snapshot that declares render
/// texture sampling but none of the earlier blocks still keeps the decoder's
/// position rules unambiguous.
const CAPABILITY_RENDER_TEXTURE_TAIL: u8 = 0x20;

/// Maximum texture formats one capability snapshot may declare as render-pass
/// sampling sources.
///
/// The contract's own list is the closed four-value [`TextureFormat`] family,
/// so this bound can never refuse a well-formed snapshot; it only stops a
/// corrupt count from driving the decoder — the same rule
/// [`MAX_SUPPORTED_COLOR_FORMATS`] states for the attachment formats.
pub const MAX_SUPPORTED_RENDER_TEXTURE_FORMATS: usize = 8;

/// Maximum number of bytes one present target's sentinel may carry.
///
/// The contract's own rule is stronger — a sentinel is exactly one tightly
/// packed texel of its format (`PresentTarget::validate_shape`, four bytes for
/// every admitted format) — but that rule needs the format, which is decoded
/// next to the bytes. This bound is the protocol's guard instead: it stops a
/// corrupt length from being read as a payload, so a malformed present section
/// is refused as [`CodecError::PresentSentinelLength`] before it is mistaken
/// for the rest of the pass. It is deliberately looser than the contract rule
/// so the contract, not the codec, reports a sentinel that is merely the wrong
/// size for its format.
pub const MAX_PRESENT_SENTINEL_BYTES: usize = 64;

/// Maximum heap placements one trace payload may carry.
///
/// The contract's own rule is stronger — a first-increment heap refuses
/// aliasing, so placements are pairwise disjoint — but that rule needs the
/// whole list decoded. This bound is the protocol's guard instead: it stops a
/// corrupt count from making the decoder allocate before it has read a single
/// placement.
pub const MAX_HEAP_PLACEMENTS: usize = 4096;

/// Maximum heap storage modes one capability snapshot may declare.
///
/// [`StorageMode`] is a closed three-value family, so this bound can never
/// refuse a well-formed snapshot; it only stops a corrupt count from driving
/// the decoder (`docs/25-heaps与ICB设计.md` §4.1).
pub const MAX_SUPPORTED_HEAP_STORAGE_MODES: usize = 8;

/// Maximum indirect-command kinds one ICB or capability snapshot may declare.
///
/// [`IndirectCommandKind`] is a closed three-value family, so this bound can
/// never refuse a well-formed value; it only stops a corrupt count from
/// driving the decoder (`docs/25-heaps与ICB设计.md` §4.1, §4.3).
pub const MAX_SUPPORTED_INDIRECT_COMMANDS: usize = 8;

/// Maximum number of queue tiers one frame may carry.
///
/// The bound is a protocol limit, not a device limit: a marking describes the
/// owner's view of a device queue table, and the Vulkan provider caps a device
/// at eight queues. Refusing a longer frame keeps a corrupt length from making
/// the decoder allocate; a provider may still read a short marking, because the
/// expansion pads it (`metal_api_core::provider::queue_priorities_for_device`).
pub const MAX_QUEUE_PRIORITIES: usize = 64;

/// Stateless encoder/decoder for command frames.
pub struct CommandCodec;

/// One decoded chunk frame.
#[derive(Debug)]
pub(crate) struct ChunkPayload {
    pub(crate) transfer_id: u64,
    pub(crate) offset: u64,
    pub(crate) total: u64,
    pub(crate) bytes: Vec<u8>,
}

impl CommandCodec {
    /// Encode one complete request frame.
    pub fn encode_request(request: &CommandRequest) -> Result<Vec<u8>, CodecError> {
        frame(REQUEST_FRAME, Self::encode_request_payload(request)?)
    }

    /// Encode one request payload without the frame header.
    ///
    /// A transport that cannot fit the payload into [`MAX_COMMAND_FRAME`]
    /// splits it into chunk frames with `CommandCodec::write_chunk_frame`.
    pub fn encode_request_payload(request: &CommandRequest) -> Result<Vec<u8>, CodecError> {
        let mut encoder = Encoder::new();
        match request {
            CommandRequest::Capabilities => encoder.u8(CAPABILITIES_REQUEST),
            CommandRequest::Health => encoder.u8(HEALTH_REQUEST),
            CommandRequest::Compile { request } => {
                encoder.u8(COMPILE_REQUEST);
                put_compile_request(&mut encoder, request);
            }
            CommandRequest::ImportStagedLease { staged } => {
                encoder.u8(IMPORT_STAGED_LEASE_REQUEST);
                put_staged_lease(&mut encoder, staged);
            }
            CommandRequest::ReleaseStagedLease { lease_id } => {
                encoder.u8(RELEASE_STAGED_LEASE_REQUEST);
                encoder.u64(lease_id.get());
            }
            CommandRequest::ImportBorrowedLease { reservation } => {
                encoder.u8(IMPORT_BORROWED_LEASE_REQUEST);
                put_reservation(&mut encoder, reservation);
            }
            CommandRequest::ReleaseBorrowedLease { lease_id } => {
                encoder.u8(RELEASE_BORROWED_LEASE_REQUEST);
                encoder.u64(lease_id.get());
            }
            CommandRequest::SetQueuePriorities { tiers } => {
                encoder.u8(SET_QUEUE_PRIORITIES_REQUEST);
                put_queue_priorities(&mut encoder, tiers)?;
            }
            CommandRequest::Submit { trace, resources } => {
                encoder.u8(if trace.has_heap_or_icb() {
                    SUBMIT_HEAP_ICB_REQUEST
                } else if trace.has_render_passes() {
                    SUBMIT_RENDER_REQUEST
                } else {
                    SUBMIT_REQUEST
                });
                put_trace(&mut encoder, trace)?;
                put_resources(&mut encoder, resources);
            }
            CommandRequest::Wait { token, timeout } => {
                encoder.u8(WAIT_REQUEST);
                put_token(&mut encoder, token);
                encoder.u64(u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX));
            }
            CommandRequest::Readback { token } => {
                encoder.u8(READBACK_REQUEST);
                put_token(&mut encoder, token);
            }
            CommandRequest::Cancel { token } => {
                encoder.u8(CANCEL_REQUEST);
                put_token(&mut encoder, token);
            }
            CommandRequest::ReleasePipeline { pipeline } => {
                encoder.u8(RELEASE_PIPELINE_REQUEST);
                put_pipeline(&mut encoder, pipeline);
            }
            CommandRequest::ReleaseCompletion { token } => {
                encoder.u8(RELEASE_COMPLETION_REQUEST);
                put_token(&mut encoder, token);
            }
        }
        Ok(encoder.bytes)
    }

    /// Decode one complete request frame.
    pub fn decode_request(frame: &[u8]) -> Result<CommandRequest, CodecError> {
        Self::decode_request_payload(unframe(frame, REQUEST_FRAME)?)
    }

    /// Decode one request payload without the frame header.
    pub fn decode_request_payload(payload: &[u8]) -> Result<CommandRequest, CodecError> {
        decode_request_payload(payload)
    }

    /// Encode one complete response frame.
    pub fn encode_response(response: &CommandResponse) -> Result<Vec<u8>, CodecError> {
        frame(RESPONSE_FRAME, Self::encode_response_payload(response)?)
    }

    /// Encode one response payload without the frame header.
    pub fn encode_response_payload(response: &CommandResponse) -> Result<Vec<u8>, CodecError> {
        let mut encoder = Encoder::new();
        match response {
            CommandResponse::Capabilities {
                epoch,
                capabilities,
            } => {
                if capabilities.declares_render_support() {
                    encoder.u8(RENDER_CAPABILITIES_RESPONSE);
                    put_epoch(&mut encoder, *epoch);
                    put_capabilities(&mut encoder, capabilities)?;
                } else {
                    encoder.u8(CAPABILITIES_RESPONSE);
                    put_epoch(&mut encoder, *epoch);
                    put_capabilities_legacy(&mut encoder, capabilities);
                }
            }
            CommandResponse::Health { health } => {
                encoder.u8(HEALTH_RESPONSE);
                put_health(&mut encoder, *health);
            }
            CommandResponse::Compiled { pipeline } => {
                encoder.u8(COMPILED_RESPONSE);
                put_pipeline(&mut encoder, pipeline);
            }
            CommandResponse::Imported => encoder.u8(IMPORTED_RESPONSE),
            CommandResponse::Submitted { submission } => {
                encoder.u8(SUBMITTED_RESPONSE);
                put_submission(&mut encoder, submission);
            }
            CommandResponse::Observed { disposition } => {
                encoder.u8(OBSERVED_RESPONSE);
                put_disposition(&mut encoder, *disposition);
            }
            CommandResponse::Readback { readback } => {
                encoder.u8(READBACK_RESPONSE);
                put_readback(&mut encoder, readback);
            }
            CommandResponse::QueuePriorities { installed } => {
                encoder.u8(QUEUE_PRIORITIES_RESPONSE);
                put_queue_priorities(&mut encoder, installed)?;
            }
            CommandResponse::Released => encoder.u8(RELEASED_RESPONSE),
            CommandResponse::Error { error } => {
                encoder.u8(ERROR_RESPONSE);
                put_error(&mut encoder, error);
            }
        }
        Ok(encoder.bytes)
    }

    /// Decode one complete response frame.
    pub fn decode_response(frame: &[u8]) -> Result<CommandResponse, CodecError> {
        Self::decode_response_payload(unframe(frame, RESPONSE_FRAME)?)
    }

    /// Decode one response payload without the frame header.
    pub fn decode_response_payload(payload: &[u8]) -> Result<CommandResponse, CodecError> {
        decode_response_payload(payload)
    }

    /// Write one request frame.
    pub fn write_request<W: Write>(
        writer: &mut W,
        request: &CommandRequest,
    ) -> Result<(), CodecError> {
        writer.write_all(&Self::encode_request(request)?)?;
        Ok(())
    }

    /// Read one request frame.
    pub fn read_request<R: Read>(reader: &mut R) -> Result<CommandRequest, CodecError> {
        let (kind, payload) = read_frame(reader)?;
        if kind != REQUEST_FRAME {
            return Err(CodecError::UnknownCommandTag(kind));
        }
        decode_request_payload(&payload)
    }

    /// Write one request payload as a single frame.
    pub(crate) fn write_request_payload<W: Write>(
        writer: &mut W,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        write_framed(writer, REQUEST_FRAME, payload)
    }

    /// Write one response payload as a single frame.
    pub(crate) fn write_response_payload<W: Write>(
        writer: &mut W,
        payload: &[u8],
    ) -> Result<(), CodecError> {
        write_framed(writer, RESPONSE_FRAME, payload)
    }

    /// Write one chunk of a request payload.
    pub(crate) fn write_chunk_frame<W: Write>(
        writer: &mut W,
        transfer_id: u64,
        offset: u64,
        total: u64,
        bytes: &[u8],
    ) -> Result<(), CodecError> {
        let mut encoder = Encoder::new();
        encoder.u64(transfer_id);
        encoder.u64(offset);
        encoder.u64(total);
        encoder.blob(bytes);
        write_framed(writer, CHUNK_FRAME, &encoder.bytes)
    }

    /// Read one frame without interpreting its kind.
    pub(crate) fn read_raw_frame<R: Read>(reader: &mut R) -> Result<(u8, Vec<u8>), CodecError> {
        read_frame(reader)
    }

    /// Decode one chunk frame payload.
    pub(crate) fn decode_chunk_payload(payload: &[u8]) -> Result<ChunkPayload, CodecError> {
        let mut decoder = Decoder::new(payload);
        let transfer_id = decoder.u64()?;
        let offset = decoder.u64()?;
        let total = decoder.u64()?;
        let bytes = decoder.blob()?;
        decoder.finish()?;
        Ok(ChunkPayload {
            transfer_id,
            offset,
            total,
            bytes,
        })
    }

    /// Frame kind of a request frame.
    pub(crate) const fn request_frame_kind() -> u8 {
        REQUEST_FRAME
    }

    /// Frame kind of a response frame.
    pub(crate) const fn response_frame_kind() -> u8 {
        RESPONSE_FRAME
    }

    /// Frame kind of a chunk frame.
    pub(crate) const fn chunk_frame_kind() -> u8 {
        CHUNK_FRAME
    }

    /// Write one response frame.
    pub fn write_response<W: Write>(
        writer: &mut W,
        response: &CommandResponse,
    ) -> Result<(), CodecError> {
        writer.write_all(&Self::encode_response(response)?)?;
        Ok(())
    }

    /// Read one response frame.
    pub fn read_response<R: Read>(reader: &mut R) -> Result<CommandResponse, CodecError> {
        let (kind, payload) = read_frame(reader)?;
        if kind != RESPONSE_FRAME {
            return Err(CodecError::UnknownCommandTag(kind));
        }
        decode_response_payload(&payload)
    }
}

fn decode_request_payload(payload: &[u8]) -> Result<CommandRequest, CodecError> {
    let mut decoder = Decoder::new(payload);
    let tag = decoder.u8()?;
    let request = match tag {
        CAPABILITIES_REQUEST => CommandRequest::Capabilities,
        HEALTH_REQUEST => CommandRequest::Health,
        COMPILE_REQUEST => CommandRequest::Compile {
            request: get_compile_request(&mut decoder)?,
        },
        IMPORT_STAGED_LEASE_REQUEST => CommandRequest::ImportStagedLease {
            staged: get_staged_lease(&mut decoder)?,
        },
        RELEASE_STAGED_LEASE_REQUEST => CommandRequest::ReleaseStagedLease {
            lease_id: LeaseId::new(decoder.u64()?),
        },
        IMPORT_BORROWED_LEASE_REQUEST => CommandRequest::ImportBorrowedLease {
            reservation: get_reservation(&mut decoder)?,
        },
        RELEASE_BORROWED_LEASE_REQUEST => CommandRequest::ReleaseBorrowedLease {
            lease_id: LeaseId::new(decoder.u64()?),
        },
        SET_QUEUE_PRIORITIES_REQUEST => CommandRequest::SetQueuePriorities {
            tiers: get_queue_priorities(&mut decoder)?,
        },
        SUBMIT_REQUEST => CommandRequest::Submit {
            trace: get_trace(&mut decoder)?,
            resources: get_resources(&mut decoder)?,
        },
        SUBMIT_RENDER_REQUEST => CommandRequest::Submit {
            trace: get_trace_tagged(&mut decoder, false)?,
            resources: get_resources(&mut decoder)?,
        },
        SUBMIT_HEAP_ICB_REQUEST => CommandRequest::Submit {
            trace: get_trace_tagged(&mut decoder, true)?,
            resources: get_resources(&mut decoder)?,
        },
        WAIT_REQUEST => CommandRequest::Wait {
            token: get_token(&mut decoder)?,
            timeout: std::time::Duration::from_millis(decoder.u64()?),
        },
        READBACK_REQUEST => CommandRequest::Readback {
            token: get_token(&mut decoder)?,
        },
        CANCEL_REQUEST => CommandRequest::Cancel {
            token: get_token(&mut decoder)?,
        },
        RELEASE_PIPELINE_REQUEST => CommandRequest::ReleasePipeline {
            pipeline: get_pipeline(&mut decoder)?,
        },
        RELEASE_COMPLETION_REQUEST => CommandRequest::ReleaseCompletion {
            token: get_token(&mut decoder)?,
        },
        other => return Err(CodecError::UnknownCommandTag(other)),
    };
    decoder.finish()?;
    Ok(request)
}

fn decode_response_payload(payload: &[u8]) -> Result<CommandResponse, CodecError> {
    let mut decoder = Decoder::new(payload);
    let tag = decoder.u8()?;
    let response = match tag {
        CAPABILITIES_RESPONSE => {
            let epoch = get_epoch(&mut decoder)?;
            CommandResponse::Capabilities {
                epoch,
                capabilities: get_capabilities_legacy(&mut decoder)?,
            }
        }
        RENDER_CAPABILITIES_RESPONSE => CommandResponse::Capabilities {
            epoch: get_epoch(&mut decoder)?,
            capabilities: get_capabilities(&mut decoder)?,
        },
        HEALTH_RESPONSE => CommandResponse::Health {
            health: get_health(&mut decoder)?,
        },
        COMPILED_RESPONSE => CommandResponse::Compiled {
            pipeline: get_pipeline(&mut decoder)?,
        },
        IMPORTED_RESPONSE => CommandResponse::Imported,
        SUBMITTED_RESPONSE => CommandResponse::Submitted {
            submission: get_submission(&mut decoder)?,
        },
        OBSERVED_RESPONSE => CommandResponse::Observed {
            disposition: get_disposition(&mut decoder)?,
        },
        READBACK_RESPONSE => CommandResponse::Readback {
            readback: get_readback(&mut decoder)?,
        },
        QUEUE_PRIORITIES_RESPONSE => CommandResponse::QueuePriorities {
            installed: get_queue_priorities(&mut decoder)?,
        },
        RELEASED_RESPONSE => CommandResponse::Released,
        ERROR_RESPONSE => CommandResponse::Error {
            error: get_error(&mut decoder)?,
        },
        other => return Err(CodecError::UnknownCommandTag(other)),
    };
    decoder.finish()?;
    Ok(response)
}

fn reframe(kind: u8, payload: Vec<u8>) -> Vec<u8> {
    let mut frame = Vec::with_capacity(9 + payload.len());
    frame.extend_from_slice(&COMMAND_FRAME_MAGIC);
    frame.push(kind);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn frame(kind: u8, payload: Vec<u8>) -> Result<Vec<u8>, CodecError> {
    if payload.len() > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length: payload.len(),
            maximum: MAX_COMMAND_FRAME,
        });
    }
    Ok(reframe(kind, payload))
}

fn write_framed<W: Write>(writer: &mut W, kind: u8, payload: &[u8]) -> Result<(), CodecError> {
    if payload.len() > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length: payload.len(),
            maximum: MAX_COMMAND_FRAME,
        });
    }
    let mut header = [0u8; 9];
    header[..4].copy_from_slice(&COMMAND_FRAME_MAGIC);
    header[4] = kind;
    header[5..].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    writer.write_all(&header).map_err(CodecError::Io)?;
    writer.write_all(payload).map_err(CodecError::Io)?;
    Ok(())
}

fn unframe(frame: &[u8], expected: u8) -> Result<&[u8], CodecError> {
    if frame.len() < 9 {
        return Err(CodecError::TruncatedFrame {
            expected: 9,
            actual: frame.len(),
        });
    }
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&frame[..4]);
    if magic != COMMAND_FRAME_MAGIC {
        return Err(CodecError::BadMagic(magic));
    }
    let kind = frame[4];
    if kind != expected {
        return Err(CodecError::UnknownCommandTag(kind));
    }
    let length = u32::from_be_bytes([frame[5], frame[6], frame[7], frame[8]]) as usize;
    if length > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length,
            maximum: MAX_COMMAND_FRAME,
        });
    }
    if frame.len() != 9 + length {
        return Err(CodecError::TruncatedFrame {
            expected: 9 + length,
            actual: frame.len(),
        });
    }
    Ok(&frame[9..])
}

fn read_frame<R: Read>(reader: &mut R) -> Result<(u8, Vec<u8>), CodecError> {
    let mut header = [0u8; 9];
    read_header(reader, &mut header)?;
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&header[..4]);
    if magic != COMMAND_FRAME_MAGIC {
        return Err(CodecError::BadMagic(magic));
    }
    let kind = header[4];
    let length = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) as usize;
    if length > MAX_COMMAND_FRAME {
        return Err(CodecError::FrameTooLarge {
            length,
            maximum: MAX_COMMAND_FRAME,
        });
    }
    let mut payload = vec![0u8; length];
    read_payload(reader, &mut payload)?;
    Ok((kind, payload))
}

fn read_header<R: Read>(reader: &mut R, header: &mut [u8; 9]) -> Result<(), CodecError> {
    let mut read = 0;
    while read < header.len() {
        match reader.read(&mut header[read..]) {
            Ok(0) if read == 0 => return Err(CodecError::Eof),
            Ok(0) => {
                return Err(CodecError::TruncatedFrame {
                    expected: header.len(),
                    actual: read,
                })
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CodecError::Io(error)),
        }
    }
    Ok(())
}

fn read_payload<R: Read>(reader: &mut R, payload: &mut [u8]) -> Result<(), CodecError> {
    let mut read = 0;
    while read < payload.len() {
        match reader.read(&mut payload[read..]) {
            Ok(0) => {
                return Err(CodecError::TruncatedFrame {
                    expected: payload.len(),
                    actual: read,
                })
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(CodecError::Io(error)),
        }
    }
    Ok(())
}

struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn blob(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    fn text(&mut self, value: &str) {
        self.blob(value.as_bytes());
    }

    fn opt_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.u64(value);
            }
            None => self.u8(0),
        }
    }

    fn opt_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.text(value);
            }
            None => self.u8(0),
        }
    }

    fn array3(&mut self, value: [u64; 3]) {
        for axis in value {
            self.u64(axis);
        }
    }

    fn opt_array3(&mut self, value: Option<[u64; 3]>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.array3(value);
            }
            None => self.u8(0),
        }
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CodecError> {
        let remaining = self.remaining();
        if length > remaining {
            return Err(CodecError::TruncatedPayload {
                needed: length,
                remaining,
            });
        }
        let start = self.position;
        self.position += length;
        Ok(&self.bytes[start..self.position])
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_be_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_be_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_be_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, CodecError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(i64::from_be_bytes(bytes))
    }

    fn bool(&mut self) -> Result<bool, CodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(CodecError::UnknownEnumValue {
                field: "boolean",
                value,
            }),
        }
    }

    fn blob(&mut self) -> Result<Vec<u8>, CodecError> {
        let length = usize::try_from(self.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: self.remaining(),
        })?;
        Ok(self.take(length)?.to_vec())
    }

    fn text(&mut self) -> Result<String, CodecError> {
        String::from_utf8(self.blob()?).map_err(CodecError::InvalidUtf8)
    }

    fn opt_u64(&mut self) -> Result<Option<u64>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u64()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional u64",
                value,
            }),
        }
    }

    fn opt_text(&mut self) -> Result<Option<String>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.text()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional text",
                value,
            }),
        }
    }

    fn array3(&mut self) -> Result<[u64; 3], CodecError> {
        Ok([self.u64()?, self.u64()?, self.u64()?])
    }

    fn opt_array3(&mut self) -> Result<Option<[u64; 3]>, CodecError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.array3()?)),
            value => Err(CodecError::UnknownEnumValue {
                field: "optional u64 array",
                value,
            }),
        }
    }

    fn finish(self) -> Result<(), CodecError> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(CodecError::TrailingPayload {
                extra: self.bytes.len() - self.position,
            })
        }
    }
}

fn put_epoch(encoder: &mut Encoder, epoch: DeviceEpoch) {
    encoder.u64(epoch.get());
}

fn get_epoch(decoder: &mut Decoder<'_>) -> Result<DeviceEpoch, CodecError> {
    Ok(DeviceEpoch::new(decoder.u64()?))
}

fn put_token(encoder: &mut Encoder, token: &CompletionToken) {
    put_epoch(encoder, token.device_epoch);
    encoder.u64(token.submission_id.get());
}

fn get_token(decoder: &mut Decoder<'_>) -> Result<CompletionToken, CodecError> {
    Ok(CompletionToken {
        device_epoch: get_epoch(decoder)?,
        submission_id: SubmissionId::new(decoder.u64()?),
    })
}

/// One queue-priority table: a bounded count followed by one rank per tier.
///
/// The count is explicit so the decoder can refuse an oversized table before
/// reading it, and each tier travels as its scheduler rank
/// ([`QueuePriority::rank`]) rather than as a dense enum byte, so the wire value
/// is the same number the policy compares.
fn put_queue_priorities(encoder: &mut Encoder, tiers: &[QueuePriority]) -> Result<(), CodecError> {
    if tiers.len() > MAX_QUEUE_PRIORITIES {
        return Err(CodecError::QueuePriorityCount {
            count: tiers.len(),
            maximum: MAX_QUEUE_PRIORITIES,
        });
    }
    encoder.u64(tiers.len() as u64);
    for tier in tiers {
        encoder.u8(tier.rank());
    }
    Ok(())
}

fn get_queue_priorities(decoder: &mut Decoder<'_>) -> Result<Vec<QueuePriority>, CodecError> {
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::QueuePriorityCount {
        count: usize::MAX,
        maximum: MAX_QUEUE_PRIORITIES,
    })?;
    if count > MAX_QUEUE_PRIORITIES {
        return Err(CodecError::QueuePriorityCount {
            count,
            maximum: MAX_QUEUE_PRIORITIES,
        });
    }
    let mut tiers = Vec::with_capacity(count);
    for _ in 0..count {
        let rank = decoder.u8()?;
        // A rank this build does not know is refused, never folded to
        // `Default`: the frame carries a tier the decoder cannot honour.
        let tier = QueuePriority::from_rank(rank).ok_or(CodecError::UnknownEnumValue {
            field: "queue priority",
            value: rank,
        })?;
        tiers.push(tier);
    }
    Ok(tiers)
}

fn put_digest(encoder: &mut Encoder, digest: &SemanticDigest) {
    encoder.text(digest.scheme());
    encoder.blob(digest.bytes());
}

fn get_digest(decoder: &mut Decoder<'_>) -> Result<SemanticDigest, CodecError> {
    Ok(SemanticDigest::new(decoder.text()?, decoder.blob()?)?)
}

fn put_function_source(encoder: &mut Encoder, source: FunctionSource) {
    encoder.u8(match source {
        FunctionSource::SanitizedLl => 0,
        FunctionSource::BinaryAir => 1,
        FunctionSource::MetalSource => 2,
        FunctionSource::Metallib => 3,
    });
}

fn get_function_source(decoder: &mut Decoder<'_>) -> Result<FunctionSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(FunctionSource::SanitizedLl),
        1 => Ok(FunctionSource::BinaryAir),
        2 => Ok(FunctionSource::MetalSource),
        3 => Ok(FunctionSource::Metallib),
        value => Err(CodecError::UnknownEnumValue {
            field: "function source",
            value,
        }),
    }
}

fn put_shader_source(encoder: &mut Encoder, source: &ShaderSource) {
    match source {
        ShaderSource::SanitizedLl(source) => {
            encoder.u8(0);
            encoder.text(source);
        }
        ShaderSource::BinaryAir(source) => {
            encoder.u8(1);
            encoder.blob(source);
        }
        ShaderSource::MetalSource(source) => {
            encoder.u8(2);
            encoder.text(source);
        }
    }
}

fn get_shader_source(decoder: &mut Decoder<'_>) -> Result<ShaderSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(ShaderSource::SanitizedLl(decoder.text()?)),
        1 => Ok(ShaderSource::BinaryAir(decoder.blob()?)),
        2 => Ok(ShaderSource::MetalSource(decoder.text()?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "shader source",
            value,
        }),
    }
}

fn put_compile_request(encoder: &mut Encoder, request: &PipelineCompileRequest) {
    encoder.text(&request.entry_name);
    put_digest(encoder, &request.logical_digest);
    put_shader_source(encoder, &request.source);
}

fn get_compile_request(decoder: &mut Decoder<'_>) -> Result<PipelineCompileRequest, CodecError> {
    Ok(PipelineCompileRequest {
        entry_name: decoder.text()?,
        logical_digest: get_digest(decoder)?,
        source: get_shader_source(decoder)?,
    })
}

fn put_function_identity(encoder: &mut Encoder, function: &FunctionIdentity) {
    put_digest(encoder, &function.logical_digest);
    encoder.text(&function.entry_name);
    put_function_source(encoder, function.source);
}

fn get_function_identity(decoder: &mut Decoder<'_>) -> Result<FunctionIdentity, CodecError> {
    Ok(FunctionIdentity {
        logical_digest: get_digest(decoder)?,
        entry_name: decoder.text()?,
        source: get_function_source(decoder)?,
    })
}

fn put_dispatch_kind(encoder: &mut Encoder, kind: DispatchKind) {
    encoder.u8(match kind {
        DispatchKind::ThreadsExact => 0,
        DispatchKind::Threadgroups => 1,
    });
}

fn get_dispatch_kind(decoder: &mut Decoder<'_>) -> Result<DispatchKind, CodecError> {
    match decoder.u8()? {
        0 => Ok(DispatchKind::ThreadsExact),
        1 => Ok(DispatchKind::Threadgroups),
        value => Err(CodecError::UnknownEnumValue {
            field: "dispatch kind",
            value,
        }),
    }
}

fn put_access(encoder: &mut Encoder, access: BufferAccess) {
    encoder.u8(match access {
        BufferAccess::Read => 0,
        BufferAccess::Write => 1,
        BufferAccess::ReadWrite => 2,
        BufferAccess::Unused => 3,
    });
}

fn get_access(decoder: &mut Decoder<'_>) -> Result<BufferAccess, CodecError> {
    match decoder.u8()? {
        0 => Ok(BufferAccess::Read),
        1 => Ok(BufferAccess::Write),
        2 => Ok(BufferAccess::ReadWrite),
        3 => Ok(BufferAccess::Unused),
        value => Err(CodecError::UnknownEnumValue {
            field: "buffer access",
            value,
        }),
    }
}

fn put_footprint(encoder: &mut Encoder, footprint: &FootprintProof) {
    match footprint {
        FootprintProof::Static { max_bytes } => {
            encoder.u8(0);
            encoder.u64(*max_bytes);
        }
        FootprintProof::Affine { accesses } => {
            encoder.u8(1);
            encoder.u64(accesses.len() as u64);
            for access in accesses {
                encoder.u64(access.base_offset);
                encoder.u64(access.access_size);
                encoder.u64(access.terms.len() as u64);
                for term in &access.terms {
                    encoder.u8(term.axis);
                    encoder.u64(term.stride);
                }
            }
        }
        FootprintProof::Unbounded => encoder.u8(2),
    }
}

fn get_footprint(decoder: &mut Decoder<'_>) -> Result<FootprintProof, CodecError> {
    match decoder.u8()? {
        0 => Ok(FootprintProof::Static {
            max_bytes: decoder.u64()?,
        }),
        1 => {
            let count =
                usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
                    needed: usize::MAX,
                    remaining: decoder.remaining(),
                })?;
            let mut accesses = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                let base_offset = decoder.u64()?;
                let access_size = decoder.u64()?;
                let terms =
                    usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
                        needed: usize::MAX,
                        remaining: decoder.remaining(),
                    })?;
                let mut affine_terms = Vec::with_capacity(terms.min(1024));
                for _ in 0..terms {
                    affine_terms.push(AffineTerm {
                        axis: decoder.u8()?,
                        stride: decoder.u64()?,
                    });
                }
                accesses.push(AffineAccess {
                    base_offset,
                    access_size,
                    terms: affine_terms,
                });
            }
            Ok(FootprintProof::Affine { accesses })
        }
        2 => Ok(FootprintProof::Unbounded),
        value => Err(CodecError::UnknownEnumValue {
            field: "footprint proof",
            value,
        }),
    }
}

/// Encode the compute half of one pipeline-table entry.
///
/// The pre-render layout carries exactly these bytes, so this function stays
/// the legacy encoder for `Compiled`, `ReleasePipeline` and any compute-only
/// trace; the render half travels through [`put_pipeline_tagged`] only.
fn put_pipeline(encoder: &mut Encoder, pipeline: &CompiledComputePipeline) {
    put_epoch(encoder, pipeline.device_epoch);
    encoder.u64(pipeline.pipeline_id.get());
    put_function_identity(encoder, &pipeline.function);
    put_contract(encoder, &pipeline.contract);
}

/// Encode one pipeline-table entry of the tagged layout: the entry kind, then
/// the compute half, then the render half when the entry carries one.
fn put_pipeline_tagged(
    encoder: &mut Encoder,
    pipeline: &CompiledComputePipeline,
) -> Result<(), CodecError> {
    match &pipeline.render {
        Some(contract) => {
            encoder.u8(render_pipeline_kind(contract)?);
            put_pipeline(encoder, pipeline);
            put_render_pipeline_contract(encoder, contract)?;
        }
        None => {
            encoder.u8(PIPELINE_KIND_COMPUTE);
            put_pipeline(encoder, pipeline);
        }
    }
    Ok(())
}

/// The pipeline-table tag one render contract wears.
///
/// The tag is a function of the format count because that count is what the
/// pre-MRT body cannot express: one format keeps `PIPELINE_KIND_RENDER` and the
/// exact bytes it always had, two to [`MAX_COLOR_ATTACHMENTS`] take
/// `PIPELINE_KIND_RENDER_MRT`. A contract with no format at all, or with more
/// than the render track admits, is refused here — before the tag is written —
/// so a refused registration never emits a partial entry.
fn render_pipeline_kind(contract: &RenderPipelineContract) -> Result<u8, CodecError> {
    match bounded_color_format_count(contract.color_formats.len() as u64)? {
        1 => Ok(PIPELINE_KIND_RENDER),
        _ => Ok(PIPELINE_KIND_RENDER_MRT),
    }
}

fn put_render_pipeline_contract(
    encoder: &mut Encoder,
    contract: &RenderPipelineContract,
) -> Result<(), CodecError> {
    // The entry keeps the field order the single-format shape established
    // (both entry names, then the format, then the vertex layout); only the
    // format field changes shape, from one byte to a length-prefixed list, and
    // only when the tag says so.
    bounded_color_format_count(contract.color_formats.len() as u64)?;
    encoder.text(&contract.vertex_entry);
    encoder.text(&contract.fragment_entry);
    match contract.color_formats.as_slice() {
        [format] => put_attachment_format(encoder, *format),
        formats => {
            encoder.u64(formats.len() as u64);
            for format in formats {
                put_attachment_format(encoder, *format);
            }
        }
    }
    put_vertex_layout(encoder, &contract.vertex_layout)
}

/// Bound one contract's colour-format list.
///
/// The list is read as a length prefix on the wire and as a plain slice when
/// encoding, so the same bound serves both directions: an empty list describes
/// no attachment and a list longer than [`MAX_COLOR_ATTACHMENTS`] describes a
/// pass no render track admits. Both are refused with the count as written, and
/// on the decode side before a single format byte — or a `Vec` of that length —
/// is produced.
fn bounded_color_format_count(value: u64) -> Result<usize, CodecError> {
    let count = usize::try_from(value).unwrap_or(usize::MAX);
    if count == 0 || count > MAX_COLOR_ATTACHMENTS {
        return Err(CodecError::RenderPipelineFormatCount {
            count,
            maximum: MAX_COLOR_ATTACHMENTS,
        });
    }
    Ok(count)
}

/// Encode one vertex layout.
///
/// [`VertexLayout::None`] keeps the single `0x00` byte the pre-vertex pipeline
/// payload wrote, so every existing registration keeps its exact bytes; the
/// buffer variant appends its own length-prefixed block after the tag.
fn put_vertex_layout(encoder: &mut Encoder, layout: &VertexLayout) -> Result<(), CodecError> {
    match layout {
        VertexLayout::None => {
            encoder.u8(VERTEX_LAYOUT_NONE);
            Ok(())
        }
        VertexLayout::Buffers(buffers) => {
            if buffers.len() > MAX_VERTEX_BUFFERS {
                return Err(CodecError::VertexBufferCount {
                    count: buffers.len(),
                    maximum: MAX_VERTEX_BUFFERS,
                });
            }
            // Only a layout that declares a per-instance binding takes the
            // stepped tag; every all-per-vertex layout keeps the pre-v31 bytes
            // (`research/docs/23` §3.3, v31).
            let stepped = buffers
                .iter()
                .any(|buffer| buffer.step != VertexStep::PerVertex);
            encoder.u8(if stepped {
                VERTEX_LAYOUT_BUFFERS_STEPPED
            } else {
                VERTEX_LAYOUT_BUFFERS
            });
            encoder.u64(buffers.len() as u64);
            for buffer in buffers {
                if buffer.attributes.len() > MAX_VERTEX_ATTRIBUTES {
                    return Err(CodecError::VertexAttributeCount {
                        count: buffer.attributes.len(),
                        maximum: MAX_VERTEX_ATTRIBUTES,
                    });
                }
                encoder.u64(buffer.stride);
                if stepped {
                    encoder.u8(buffer.step.code());
                }
                encoder.u64(buffer.attributes.len() as u64);
                for attribute in &buffer.attributes {
                    encoder.u32(attribute.location);
                    encoder.u64(attribute.offset);
                    encoder.u8(attribute.format.code());
                }
            }
            Ok(())
        }
    }
}

/// Decode the compute half of one pipeline-table entry.
///
/// `render` is always `None`: the pre-render entry has no render half, and
/// [`get_pipeline_tagged`] fills it in for the layout that carries one.
fn get_pipeline(decoder: &mut Decoder<'_>) -> Result<CompiledComputePipeline, CodecError> {
    Ok(CompiledComputePipeline {
        device_epoch: get_epoch(decoder)?,
        pipeline_id: PipelineId::new(decoder.u64()?),
        function: get_function_identity(decoder)?,
        contract: get_contract(decoder)?,
        render: None,
    })
}

/// Decode one pipeline-table entry of the tagged layout.
fn get_pipeline_tagged(decoder: &mut Decoder<'_>) -> Result<CompiledComputePipeline, CodecError> {
    let kind = decoder.u8()?;
    let mut pipeline = get_pipeline(decoder)?;
    pipeline.render = match kind {
        PIPELINE_KIND_COMPUTE => None,
        PIPELINE_KIND_RENDER | PIPELINE_KIND_RENDER_MRT => {
            Some(get_render_pipeline_contract(decoder, kind)?)
        }
        tag => return Err(CodecError::UnknownPipelineTag(tag)),
    };
    Ok(pipeline)
}

/// Decode the render half of one pipeline-table entry.
///
/// `kind` selects the shape of the format field: `PIPELINE_KIND_RENDER` is a
/// single byte, `PIPELINE_KIND_RENDER_MRT` a length-prefixed list. A list
/// length outside `1..=MAX_COLOR_ATTACHMENTS` is refused rather than read as a
/// shorter or longer entry, so a corrupt prefix cannot shift the vertex layout
/// that follows it.
fn get_render_pipeline_contract(
    decoder: &mut Decoder<'_>,
    kind: u8,
) -> Result<RenderPipelineContract, CodecError> {
    let vertex_entry = decoder.text()?;
    let fragment_entry = decoder.text()?;
    let color_formats = match kind {
        PIPELINE_KIND_RENDER_MRT => {
            let count = bounded_color_format_count(decoder.u64()?)?;
            let mut formats = Vec::with_capacity(count);
            for _ in 0..count {
                formats.push(get_attachment_format(decoder)?);
            }
            formats
        }
        _ => vec![get_attachment_format(decoder)?],
    };
    Ok(RenderPipelineContract {
        vertex_entry,
        fragment_entry,
        color_formats,
        vertex_layout: get_vertex_layout(decoder)?,
    })
}

fn get_vertex_layout(decoder: &mut Decoder<'_>) -> Result<VertexLayout, CodecError> {
    match decoder.u8()? {
        VERTEX_LAYOUT_NONE => Ok(VertexLayout::None),
        kind @ (VERTEX_LAYOUT_BUFFERS | VERTEX_LAYOUT_BUFFERS_STEPPED) => {
            // The stepped tag adds one step byte per binding and leaves every
            // other field in place (`research/docs/23` §3.3, v31).
            let stepped = kind == VERTEX_LAYOUT_BUFFERS_STEPPED;
            let count = bounded_vertex_buffer_count(decoder.u64()?)?;
            let mut buffers = Vec::with_capacity(count);
            for _ in 0..count {
                let stride = decoder.u64()?;
                let step = if stepped {
                    let code = decoder.u8()?;
                    VertexStep::from_code(code).ok_or(CodecError::UnknownEnumValue {
                        field: "vertex step",
                        value: code,
                    })?
                } else {
                    VertexStep::PerVertex
                };
                let attribute_count = bounded_vertex_attribute_count(decoder.u64()?)?;
                let mut attributes = Vec::with_capacity(attribute_count);
                for _ in 0..attribute_count {
                    let location = decoder.u32()?;
                    let offset = decoder.u64()?;
                    let format = get_vertex_format(decoder)?;
                    attributes.push(VertexAttribute {
                        location,
                        offset,
                        format,
                    });
                }
                buffers.push(VertexBufferLayout {
                    stride,
                    step,
                    attributes,
                });
            }
            Ok(VertexLayout::Buffers(buffers))
        }
        value => Err(CodecError::UnknownEnumValue {
            field: "vertex layout",
            value,
        }),
    }
}

fn get_vertex_format(decoder: &mut Decoder<'_>) -> Result<VertexFormat, CodecError> {
    let code = decoder.u8()?;
    VertexFormat::from_code(code).ok_or(CodecError::UnknownEnumValue {
        field: "vertex format",
        value: code,
    })
}

fn get_index_format(decoder: &mut Decoder<'_>) -> Result<IndexFormat, CodecError> {
    let code = decoder.u8()?;
    IndexFormat::from_code(code).ok_or(CodecError::UnknownEnumValue {
        field: "index format",
        value: code,
    })
}

/// Read one vertex-buffer count and refuse it before it can drive an
/// allocation.
///
/// The bound is the contract's own cap ([`MAX_VERTEX_BUFFERS`]), which a
/// well-formed trace cannot exceed; a corrupt count is refused here instead of
/// sizing a `Vec` the payload cannot fill.
fn bounded_vertex_buffer_count(value: u64) -> Result<usize, CodecError> {
    let count = usize::try_from(value).unwrap_or(usize::MAX);
    if count > MAX_VERTEX_BUFFERS {
        return Err(CodecError::VertexBufferCount {
            count,
            maximum: MAX_VERTEX_BUFFERS,
        });
    }
    Ok(count)
}

/// Sibling of [`bounded_vertex_buffer_count`] for one layout's attribute list.
fn bounded_vertex_attribute_count(value: u64) -> Result<usize, CodecError> {
    let count = usize::try_from(value).unwrap_or(usize::MAX);
    if count > MAX_VERTEX_ATTRIBUTES {
        return Err(CodecError::VertexAttributeCount {
            count,
            maximum: MAX_VERTEX_ATTRIBUTES,
        });
    }
    Ok(count)
}

fn put_contract(encoder: &mut Encoder, contract: &PipelineContract) {
    put_dispatch_kind(encoder, contract.dispatch_kind);
    encoder.opt_array3(contract.required_local_size);
    encoder.opt_array3(contract.fixed_grid);
    encoder.u32(contract.push_constant_offset);
    encoder.u32(contract.push_constant_bytes);
    encoder.u64(contract.buffer_bindings.len() as u64);
    for binding in &contract.buffer_bindings {
        encoder.u32(binding.metal_binding);
        put_access(encoder, binding.access);
        put_footprint(encoder, &binding.footprint);
    }
    encoder.u64(contract.shader_capabilities.len() as u64);
    for capability in &contract.shader_capabilities {
        encoder.text(capability);
    }
    match &contract.translator_revision {
        Some(digest) => {
            encoder.u8(1);
            put_digest(encoder, digest);
        }
        None => encoder.u8(0),
    }
}

fn get_contract(decoder: &mut Decoder<'_>) -> Result<PipelineContract, CodecError> {
    let dispatch_kind = get_dispatch_kind(decoder)?;
    let required_local_size = decoder.opt_array3()?;
    let fixed_grid = decoder.opt_array3()?;
    let push_constant_offset = decoder.u32()?;
    let push_constant_bytes = decoder.u32()?;
    let binding_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut buffer_bindings = Vec::with_capacity(binding_count.min(1024));
    for _ in 0..binding_count {
        buffer_bindings.push(BufferBindingContract {
            metal_binding: decoder.u32()?,
            access: get_access(decoder)?,
            footprint: get_footprint(decoder)?,
        });
    }
    let capability_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut shader_capabilities = Vec::with_capacity(capability_count.min(1024));
    for _ in 0..capability_count {
        shader_capabilities.push(decoder.text()?);
    }
    let translator_revision = match decoder.u8()? {
        0 => None,
        1 => Some(get_digest(decoder)?),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "translator revision",
                value,
            })
        }
    };
    Ok(PipelineContract {
        dispatch_kind,
        required_local_size,
        fixed_grid,
        push_constant_offset,
        push_constant_bytes,
        buffer_bindings,
        shader_capabilities,
        translator_revision,
    })
}

fn put_view(encoder: &mut Encoder, view: &BufferView) {
    encoder.u64(view.view_id.get());
    encoder.u32(view.metal_binding);
    encoder.u64(view.allocation_id.get());
    encoder.u64(view.offset);
    encoder.u64(view.length);
    put_access(encoder, view.access);
    encoder.opt_u64(view.attribute_stride);
    match &view.source {
        BufferSource::OwnedBytes(bytes) => {
            encoder.u8(0);
            encoder.blob(bytes);
        }
        BufferSource::StagedLease(lease_id) => {
            encoder.u8(1);
            encoder.u64(lease_id.get());
        }
        BufferSource::BorrowedNoCopy(lease_id) => {
            encoder.u8(2);
            encoder.u64(lease_id.get());
        }
    }
}

fn get_view(decoder: &mut Decoder<'_>) -> Result<BufferView, CodecError> {
    let view_id = ViewId::new(decoder.u64()?);
    let metal_binding = decoder.u32()?;
    let allocation_id = AllocationId::new(decoder.u64()?);
    let offset = decoder.u64()?;
    let length = decoder.u64()?;
    let access = get_access(decoder)?;
    let attribute_stride = decoder.opt_u64()?;
    let source = match decoder.u8()? {
        0 => BufferSource::OwnedBytes(decoder.blob()?),
        1 => BufferSource::StagedLease(LeaseId::new(decoder.u64()?)),
        2 => BufferSource::BorrowedNoCopy(LeaseId::new(decoder.u64()?)),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "buffer source",
                value,
            })
        }
    };
    Ok(BufferView {
        view_id,
        metal_binding,
        allocation_id,
        offset,
        length,
        access,
        attribute_stride,
        source,
    })
}

fn put_texture_type(encoder: &mut Encoder, texture_type: TextureType) {
    encoder.u8(match texture_type {
        TextureType::D1 => 0,
        TextureType::D1Array => 1,
        TextureType::D2 => 2,
        TextureType::D2Array => 3,
        TextureType::D2Multisample => 4,
        TextureType::D2MultisampleArray => 5,
        TextureType::D3 => 6,
    });
}

fn get_texture_type(decoder: &mut Decoder<'_>) -> Result<TextureType, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureType::D1),
        1 => Ok(TextureType::D1Array),
        2 => Ok(TextureType::D2),
        3 => Ok(TextureType::D2Array),
        4 => Ok(TextureType::D2Multisample),
        5 => Ok(TextureType::D2MultisampleArray),
        6 => Ok(TextureType::D3),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture type",
            value,
        }),
    }
}

fn put_texture_format(encoder: &mut Encoder, format: TextureFormat) {
    encoder.u8(match format {
        TextureFormat::R32Uint => 0,
        TextureFormat::R32Float => 1,
        TextureFormat::Rgba8Unorm => 2,
        TextureFormat::Bgra8Unorm => 3,
    });
}

fn get_texture_format(decoder: &mut Decoder<'_>) -> Result<TextureFormat, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureFormat::R32Uint),
        1 => Ok(TextureFormat::R32Float),
        2 => Ok(TextureFormat::Rgba8Unorm),
        3 => Ok(TextureFormat::Bgra8Unorm),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture format",
            value,
        }),
    }
}

fn put_texture_access(encoder: &mut Encoder, access: TextureAccess) {
    encoder.u8(match access {
        TextureAccess::Sampled => 0,
        TextureAccess::Storage => 1,
        TextureAccess::Unused => 2,
    });
}

fn get_texture_access(decoder: &mut Decoder<'_>) -> Result<TextureAccess, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureAccess::Sampled),
        1 => Ok(TextureAccess::Storage),
        2 => Ok(TextureAccess::Unused),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture access",
            value,
        }),
    }
}

fn put_texture_source(encoder: &mut Encoder, source: &TextureSource) {
    match source {
        TextureSource::OwnedBytes(bytes) => {
            encoder.u8(0);
            encoder.blob(bytes);
        }
        TextureSource::StagedLease(lease_id) => {
            encoder.u8(1);
            encoder.u64(lease_id.get());
        }
        TextureSource::BorrowedNoCopy(lease_id) => {
            encoder.u8(2);
            encoder.u64(lease_id.get());
        }
    }
}

fn get_texture_source(decoder: &mut Decoder<'_>) -> Result<TextureSource, CodecError> {
    match decoder.u8()? {
        0 => Ok(TextureSource::OwnedBytes(decoder.blob()?)),
        1 => Ok(TextureSource::StagedLease(LeaseId::new(decoder.u64()?))),
        2 => Ok(TextureSource::BorrowedNoCopy(LeaseId::new(decoder.u64()?))),
        value => Err(CodecError::UnknownEnumValue {
            field: "texture source",
            value,
        }),
    }
}

fn put_texture(encoder: &mut Encoder, texture: &TextureView) {
    encoder.u64(texture.view_id.get());
    encoder.u32(texture.metal_binding);
    encoder.u64(texture.allocation_id.get());
    put_texture_type(encoder, texture.texture_type);
    put_texture_format(encoder, texture.format);
    encoder.u64(texture.width);
    encoder.u64(texture.height);
    encoder.u64(texture.depth);
    encoder.u64(texture.array_length);
    encoder.u64(texture.sample_count);
    put_texture_access(encoder, texture.access);
    put_texture_source(encoder, &texture.source);
}

fn get_texture(decoder: &mut Decoder<'_>) -> Result<TextureView, CodecError> {
    Ok(TextureView {
        view_id: ViewId::new(decoder.u64()?),
        metal_binding: decoder.u32()?,
        allocation_id: AllocationId::new(decoder.u64()?),
        texture_type: get_texture_type(decoder)?,
        format: get_texture_format(decoder)?,
        width: decoder.u64()?,
        height: decoder.u64()?,
        depth: decoder.u64()?,
        array_length: decoder.u64()?,
        sample_count: decoder.u64()?,
        access: get_texture_access(decoder)?,
        source: get_texture_source(decoder)?,
    })
}

fn put_dispatch(encoder: &mut Encoder, dispatch: &Dispatch) {
    put_dispatch_kind(encoder, dispatch.kind);
    encoder.array3(dispatch.grid);
    encoder.array3(dispatch.threads_per_threadgroup);
}

fn get_dispatch(decoder: &mut Decoder<'_>) -> Result<Dispatch, CodecError> {
    Ok(Dispatch {
        kind: get_dispatch_kind(decoder)?,
        grid: decoder.array3()?,
        threads_per_threadgroup: decoder.array3()?,
    })
}

fn put_dispatch_type(encoder: &mut Encoder, dispatch_type: DispatchType) {
    encoder.u8(match dispatch_type {
        DispatchType::Serial => 0,
        DispatchType::Concurrent => 1,
    });
}

fn get_dispatch_type(decoder: &mut Decoder<'_>) -> Result<DispatchType, CodecError> {
    match decoder.u8()? {
        0 => Ok(DispatchType::Serial),
        1 => Ok(DispatchType::Concurrent),
        value => Err(CodecError::UnknownEnumValue {
            field: "dispatch type",
            value,
        }),
    }
}

fn put_completion_policy(encoder: &mut Encoder, policy: CompletionPolicy) {
    encoder.u8(match policy {
        CompletionPolicy::HostReadback => 0,
        CompletionPolicy::SubmitOnly => 1,
    });
}

fn get_completion_policy(decoder: &mut Decoder<'_>) -> Result<CompletionPolicy, CodecError> {
    match decoder.u8()? {
        0 => Ok(CompletionPolicy::HostReadback),
        1 => Ok(CompletionPolicy::SubmitOnly),
        value => Err(CodecError::UnknownEnumValue {
            field: "completion policy",
            value,
        }),
    }
}

/// Encode one trace, choosing the pass layout from its own contents.
///
/// A compute-only trace writes the pre-render pass bytes; a render-bearing
/// trace tags every pass. The caller writes the matching payload tag, so
/// `SUBMIT_REQUEST` frames keep their exact legacy bytes and
/// `SUBMIT_RENDER_REQUEST` frames are self-describing.
fn put_trace(encoder: &mut Encoder, trace: &ComputeTrace) -> Result<(), CodecError> {
    let tagged = trace.has_render_passes() || trace.has_heap_or_icb();
    if tagged && trace.passes.len() > MAX_TAGGED_TRACE_PASSES {
        return Err(CodecError::TracePassCount {
            count: trace.passes.len(),
            maximum: MAX_TAGGED_TRACE_PASSES,
        });
    }
    encoder.u16(trace.schema_version);
    put_epoch(encoder, trace.device_epoch);
    encoder.u64(trace.operation_id.get());
    encoder.u64(trace.pipelines.len() as u64);
    for pipeline in &trace.pipelines {
        if tagged {
            put_pipeline_tagged(encoder, pipeline)?;
        } else {
            put_pipeline(encoder, pipeline);
        }
    }
    put_dispatch_type(encoder, trace.encoder_dispatch_type);
    encoder.u64(trace.passes.len() as u64);
    for pass in &trace.passes {
        match pass {
            TracePass::Compute(pass) => {
                if tagged {
                    encoder.u8(PASS_KIND_COMPUTE);
                }
                put_compute_pass(encoder, pass);
            }
            TracePass::Render(pass) => {
                if pass.color_attachments.len() > MAX_COLOR_ATTACHMENTS {
                    return Err(CodecError::ColorAttachmentCount {
                        count: pass.color_attachments.len(),
                        maximum: MAX_COLOR_ATTACHMENTS,
                    });
                }
                // `tagged` is true whenever a render entry exists, so the tag
                // below always belongs to the extended layout.
                // The pass kind carries the present half: an offscreen pass
                // keeps `PASS_KIND_RENDER` and its previous bytes exactly, and
                // a presenting pass is a tag an older decoder refuses
                // (`docs/24` §4.1, §4.3).
                //
                // A pass that binds a caller-held vertex stream takes the
                // feature-tagged kind instead, because two of the three
                // optional sections would otherwise multiply the tag space
                // (`docs/23` §3.3). Its present half travels as a feature bit,
                // so the pair stays orthogonal.
                let has_vertex_input = !pass.vertex_buffers.is_empty() || pass.indices.is_some();
                // The instancing tail follows the same rule (`docs/23` §3.3,
                // v31): a single-instance pass is the shape every earlier
                // increment wrote, so only a multi-instance draw takes the
                // extended kind and appends its own count.
                let has_instancing = pass.instance_count != 1;
                // The base vertex follows the same rule (`docs/23` §3.3, v34):
                // only a draw that offsets its indices takes the extended kind
                // and appends its own field.
                let has_base_vertex = pass.base_vertex != 0;
                // The depth block follows the same rule (`docs/23` §3.3, v36):
                // only a pass that carries a depth attachment takes the
                // extended kind and appends its shape.
                let has_depth = pass.depth.is_some();
                // The culling state follows the same rule (`docs/23` §3.3,
                // v39): only a pass that culls something takes the extended
                // kind and appends its state.
                let has_cull = pass.cull.is_some();
                let has_blend = pass.blend.is_some();
                // The depth store action and the depth identity are the first
                // sections that do not fit the narrow feature byte
                // (`docs/23` §3.3, v43): both travel under the wide tag, whose
                // word has a second byte for exactly this purpose, and neither
                // is ever written by a frame that only needs narrow bits.
                let has_depth_store = pass
                    .depth
                    .as_ref()
                    .is_some_and(|depth| depth.store.is_some());
                let has_depth_resource = pass
                    .depth
                    .as_ref()
                    .is_some_and(|depth| depth.identity.is_some());
                // The stencil block is the same story (`docs/23` §3.3, v47): a
                // pass with no stencil attachment never sets the bit, so its
                // bytes stay exactly what they were.
                let has_stencil = pass.stencil.is_some();
                // The stencil store action and identity follow the depth pair's
                // own rule (`docs/23` §3.3, v43/v49).
                let has_stencil_store = pass
                    .stencil
                    .as_ref()
                    .is_some_and(|stencil| stencil.store.is_some());
                let has_stencil_resource = pass
                    .stencil
                    .as_ref()
                    .is_some_and(|stencil| stencil.identity.is_some());
                // The multisample state is the pass's own, so its bit is the
                // pass's own statement too (`research/docs/23` §3.3, v51).
                let has_multisample = pass.multisample.is_some();
                // The depth resolve is the stored depth surface's own tail
                // (`research/docs/23` §3.3, v57): a pass that never resolves
                // never sets the bit, and the contract refuses the bit without
                // the stored multisampled depth surface it reduces.
                let has_depth_resolve = pass.depth_resolve.is_some();
                // The stencil resolve is the stored stencil surface's own tail
                // (`research/docs/23` §3.3, v60): a pass that never resolves
                // never sets the bit, and the contract refuses the bit without
                // the stored multisampled stencil surface it reduces.
                let has_stencil_resolve = pass.stencil_resolve.is_some();
                // The sampled-texture block is the section the wide word had
                // no bit for (`research/docs/23` §3.3, v70): a pass that binds
                // no fragment texture never sets it, so every pre-v70 frame
                // keeps its exact bytes and only a texture-bearing pass takes
                // the tag of its own.
                let has_render_textures = !pass.textures.is_empty();
                if has_render_textures && pass.textures.len() > MAX_RENDER_TEXTURES {
                    return Err(CodecError::RenderTextureCount {
                        count: pass.textures.len(),
                        maximum: MAX_RENDER_TEXTURES,
                    });
                }
                let wide = has_depth_store
                    || has_depth_resource
                    || has_stencil
                    || has_stencil_store
                    || has_stencil_resource
                    || has_multisample
                    || has_depth_resolve
                    || has_stencil_resolve;
                if has_vertex_input
                    || pass.scissor.is_some()
                    || has_instancing
                    || has_base_vertex
                    || has_depth
                    || has_cull
                    || has_blend
                    || wide
                    || has_render_textures
                {
                    let mut features = if has_vertex_input {
                        RENDER_FEATURE_VERTEX_INPUT
                    } else {
                        0
                    };
                    if pass.present.is_some() {
                        features |= RENDER_FEATURE_PRESENT;
                    }
                    if pass.scissor.is_some() {
                        features |= RENDER_FEATURE_SCISSOR;
                    }
                    if has_instancing {
                        features |= RENDER_FEATURE_INSTANCING;
                    }
                    if has_base_vertex {
                        features |= RENDER_FEATURE_BASE_VERTEX;
                    }
                    if has_depth {
                        features |= RENDER_FEATURE_DEPTH;
                    }
                    if has_cull {
                        features |= RENDER_FEATURE_CULL;
                    }
                    if has_blend {
                        features |= RENDER_FEATURE_BLEND;
                    }
                    if wide || has_render_textures {
                        // The wide word's low byte is the narrow byte, so a
                        // decoder reads both tags through one section walker
                        // and only the extra bits differ.
                        let mut wide_features = u16::from(features);
                        if has_depth_store {
                            wide_features |= RENDER_WIDE_FEATURE_DEPTH_STORE;
                        }
                        if has_depth_resource {
                            wide_features |= RENDER_WIDE_FEATURE_DEPTH_RESOURCE;
                        }
                        if has_stencil {
                            wide_features |= RENDER_WIDE_FEATURE_STENCIL;
                        }
                        if has_stencil_store {
                            wide_features |= RENDER_WIDE_FEATURE_STENCIL_STORE;
                        }
                        if has_stencil_resource {
                            wide_features |= RENDER_WIDE_FEATURE_STENCIL_RESOURCE;
                        }
                        if has_multisample {
                            wide_features |= RENDER_WIDE_FEATURE_MULTISAMPLE;
                        }
                        if has_depth_resolve {
                            wide_features |= RENDER_WIDE_FEATURE_DEPTH_RESOLVE;
                        }
                        if has_stencil_resolve {
                            wide_features |= RENDER_WIDE_FEATURE_STENCIL_RESOLVE;
                        }
                        if has_render_textures {
                            // The wide word is full, so the texture block
                            // rides the tag of its own: the same `u16` word
                            // (same meanings, so the section walker is
                            // shared) followed by the block
                            // (`research/docs/23` §3.3, v70).
                            encoder.u8(PASS_KIND_RENDER_SAMPLED);
                            encoder.u16(wide_features);
                            encoder.u8(u8::try_from(pass.textures.len()).map_err(|_| {
                                CodecError::RenderTextureCount {
                                    count: pass.textures.len(),
                                    maximum: MAX_RENDER_TEXTURES,
                                }
                            })?);
                            for texture in &pass.textures {
                                put_texture(encoder, texture);
                            }
                        } else {
                            encoder.u8(PASS_KIND_RENDER_EXT_WIDE);
                            encoder.u16(wide_features);
                        }
                    } else {
                        encoder.u8(PASS_KIND_RENDER_EXT);
                        encoder.u8(features);
                    }
                    put_render_pass(encoder, pass, false)?;
                    if has_vertex_input {
                        put_vertex_input(encoder, pass)?;
                    }
                    if let Some(present) = &pass.present {
                        put_present_descriptor(encoder, present)?;
                    }
                    if let Some([x, y, width, height]) = pass.scissor {
                        for dimension in [x, y, width, height] {
                            encoder.u32(dimension);
                        }
                    }
                    if has_instancing {
                        encoder.u32(pass.instance_count);
                    }
                    if has_base_vertex {
                        encoder.u32(pass.base_vertex);
                    }
                    if let Some(depth) = &pass.depth {
                        put_depth_block(encoder, depth, pass.depth_test.as_ref())?;
                        // The two wide sections follow the depth block, in the
                        // order their bits are declared: the store action
                        // first, then the identity the storing shape lands on
                        // (`docs/23` §3.3, v43).
                        if let Some(store) = depth.store {
                            encoder.u8(store.code());
                        }
                        if let Some(identity) = &depth.identity {
                            encoder.u64(identity.allocation_id.get());
                            encoder.u64(identity.view_id.get());
                        }
                    }
                    // The stencil block follows every depth section, so the two
                    // surfaces' state cannot be read in either order by a
                    // decoder that knows both bits (`docs/23` §3.3, v47).
                    if let Some(stencil) = &pass.stencil {
                        put_stencil_block(encoder, stencil, pass.stencil_test.as_ref());
                        if let Some(store) = stencil.store {
                            encoder.u8(u8::from(store == StoreOp::Store));
                        }
                        if let Some(identity) = &stencil.identity {
                            encoder.u64(identity.allocation_id.get());
                            encoder.u64(identity.view_id.get());
                        }
                    }
                    // The multisample section follows the stencil sections and
                    // precedes culling, in the same order the decoder walks
                    // (`research/docs/23` §3.3, v51): one sample count code.
                    if let Some(multisample) = &pass.multisample {
                        encoder.u8(multisample.sample_count.code());
                    }
                    // The depth resolve section follows the multisample
                    // section and precedes culling, in the same order the
                    // decoder walks (`research/docs/23` §3.3, v57): one
                    // filter code.
                    if let Some(resolve) = &pass.depth_resolve {
                        encoder.u8(resolve.filter.code());
                    }
                    // The stencil resolve section follows the depth resolve
                    // section and precedes culling, in the same order the
                    // decoder walks (`research/docs/23` §3.3, v60): one
                    // filter code.
                    if let Some(resolve) = &pass.stencil_resolve {
                        encoder.u8(resolve.filter.code());
                    }
                    if let Some(cull) = &pass.cull {
                        encoder.u8(cull.mode.code());
                        encoder.u8(cull.winding.code());
                    }
                    if let Some(blend) = &pass.blend {
                        encoder.u64(blend.attachments.len() as u64);
                        for attachment in &blend.attachments {
                            encoder.u8(attachment.source_rgb.code());
                            encoder.u8(attachment.destination_rgb.code());
                            encoder.u8(attachment.source_alpha.code());
                            encoder.u8(attachment.destination_alpha.code());
                            encoder.u8(attachment.operation.code());
                        }
                    }
                    continue;
                }
                encoder.u8(if pass.present.is_some() {
                    PASS_KIND_RENDER_PRESENT
                } else {
                    PASS_KIND_RENDER
                });
                put_render_pass(encoder, pass, true)?;
            }
        }
    }
    put_completion_policy(encoder, trace.completion_policy);
    if trace.has_heap_or_icb() {
        put_heap_icb_tail(encoder, trace)?;
    }
    Ok(())
}

fn put_compute_pass(encoder: &mut Encoder, pass: &ComputePass) {
    encoder.u64(pass.pipeline.get());
    encoder.u64(pass.buffers.len() as u64);
    for view in &pass.buffers {
        put_view(encoder, view);
    }
    encoder.u64(pass.textures.len() as u64);
    for texture in &pass.textures {
        put_texture(encoder, texture);
    }
    put_dispatch(encoder, &pass.dispatch);
}

/// Encode one render pass's base payload: the pipeline, the colour
/// attachments, the viewport and the vertex/index count.
///
/// `with_present` is what keeps the two tag families byte-exact. A legacy
/// tagged pass writes its present tail here, exactly as it did before the
/// vertex-input tag existed; an extended pass writes it after the vertex-input
/// block instead, because that block sits between the base payload and the
/// tail.
fn put_render_pass(
    encoder: &mut Encoder,
    pass: &RenderPassDescriptor,
    with_present: bool,
) -> Result<(), CodecError> {
    encoder.u64(pass.pipeline.get());
    encoder.u64(pass.color_attachments.len() as u64);
    for attachment in &pass.color_attachments {
        put_render_attachment(encoder, attachment);
    }
    for dimension in pass.viewport {
        encoder.u32(dimension);
    }
    encoder.u32(pass.vertices);
    if with_present {
        if let Some(present) = &pass.present {
            put_present_descriptor(encoder, present)?;
        }
    }
    Ok(())
}

/// Encode the vertex-input block of an extended render pass: the bound vertex
/// streams by their own view identity, then the index buffer when the draw is
/// indexed.
///
/// Each stream travels as a full [`BufferView`] — identity, range and source
/// bytes — because a render input declares its own bytes rather than
/// referencing a compute binding (`research/docs/23` §3.6). The encoder reuses
/// the compute path's view encoding for the same reason: one declaration shape
/// means one set of source rules.
fn put_vertex_input(encoder: &mut Encoder, pass: &RenderPassDescriptor) -> Result<(), CodecError> {
    if pass.vertex_buffers.len() > MAX_VERTEX_BUFFERS {
        return Err(CodecError::VertexBufferCount {
            count: pass.vertex_buffers.len(),
            maximum: MAX_VERTEX_BUFFERS,
        });
    }
    encoder.u64(pass.vertex_buffers.len() as u64);
    for view in &pass.vertex_buffers {
        put_view(encoder, view);
    }
    match &pass.indices {
        Some(indices) => {
            encoder.u8(1);
            put_view(encoder, &indices.view);
            encoder.u8(indices.format.code());
        }
        None => encoder.u8(0),
    }
    Ok(())
}

/// Encode one present action as the render pass's trailing action
/// (`research/docs/24` §3.5, shape one).
///
/// Every field is written in the order the Step 1 value type declares it, and
/// the two optional halves are spelled as explicit tags rather than as a
/// length-prefixed block, so a decoder reads `InitialState::Undefined` and
/// `AcquirePolicy::Blocking` — the first increment's only admitted values —
/// without having to interpret a sentinel budget it is not going to use.
fn put_present_descriptor(
    encoder: &mut Encoder,
    present: &PresentDescriptor,
) -> Result<(), CodecError> {
    let target = &present.target;
    encoder.u64(target.allocation_id.get());
    encoder.u64(target.view_id.get());
    put_attachment_format(encoder, target.format);
    encoder.u64(target.width);
    encoder.u64(target.height);
    encoder.u32(target.image_count);
    match &target.initial {
        InitialState::Sentinel(bytes) => {
            if bytes.len() > MAX_PRESENT_SENTINEL_BYTES {
                return Err(CodecError::PresentSentinelLength {
                    length: bytes.len(),
                    maximum: MAX_PRESENT_SENTINEL_BYTES,
                });
            }
            encoder.u8(0);
            encoder.blob(bytes);
        }
        InitialState::Undefined => encoder.u8(1),
    }
    encoder.u64(present.source.get());
    encoder.u8(present.mode.code());
    // `AcquirePolicy::Timeout` carries its nanosecond budget, which is exactly
    // what the Step 1 value type keeps expressible so a refusal can name the
    // deadline instead of the trace losing the request (`docs/24` §3.1).
    encoder.u8(present.acquire.code());
    if let Some(nanos) = present.acquire.timeout_nanos() {
        encoder.u64(nanos);
    }
    Ok(())
}

fn put_render_attachment(encoder: &mut Encoder, attachment: &RenderAttachment) {
    encoder.u64(attachment.view_id.get());
    encoder.u64(attachment.allocation_id.get());
    put_attachment_format(encoder, attachment.format);
    encoder.u64(attachment.width);
    encoder.u64(attachment.height);
    put_load_op(encoder, attachment.load);
    put_store_op(encoder, attachment.store);
}

fn put_attachment_format(encoder: &mut Encoder, format: AttachmentFormat) {
    encoder.u8(format.code());
}

fn put_load_op(encoder: &mut Encoder, load: LoadOp) {
    match load {
        LoadOp::Clear(color) => {
            encoder.u8(0);
            // Fixed four-byte payload: no length prefix, so the clear value
            // cannot be separated from its tag by a malformed length.
            encoder.bytes.extend_from_slice(&color.bytes);
        }
        LoadOp::Load => encoder.u8(1),
        LoadOp::DontCare => encoder.u8(2),
        // R7 (`research/docs/23` §76): the resident load carries no payload —
        // the identity is the attachment's own, so the tag is the whole
        // declaration, exactly like `Load`'s.
        LoadOp::Resident => encoder.u8(3),
    }
}

fn put_store_op(encoder: &mut Encoder, store: StoreOp) {
    encoder.u8(match store {
        StoreOp::Store => 0,
        StoreOp::DontCare => 1,
        // R7 (`research/docs/23` §76): the resident store's source of bytes is
        // the pass's own raster, so the tag is the whole declaration.
        StoreOp::Resident => 2,
    });
}

/// Encode the heap/ICB tail that follows the completion policy of a
/// [`SUBMIT_HEAP_ICB_REQUEST`] frame.
///
/// Each optional payload is spelled as an explicit presence byte rather than a
/// length-prefixed block, so a frame cannot claim a heap section it did not
/// write and an older decoder that somehow reaches the tail refuses the
/// unknown presence byte instead of reading garbage as the completion policy
/// (`research/docs/25-heaps与ICB设计.md` §4.5).
fn put_heap_icb_tail(encoder: &mut Encoder, trace: &ComputeTrace) -> Result<(), CodecError> {
    encoder.bool(trace.heap.is_some());
    if let Some(heap) = &trace.heap {
        put_heap_payload(encoder, heap)?;
    }
    encoder.bool(trace.indirect.is_some());
    if let Some(indirect) = &trace.indirect {
        put_indirect_payload(encoder, indirect)?;
    }
    Ok(())
}

/// Encode one heap payload: the descriptor followed by its placements.
fn put_heap_payload(encoder: &mut Encoder, heap: &HeapPayload) -> Result<(), CodecError> {
    encoder.u64(heap.descriptor.size);
    put_storage_mode(encoder, heap.descriptor.storage_mode);
    encoder.bool(heap.descriptor.allows_aliasing);
    if heap.placements.len() > MAX_HEAP_PLACEMENTS {
        return Err(CodecError::HeapPlacementCount {
            count: heap.placements.len(),
            maximum: MAX_HEAP_PLACEMENTS,
        });
    }
    encoder.u64(heap.placements.len() as u64);
    for placement in &heap.placements {
        put_heap_placement(encoder, placement);
    }
    Ok(())
}

/// Encode one heap placement: its heap identity, offset and resource.
fn put_heap_placement(encoder: &mut Encoder, placement: &HeapPlacement) {
    encoder.u64(placement.heap_id.get());
    encoder.u64(placement.offset);
    put_heap_resource(encoder, placement.resource);
}

/// Encode the resource a heap placement puts at an offset. The kind is written
/// before the byte extent so an unknown kind is a decoder refusal rather than a
/// silent default.
fn put_heap_resource(encoder: &mut Encoder, resource: HeapResource) {
    match resource {
        HeapResource::Buffer { byte_size } => {
            encoder.u8(0);
            encoder.u64(byte_size);
        }
        HeapResource::Texture { byte_size } => {
            encoder.u8(1);
            encoder.u64(byte_size);
        }
    }
}

/// Encode one indirect-command payload: the buffer descriptor, one command and
/// the replay range.
fn put_indirect_payload(
    encoder: &mut Encoder,
    indirect: &IndirectCommandPayload,
) -> Result<(), CodecError> {
    encoder.u32(indirect.buffer.max_commands);
    if indirect.buffer.kinds.len() > MAX_SUPPORTED_INDIRECT_COMMANDS {
        return Err(CodecError::IndirectCommandKindCount {
            count: indirect.buffer.kinds.len(),
            maximum: MAX_SUPPORTED_INDIRECT_COMMANDS,
        });
    }
    encoder.u64(indirect.buffer.kinds.len() as u64);
    for kind in &indirect.buffer.kinds {
        encoder.u8(kind.code());
    }
    put_indirect_command(encoder, &indirect.command);
    encoder.u32(indirect.range.start);
    encoder.u32(indirect.range.count);
    Ok(())
}

/// Encode one indirect command as its kind followed by the closed variant's
/// fields.
fn put_indirect_command(encoder: &mut Encoder, command: &IndirectCommandDescriptor) {
    match *command {
        IndirectCommandDescriptor::Draw {
            vertex_count,
            instance_count,
        } => {
            encoder.u8(IndirectCommandKind::Draw.code());
            encoder.u32(vertex_count);
            encoder.u32(instance_count);
        }
        IndirectCommandDescriptor::DrawIndexed {
            index_count,
            instance_count,
        } => {
            encoder.u8(IndirectCommandKind::DrawIndexed.code());
            encoder.u32(index_count);
            encoder.u32(instance_count);
        }
        IndirectCommandDescriptor::Dispatch { threadgroups } => {
            encoder.u8(IndirectCommandKind::Dispatch.code());
            encoder.u32(threadgroups[0]);
            encoder.u32(threadgroups[1]);
            encoder.u32(threadgroups[2]);
        }
    }
}

fn get_trace(decoder: &mut Decoder<'_>) -> Result<ComputeTrace, CodecError> {
    let schema_version = decoder.u16()?;
    let device_epoch = get_epoch(decoder)?;
    let operation_id = OperationId::new(decoder.u64()?);
    let pipeline_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut pipelines = Vec::with_capacity(pipeline_count.min(1024));
    for _ in 0..pipeline_count {
        pipelines.push(get_pipeline(decoder)?);
    }
    let encoder_dispatch_type = get_dispatch_type(decoder)?;
    let pass_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut passes = Vec::with_capacity(pass_count.min(1024));
    for _ in 0..pass_count {
        passes.push(TracePass::Compute(get_compute_pass(decoder)?));
    }
    Ok(ComputeTrace {
        schema_version,
        device_epoch,
        operation_id,
        pipelines,
        encoder_dispatch_type,
        passes,
        completion_policy: get_completion_policy(decoder)?,
        heap: None,
        indirect: None,
    })
}

/// Decode a trace whose pass list is tagged. Only a `SUBMIT_RENDER_REQUEST`
/// or `SUBMIT_HEAP_ICB_REQUEST` frame uses this layout; a `SUBMIT_REQUEST`
/// frame keeps decoding through [`get_trace`] and stays compute-only. When the
/// frame tag is the heap/ICB one, the completion policy is followed by the
/// tagged heap/ICB tail, which `has_heap_icb_tail` selects.
fn get_trace_tagged(
    decoder: &mut Decoder<'_>,
    has_heap_icb_tail: bool,
) -> Result<ComputeTrace, CodecError> {
    let schema_version = decoder.u16()?;
    let device_epoch = get_epoch(decoder)?;
    let operation_id = OperationId::new(decoder.u64()?);
    let pipeline_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut pipelines = Vec::with_capacity(pipeline_count.min(1024));
    for _ in 0..pipeline_count {
        pipelines.push(get_pipeline_tagged(decoder)?);
    }
    let encoder_dispatch_type = get_dispatch_type(decoder)?;
    let pass_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    if pass_count > MAX_TAGGED_TRACE_PASSES {
        return Err(CodecError::TracePassCount {
            count: pass_count,
            maximum: MAX_TAGGED_TRACE_PASSES,
        });
    }
    let mut passes = Vec::with_capacity(pass_count.min(1024));
    for _ in 0..pass_count {
        passes.push(match decoder.u8()? {
            PASS_KIND_COMPUTE => TracePass::Compute(get_compute_pass(decoder)?),
            PASS_KIND_RENDER => TracePass::Render(get_render_pass(decoder, false)?),
            // The present half is a property of the tag, not of a field inside
            // the render payload, so a frame cannot claim a present section it
            // did not write (`docs/24` §4.1).
            PASS_KIND_RENDER_PRESENT => TracePass::Render(get_render_pass(decoder, true)?),
            // The extended kind carries a feature byte: bit 0 is the
            // vertex-input block and bit 1 the present tail, in that order
            // after the base payload. A bit this version does not know is a
            // decoder refusal, so a future section cannot be skipped silently.
            PASS_KIND_RENDER_EXT => {
                // As of v40 every bit of the narrow byte is known, so there is
                // no unknown bit left for a decoder to refuse there: the next
                // optional section needed a second byte or a tag of its own,
                // and v43 chose the wide tag below. `RENDER_FEATURE_KNOWN`
                // stays as the record of the narrow byte's contents for that
                // decision.
                let features = decoder.u8()?;
                TracePass::Render(get_render_ext_pass(decoder, u16::from(features))?)
            }
            // The wide kind carries a `u16` feature word whose low byte is the
            // narrow byte above (`docs/23` §3.3, v43). Its high bits name the
            // sections the narrow byte had no room for, so the two tags share
            // one section walker and only the masks differ. An unknown high bit
            // is refused here exactly as an unknown tag is: a section this
            // decoder does not know cannot be skipped to reach the ones after
            // it.
            PASS_KIND_RENDER_EXT_WIDE => {
                let features = decoder.u16()?;
                let unknown = features & !RENDER_WIDE_FEATURE_KNOWN;
                if unknown != 0 {
                    return Err(CodecError::UnknownRenderFeature(unknown));
                }
                TracePass::Render(get_render_ext_pass(decoder, features)?)
            }
            // The sampled kind carries the same wide feature word as the tag
            // above and, right after it, the texture block the wide word had
            // no bit for (`research/docs/23` §3.3, v70). An unknown bit is
            // refused exactly as it is there: a section this decoder does not
            // know cannot be skipped to reach the ones after it.
            PASS_KIND_RENDER_SAMPLED => {
                let features = decoder.u16()?;
                let unknown = features & !RENDER_WIDE_FEATURE_KNOWN;
                if unknown != 0 {
                    return Err(CodecError::UnknownRenderFeature(unknown));
                }
                TracePass::Render(get_render_sampled_pass(decoder, features)?)
            }
            tag => return Err(CodecError::UnknownPassTag(tag)),
        });
    }
    let completion_policy = get_completion_policy(decoder)?;
    let (heap, indirect) = if has_heap_icb_tail {
        get_heap_icb_tail(decoder)?
    } else {
        (None, None)
    };
    Ok(ComputeTrace {
        schema_version,
        device_epoch,
        operation_id,
        pipelines,
        encoder_dispatch_type,
        passes,
        completion_policy,
        heap,
        indirect,
    })
}

fn get_compute_pass(decoder: &mut Decoder<'_>) -> Result<ComputePass, CodecError> {
    let pipeline = PipelineId::new(decoder.u64()?);
    let view_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut buffers = Vec::with_capacity(view_count.min(1024));
    for _ in 0..view_count {
        buffers.push(get_view(decoder)?);
    }
    let texture_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    let mut textures = Vec::with_capacity(texture_count.min(1024));
    for _ in 0..texture_count {
        textures.push(get_texture(decoder)?);
    }
    Ok(ComputePass {
        pipeline,
        buffers,
        dispatch: get_dispatch(decoder)?,
        textures,
    })
}

/// Decode one extended render pass's optional sections, driven by the feature
/// word its tag carried (`research/docs/23` §3.3).
///
/// Both extended tags walk the same sections in the same order — vertex input,
/// present, scissor, instancing, base vertex, depth, depth store, depth
/// identity, cull, blend — and differ only in the width of the word that says
/// which are present: [`PASS_KIND_RENDER_EXT`] hands its byte widened to this
/// `u16`, [`PASS_KIND_RENDER_EXT_WIDE`] the word itself. The two wide sections
/// are read where the encoder writes them, immediately after the depth block
/// and before culling.
///
/// A wide feature that narrows the depth attachment without a depth section is
/// refused rather than read as a frame with an attachment: the store action and
/// the identity describe a surface this pass never opens.
fn get_render_ext_pass(
    decoder: &mut Decoder<'_>,
    features: u16,
) -> Result<RenderPassDescriptor, CodecError> {
    let mut pass = get_render_pass(decoder, false)?;
    get_render_ext_sections(decoder, features, &mut pass)?;
    Ok(pass)
}

/// Decode the sampled-texture render pass of [`PASS_KIND_RENDER_SAMPLED`]
/// (`research/docs/23` §3.3, v70).
///
/// The layout repeats the wide tag's — the same `u16` feature word, then the
/// section walker below — with the texture block read immediately after the
/// word, where the encoder writes it: a `u8` count and that many full
/// [`TextureView`]s. A count above the contract's own cap is refused with
/// [`CodecError::RenderTextureCount`] before a single texture is read, so a
/// corrupt count cannot drive the decoder.
fn get_render_sampled_pass(
    decoder: &mut Decoder<'_>,
    features: u16,
) -> Result<RenderPassDescriptor, CodecError> {
    let texture_count = usize::from(decoder.u8()?);
    if texture_count > MAX_RENDER_TEXTURES {
        return Err(CodecError::RenderTextureCount {
            count: texture_count,
            maximum: MAX_RENDER_TEXTURES,
        });
    }
    let mut textures = Vec::with_capacity(texture_count);
    for _ in 0..texture_count {
        textures.push(get_texture(decoder)?);
    }
    let mut pass = get_render_pass(decoder, false)?;
    pass.textures = textures;
    get_render_ext_sections(decoder, features, &mut pass)?;
    Ok(pass)
}

/// Decode the optional sections an extended render pass's feature word names,
/// in the one order both extended tags write them.
fn get_render_ext_sections(
    decoder: &mut Decoder<'_>,
    features: u16,
    pass: &mut RenderPassDescriptor,
) -> Result<(), CodecError> {
    if features & u16::from(RENDER_FEATURE_VERTEX_INPUT) != 0 {
        get_vertex_input(decoder, pass)?;
    }
    if features & u16::from(RENDER_FEATURE_PRESENT) != 0 {
        pass.present = Some(get_present_descriptor(decoder)?);
    }
    if features & u16::from(RENDER_FEATURE_SCISSOR) != 0 {
        pass.scissor = Some([
            decoder.u32()?,
            decoder.u32()?,
            decoder.u32()?,
            decoder.u32()?,
        ]);
    }
    if features & u16::from(RENDER_FEATURE_INSTANCING) != 0 {
        pass.instance_count = decoder.u32()?;
    }
    if features & u16::from(RENDER_FEATURE_BASE_VERTEX) != 0 {
        pass.base_vertex = decoder.u32()?;
    }
    if features & u16::from(RENDER_FEATURE_DEPTH) != 0 {
        let (depth, test) = get_depth_block(decoder)?;
        pass.depth = Some(depth);
        pass.depth_test = test;
    }
    let depth_features =
        features & (RENDER_WIDE_FEATURE_DEPTH_STORE | RENDER_WIDE_FEATURE_DEPTH_RESOURCE);
    if depth_features != 0 && pass.depth.is_none() {
        return Err(CodecError::DepthFeatureWithoutAttachment(depth_features));
    }
    if features & RENDER_WIDE_FEATURE_DEPTH_STORE != 0 {
        let code = decoder.u8()?;
        let store = DepthStoreOp::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "depth store action",
            value: code,
        })?;
        // The presence check above proved the attachment is there; the
        // `ok_or` keeps this arm total rather than adding a second unwrap.
        let depth = pass
            .depth
            .as_mut()
            .ok_or(CodecError::DepthFeatureWithoutAttachment(depth_features))?;
        depth.store = Some(store);
    }
    if features & RENDER_WIDE_FEATURE_DEPTH_RESOURCE != 0 {
        let allocation_id = AllocationId::new(decoder.u64()?);
        let view_id = ViewId::new(decoder.u64()?);
        let depth = pass
            .depth
            .as_mut()
            .ok_or(CodecError::DepthFeatureWithoutAttachment(depth_features))?;
        depth.identity = Some(RenderDepthIdentity {
            allocation_id,
            view_id,
        });
    }
    if features & RENDER_WIDE_FEATURE_STENCIL != 0 {
        let (stencil, test) = get_stencil_block(decoder)?;
        pass.stencil = Some(stencil);
        pass.stencil_test = test;
    }
    let stencil_features =
        features & (RENDER_WIDE_FEATURE_STENCIL_STORE | RENDER_WIDE_FEATURE_STENCIL_RESOURCE);
    if stencil_features != 0 && pass.stencil.is_none() {
        return Err(CodecError::DepthFeatureWithoutAttachment(stencil_features));
    }
    if features & RENDER_WIDE_FEATURE_STENCIL_STORE != 0 {
        let code = decoder.u8()?;
        let store = match code {
            0 => StoreOp::DontCare,
            1 => StoreOp::Store,
            value => {
                return Err(CodecError::UnknownEnumValue {
                    field: "stencil store action",
                    value,
                })
            }
        };
        let stencil = pass
            .stencil
            .as_mut()
            .ok_or(CodecError::DepthFeatureWithoutAttachment(stencil_features))?;
        stencil.store = Some(store);
    }
    if features & RENDER_WIDE_FEATURE_STENCIL_RESOURCE != 0 {
        let allocation_id = AllocationId::new(decoder.u64()?);
        let view_id = ViewId::new(decoder.u64()?);
        let stencil = pass
            .stencil
            .as_mut()
            .ok_or(CodecError::DepthFeatureWithoutAttachment(stencil_features))?;
        stencil.identity = Some(RenderStencilIdentity {
            allocation_id,
            view_id,
        });
    }
    // The multisample section sits between the stencil sections and culling
    // (`research/docs/23` §3.3, v51): one sample count code, refused when it
    // names a count this version does not know.
    if features & RENDER_WIDE_FEATURE_MULTISAMPLE != 0 {
        let code = decoder.u8()?;
        let sample_count = SampleCount::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "multisample sample count",
            value: code,
        })?;
        pass.multisample = Some(MultisampleState { sample_count });
    }
    // The depth resolve section sits between the multisample section and
    // culling (`research/docs/23` §3.3, v57): one filter code, refused when it
    // names a filter this version does not know. The bit describes the depth
    // attachment's resolve, so a frame that sets it without a depth section
    // names a surface the pass never opens and is refused like the depth store
    // and identity bits are.
    if features & RENDER_WIDE_FEATURE_DEPTH_RESOLVE != 0 {
        if pass.depth.is_none() {
            return Err(CodecError::DepthFeatureWithoutAttachment(
                RENDER_WIDE_FEATURE_DEPTH_RESOLVE,
            ));
        }
        let code = decoder.u8()?;
        let filter = DepthResolveFilter::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "depth resolve filter",
            value: code,
        })?;
        pass.depth_resolve = Some(MultisampleDepthResolve { filter });
    }
    // The stencil resolve section follows the depth resolve section and
    // precedes culling (`research/docs/23` §3.3, v60): one filter code,
    // refused when it names a filter this version does not know. The bit
    // describes the stencil attachment's resolve, so a frame that sets it
    // without a stencil section names a surface the pass never opens and is
    // refused like the stencil store and resource bits are.
    if features & RENDER_WIDE_FEATURE_STENCIL_RESOLVE != 0 {
        if pass.stencil.is_none() {
            return Err(CodecError::DepthFeatureWithoutAttachment(
                RENDER_WIDE_FEATURE_STENCIL_RESOLVE,
            ));
        }
        let code = decoder.u8()?;
        let filter = StencilResolveFilter::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "stencil resolve filter",
            value: code,
        })?;
        pass.stencil_resolve = Some(MultisampleStencilResolve { filter });
    }
    if features & u16::from(RENDER_FEATURE_CULL) != 0 {
        let mode = CullMode::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
            field: "cull mode",
            value: 0,
        })?;
        let winding = Winding::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
            field: "front-facing winding",
            value: 0,
        })?;
        pass.cull = Some(RenderPassCull { mode, winding });
    }
    if features & u16::from(RENDER_FEATURE_BLEND) != 0 {
        let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
        if count > MAX_COLOR_ATTACHMENTS {
            return Err(CodecError::ColorAttachmentCount {
                count,
                maximum: MAX_COLOR_ATTACHMENTS,
            });
        }
        let mut attachments = Vec::with_capacity(count);
        for _ in 0..count {
            let source_rgb =
                BlendFactor::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "blend source rgb factor",
                    value: 0,
                })?;
            let destination_rgb =
                BlendFactor::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "blend destination rgb factor",
                    value: 0,
                })?;
            let source_alpha =
                BlendFactor::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "blend source alpha factor",
                    value: 0,
                })?;
            let destination_alpha =
                BlendFactor::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "blend destination alpha factor",
                    value: 0,
                })?;
            let operation =
                BlendOperation::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "blend operation",
                    value: 0,
                })?;
            attachments.push(BlendAttachment {
                source_rgb,
                destination_rgb,
                source_alpha,
                destination_alpha,
                operation,
            });
        }
        pass.blend = Some(RenderPassBlend { attachments });
    }
    Ok(())
}

fn get_render_pass(
    decoder: &mut Decoder<'_>,
    has_present: bool,
) -> Result<RenderPassDescriptor, CodecError> {
    let pipeline = PipelineId::new(decoder.u64()?);
    let attachment_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    if attachment_count > MAX_COLOR_ATTACHMENTS {
        return Err(CodecError::ColorAttachmentCount {
            count: attachment_count,
            maximum: MAX_COLOR_ATTACHMENTS,
        });
    }
    let mut color_attachments = Vec::with_capacity(attachment_count);
    for _ in 0..attachment_count {
        color_attachments.push(get_render_attachment(decoder)?);
    }
    let viewport = [
        decoder.u32()?,
        decoder.u32()?,
        decoder.u32()?,
        decoder.u32()?,
    ];
    let vertices = decoder.u32()?;
    let present = if has_present {
        Some(get_present_descriptor(decoder)?)
    } else {
        None
    };
    Ok(RenderPassDescriptor {
        blend: None,
        cull: None,
        // A frame without the wide multisample bit runs the single-sample
        // raster every pre-v51 frame ran (`research/docs/23` §3.3, v51).
        multisample: None,
        // A frame without the wide depth resolve bit states the API default
        // filter, exactly as a pass that never resolves does
        // (`research/docs/23` §3.3, v57).
        depth_resolve: None,
        // A frame without the wide stencil resolve bit states the API default
        // filter, exactly as a pass that never resolves does
        // (`research/docs/23` §3.3, v60).
        stencil_resolve: None,
        stencil: None,
        stencil_test: None,
        pipeline,
        color_attachments,
        viewport,
        scissor: None,
        vertices,
        vertex_buffers: Vec::new(),
        indices: None,
        // The single instance every pre-v31 frame drew and the zero offset
        // every pre-v34 frame read its indices through. A frame that carries
        // the matching feature bit overwrites each after the optional sections
        // are read (`research/docs/23` §3.3, v31/v34).
        instance_count: 1,
        base_vertex: 0,
        // A frame without the depth bit declares no depth attachment and no
        // depth state (`research/docs/23` §3.3, v36).
        depth: None,
        depth_test: None,
        // A frame that is not the sampled-texture tag binds no fragment
        // texture (`research/docs/23` §3.3, v70).
        textures: Vec::new(),
        present,
    })
}

/// Encode the depth block of an extended render pass: the rail-owned depth
/// attachment's shape and, when the pass declares one, its depth state.
///
/// The attachment travels as `(format, width, height, load)` rather than as a
/// resource identity, because the first depth increment's attachment is
/// rail-owned: nothing reads it back yet, and the readback channel is what
/// would add the identity fields (`research/docs/23` §3.3, v36). The load
/// operation's clear value is written as its IEEE-754 bits, so a frame is
/// byte-stable and a decoder reads exactly the value the encoder wrote.
fn put_depth_block(
    encoder: &mut Encoder,
    depth: &RenderDepthAttachment,
    test: Option<&DepthTest>,
) -> Result<(), CodecError> {
    encoder.u8(depth.format.code());
    encoder.u64(depth.width);
    encoder.u64(depth.height);
    match depth.load {
        DepthLoadOp::Clear(bits) => {
            encoder.u8(0);
            encoder.u32(bits);
        }
        DepthLoadOp::Load => encoder.u8(1),
    }
    match test {
        Some(test) => {
            encoder.u8(1);
            encoder.u8(test.compare.code());
            encoder.u8(u8::from(test.write));
        }
        None => encoder.u8(0),
    }
    Ok(())
}

/// Encode the stencil block of an extended render pass: the rail-owned stencil
/// attachment's shape and, when the pass declares one, its stencil state
/// (`research/docs/23` §3.3, v47).
///
/// The block is the depth block's shape one byte wide: format, extent and load
/// operation, then the pass's stencil state when it states one. The state's
/// read/write masks and reference are the byte-wide values both APIs state.
fn put_stencil_block(
    encoder: &mut Encoder,
    stencil: &RenderStencilAttachment,
    test: Option<&StencilTest>,
) {
    encoder.u8(stencil.format.code());
    encoder.u64(stencil.width);
    encoder.u64(stencil.height);
    match stencil.load {
        StencilLoadOp::Clear(value) => {
            encoder.u8(0);
            encoder.u8(value);
        }
        StencilLoadOp::Load => encoder.u8(1),
    }
    match test {
        Some(test) => {
            encoder.u8(1);
            encoder.u8(test.compare.code());
            encoder.u8(test.fail_op.code());
            encoder.u8(test.depth_fail_op.code());
            encoder.u8(test.pass_op.code());
            encoder.u8(test.read_mask);
            encoder.u8(test.write_mask);
            encoder.u8(test.reference);
        }
        None => encoder.u8(0),
    }
}

/// Decode the stencil block. An unknown format, comparison, operation or
/// presence byte is a typed refusal rather than a default.
fn get_stencil_block(
    decoder: &mut Decoder<'_>,
) -> Result<(RenderStencilAttachment, Option<StencilTest>), CodecError> {
    let format = StencilFormat::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
        field: "stencil format",
        value: 0,
    })?;
    let width = decoder.u64()?;
    let height = decoder.u64()?;
    let load = match decoder.u8()? {
        0 => StencilLoadOp::Clear(decoder.u8()?),
        1 => StencilLoadOp::Load,
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "stencil load op",
                value,
            })
        }
    };
    let test = match decoder.u8()? {
        0 => None,
        1 => {
            let compare =
                StencilCompare::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "stencil compare function",
                    value: 0,
                })?;
            let fail_op =
                StencilOp::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "stencil fail operation",
                    value: 0,
                })?;
            let depth_fail_op =
                StencilOp::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "stencil depth fail operation",
                    value: 0,
                })?;
            let pass_op =
                StencilOp::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "stencil pass operation",
                    value: 0,
                })?;
            let read_mask = decoder.u8()?;
            let write_mask = decoder.u8()?;
            let reference = decoder.u8()?;
            Some(StencilTest {
                compare,
                fail_op,
                depth_fail_op,
                pass_op,
                read_mask,
                write_mask,
                reference,
            })
        }
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "stencil state presence",
                value,
            })
        }
    };
    Ok((
        RenderStencilAttachment {
            format,
            width,
            height,
            load,
            // The base block never carries the v49 sections: the store action
            // and the identity travel under the wide tag's own bits.
            store: None,
            identity: None,
        },
        test,
    ))
}

/// Decode the depth block. An unknown format, compare function or presence byte
/// is a typed refusal rather than a default.
fn get_depth_block(
    decoder: &mut Decoder<'_>,
) -> Result<(RenderDepthAttachment, Option<DepthTest>), CodecError> {
    let format = DepthFormat::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
        field: "depth format",
        value: 0,
    })?;
    let width = decoder.u64()?;
    let height = decoder.u64()?;
    let load = match decoder.u8()? {
        0 => DepthLoadOp::Clear(decoder.u32()?),
        1 => DepthLoadOp::Load,
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "depth load op",
                value,
            })
        }
    };
    let test = match decoder.u8()? {
        0 => None,
        1 => {
            let compare =
                CompareFunction::from_code(decoder.u8()?).ok_or(CodecError::UnknownEnumValue {
                    field: "depth compare function",
                    value: 0,
                })?;
            let write = match decoder.u8()? {
                0 => false,
                1 => true,
                value => {
                    return Err(CodecError::UnknownEnumValue {
                        field: "depth write enable",
                        value,
                    })
                }
            };
            Some(DepthTest { compare, write })
        }
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "depth state presence",
                value,
            })
        }
    };
    Ok((
        RenderDepthAttachment {
            format,
            width,
            height,
            load,
            // The base block never carries the v43 sections: the store action
            // and the identity travel under the wide tag's own bits, so a frame
            // that states neither decodes to exactly the pre-v43 value.
            store: None,
            identity: None,
        },
        test,
    ))
}

/// Decode the vertex-input block of an extended render pass: the bound vertex
/// streams by their own view identity, then the index buffer when the draw is
/// indexed.
///
/// The counts are bounded by the contract's own caps before they can drive an
/// allocation, and an unknown index-width code is refused rather than defaulted
/// — the same rules the rest of the tagged payload follows.
fn get_vertex_input(
    decoder: &mut Decoder<'_>,
    pass: &mut RenderPassDescriptor,
) -> Result<(), CodecError> {
    let count = bounded_vertex_buffer_count(decoder.u64()?)?;
    let mut vertex_buffers = Vec::with_capacity(count);
    for _ in 0..count {
        vertex_buffers.push(get_view(decoder)?);
    }
    pass.vertex_buffers = vertex_buffers;
    pass.indices = match decoder.u8()? {
        0 => None,
        1 => Some(IndexBufferBinding {
            view: get_view(decoder)?,
            format: get_index_format(decoder)?,
        }),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "index buffer presence",
                value,
            })
        }
    };
    Ok(())
}

/// Decode one present action, the render pass's trailing action
/// (`research/docs/24` §3.5, shape one).
///
/// Every closed value is read through the same `from_code` inverse the Step 1
/// tests pinned, so an unknown mode, initial state or acquire policy is a
/// decoder refusal rather than a silent default. The sentinel's length is
/// bounded by [`MAX_PRESENT_SENTINEL_BYTES`] before it is read, so a corrupt
/// length cannot be mistaken for the remainder of the frame.
fn get_present_descriptor(decoder: &mut Decoder<'_>) -> Result<PresentDescriptor, CodecError> {
    let allocation_id = AllocationId::new(decoder.u64()?);
    let view_id = ViewId::new(decoder.u64()?);
    let format = get_attachment_format(decoder)?;
    let width = decoder.u64()?;
    let height = decoder.u64()?;
    let image_count = decoder.u32()?;
    let initial = match decoder.u8()? {
        0 => {
            let length =
                usize::try_from(decoder.u64()?).map_err(|_| CodecError::PresentSentinelLength {
                    length: usize::MAX,
                    maximum: MAX_PRESENT_SENTINEL_BYTES,
                })?;
            if length > MAX_PRESENT_SENTINEL_BYTES {
                return Err(CodecError::PresentSentinelLength {
                    length,
                    maximum: MAX_PRESENT_SENTINEL_BYTES,
                });
            }
            InitialState::Sentinel(decoder.take(length)?.to_vec())
        }
        1 => InitialState::Undefined,
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "present initial state",
                value,
            })
        }
    };
    let source = ViewId::new(decoder.u64()?);
    let mode_code = decoder.u8()?;
    let mode = PresentMode::from_code(mode_code).ok_or(CodecError::UnknownEnumValue {
        field: "present mode",
        value: mode_code,
    })?;
    let acquire_code = decoder.u8()?;
    let acquire = match acquire_code {
        0 => AcquirePolicy::Blocking,
        1 => AcquirePolicy::Timeout(decoder.u64()?),
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "present acquire policy",
                value,
            })
        }
    };
    Ok(PresentDescriptor {
        target: PresentTarget {
            allocation_id,
            view_id,
            format,
            width,
            height,
            image_count,
            initial,
        },
        source,
        mode,
        acquire,
    })
}

type HeapIcbTail = (
    Option<Box<HeapPayload>>,
    Option<Box<IndirectCommandPayload>>,
);

/// Decode the heap/ICB tail that follows the completion policy of a
/// [`SUBMIT_HEAP_ICB_REQUEST`] frame.
fn get_heap_icb_tail(decoder: &mut Decoder<'_>) -> Result<HeapIcbTail, CodecError> {
    let heap = if decoder.bool()? {
        Some(Box::new(get_heap_payload(decoder)?))
    } else {
        None
    };
    let indirect = if decoder.bool()? {
        Some(Box::new(get_indirect_payload(decoder)?))
    } else {
        None
    };
    Ok((heap, indirect))
}

/// Decode one heap payload: the descriptor followed by its placements.
fn get_heap_payload(decoder: &mut Decoder<'_>) -> Result<HeapPayload, CodecError> {
    let descriptor = HeapDescriptor {
        size: decoder.u64()?,
        storage_mode: get_storage_mode(decoder)?,
        allows_aliasing: decoder.bool()?,
    };
    let placement_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    if placement_count > MAX_HEAP_PLACEMENTS {
        return Err(CodecError::HeapPlacementCount {
            count: placement_count,
            maximum: MAX_HEAP_PLACEMENTS,
        });
    }
    let mut placements = Vec::with_capacity(placement_count);
    for _ in 0..placement_count {
        placements.push(get_heap_placement(decoder)?);
    }
    Ok(HeapPayload {
        descriptor,
        placements,
    })
}

/// Decode one heap placement: its heap identity, offset and resource.
fn get_heap_placement(decoder: &mut Decoder<'_>) -> Result<HeapPlacement, CodecError> {
    let heap_id = HeapId::new(decoder.u64()?);
    let offset = decoder.u64()?;
    let resource = get_heap_resource(decoder)?;
    Ok(HeapPlacement {
        heap_id,
        offset,
        resource,
    })
}

/// Decode the resource a heap placement puts at an offset. An unknown kind is a
/// decoder refusal, not a silent default.
fn get_heap_resource(decoder: &mut Decoder<'_>) -> Result<HeapResource, CodecError> {
    let kind = decoder.u8()?;
    let byte_size = decoder.u64()?;
    match kind {
        0 => Ok(HeapResource::Buffer { byte_size }),
        1 => Ok(HeapResource::Texture { byte_size }),
        value => Err(CodecError::UnknownEnumValue {
            field: "heap resource kind",
            value,
        }),
    }
}

/// Decode one indirect-command payload: the buffer descriptor, one command and
/// the replay range.
fn get_indirect_payload(decoder: &mut Decoder<'_>) -> Result<IndirectCommandPayload, CodecError> {
    let max_commands = decoder.u32()?;
    let kind_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    if kind_count > MAX_SUPPORTED_INDIRECT_COMMANDS {
        return Err(CodecError::IndirectCommandKindCount {
            count: kind_count,
            maximum: MAX_SUPPORTED_INDIRECT_COMMANDS,
        });
    }
    let mut kinds = Vec::with_capacity(kind_count);
    for _ in 0..kind_count {
        let code = decoder.u8()?;
        let kind = IndirectCommandKind::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "indirect command kind",
            value: code,
        })?;
        kinds.push(kind);
    }
    let command = get_indirect_command(decoder)?;
    let range = IndirectCommandRange {
        start: decoder.u32()?,
        count: decoder.u32()?,
    };
    Ok(IndirectCommandPayload {
        buffer: IndirectCommandBufferDescriptor {
            max_commands,
            kinds,
        },
        command,
        range,
    })
}

/// Decode one indirect command: the kind first, then the closed variant's
/// fields.
fn get_indirect_command(
    decoder: &mut Decoder<'_>,
) -> Result<IndirectCommandDescriptor, CodecError> {
    let code = decoder.u8()?;
    match IndirectCommandKind::from_code(code) {
        Some(IndirectCommandKind::Draw) => Ok(IndirectCommandDescriptor::Draw {
            vertex_count: decoder.u32()?,
            instance_count: decoder.u32()?,
        }),
        Some(IndirectCommandKind::DrawIndexed) => Ok(IndirectCommandDescriptor::DrawIndexed {
            index_count: decoder.u32()?,
            instance_count: decoder.u32()?,
        }),
        Some(IndirectCommandKind::Dispatch) => Ok(IndirectCommandDescriptor::Dispatch {
            threadgroups: [decoder.u32()?, decoder.u32()?, decoder.u32()?],
        }),
        None => Err(CodecError::UnknownEnumValue {
            field: "indirect command kind",
            value: code,
        }),
    }
}

fn get_render_attachment(decoder: &mut Decoder<'_>) -> Result<RenderAttachment, CodecError> {
    let view_id = ViewId::new(decoder.u64()?);
    let allocation_id = AllocationId::new(decoder.u64()?);
    let format = get_attachment_format(decoder)?;
    let width = decoder.u64()?;
    let height = decoder.u64()?;
    let load = get_load_op(decoder)?;
    let store = get_store_op(decoder)?;
    Ok(RenderAttachment {
        view_id,
        allocation_id,
        format,
        width,
        height,
        load,
        store,
    })
}

fn get_attachment_format(decoder: &mut Decoder<'_>) -> Result<AttachmentFormat, CodecError> {
    let code = decoder.u8()?;
    AttachmentFormat::from_code(code).ok_or(CodecError::UnknownEnumValue {
        field: "attachment format",
        value: code,
    })
}

fn get_load_op(decoder: &mut Decoder<'_>) -> Result<LoadOp, CodecError> {
    match decoder.u8()? {
        0 => {
            let mut bytes = [0_u8; ClearColor::BYTES];
            bytes.copy_from_slice(decoder.take(ClearColor::BYTES)?);
            Ok(LoadOp::Clear(ClearColor::new(bytes)))
        }
        1 => Ok(LoadOp::Load),
        2 => Ok(LoadOp::DontCare),
        3 => Ok(LoadOp::Resident),
        value => Err(CodecError::UnknownEnumValue {
            field: "attachment load op",
            value,
        }),
    }
}

fn get_store_op(decoder: &mut Decoder<'_>) -> Result<StoreOp, CodecError> {
    match decoder.u8()? {
        0 => Ok(StoreOp::Store),
        1 => Ok(StoreOp::DontCare),
        2 => Ok(StoreOp::Resident),
        value => Err(CodecError::UnknownEnumValue {
            field: "attachment store op",
            value,
        }),
    }
}

fn put_resources(encoder: &mut Encoder, resources: &ResourceTableSnapshot) {
    let allocations: Vec<_> = resources.allocations().collect();
    encoder.u64(allocations.len() as u64);
    for allocation in &allocations {
        encoder.u64(allocation.allocation_id.get());
        put_epoch(encoder, allocation.owner_epoch);
        encoder.u64(allocation.size);
    }
    let leases: Vec<_> = resources.leases().collect();
    encoder.u64(leases.len() as u64);
    for reservation in &leases {
        put_reservation(encoder, reservation);
    }
}

fn get_resources(decoder: &mut Decoder<'_>) -> Result<ResourceTableSnapshot, CodecError> {
    let mut resources = ResourceTableSnapshot::new();
    let allocation_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..allocation_count {
        resources.insert_allocation(AllocationRecord {
            allocation_id: AllocationId::new(decoder.u64()?),
            owner_epoch: get_epoch(decoder)?,
            size: decoder.u64()?,
        })?;
    }
    let lease_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..lease_count {
        resources.insert_lease(get_reservation(decoder)?)?;
    }
    Ok(resources)
}

fn put_reservation(encoder: &mut Encoder, reservation: &LeaseReservation) {
    encoder.u64(reservation.lease.lease_id.get());
    encoder.u64(reservation.lease.allocation_id.get());
    put_epoch(encoder, reservation.lease.owner_epoch);
    encoder.u64(reservation.offset);
    encoder.u64(reservation.length);
}

fn get_reservation(decoder: &mut Decoder<'_>) -> Result<LeaseReservation, CodecError> {
    Ok(LeaseReservation {
        lease: BufferLease {
            lease_id: LeaseId::new(decoder.u64()?),
            allocation_id: AllocationId::new(decoder.u64()?),
            owner_epoch: get_epoch(decoder)?,
        },
        offset: decoder.u64()?,
        length: decoder.u64()?,
    })
}

fn put_staged_lease(encoder: &mut Encoder, staged: &StagedLease) {
    put_reservation(encoder, &staged.reservation);
    encoder.blob(&staged.bytes);
}

fn get_staged_lease(decoder: &mut Decoder<'_>) -> Result<StagedLease, CodecError> {
    Ok(StagedLease::new(
        get_reservation(decoder)?,
        decoder.blob()?,
    )?)
}

fn put_health(encoder: &mut Encoder, health: ProviderHealth) {
    encoder.u8(match health {
        ProviderHealth::Usable => 0,
        ProviderHealth::DeviceLost => 1,
        ProviderHealth::Exhausted => 2,
    });
}

fn get_health(decoder: &mut Decoder<'_>) -> Result<ProviderHealth, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderHealth::Usable),
        1 => Ok(ProviderHealth::DeviceLost),
        2 => Ok(ProviderHealth::Exhausted),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider health",
            value,
        }),
    }
}

fn put_writeback(encoder: &mut Encoder, writeback: &BufferWriteback) {
    encoder.u64(writeback.view_id.get());
    encoder.u64(writeback.allocation_id.get());
    encoder.u64(writeback.offset);
    encoder.blob(&writeback.bytes);
}

fn get_writeback(decoder: &mut Decoder<'_>) -> Result<BufferWriteback, CodecError> {
    Ok(BufferWriteback {
        view_id: ViewId::new(decoder.u64()?),
        allocation_id: AllocationId::new(decoder.u64()?),
        offset: decoder.u64()?,
        bytes: decoder.blob()?,
    })
}

fn put_disposition(encoder: &mut Encoder, disposition: CompletionDisposition) {
    match disposition {
        CompletionDisposition::NotSubmitted => encoder.u8(0),
        CompletionDisposition::Submitted { token } => {
            encoder.u8(1);
            put_token(encoder, &token);
        }
        CompletionDisposition::CompletedVisible { token } => {
            encoder.u8(2);
            put_token(encoder, &token);
        }
        CompletionDisposition::Cancelled { token } => {
            encoder.u8(3);
            put_token(encoder, &token);
        }
        CompletionDisposition::TimedOut { token } => {
            encoder.u8(4);
            put_token(encoder, &token);
        }
        CompletionDisposition::Failed { token } => {
            encoder.u8(5);
            put_optional_token(encoder, token);
        }
        CompletionDisposition::DeviceLost { token } => {
            encoder.u8(6);
            put_optional_token(encoder, token);
        }
        CompletionDisposition::SubmittedUnknown { token } => {
            encoder.u8(7);
            put_optional_token(encoder, token);
        }
    }
}

fn put_optional_token(encoder: &mut Encoder, token: Option<CompletionToken>) {
    match token {
        Some(token) => {
            encoder.u8(1);
            put_token(encoder, &token);
        }
        None => encoder.u8(0),
    }
}

fn get_optional_token(decoder: &mut Decoder<'_>) -> Result<Option<CompletionToken>, CodecError> {
    match decoder.u8()? {
        0 => Ok(None),
        1 => Ok(Some(get_token(decoder)?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "optional completion token",
            value,
        }),
    }
}

fn get_disposition(decoder: &mut Decoder<'_>) -> Result<CompletionDisposition, CodecError> {
    Ok(match decoder.u8()? {
        0 => CompletionDisposition::NotSubmitted,
        1 => CompletionDisposition::Submitted {
            token: get_token(decoder)?,
        },
        2 => CompletionDisposition::CompletedVisible {
            token: get_token(decoder)?,
        },
        3 => CompletionDisposition::Cancelled {
            token: get_token(decoder)?,
        },
        4 => CompletionDisposition::TimedOut {
            token: get_token(decoder)?,
        },
        5 => CompletionDisposition::Failed {
            token: get_optional_token(decoder)?,
        },
        6 => CompletionDisposition::DeviceLost {
            token: get_optional_token(decoder)?,
        },
        7 => CompletionDisposition::SubmittedUnknown {
            token: get_optional_token(decoder)?,
        },
        value => {
            return Err(CodecError::UnknownEnumValue {
                field: "completion disposition",
                value,
            })
        }
    })
}

fn put_submission(encoder: &mut Encoder, submission: &ProviderSubmission) {
    put_disposition(encoder, submission.completion);
    encoder.u64(submission.writebacks.len() as u64);
    for writeback in &submission.writebacks {
        put_writeback(encoder, writeback);
    }
}

fn get_submission(decoder: &mut Decoder<'_>) -> Result<ProviderSubmission, CodecError> {
    let completion = get_disposition(decoder)?;
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut writebacks = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        writebacks.push(get_writeback(decoder)?);
    }
    Ok(ProviderSubmission {
        completion,
        writebacks,
    })
}

fn put_readback(encoder: &mut Encoder, readback: &CompletionReadback) {
    put_disposition(encoder, readback.completion);
    encoder.u64(readback.writebacks.len() as u64);
    for writeback in &readback.writebacks {
        put_writeback(encoder, writeback);
    }
}

fn get_readback(decoder: &mut Decoder<'_>) -> Result<CompletionReadback, CodecError> {
    let completion = get_disposition(decoder)?;
    let count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut writebacks = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        writebacks.push(get_writeback(decoder)?);
    }
    Ok(CompletionReadback {
        completion,
        writebacks,
    })
}

fn put_phase(encoder: &mut Encoder, phase: ProviderPhase) {
    encoder.u8(match phase {
        ProviderPhase::Resolve => 0,
        ProviderPhase::Compile => 1,
        ProviderPhase::Encode => 2,
        ProviderPhase::Submit => 3,
        ProviderPhase::Wait => 4,
        ProviderPhase::Readback => 5,
    });
}

fn get_phase(decoder: &mut Decoder<'_>) -> Result<ProviderPhase, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderPhase::Resolve),
        1 => Ok(ProviderPhase::Compile),
        2 => Ok(ProviderPhase::Encode),
        3 => Ok(ProviderPhase::Submit),
        4 => Ok(ProviderPhase::Wait),
        5 => Ok(ProviderPhase::Readback),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider phase",
            value,
        }),
    }
}

fn put_class(encoder: &mut Encoder, class: ProviderErrorClass) {
    encoder.u8(match class {
        ProviderErrorClass::Args => 0,
        ProviderErrorClass::Capability => 1,
        ProviderErrorClass::Resource => 2,
        ProviderErrorClass::Compile => 3,
        ProviderErrorClass::Execute => 4,
        ProviderErrorClass::DeviceLost => 5,
        ProviderErrorClass::Internal => 6,
    });
}

fn get_class(decoder: &mut Decoder<'_>) -> Result<ProviderErrorClass, CodecError> {
    match decoder.u8()? {
        0 => Ok(ProviderErrorClass::Args),
        1 => Ok(ProviderErrorClass::Capability),
        2 => Ok(ProviderErrorClass::Resource),
        3 => Ok(ProviderErrorClass::Compile),
        4 => Ok(ProviderErrorClass::Execute),
        5 => Ok(ProviderErrorClass::DeviceLost),
        6 => Ok(ProviderErrorClass::Internal),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider error class",
            value,
        }),
    }
}

fn put_retryability(encoder: &mut Encoder, retryability: Retryability) {
    encoder.u8(match retryability {
        Retryability::Never => 0,
        Retryability::RetrySameTrace => 1,
        Retryability::RetryAfterRecreate => 2,
        Retryability::Unknown => 3,
    });
}

fn get_retryability(decoder: &mut Decoder<'_>) -> Result<Retryability, CodecError> {
    match decoder.u8()? {
        0 => Ok(Retryability::Never),
        1 => Ok(Retryability::RetrySameTrace),
        2 => Ok(Retryability::RetryAfterRecreate),
        3 => Ok(Retryability::Unknown),
        value => Err(CodecError::UnknownEnumValue {
            field: "retryability",
            value,
        }),
    }
}

fn put_field_value(encoder: &mut Encoder, value: &FieldValue) {
    match value {
        FieldValue::Unsigned(value) => {
            encoder.u8(0);
            encoder.u64(*value);
        }
        FieldValue::Signed(value) => {
            encoder.u8(1);
            encoder.i64(*value);
        }
        FieldValue::Bool(value) => {
            encoder.u8(2);
            encoder.bool(*value);
        }
        FieldValue::Text(value) => {
            encoder.u8(3);
            encoder.text(value);
        }
    }
}

fn get_field_value(decoder: &mut Decoder<'_>) -> Result<FieldValue, CodecError> {
    match decoder.u8()? {
        0 => Ok(FieldValue::Unsigned(decoder.u64()?)),
        1 => Ok(FieldValue::Signed(decoder.i64()?)),
        2 => Ok(FieldValue::Bool(decoder.bool()?)),
        3 => Ok(FieldValue::Text(decoder.text()?)),
        value => Err(CodecError::UnknownEnumValue {
            field: "provider error field value",
            value,
        }),
    }
}

fn put_error(encoder: &mut Encoder, error: &ProviderError) {
    put_phase(encoder, error.phase);
    put_class(encoder, error.class);
    encoder.text(&error.slug);
    encoder.u64(error.fields.len() as u64);
    for (name, value) in &error.fields {
        encoder.text(name);
        put_field_value(encoder, value);
    }
    put_retryability(encoder, error.retryability);
    put_disposition(encoder, error.completion);
    encoder.opt_text(error.detail.as_deref());
}

fn get_error(decoder: &mut Decoder<'_>) -> Result<ProviderError, CodecError> {
    let phase = get_phase(decoder)?;
    let class = get_class(decoder)?;
    let slug = decoder.text()?;
    let mut error = ProviderError::new(phase, class, slug)?;
    let field_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    for _ in 0..field_count {
        error
            .fields
            .insert(decoder.text()?, get_field_value(decoder)?);
    }
    error.retryability = get_retryability(decoder)?;
    error.completion = get_disposition(decoder)?;
    error.detail = decoder.opt_text()?;
    Ok(error)
}

fn put_alias_mode(encoder: &mut Encoder, mode: AliasMode) {
    encoder.u8(match mode {
        AliasMode::Refused => 0,
        AliasMode::DistinctViews => 1,
        AliasMode::ExplicitPolicy => 2,
    });
}

fn get_alias_mode(decoder: &mut Decoder<'_>) -> Result<AliasMode, CodecError> {
    match decoder.u8()? {
        0 => Ok(AliasMode::Refused),
        1 => Ok(AliasMode::DistinctViews),
        2 => Ok(AliasMode::ExplicitPolicy),
        value => Err(CodecError::UnknownEnumValue {
            field: "alias mode",
            value,
        }),
    }
}

fn put_storage_mode(encoder: &mut Encoder, mode: StorageMode) {
    encoder.u8(match mode {
        StorageMode::OwnedBytes => 0,
        StorageMode::StagedLease => 1,
        StorageMode::BorrowedNoCopy => 2,
    });
}

fn get_storage_mode(decoder: &mut Decoder<'_>) -> Result<StorageMode, CodecError> {
    match decoder.u8()? {
        0 => Ok(StorageMode::OwnedBytes),
        1 => Ok(StorageMode::StagedLease),
        2 => Ok(StorageMode::BorrowedNoCopy),
        value => Err(CodecError::UnknownEnumValue {
            field: "storage mode",
            value,
        }),
    }
}

/// Encode a capability snapshot, including its render bits.
///
/// Only [`RENDER_CAPABILITIES_RESPONSE`] frames use this layout. A snapshot
/// whose render bits are all at their defaults is written by
/// [`put_capabilities_legacy`] under the legacy tag, which is what keeps both
/// current providers byte-identical on the wire.
fn put_capabilities(
    encoder: &mut Encoder,
    capabilities: &ProviderCapabilities,
) -> Result<(), CodecError> {
    put_capabilities_legacy(encoder, capabilities);
    if capabilities.supported_color_formats.len() > MAX_SUPPORTED_COLOR_FORMATS {
        return Err(CodecError::SupportedColorFormatCount {
            count: capabilities.supported_color_formats.len(),
            maximum: MAX_SUPPORTED_COLOR_FORMATS,
        });
    }
    encoder.bool(capabilities.supports_render_passes);
    encoder.u32(capabilities.max_color_attachments);
    encoder.u64(capabilities.max_attachment_dimension[0]);
    encoder.u64(capabilities.max_attachment_dimension[1]);
    encoder.u64(capabilities.supported_color_formats.len() as u64);
    for format in &capabilities.supported_color_formats {
        put_attachment_format(encoder, *format);
    }
    // The present bits (`research/docs/24` §4.2) travel in the same extended
    // payload as the render bits. A snapshot whose present bits all stay at
    // their defaults never reaches this function, because
    // `declares_render_support` treats them as part of the same question —
    // which is what keeps both current providers' frames at their previous
    // bytes while still carrying a future presenter's declaration.
    encoder.bool(capabilities.supports_presentation);
    encoder.u32(capabilities.max_present_targets);
    if capabilities.supported_present_modes.len() > MAX_SUPPORTED_PRESENT_MODES {
        return Err(CodecError::PresentModeCount {
            count: capabilities.supported_present_modes.len(),
            maximum: MAX_SUPPORTED_PRESENT_MODES,
        });
    }
    encoder.u64(capabilities.supported_present_modes.len() as u64);
    for mode in &capabilities.supported_present_modes {
        encoder.u8(mode.code());
    }
    encoder.u32(capabilities.max_present_image_count);
    // The heap and ICB bits (`research/docs/25-heaps与ICB设计.md` §4.1) are an
    // *optional tail* of the extended payload: a snapshot whose heap and ICB
    // bits all stay at their defaults keeps the exact bytes of the pre-heap
    // frame, so a provider that only declares render and/or present bits (both
    // current providers) is wire-identical to what it was before this
    // increment. The decoder reads the tail only while bytes remain, which is
    // the same additive rule the render and present bits follow: an old
    // decoder sees the frame it always saw, and a new tail is refused with a
    // typed error rather than silently truncated.
    // The vertex-input bits (`research/docs/23` §3.3) extend the same optional
    // tail. A snapshot that declares them without declaring heap or ICB support
    // still writes the tail's heap/ICB half at its defaults, so the decoder can
    // read the vertex block by position instead of guessing which optional
    // section the remaining bytes carry.
    if capabilities.declares_heap_support()
        || capabilities.declares_icb_support()
        || capabilities.declares_vertex_input_support()
        || capabilities.declares_instancing_support()
        // A snapshot that declares multisampling but none of the blocks before
        // it still has to write the heap/ICB half, because the decoder reads
        // that half by position before the multisample tag
        // (`research/docs/23` §3.3, v51). Leaving this out would drop the two
        // bits on the wire entirely — the exact "declared a bit that travels
        // as the old bytes" failure the comment above names.
        || capabilities.declares_multisample_support()
        // The depth-resolve block follows the same rule: a snapshot that
        // declares only the two depth-resolve bits still has to write the
        // heap/ICB half the decoder reads by position before the tag, and this
        // guard is what keeps the declaration from being silently dropped
        // (`research/docs/23` §3.3, v57).
        || capabilities.declares_depth_resolve_support()
        // The stencil-resolve block follows the same rule: a snapshot that
        // declares only the two stencil-resolve bits still has to write the
        // heap/ICB half the decoder reads by position before the tag, and this
        // guard is what keeps the declaration from being silently dropped
        // (`research/docs/23` §3.3, v60).
        || capabilities.declares_stencil_resolve_support()
        // The render-sampler block follows the same rule one more time: a
        // snapshot that declares only the three render-sampler bits still has
        // to write the heap/ICB half the decoder reads by position before the
        // tag, and this guard is what keeps the declaration from being
        // silently dropped (`research/docs/23` §3.3, v70).
        || capabilities.declares_render_texture_support()
    {
        encoder.bool(capabilities.supports_heaps);
        encoder.u64(capabilities.max_heap_bytes);
        if capabilities.supported_heap_storage_modes.len() > MAX_SUPPORTED_HEAP_STORAGE_MODES {
            return Err(CodecError::HeapStorageModeCount {
                count: capabilities.supported_heap_storage_modes.len(),
                maximum: MAX_SUPPORTED_HEAP_STORAGE_MODES,
            });
        }
        encoder.u64(capabilities.supported_heap_storage_modes.len() as u64);
        for mode in &capabilities.supported_heap_storage_modes {
            put_storage_mode(encoder, *mode);
        }
        encoder.bool(capabilities.supports_heap_aliasing);
        encoder.bool(capabilities.supports_indirect_command_buffers);
        encoder.u32(capabilities.max_indirect_commands);
        if capabilities.supported_indirect_commands.len() > MAX_SUPPORTED_INDIRECT_COMMANDS {
            return Err(CodecError::IndirectCommandKindCount {
                count: capabilities.supported_indirect_commands.len(),
                maximum: MAX_SUPPORTED_INDIRECT_COMMANDS,
            });
        }
        encoder.u64(capabilities.supported_indirect_commands.len() as u64);
        for kind in &capabilities.supported_indirect_commands {
            encoder.u8(kind.code());
        }
        if capabilities.declares_vertex_input_support() {
            encoder.u8(CAPABILITY_VERTEX_INPUT_TAIL);
            encoder.u32(capabilities.max_vertex_buffers);
            if capabilities.supported_vertex_formats.len() > MAX_SUPPORTED_VERTEX_FORMATS {
                return Err(CodecError::SupportedVertexFormatCount {
                    count: capabilities.supported_vertex_formats.len(),
                    maximum: MAX_SUPPORTED_VERTEX_FORMATS,
                });
            }
            encoder.u64(capabilities.supported_vertex_formats.len() as u64);
            for format in &capabilities.supported_vertex_formats {
                encoder.u8(format.code());
            }
            if capabilities.supported_index_formats.len() > MAX_SUPPORTED_INDEX_FORMATS {
                return Err(CodecError::SupportedIndexFormatCount {
                    count: capabilities.supported_index_formats.len(),
                    maximum: MAX_SUPPORTED_INDEX_FORMATS,
                });
            }
            encoder.u64(capabilities.supported_index_formats.len() as u64);
            for format in &capabilities.supported_index_formats {
                encoder.u8(format.code());
            }
        }
        // The instancing block is the tail's newest section and follows the
        // vertex-input half when the snapshot declares either of its two bits
        // (`research/docs/23` §3.3, v31).
        if capabilities.declares_instancing_support() {
            encoder.u8(CAPABILITY_INSTANCING_TAIL);
            encoder.bool(capabilities.supports_render_instancing);
            encoder.u32(capabilities.max_render_instances);
        }
        // The multisample block is the tail's newest section and follows the
        // instancing half when the snapshot declares either of its two bits
        // (`research/docs/23` §3.3, v51).
        if capabilities.declares_multisample_support() {
            encoder.u8(CAPABILITY_MULTISAMPLE_TAIL);
            encoder.bool(capabilities.supports_render_multisample);
            encoder.u32(capabilities.max_render_sample_count);
        }
        // The depth-resolve block is the tail's newest section and follows the
        // multisample half when the snapshot declares either of its two bits
        // (`research/docs/23` §3.3, v57).
        if capabilities.declares_depth_resolve_support() {
            encoder.u8(CAPABILITY_DEPTH_RESOLVE_TAIL);
            encoder.bool(capabilities.supports_render_depth_resolve);
            encoder.u32(capabilities.depth_resolve_modes);
        }
        // The stencil-resolve block is the tail's newest section and follows
        // the depth-resolve half when the snapshot declares either of its two
        // bits (`research/docs/23` §3.3, v60).
        if capabilities.declares_stencil_resolve_support() {
            encoder.u8(CAPABILITY_STENCIL_RESOLVE_TAIL);
            encoder.bool(capabilities.supports_render_stencil_resolve);
            encoder.u32(capabilities.stencil_resolve_modes);
        }
        // The render-sampler block is the tail's newest section and follows
        // the stencil-resolve half when the snapshot declares any of its three
        // bits (`research/docs/23` §3.3, v70).
        if capabilities.declares_render_texture_support() {
            encoder.u8(CAPABILITY_RENDER_TEXTURE_TAIL);
            encoder.bool(capabilities.supports_render_texture_sampling);
            encoder.u32(capabilities.max_render_textures);
            if capabilities.supported_render_texture_formats.len()
                > MAX_SUPPORTED_RENDER_TEXTURE_FORMATS
            {
                return Err(CodecError::RenderTextureFormatCount {
                    count: capabilities.supported_render_texture_formats.len(),
                    maximum: MAX_SUPPORTED_RENDER_TEXTURE_FORMATS,
                });
            }
            encoder.u64(capabilities.supported_render_texture_formats.len() as u64);
            for format in &capabilities.supported_render_texture_formats {
                put_texture_format(encoder, *format);
            }
        }
    }
    Ok(())
}

fn put_capabilities_legacy(encoder: &mut Encoder, capabilities: &ProviderCapabilities) {
    encoder.u32(capabilities.max_passes);
    encoder.bool(capabilities.supports_threads_exact);
    encoder.bool(capabilities.supports_threadgroups);
    encoder.bool(capabilities.supports_serial);
    encoder.bool(capabilities.supports_concurrent);
    encoder.array3(capabilities.max_local_size);
    encoder.u64(capabilities.max_invocations);
    encoder.array3(capabilities.max_group_count);
    encoder.u32(capabilities.max_storage_buffer_descriptors);
    encoder.u64(capabilities.max_buffer_range);
    encoder.u32(capabilities.max_push_constant_bytes);
    put_alias_mode(encoder, capabilities.alias_mode);
    encoder.u64(capabilities.storage_modes.len() as u64);
    for mode in &capabilities.storage_modes {
        put_storage_mode(encoder, *mode);
    }
    encoder.bool(capabilities.host_readback);
    encoder.bool(capabilities.submit_only);
}

/// Decode the pre-render capability payload and fill the render bits with the
/// defaults a provider that never declared them must be treated as having.
fn get_capabilities_legacy(decoder: &mut Decoder<'_>) -> Result<ProviderCapabilities, CodecError> {
    let max_passes = decoder.u32()?;
    let supports_threads_exact = decoder.bool()?;
    let supports_threadgroups = decoder.bool()?;
    let supports_serial = decoder.bool()?;
    let supports_concurrent = decoder.bool()?;
    let max_local_size = decoder.array3()?;
    let max_invocations = decoder.u64()?;
    let max_group_count = decoder.array3()?;
    let max_storage_buffer_descriptors = decoder.u32()?;
    let max_buffer_range = decoder.u64()?;
    let max_push_constant_bytes = decoder.u32()?;
    let alias_mode = get_alias_mode(decoder)?;
    let mode_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
        needed: usize::MAX,
        remaining: decoder.remaining(),
    })?;
    let mut storage_modes = Vec::with_capacity(mode_count.min(1024));
    for _ in 0..mode_count {
        storage_modes.push(get_storage_mode(decoder)?);
    }
    Ok(ProviderCapabilities {
        max_passes,
        supports_threads_exact,
        supports_threadgroups,
        supports_serial,
        supports_concurrent,
        max_local_size,
        max_invocations,
        max_group_count,
        max_storage_buffer_descriptors,
        max_buffer_range,
        max_push_constant_bytes,
        alias_mode,
        storage_modes,
        host_readback: decoder.bool()?,
        submit_only: decoder.bool()?,
        supports_render_passes: false,
        max_color_attachments: 0,
        max_attachment_dimension: [0, 0],
        supported_color_formats: Vec::new(),
        max_vertex_buffers: 0,
        supported_vertex_formats: Vec::new(),
        supported_index_formats: Vec::new(),
        // A legacy payload cannot have declared instancing either: both bits
        // take the "cannot instance" defaults, so a legacy provider is refused
        // a multi-instance pass instead of executing it once
        // (`research/docs/23` §3.3, v31).
        supports_render_instancing: false,
        max_render_instances: 0,
        // A legacy payload cannot have declared multisampling either: the two
        // bits take the "cannot multisample" defaults, so a legacy provider is
        // refused a multisampled pass instead of executing it as a
        // single-sample draw (`research/docs/23` §3.3, v51).
        supports_render_multisample: false,
        max_render_sample_count: 0,
        // A legacy payload cannot have declared the depth resolve either:
        // both bits take the "cannot resolve" defaults, so a legacy provider is
        // refused a resolving pass instead of executing it with a filter the
        // caller did not state (`research/docs/23` §3.3, v57).
        supports_render_depth_resolve: false,
        depth_resolve_modes: 0,
        // A legacy payload cannot have declared the stencil resolve either:
        // both bits take the "cannot resolve" defaults, so a legacy provider is
        // refused a resolving pass instead of executing it with a filter the
        // caller did not state (`research/docs/23` §3.3, v60).
        supports_render_stencil_resolve: false,
        stencil_resolve_modes: 0,
        // A legacy payload cannot have declared the render sampler either:
        // all three bits take the "cannot sample" defaults, so a legacy
        // provider is refused a texture-bearing pass instead of being
        // executed with a cleared sampling result the caller did not state
        // (`research/docs/23` §3.3, v70).
        supports_render_texture_sampling: false,
        max_render_textures: 0,
        supported_render_texture_formats: Vec::new(),
        // A legacy payload cannot have declared presentation, so the present
        // bits take the same "cannot present" defaults the render bits take
        // here (`docs/24` §4.2): a decoder that predates the present tag reads
        // a provider as present-refusing, which is exactly what that provider
        // was.
        supports_presentation: false,
        max_present_targets: 0,
        supported_present_modes: Vec::new(),
        max_present_image_count: 0,
        supports_heaps: false,
        max_heap_bytes: 0,
        supported_heap_storage_modes: Vec::new(),
        supports_heap_aliasing: false,
        supports_indirect_command_buffers: false,
        max_indirect_commands: 0,
        supported_indirect_commands: Vec::new(),
    })
}

fn get_capabilities(decoder: &mut Decoder<'_>) -> Result<ProviderCapabilities, CodecError> {
    let mut capabilities = get_capabilities_legacy(decoder)?;
    capabilities.supports_render_passes = decoder.bool()?;
    capabilities.max_color_attachments = decoder.u32()?;
    capabilities.max_attachment_dimension = [decoder.u64()?, decoder.u64()?];
    let format_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::TruncatedPayload {
            needed: usize::MAX,
            remaining: decoder.remaining(),
        })?;
    if format_count > MAX_SUPPORTED_COLOR_FORMATS {
        return Err(CodecError::SupportedColorFormatCount {
            count: format_count,
            maximum: MAX_SUPPORTED_COLOR_FORMATS,
        });
    }
    let mut supported_color_formats = Vec::with_capacity(format_count);
    for _ in 0..format_count {
        supported_color_formats.push(get_attachment_format(decoder)?);
    }
    capabilities.supported_color_formats = supported_color_formats;
    capabilities.supports_presentation = decoder.bool()?;
    capabilities.max_present_targets = decoder.u32()?;
    let mode_count = usize::try_from(decoder.u64()?).map_err(|_| CodecError::PresentModeCount {
        count: usize::MAX,
        maximum: MAX_SUPPORTED_PRESENT_MODES,
    })?;
    if mode_count > MAX_SUPPORTED_PRESENT_MODES {
        return Err(CodecError::PresentModeCount {
            count: mode_count,
            maximum: MAX_SUPPORTED_PRESENT_MODES,
        });
    }
    let mut supported_present_modes = Vec::with_capacity(mode_count);
    for _ in 0..mode_count {
        let code = decoder.u8()?;
        let mode = PresentMode::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "present mode",
            value: code,
        })?;
        supported_present_modes.push(mode);
    }
    capabilities.supported_present_modes = supported_present_modes;
    capabilities.max_present_image_count = decoder.u32()?;
    // The heap and ICB tail is optional: a frame that ended after the present
    // bits is a snapshot with no heap or ICB declaration, and its bits keep the
    // "cannot" defaults (`research/docs/25` §4.1). A frame with any bytes left
    // must carry the whole tail; a truncated one is a typed error.
    if decoder.remaining() == 0 {
        return Ok(capabilities);
    }
    capabilities.supports_heaps = decoder.bool()?;
    capabilities.max_heap_bytes = decoder.u64()?;
    let heap_mode_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::HeapStorageModeCount {
            count: usize::MAX,
            maximum: MAX_SUPPORTED_HEAP_STORAGE_MODES,
        })?;
    if heap_mode_count > MAX_SUPPORTED_HEAP_STORAGE_MODES {
        return Err(CodecError::HeapStorageModeCount {
            count: heap_mode_count,
            maximum: MAX_SUPPORTED_HEAP_STORAGE_MODES,
        });
    }
    let mut supported_heap_storage_modes = Vec::with_capacity(heap_mode_count);
    for _ in 0..heap_mode_count {
        supported_heap_storage_modes.push(get_storage_mode(decoder)?);
    }
    capabilities.supported_heap_storage_modes = supported_heap_storage_modes;
    capabilities.supports_heap_aliasing = decoder.bool()?;
    capabilities.supports_indirect_command_buffers = decoder.bool()?;
    capabilities.max_indirect_commands = decoder.u32()?;
    let icb_kind_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::IndirectCommandKindCount {
            count: usize::MAX,
            maximum: MAX_SUPPORTED_INDIRECT_COMMANDS,
        })?;
    if icb_kind_count > MAX_SUPPORTED_INDIRECT_COMMANDS {
        return Err(CodecError::IndirectCommandKindCount {
            count: icb_kind_count,
            maximum: MAX_SUPPORTED_INDIRECT_COMMANDS,
        });
    }
    let mut supported_indirect_commands = Vec::with_capacity(icb_kind_count);
    for _ in 0..icb_kind_count {
        let code = decoder.u8()?;
        let kind = IndirectCommandKind::from_code(code).ok_or(CodecError::UnknownEnumValue {
            field: "indirect command kind",
            value: code,
        })?;
        supported_indirect_commands.push(kind);
    }
    capabilities.supported_indirect_commands = supported_indirect_commands;
    // The vertex-input, instancing, multisample, depth-resolve and
    // stencil-resolve blocks are the tail's five optional sections
    // (`research/docs/23` §3.3, v31/v51/v57/v60): each is read only when
    // bytes remain, and each carries its own tag, so a snapshot that declares
    // stencil resolve without the four blocks before it writes its own tag
    // directly and one that declares none keeps the shorter frame. The walk
    // stays ordered — each section is only read where the encoder writes it —
    // rather than a tag-keyed loop, so a frame that reorders the sections is
    // refused instead of silently accepted.
    if decoder.remaining() == 0 {
        return Ok(capabilities);
    }
    // The walk stays ordered — each section is read exactly where the encoder
    // writes it — so the tag is the mutable cursor the optional blocks move
    // forward rather than a chain of shadows.
    let mut tag = decoder.u8()?;
    if tag == CAPABILITY_VERTEX_INPUT_TAIL {
        capabilities.max_vertex_buffers = decoder.u32()?;
        let vertex_format_count = usize::try_from(decoder.u64()?).map_err(|_| {
            CodecError::SupportedVertexFormatCount {
                count: usize::MAX,
                maximum: MAX_SUPPORTED_VERTEX_FORMATS,
            }
        })?;
        if vertex_format_count > MAX_SUPPORTED_VERTEX_FORMATS {
            return Err(CodecError::SupportedVertexFormatCount {
                count: vertex_format_count,
                maximum: MAX_SUPPORTED_VERTEX_FORMATS,
            });
        }
        let mut supported_vertex_formats = Vec::with_capacity(vertex_format_count);
        for _ in 0..vertex_format_count {
            supported_vertex_formats.push(get_vertex_format(decoder)?);
        }
        capabilities.supported_vertex_formats = supported_vertex_formats;
        let index_format_count =
            usize::try_from(decoder.u64()?).map_err(|_| CodecError::SupportedIndexFormatCount {
                count: usize::MAX,
                maximum: MAX_SUPPORTED_INDEX_FORMATS,
            })?;
        if index_format_count > MAX_SUPPORTED_INDEX_FORMATS {
            return Err(CodecError::SupportedIndexFormatCount {
                count: index_format_count,
                maximum: MAX_SUPPORTED_INDEX_FORMATS,
            });
        }
        let mut supported_index_formats = Vec::with_capacity(index_format_count);
        for _ in 0..index_format_count {
            supported_index_formats.push(get_index_format(decoder)?);
        }
        capabilities.supported_index_formats = supported_index_formats;
        if decoder.remaining() == 0 {
            return Ok(capabilities);
        }
        tag = decoder.u8()?;
    }
    if tag != CAPABILITY_INSTANCING_TAIL
        && tag != CAPABILITY_MULTISAMPLE_TAIL
        && tag != CAPABILITY_DEPTH_RESOLVE_TAIL
        && tag != CAPABILITY_STENCIL_RESOLVE_TAIL
        && tag != CAPABILITY_RENDER_TEXTURE_TAIL
    {
        return Err(CodecError::UnknownCapabilityTail(tag));
    }
    if tag == CAPABILITY_INSTANCING_TAIL {
        capabilities.supports_render_instancing = decoder.bool()?;
        capabilities.max_render_instances = decoder.u32()?;
        if decoder.remaining() == 0 {
            return Ok(capabilities);
        }
        tag = decoder.u8()?;
        if tag != CAPABILITY_MULTISAMPLE_TAIL
            && tag != CAPABILITY_DEPTH_RESOLVE_TAIL
            && tag != CAPABILITY_STENCIL_RESOLVE_TAIL
            && tag != CAPABILITY_RENDER_TEXTURE_TAIL
        {
            return Err(CodecError::UnknownCapabilityTail(tag));
        }
    }
    if tag == CAPABILITY_MULTISAMPLE_TAIL {
        capabilities.supports_render_multisample = decoder.bool()?;
        capabilities.max_render_sample_count = decoder.u32()?;
        if decoder.remaining() == 0 {
            return Ok(capabilities);
        }
        tag = decoder.u8()?;
        if tag != CAPABILITY_DEPTH_RESOLVE_TAIL
            && tag != CAPABILITY_STENCIL_RESOLVE_TAIL
            && tag != CAPABILITY_RENDER_TEXTURE_TAIL
        {
            return Err(CodecError::UnknownCapabilityTail(tag));
        }
    }
    // The depth-resolve block is the tail's fourth optional section
    // (`research/docs/23` §3.3, v57): it follows the multisample block when
    // present, and reads the bool plus the filter bitmask. A snapshot that
    // declares stencil resolve without depth resolve skips it.
    if tag == CAPABILITY_DEPTH_RESOLVE_TAIL {
        capabilities.supports_render_depth_resolve = decoder.bool()?;
        capabilities.depth_resolve_modes = decoder.u32()?;
        if decoder.remaining() == 0 {
            return Ok(capabilities);
        }
        tag = decoder.u8()?;
        if tag != CAPABILITY_STENCIL_RESOLVE_TAIL && tag != CAPABILITY_RENDER_TEXTURE_TAIL {
            return Err(CodecError::UnknownCapabilityTail(tag));
        }
    }
    // The stencil-resolve block is the tail's fifth optional section
    // (`research/docs/23` §3.3, v60): it follows the depth-resolve block when
    // present, and reads the bool plus the filter bitmask. The render-sampler
    // block may follow it, so the walk continues while bytes remain.
    if tag == CAPABILITY_STENCIL_RESOLVE_TAIL {
        capabilities.supports_render_stencil_resolve = decoder.bool()?;
        capabilities.stencil_resolve_modes = decoder.u32()?;
        if decoder.remaining() == 0 {
            return Ok(capabilities);
        }
        tag = decoder.u8()?;
    }
    // The render-sampler block is the tail's sixth and newest optional section
    // (`research/docs/23` §3.3, v70): it follows the stencil-resolve block when
    // present, and reads the bool, the binding cap and the admitted texture
    // formats.
    if tag != CAPABILITY_RENDER_TEXTURE_TAIL {
        return Err(CodecError::UnknownCapabilityTail(tag));
    }
    capabilities.supports_render_texture_sampling = decoder.bool()?;
    capabilities.max_render_textures = decoder.u32()?;
    let render_texture_format_count =
        usize::try_from(decoder.u64()?).map_err(|_| CodecError::RenderTextureFormatCount {
            count: usize::MAX,
            maximum: MAX_SUPPORTED_RENDER_TEXTURE_FORMATS,
        })?;
    if render_texture_format_count > MAX_SUPPORTED_RENDER_TEXTURE_FORMATS {
        return Err(CodecError::RenderTextureFormatCount {
            count: render_texture_format_count,
            maximum: MAX_SUPPORTED_RENDER_TEXTURE_FORMATS,
        });
    }
    let mut supported_render_texture_formats = Vec::with_capacity(render_texture_format_count);
    for _ in 0..render_texture_format_count {
        supported_render_texture_formats.push(get_texture_format(decoder)?);
    }
    capabilities.supported_render_texture_formats = supported_render_texture_formats;
    Ok(capabilities)
}
