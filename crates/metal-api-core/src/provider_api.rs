//! Experimental compute objects over the shared provider contract.
//!
//! Dispatch records freeze pipeline and view bindings. Buffer contents are read
//! once at commit, when the complete command is admitted and submitted once.
//! A provider that completes inside `submit` finishes the command at commit. A
//! provider that returns `Submitted` leaves the command pending; the first
//! `wait_until_completed` observes completion, validates `readback` against the
//! exact submitted trace and then lands writebacks. Buffer reservations are held
//! from commit through that boundary, so concurrent CPU access and commands
//! using those buffers wait for completion instead of racing the GPU. Dropping
//! a pending command releases its reservations and completion record without
//! claiming that unknown GPU work retired. Cancelling a pending command
//! releases the provider observation slot and the host reservations without
//! claiming device retirement. This module does not extend the older
//! [`crate::ComputeExecutor`] object API.

use crate::provider::{
    self as contract, AcquirePolicy, AllocationId, AllocationRecord, AttachmentFormat,
    BufferAccess, BufferRange, BufferSource, BufferWriteback, ClearColor, CompiledComputePipeline,
    CompletionDisposition, CompletionPolicy, CompletionToken, ComputeTrace, ContractError,
    Dispatch, DispatchKind, DispatchType, HeapDescriptor, HeapId, HeapPayload, HeapPlacement,
    HeapResource, IndexBufferBinding, IndexFormat, IndirectCommandBufferDescriptor,
    IndirectCommandDescriptor, IndirectCommandKind, IndirectCommandPayload, IndirectCommandRange,
    InitialState, LeaseId, LeaseReservation, LoadOp, OperationId, PipelineCompileRequest,
    PipelineId, PipelineProvider, PresentDescriptor, PresentMode, PresentTarget,
    ProviderCapabilities, ProviderError, ProviderHealth, ProviderSubmission, RenderAttachment,
    RenderPassDescriptor, RenderPipelineStage, ResourceTableSnapshot, SamplerPolicy,
    StageBufferView, StorageMode, StoreOp, ViewId, FULL_SCREEN_TRIANGLE_VERTICES,
    MAX_COLOR_ATTACHMENTS, MAX_PRESENT_IMAGE_COUNT, MAX_RENDER_SAMPLERS, MAX_RENDER_STAGE_BUFFERS,
    MAX_RENDER_STAGE_BUFFER_INDEX, MAX_RENDER_TEXTURE_INDEX, MAX_SERIAL_RESOURCES,
    MAX_VERTEX_BUFFERS, PROVIDER_SCHEMA_VERSION,
};
use crate::{ApiError, CommandBufferStatus, Size};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

/// The indirect command kinds the render encoder replays.
///
/// A named set rather than a single `expected` kind: one encoder replays both
/// direct draw shapes an ICB payload can hold (`research/docs/25` §4.3), so a
/// refusal that spelled only one of them would read as though the other were
/// inadmissible.
static RENDER_DRAW_KINDS: [IndirectCommandKind; 2] =
    [IndirectCommandKind::Draw, IndirectCommandKind::DrawIndexed];

/// The indirect command kinds the compute encoder replays from an ICB.
static COMPUTE_DISPATCH_KINDS: [IndirectCommandKind; 1] = [IndirectCommandKind::Dispatch];

/// Typed object, contract or provider failure. Provider fields remain intact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Api(ApiError),
    Contract(ContractError),
    Provider(ProviderError),
    ForeignBuffer,
    ForeignTexture,
    IdentityExhausted,
    InvalidPipelineMetadata,
    PassLimit {
        requested: usize,
        maximum: usize,
    },
    ProviderPanicked,
    CompletionUnavailable(CompletionDisposition),
    CompletionObservationMismatch,
    ForeignHeap,
    ForeignIndirectCommandBuffer,
    HeapPlacementDuplicate {
        allocation: AllocationId,
    },
    IndirectKindMismatch {
        /// Every command kind this encoder replays. The set, not one member of
        /// it: the render encoder replays `Draw` and `DrawIndexed` alike.
        accepted: &'static [IndirectCommandKind],
        actual: IndirectCommandKind,
    },
    IndirectAlreadyRecorded,
    IndirectDirectConflict,
    /// One binding index holds two views, so the pass's positional binding
    /// order would name two streams for one layout entry.
    VertexBufferAlreadyBound {
        index: u32,
    },
    /// A binding index is at or past [`MAX_VERTEX_BUFFERS`], which is the cap
    /// the pass's own descriptor carries.
    VertexBufferIndexOutOfRange {
        index: u32,
        maximum: usize,
    },
    /// An index buffer is already bound. One pass draws through one index
    /// buffer, exactly as its descriptor holds one `indices` binding.
    IndexBufferAlreadyBound,
    /// One texture binding index holds two textures, so the pass's binding list
    /// would name two sampled surfaces for one `[[texture(n)]]` argument
    /// (`research/docs/23` §3.3, v70/v104).
    FragmentTextureAlreadyBound {
        index: u32,
    },
    /// A texture binding index is at or past [`MAX_RENDER_TEXTURE_INDEX`],
    /// which is the fragment stage's own `[[texture(n)]]` bound
    /// (`research/docs/23` §3.3, v104).
    FragmentTextureIndexOutOfRange {
        index: u32,
        maximum: usize,
    },
    /// One runtime sampler index holds two states, so the pass's canonical
    /// sampler order would name two states for one `[[sampler(n)]]` argument
    /// (`research/docs/23` §3.3, v102).
    FragmentSamplerAlreadyBound {
        index: u32,
    },
    /// A runtime sampler index is at or past [`MAX_RENDER_SAMPLERS`], which is
    /// the Metal sampler argument table's own size.
    FragmentSamplerIndexOutOfRange {
        index: u32,
        maximum: usize,
    },
    /// One stage-buffer slot already holds a binding, so a second one would
    /// name two sources for one `[[buffer(N)]]` slot
    /// (`research/docs/23` §3.3, v87).
    StageBufferAlreadyBound {
        stage: RenderPipelineStage,
        index: u32,
    },
    /// A lease-bound slot the pipeline declares writable has no landing the
    /// object API could publish (`research/docs/23` §3.3, v86): the writeback
    /// channel lands in the bytes' own host image, and an imported lease has
    /// none — its bytes live in the provider's staged copy or in the owner's
    /// own mapping. A writable stage buffer is the caller-held arm's shape.
    WritableStageBufferLeaseUnsupported {
        stage: RenderPipelineStage,
        index: u32,
    },
    /// A lease-bound compute slot the pipeline declares writable has no
    /// landing the object API could publish (`research/docs/23` §90, E-TX9b).
    /// The refusal is the stage-buffer one's own reason: the writeback channel
    /// lands in the bytes' own host image, and an imported lease has none —
    /// its bytes live in the provider's staged copy or in the owner's own
    /// mapping. A writable slot is the caller-held arm's shape.
    WritableComputeLeaseUnsupported {
        index: u32,
    },
    /// One owner-window attachment names a window *no pass of the same command
    /// declares* (`research/docs/23` §114, E-TX9b).
    ///
    /// The window is resolved from the trace's *serial view list* — the
    /// declaration the attachment's own `(allocation, view)` identity already
    /// names — so a command whose declaring pass binds a different reservation
    /// (or none at all) states a landing the rail cannot resolve. The object
    /// rail refuses it here, where both halves are still visible, rather than
    /// one rail deeper where it would read as a generic missing declaration.
    WindowAttachmentUndeclared {
        lease: LeaseId,
        view: ViewId,
    },
    /// One owner-window attachment's window is not the attachment's own
    /// tightly packed byte extent (`research/docs/23` §114, E-TX8/E-TX9b).
    ///
    /// The frame the pass read back is exactly `width * height * bytes_per_texel`,
    /// and the window is where all of it lands; a window of any other length
    /// would have to be truncated or padded, which this arm refuses by name
    /// instead of doing silently.
    WindowAttachmentExtentMismatch {
        lease: LeaseId,
        expected: u64,
        declared: u64,
    },
    /// One owner-window attachment named a *copy* arm
    /// (`StageBufferLeaseArm::StagedLease`, `research/docs/23` §114/§90,
    /// E-TX9b). The store's window has to be the owner's own pages: a staged
    /// lease is the provider's copy of one reservation, and landing the frame
    /// there would write into a buffer no owner's ledger protects — exactly
    /// the shape the render rail refuses as `staged_lease`.
    WindowAttachmentNamesACopyArm {
        lease: LeaseId,
    },
    /// A vertex-buffer draw has no stream bound. The `vertex_id`-only shape is
    /// [`RenderCommandEncoder::draw_render_pass`], which binds no input at all.
    MissingVertexBuffer,
    /// A trace-view texture has no host bytes to read back
    /// (`research/docs/23` §110, E-TX3): its texels exist only on the trace
    /// that produces them, so the observation is the frame the sampling pass
    /// lands rather than a [`Texture::read`] of the handle.
    TraceViewTextureHasNoHostBytes {
        view: ViewId,
    },
    /// A trace-view texture was bound to a compute encoder
    /// (`research/docs/23` §110, E-TX3): the arm is the render sampler's, and
    /// this increment states no compute-side resolution of the trace's own
    /// production. The refusal happens here rather than one rail deeper, where
    /// it would read as "this source needs an owned byte copy".
    TraceViewTextureIsNotAComputeBinding {
        view: ViewId,
    },
    /// An indexed draw has no index buffer bound, so nothing says which indices
    /// it selects through.
    MissingIndexBuffer,
    /// An indirect replay supplies its own draw input, but the encoder bound
    /// streams, an index buffer, a sampled texture or a runtime sampler state
    /// for a direct draw.
    IndirectReplayInputConflict {
        vertex_buffers: usize,
        index_buffer: bool,
        fragment_textures: usize,
        fragment_samplers: usize,
    },
    /// A multi-attachment draw recorded no colour attachments, so the pass it
    /// would become has no target for any fragment output to land in — and no
    /// stored depth attachment to land its texels in either
    /// (`research/docs/23` §3.3, v46: the depth-only recording is the shape
    /// this refusal admits).
    EmptyRenderAttachmentList,
    /// A multi-attachment draw recorded more than
    /// [`MAX_COLOR_ATTACHMENTS`], the cap the pass's own descriptor carries.
    RenderAttachmentLimitExceeded {
        requested: usize,
        maximum: usize,
    },
    /// The same attachment — one `(allocation, view)` identity — appears twice
    /// in one recorded draw, so two locations would name one target.
    DuplicateRenderAttachment,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Api(error) => error.fmt(f),
            Self::Contract(error) => error.fmt(f),
            Self::Provider(error) => write!(
                f,
                "provider {:?}/{}: {:?}",
                error.phase, error.slug, error.detail
            ),
            Self::ForeignBuffer => f.write_str("buffer belongs to a different object device"),
            Self::ForeignTexture => f.write_str("texture belongs to a different object device"),
            Self::IdentityExhausted => f.write_str("object identity space exhausted"),
            Self::InvalidPipelineMetadata => {
                f.write_str("provider returned inconsistent pipeline metadata")
            }
            Self::PassLimit { requested, maximum } => {
                write!(f, "command has {requested} passes; maximum is {maximum}")
            }
            Self::ProviderPanicked => f.write_str("provider panicked"),
            Self::CompletionUnavailable(disposition) => write!(
                f,
                "provider completion did not produce host-visible results: {disposition:?}"
            ),
            Self::CompletionObservationMismatch => {
                f.write_str("provider wait disagrees with submitted completion")
            }
            Self::ForeignHeap => f.write_str("heap belongs to a different object device"),
            Self::ForeignIndirectCommandBuffer => {
                f.write_str("indirect command buffer belongs to a different object device")
            }
            Self::HeapPlacementDuplicate { allocation } => write!(
                f,
                "allocation {} is placed in the heap more than once",
                allocation.get()
            ),
            Self::IndirectKindMismatch { accepted, actual } => write!(
                f,
                "the indirect buffer holds {actual:?} commands; this encoder replays {accepted:?}"
            ),
            Self::IndirectAlreadyRecorded => {
                f.write_str("command buffer already carries an indirect command buffer")
            }
            Self::IndirectDirectConflict => {
                f.write_str("one encoder cannot mix direct and indirect dispatch or draw")
            }
            Self::VertexBufferAlreadyBound { index } => {
                write!(f, "vertex buffer binding {index} is bound twice")
            }
            Self::VertexBufferIndexOutOfRange { index, maximum } => write!(
                f,
                "vertex buffer binding {index} is past the {maximum}-stream limit"
            ),
            Self::IndexBufferAlreadyBound => {
                f.write_str("an index buffer is already bound on this encoder")
            }
            Self::TraceViewTextureHasNoHostBytes { view } => write!(
                f,
                "texture view {view:?} is the trace's own production: its texels have no host \
                 copy to read back"
            ),
            Self::TraceViewTextureIsNotAComputeBinding { view } => write!(
                f,
                "texture view {view:?} is the trace's own production: the arm is the render \
                 sampler's, not a compute binding"
            ),
            Self::FragmentTextureAlreadyBound { index } => {
                write!(f, "fragment texture binding {index} is bound twice")
            }
            Self::FragmentTextureIndexOutOfRange { index, maximum } => write!(
                f,
                "fragment texture binding {index} is past the {maximum}-texture limit"
            ),
            Self::FragmentSamplerAlreadyBound { index } => {
                write!(f, "runtime sampler {index} is bound twice")
            }
            Self::FragmentSamplerIndexOutOfRange { index, maximum } => write!(
                f,
                "runtime sampler binding {index} is past the {maximum}-sampler limit"
            ),
            Self::StageBufferAlreadyBound { stage, index } => {
                write!(f, "stage buffer {}/{} is bound twice", stage.name(), index)
            }
            Self::WritableStageBufferLeaseUnsupported { stage, index } => write!(
                f,
                "the pipeline declares stage buffer {}/{} writable, and an imported lease has \
                 no host landing the object API could publish",
                stage.name(),
                index
            ),
            Self::WritableComputeLeaseUnsupported { index } => write!(
                f,
                "the pipeline declares compute binding {index} writable, and an imported lease \
                 has no host landing the object API could publish"
            ),
            Self::WindowAttachmentUndeclared { lease, view } => write!(
                f,
                "the owner-window attachment lands in lease {} (view {}), and no pass of \
                 this command declares that window",
                lease.get(),
                view.get()
            ),
            Self::WindowAttachmentExtentMismatch {
                lease,
                expected,
                declared,
            } => write!(
                f,
                "the owner-window attachment's window (lease {}) carries {declared} bytes, \
                 and the attachment's own tightly packed extent is {expected}",
                lease.get()
            ),
            Self::WindowAttachmentNamesACopyArm { lease } => write!(
                f,
                "the owner-window attachment names lease {} through a copy arm; a stored \
                 frame lands in the owner's own registered window",
                lease.get()
            ),
            Self::MissingVertexBuffer => f.write_str(
                "a vertex-buffer draw needs a bound vertex stream; the vertex_id-only shape is \
                 draw_render_pass",
            ),
            Self::MissingIndexBuffer => {
                f.write_str("an indexed draw needs an index buffer bound on this encoder")
            }
            Self::IndirectReplayInputConflict {
                vertex_buffers,
                index_buffer,
                fragment_textures,
                fragment_samplers,
            } => {
                write!(
                    f,
                    "an indirect replay supplies its own draw input, but the encoder binds \
                     {vertex_buffers} vertex buffer(s)"
                )?;
                if *index_buffer {
                    f.write_str(" and an index buffer")?;
                }
                if *fragment_textures != 0 {
                    write!(f, " and {fragment_textures} sampled texture(s)")?;
                }
                if *fragment_samplers != 0 {
                    write!(f, " and {fragment_samplers} runtime sampler state(s)")?;
                }
                Ok(())
            }
            Self::EmptyRenderAttachmentList => {
                f.write_str("a draw needs at least one colour attachment")
            }
            Self::RenderAttachmentLimitExceeded { requested, maximum } => {
                write!(
                    f,
                    "a draw records {requested} colour attachments; maximum is {maximum}"
                )
            }
            Self::DuplicateRenderAttachment => {
                f.write_str("one colour attachment (allocation and view) is recorded twice")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Api(error) => Some(error),
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ApiError> for Error {
    fn from(error: ApiError) -> Self {
        Self::Api(error)
    }
}
impl From<ContractError> for Error {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}
impl From<ProviderError> for Error {
    fn from(error: ProviderError) -> Self {
        Self::Provider(error)
    }
}

fn next_id() -> Result<u64, Error> {
    NEXT_OBJECT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| Error::IdentityExhausted)
}

fn lock<'a, T>(mutex: &'a Mutex<T>, owner: &'static str) -> Result<MutexGuard<'a, T>, Error> {
    mutex
        .lock()
        .map_err(|_| ApiError::StatePoisoned(owner).into())
}

fn provider_call<T>(call: impl FnOnce() -> Result<T, ProviderError>) -> Result<T, Error> {
    catch_unwind(AssertUnwindSafe(call))
        .map_err(|_| Error::ProviderPanicked)?
        .map_err(Error::Provider)
}

struct DeviceState {
    provider: Arc<dyn PipelineProvider>,
    epoch: contract::DeviceEpoch,
    capabilities: ProviderCapabilities,
}

/// One object namespace. Clones share ownership; separately wrapping the same
/// provider creates a different namespace and does not permit object mixing.
#[derive(Clone)]
pub struct Device {
    state: Arc<DeviceState>,
}

impl Device {
    pub fn new(provider: Arc<dyn PipelineProvider>) -> Self {
        Self {
            state: Arc::new(DeviceState {
                epoch: provider.device_epoch(),
                capabilities: provider.capabilities(),
                provider,
            }),
        }
    }

    /// Current provider health. A caller must recreate the device/provider
    /// pair when this is not [`ProviderHealth::Usable`]; tokens already
    /// observed as terminal remain observable per the provider contract.
    pub fn health(&self) -> ProviderHealth {
        self.state.provider.health()
    }

    pub fn compile_pipeline(&self, request: PipelineCompileRequest) -> Result<Pipeline, Error> {
        request.validate()?;
        let expected = (
            request.entry_name.clone(),
            request.logical_digest.clone(),
            request.source.kind(),
        );
        let metadata = provider_call(|| self.state.provider.compile(request))?;
        // The owner is created before validating metadata so invalid returned
        // registrations also receive best-effort retirement.
        let pipeline = Pipeline {
            inner: Arc::new(PipelineInner {
                owner: Arc::clone(&self.state),
                metadata,
            }),
        };
        let metadata = pipeline.metadata();
        if metadata.device_epoch != self.state.epoch
            || metadata.device_epoch.is_zero()
            || metadata.pipeline_id.is_zero()
            || metadata.function.entry_name != expected.0
            || metadata.function.logical_digest != expected.1
            || metadata.function.source != expected.2
        {
            return Err(Error::InvalidPipelineMetadata);
        }
        metadata.function.validate()?;
        metadata.contract.validate()?;
        Ok(pipeline)
    }

    /// Wrap a provider-registered render pipeline so a [`RenderCommandEncoder`]
    /// can name it. The render rail is a concrete-context entry point
    /// (`VulkanComputeProvider::register_render_pipeline` is not part of
    /// [`PipelineProvider`]), so the caller hands the returned table metadata
    /// back here instead of asking this device to compile a render pipeline.
    ///
    /// The handle owns nothing on the provider side: the caller keeps the
    /// registration alive for the command buffers that name it and releases it
    /// through the concrete context when it is no longer needed. A metadata
    /// value without a render half, from another device epoch, or with a zero
    /// identity is refused here rather than surfacing as a provider registry
    /// mismatch later.
    pub fn render_pipeline(
        &self,
        metadata: &CompiledComputePipeline,
    ) -> Result<RenderPipeline, Error> {
        if metadata.device_epoch != self.state.epoch || metadata.device_epoch.is_zero() {
            return Err(Error::InvalidPipelineMetadata);
        }
        if metadata.pipeline_id.is_zero() {
            return Err(Error::InvalidPipelineMetadata);
        }
        let render = metadata
            .render
            .as_ref()
            .ok_or(Error::InvalidPipelineMetadata)?;
        render.validate()?;
        Ok(RenderPipeline {
            inner: Arc::new(RenderPipelineInner {
                owner: Arc::clone(&self.state),
                metadata: metadata.clone(),
            }),
        })
    }

    pub fn new_buffer_with_bytes(&self, bytes: Vec<u8>) -> Result<Buffer, Error> {
        if bytes.is_empty() {
            return Err(ApiError::EmptyBuffer.into());
        }
        let length = bytes.len();
        Ok(Buffer {
            inner: Arc::new(BufferInner {
                owner: Arc::clone(&self.state),
                allocation_id: AllocationId::new(next_id()?),
                length,
                bytes: Mutex::new(bytes),
                reservations: Mutex::new(Vec::new()),
                available: Condvar::new(),
            }),
        })
    }

    /// Declare one sampled texture with its initial contents. Only the first
    /// increment's shape is accepted here (`research/docs/16` §4.7): a 2D,
    /// single-sample `R32Uint` texture whose byte length matches the tightly
    /// packed extent.
    pub fn new_texture_with_bytes(
        &self,
        format: contract::TextureFormat,
        width: u64,
        height: u64,
        bytes: Vec<u8>,
    ) -> Result<Texture, Error> {
        self.new_texture(
            contract::TextureAccess::Sampled,
            format,
            width,
            height,
            bytes,
        )
    }

    /// Declare one storage texture — the object a kernel's
    /// `texture2d<float, access::write|read_write>` argument binds — with its
    /// initial contents. The shape validation is the sampled constructor's;
    /// what changes is the access the handle's views carry and the fact that
    /// the bytes are a landing rather than a read-only source
    /// (`research/docs/26` §21.4, C2): when a command that names this texture
    /// completes, its whole tightly packed extent is written back into the
    /// handle, and [`Texture::read`] observes those bytes.
    pub fn new_storage_texture_with_bytes(
        &self,
        format: contract::TextureFormat,
        width: u64,
        height: u64,
        bytes: Vec<u8>,
    ) -> Result<Texture, Error> {
        self.new_texture(
            contract::TextureAccess::Storage,
            format,
            width,
            height,
            bytes,
        )
    }

    /// Declare one texel-fetch texture — the object a fragment stage's
    /// `texture2d<T, access::read>` argument binds — with its initial contents
    /// (`research/docs/23` §3.3, v105).
    ///
    /// The handle is read-only exactly as the sampled one is, so what changes
    /// is the access every view carries: `Fetched` is the arm the pass's
    /// declaration pairs with, and the Vulkan rail binds the image alone for it
    /// — the module's own `OpImageFetch` reads the texels and no `SAMPLER` slot
    /// is declared. A sampler-free binding is never a landing, so this
    /// constructor's bytes stay its declared initial contents.
    pub fn new_fetched_texture_with_bytes(
        &self,
        format: contract::TextureFormat,
        width: u64,
        height: u64,
        bytes: Vec<u8>,
    ) -> Result<Texture, Error> {
        self.new_texture(
            contract::TextureAccess::Fetched,
            format,
            width,
            height,
            bytes,
        )
    }

    /// Declare one sampled texture whose texels are the trace's own GPU
    /// output (`research/docs/23` §110, E-TX3).
    ///
    /// `view` is the *producing* view: the recorded pass that renders into it
    /// with [`StoreOp::Store`] is the pass whose
    /// bytes every later sampled binding of this handle reads. The handle
    /// therefore shares that view's identity — its allocation and its view id
    /// — and carries no bytes of its own, exactly as the trace rail's
    /// `TextureSource::TraceView` arm states. The producer has to be recorded
    /// *before* the pass that samples the handle, because the arm is defined
    /// by the trace's own order: `ComputeTrace::validate_serial_buffer_reuse`
    /// refuses an unwritten identity, a store that follows the read, and a
    /// declaration that restates another shape.
    ///
    /// `format`, `width` and `height` are the sampled declaration's own
    /// restatement of the stored surface, and they have to match it, exactly
    /// as the trace rail states: `view`'s byte length is held to the tightly
    /// packed extent `width * height * format` here, and core admission holds
    /// the rest to the attachment's declaration.
    pub fn new_trace_view_texture(
        &self,
        view: &BufferView,
        format: contract::TextureFormat,
        width: u64,
        height: u64,
    ) -> Result<Texture, Error> {
        if !Arc::ptr_eq(&self.state, &view.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        if width == 0 || height == 0 {
            return Err(ApiError::ZeroSize.into());
        }
        let expected = width
            .checked_mul(height)
            .and_then(|extent| extent.checked_mul(format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("texture extent"))?;
        if view.length as u64 != expected {
            return Err(ContractError::SourceLengthMismatch {
                view: view.view_id,
                expected,
                actual: view.length as u64,
            }
            .into());
        }
        Ok(Texture {
            inner: Arc::new(TextureInner {
                owner: Arc::clone(&self.state),
                allocation_id: view.allocation_id(),
                view_id: view.view_id(),
                format,
                width,
                height,
                length: view.length,
                access: contract::TextureAccess::Sampled,
                origin: TextureOrigin::TraceView,
                bytes: Mutex::new(Vec::new()),
                reservations: Mutex::new(Vec::new()),
                available: Condvar::new(),
            }),
        })
    }

    fn new_texture(
        &self,
        access: contract::TextureAccess,
        format: contract::TextureFormat,
        width: u64,
        height: u64,
        bytes: Vec<u8>,
    ) -> Result<Texture, Error> {
        if width == 0 || height == 0 {
            return Err(ApiError::ZeroSize.into());
        }
        if bytes.is_empty() {
            return Err(ApiError::EmptyBuffer.into());
        }
        let expected = width
            .checked_mul(height)
            .and_then(|extent| extent.checked_mul(format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("texture extent"))?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected {
            return Err(ContractError::SourceLengthMismatch {
                view: ViewId::new(u64::MAX),
                expected,
                actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            }
            .into());
        }
        Ok(Texture {
            inner: Arc::new(TextureInner {
                owner: Arc::clone(&self.state),
                allocation_id: AllocationId::new(next_id()?),
                view_id: ViewId::new(next_id()?),
                format,
                width,
                height,
                length: bytes.len(),
                access,
                origin: TextureOrigin::DeclaredBytes,
                bytes: Mutex::new(bytes),
                reservations: Mutex::new(Vec::new()),
                available: Condvar::new(),
            }),
        })
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue {
            owner: Arc::clone(&self.state),
        }
    }

    /// Declare one heap (`research/docs/25` §6 Step 6). The first increment is
    /// fixed-size, and the aliasing flag is a declaration rather than a
    /// capability: `allows_aliasing = true` is well formed here, but whether
    /// the snapshot can execute it is answered at
    /// [`CommandBuffer::commit`] admission, so a provider whose
    /// `supports_heap_aliasing` stays `false` still refuses the commit
    /// fail closed.
    pub fn new_heap(
        &self,
        size: u64,
        storage_mode: StorageMode,
        allows_aliasing: bool,
    ) -> Result<Heap, Error> {
        let descriptor = HeapDescriptor {
            size,
            storage_mode,
            allows_aliasing,
        };
        descriptor.validate()?;
        Ok(Heap {
            inner: Arc::new(HeapInner {
                owner: Arc::clone(&self.state),
                heap_id: HeapId::new(next_id()?),
                descriptor,
                placements: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Declare one indirect command buffer (`research/docs/25` §6 Step 6).
    ///
    /// `kind` is the explicit statement of what the single command replays; it
    /// has to agree with `command`'s own kind and appear in `kinds`, so the
    /// caller cannot split the command's shape across two places. Structural
    /// validation runs here; capability refusals stay with admission.
    pub fn new_indirect_command_buffer(
        &self,
        kind: IndirectCommandKind,
        max_commands: u32,
        kinds: Vec<IndirectCommandKind>,
        range: IndirectCommandRange,
        command: IndirectCommandDescriptor,
    ) -> Result<IndirectCommandBuffer, Error> {
        if command.kind() != kind {
            return Err(ContractError::IcbCommandKindUnsupported(command.kind()).into());
        }
        let payload = IndirectCommandPayload {
            buffer: IndirectCommandBufferDescriptor {
                max_commands,
                kinds,
            },
            command,
            range,
        };
        payload.validate()?;
        Ok(IndirectCommandBuffer {
            inner: Arc::new(IndirectCommandBufferInner {
                owner: Arc::clone(&self.state),
                payload,
            }),
        })
    }
}

struct PipelineInner {
    owner: Arc<DeviceState>,
    metadata: CompiledComputePipeline,
}
impl Drop for PipelineInner {
    fn drop(&mut self) {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            self.owner.provider.release_pipeline(&self.metadata)
        }));
    }
}

/// Registered pipeline retained by every recorded dispatch that uses it.
#[derive(Clone)]
pub struct Pipeline {
    inner: Arc<PipelineInner>,
}
impl Pipeline {
    pub fn metadata(&self) -> &CompiledComputePipeline {
        &self.inner.metadata
    }
}

struct RenderPipelineInner {
    owner: Arc<DeviceState>,
    metadata: CompiledComputePipeline,
}

/// A provider-registered render pipeline a [`RenderCommandEncoder`] names.
///
/// This is the render sibling of [`Pipeline`]: it carries the table entry a
/// render pass has to reference and pins it to one device, but it does not own
/// the provider registration. The concrete context that registered it owns
/// retirement (`VulkanComputeProvider::release_render_pipeline`), exactly as
/// the compute rail's render plan does for the trace path.
#[derive(Clone)]
pub struct RenderPipeline {
    inner: Arc<RenderPipelineInner>,
}
impl RenderPipeline {
    pub fn metadata(&self) -> &CompiledComputePipeline {
        &self.inner.metadata
    }
}

/// What a present action pre-seeds its target with (`research/docs/24` §3.1).
///
/// The first increment's render encoder accepts the sentinel as a four-byte
/// value, mirroring the contract's [`InitialState`] but keeping the object API
/// free of the wire-level `Vec<u8>` that carries it there.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentInitial {
    /// Leave the target's previous contents undefined.
    Undefined,
    /// Pre-fill the target with these tightly packed texel bytes before the
    /// pass runs.
    Sentinel([u8; 4]),
}

/// How one recorded render pass establishes its attachment's contents.
///
/// The contract's `LoadOp` is the trace-side shape; this is the encoder-side
/// one. `Clear` carries the colour the pass starts from, and `Load` means the
/// attachment keeps what it already holds: the encoder snapshots the
/// attachment view's bytes at commit and the trace's own view declaration
/// carries them, so the provider uploads them before the render pass opens
/// (`research/docs/23` §3.3). `DontCare` declares the pre-pass contents
/// undefined: the pass neither reads nor presets them, and the draw alone
/// defines what the attachment stores (`research/docs/23` §3.1, v20).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderAttachmentLoad {
    /// Fill every texel with this colour before drawing.
    Clear([u8; 4]),
    /// Keep the attachment's current contents.
    Load,
    /// Leave the attachment's current contents undefined.
    DontCare,
}

/// Which lease arm one stage-buffer slot's bytes come from
/// (`research/docs/23` §90, R9i).
///
/// The two arms are the ones the trace contract's `BufferSource` states for a
/// lease: the provider's own staged copy of the owner's reservation
/// (`staged_lease`), or the owner's own mapping imported without a copy
/// (`borrowed_no_copy`). A caller-held view is the third arm and has its own
/// entry point ([`RenderCommandEncoder::set_stage_buffer`]), because its bytes
/// are the caller's host image rather than a provider registry's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageBufferLeaseArm {
    /// The provider copied the owner's reservation into its own storage when
    /// the lease was imported.
    StagedLease,
    /// The provider holds the owner's own mapping; the pass reads (or, on the
    /// trace rail, writes) it where the owner registered it.
    BorrowedNoCopy,
}

/// One imported lease bound to a stage-buffer slot
/// (`research/docs/23` §3.3, v87; §90, R9i).
///
/// The caller imports the lease through the same channel a compute case's
/// `storage_mode` uses ([`crate::provider::LeaseImporter`] or
/// [`crate::provider::NoCopyLeaseImporter`]) and hands the reservation here;
/// the object API never imports anything itself, exactly as it never uploads a
/// texture. The reservation's allocation, offset and length are the view the
/// pass states, and the lease's identity becomes the view's own id — the rule
/// the trace rail's captured views follow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageBufferLease {
    /// The provider-registry reservation the bytes come from.
    pub reservation: LeaseReservation,
    /// The owner allocation's registered byte length. The trace's resource
    /// table states it, and a reservation reaching past it is refused by name
    /// (`research/docs/23` §90).
    pub allocation_size: u64,
    /// Which import arm holds the bytes.
    pub arm: StageBufferLeaseArm,
}

/// One colour attachment a multi-attachment draw records.
///
/// The attachment list's position is the attachment's `location` — entry `i`
/// is the target the fragment stage's output `i` lands in — so the list the
/// encoder holds is also the list the pass's descriptor carries. Each entry
/// names its own view, format and load operation, while `width` and `height`
/// stay pass-wide exactly as the contract's shared viewport makes them
/// ([`RenderPassDescriptor`]).
#[derive(Clone, Copy)]
pub struct RenderColorAttachment<'a> {
    /// The buffer view that carries the attachment's identity and byte range.
    pub view: &'a BufferView,
    /// The attachment's colour format, which has to equal the recorded
    /// pipeline's compiled format at the same location.
    pub format: AttachmentFormat,
    /// How the pass establishes this attachment's contents.
    pub load: RenderAttachmentLoad,
    /// How the pass hands this attachment on: `Store` keeps the pass's writes
    /// observable, `DontCare` makes the attachment disappear from the
    /// observable surface instead of passing as "landed correctly". The pass
    /// still has to store at least one attachment — an all-`DontCare` list is
    /// [`ContractError::AllRenderAttachmentsDiscarded`] — so the caller spells
    /// one store decision per location.
    pub store: StoreOp,
}

/// One recorded colour attachment: the buffer view that carries the attachment
/// identity and byte range, plus the format and load the encoder restates.
///
/// [`RenderColorAttachment`] is the recording-time shape a caller spells;
/// this is the owned copy the recorded [`RenderTarget`] keeps, so the draw
/// stays valid after the caller's views are dropped.
#[derive(Clone)]
struct RenderTargetAttachment {
    source: RenderAttachmentSource,
    format: AttachmentFormat,
    load: RenderAttachmentLoad,
    store: StoreOp,
}

/// Where one recorded colour attachment's bytes live
/// (`research/docs/23` §114, E-TX8/E-TX9b).
///
/// The two arms the object rail can record: the caller's own host image — the
/// shape every pre-E-TX9b recording states — and the owner's registered
/// window, whose frame lands in the guest's own pages instead of only in the
/// writeback channel. The window arm's identity is the lease's own, exactly as
/// a lease-bound stage buffer's view is, so the `(allocation, view)` pair the
/// pass's descriptor carries and the serial view list's declaration are one
/// fact.
#[derive(Clone)]
enum RenderAttachmentSource {
    /// A caller-held buffer view.
    View(BufferView),
    /// The owner's registered window.
    Window(StageBufferLease),
}

impl RenderAttachmentSource {
    fn view_id(&self) -> ViewId {
        match self {
            Self::View(view) => view.view_id(),
            Self::Window(lease) => ViewId::new(lease.reservation.lease.lease_id.get()),
        }
    }

    fn allocation_id(&self) -> AllocationId {
        match self {
            Self::View(view) => view.allocation_id(),
            Self::Window(lease) => lease.reservation.lease.allocation_id,
        }
    }
}

/// One colour attachment a multi-attachment draw records
/// (`research/docs/23` §114/§115, E-TX9b).
///
/// The list stays positional — entry `i` is location `i` — and each entry
/// states where its bytes come from: the caller's own buffer view or the
/// owner's registered window. The window arm is the only one that spells
/// [`StoreOp::Borrowed`]; the view arm keeps the store decision the caller
/// states, exactly as [`RenderColorAttachment`] already carries it.
#[derive(Clone, Copy)]
pub enum RenderAttachmentDeclaration<'a> {
    /// The attachment's bytes and landing are the caller's own buffer view.
    View(RenderColorAttachment<'a>),
    /// The attachment's bytes and landing are the owner's registered window.
    Window(RenderWindowAttachment),
}

/// One colour attachment whose stored bytes land in the *owner's registered
/// window* (`research/docs/23` §114/§115, E-TX8/E-TX9b).
///
/// The arm is the store sibling of the load channel R5b/§74 and E-TX6 opened:
/// a guest-backed surface's bytes live in the guest's own pages, and those
/// pages reach the provider as an owner-issued window. A recorded pass states
/// it by declaring *this* window on a pass of the same command and naming its
/// identity as the attachment — the serial view list is where the rail
/// resolves the landing, so the declaration the pass begins from and the
/// memory it lands in cannot disagree.
///
/// The window has to be the attachment's own tightly packed byte extent, and
/// its arm has to be the no-copy one: the frame lands in the owner's live
/// pages, not in a copy the provider staged. A shape this entry cannot state
/// is refused by name at the recorder rather than silently truncated, and the
/// count of colour attachments, their formats and the pass-wide viewport are
/// the multi-attachment entry's own rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderWindowAttachment {
    /// The owner's window the frame lands in
    /// ([`StageBufferLeaseArm::BorrowedNoCopy`]: the provider holds the
    /// owner's own mapping, so the bytes it writes are the guest's).
    pub lease: StageBufferLease,
    /// The attachment's colour format, which has to equal the recorded
    /// pipeline's compiled format at the same location.
    pub format: AttachmentFormat,
    /// How the pass establishes this attachment's contents: `Clear`, `Load`
    /// (which uploads the window's own current bytes, exactly as E-TX6's load
    /// half states) or `DontCare`.
    pub load: RenderAttachmentLoad,
}

/// The colour attachments one recorded render pass stores into, plus the
/// pass-wide viewport and the draw it replays. The attachment list is
/// positional: entry `i` is location `i`, up to [`MAX_COLOR_ATTACHMENTS`].
#[derive(Clone)]
struct RenderTarget {
    attachments: Vec<RenderTargetAttachment>,
    width: u64,
    height: u64,
    /// The scissor rectangle the pass clips to, or `None` for the whole
    /// attachment (`research/docs/23` §3.3, v29/v30). It is encoder state: a
    /// [`RenderEncoder::set_scissor`] call applies it to every pass recorded
    /// afterwards, exactly as Metal's `setScissorRect` does.
    scissor: Option<[u32; 4]>,
    present: Option<PresentInitial>,
    draw: RenderDraw,
    /// The sampled textures the pass's fragment stage reads, in binding order
    /// (`research/docs/23` §3.3, v70). The recording takes the encoder's own
    /// bindings at record time; the bytes they carry are the snapshot the
    /// texture declaration took, exactly as a compute binding's are.
    textures: Vec<contract::TextureView>,
    /// The runtime samplers the pass's fragment stage executes with, in
    /// canonical order (`research/docs/23` §3.3, v102). The recording takes the
    /// encoder's own states at record time, exactly as it takes the textures:
    /// Metal binds the sampler object when the draw is encoded, so the state is
    /// a request fact and the encoder is where the request states it.
    samplers: Vec<contract::RenderSamplerBinding>,
    /// The stage buffers the pass's two stages read — and the writable ones
    /// they land — in canonical order (`research/docs/23` §3.3, v83-v86): one
    /// entry per `[[buffer(N)]]` slot, the pair of the pipeline's own
    /// declaration and the bytes' source arm. Empty for every pre-v83
    /// recording, which is the shape every other pass keeps.
    stage_buffers: Vec<RenderStageSlot>,
}

/// One stage-buffer slot a recorded pass binds (`research/docs/23` §3.3, v87).
///
/// The stage and the index inside that stage's own Metal buffer namespace are
/// the slot; `access` is the *pipeline's* answer, resolved when the pass is
/// recorded from the declaration the recorded pipeline carries, because a
/// Metal encoder's binding call states no access of its own. The source is
/// where the bytes live: a caller-held view the commit snapshots, or an
/// imported lease the provider already holds.
#[derive(Clone)]
struct RenderStageSlot {
    stage: RenderPipelineStage,
    index: u32,
    access: BufferAccess,
    source: RenderStageSource,
}

/// Where one recorded slot's bytes come from (`research/docs/23` §90, R9i).
///
/// The two arms a render pass can name: the caller's own host image, snapshotted
/// at commit under the command's reservations, or a lease whose bytes the
/// provider registry already holds. The borrowed arm's window stays the
/// caller's to keep alive until the submission has been waited for, exactly as
/// the compute rail's `BufferSource` does.
#[derive(Clone)]
enum RenderStageSource {
    /// A caller-held buffer view.
    View(BufferView),
    /// An imported lease, staged or borrowed.
    Lease(StageBufferLease),
}

/// One vertex stream a recorded draw reads: the encoder's own view plus the
/// binding index the pass's positional list has to carry it at.
///
/// The index is stored beside the view instead of being re-derived while the
/// list is built, so the label a view spells and the position it sits at come
/// from one fact: entry `i` of `RenderPassDescriptor::vertex_buffers` is binding
/// `i`, which is what the contract holds a view's own `metal_binding` to
/// (`research/docs/23` §3.6).
#[derive(Clone)]
struct RenderStream {
    binding: u32,
    view: BufferView,
}

/// The index buffer one recorded draw selects through: the encoder's own view
/// plus the width of the indices it holds. An index buffer carries no binding
/// index, so the pass spells it with `metal_binding` zero.
#[derive(Clone)]
struct RenderIndex {
    view: BufferView,
    format: IndexFormat,
}

/// The depth surface one recorded pass opens (`research/docs/23` §3.3,
/// v36/v37/v44).
///
/// The recorded surface mirrors the trace contract's: what a caller states is
/// the extent and how the pass establishes the surface's contents, and — from
/// v44 on — whether the pass keeps it and where its texels land. The two
/// optional fields are the contract's own decision in the same shape: `store`
/// alone is the pre-v44 rail-owned surface dropped with the pass, and a storing
/// surface names its [`contract::RenderDepthIdentity`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderDepthAttachment {
    pub width: u64,
    pub height: u64,
    pub load: RenderDepthLoad,
    /// The store action the pass asks for, or `None` for the rail-owned shape
    /// every pre-v44 recording means (`research/docs/23` §3.3, v43).
    pub store: Option<contract::DepthStoreOp>,
    /// The stored surface's resource identity, present exactly when the
    /// recording observes the texels (`research/docs/23` §3.3, v43).
    pub identity: Option<contract::RenderDepthIdentity>,
}

/// How a recorded pass establishes its depth surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderDepthLoad {
    /// Clear every depth texel to this value.
    Clear(f32),
    /// Keep the surface's previous contents.
    Load,
}

/// The depth state a recorded pass tests and writes with
/// (`research/docs/23` §3.3, v36/v37).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderDepthTest {
    pub compare: contract::CompareFunction,
    pub write: bool,
}

/// The stencil surface one recorded pass opens (`research/docs/23` §3.3,
/// v47/v48).
///
/// The depth surface's sibling one byte wide: rail-owned like it, so what a
/// caller states is the extent and how the pass establishes the surface's
/// contents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderStencilAttachment {
    pub width: u64,
    pub height: u64,
    pub load: RenderStencilLoad,
    /// The store action the pass asks for, or `None` for the rail-owned shape
    /// every pre-v49 recording means (`research/docs/23` §3.3, v49).
    pub store: Option<StoreOp>,
    /// The stored surface's resource identity, present exactly when the
    /// recording observes the texels.
    pub identity: Option<contract::RenderStencilIdentity>,
}

/// How a recorded pass establishes its stencil surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RenderStencilLoad {
    /// Clear every stencil texel to this value.
    Clear(u8),
    /// Keep the surface's previous contents.
    Load,
}

/// The stencil state a recorded pass tests and writes with
/// (`research/docs/23` §3.3, v47/v48).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderStencilTest {
    pub compare: contract::StencilCompare,
    pub fail_op: contract::StencilOp,
    pub depth_fail_op: contract::StencilOp,
    pub pass_op: contract::StencilOp,
    pub read_mask: u8,
    pub write_mask: u8,
    pub reference: u8,
}

/// The draw one recorded render pass replays: the counts, plus the views of the
/// streams and index buffer it reads.
///
/// Every draw input carries its own bytes --
/// `RenderPassDescriptor::vertex_buffers` is the positional `Vec<BufferView>`
/// and `IndexBufferBinding` embeds one -- so an object-API trace needs no
/// compute pass to declare them (`research/docs/23` §3.6). The bytes those views
/// carry are taken at commit, under the command's own reservations, exactly as a
/// compute binding's are; recording answers the contract's shape questions only.
#[derive(Clone)]
struct RenderDraw {
    /// Vertices of a non-indexed draw, or indices of an indexed one. The
    /// descriptor carries both in one `u32` field, which is why this one does
    /// too (`RenderPassDescriptor::vertices`).
    vertices: u32,
    /// One entry per bound stream, in binding order: entry `i` is binding `i`
    /// of the pipeline's vertex layout, the way the descriptor is positional.
    vertex_buffers: Vec<RenderStream>,
    /// The index buffer this draw selects through, or `None` for a non-indexed
    /// draw.
    indices: Option<RenderIndex>,
    /// Instances the draw runs (`research/docs/23` §3.3, v31). Every draw the
    /// object API records today is the single-instance shape, so this stays `1`
    /// until the encoder gains the instanced draw call; the field exists here
    /// because the pass it lands in carries it.
    instance_count: u32,
    /// Vertex offset every index is read through (`research/docs/23` §3.3,
    /// v34). `0` is the shape every object-API draw records today; the field
    /// exists here because the pass it lands in carries it.
    base_vertex: u32,
    /// The culling state this pass draws with, or `None` for "keep every
    /// triangle" — the shape every draw the object API could record before v41
    /// had (`research/docs/23` §3.3, v39/v41).
    cull: Option<contract::RenderPassCull>,
    /// The blend state this pass draws with, or `None` for "write the fragment
    /// output" — the shape every draw the object API could record before v42
    /// had (`research/docs/23` §3.3, v40/v42).
    blend: Option<contract::RenderPassBlend>,
    /// The depth surface this pass opens, or `None` for a pass with no depth
    /// surface — the shape every draw the object API could record before v37
    /// had (`research/docs/23` §3.3, v36/v37).
    depth: Option<RenderDepthAttachment>,
    /// The depth state the pass tests with, or `None` for no test.
    depth_test: Option<RenderDepthTest>,
    /// The stencil surface this pass opens, or `None` for a pass with no
    /// stencil surface — the shape every draw the object API could record
    /// before v48 had (`research/docs/23` §3.3, v47/v48).
    stencil: Option<RenderStencilAttachment>,
    /// The stencil state the pass tests and writes with, or `None` for no test.
    stencil_test: Option<RenderStencilTest>,
    /// The pass-wide multisample raster this pass renders with, or `None` for
    /// the single-sample raster every draw the object API could record before
    /// v52 had (`research/docs/23` §3.3, v51/v52).
    multisample: Option<contract::MultisampleState>,
    /// The resolve the pass applies to a stored multisampled depth surface, or
    /// `None` for the API default [`contract::DepthResolveFilter::Sample0`]
    /// (`research/docs/23` §3.3, v57/v58). Only the recording entry that
    /// names it sets this field, so every earlier recording keeps the absent
    /// shape and its exact bytes.
    depth_resolve: Option<contract::MultisampleDepthResolve>,
    /// The resolve the pass applies to a stored multisampled stencil surface,
    /// or `None` for the API default
    /// [`contract::StencilResolveFilter::Sample0`] (`research/docs/23` §3.3,
    /// v60/v70). Only the recording entry that names it sets this field, so
    /// every earlier recording keeps the absent shape and its exact bytes.
    stencil_resolve: Option<contract::MultisampleStencilResolve>,
}

/// The pass-shaped view one bound draw input becomes.
///
/// A draw input is read-only: the object API has no way to declare a write on a
/// bound view (a compute binding's access comes from the reflected pipeline
/// contract, and a render input has no such reflection), so the access the trace
/// carries is the contract's own read. A view that reached the trace writable
/// would be refused by `validate_vertex_buffer_binding` as
/// [`ContractError::RenderInputAccessUnsupported`].
fn draw_input_view(view: &BufferView, metal_binding: u32, bytes: Vec<u8>) -> contract::BufferView {
    contract::BufferView {
        view_id: view.view_id(),
        metal_binding,
        allocation_id: view.allocation_id(),
        offset: view.offset as u64,
        length: view.length as u64,
        access: BufferAccess::Read,
        attribute_stride: None,
        source: BufferSource::OwnedBytes(bytes),
    }
}

/// The recording-time byte source for a pass's draw inputs.
///
/// A view declares the length its bytes have, and the contract's shape rules
/// compare exactly that, so a placeholder of the declared length answers them
/// without reading the host buffer. The bytes that reach the trace are the ones
/// the commit snapshot takes under its reservations ([`RenderDraw`]); a recorded
/// pass is only the shape proof [`RenderPassDescriptor::validate`] and the
/// pipeline's agreement read.
fn shape_only_bytes(view: &BufferView) -> Vec<u8> {
    vec![0; view.length]
}

impl RenderDraw {
    /// The milestone's `vertex_id` draw: three generated vertices, no bound
    /// stream and no caller-held index buffer. Shared by the direct vertex_id
    /// pass and by an ICB replay whose command carries the input itself.
    fn vertex_id() -> Self {
        Self {
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            vertex_buffers: Vec::new(),
            indices: None,
            instance_count: 1,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        }
    }
}

impl RenderTarget {
    /// The pass this target becomes.
    ///
    /// `input_bytes` supplies the bytes each draw input's own view carries: the
    /// commit snapshot hands back the host bytes its reservations hold, which is
    /// what the trace keeps, while recording hands back a placeholder of the
    /// declared length ([`shape_only_bytes`]) because the shape rules never read
    /// a byte's value. Only the bytes behind the descriptor differ between the
    /// two; the descriptor itself is the same value, and the bytes that reach a
    /// provider are taken once, at commit.
    fn descriptor(
        &self,
        pipeline_id: PipelineId,
        input_bytes: &dyn Fn(&BufferView) -> Vec<u8>,
    ) -> Result<RenderPassDescriptor, Error> {
        // One attachment per location, in list order: the descriptor's entry
        // `i` is location `i`, which is also the order the recorded pipeline's
        // compiled formats are checked against (`validate_against`).
        let color_attachments = self
            .attachments
            .iter()
            .map(|attachment| RenderAttachment {
                view_id: attachment.source.view_id(),
                allocation_id: attachment.source.allocation_id(),
                format: attachment.format,
                width: self.width,
                height: self.height,
                load: match attachment.load {
                    RenderAttachmentLoad::Clear(bytes) => LoadOp::Clear(ClearColor::new(bytes)),
                    RenderAttachmentLoad::Load => LoadOp::Load,
                    RenderAttachmentLoad::DontCare => LoadOp::DontCare,
                },
                store: attachment.store,
            })
            .collect();
        // The present tail hands on the location-0 attachment, which is the
        // single-attachment shape v16/v17 published: the target restates the
        // first attachment's view, allocation, format and extent. A recording
        // with no colour attachment is the depth-only shape (`v46`), which
        // carries no present action — the recording entry refuses that pairing
        // — so location 0 exists whenever this arm is taken.
        let present = match (self.present, self.attachments.first()) {
            (Some(initial), Some(first)) => Some(PresentDescriptor {
                target: PresentTarget {
                    allocation_id: first.source.allocation_id(),
                    view_id: first.source.view_id(),
                    format: first.format,
                    width: self.width,
                    height: self.height,
                    image_count: MAX_PRESENT_IMAGE_COUNT,
                    initial: match initial {
                        PresentInitial::Undefined => InitialState::Undefined,
                        PresentInitial::Sentinel(bytes) => InitialState::Sentinel(bytes.to_vec()),
                    },
                },
                source: first.source.view_id(),
                mode: PresentMode::Fifo,
                acquire: AcquirePolicy::Blocking,
            }),
            (Some(_), None) => return Err(Error::EmptyRenderAttachmentList),
            (None, _) => None,
        };
        let vertex_buffers = self
            .draw
            .vertex_buffers
            .iter()
            .map(|stream| draw_input_view(&stream.view, stream.binding, input_bytes(&stream.view)))
            .collect();
        let indices = self
            .draw
            .indices
            .as_ref()
            .map(|indices| IndexBufferBinding {
                view: draw_input_view(&indices.view, 0, input_bytes(&indices.view)),
                format: indices.format,
            });
        let descriptor = RenderPassDescriptor {
            // The pass's own stage buffers (`research/docs/23` §3.3, v83-v87):
            // one view per slot the encoder bound, in canonical order — the
            // stage's own ordinal first, the binding inside it second, which is
            // the order the contract's own binding walk states. A caller-held
            // view's bytes are taken here, exactly as a draw input's are,
            // while a lease names the import the provider already holds.
            stage_buffers: self
                .stage_buffers
                .iter()
                .map(|slot| StageBufferView {
                    stage: slot.stage,
                    view: match &slot.source {
                        // A caller-held slot states the *declaration's* access,
                        // not the draw inputs' read-only shape: a writable slot
                        // is the landing the writeback channel publishes
                        // (`research/docs/23` §3.3, v86).
                        RenderStageSource::View(view) => contract::BufferView {
                            view_id: view.view_id(),
                            metal_binding: slot.index,
                            allocation_id: view.allocation_id(),
                            offset: view.offset as u64,
                            length: view.length as u64,
                            access: slot.access,
                            attribute_stride: None,
                            source: BufferSource::OwnedBytes(input_bytes(view)),
                        },
                        RenderStageSource::Lease(lease) => contract::BufferView {
                            // The lease's own identity is the view's id, the
                            // rule the trace rail's captured views follow: the
                            // reservation names the bytes, so the view cannot
                            // name a different set.
                            view_id: ViewId::new(lease.reservation.lease.lease_id.get()),
                            metal_binding: slot.index,
                            allocation_id: lease.reservation.lease.allocation_id,
                            offset: lease.reservation.offset,
                            length: lease.reservation.length,
                            access: slot.access,
                            attribute_stride: None,
                            source: match lease.arm {
                                StageBufferLeaseArm::StagedLease => {
                                    BufferSource::StagedLease(lease.reservation.lease.lease_id)
                                }
                                StageBufferLeaseArm::BorrowedNoCopy => {
                                    BufferSource::BorrowedNoCopy(lease.reservation.lease.lease_id)
                                }
                            },
                        },
                    },
                })
                .collect(),
            // The pass-wide multisample raster travels with the recording
            // (`research/docs/23` §3.3, v51/v52): the entry that names it is
            // the only one that sets this field, so every earlier recording
            // keeps the single-sample shape and its exact bytes.
            multisample: self.draw.multisample,
            // The resolve travels with the recording (`research/docs/23` §3.3,
            // v57/v58): the entry that names it is the only one that sets this
            // field, so every earlier recording keeps the API default filter
            // and its exact bytes.
            depth_resolve: self.draw.depth_resolve,
            // The stencil resolve travels with the recording exactly as the
            // depth resolve above does (`research/docs/23` §3.3, v60/v70): the
            // entry that names it is the only one that sets this field, so
            // every earlier recording keeps the API default filter and its
            // exact bytes.
            stencil_resolve: self.draw.stencil_resolve,
            // The blend state is the pass's own, exactly as the culling and
            // depth entries state theirs (`research/docs/23` §3.3, v40/v42).
            blend: self.draw.blend.clone(),
            // The culling state is the pass's own, exactly as the depth entry
            // states its surface (`research/docs/23` §3.3, v39/v41).
            cull: self.draw.cull,
            // The object API's depth surface states the shape directly: it has
            // no trace identity, so this is the same rail-owned description the
            // trace contract carries (`research/docs/23` §3.3, v36/v37).
            depth: self
                .draw
                .depth
                .map(|depth| contract::RenderDepthAttachment {
                    format: contract::DepthFormat::Depth32Float,
                    width: depth.width,
                    height: depth.height,
                    load: match depth.load {
                        RenderDepthLoad::Clear(value) => contract::DepthLoadOp::clear(value),
                        RenderDepthLoad::Load => contract::DepthLoadOp::Load,
                    },
                    // The recording's own store action and landing identity
                    // travel straight into the contract's depth attachment
                    // (`research/docs/23` §3.3, v43/v44): one description, so
                    // the recorded pass and the trace it becomes cannot
                    // disagree about whether the surface is kept.
                    store: depth.store,
                    identity: depth.identity,
                }),
            depth_test: self.draw.depth_test.map(|test| contract::DepthTest {
                compare: test.compare,
                write: test.write,
            }),
            // The object API records no stencil surface yet (`research/docs/23`
            // §3.3, v48): the recording's own surface and state travel straight
            // into the contract's stencil fields, exactly as the depth pair
            // above does. A recording that states neither keeps the pre-v48
            // shape.
            stencil: self
                .draw
                .stencil
                .map(|stencil| contract::RenderStencilAttachment {
                    format: contract::StencilFormat::Stencil8,
                    width: stencil.width,
                    height: stencil.height,
                    load: match stencil.load {
                        RenderStencilLoad::Clear(value) => contract::StencilLoadOp::clear(value),
                        RenderStencilLoad::Load => contract::StencilLoadOp::Load,
                    },
                    // The object API records no stored stencil surface yet
                    // (`research/docs/23` §3.3, v49): the recording's own store
                    // action and landing identity travel into the contract,
                    // exactly as the depth pair above does.
                    store: stencil.store,
                    identity: stencil.identity,
                }),
            stencil_test: self.draw.stencil_test.map(|test| contract::StencilTest {
                compare: test.compare,
                fail_op: test.fail_op,
                depth_fail_op: test.depth_fail_op,
                pass_op: test.pass_op,
                read_mask: test.read_mask,
                write_mask: test.write_mask,
                reference: test.reference,
            }),
            scissor: self.scissor,
            pipeline: pipeline_id,
            color_attachments,
            viewport: [
                0,
                0,
                u32::try_from(self.width)
                    .map_err(|_| ContractError::ArithmeticOverflow("attachment width"))?,
                u32::try_from(self.height)
                    .map_err(|_| ContractError::ArithmeticOverflow("attachment height"))?,
            ],
            vertices: self.draw.vertices,
            vertex_buffers,
            indices,
            base_vertex: self.draw.base_vertex,
            instance_count: self.draw.instance_count,
            // The fragment stage's sampled textures travel with the recording
            // (`research/docs/23` §3.3, v70): the encoder's own bindings, in
            // ascending binding order, each carrying the bytes its declaration
            // snapshotted.
            textures: self.textures.clone(),
            samplers: self.samplers.clone(),
            present,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
}

struct BufferInner {
    owner: Arc<DeviceState>,
    allocation_id: AllocationId,
    length: usize,
    bytes: Mutex<Vec<u8>>,
    reservations: Mutex<Vec<RangeHold>>,
    available: Condvar,
}

/// One texture declared through the object API. The bytes are the snapshot the
/// provider uploads; a storage texture's handle is also the landing a completed
/// command writes back into (`research/docs/26` §21.4, C2), exactly as a
/// `Buffer`'s bytes are its landing. The whole texture is one in-flight
/// reservation unit (`research/docs/16` §2), so a landing cannot interleave
/// with CPU access or with a sibling command that names the same texture.
struct TextureInner {
    owner: Arc<DeviceState>,
    allocation_id: AllocationId,
    view_id: ViewId,
    format: contract::TextureFormat,
    width: u64,
    height: u64,
    length: usize,
    access: contract::TextureAccess,
    /// Where every view this handle declares takes its texels from
    /// (`research/docs/23` §110, E-TX3).
    origin: TextureOrigin,
    bytes: Mutex<Vec<u8>>,
    reservations: Mutex<Vec<RangeHold>>,
    available: Condvar,
}

/// Where one texture handle's texels come from.
///
/// The object API owns two of the contract's four source arms: the bytes the
/// constructor was handed (`OwnedBytes`), and — the shape this increment adds
/// — the trace's own production (`TraceView`), whose texels do not exist until
/// the command that produces them runs. The lease arms are the trace rail's:
/// the object API never imports owner backing, exactly as it never uploads a
/// texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextureOrigin {
    /// The handle's bytes are its declared initial contents.
    DeclaredBytes,
    /// The handle names one view the trace's own earlier pass stores
    /// (`research/docs/23` §110, E-TX3): it carries no bytes of its own and
    /// shares the producing view's identity.
    TraceView,
}

/// A texture handle: a sampled or texel-fetched texture is a read-only source,
/// and a storage texture a compute write target whose completion updates this
/// handle's own bytes. Clone is cheap and shares the same allocation.
#[derive(Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
}

impl Texture {
    pub fn allocation_id(&self) -> AllocationId {
        self.inner.allocation_id
    }

    pub fn view_id(&self) -> ViewId {
        self.inner.view_id
    }

    pub fn format(&self) -> contract::TextureFormat {
        self.inner.format
    }

    pub fn dimensions(&self) -> (u64, u64) {
        (self.inner.width, self.inner.height)
    }

    /// The access every view this handle declares carries: `Sampled` for a
    /// texture declared through [`Device::new_texture_with_bytes`], `Fetched`
    /// for one declared through [`Device::new_fetched_texture_with_bytes`], and
    /// `Storage` for one declared through
    /// [`Device::new_storage_texture_with_bytes`].
    pub fn access(&self) -> contract::TextureAccess {
        self.inner.access
    }

    /// The bytes this texture currently holds: its declared initial contents
    /// until a completed storage landing replaces them, exactly as
    /// [`Buffer::read`] observes a buffer writeback. Waits while an in-flight
    /// command holds a conflicting access (a read waits only on a writer), so a
    /// caller never observes half a landing.
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        // A trace-view handle has no host image at all: its texels are the
        // bytes a pass of the same command stores, and those bytes are
        // observed through the pass that samples them
        // (`research/docs/23` §110, E-TX3). Returning an empty or stale
        // vector would read as "the texture holds nothing".
        if self.inner.origin == TextureOrigin::TraceView {
            return Err(Error::TraceViewTextureHasNoHostBytes {
                view: self.inner.view_id,
            });
        }
        Ok(self.lock_unreserved(false)?.clone())
    }

    /// Wait until no in-flight reservation conflicts with the whole texture,
    /// then hold the bytes. `Buffer::lock_unreserved`'s whole-resource sibling:
    /// the first increment reserves the whole texture, so the recheck is the
    /// same read-read-allowed rule on one `[0, length)` range.
    fn lock_unreserved(&self, write: bool) -> Result<MutexGuard<'_, Vec<u8>>, Error> {
        let end = self.inner.length;
        loop {
            let mut reservations = lock(&self.inner.reservations, "provider texture reservation")?;
            while ranges_conflict(&reservations, 0, end, write) {
                reservations = self
                    .inner
                    .available
                    .wait(reservations)
                    .map_err(|_| ApiError::StatePoisoned("provider texture reservation"))?;
            }
            drop(reservations);
            let bytes = lock(&self.inner.bytes, "provider texture")?;
            let clear = !ranges_conflict(
                &lock(&self.inner.reservations, "provider texture reservation")?,
                0,
                end,
                write,
            );
            if clear {
                return Ok(bytes);
            }
            drop(bytes);
        }
    }

    /// Register the whole texture against one submission, waiting while any
    /// in-flight command holds a conflicting access. A sampled texture is a
    /// read; a storage texture is the write its landing performs, so two
    /// storage commands naming one texture serialize instead of interleaving.
    fn reserve(&self) -> Result<TextureReservation, Error> {
        let identity = next_id()?;
        let write = self.inner.access == contract::TextureAccess::Storage;
        let end = self.inner.length;
        let mut reservations = lock(&self.inner.reservations, "provider texture reservation")?;
        while ranges_conflict(&reservations, 0, end, write) {
            reservations = self
                .inner
                .available
                .wait(reservations)
                .map_err(|_| ApiError::StatePoisoned("provider texture reservation"))?;
        }
        reservations.push(RangeHold {
            reservation: identity,
            start: 0,
            end,
            write,
        });
        Ok(TextureReservation {
            inner: Arc::clone(&self.inner),
            identity,
        })
    }

    /// The contract view for one binding, mirroring `BufferView`'s snapshot.
    fn view(&self, metal_binding: u32) -> Result<contract::TextureView, Error> {
        let source = match self.inner.origin {
            TextureOrigin::DeclaredBytes => contract::TextureSource::OwnedBytes(
                lock(&self.inner.bytes, "provider texture")?.clone(),
            ),
            // The producer is the view whose identity this handle shares, so
            // the declaration resolves against the trace's own stores without
            // a second name (`research/docs/23` §110, E-TX3).
            TextureOrigin::TraceView => contract::TextureSource::TraceView,
        };
        Ok(contract::TextureView {
            view_id: self.inner.view_id,
            metal_binding,
            allocation_id: self.inner.allocation_id,
            texture_type: contract::TextureType::D2,
            format: self.inner.format,
            width: self.inner.width,
            height: self.inner.height,
            depth: 1,
            array_length: 1,
            sample_count: 1,
            access: self.inner.access,
            source,
        })
    }
}

/// Commit-through-completion reservation for one whole texture.
/// [`BufferReservation`]'s sibling: the guard is held from the commit-time
/// snapshot through the landing, and dropping it wakes CPU readers and sibling
/// commands even when a pending command is abandoned.
struct TextureReservation {
    inner: Arc<TextureInner>,
    identity: u64,
}
impl TextureReservation {
    fn allocation_id(&self) -> AllocationId {
        self.inner.allocation_id
    }
    fn view_id(&self) -> ViewId {
        self.inner.view_id
    }
    fn lock_bytes(&self) -> Result<MutexGuard<'_, Vec<u8>>, Error> {
        lock(&self.inner.bytes, "provider texture")
    }
}
impl Drop for TextureReservation {
    fn drop(&mut self) {
        fn release(holds: &mut Vec<RangeHold>, identity: u64) {
            holds.retain(|hold| hold.reservation != identity);
        }
        match self.inner.reservations.lock() {
            Ok(mut reservations) => release(&mut reservations, self.identity),
            Err(poisoned) => release(&mut poisoned.into_inner(), self.identity),
        }
        self.inner.available.notify_all();
    }
}

/// One in-flight byte range held by a submission, tagged with the reservation
/// that owns it so release removes exactly its own entries.
struct RangeHold {
    reservation: u64,
    start: usize,
    end: usize,
    write: bool,
}

impl RangeHold {
    /// Ranged hazard rule from `research/docs/14` §3.2: an overlap conflicts
    /// when at least one side writes, so read-read pairs never conflict.
    fn conflicts(&self, start: usize, end: usize, write: bool) -> bool {
        self.start < end && start < self.end && (write || self.write)
    }
}

/// True when any in-flight range conflicts with `[start, end)`.
fn ranges_conflict(holds: &[RangeHold], start: usize, end: usize, write: bool) -> bool {
    holds.iter().any(|hold| hold.conflicts(start, end, write))
}

/// Fixed-size CPU storage. Reads and writes wait while a command uses it.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}
impl Buffer {
    pub fn allocation_id(&self) -> AllocationId {
        self.inner.allocation_id
    }
    pub fn read(&self) -> Result<Vec<u8>, Error> {
        Ok(self.lock_unreserved(0, self.inner.length, false)?.clone())
    }
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = checked_range(offset, bytes.len(), self.inner.length)?;
        self.lock_unreserved(offset, end, true)?[offset..end].copy_from_slice(bytes);
        Ok(())
    }
    /// Allocate a distinct logical view. Clone the returned view to reuse its
    /// identity across dispatches; creating another view is an alias.
    pub fn view(&self, offset: usize, length: usize) -> Result<BufferView, Error> {
        if length == 0 {
            return Err(ContractError::ZeroLength("buffer view").into());
        }
        checked_range(offset, length, self.inner.length)?;
        Ok(BufferView {
            buffer: self.clone(),
            view_id: ViewId::new(next_id()?),
            offset,
            length,
        })
    }

    /// Wait until no in-flight range conflicts with `[start, end)`, then hold
    /// the bytes. A reservation may be acquired between the conflict check and
    /// this lock; re-check under the bytes guard and retry so CPU access is
    /// linearized either completely before the commit-time snapshot or
    /// completely after the command releases its range.
    fn lock_unreserved(
        &self,
        start: usize,
        end: usize,
        write: bool,
    ) -> Result<MutexGuard<'_, Vec<u8>>, Error> {
        loop {
            let mut reservations = lock(&self.inner.reservations, "provider buffer reservation")?;
            while ranges_conflict(&reservations, start, end, write) {
                reservations = self
                    .inner
                    .available
                    .wait(reservations)
                    .map_err(|_| ApiError::StatePoisoned("provider buffer reservation"))?;
            }
            drop(reservations);
            let bytes = lock(&self.inner.bytes, "provider buffer")?;
            let clear = !ranges_conflict(
                &lock(&self.inner.reservations, "provider buffer reservation")?,
                start,
                end,
                write,
            );
            if clear {
                return Ok(bytes);
            }
            drop(bytes);
        }
    }

    /// Register every range of one submission against this allocation, waiting
    /// while any of them conflicts with an in-flight range. Disjoint commands
    /// of one allocation therefore proceed instead of serializing on the whole
    /// allocation; `research/docs/14` §5 step 3.
    fn reserve_ranges(&self, ranges: &[(usize, usize, bool)]) -> Result<BufferReservation, Error> {
        let identity = next_id()?;
        let mut reservations = lock(&self.inner.reservations, "provider buffer reservation")?;
        while ranges
            .iter()
            .any(|&(start, end, write)| ranges_conflict(&reservations, start, end, write))
        {
            reservations = self
                .inner
                .available
                .wait(reservations)
                .map_err(|_| ApiError::StatePoisoned("provider buffer reservation"))?;
        }
        for &(start, end, write) in ranges {
            reservations.push(RangeHold {
                reservation: identity,
                start,
                end,
                write,
            });
        }
        Ok(BufferReservation {
            inner: Arc::clone(&self.inner),
            identity,
        })
    }
}

/// Commit-through-completion reservation for a set of byte ranges of one
/// allocation. The guard is `Send` so finalization may run on the waiting
/// thread, and dropping it wakes CPU accessors and sibling commands even when a
/// pending command is abandoned.
struct BufferReservation {
    inner: Arc<BufferInner>,
    identity: u64,
}
impl BufferReservation {
    fn allocation_id(&self) -> AllocationId {
        self.inner.allocation_id
    }
    fn lock_bytes(&self) -> Result<MutexGuard<'_, Vec<u8>>, Error> {
        lock(&self.inner.bytes, "provider buffer")
    }
}
impl Drop for BufferReservation {
    fn drop(&mut self) {
        fn release(holds: &mut Vec<RangeHold>, identity: u64) {
            holds.retain(|hold| hold.reservation != identity);
        }
        match self.inner.reservations.lock() {
            Ok(mut reservations) => release(&mut reservations, self.identity),
            Err(poisoned) => release(&mut poisoned.into_inner(), self.identity),
        }
        self.inner.available.notify_all();
    }
}

fn checked_range(offset: usize, length: usize, allocation_length: usize) -> Result<usize, Error> {
    offset
        .checked_add(length)
        .filter(|end| *end <= allocation_length)
        .ok_or_else(|| {
            ApiError::BufferOffsetOutOfBounds {
                offset,
                length: allocation_length,
            }
            .into()
        })
}

/// Reserve every range a command touches, allocating in identity order. All
/// commands use the same order, so overlapping ranges serialize while disjoint
/// ranges of one allocation proceed. A binding whose access the reflected
/// contract does not describe is treated as a write, so an unknown binding can
/// never widen concurrency.
/// Distinct textures referenced by the recorded passes, in first-use order.
/// The provider uploads each allocation once per submission.
fn collect_textures(passes: &[RecordedPass]) -> Vec<Texture> {
    let mut textures = Vec::new();
    let mut seen = BTreeSet::new();
    for pass in passes {
        if let RecordedPass::Compute {
            textures: bound, ..
        } = pass
        {
            for texture in bound.values() {
                if seen.insert(texture.view_id()) {
                    textures.push(texture.clone());
                }
            }
        }
    }
    textures
}

/// The distinct view identities every recorded pass touches, in identity order.
///
/// A render pass contributes its attachment view exactly like a compute pass
/// contributes its bindings, so the serial-resource budget covers both rails
/// from one walk instead of growing a render-only counter.
fn recorded_view_ids(passes: &[RecordedPass]) -> BTreeSet<ViewId> {
    let mut ids = BTreeSet::new();
    for pass in passes {
        match pass {
            RecordedPass::Compute { buffers, .. } => {
                ids.extend(buffers.values().map(ComputeBindingSource::view_id));
            }
            RecordedPass::Render { target, .. } => {
                ids.extend(
                    target
                        .attachments
                        .iter()
                        .map(|attachment| attachment.source.view_id()),
                );
            }
        }
    }
    ids
}

/// Reserve every range every recorded pass touches, in identity order.
///
/// A render pass reserves a `Clear` or `DontCare` attachment as a write (the
/// pass lands texels there without reading the old ones) and a `Load`
/// attachment as a read: a load snapshots the attachment's current bytes at
/// commit, exactly as the draw inputs do, and a read range only excludes a
/// conflicting write (`research/docs/14` §3.2), so a CPU reader is not made to
/// wait out the whole window. Each stream and index buffer a pass draws through
/// is reserved as a read for the same reason.
fn reserve_buffers(passes: &[RecordedPass]) -> Result<Vec<BufferReservation>, Error> {
    let mut by_allocation =
        BTreeMap::<AllocationId, (&Buffer, BTreeMap<(usize, usize), bool>)>::new();
    for pass in passes {
        match pass {
            RecordedPass::Compute {
                pipeline, buffers, ..
            } => {
                let metadata = pipeline.metadata();
                for (binding, view) in buffers {
                    // A lease-bound slot owns no host image of the object API's:
                    // its bytes live in the provider's staged copy or in the
                    // owner's own mapping, exactly as a lease-bound render
                    // stage buffer's do (`research/docs/23` §90, R9i/E-TX9b),
                    // so there is no range here for a CPU reader to wait on.
                    let ComputeBindingSource::View(view) = view else {
                        continue;
                    };
                    let write = metadata
                        .contract
                        .buffer_bindings
                        .iter()
                        .find(|value| value.metal_binding == *binding)
                        .is_none_or(|value| value.access != BufferAccess::Read);
                    by_allocation
                        .entry(view.allocation_id())
                        .or_insert_with(|| (&view.buffer, BTreeMap::new()))
                        .1
                        .entry((view.offset, view.offset + view.length))
                        .and_modify(|existing| *existing |= write)
                        .or_insert(write);
                }
            }
            RecordedPass::Render { target, .. } => {
                for attachment in &target.attachments {
                    // A window-bound attachment owns no host image of the
                    // object API's: its bytes are the owner's own pages, which
                    // this rail never holds a guard for — the same reason a
                    // lease-bound stage buffer reserves no range
                    // (`research/docs/23` §90, R9i/E-TX9b).
                    let RenderAttachmentSource::View(view) = &attachment.source else {
                        continue;
                    };
                    let write = !matches!(attachment.load, RenderAttachmentLoad::Load);
                    by_allocation
                        .entry(view.allocation_id())
                        .or_insert_with(|| (&view.buffer, BTreeMap::new()))
                        .1
                        .entry((view.offset, view.offset + view.length))
                        .and_modify(|existing| *existing |= write)
                        .or_insert(write);
                }
                // The draw's own inputs are read: the commit snapshot copies
                // their bytes into the trace, and a read range only excludes a
                // conflicting write (`research/docs/14` §3.2), so a CPU reader
                // of a stream is not made to wait out the whole window.
                let inputs = target
                    .draw
                    .vertex_buffers
                    .iter()
                    .map(|stream| &stream.view)
                    .chain(target.draw.indices.iter().map(|indices| &indices.view));
                for view in inputs {
                    by_allocation
                        .entry(view.allocation_id())
                        .or_insert_with(|| (&view.buffer, BTreeMap::new()))
                        .1
                        .entry((view.offset, view.offset + view.length))
                        .or_insert(false);
                }
                // A caller-held stage buffer is reserved like the inputs
                // above, with one difference: the slot the pipeline declares
                // writable is a *landing*, so its range is a write — the
                // writeback channel copies the pass's bytes into this very
                // image, and a CPU reader must wait the window out
                // (`research/docs/23` §3.3, v86). A lease-bound slot owns no
                // host image here: its bytes live in the provider's registry.
                for slot in &target.stage_buffers {
                    let RenderStageSource::View(view) = &slot.source else {
                        continue;
                    };
                    let write = slot.access.is_writable();
                    by_allocation
                        .entry(view.allocation_id())
                        .or_insert_with(|| (&view.buffer, BTreeMap::new()))
                        .1
                        .entry((view.offset, view.offset + view.length))
                        .and_modify(|existing| *existing |= write)
                        .or_insert(write);
                }
            }
        }
    }
    by_allocation
        .into_values()
        .map(|(buffer, ranges)| {
            let ranges = ranges
                .into_iter()
                .map(|((start, end), write)| (start, end, write))
                .collect::<Vec<_>>();
            buffer.reserve_ranges(&ranges)
        })
        .collect()
}

/// Reserve every texture a command touches, in identity order.
///
/// One range per texture — the whole resource (`research/docs/16` §2) — with
/// the access the object itself declares: a sampled texture is a read and a
/// storage texture the write its landing performs. Identity order is the same
/// cycle guard `reserve_buffers` uses: two commands naming overlapping textures
/// take their holds in one order, so neither can wait on the other's hold.
fn reserve_textures(textures: &[Texture]) -> Result<Vec<TextureReservation>, Error> {
    let mut ordered = textures.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|texture| texture.allocation_id());
    let mut reservations = Vec::with_capacity(ordered.len());
    for texture in ordered {
        reservations.push(texture.reserve()?);
    }
    Ok(reservations)
}

/// Everything one commit holds in flight until its landing: the byte ranges of
/// the buffers it touches and the whole-resource holds of the textures it
/// names. One value keeps both halves together through the deferred completion,
/// so a storage image's writeback lands under the same guards a buffer's does
/// (`research/docs/26` §21.4, C2).
struct CommandReservations {
    buffers: Vec<BufferReservation>,
    textures: Vec<TextureReservation>,
}

/// Validate every writeback range before copying any host byte, then land all
/// of them under the reservations held by the caller: buffer ranges land in the
/// buffer's own bytes, and a storage image lands its whole tightly packed extent
/// in the texture's bytes (`research/docs/26` §21.4, C2). Both are the same
/// identity-keyed channel; the identity decides the target, and an identity no
/// reservation holds is refused by name rather than dropped.
/// The identities a command's own lease declarations carry
/// (`research/docs/23` §90, R9i; §114, E-TX8/E-TX9b).
///
/// The writeback channel publishes the bytes a lease-bound view's landing
/// holds — a `StoreOp::Borrowed` attachment's frame in particular — and this
/// rail holds no host image for those bytes: they live in the provider's
/// staged copy or in the owner's own pages, and the rail's landing has already
/// written the frame into the window. A writeback for one of these identities
/// therefore copies nothing into the object API's own resources; the bytes it
/// carries are the ones the owner's memory holds.
fn lease_bound_identities(trace: &ComputeTrace) -> BTreeSet<(AllocationId, ViewId)> {
    trace
        .serial_resources()
        .unwrap_or_default()
        .into_iter()
        .filter(|view| {
            matches!(
                view.source,
                BufferSource::StagedLease(_)
                    | BufferSource::BorrowedNoCopy(_)
                    | BufferSource::GuestRuns(_)
            )
        })
        .map(|view| (view.allocation_id, view.view_id))
        .collect()
}

fn apply_writebacks(
    reservations: &[BufferReservation],
    textures: &[TextureReservation],
    writebacks: &[BufferWriteback],
    lease_bound: &BTreeSet<(AllocationId, ViewId)>,
) -> Result<(), Error> {
    if writebacks.is_empty() {
        return Ok(());
    }
    let positions = reservations
        .iter()
        .enumerate()
        .map(|(position, reservation)| (reservation.allocation_id(), position))
        .collect::<BTreeMap<_, _>>();
    let mut writes = Vec::with_capacity(writebacks.len());
    let mut texture_writes = Vec::new();
    for writeback in writebacks {
        let Some(position) = positions.get(&writeback.allocation_id) else {
            // A writeback a *lease-bound declaration of this same command*
            // carries has no host image here (`research/docs/23` §90, R9i;
            // §114, E-TX8/E-TX9b): the bytes live in the provider's staged
            // copy or in the owner's own pages, and the rail's landing has
            // already written the frame into the owner's window. The channel
            // carries the same bytes, so copying nothing loses nothing.
            if lease_bound.contains(&(writeback.allocation_id, writeback.view_id)) {
                continue;
            }
            // A texture identity keys the second target: core admission has
            // already refused an unknown identity, a read-only texture, a short
            // landing and an offset landing against the trace, so this arm
            // rechecks the shape of the one target kind left.
            let texture = textures
                .iter()
                .find(|texture| {
                    texture.allocation_id() == writeback.allocation_id
                        && texture.view_id() == writeback.view_id
                })
                .ok_or(ContractError::UnknownWriteback {
                    allocation: writeback.allocation_id,
                    view: writeback.view_id,
                })?;
            if writeback.offset != 0 {
                return Err(ContractError::WritebackRangeOutOfBounds {
                    view: writeback.view_id,
                    offset: writeback.offset,
                    end: writeback
                        .offset
                        .saturating_add(writeback.bytes.len() as u64),
                    view_offset: 0,
                    view_end: texture.inner.length as u64,
                }
                .into());
            }
            if writeback.bytes.len() != texture.inner.length {
                return Err(ContractError::IncompleteWriteback(writeback.view_id).into());
            }
            texture_writes.push((texture, &writeback.bytes));
            continue;
        };
        let position = *position;
        let offset = usize::try_from(writeback.offset)
            .map_err(|_| ContractError::ArithmeticOverflow("writeback offset"))?;
        let end = checked_range(
            offset,
            writeback.bytes.len(),
            reservations[position].inner.length,
        )?;
        writes.push((position, offset, end, &writeback.bytes));
    }
    let mut guards = Vec::with_capacity(reservations.len());
    for reservation in reservations {
        guards.push(reservation.lock_bytes()?);
    }
    for (position, offset, end, bytes) in writes {
        guards[position][offset..end].copy_from_slice(bytes);
    }
    for (texture, bytes) in texture_writes {
        texture.lock_bytes()?.copy_from_slice(bytes);
    }
    Ok(())
}

/// Immutable range and identity in one buffer allocation.
#[derive(Clone)]
pub struct BufferView {
    buffer: Buffer,
    view_id: ViewId,
    offset: usize,
    length: usize,
}
impl BufferView {
    pub fn view_id(&self) -> ViewId {
        self.view_id
    }
    pub fn allocation_id(&self) -> AllocationId {
        self.buffer.allocation_id()
    }
    /// Byte interval of this view inside its allocation, for range hazards.
    fn range(&self) -> BufferRange {
        BufferRange::new(
            self.buffer.allocation_id(),
            self.offset as u64,
            self.length as u64,
        )
    }
}

/// A heap and the resources placed inside it (`research/docs/25` §6 Step 6).
///
/// [`Device::new_heap`] fixes the descriptor for the heap's lifetime.
/// [`Heap::place`] records one buffer placement, keeping the placement's
/// allocation identity so commit can publish the placements in ascending
/// allocation order — the exact order the provider's placement map reads
/// (`compute_provider.rs::plan_heap_placements`). The heap's identity and
/// descriptor are shared across clones, so several commands can name the same
/// slab.
#[derive(Clone)]
pub struct Heap {
    inner: Arc<HeapInner>,
}

struct HeapInner {
    owner: Arc<DeviceState>,
    heap_id: HeapId,
    descriptor: HeapDescriptor,
    placements: Mutex<Vec<HeapPlacementRecord>>,
}

#[derive(Clone, Copy)]
struct HeapPlacementRecord {
    allocation_id: AllocationId,
    offset: u64,
    byte_size: u64,
}

impl Heap {
    /// The neutral heap identifier shared by every placement.
    pub fn heap_id(&self) -> HeapId {
        self.inner.heap_id
    }

    /// The fixed descriptor this heap was declared with.
    pub fn descriptor(&self) -> HeapDescriptor {
        self.inner.descriptor
    }

    /// Place one whole allocation at `offset` inside this heap.
    ///
    /// The buffer has to belong to the heap's device, an allocation may only
    /// be placed once, `offset + buffer.length` has to fit the slab, and — the
    /// first increment always refuses aliasing — the byte interval has to stay
    /// disjoint from every earlier placement. Alignment is a provider concern
    /// and is not stored here (`research/docs/25` §4.2).
    pub fn place(&self, buffer: &Buffer, offset: u64) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.inner.owner, &buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        let allocation_id = buffer.allocation_id();
        let byte_size = u64::try_from(buffer.inner.length)
            .map_err(|_| ContractError::ArithmeticOverflow("heap placement size"))?;
        let mut placements = lock(&self.inner.placements, "provider heap placements")?;
        if placements
            .iter()
            .any(|placement| placement.allocation_id == allocation_id)
        {
            return Err(Error::HeapPlacementDuplicate {
                allocation: allocation_id,
            });
        }
        let candidate = HeapPlacement {
            heap_id: self.inner.heap_id,
            offset,
            resource: HeapResource::Buffer { byte_size },
        };
        candidate.validate_against(&self.inner.descriptor)?;
        if !self.inner.descriptor.allows_aliasing {
            let overlap = placements.iter().find(|placement| {
                candidate.overlaps(&HeapPlacement {
                    heap_id: self.inner.heap_id,
                    offset: placement.offset,
                    resource: HeapResource::Buffer {
                        byte_size: placement.byte_size,
                    },
                })
            });
            if let Some(previous) = overlap {
                return Err(ContractError::HeapPlacementOverlap {
                    heap: self.inner.heap_id,
                    first_offset: previous.offset,
                    first_size: previous.byte_size,
                    second_offset: offset,
                    second_size: byte_size,
                }
                .into());
            }
        }
        placements.push(HeapPlacementRecord {
            allocation_id,
            offset,
            byte_size,
        });
        Ok(())
    }

    /// The trace payload this heap contributes, or `None` when nothing has been
    /// placed yet. Placements are published in ascending allocation order, the
    /// order the provider's placement map zips against the trace's owned
    /// allocations.
    fn payload(&self) -> Option<HeapPayload> {
        let mut records = lock(&self.inner.placements, "provider heap placements")
            .ok()?
            .clone();
        if records.is_empty() {
            return None;
        }
        records.sort_by_key(|record| record.allocation_id);
        Some(HeapPayload {
            descriptor: self.inner.descriptor,
            placements: records
                .into_iter()
                .map(|record| HeapPlacement {
                    heap_id: self.inner.heap_id,
                    offset: record.offset,
                    resource: HeapResource::Buffer {
                        byte_size: record.byte_size,
                    },
                })
                .collect(),
        })
    }
}

/// One indirect command buffer (`research/docs/25` §6 Step 6).
///
/// The object wraps the neutral [`IndirectCommandPayload`] verbatim: a fixed
/// command cap and kind whitelist plus the single command and replay range the
/// first increment encodes. Encoders replay it with
/// [`ComputeCommandEncoder::dispatch_indirect`] or
/// [`RenderCommandEncoder::draw_indirect`].
#[derive(Clone)]
pub struct IndirectCommandBuffer {
    inner: Arc<IndirectCommandBufferInner>,
}

struct IndirectCommandBufferInner {
    owner: Arc<DeviceState>,
    payload: IndirectCommandPayload,
}

impl IndirectCommandBuffer {
    /// The neutral payload the command replays.
    pub fn payload(&self) -> &IndirectCommandPayload {
        &self.inner.payload
    }

    /// The kind of the single command this buffer encodes.
    pub fn command_kind(&self) -> IndirectCommandKind {
        self.inner.payload.command.kind()
    }
}

#[derive(Clone)]
pub struct CommandQueue {
    owner: Arc<DeviceState>,
}
impl CommandQueue {
    pub fn command_buffer(&self) -> CommandBuffer {
        CommandBuffer {
            shared: Arc::new(CommandShared {
                owner: Arc::clone(&self.owner),
                inner: Mutex::new(CommandInner {
                    passes: Vec::new(),
                    encoder_open: false,
                    heap: None,
                    indirect: None,
                    recording_error: None,
                    status: CommandBufferStatus::Recording,
                    failure: None,
                    submission: None,
                    completion: None,
                    pending: None,
                }),
                completion: Condvar::new(),
            }),
        }
    }
}

#[derive(Clone)]
enum RecordedPass {
    Compute {
        pipeline: Pipeline,
        buffers: BTreeMap<u32, ComputeBindingSource>,
        textures: BTreeMap<u32, Texture>,
        dispatch: Dispatch,
    },
    Render {
        pipeline: RenderPipeline,
        /// Boxed for the same reason [`ComputeTrace::heap`] is: the recorded
        /// render pass carries the whole pass contract, and a per-increment
        /// section ([`RenderTarget::textures`] is v70's) would otherwise grow
        /// every recorded pass, compute ones included
        /// (`research/docs/23` §3.3).
        target: Box<RenderTarget>,
    },
}
struct CommandInner {
    passes: Vec<RecordedPass>,
    encoder_open: bool,
    heap: Option<Heap>,
    indirect: Option<IndirectCommandBuffer>,
    recording_error: Option<Error>,
    status: CommandBufferStatus,
    failure: Option<Error>,
    submission: Option<ProviderSubmission>,
    completion: Option<CompletionToken>,
    pending: Option<PendingCompletion>,
}

/// Submitted work whose completion and host readback have not been observed.
/// The reservations keep CPU access blocked until finalization lands or the
/// command is dropped.
struct PendingCompletion {
    token: CompletionToken,
    trace: ComputeTrace,
    reservations: CommandReservations,
}

enum ExecutionOutcome {
    Completed(ProviderSubmission),
    Pending(ProviderSubmission, PendingCompletion),
}
struct CommandShared {
    owner: Arc<DeviceState>,
    inner: Mutex<CommandInner>,
    completion: Condvar,
}
impl Drop for CommandShared {
    fn drop(&mut self) {
        let inner = match self.inner.get_mut() {
            Ok(inner) => inner,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(token) = inner.completion {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                self.owner.provider.release_completion(token)
            }));
        }
    }
}

/// Single-use command buffer. A synchronous provider completes at commit; a
/// provider returning `Submitted` completes on the first `wait_until_completed`
/// after its readback passes exact-trace validation. [`CommandBuffer::cancel`]
/// abandons a pending observation without landing results. No buffer bytes
/// change if admission, execution, completion observation or readback
/// validation fails.
pub struct CommandBuffer {
    shared: Arc<CommandShared>,
}
impl CommandBuffer {
    pub fn status(&self) -> Result<CommandBufferStatus, Error> {
        Ok(lock(&self.shared.inner, "provider command")?.status)
    }

    /// Attach a heap to this command. Its placements are snapshotted at
    /// [`CommandBuffer::commit`]: the heap may only belong to this device, and
    /// a committed command refuses the set like any other recording action.
    pub fn set_heap(&self, heap: &Heap) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.shared.owner, &heap.inner.owner) {
            return Err(Error::ForeignHeap);
        }
        let mut inner = lock(&self.shared.inner, "provider command")?;
        if inner.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted.into());
        }
        inner.heap = Some(heap.clone());
        Ok(())
    }

    pub fn compute_command_encoder(&self) -> Result<ComputeCommandEncoder, Error> {
        let mut inner = lock(&self.shared.inner, "provider command")?;
        if inner.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted.into());
        }
        if inner.encoder_open {
            return Err(ApiError::EncoderAlreadyOpen.into());
        }
        inner.encoder_open = true;
        Ok(ComputeCommandEncoder {
            shared: Arc::clone(&self.shared),
            pipeline: None,
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
            dispatch_count: 0,
            indirect: false,
            ended: false,
        })
    }

    /// Open a render encoder on this command buffer.
    ///
    /// The encoder records the first increment's single render shape: one
    /// colour attachment, a covering viewport, one draw — the three-vertex
    /// full-screen triangle, a draw over bound vertex streams, or an indexed
    /// draw — and, optionally, one present tail action. It shares the command
    /// buffer's single open-encoder slot with the compute encoder, so a render
    /// pass can follow the declaring compute passes that reserved its
    /// attachment buffer.
    pub fn render_command_encoder(&self) -> Result<RenderCommandEncoder, Error> {
        let mut inner = lock(&self.shared.inner, "provider command")?;
        if inner.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted.into());
        }
        if inner.encoder_open {
            return Err(ApiError::EncoderAlreadyOpen.into());
        }
        inner.encoder_open = true;
        Ok(RenderCommandEncoder {
            shared: Arc::clone(&self.shared),
            pipeline: None,
            draw_count: 0,
            indirect: false,
            ended: false,
            scissor: None,
            vertex_buffers: BTreeMap::new(),
            index_buffer: None,
            fragment_textures: BTreeMap::new(),
            fragment_samplers: BTreeMap::new(),
            stage_buffers: BTreeMap::new(),
        })
    }

    pub fn commit(&self) -> Result<(), Error> {
        let (passes, heap, indirect) = {
            let mut inner = lock(&self.shared.inner, "provider command")?;
            if inner.status != CommandBufferStatus::Recording {
                return Err(ApiError::CommandBufferAlreadyCommitted.into());
            }
            if inner.encoder_open {
                return Err(ApiError::EncoderNotEnded.into());
            }
            if let Some(error) = inner.recording_error.clone() {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
                self.shared.completion.notify_all();
                return Err(error);
            }
            if inner.passes.is_empty() {
                return Err(ApiError::NoEncodedCommands.into());
            }
            inner.status = CommandBufferStatus::Committed;
            (
                inner.passes.clone(),
                inner.heap.clone(),
                inner.indirect.clone(),
            )
        };
        let mut token = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let textures = collect_textures(&passes);
            let reservations = CommandReservations {
                buffers: reserve_buffers(&passes)?,
                textures: reserve_textures(&textures)?,
            };
            self.execute(&passes, reservations, textures, heap, indirect, &mut token)
        }))
        .unwrap_or(Err(Error::ProviderPanicked));
        let mut inner = lock(&self.shared.inner, "provider command")?;
        inner.completion = token;
        let result = match result {
            Ok(ExecutionOutcome::Completed(submission)) => {
                inner.submission = Some(submission);
                inner.status = CommandBufferStatus::Completed;
                Ok(())
            }
            Ok(ExecutionOutcome::Pending(submission, pending)) => {
                inner.submission = Some(submission);
                inner.pending = Some(pending);
                // Status remains Committed until host-visible results land.
                Ok(())
            }
            Err(error) => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
                Err(error)
            }
        };
        self.shared.completion.notify_all();
        result
    }

    /// Cancel a committed command whose deferred completion has not landed.
    ///
    /// The provider releases its observation slot and the host reservations
    /// covering the committed buffers are dropped, so CPU access resumes
    /// without waiting for device retirement. The command becomes `Failed`
    /// with [`Error::CompletionUnavailable`] carrying `Cancelled`. Device work
    /// that was already submitted may still run; the provider reclaims its
    /// backing when the device retires it. A command whose result already
    /// landed, or whose provider refuses cancellation, cannot be cancelled and
    /// reports the corresponding error.
    ///
    /// A present tail action was performed at `submit` time: cancel does not
    /// roll back its acquire/present count or the target's terminal layout. It
    /// abandons only the observation and the landing of the writebacks.
    pub fn cancel(&self) -> Result<(), Error> {
        let pending = {
            let mut inner = lock(&self.shared.inner, "provider command")?;
            loop {
                match inner.status {
                    CommandBufferStatus::Recording => {
                        return Err(ApiError::CommandBufferNotCommitted.into())
                    }
                    CommandBufferStatus::Completed | CommandBufferStatus::Failed => {
                        return Err(inner
                            .failure
                            .clone()
                            .unwrap_or(ApiError::CommandBufferNotCompleted.into()))
                    }
                    CommandBufferStatus::Committed => {
                        if let Some(pending) = inner.pending.take() {
                            break pending;
                        }
                        inner = self
                            .shared
                            .completion
                            .wait(inner)
                            .map_err(|_| ApiError::StatePoisoned("provider command"))?;
                    }
                }
            }
        };
        let observed = provider_call(|| self.shared.owner.provider.cancel(pending.token));
        let mut inner = lock(&self.shared.inner, "provider command")?;
        let result = match observed {
            Ok(CompletionDisposition::Cancelled { token }) if token == pending.token => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(Error::CompletionUnavailable(
                    CompletionDisposition::Cancelled { token },
                ));
                Ok(())
            }
            Ok(other) => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(Error::CompletionUnavailable(other));
                Err(Error::CompletionUnavailable(other))
            }
            Err(error) => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
                Err(error)
            }
        };
        drop(pending);
        self.shared.completion.notify_all();
        result
    }

    fn execute(
        &self,
        passes: &[RecordedPass],
        reservations: CommandReservations,
        textures: Vec<Texture>,
        heap: Option<Heap>,
        indirect: Option<IndirectCommandBuffer>,
        token: &mut Option<CompletionToken>,
    ) -> Result<ExecutionOutcome, Error> {
        let owner = &self.shared.owner;
        let mut positions = BTreeMap::new();
        let mut resources = ResourceTableSnapshot::new();
        for (position, reservation) in reservations.buffers.iter().enumerate() {
            positions.insert(reservation.allocation_id(), position);
            resources.insert_allocation(AllocationRecord {
                allocation_id: reservation.allocation_id(),
                owner_epoch: owner.epoch,
                size: reservation.inner.length as u64,
            })?;
        }
        for texture in &textures {
            // A trace-view texture shares the producing view's allocation
            // (`research/docs/23` §110, E-TX3), and the buffer reservations
            // above have already registered it: the identity is one
            // allocation, so the existing record is the right one and a
            // second insert of it would be refused as a duplicate of itself.
            if resources.allocation(texture.inner.allocation_id).is_some() {
                continue;
            }
            resources.insert_allocation(AllocationRecord {
                allocation_id: texture.inner.allocation_id,
                owner_epoch: owner.epoch,
                size: u64::try_from(texture.inner.length).unwrap_or(u64::MAX),
            })?;
        }
        // A lease-bound stage buffer's allocation is the *owner's*
        // registration, so the trace's own table has to carry it before the
        // reservation can be admitted: `ResourceTableSnapshot::insert_lease`
        // resolves the lease's allocation, and a snapshot without it refuses
        // the reservation by name (`research/docs/23` §90, R9i). The owner's
        // registered length is the caller's own statement, exactly as the
        // suite's `allocation_size` is on the trace rail.
        for pass in passes {
            let RecordedPass::Render { target, .. } = pass else {
                continue;
            };
            for slot in &target.stage_buffers {
                let RenderStageSource::Lease(lease) = &slot.source else {
                    continue;
                };
                let allocation = lease.reservation.lease.allocation_id;
                if resources.allocation(allocation).is_none() {
                    resources.insert_allocation(AllocationRecord {
                        allocation_id: allocation,
                        owner_epoch: owner.epoch,
                        size: lease.allocation_size,
                    })?;
                }
                if resources.lease(lease.reservation.lease.lease_id).is_none() {
                    resources.insert_lease(lease.reservation)?;
                }
            }
        }
        // A lease-bound compute binding's allocation is the owner's own
        // registration, exactly as a lease-bound stage buffer's is: the
        // trace's own table has to carry it before the reservation can be
        // admitted (`research/docs/23` §90, R9i/E-TX9b). The owner's
        // registered length is the caller's own statement.
        for pass in passes {
            let RecordedPass::Compute { buffers, .. } = pass else {
                continue;
            };
            for source in buffers.values() {
                let ComputeBindingSource::Lease(lease) = source else {
                    continue;
                };
                let allocation = lease.reservation.lease.allocation_id;
                if resources.allocation(allocation).is_none() {
                    resources.insert_allocation(AllocationRecord {
                        allocation_id: allocation,
                        owner_epoch: owner.epoch,
                        size: lease.allocation_size,
                    })?;
                }
                if resources.lease(lease.reservation.lease.lease_id).is_none() {
                    resources.insert_lease(lease.reservation)?;
                }
            }
        }
        // The host bytes must stay stable only while the trace snapshots them:
        // every view copies its bytes into the trace, so the provider never
        // reads the host buffer again. Conflicting CPU access is excluded for
        // the whole commit-to-completion window by the range reservations, and
        // `lock_unreserved` re-checks them under these guards, so releasing the
        // guards before `submit` cannot admit a conflicting CPU write. It does
        // let a sibling command with a disjoint range of the same allocation
        // take its own snapshot while this one is still inside `submit`.
        let mut guards = Vec::with_capacity(reservations.buffers.len());
        for reservation in &reservations.buffers {
            guards.push(reservation.lock_bytes()?);
        }
        let mut pipelines = BTreeMap::<PipelineId, CompiledComputePipeline>::new();
        let mut trace_passes = Vec::with_capacity(passes.len());
        for pass in passes {
            match pass {
                RecordedPass::Compute {
                    pipeline,
                    buffers,
                    textures,
                    dispatch,
                } => {
                    let metadata = pipeline.metadata();
                    if let Some(previous) = pipelines.insert(metadata.pipeline_id, metadata.clone())
                    {
                        if previous != *metadata {
                            return Err(Error::InvalidPipelineMetadata);
                        }
                    }
                    let mut views = Vec::with_capacity(buffers.len());
                    for (binding, source) in buffers {
                        let reflected = metadata
                            .contract
                            .buffer_bindings
                            .iter()
                            .find(|value| value.metal_binding == *binding)
                            .ok_or(ContractError::UnknownBinding(*binding))?;
                        let view = match source {
                            ComputeBindingSource::View(view) => {
                                let bytes = &guards[positions[&view.allocation_id()]];
                                contract::BufferView {
                                    view_id: view.view_id,
                                    metal_binding: *binding,
                                    allocation_id: view.allocation_id(),
                                    offset: view.offset as u64,
                                    length: view.length as u64,
                                    access: reflected.access,
                                    attribute_stride: None,
                                    source: BufferSource::OwnedBytes(
                                        bytes[view.offset..view.offset + view.length].to_vec(),
                                    ),
                                }
                            }
                            ComputeBindingSource::Lease(lease) => contract::BufferView {
                                // The lease's own identity is the view's id,
                                // the rule the render stage-buffer arm states:
                                // the reservation names the bytes, so the view
                                // cannot name a different set (`research/docs/23`
                                // §90, R9i; §114, E-TX9b).
                                view_id: ViewId::new(lease.reservation.lease.lease_id.get()),
                                metal_binding: *binding,
                                allocation_id: lease.reservation.lease.allocation_id,
                                offset: lease.reservation.offset,
                                length: lease.reservation.length,
                                access: reflected.access,
                                attribute_stride: None,
                                source: match lease.arm {
                                    StageBufferLeaseArm::StagedLease => {
                                        BufferSource::StagedLease(lease.reservation.lease.lease_id)
                                    }
                                    StageBufferLeaseArm::BorrowedNoCopy => {
                                        BufferSource::BorrowedNoCopy(
                                            lease.reservation.lease.lease_id,
                                        )
                                    }
                                },
                            },
                        };
                        views.push(view);
                    }
                    trace_passes.push(contract::TracePass::Compute(contract::ComputePass {
                        pipeline: metadata.pipeline_id,
                        buffers: views,
                        dispatch: *dispatch,
                        textures: textures
                            .iter()
                            .map(|(binding, texture)| texture.view(*binding))
                            .collect::<Result<Vec<_>, _>>()?,
                    }));
                }
                RecordedPass::Render { pipeline, target } => {
                    let metadata = pipeline.metadata();
                    if let Some(previous) = pipelines.insert(metadata.pipeline_id, metadata.clone())
                    {
                        if previous != *metadata {
                            return Err(Error::InvalidPipelineMetadata);
                        }
                    }
                    // A draw input carries its own bytes, so the pass the trace
                    // keeps is the one holding the host bytes this snapshot
                    // takes. The reservations above hold every range the draw
                    // reads, so the bytes are the ones the caller had when the
                    // command committed — the same boundary a compute binding's
                    // bytes are taken at (`research/docs/23` §3.6).
                    let input_bytes = |view: &BufferView| -> Vec<u8> {
                        let bytes = &guards[positions[&view.allocation_id()]];
                        bytes[view.offset..view.offset + view.length].to_vec()
                    };
                    trace_passes.push(contract::TracePass::Render(
                        target.descriptor(metadata.pipeline_id, &input_bytes)?,
                    ));
                }
            }
        }
        // Snapshot complete: no later step of this command reads the host bytes.
        drop(guards);
        // The owner-window store resolves its landing from the trace's own
        // serial view list (`research/docs/23` §114, E-TX8), so the object rail
        // states here that the command's own declarations carry the window its
        // frame lands in. A landing no pass declares is refused by name — the
        // same `(allocation, view)` identity the rail looks up, answered while
        // both halves are still one trace — instead of being deferred to the
        // rail, where it would read as a missing declaration
        // (`research/docs/23` §115, E-TX9b).
        let declared_windows = trace_passes
            .iter()
            .filter_map(contract::TracePass::as_compute)
            .flat_map(|pass| pass.buffers.iter())
            .filter(|view| {
                matches!(
                    view.source,
                    BufferSource::BorrowedNoCopy(_) | BufferSource::GuestRuns(_)
                )
            })
            .map(|view| (view.view_id, view.allocation_id))
            .collect::<BTreeSet<_>>();
        for pass in trace_passes
            .iter()
            .filter_map(contract::TracePass::as_render)
        {
            for attachment in &pass.color_attachments {
                // Which declaration holds the window is the store arm's own
                // answer (`research/docs/23` §115 及其后的增量，E-TX8/E-TX13):
                // the attachment's own identity for `Borrowed`, the second view
                // the landing arm carries for `BorrowedLanding`. Either way the
                // identity has to appear in the trace's own declaration list,
                // and the refusal names the identity that is missing.
                let Some(landing) = attachment.landing_identity() else {
                    continue;
                };
                if !declared_windows.contains(&(landing.view_id, landing.allocation_id)) {
                    return Err(Error::WindowAttachmentUndeclared {
                        lease: LeaseId::new(landing.view_id.get()),
                        view: landing.view_id,
                    });
                }
            }
        }
        let heap_payload = heap.and_then(|heap| heap.payload()).map(Box::new);
        let indirect_payload = indirect.map(|icb| Box::new(icb.payload().clone()));
        let trace = contract::ComputeTrace {
            schema_version: PROVIDER_SCHEMA_VERSION,
            device_epoch: owner.epoch,
            operation_id: OperationId::new(next_id()?),
            pipelines: pipelines.into_values().collect(),
            encoder_dispatch_type: DispatchType::Serial,
            passes: trace_passes,
            completion_policy: CompletionPolicy::HostReadback,
            heap: heap_payload,
            indirect: indirect_payload,
        };
        let admitted = owner
            .capabilities
            .validate_trace(trace.clone(), resources)?;
        let result = provider_call(|| owner.provider.submit(admitted));
        let observed_token = match &result {
            Ok(value) => value.completion.token(),
            Err(Error::Provider(error)) => error.completion.token(),
            _ => None,
        };
        *token = observed_token
            .filter(|token| token.device_epoch == owner.epoch && token.validate().is_ok());
        let submission = result?;
        submission.validate_for_trace(&trace)?;
        match submission.completion {
            CompletionDisposition::CompletedVisible { token: completed } => {
                if token.as_ref() != Some(&completed) {
                    return Err(Error::CompletionObservationMismatch);
                }
                let observed = provider_call(|| owner.provider.wait(completed, Duration::ZERO))?;
                if observed != submission.completion {
                    return Err(Error::CompletionObservationMismatch);
                }
                apply_writebacks(
                    &reservations.buffers,
                    &reservations.textures,
                    &submission.writebacks,
                    &lease_bound_identities(&trace),
                )?;
                Ok(ExecutionOutcome::Completed(submission))
            }
            CompletionDisposition::Submitted { token: submitted } => {
                if token.as_ref() != Some(&submitted) {
                    return Err(Error::CompletionObservationMismatch);
                }
                Ok(ExecutionOutcome::Pending(
                    submission,
                    PendingCompletion {
                        token: submitted,
                        trace,
                        reservations,
                    },
                ))
            }
            _ => Err(Error::CompletionUnavailable(submission.completion)),
        }
    }

    pub fn wait_until_completed(&self) -> Result<(), Error> {
        let pending = {
            let mut inner = lock(&self.shared.inner, "provider command")?;
            loop {
                match inner.status {
                    CommandBufferStatus::Recording => {
                        return Err(ApiError::CommandBufferNotCommitted.into())
                    }
                    CommandBufferStatus::Completed => return Ok(()),
                    CommandBufferStatus::Failed => {
                        return Err(inner
                            .failure
                            .clone()
                            .unwrap_or(ApiError::CommandBufferNotCompleted.into()))
                    }
                    CommandBufferStatus::Committed => {
                        if let Some(pending) = inner.pending.take() {
                            break pending;
                        }
                        inner = self
                            .shared
                            .completion
                            .wait(inner)
                            .map_err(|_| ApiError::StatePoisoned("provider command"))?;
                    }
                }
            }
        };
        let result = catch_unwind(AssertUnwindSafe(|| self.finalize(&pending)))
            .unwrap_or(Err(Error::ProviderPanicked));
        let mut inner = lock(&self.shared.inner, "provider command")?;
        let result = match result {
            Ok(submission) => {
                inner.submission = Some(submission);
                inner.status = CommandBufferStatus::Completed;
                Ok(())
            }
            Err(error) => {
                inner.status = CommandBufferStatus::Failed;
                inner.failure = Some(error.clone());
                Err(error)
            }
        };
        self.shared.completion.notify_all();
        result
    }

    /// Observe a pending token, validate readback against the exact submitted
    /// trace and land every writeback before releasing reservations.
    fn finalize(&self, pending: &PendingCompletion) -> Result<ProviderSubmission, Error> {
        let owner = &self.shared.owner;
        let mut timeout = Duration::from_millis(10);
        loop {
            let observed = provider_call(|| owner.provider.wait(pending.token, timeout))?;
            match observed {
                CompletionDisposition::CompletedVisible { token } if token == pending.token => {
                    break
                }
                CompletionDisposition::TimedOut { token } if token == pending.token => {
                    timeout = timeout.saturating_mul(2).min(Duration::from_millis(100));
                }
                CompletionDisposition::Failed { .. }
                | CompletionDisposition::DeviceLost { .. }
                | CompletionDisposition::Cancelled { .. }
                | CompletionDisposition::SubmittedUnknown { .. } => {
                    return Err(Error::CompletionUnavailable(observed))
                }
                _ => return Err(Error::CompletionObservationMismatch),
            }
        }
        let readback = provider_call(|| owner.provider.readback(pending.token))?;
        readback.validate_for_trace(&pending.trace)?;
        match readback.completion {
            CompletionDisposition::CompletedVisible { token } if token == pending.token => {}
            _ => return Err(Error::CompletionObservationMismatch),
        }
        apply_writebacks(
            &pending.reservations.buffers,
            &pending.reservations.textures,
            &readback.writebacks,
            &lease_bound_identities(&pending.trace),
        )?;
        Ok(ProviderSubmission {
            completion: readback.completion,
            writebacks: readback.writebacks,
        })
    }

    pub fn submission(&self) -> Result<ProviderSubmission, Error> {
        let inner = lock(&self.shared.inner, "provider command")?;
        inner.submission.clone().ok_or_else(|| {
            inner
                .failure
                .clone()
                .unwrap_or(ApiError::CommandBufferNotCompleted.into())
        })
    }
}

/// Where one recorded compute slot's bytes come from (`research/docs/23` §90,
/// R9i; §114, E-TX9b).
///
/// The two arms a dispatch can name: the caller's own host image, snapshotted
/// at commit under the command's reservations, or a lease the provider
/// registry already holds. The lease arm is the same one a render stage buffer
/// has ([`RenderStageSource`]), and it exists on the compute rail for the
/// owner-window store's own reason: the window a `StoreOp::Borrowed`
/// attachment's frame lands in is resolved from the trace's *serial view
/// list*, so a command that lands a frame in the owner's pages declares that
/// window with a pass — and the pass that declares an input the render pass
/// also names is the compute one.
///
/// The lease's own identity is the view's id, exactly as it is on the render
/// stage-buffer arm: the reservation names the bytes, so the view cannot name
/// a different set.
#[derive(Clone)]
enum ComputeBindingSource {
    /// A caller-held buffer view, snapshotted at commit.
    View(BufferView),
    /// An imported lease the provider registry already holds.
    Lease(StageBufferLease),
}

impl ComputeBindingSource {
    fn view_id(&self) -> ViewId {
        match self {
            Self::View(view) => view.view_id(),
            Self::Lease(lease) => ViewId::new(lease.reservation.lease.lease_id.get()),
        }
    }

    /// Byte interval of this binding inside its allocation, for range hazards.
    /// A lease-bound slot's interval is its reservation, since that is the
    /// window the provider reads.
    fn range(&self) -> BufferRange {
        match self {
            Self::View(view) => view.range(),
            Self::Lease(lease) => BufferRange::new(
                lease.reservation.lease.allocation_id,
                lease.reservation.offset,
                lease.reservation.length,
            ),
        }
    }
}

/// Encoder state persists across dispatches. Call `clear_buffers` when changing
/// to a pipeline with a different layout; extra bindings are refused.
pub struct ComputeCommandEncoder {
    shared: Arc<CommandShared>,
    pipeline: Option<Pipeline>,
    buffers: BTreeMap<u32, ComputeBindingSource>,
    textures: BTreeMap<u32, Texture>,
    dispatch_count: usize,
    indirect: bool,
    ended: bool,
}
impl ComputeCommandEncoder {
    pub fn set_compute_pipeline_state(&mut self, pipeline: &Pipeline) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &pipeline.inner.owner) {
            return Err(ApiError::ForeignPipeline.into());
        }
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }
    pub fn set_buffer(&mut self, index: u32, view: &BufferView) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &view.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        self.bind_buffer_source(index, ComputeBindingSource::View(view.clone()))
    }

    /// Bind one imported lease to `index` (`research/docs/23` §90, R9i; §114,
    /// E-TX9b).
    ///
    /// The lease is the caller's own import: it goes through the same channel a
    /// compute case's `storage_mode` uses, and the reservation the caller hands
    /// here is the one the provider's registry resolved. The object API does
    /// not import anything itself, and the owner's window stays the caller's to
    /// keep alive until the submission has been waited for.
    ///
    /// The lease's own identity is the view's id — the rule the render
    /// stage-buffer arm states — so a pass that names the owner's window here
    /// declares exactly the `(allocation, view)` pair a
    /// [`RenderWindowAttachment`] of the same command lands its frame in. The
    /// window a `StoreOp::Borrowed` attachment resolves is read from the
    /// trace's *serial view list*, and this is the object rail's channel into
    /// that list.
    ///
    /// A reservation whose range is not inside the owner allocation the caller
    /// states is refused by name ([`ContractError::LeaseRangeOutOfBounds`]),
    /// exactly as the trace's own resource table refuses it. A slot the
    /// pipeline declares *writable* is
    /// [`Error::WritableComputeLeaseUnsupported`], refused when the pass is
    /// recorded because only the pipeline's own contract states the slot's
    /// access.
    pub fn set_buffer_lease(&mut self, index: u32, lease: StageBufferLease) -> Result<(), Error> {
        self.ensure_open()?;
        if lease.allocation_size == 0 {
            return Err(ContractError::ZeroLength("lease allocation").into());
        }
        let lease_id = lease.reservation.lease.lease_id;
        let end = lease.reservation.end()?;
        if end > lease.allocation_size {
            return Err(ContractError::LeaseRangeOutOfBounds {
                lease: lease_id,
                end,
                allocation_size: lease.allocation_size,
            }
            .into());
        }
        self.bind_buffer_source(index, ComputeBindingSource::Lease(lease))
    }

    /// The two refusals every compute binding shares: one index holds one
    /// source, and two bindings of one allocation may not alias.
    fn bind_buffer_source(
        &mut self,
        index: u32,
        source: ComputeBindingSource,
    ) -> Result<(), Error> {
        let range = source.range();
        if let Some((first, _)) = self
            .buffers
            .iter()
            .find(|(other, bound)| **other != index && bound.range().overlaps(&range))
        {
            return Err(ApiError::AliasedBufferBindings {
                first: *first,
                second: index,
            }
            .into());
        }
        self.buffers.insert(index, source);
        Ok(())
    }

    /// Refuse a lease-bound slot the pipeline's own contract declares writable
    /// (`research/docs/23` §90, R9i/E-TX9b).
    ///
    /// The access of a slot is the recorded pipeline's declaration — the pair
    /// rule [`ComputeCommandEncoder::dispatch_threads`] already enforces
    /// between the bindings and the contract — so this is where the answer is
    /// known. A writable slot's landing is the bytes' own host image, and a
    /// lease has none.
    fn refuse_writable_lease_slots(&self, pipeline: &Pipeline) -> Result<(), Error> {
        for (index, source) in &self.buffers {
            if !matches!(source, ComputeBindingSource::Lease(_)) {
                continue;
            }
            let writable = pipeline
                .metadata()
                .contract
                .buffer_bindings
                .iter()
                .find(|slot| slot.metal_binding == *index)
                .is_some_and(|slot| slot.access != BufferAccess::Read);
            if writable {
                return Err(Error::WritableComputeLeaseUnsupported { index: *index });
            }
        }
        Ok(())
    }
    pub fn clear_buffers(&mut self) -> Result<(), Error> {
        self.ensure_open()?;
        self.buffers.clear();
        Ok(())
    }
    /// Bind one sampled texture to a Metal argument index. A texture and a
    /// buffer may share an index because the translator reports them in one
    /// argument namespace (`research/docs/16` §4.7).
    pub fn set_texture(&mut self, index: u32, texture: &Texture) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &texture.inner.owner) {
            return Err(Error::ForeignTexture);
        }
        // A trace-view texture is the render sampler's arm
        // (`research/docs/23` §110, E-TX3): its bytes exist only as the
        // trace's own production, and a compute binding would have to state
        // how that production reaches the compute rail's image upload. This
        // increment does not, so it is refused by name here.
        if texture.inner.origin == TextureOrigin::TraceView {
            return Err(Error::TraceViewTextureIsNotAComputeBinding {
                view: texture.inner.view_id,
            });
        }
        self.textures.insert(index, texture.clone());
        Ok(())
    }
    pub fn clear_textures(&mut self) -> Result<(), Error> {
        self.ensure_open()?;
        self.textures.clear();
        Ok(())
    }
    pub fn dispatch_threads(&mut self, grid: Size, local: Size) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let pipeline = self.pipeline.as_ref().ok_or(ApiError::MissingPipeline)?;
        for slot in &pipeline.metadata().contract.buffer_bindings {
            if !self.buffers.contains_key(&slot.metal_binding) {
                return Err(ContractError::MissingBinding(slot.metal_binding).into());
            }
        }
        for binding in self.buffers.keys() {
            if !pipeline
                .metadata()
                .contract
                .buffer_bindings
                .iter()
                .any(|slot| slot.metal_binding == *binding)
            {
                return Err(ContractError::UnknownBinding(*binding).into());
            }
        }
        // The texture face gets the same two directions the buffer face does,
        // at the same recorder boundary (`research/docs/26` §21.3, step 1): a
        // declaration the pass never bound would leave a descriptor the module
        // reads undefined, and a bound texture the declaration never named
        // would fill a slot the module said nothing about. Access, shape and
        // format stay with the pass's own admission, exactly as buffer access
        // does.
        for slot in &pipeline.metadata().contract.texture_bindings {
            if !self.textures.contains_key(&slot.metal_binding) {
                return Err(ContractError::MissingTextureBinding {
                    binding: slot.metal_binding,
                }
                .into());
            }
        }
        for binding in self.textures.keys() {
            if !pipeline
                .metadata()
                .contract
                .texture_bindings
                .iter()
                .any(|slot| slot.metal_binding == *binding)
            {
                return Err(ContractError::UndeclaredTextureBinding { binding: *binding }.into());
            }
        }
        self.refuse_writable_lease_slots(pipeline)?;
        let mut inner = lock(&self.shared.inner, "provider command")?;
        let maximum = usize::try_from(self.shared.owner.capabilities.max_passes)
            .unwrap_or(usize::MAX)
            .min(8);
        if inner.passes.len() >= maximum {
            return Err(Error::PassLimit {
                requested: inner.passes.len() + 1,
                maximum,
            });
        }
        let mut unique = recorded_view_ids(&inner.passes);
        unique.extend(self.buffers.values().map(ComputeBindingSource::view_id));
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass::Compute {
            pipeline: pipeline.clone(),
            buffers: self.buffers.clone(),
            textures: self.textures.clone(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: grid.dimensions().map(u64::from),
                threads_per_threadgroup: local.dimensions().map(u64::from),
            },
        });
        self.dispatch_count += 1;
        Ok(())
    }

    /// Replay one indirect dispatch from an encoded command instead of a direct
    /// [`Self::dispatch_threads`]. The replayed threadgroups are the ICB's own;
    /// `grid` and `local` restate the pass the footprint proofs describe, so
    /// the provider can check the replay's group count against the planned one
    /// (`research/docs/25` §6 Step 4).
    pub fn dispatch_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        grid: Size,
        local: Size,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &icb.inner.owner) {
            return Err(Error::ForeignIndirectCommandBuffer);
        }
        if icb.command_kind() != IndirectCommandKind::Dispatch {
            return Err(Error::IndirectKindMismatch {
                accepted: &COMPUTE_DISPATCH_KINDS,
                actual: icb.command_kind(),
            });
        }
        if self.dispatch_count > 0 || self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let pipeline = self.pipeline.as_ref().ok_or(ApiError::MissingPipeline)?;
        for slot in &pipeline.metadata().contract.buffer_bindings {
            if !self.buffers.contains_key(&slot.metal_binding) {
                return Err(ContractError::MissingBinding(slot.metal_binding).into());
            }
        }
        for binding in self.buffers.keys() {
            if !pipeline
                .metadata()
                .contract
                .buffer_bindings
                .iter()
                .any(|slot| slot.metal_binding == *binding)
            {
                return Err(ContractError::UnknownBinding(*binding).into());
            }
        }
        // The indirect replay's binding table is the encoder's, exactly as a
        // direct dispatch's is, so the texture face is paired here too.
        for slot in &pipeline.metadata().contract.texture_bindings {
            if !self.textures.contains_key(&slot.metal_binding) {
                return Err(ContractError::MissingTextureBinding {
                    binding: slot.metal_binding,
                }
                .into());
            }
        }
        for binding in self.textures.keys() {
            if !pipeline
                .metadata()
                .contract
                .texture_bindings
                .iter()
                .any(|slot| slot.metal_binding == *binding)
            {
                return Err(ContractError::UndeclaredTextureBinding { binding: *binding }.into());
            }
        }
        self.refuse_writable_lease_slots(pipeline)?;
        let mut inner = lock(&self.shared.inner, "provider command")?;
        if inner.indirect.is_some() {
            return Err(Error::IndirectAlreadyRecorded);
        }
        let maximum = usize::try_from(self.shared.owner.capabilities.max_passes)
            .unwrap_or(usize::MAX)
            .min(8);
        if inner.passes.len() >= maximum {
            return Err(Error::PassLimit {
                requested: inner.passes.len() + 1,
                maximum,
            });
        }
        let mut unique = recorded_view_ids(&inner.passes);
        unique.extend(self.buffers.values().map(ComputeBindingSource::view_id));
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass::Compute {
            pipeline: pipeline.clone(),
            buffers: self.buffers.clone(),
            textures: self.textures.clone(),
            dispatch: Dispatch {
                kind: DispatchKind::ThreadsExact,
                grid: grid.dimensions().map(u64::from),
                threads_per_threadgroup: local.dimensions().map(u64::from),
            },
        });
        inner.indirect = Some(icb.clone());
        self.dispatch_count += 1;
        self.indirect = true;
        Ok(())
    }
    pub fn end_encoding(mut self) -> Result<(), Error> {
        self.ensure_open()?;
        let mut inner = lock(&self.shared.inner, "provider command")?;
        let result = if self.dispatch_count == 0 {
            Err(Error::Api(ApiError::MissingDispatch))
        } else {
            Ok(())
        };
        inner.encoder_open = false;
        if let Err(error) = &result {
            inner.recording_error = Some(error.clone());
        }
        self.ended = true;
        result
    }
    fn ensure_open(&self) -> Result<(), Error> {
        if self.ended {
            return Err(ApiError::EncoderAlreadyEnded.into());
        }
        if lock(&self.shared.inner, "provider command")?.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted.into());
        }
        Ok(())
    }
}
impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        if !self.ended {
            if let Ok(mut inner) = self.shared.inner.lock() {
                inner.encoder_open = false;
                inner.recording_error = Some(ApiError::EncoderNotEnded.into());
            }
        }
    }
}

/// Records the first increment's render shape on one command buffer.
///
/// [`RenderCommandEncoder`] is the render sibling of [`ComputeCommandEncoder`]:
/// it persists a pipeline selection across draws, refuses foreign objects, and
/// hands the recorded pass to the command buffer's commit-time reservation and
/// submission exactly as the compute encoder does. One draw records one
/// colour-attachment render pass with the covering viewport and an optional
/// present tail. That present tail executes when the command is submitted: its
/// acquire/present count and target terminal layout are not rolled back by a
/// later `cancel` or deadline.
///
/// The encoder also persists vertex and index bindings the way Metal's
/// `MTLRenderCommandEncoder` does (`setVertexBuffer(_:offset:index:)` /
/// `setIndexBuffer`): a direct draw records the bound streams and index buffer
/// in its pass, and an indirect replay refuses an encoder that bound them,
/// because the replay reads its draw input from the ICB's own payload.
pub struct RenderCommandEncoder {
    shared: Arc<CommandShared>,
    pipeline: Option<RenderPipeline>,
    draw_count: usize,
    indirect: bool,
    ended: bool,
    /// Bound vertex streams by binding index. The map's ascending keys are the
    /// pass's binding order, which the descriptor makes positional.
    vertex_buffers: BTreeMap<u32, BufferView>,
    /// The one index buffer this encoder draws through, with the width of the
    /// indices it holds.
    index_buffer: Option<(BufferView, IndexFormat)>,
    /// The sampled textures the fragment stage reads, by binding index
    /// (`research/docs/23` §3.3, v70). Like the vertex streams, a binding is
    /// direct-draw state that travels with the pass the draw records.
    fragment_textures: BTreeMap<u32, Texture>,
    /// The runtime sampler states the fragment stage executes with, by Metal
    /// `[[sampler(n)]]` index (`research/docs/23` §3.3, v102). Like the
    /// textures, a state is direct-draw state that travels with the pass the
    /// draw records; the map's ascending keys are its canonical order.
    fragment_samplers: BTreeMap<u32, SamplerPolicy>,
    /// The stage buffers the two stages read — and the writable ones they land
    /// — by `(stage ordinal, index)` (`research/docs/23` §3.3, v87). The key's
    /// first half is the stage's own ordinal, so the map's iteration order is
    /// the contract's canonical one (vertex before fragment, ascending index)
    /// without a second list that could disagree with it.
    stage_buffers: BTreeMap<(u8, u32), RenderStageBinding>,
    /// The scissor rectangle every pass recorded afterwards clips to, or `None`
    /// for the whole attachment (`research/docs/23` §3.3, v29/v30). Like
    /// Metal's own encoder state it applies to the draws that follow the call.
    scissor: Option<[u32; 4]>,
}

/// One stage-buffer binding the encoder holds (`research/docs/23` §3.3, v87).
///
/// The stage is kept beside the source because the map's key carries its
/// ordinal rather than the enum: the ordinal is what orders the map, and the
/// enum is what the descriptor states.
#[derive(Clone)]
struct RenderStageBinding {
    stage: RenderPipelineStage,
    source: RenderStageSource,
}
impl RenderCommandEncoder {
    /// Clip every draw recorded afterwards to `rect` (`[x, y, width, height]`
    /// in framebuffer coordinates), or clear the clipping with `None`.
    ///
    /// The rectangle is encumbrance-free encoder state: it is validated against
    /// the attachment's extent when the draw is recorded (the contract's
    /// [`ContractError::ScissorOutOfBounds`]) and then travels into the pass's
    /// own descriptor, so both rails clip the same rectangle
    /// (`research/docs/23` §3.3, v29/v30).
    pub fn set_scissor(&mut self, rect: Option<[u32; 4]>) -> Result<(), Error> {
        self.ensure_open()?;
        self.scissor = rect;
        Ok(())
    }

    pub fn set_render_pipeline_state(&mut self, pipeline: &RenderPipeline) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &pipeline.inner.owner) {
            return Err(ApiError::ForeignPipeline.into());
        }
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }

    /// Bind one vertex stream to `index` for the direct draws that follow.
    ///
    /// `index` is the binding the pipeline's vertex layout names, and the pass's
    /// descriptor carries the bound streams in ascending index order
    /// (`RenderPassDescriptor::vertex_buffers` is positional, and the entry a
    /// stream lands at also carries this index as its own `metal_binding`). The
    /// checks run in the order a heap placement's do — ownership, then the
    /// binding the encoder already holds — because each one is answerable here,
    /// without a provider and without a pass to attach it to: a buffer from
    /// another device is [`Error::ForeignBuffer`], a repeated index is
    /// [`Error::VertexBufferAlreadyBound`], and an index at or past
    /// [`MAX_VERTEX_BUFFERS`] is [`Error::VertexBufferIndexOutOfRange`].
    ///
    /// The stream's bytes are not copied here: they travel with the pass the
    /// draw records, taken at commit under the command's reservations, exactly
    /// as a compute binding's are.
    ///
    /// A binding is direct-draw state, so it is refused once an indirect replay
    /// has been recorded, exactly as [`Self::draw_render_pass`] is.
    pub fn set_vertex_buffer(&mut self, index: u32, buffer: &BufferView) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if !Arc::ptr_eq(&self.shared.owner, &buffer.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        if self.vertex_buffers.contains_key(&index) {
            return Err(Error::VertexBufferAlreadyBound { index });
        }
        if usize::try_from(index).unwrap_or(usize::MAX) >= MAX_VERTEX_BUFFERS {
            return Err(Error::VertexBufferIndexOutOfRange {
                index,
                maximum: MAX_VERTEX_BUFFERS,
            });
        }
        self.vertex_buffers.insert(index, buffer.clone());
        Ok(())
    }

    /// Bind the index buffer the indexed direct draws select through.
    ///
    /// One index buffer per encoder, exactly as the pass it records holds one
    /// `indices` binding: the descriptor carries this view's identity and
    /// `format`. A buffer from another device is [`Error::ForeignBuffer`] and a
    /// second binding is [`Error::IndexBufferAlreadyBound`]; like a vertex
    /// stream, an index buffer is direct-draw state and is refused once an
    /// indirect replay has been recorded.
    pub fn set_index_buffer(
        &mut self,
        buffer: &BufferView,
        format: IndexFormat,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if !Arc::ptr_eq(&self.shared.owner, &buffer.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        if self.index_buffer.is_some() {
            return Err(Error::IndexBufferAlreadyBound);
        }
        self.index_buffer = Some((buffer.clone(), format));
        Ok(())
    }

    /// Bind one sampled texture the fragment stage reads at `index`
    /// (`research/docs/23` §3.3, v70/v104).
    ///
    /// `index` is the fragment stage's own `[[texture(index)]]` argument — the
    /// same index the pipeline's declaration names and the pass's texture list
    /// carries, canonical order and all — so the bindings a recorder holds need
    /// not start at zero: the census's `[[texture(3)]]` shape binds one texture
    /// and nothing below it. A texture from another device is
    /// [`Error::ForeignTexture`]. A repeated index is
    /// [`Error::FragmentTextureAlreadyBound`] and an index at or past
    /// [`MAX_RENDER_TEXTURE_INDEX`] is
    /// [`Error::FragmentTextureIndexOutOfRange`], exactly as the vertex
    /// streams' two refusals spell their own binding.
    ///
    /// Like a compute binding, the texture's bytes are not copied here: the
    /// view the pass carries takes them at commit, under the command's own
    /// reservations, so a host write between recording and commit is refused
    /// by the same hazard rule every other binding follows. A binding is
    /// direct-draw state, so it is refused once an indirect replay has been
    /// recorded, exactly as [`Self::set_vertex_buffer`] is.
    pub fn set_fragment_texture(&mut self, index: u32, texture: &Texture) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if !Arc::ptr_eq(&self.shared.owner, &texture.inner.owner) {
            return Err(Error::ForeignTexture);
        }
        if self.fragment_textures.contains_key(&index) {
            return Err(Error::FragmentTextureAlreadyBound { index });
        }
        if index >= MAX_RENDER_TEXTURE_INDEX {
            return Err(Error::FragmentTextureIndexOutOfRange {
                index,
                maximum: MAX_RENDER_TEXTURE_INDEX as usize,
            });
        }
        self.fragment_textures.insert(index, texture.clone());
        Ok(())
    }

    /// Bind one runtime sampler state the fragment stage executes with at
    /// `index` (`research/docs/23` §3.3, v102).
    ///
    /// This is Metal's `setFragmentSamplerState(_:index:)`: the state a
    /// `[[sampler(index)]]` argument of the recorded pipeline's fragment stage
    /// is executed with. The value travels into the pass's sampler list, which
    /// the pipeline contract's own declarations pair with their textures — a
    /// state no declaration pairs with, or a declared runtime sampler left
    /// unbound, is refused when the pass is recorded by the contract's
    /// [`ContractError::UnpairedRuntimeSamplerBinding`] /
    /// [`ContractError::MissingRuntimeSamplerBinding`].
    ///
    /// A repeated index is [`Error::FragmentSamplerAlreadyBound`] and an index
    /// at or past [`MAX_RENDER_SAMPLERS`] is
    /// [`Error::FragmentSamplerIndexOutOfRange`], exactly as the texture
    /// binding's two refusals spell their own binding. A binding is direct-draw
    /// state, so it is refused once an indirect replay has been recorded.
    pub fn set_fragment_sampler_state(
        &mut self,
        index: u32,
        policy: SamplerPolicy,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.fragment_samplers.contains_key(&index) {
            return Err(Error::FragmentSamplerAlreadyBound { index });
        }
        if usize::try_from(index).unwrap_or(usize::MAX) >= MAX_RENDER_SAMPLERS {
            return Err(Error::FragmentSamplerIndexOutOfRange {
                index,
                maximum: MAX_RENDER_SAMPLERS,
            });
        }
        self.fragment_samplers.insert(index, policy);
        Ok(())
    }

    /// Bind one caller-held view to `stage`'s `[[buffer(index)]]` slot
    /// (`research/docs/23` §3.3, v83-v87).
    ///
    /// The encoder states the slot and the bytes; *how* the stage uses them —
    /// a read, a write, or both — is the recorded pipeline's own declaration,
    /// exactly as Metal's `setBuffer(_:offset:index:)` states no access and the
    /// function's argument does. A slot the recorded pipeline does not declare
    /// is refused when the pass is recorded, by the contract's own
    /// [`ContractError::UndeclaredStageBufferBinding`].
    ///
    /// `index` is the binding inside that stage's own namespace — two stages'
    /// index spaces overlap, which is why the stage is part of the slot — and
    /// a view from another device is [`Error::ForeignBuffer`]. A repeated slot
    /// is [`Error::StageBufferAlreadyBound`], an index at or past
    /// [`MAX_RENDER_STAGE_BUFFER_INDEX`] is
    /// [`ContractError::RenderStageBufferIndexExceeded`], and a binding past
    /// the pass's own [`MAX_RENDER_STAGE_BUFFERS`] ceiling is the contract's
    /// [`ContractError::RenderStageBufferLimitExceeded`].
    ///
    /// Like a vertex stream, the view's bytes are not copied here: a
    /// caller-held slot's bytes are taken once at commit, under the command's
    /// own reservations, so a host write between recording and commit is
    /// refused by the same hazard rule every other binding follows. A binding
    /// is direct-draw state, so it is refused once an indirect replay has been
    /// recorded, exactly as [`Self::set_vertex_buffer`] is.
    pub fn set_stage_buffer(
        &mut self,
        stage: RenderPipelineStage,
        index: u32,
        view: &BufferView,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if !Arc::ptr_eq(&self.shared.owner, &view.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        self.reserve_stage_buffer_slot(stage, index)?;
        self.stage_buffers.insert(
            (stage.code(), index),
            RenderStageBinding {
                stage,
                source: RenderStageSource::View(view.clone()),
            },
        );
        Ok(())
    }

    /// Bind one imported lease to `stage`'s `[[buffer(index)]]` slot
    /// (`research/docs/23` §3.3, v87; §90, R9i).
    ///
    /// The lease is the caller's own import: it goes through the same channel a
    /// compute case's `storage_mode` uses, and the reservation the caller hands
    /// here is the one the provider's registry resolved. The object API does
    /// not import anything itself, and the owner's window — a borrowed lease's
    /// mapping in particular — stays the caller's to keep alive until the
    /// submission has been waited for.
    ///
    /// A reservation whose range is not inside the owner allocation the caller
    /// states is refused by name ([`ContractError::LeaseRangeOutOfBounds`]),
    /// exactly as the trace's own resource table refuses it. A slot the
    /// pipeline declares *writable* is
    /// [`Error::WritableStageBufferLeaseUnsupported`]: the writeback channel
    /// lands in the bytes' own host image, and an imported lease has none.
    pub fn set_stage_buffer_lease(
        &mut self,
        stage: RenderPipelineStage,
        index: u32,
        lease: StageBufferLease,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if lease.allocation_size == 0 {
            return Err(ContractError::ZeroLength("lease allocation").into());
        }
        let lease_id = lease.reservation.lease.lease_id;
        let end = lease.reservation.end()?;
        if end > lease.allocation_size {
            return Err(ContractError::LeaseRangeOutOfBounds {
                lease: lease_id,
                end,
                allocation_size: lease.allocation_size,
            }
            .into());
        }
        self.reserve_stage_buffer_slot(stage, index)?;
        self.stage_buffers.insert(
            (stage.code(), index),
            RenderStageBinding {
                stage,
                source: RenderStageSource::Lease(lease),
            },
        );
        Ok(())
    }

    /// The two refusals every stage-buffer binding shares: one slot holds one
    /// source, and an index has to be inside the contract's own ceiling.
    fn reserve_stage_buffer_slot(
        &self,
        stage: RenderPipelineStage,
        index: u32,
    ) -> Result<(), Error> {
        if self.stage_buffers.contains_key(&(stage.code(), index)) {
            return Err(Error::StageBufferAlreadyBound { stage, index });
        }
        if index >= MAX_RENDER_STAGE_BUFFER_INDEX {
            return Err(ContractError::RenderStageBufferIndexExceeded {
                stage,
                index,
                maximum: MAX_RENDER_STAGE_BUFFER_INDEX,
            }
            .into());
        }
        // The count is the *stage's* own (`research/docs/23` §117, E-SB2): the
        // encoder holds one slot per `(stage, index)` pair, and a stage's
        // namespace is bounded by [`MAX_RENDER_STAGE_BUFFERS`] — the other
        // stage's bindings are a second list, not more of this one.
        let stage_count = self
            .stage_buffers
            .keys()
            .filter(|(code, _)| *code == stage.code())
            .count();
        if stage_count >= MAX_RENDER_STAGE_BUFFERS {
            return Err(ContractError::RenderStageBufferLimitExceeded {
                stage: Some(stage),
                requested: stage_count + 1,
                maximum: MAX_RENDER_STAGE_BUFFERS,
            }
            .into());
        }
        Ok(())
    }

    /// The bound stage buffers as the pass's own slots, in canonical order.
    ///
    /// The access of each slot is the recorded pipeline's own declaration —
    /// the pair rule `RenderPipelineContract::validate_against` states is the
    /// one the pass cannot answer by itself — so the resolution happens here,
    /// where the pipeline is known. A slot the pipeline does not declare is
    /// refused by name, and a lease-bound slot the pipeline declares writable
    /// is [`Error::WritableStageBufferLeaseUnsupported`]: an imported lease has
    /// no host image for the writeback channel to land in.
    fn bound_stage_buffers(
        &self,
        pipeline: &RenderPipeline,
    ) -> Result<Vec<RenderStageSlot>, Error> {
        let render = pipeline
            .metadata()
            .render
            .as_ref()
            .ok_or(Error::InvalidPipelineMetadata)?;
        self.stage_buffers
            .iter()
            .map(|((_, index), bound)| {
                let declared = render
                    .stage_buffers
                    .iter()
                    .find(|declared| declared.stage == bound.stage && declared.index == *index)
                    .ok_or(ContractError::UndeclaredStageBufferBinding {
                        stage: bound.stage,
                        index: *index,
                    })?;
                if declared.access.is_writable()
                    && matches!(bound.source, RenderStageSource::Lease(_))
                {
                    return Err(Error::WritableStageBufferLeaseUnsupported {
                        stage: bound.stage,
                        index: *index,
                    });
                }
                Ok(RenderStageSlot {
                    stage: bound.stage,
                    index: *index,
                    access: declared.access,
                    source: bound.source.clone(),
                })
            })
            .collect()
    }

    /// The bound streams as the pass's own inputs, in binding order.
    ///
    /// The map's keys are the binding indices, so iterating them in key order
    /// yields entry `i` = binding `i` — the positional rule the descriptor
    /// states, without a second list that could disagree with this one. A set
    /// that skips an index has no view to put at the position it skips, so the
    /// draw is refused by the contract's own
    /// [`ContractError::VertexBufferBindingMismatch`] rather than quietly
    /// shifting a stream's bytes to another binding.
    fn bound_vertex_buffers(&self) -> Vec<RenderStream> {
        self.vertex_buffers
            .iter()
            .map(|(binding, view)| RenderStream {
                binding: *binding,
                view: view.clone(),
            })
            .collect()
    }

    /// Record the milestone's single render pass.
    ///
    /// `attachment` names the buffer view the attachment lands in (its
    /// allocation/view identity and byte range); `format`, `width`, `height`
    /// and `load` restate the attachment shape the render contract fixes, and
    /// `present` selects the optional present tail action on that same
    /// attachment. Every other shape is refused here with a typed error rather
    /// than deferred to provider admission.
    ///
    /// A present tail is an action of the command's `submit`, not of its
    /// `wait`: when the command is submitted the provider counts the one
    /// acquire and one present and advances the target to its terminal layout.
    /// [`CommandBuffer::wait_until_completed`] only makes the attachment
    /// writeback host-visible; [`CommandBuffer::cancel`] or a deadline abandons
    /// that observation without rolling the present action back.
    ///
    /// This shape draws from `vertex_id`: it names no stream and no index
    /// buffer, so the pass the caller records carries neither, whatever the
    /// encoder holds. A draw whose vertex stage reads bound streams is
    /// [`Self::draw_primitives`], and one that selects through an index buffer
    /// is [`Self::draw_indexed_primitives`].
    ///
    /// The single-attachment shape this call publishes fixes
    /// [`StoreOp::Store`]: a per-attachment store decision needs a
    /// multi-attachment call, [`Self::draw_primitives_with_attachments`] or
    /// [`Self::draw_indexed_primitives_with_attachments`].
    pub fn draw_render_pass(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        self.record_render_pass(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            present,
            RenderDraw::vertex_id(),
            None,
        )
    }

    /// Record the milestone's render pass over the bound vertex streams.
    ///
    /// The pass's descriptor carries the bound streams in binding order, so the
    /// registered pipeline's vertex layout and the pass agree entry by entry
    /// (that agreement is checked here, not left to admission). At least one
    /// stream has to be bound: the `vertex_id`-only shape is
    /// [`Self::draw_render_pass`], and keeping it out of this method leaves each
    /// shape with exactly one call that records it.
    ///
    /// A count below the milestone's three is refused here with the contract's
    /// own [`ContractError::DrawVertexCountBelowMinimum`], so both direct draw
    /// shapes report that one refusal instead of the descriptor's two.
    ///
    /// The single-attachment shape this call publishes fixes
    /// [`StoreOp::Store`], exactly as [`Self::draw_render_pass`] does; the
    /// multi-attachment shape that spells one store per location is
    /// [`Self::draw_primitives_with_attachments`].
    /// The count checks every direct draw shares, in one place so a new draw
    /// entry (the instanced pair below) cannot widen them by accident: the draw
    /// covers at least the reviewed triangle, and it runs at least one
    /// instance — zero instances is the "nothing landed" shape the contract
    /// refuses as [`ContractError::ZeroLength`], so the encoder refuses it
    /// before a pass exists.
    fn admit_draw_counts(count: u32, instance_count: u32) -> Result<(), Error> {
        if count < FULL_SCREEN_TRIANGLE_VERTICES {
            return Err(ContractError::DrawVertexCountBelowMinimum {
                minimum: FULL_SCREEN_TRIANGLE_VERTICES,
                actual: count,
            }
            .into());
        }
        if instance_count == 0 {
            return Err(ContractError::ZeroLength("render instance count").into());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        vertex_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.draw_primitives_with_attachments(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            vertex_count,
            present,
        )
    }

    /// Record a multi-attachment render pass over the bound vertex streams.
    ///
    /// [`Self::draw_primitives`] is the single-attachment shape of this call:
    /// it wraps one attachment in the list and delegates here, so every direct
    /// vertex-buffer draw shares one recording path. `attachments` is
    /// positional — entry `i` is location `i` — and the recorded pipeline's
    /// compiled formats have to agree entry by entry, which the render contract
    /// checks at recording time exactly as the single-attachment shape does. An
    /// empty list is [`Error::EmptyRenderAttachmentList`], more than
    /// [`MAX_COLOR_ATTACHMENTS`] entries is
    /// [`Error::RenderAttachmentLimitExceeded`], and one `(allocation, view)`
    /// identity recorded twice is [`Error::DuplicateRenderAttachment`].
    /// Each entry's `store` reaches the pass at its own location, and an
    /// all-`DontCare` list is [`ContractError::AllRenderAttachmentsDiscarded`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_with_attachments(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, 1)?;
        let draw = RenderDraw {
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count: 1,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass over the bound vertex streams
    /// whose entries state their own source (`research/docs/23` §115, E-TX9b).
    ///
    /// The declaration sibling of [`Self::draw_primitives_with_attachments`]:
    /// the list is positional the same way — entry `i` is location `i` — and
    /// each entry is either a caller-held view
    /// ([`RenderAttachmentDeclaration::View`], which is exactly the shape the
    /// view-only entry takes) or the owner's registered window
    /// ([`RenderAttachmentDeclaration::Window`]), whose frame lands in the
    /// guest's own pages under the store arm [`StoreOp::Borrowed`]. Every rule
    /// the view-only entry states — the attachment count, the extent each
    /// entry fills, the format agreement with the recorded pipeline, the
    /// all-`DontCare` refusal — holds here unchanged, because the entries are
    /// the same shape.
    ///
    /// A window entry is a *declaration*: the serial view list is where the
    /// rail resolves the window, so the same command has to declare it on a
    /// pass of its own (the object rail's declaring pass, exactly as a
    /// compute binding's bytes are declared). A command whose declarations do
    /// not name the window is refused by name when it commits
    /// ([`Error::WindowAttachmentUndeclared`]) instead of landing nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_with_declared_attachments(
        &mut self,
        attachments: &[RenderAttachmentDeclaration<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, 1)?;
        let draw = RenderDraw {
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count: 1,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_declared_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// whose entries state their own source (`research/docs/23` §115, E-TX9b).
    ///
    /// [`Self::draw_primitives_with_declared_attachments`]'s indexed sibling:
    /// the draw selects through the bound index buffer, exactly as
    /// [`Self::draw_indexed_primitives_with_attachments`] does, and the
    /// attachment list is the declaration list above. This is the census's
    /// shape — a stream-fed vertex stage, an index buffer and one colour
    /// attachment backed by the guest's own pages — with the attachment's
    /// source stated instead of fixed.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_declared_attachments(
        &mut self,
        attachments: &[RenderAttachmentDeclaration<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, 1)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count: 1,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_declared_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// that masks its fragments with a stencil test (`research/docs/23` §3.3,
    /// v47/v48).
    ///
    /// The stencil sibling of [`Self::draw_indexed_primitives_with_depth`],
    /// matching the reviewed stencil fixture's shape: the pass opens the
    /// rail-owned `stencil8` surface `stencil` describes and tests and writes
    /// it with `stencil_test`, or opens it with no test when that is `None`.
    /// Every other rule — the positional attachment list, the pipeline's
    /// compiled formats, the bound streams and the index buffer — is
    /// [`Self::draw_indexed_primitives_with_attachments`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_stencil(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        stencil: RenderStencilAttachment,
        stencil_test: Option<RenderStencilTest>,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            blend: None,
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: Some(stencil),
            stencil_test,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass over the bound vertex streams,
    /// run once per instance (`research/docs/23` §3.3, v31/v32).
    ///
    /// Metal's own entry point is
    /// `drawPrimitives(type:vertexStart:vertexCount:instanceCount:)`: the
    /// instance count belongs to the draw call rather than to encoder state,
    /// which is why this is a second recording entry and not a `set_` method
    /// like [`Self::set_scissor`]. Every other rule is
    /// [`Self::draw_primitives_with_attachments`]'s: the attachment list is
    /// positional, the pipeline's compiled formats have to agree entry by
    /// entry, and a list with no store is refused. `instance_count` of zero is
    /// [`ContractError::ZeroLength`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_instanced_with_attachments(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, instance_count)?;
        let draw = RenderDraw {
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass over the bound vertex streams that
    /// opens a depth surface (`research/docs/23` §3.3, v36/v37).
    ///
    /// The depth counterpart of
    /// [`Self::draw_primitives_with_attachments`]: the pass opens the
    /// rail-owned surface `depth` describes and tests with `depth_test`, or
    /// opens it with no test when that is `None`. Every other rule — the
    /// positional attachment list, the pipeline's compiled formats, the bound
    /// streams — is that entry's.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_with_depth(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        instance_count: u32,
        depth: RenderDepthAttachment,
        depth_test: Option<RenderDepthTest>,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, instance_count)?;
        let draw = RenderDraw {
            blend: None,
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count,
            base_vertex: 0,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the single-attachment shape of
    /// [`Self::draw_primitives_instanced_with_attachments`], fixing
    /// [`StoreOp::Store`] exactly as [`Self::draw_primitives`] does.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_instanced(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        vertex_count: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.draw_primitives_instanced_with_attachments(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            vertex_count,
            instance_count,
            present,
        )
    }

    /// Record the milestone's render pass through the bound index buffer.
    ///
    /// `index_count` is the number of indices the draw consumes and is what the
    /// descriptor's `vertices` field carries, the way the frozen contract
    /// spells `drawIndexedPrimitives(indexCount:)`. Vertex streams are optional
    /// here: a draw with none bound selects through `vertex_id`, which the
    /// contract admits at exactly [`FULL_SCREEN_TRIANGLE_VERTICES`] indices, so
    /// an index-only pass is the indexed sibling of
    /// [`Self::draw_render_pass`].
    ///
    /// An encoder with no index buffer bound is refused with
    /// [`Error::MissingIndexBuffer`] before any other shape is looked at.
    ///
    /// The single-attachment shape this call publishes fixes
    /// [`StoreOp::Store`]; the multi-attachment shape that spells one store per
    /// location is [`Self::draw_indexed_primitives_with_attachments`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        index_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.draw_indexed_primitives_with_attachments(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            index_count,
            present,
        )
    }

    /// Record a multi-attachment render pass through the bound index buffer.
    ///
    /// [`Self::draw_indexed_primitives`] is the single-attachment shape of this
    /// call: it wraps one attachment in the list and delegates here.
    /// `index_count` is the number of indices the draw consumes and is what the
    /// descriptor's `vertices` field carries, the way the frozen contract
    /// spells `drawIndexedPrimitives(indexCount:)`; vertex streams are optional
    /// exactly as they are for the single-attachment shape. The attachment
    /// list's refusals are the ones
    /// [`Self::draw_primitives_with_attachments`] states, including the
    /// all-`DontCare` [`ContractError::AllRenderAttachmentsDiscarded`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_attachments(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, 1)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count: 1,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// with the pass-wide multisample raster (`research/docs/23` §3.3,
    /// v51/v52).
    ///
    /// The recording carries the same one-field state the trace contract does:
    /// the colour attachments are rendered with a four-sample raster and the
    /// covered samples are resolved into the attachments themselves, which is
    /// what the caller's views observe. Every other rule is
    /// [`Self::draw_indexed_primitives_with_attachments`]'s, and the contract's
    /// own admission refuses the shapes this increment does not review — a
    /// depth or stencil surface beside the raster, and a present action behind
    /// it — exactly as it does for a trace.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        multisample: contract::MultisampleState,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        // A stated single-sample raster is what the absent field already means,
        // so the recording refuses it with the contract's own error instead of
        // recording a second encoding of the shape every earlier entry states
        // (`research/docs/23` §3.3, v51).
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: Some(multisample),
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the depth-bearing sibling of
    /// [`Self::draw_indexed_primitives_with_multisample`]
    /// (`research/docs/23` §3.3, v53/v54).
    ///
    /// The recording carries both halves the trace contract states: the
    /// pass-wide four-sample raster and the rail-owned depth surface the pass
    /// tests and writes. The surface is one description, exactly as the
    /// v37/v44 entries state theirs, and the contract's own admission refuses
    /// the shapes this increment does not review — a stored depth surface
    /// beside the raster (the depth resolve filters are a later increment), a
    /// single-sample state, a stencil surface and a present action. Every other
    /// rule is [`Self::draw_indexed_primitives_with_depth`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample_depth(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        depth: RenderDepthAttachment,
        depth_test: Option<RenderDepthTest>,
        multisample: contract::MultisampleState,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        // The single-sample state is what the absent field means, exactly as
        // the colour-only entry states it (`research/docs/23` §3.3, v51).
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        // A multisampled depth surface is rail-owned in this increment: keeping
        // it would need the depth resolve filters, so the recording refuses the
        // stored shape with the contract's own error rather than recording a
        // pass admission would refuse.
        if depth.store == Some(contract::DepthStoreOp::Store) {
            return Err(ContractError::MultisampleDepthStoreUnsupported.into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: None,
            stencil_test: None,
            multisample: Some(multisample),
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the stored-depth sibling of
    /// [`Self::draw_indexed_primitives_with_multisample_depth`]
    /// (`research/docs/23` §3.3, v57/v58).
    ///
    /// The recording carries the resolve the v57 increment reviewed: the
    /// pass-wide four-sample raster keeps its depth surface, and `filter`
    /// states how the stored four-sample texels reduce into the surface the
    /// caller's identity names. The surface has to be stored — the `store`
    /// action and the landing identity are one decision, exactly as the
    /// contract's own admission states them — and the recording refuses every
    /// other shape with the contract's own errors instead of recording a pass
    /// admission would refuse. Every other rule is
    /// [`Self::draw_indexed_primitives_with_multisample_depth`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample_depth_resolve(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        depth: RenderDepthAttachment,
        depth_test: Option<RenderDepthTest>,
        multisample: contract::MultisampleState,
        filter: contract::DepthResolveFilter,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        // The single-sample state is what the absent field means, exactly as
        // the colour-only entry states it (`research/docs/23` §3.3, v51).
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        // The resolve only means something beside a stored multisampled depth
        // surface (`research/docs/23` §3.3, v57): a filter beside a surface the
        // pass drops, or beside no surface at all, is refused with the
        // contract's own error rather than silently ignored.
        if depth.store != Some(contract::DepthStoreOp::Store) {
            return Err(ContractError::DepthResolveWithoutStoredDepth {
                store: depth.store.map(contract::DepthStoreOp::code),
            }
            .into());
        }
        // The store action and the landing identity are one decision in two
        // fields (`research/docs/23` §3.3, v43): keeping the surface needs a
        // landing, so a stored surface without one is refused here exactly as
        // the contract's own admission refuses it.
        if depth.identity.is_none() {
            return Err(ContractError::DepthStoreIdentityMismatch {
                store: depth.store.map(contract::DepthStoreOp::code),
                identity: depth.identity.is_some(),
            }
            .into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: None,
            stencil_test: None,
            multisample: Some(multisample),
            depth_resolve: Some(contract::MultisampleDepthResolve { filter }),
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the stencil-bearing sibling of
    /// [`Self::draw_indexed_primitives_with_multisample`]
    /// (`research/docs/23` §3.3, v55/v56).
    ///
    /// The depth sibling's rule one byte wide: the recording carries the
    /// pass-wide four-sample raster and the rail-owned stencil surface the pass
    /// tests and writes. The contract's own admission refuses a stored stencil
    /// surface (the stencil resolve is a later increment), a single-sample
    /// state, a combined depth-stencil surface and a present action, and this
    /// entry refuses them at recording time where they are its own arguments.
    /// Every other rule is [`Self::draw_indexed_primitives_with_stencil`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample_stencil(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        stencil: RenderStencilAttachment,
        stencil_test: Option<RenderStencilTest>,
        multisample: contract::MultisampleState,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        if stencil.store == Some(StoreOp::Store) {
            return Err(ContractError::MultisampleStencilStoreUnsupported.into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: Some(stencil),
            stencil_test,
            multisample: Some(multisample),
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the combined-surface sibling of
    /// [`Self::draw_indexed_primitives_with_multisample_depth`]
    /// (`research/docs/23` §3.3, v66/v68).
    ///
    /// One entry carries both faces of the combined depth-stencil surface the
    /// v66 increment reviewed: the pass-wide four-sample raster and the single
    /// rail-owned surface the pass tests and writes through both faces. The
    /// two faces share that surface, so their store decisions are one decision
    /// and a lopsided pair is refused with the contract's own error rather
    /// than read as either reviewed shape. This entry carries the v66 shape —
    /// both faces dropped with the pass, observed through the colour resolve —
    /// and states no resolve for either face, so the v60 stored shape (both
    /// faces kept through their own resolves) is refused here with the
    /// contract's own error instead of recording a pass admission would
    /// refuse. A single-sample state is refused as every multisample sibling
    /// refuses it. Every other rule is
    /// [`Self::draw_indexed_primitives_with_multisample_stencil`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample_depth_stencil(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        multisample: contract::MultisampleState,
        depth: RenderDepthAttachment,
        stencil: RenderStencilAttachment,
        depth_test: Option<RenderDepthTest>,
        stencil_test: Option<RenderStencilTest>,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        // The single-sample state is what the absent field means, exactly as
        // the colour-only entry states it (`research/docs/23` §3.3, v51).
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        // The two faces share one surface, so the pass keeps both or neither
        // (`research/docs/23` §3.3, v66): a pair that stores one face while
        // dropping the other is refused with the contract's own error, in the
        // same shape the contract's own admission refuses it.
        let depth_stored = depth.store == Some(contract::DepthStoreOp::Store);
        let stencil_stored = stencil.store == Some(StoreOp::Store);
        if depth_stored != stencil_stored {
            return Err(ContractError::MultisampleCombinedSurfaceUnsupported {
                depth_stored,
                stencil_stored,
            }
            .into());
        }
        // A kept face's texels are only observable through its own resolve,
        // and this entry states neither — so the stored shape is refused at
        // recording time with the contract's own error, the same one admission
        // would answer the recorded pass with, rather than silently dropping
        // the run's landing (`research/docs/23` §3.3, v57/v66).
        if depth_stored {
            return Err(ContractError::MultisampleDepthStoreUnsupported.into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: Some(stencil),
            stencil_test,
            multisample: Some(multisample),
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the stored sibling of
    /// [`Self::draw_indexed_primitives_with_multisample_depth_stencil`]
    /// (`research/docs/23` §3.3, v60/v70).
    ///
    /// The same combined depth-stencil pair, kept by the pass: both faces of
    /// the one rail-owned surface are stored through the resolve each states,
    /// which is the v60 shape the trace rails execute
    /// (`msaa_stencil_resolve_sample0_4x4`, and the two-filter
    /// `msaa_stencil_resolve_drs_4x4` on the rails whose devices admit both
    /// filters). The filters travel with their own surface, exactly as the
    /// depth-only resolving entry states its one filter, so a recording that
    /// names a filter for a face it drops is refused rather than read as a
    /// pass that resolves nothing.
    ///
    /// The two faces share one surface, so their store decisions are one
    /// decision: a lopsided pair is refused with the contract's own
    /// [`ContractError::MultisampleCombinedSurfaceUnsupported`], and a pair
    /// that keeps neither face is the v68 entry's shape — refused here with
    /// the contract's own "resolve beside a dropped surface" error instead of
    /// silently dropping the two filters. A kept face names where its texels
    /// land, and a half-stated pair is refused with the contract's own
    /// identity error, exactly as the depth-only and stencil-only entries
    /// refuse it. A single-sample state is refused as every multisample
    /// sibling refuses it. Every other rule is
    /// [`Self::draw_indexed_primitives_with_multisample_depth_stencil`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_multisample_depth_stencil_resolve(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        multisample: contract::MultisampleState,
        depth: RenderDepthAttachment,
        depth_filter: contract::DepthResolveFilter,
        stencil: RenderStencilAttachment,
        stencil_filter: contract::StencilResolveFilter,
        depth_test: Option<RenderDepthTest>,
        stencil_test: Option<RenderStencilTest>,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if multisample.sample_count == contract::SampleCount::One {
            return Err(ContractError::SingleSampleMultisampleState.into());
        }
        // The two faces share one surface, so the pass keeps both or neither
        // (`research/docs/23` §3.3, v66): a pair that stores one face while
        // dropping the other is refused with the contract's own error, in the
        // same shape the contract's own admission refuses it.
        let depth_stored = depth.store == Some(contract::DepthStoreOp::Store);
        let stencil_stored = stencil.store == Some(StoreOp::Store);
        if depth_stored != stencil_stored {
            return Err(ContractError::MultisampleCombinedSurfaceUnsupported {
                depth_stored,
                stencil_stored,
            }
            .into());
        }
        // This entry is the resolving one: a resolve only means something
        // beside a stored surface (`research/docs/23` §3.3, v57/v60), so the
        // pair that keeps neither face is refused with the contract's own
        // error instead of recording two filters admission would refuse.
        if !depth_stored {
            return Err(ContractError::DepthResolveWithoutStoredDepth {
                store: depth.store.map(contract::DepthStoreOp::code),
            }
            .into());
        }
        // The store action and the landing identity are one decision in two
        // fields (`research/docs/23` §3.3, v43/v49), for both faces at once:
        // keeping the surface without saying where its texels land is refused
        // here exactly as the contract's own admission refuses it.
        if depth.identity.is_none() {
            return Err(ContractError::DepthStoreIdentityMismatch {
                store: depth.store.map(contract::DepthStoreOp::code),
                identity: false,
            }
            .into());
        }
        if stencil.identity.is_none() {
            return Err(ContractError::StencilStoreIdentityMismatch {
                store: stencil.store,
                identity: false,
            }
            .into());
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: Some(stencil),
            stencil_test,
            multisample: Some(multisample),
            depth_resolve: Some(contract::MultisampleDepthResolve {
                filter: depth_filter,
            }),
            stencil_resolve: Some(contract::MultisampleStencilResolve {
                filter: stencil_filter,
            }),
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer,
    /// run once per instance (`research/docs/23` §3.3, v31/v32).
    ///
    /// The indexed sibling of
    /// [`Self::draw_primitives_instanced_with_attachments`], matching Metal's
    /// `drawIndexedPrimitives(...:instanceCount:)`. The bound index buffer is
    /// still required ([`Error::MissingIndexBuffer`]), the pipeline's layout
    /// still decides how each stream advances, and every other rule is
    /// [`Self::draw_indexed_primitives_with_attachments`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_instanced_with_attachments(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer,
    /// reading every index through `base_vertex` (`research/docs/23` §3.3,
    /// v35).
    ///
    /// Metal's own entry point is
    /// `drawIndexedPrimitives(…:baseVertex:baseInstance:)`; the offset belongs
    /// to the draw call rather than to encoder state, which is why this is a
    /// third recording entry rather than a `set_` method. Only an indexed draw
    /// may carry it — the contract refuses a base vertex without an index
    /// buffer, and an encoder that reaches here without one is refused with
    /// [`Error::MissingIndexBuffer`] exactly as the other indexed entries are.
    /// Every other rule is
    /// [`Self::draw_indexed_primitives_with_attachments`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_base_vertex_with_attachments(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        base_vertex: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex,
            blend: None,
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass over the bound vertex streams that
    /// blends its fragment output (`research/docs/23` §3.3, v40/v42).
    ///
    /// `blend` states one entry per colour attachment, in location order,
    /// exactly as the trace contract's list does; the pass the descriptor
    /// builds carries that list, and the providers build their pipeline from
    /// it. Every other rule is
    /// [`Self::draw_primitives_with_attachments`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_with_blend(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        instance_count: u32,
        blend: &[contract::BlendAttachment],
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, instance_count)?;
        let draw = RenderDraw {
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count,
            base_vertex: 0,
            blend: Some(contract::RenderPassBlend {
                attachments: blend.to_vec(),
            }),
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// that blends its fragment output (`research/docs/23` §3.3, v40/v42).
    ///
    /// The indexed sibling of [`Self::draw_primitives_with_blend`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_blend(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        blend: &[contract::BlendAttachment],
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: Some(contract::RenderPassBlend {
                attachments: blend.to_vec(),
            }),
            cull: None,
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass over the bound vertex streams that
    /// culls triangles (`research/docs/23` §3.3, v39/v41).
    ///
    /// The culling counterpart of
    /// [`Self::draw_primitives_with_attachments`]: the pass runs with the mode
    /// and the front-facing winding `cull` states, and every other rule is that
    /// entry's.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_primitives_with_cull(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        vertex_count: u32,
        instance_count: u32,
        cull: contract::RenderPassCull,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        if self.vertex_buffers.is_empty() {
            return Err(Error::MissingVertexBuffer);
        }
        Self::admit_draw_counts(vertex_count, instance_count)?;
        let draw = RenderDraw {
            vertices: vertex_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: None,
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: Some(cull),
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// that culls triangles (`research/docs/23` §3.3, v39/v41).
    ///
    /// The indexed sibling of [`Self::draw_primitives_with_cull`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_cull(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        cull: contract::RenderPassCull,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            blend: None,
            cull: Some(cull),
            depth: None,
            depth_test: None,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record a multi-attachment render pass through the bound index buffer
    /// that opens a depth surface (`research/docs/23` §3.3, v36/v37).
    ///
    /// The indexed sibling of [`Self::draw_primitives_with_depth`], matching
    /// the reviewed depth fixture's shape: the draw selects its vertices
    /// through the bound index buffer and the pass opens the depth surface
    /// `depth` describes. Every other rule is
    /// [`Self::draw_indexed_primitives_with_attachments`]'s.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_with_depth(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        index_count: u32,
        instance_count: u32,
        depth: RenderDepthAttachment,
        depth_test: Option<RenderDepthTest>,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let (index_view, index_format) = self
            .index_buffer
            .as_ref()
            .ok_or(Error::MissingIndexBuffer)?;
        Self::admit_draw_counts(index_count, instance_count)?;
        let draw = RenderDraw {
            blend: None,
            vertices: index_count,
            vertex_buffers: self.bound_vertex_buffers(),
            indices: Some(RenderIndex {
                view: index_view.clone(),
                format: *index_format,
            }),
            instance_count,
            base_vertex: 0,
            cull: None,
            depth: Some(depth),
            depth_test,
            stencil: None,
            stencil_test: None,
            multisample: None,
            depth_resolve: None,
            stencil_resolve: None,
        };
        self.record_render_pass(attachments, width, height, present, draw, None)
    }

    /// Record the single-attachment shape of
    /// [`Self::draw_indexed_primitives_base_vertex_with_attachments`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_base_vertex(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        index_count: u32,
        base_vertex: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.draw_indexed_primitives_base_vertex_with_attachments(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            index_count,
            base_vertex,
            instance_count,
            present,
        )
    }

    /// Record the single-attachment shape of
    /// [`Self::draw_indexed_primitives_instanced_with_attachments`].
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_primitives_instanced(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        index_count: u32,
        instance_count: u32,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.draw_indexed_primitives_instanced_with_attachments(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            index_count,
            instance_count,
            present,
        )
    }

    /// Land one render pass in the command's pass list.
    ///
    /// Every draw above ends here, so all draw shapes share one statement of
    /// what recording a pass means: every attachment belongs to this device,
    /// has the extent the restated shape implies and names a distinct
    /// `(allocation, view)` identity, the descriptor the pass becomes passes
    /// the frozen contract's own shape rules, the registered pipeline's render
    /// contract agrees with that descriptor, the command's pass list has room,
    /// and the serial-resource budget holds. Only then does the pass land.
    /// `indirect` carries the ICB a replayed pass inherits, which is also the
    /// one pass the command may replay: an ICB already recorded is
    /// [`Error::IndirectAlreadyRecorded`].
    #[allow(clippy::too_many_arguments)]
    fn record_render_pass(
        &mut self,
        attachments: &[RenderColorAttachment<'_>],
        width: u64,
        height: u64,
        present: Option<PresentInitial>,
        draw: RenderDraw,
        indirect: Option<&IndirectCommandBuffer>,
    ) -> Result<(), Error> {
        // Every pre-E-TX9b entry states caller-held views, so this wrapper is
        // the whole of their share: the declared-list form below is the one
        // recording path (`research/docs/23` §114/§115, E-TX9b).
        let declarations = attachments
            .iter()
            .copied()
            .map(RenderAttachmentDeclaration::View)
            .collect::<Vec<_>>();
        self.record_declared_render_pass(&declarations, width, height, present, draw, indirect)
    }

    /// Record one pass over a positional list of attachment *declarations*
    /// (`research/docs/23` §114/§115, E-TX9b).
    ///
    /// The list is [`Self::record_render_pass`]'s own, with each entry's
    /// source arm stated by the caller instead of fixed to a caller-held view:
    /// an entry that names the owner's registered window lands its frame in
    /// the guest's own pages ([`RenderWindowAttachment`]), and its store is the
    /// window arm [`StoreOp::Borrowed`]. Every other rule — the attachment
    /// count, the extent each entry's own bytes have to fill, the format
    /// agreement with the recorded pipeline, the all-`DontCare` refusal — is
    /// the view-only path's, because those shapes are the same shape.
    #[allow(clippy::too_many_arguments)]
    fn record_declared_render_pass(
        &mut self,
        attachments: &[RenderAttachmentDeclaration<'_>],
        width: u64,
        height: u64,
        present: Option<PresentInitial>,
        draw: RenderDraw,
        indirect: Option<&IndirectCommandBuffer>,
    ) -> Result<(), Error> {
        let pipeline = self.pipeline.clone().ok_or(ApiError::MissingPipeline)?;
        // A zero-colour-attachment recording is the depth-only pass
        // (`research/docs/23` §3.3, v46): the rasterizer still tests and writes
        // the depth surface the draw names, and that surface is the whole
        // landing. Without a *stored* depth attachment there is no target at
        // all, which stays the refusal it always was.
        let stored_depth = draw
            .depth
            .as_ref()
            .is_some_and(|depth| depth.store == Some(contract::DepthStoreOp::Store));
        if attachments.is_empty() && !stored_depth {
            return Err(Error::EmptyRenderAttachmentList);
        }
        if attachments.len() > MAX_COLOR_ATTACHMENTS {
            return Err(Error::RenderAttachmentLimitExceeded {
                requested: attachments.len(),
                maximum: MAX_COLOR_ATTACHMENTS,
            });
        }
        let mut seen = BTreeSet::new();
        let mut declared = Vec::with_capacity(attachments.len());
        for attachment in attachments {
            match attachment {
                RenderAttachmentDeclaration::View(attachment) => {
                    if !Arc::ptr_eq(&self.shared.owner, &attachment.view.buffer.inner.owner) {
                        return Err(Error::ForeignBuffer);
                    }
                    if !seen.insert((attachment.view.allocation_id(), attachment.view.view_id())) {
                        return Err(Error::DuplicateRenderAttachment);
                    }
                    let expected_bytes = width
                        .checked_mul(height)
                        .and_then(|texels| texels.checked_mul(attachment.format.bytes_per_texel()))
                        .ok_or(ContractError::ArithmeticOverflow("attachment extent"))?;
                    if u64::try_from(attachment.view.length).unwrap_or(u64::MAX) != expected_bytes {
                        return Err(ContractError::AttachmentExtentMismatch {
                            pass_index: 0,
                            view: attachment.view.view_id,
                            expected: expected_bytes,
                            declared: u64::try_from(attachment.view.length).unwrap_or(u64::MAX),
                        }
                        .into());
                    }
                    declared.push(RenderTargetAttachment {
                        source: RenderAttachmentSource::View(attachment.view.clone()),
                        format: attachment.format,
                        load: attachment.load,
                        store: attachment.store,
                    });
                }
                RenderAttachmentDeclaration::Window(window) => {
                    declared
                        .push(self.validate_window_attachment(*window, width, height, &mut seen)?);
                }
            }
        }
        let target = RenderTarget {
            attachments: declared,
            width,
            height,
            scissor: self.scissor,
            // The encoder's fragment texture bindings travel with the pass it
            // records (`research/docs/23` §3.3, v70): the map's ascending keys
            // are the binding order the descriptor makes positional, and each
            // view carries the bytes the texture declaration snapshotted.
            textures: self
                .fragment_textures
                .iter()
                .map(|(binding, texture)| texture.view(*binding))
                .collect::<Result<Vec<_>, _>>()?,
            // The encoder's runtime sampler states travel with the pass the
            // same way its textures do (`research/docs/23` §3.3, v102): the
            // map's ascending keys are the canonical order the descriptor
            // states, and each entry is the state the descriptor's sampler is
            // created with.
            samplers: self
                .fragment_samplers
                .iter()
                .map(|(binding, policy)| contract::RenderSamplerBinding::new(*binding, *policy))
                .collect(),
            // The encoder's stage-buffer bindings travel with the pass it
            // records (`research/docs/23` §3.3, v83-v87): the map's canonical
            // order is the descriptor's own, and each slot's access is the
            // recorded pipeline's declaration.
            stage_buffers: self.bound_stage_buffers(&pipeline)?,
            present,
            draw,
        };
        let pipeline_id = pipeline.metadata().pipeline_id;
        // Validate the descriptor the pass will become, so a wrong viewport,
        // vertex count, attachment shape or present shape is refused before any
        // resource is reserved; and check the pass against the registered
        // pipeline's own render contract, which is the one agreement the pass
        // cannot answer by itself because it carries an id and not the
        // pipeline's compiled layout.
        //
        // The draw's inputs are shapes here: their bytes are taken once, at
        // commit, under that command's reservations, and a recorded pass never
        // reads them.
        let descriptor = target.descriptor(pipeline_id, &shape_only_bytes)?;
        pipeline
            .metadata()
            .render
            .as_ref()
            .ok_or(Error::InvalidPipelineMetadata)?
            // The object API records shapes and resolves no lease: its index
            // bytes are the ones the descriptor carries, so the strict arm is
            // the only one it can state (`research/docs/23` §92, R9k).
            .validate_against(&descriptor, None)?;
        let mut inner = lock(&self.shared.inner, "provider command")?;
        if indirect.is_some() && inner.indirect.is_some() {
            return Err(Error::IndirectAlreadyRecorded);
        }
        let maximum = usize::try_from(self.shared.owner.capabilities.max_passes)
            .unwrap_or(usize::MAX)
            .min(8);
        if inner.passes.len() >= maximum {
            return Err(Error::PassLimit {
                requested: inner.passes.len() + 1,
                maximum,
            });
        }
        let mut unique = recorded_view_ids(&inner.passes);
        for attachment in &target.attachments {
            unique.insert(attachment.source.view_id());
        }
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass::Render {
            pipeline: pipeline.clone(),
            target: Box::new(target),
        });
        if let Some(icb) = indirect {
            inner.indirect = Some(icb.clone());
        }
        if indirect.is_some() {
            self.indirect = true;
        }
        self.draw_count += 1;
        Ok(())
    }

    /// The attachment extent one colour entry has to fill
    /// (`research/docs/23` §114, E-TX9b).
    fn attachment_extent(width: u64, height: u64, format: AttachmentFormat) -> Result<u64, Error> {
        width
            .checked_mul(height)
            .and_then(|texels| texels.checked_mul(format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("attachment extent").into())
    }

    /// Validate one owner-window attachment and record it as the target entry
    /// its own statement describes (`research/docs/23` §114/§115, E-TX9b).
    ///
    /// Four rules, each answered here rather than one rail deeper:
    ///
    /// * the arm has to be the owner's own pages
    ///   ([`StageBufferLeaseArm::BorrowedNoCopy`]); a staged lease is the
    ///   provider's copy of one reservation, so a frame landed there would
    ///   write into a buffer no owner's ledger protects
    ///   ([`Error::WindowAttachmentNamesACopyArm`]);
    /// * the reservation has to lie inside the owner allocation the caller
    ///   states ([`ContractError::LeaseRangeOutOfBounds`]);
    /// * the window has to be the attachment's own tightly packed extent
    ///   ([`Error::WindowAttachmentExtentMismatch`]) — the frame is all of
    ///   `width * height * bytes_per_texel`, so any other length would have to
    ///   be truncated or padded, which this arm refuses by name rather than
    ///   doing silently;
    /// * one identity may hold one entry
    ///   ([`Error::DuplicateRenderAttachment`]).
    ///
    /// The declared view's identity is the lease's own — `(allocation,
    /// ViewId(lease))` — exactly as a lease-bound stage buffer's is, so the
    /// `(allocation, view)` pair the pass's descriptor carries is the pair the
    /// serial view list has to declare. The store is the window arm
    /// [`StoreOp::Borrowed`]: the frame lands in the owner's registered pages,
    /// and a source that says so cannot carry any other store.
    fn validate_window_attachment(
        &self,
        window: RenderWindowAttachment,
        width: u64,
        height: u64,
        seen: &mut BTreeSet<(AllocationId, ViewId)>,
    ) -> Result<RenderTargetAttachment, Error> {
        let lease_id = window.lease.reservation.lease.lease_id;
        if window.lease.arm != StageBufferLeaseArm::BorrowedNoCopy {
            return Err(Error::WindowAttachmentNamesACopyArm { lease: lease_id });
        }
        if window.lease.allocation_size == 0 {
            return Err(ContractError::ZeroLength("lease allocation").into());
        }
        let end = window.lease.reservation.end()?;
        if end > window.lease.allocation_size {
            return Err(ContractError::LeaseRangeOutOfBounds {
                lease: lease_id,
                end,
                allocation_size: window.lease.allocation_size,
            }
            .into());
        }
        let expected_bytes = Self::attachment_extent(width, height, window.format)?;
        let window_bytes = window.lease.reservation.length;
        if window_bytes != expected_bytes {
            return Err(Error::WindowAttachmentExtentMismatch {
                lease: lease_id,
                expected: expected_bytes,
                declared: window_bytes,
            });
        }
        let identity = (
            window.lease.reservation.lease.allocation_id,
            ViewId::new(lease_id.get()),
        );
        if !seen.insert(identity) {
            return Err(Error::DuplicateRenderAttachment);
        }
        Ok(RenderTargetAttachment {
            source: RenderAttachmentSource::Window(window.lease),
            format: window.format,
            load: window.load,
            store: StoreOp::Borrowed,
        })
    }

    /// Record the milestone's single render pass replayed from one encoded
    /// indirect draw (`research/docs/25` §6 Step 4). The attachment, viewport
    /// and clear shapes are the direct [`Self::draw_render_pass`] ones; only
    /// the draw command comes from the ICB, so the vertex/instance counts are
    /// the ICB's own.
    ///
    /// Both draw kinds an ICB payload can carry are replayed: `Draw` keeps the
    /// `vertex_id` shape the first increment published, and `DrawIndexed` names
    /// the ICB's index count in the pass's `vertices` field, which is the shape
    /// the Vulkan rail replays with its own `[0, 1, 2]` index buffer. Any other
    /// kind is [`Error::IndirectKindMismatch`], and the refusal names the whole
    /// accepted set.
    ///
    /// An indirect replay reads its draw input from the ICB's payload, so an
    /// encoder that bound vertex streams or an index buffer is refused with
    /// [`Error::IndirectReplayInputConflict`]: the replayed draw would never
    /// read those bindings, and one pass may not carry two answers to what it
    /// reads.
    ///
    /// The single-attachment shape this call publishes fixes
    /// [`StoreOp::Store`], exactly as the direct [`Self::draw_render_pass`]
    /// shape it restates does.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        load: RenderAttachmentLoad,
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &icb.inner.owner) {
            return Err(Error::ForeignIndirectCommandBuffer);
        }
        let draw = match &icb.payload().command {
            // The first increment's non-indexed replay: three `vertex_id`
            // vertices, no caller-held input, exactly as the direct vertex_id
            // pass spells it. The ICB's own vertex count stays where the rail
            // reads it — the payload — because the contract's vertex_id shape
            // fixes the count at three, and carrying the payload's count here
            // would refuse an ICB the rail executes today.
            IndirectCommandDescriptor::Draw { instance_count, .. } => {
                let mut draw = RenderDraw::vertex_id();
                // The replay's counts live in the ICB payload, which is what
                // the rails read; mirroring the instance count here keeps the
                // pass's own shape (and therefore admission's instancing bits)
                // describing the draw the rail will actually replay
                // (`research/docs/23` §3.3, v31).
                draw.instance_count = *instance_count;
                draw
            }
            // The reviewed indexed replay: the descriptor's `vertices` is the
            // ICB's index count, and the rail supplies the `[0, 1, 2]` index
            // buffer the replayed draw selects through.
            IndirectCommandDescriptor::DrawIndexed {
                index_count,
                instance_count,
            } => RenderDraw {
                vertices: *index_count,
                vertex_buffers: Vec::new(),
                indices: None,
                instance_count: *instance_count,
                // The ICB's own payload has no base-vertex field in this
                // increment: the replay reads the index values the rail's own
                // buffer holds (`research/docs/25` §4.3).
                base_vertex: 0,
                // An ICB replay states neither culling, blending nor a depth
                // surface: the payload has none of them, so the replay stays
                // the v20 shape (`research/docs/25` §4.3, v37/v41/v42).
                blend: None,
                cull: None,
                depth: None,
                depth_test: None,
                stencil: None,
                stencil_test: None,
                multisample: None,
                depth_resolve: None,
                stencil_resolve: None,
            },
            other => {
                return Err(Error::IndirectKindMismatch {
                    accepted: &RENDER_DRAW_KINDS,
                    actual: other.kind(),
                })
            }
        };
        if !self.vertex_buffers.is_empty()
            || self.index_buffer.is_some()
            || !self.fragment_textures.is_empty()
            || !self.fragment_samplers.is_empty()
        {
            return Err(Error::IndirectReplayInputConflict {
                vertex_buffers: self.vertex_buffers.len(),
                index_buffer: self.index_buffer.is_some(),
                fragment_textures: self.fragment_textures.len(),
                fragment_samplers: self.fragment_samplers.len(),
            });
        }
        if self.draw_count > 0 || self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        self.record_render_pass(
            &[RenderColorAttachment {
                view: attachment,
                format,
                load,
                store: StoreOp::Store,
            }],
            width,
            height,
            present,
            draw,
            Some(icb),
        )
    }

    pub fn end_encoding(mut self) -> Result<(), Error> {
        self.ensure_open()?;
        let mut inner = lock(&self.shared.inner, "provider command")?;
        let result = if self.draw_count == 0 {
            Err(Error::Api(ApiError::MissingDispatch))
        } else {
            Ok(())
        };
        inner.encoder_open = false;
        if let Err(error) = &result {
            inner.recording_error = Some(error.clone());
        }
        self.ended = true;
        result
    }

    fn ensure_open(&self) -> Result<(), Error> {
        if self.ended {
            return Err(ApiError::EncoderAlreadyEnded.into());
        }
        if lock(&self.shared.inner, "provider command")?.status != CommandBufferStatus::Recording {
            return Err(ApiError::CommandBufferAlreadyCommitted.into());
        }
        Ok(())
    }
}
impl Drop for RenderCommandEncoder {
    fn drop(&mut self) {
        if !self.ended {
            if let Ok(mut inner) = self.shared.inner.lock() {
                inner.encoder_open = false;
                inner.recording_error = Some(ApiError::EncoderNotEnded.into());
            }
        }
    }
}

#[cfg(test)]
mod tests;
