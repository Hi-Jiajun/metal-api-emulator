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
    HeapResource, IndirectCommandBufferDescriptor, IndirectCommandDescriptor, IndirectCommandKind,
    IndirectCommandPayload, IndirectCommandRange, InitialState, LoadOp, OperationId,
    PipelineCompileRequest, PipelineId, PipelineProvider, PresentDescriptor, PresentMode,
    PresentTarget, ProviderCapabilities, ProviderError, ProviderHealth, ProviderSubmission,
    RenderAttachment, RenderPassDescriptor, ResourceTableSnapshot, StorageMode, StoreOp, ViewId,
    FULL_SCREEN_TRIANGLE_VERTICES, MAX_PRESENT_IMAGE_COUNT, MAX_SERIAL_RESOURCES,
    PROVIDER_SCHEMA_VERSION,
};
use crate::{ApiError, CommandBufferStatus, Size};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

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
        expected: IndirectCommandKind,
        actual: IndirectCommandKind,
    },
    IndirectAlreadyRecorded,
    IndirectDirectConflict,
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
            Self::IndirectKindMismatch { expected, actual } => write!(
                f,
                "encoder replays {actual:?} commands but the indirect buffer holds {expected:?}"
            ),
            Self::IndirectAlreadyRecorded => {
                f.write_str("command buffer already carries an indirect command buffer")
            }
            Self::IndirectDirectConflict => {
                f.write_str("one encoder cannot mix direct and indirect dispatch or draw")
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
                bytes,
            }),
        })
    }

    pub fn new_command_queue(&self) -> CommandQueue {
        CommandQueue {
            owner: Arc::clone(&self.state),
        }
    }

    /// Declare one heap (`research/docs/25` §6 Step 6). The first increment is
    /// fixed-size and refuses aliasing: those rules run here, while the
    /// capability questions (whether this snapshot can back the heap at all)
    /// stay with admission at [`CommandBuffer::commit`].
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

/// One recorded colour attachment: the buffer view that carries the attachment
/// identity and byte range, plus the render-contract shape the encoder restates.
#[derive(Clone)]
struct RenderTarget {
    view: BufferView,
    format: AttachmentFormat,
    width: u64,
    height: u64,
    clear: [u8; 4],
    present: Option<PresentInitial>,
}

impl RenderTarget {
    fn descriptor(&self, pipeline_id: PipelineId) -> Result<RenderPassDescriptor, Error> {
        let attachment = RenderAttachment {
            view_id: self.view.view_id,
            allocation_id: self.view.allocation_id(),
            format: self.format,
            width: self.width,
            height: self.height,
            load: LoadOp::Clear(ClearColor::new(self.clear)),
            store: StoreOp::Store,
        };
        let present = self.present.map(|initial| PresentDescriptor {
            target: PresentTarget {
                allocation_id: self.view.allocation_id(),
                view_id: self.view.view_id,
                format: self.format,
                width: self.width,
                height: self.height,
                image_count: MAX_PRESENT_IMAGE_COUNT,
                initial: match initial {
                    PresentInitial::Undefined => InitialState::Undefined,
                    PresentInitial::Sentinel(bytes) => InitialState::Sentinel(bytes.to_vec()),
                },
            },
            source: self.view.view_id,
            mode: PresentMode::Fifo,
            acquire: AcquirePolicy::Blocking,
        });
        let descriptor = RenderPassDescriptor {
            pipeline: pipeline_id,
            color_attachments: vec![attachment],
            viewport: [
                0,
                0,
                u32::try_from(self.width)
                    .map_err(|_| ContractError::ArithmeticOverflow("attachment width"))?,
                u32::try_from(self.height)
                    .map_err(|_| ContractError::ArithmeticOverflow("attachment height"))?,
            ],
            vertices: FULL_SCREEN_TRIANGLE_VERTICES,
            vertex_buffers: Vec::new(),
            indices: None,
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

/// One sampled texture declared through the object API. The snapshot is the
/// owned bytes captured at declaration time; the provider uploads them once per
/// submission (`research/docs/16` §4.7).
struct TextureInner {
    owner: Arc<DeviceState>,
    allocation_id: AllocationId,
    view_id: ViewId,
    format: contract::TextureFormat,
    width: u64,
    height: u64,
    bytes: Vec<u8>,
}

/// A sampled texture handle. Clone is cheap and shares the same allocation.
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

    /// The contract view for one binding, mirroring `BufferView`'s snapshot.
    fn view(&self, metal_binding: u32) -> contract::TextureView {
        contract::TextureView {
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
            access: contract::TextureAccess::Sampled,
            source: contract::TextureSource::OwnedBytes(self.inner.bytes.clone()),
        }
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
                ids.extend(buffers.values().map(|view| view.view_id));
            }
            RecordedPass::Render { target, .. } => {
                ids.insert(target.view.view_id);
            }
        }
    }
    ids
}

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
                let view = &target.view;
                by_allocation
                    .entry(view.allocation_id())
                    .or_insert_with(|| (&view.buffer, BTreeMap::new()))
                    .1
                    .entry((view.offset, view.offset + view.length))
                    .and_modify(|write| *write = true)
                    .or_insert(true);
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

/// Validate every writeback range before copying any host byte, then land all
/// of them under the reservations held by the caller.
fn apply_writebacks(
    reservations: &[BufferReservation],
    writebacks: &[BufferWriteback],
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
    for writeback in writebacks {
        let position = *positions
            .get(&writeback.allocation_id)
            .ok_or(ContractError::UnknownAllocation(writeback.allocation_id))?;
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
        buffers: BTreeMap<u32, BufferView>,
        textures: BTreeMap<u32, Texture>,
        dispatch: Dispatch,
    },
    Render {
        pipeline: RenderPipeline,
        target: RenderTarget,
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
    reservations: Vec<BufferReservation>,
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
    /// colour attachment, a covering viewport, the three-vertex full-screen
    /// triangle and, optionally, one present tail action. It shares the command
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
            let reservations = reserve_buffers(&passes)?;
            let textures = collect_textures(&passes);
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
        reservations: Vec<BufferReservation>,
        textures: Vec<Texture>,
        heap: Option<Heap>,
        indirect: Option<IndirectCommandBuffer>,
        token: &mut Option<CompletionToken>,
    ) -> Result<ExecutionOutcome, Error> {
        let owner = &self.shared.owner;
        let mut positions = BTreeMap::new();
        let mut resources = ResourceTableSnapshot::new();
        for (position, reservation) in reservations.iter().enumerate() {
            positions.insert(reservation.allocation_id(), position);
            resources.insert_allocation(AllocationRecord {
                allocation_id: reservation.allocation_id(),
                owner_epoch: owner.epoch,
                size: reservation.inner.length as u64,
            })?;
        }
        for texture in &textures {
            resources.insert_allocation(AllocationRecord {
                allocation_id: texture.inner.allocation_id,
                owner_epoch: owner.epoch,
                size: u64::try_from(texture.inner.bytes.len()).unwrap_or(u64::MAX),
            })?;
        }
        // The host bytes must stay stable only while the trace snapshots them:
        // every view copies its bytes into the trace, so the provider never
        // reads the host buffer again. Conflicting CPU access is excluded for
        // the whole commit-to-completion window by the range reservations, and
        // `lock_unreserved` re-checks them under these guards, so releasing the
        // guards before `submit` cannot admit a conflicting CPU write. It does
        // let a sibling command with a disjoint range of the same allocation
        // take its own snapshot while this one is still inside `submit`.
        let mut guards = Vec::with_capacity(reservations.len());
        for reservation in &reservations {
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
                    for (binding, view) in buffers {
                        let reflected = metadata
                            .contract
                            .buffer_bindings
                            .iter()
                            .find(|value| value.metal_binding == *binding)
                            .ok_or(ContractError::UnknownBinding(*binding))?;
                        let bytes = &guards[positions[&view.allocation_id()]];
                        views.push(contract::BufferView {
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
                        });
                    }
                    trace_passes.push(contract::TracePass::Compute(contract::ComputePass {
                        pipeline: metadata.pipeline_id,
                        buffers: views,
                        dispatch: *dispatch,
                        textures: textures
                            .iter()
                            .map(|(binding, texture)| texture.view(*binding))
                            .collect(),
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
                    trace_passes.push(contract::TracePass::Render(
                        target.descriptor(metadata.pipeline_id)?,
                    ));
                }
            }
        }
        // Snapshot complete: no later step of this command reads the host bytes.
        drop(guards);
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
                apply_writebacks(&reservations, &submission.writebacks)?;
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
        apply_writebacks(&pending.reservations, &readback.writebacks)?;
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

/// Encoder state persists across dispatches. Call `clear_buffers` when changing
/// to a pipeline with a different layout; extra bindings are refused.
pub struct ComputeCommandEncoder {
    shared: Arc<CommandShared>,
    pipeline: Option<Pipeline>,
    buffers: BTreeMap<u32, BufferView>,
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
        if let Some((first, _)) = self
            .buffers
            .iter()
            .find(|(other, bound)| **other != index && bound.range().overlaps(&view.range()))
        {
            return Err(ApiError::AliasedBufferBindings {
                first: *first,
                second: index,
            }
            .into());
        }
        self.buffers.insert(index, view.clone());
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
        unique.extend(self.buffers.values().map(|view| view.view_id));
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
                expected: IndirectCommandKind::Dispatch,
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
        unique.extend(self.buffers.values().map(|view| view.view_id));
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
/// submission exactly as the compute encoder does. One call to
/// [`RenderCommandEncoder::draw_render_pass`] records one colour-attachment
/// render pass with the covering viewport, the three-vertex full-screen
/// triangle and an optional present tail. That present tail executes when the
/// command is submitted: its acquire/present count and target terminal layout
/// are not rolled back by a later `cancel` or deadline.
pub struct RenderCommandEncoder {
    shared: Arc<CommandShared>,
    pipeline: Option<RenderPipeline>,
    draw_count: usize,
    indirect: bool,
    ended: bool,
}
impl RenderCommandEncoder {
    pub fn set_render_pipeline_state(&mut self, pipeline: &RenderPipeline) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &pipeline.inner.owner) {
            return Err(ApiError::ForeignPipeline.into());
        }
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }

    /// Record the milestone's single render pass.
    ///
    /// `attachment` names the buffer view the attachment lands in (its
    /// allocation/view identity and byte range); `format`, `width`, `height`
    /// and `clear` restate the attachment shape the render contract fixes, and
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
    pub fn draw_render_pass(
        &mut self,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        clear: [u8; 4],
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let pipeline = self.pipeline.as_ref().ok_or(ApiError::MissingPipeline)?;
        if !Arc::ptr_eq(&self.shared.owner, &attachment.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        let expected_bytes = width
            .checked_mul(height)
            .and_then(|texels| texels.checked_mul(format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("attachment extent"))?;
        if u64::try_from(attachment.length).unwrap_or(u64::MAX) != expected_bytes {
            return Err(ContractError::AttachmentExtentMismatch {
                pass_index: 0,
                view: attachment.view_id,
                expected: expected_bytes,
                declared: u64::try_from(attachment.length).unwrap_or(u64::MAX),
            }
            .into());
        }
        let target = RenderTarget {
            view: attachment.clone(),
            format,
            width,
            height,
            clear,
            present,
        };
        let pipeline_id = pipeline.metadata().pipeline_id;
        // Validate the descriptor the pass will become, so a wrong viewport,
        // vertex count, attachment shape or present shape is refused before any
        // resource is reserved.
        target.descriptor(pipeline_id)?;
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
        unique.insert(attachment.view_id);
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass::Render {
            pipeline: pipeline.clone(),
            target,
        });
        self.draw_count += 1;
        Ok(())
    }

    /// Record the milestone's single render pass replayed from one encoded
    /// indirect draw (`research/docs/25` §6 Step 4). The attachment, viewport
    /// and clear shapes are the direct [`Self::draw_render_pass`] ones; only
    /// the draw command comes from the ICB, so the vertex/instance counts are
    /// the ICB's own.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indirect(
        &mut self,
        icb: &IndirectCommandBuffer,
        attachment: &BufferView,
        format: AttachmentFormat,
        width: u64,
        height: u64,
        clear: [u8; 4],
        present: Option<PresentInitial>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if !Arc::ptr_eq(&self.shared.owner, &icb.inner.owner) {
            return Err(Error::ForeignIndirectCommandBuffer);
        }
        if icb.command_kind() != IndirectCommandKind::Draw {
            return Err(Error::IndirectKindMismatch {
                expected: IndirectCommandKind::Draw,
                actual: icb.command_kind(),
            });
        }
        if self.draw_count > 0 || self.indirect {
            return Err(Error::IndirectDirectConflict);
        }
        let pipeline = self.pipeline.as_ref().ok_or(ApiError::MissingPipeline)?;
        if !Arc::ptr_eq(&self.shared.owner, &attachment.buffer.inner.owner) {
            return Err(Error::ForeignBuffer);
        }
        let expected_bytes = width
            .checked_mul(height)
            .and_then(|texels| texels.checked_mul(format.bytes_per_texel()))
            .ok_or(ContractError::ArithmeticOverflow("attachment extent"))?;
        if u64::try_from(attachment.length).unwrap_or(u64::MAX) != expected_bytes {
            return Err(ContractError::AttachmentExtentMismatch {
                pass_index: 0,
                view: attachment.view_id,
                expected: expected_bytes,
                declared: u64::try_from(attachment.length).unwrap_or(u64::MAX),
            }
            .into());
        }
        let target = RenderTarget {
            view: attachment.clone(),
            format,
            width,
            height,
            clear,
            present,
        };
        let pipeline_id = pipeline.metadata().pipeline_id;
        target.descriptor(pipeline_id)?;
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
        unique.insert(attachment.view_id);
        if unique.len() > MAX_SERIAL_RESOURCES {
            return Err(ContractError::SerialResourceLimit {
                requested: unique.len(),
                maximum: MAX_SERIAL_RESOURCES,
            }
            .into());
        }
        inner.passes.push(RecordedPass::Render {
            pipeline: pipeline.clone(),
            target,
        });
        inner.indirect = Some(icb.clone());
        self.draw_count += 1;
        self.indirect = true;
        Ok(())
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
